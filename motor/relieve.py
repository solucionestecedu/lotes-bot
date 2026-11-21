"""Relieve aproximado desde datos satelitales gratuitos.

Fuente principal: AWS Terrain Tiles (formato Terrarium, zoom 14, ~10 m por píxel; la información
de origen es SRTM/Copernicus de ~30 m). Respaldo: Open-Meteo (Copernicus 90 m).
Sirve para anteproyecto. NO sustituye un levantamiento topográfico.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import requests
from contourpy import contour_generator
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.prepared import prep

from . import proyeccion

URL_TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
URL_OPEN_METEO = "https://api.open-meteo.com/v1/elevation"
ZOOM = 14
FUENTES = {
    "aws": "AWS Terrain Tiles (SRTM/Copernicus, resolución de origen ~30 m)",
    "open-meteo": "Open-Meteo (Copernicus GLO-90, ~90 m)",
    "archivo": "malla de alturas cargada",
}


class ErrorRelieve(RuntimeError):
    pass


@dataclass
class Malla:
    x0: float
    y0: float
    dx: float
    nx: int
    ny: int
    z: np.ndarray  # forma (ny, nx); fila j -> y0 + j*dx
    fuente: str = "archivo"

    @property
    def xs(self):
        return self.x0 + np.arange(self.nx) * self.dx

    @property
    def ys(self):
        return self.y0 + np.arange(self.ny) * self.dx

    def a_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "dx": self.dx, "nx": self.nx, "ny": self.ny,
                "z": [None if not np.isfinite(v) else round(float(v), 2) for v in self.z.ravel()], "fuente": self.fuente}

    @classmethod
    def de_dict(cls, d: dict) -> "Malla":
        z = np.array([np.nan if v is None else v for v in d["z"]], dtype=float).reshape(d["ny"], d["nx"])
        return cls(d["x0"], d["y0"], d.get("dx", d.get("step", 5)), d["nx"], d["ny"], z, d.get("fuente", "archivo"))

    def altura(self, x: float, y: float) -> float:
        fx, fy = (x - self.x0) / self.dx, (y - self.y0) / self.dx
        if not (0 <= fx <= self.nx - 1 and 0 <= fy <= self.ny - 1):
            return math.nan
        i, j = min(int(fx), self.nx - 2), min(int(fy), self.ny - 2)
        u, v = fx - i, fy - j
        z = self.z
        return float(z[j, i] * (1 - u) * (1 - v) + z[j, i + 1] * u * (1 - v) + z[j + 1, i] * (1 - u) * v + z[j + 1, i + 1] * u * v)

    def gradiente(self) -> tuple[np.ndarray, np.ndarray]:
        gy, gx = np.gradient(self.z, self.dx)
        return gx, gy


def malla_para(anillo, paso: float = 5.0, margen: float = 20.0, max_nodos: int = 60000) -> tuple[float, float, float, int, int]:
    xs = [p[0] for p in anillo]
    ys = [p[1] for p in anillo]
    x0, y0 = min(xs) - margen, min(ys) - margen
    w, h = max(xs) + margen - x0, max(ys) + margen - y0
    while (w / paso + 1) * (h / paso + 1) > max_nodos:
        paso *= 1.5
    return x0, y0, paso, int(math.ceil(w / paso)) + 1, int(math.ceil(h / paso)) + 1


def _tile_xy(lat, lng, z=ZOOM):
    n = 2 ** z
    x = (np.asarray(lng) + 180) / 360 * n
    lr = np.radians(np.asarray(lat))
    y = (1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / math.pi) / 2 * n
    return x * 256, y * 256  # píxel global


def decodificar_terrarium(png: bytes) -> np.ndarray:
    from PIL import Image

    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=float)
    return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768


def _desde_aws(lat, lng, http) -> np.ndarray:
    px, py = _tile_xy(lat, lng)
    tiles: dict[tuple[int, int], np.ndarray] = {}
    for tx in range(int(px.min() // 256), int(px.max() // 256) + 1):
        for ty in range(int(py.min() // 256), int(py.max() // 256) + 1):
            r = http.get(URL_TERRARIUM.format(z=ZOOM, x=tx, y=ty), timeout=20)
            r.raise_for_status()
            tiles[(tx, ty)] = decodificar_terrarium(r.content)

    def muestra(gx, gy):
        t = tiles.get((int(gx // 256), int(gy // 256)))
        return np.nan if t is None else t[int(gy % 256), int(gx % 256)]

    out = np.empty(px.shape)
    for k, (x, y) in enumerate(zip(px.ravel(), py.ravel())):
        x -= 0.5
        y -= 0.5
        i, j = math.floor(x), math.floor(y)
        u, v = x - i, y - j
        out.ravel()[k] = (muestra(i, j) * (1 - u) * (1 - v) + muestra(i + 1, j) * u * (1 - v)
                          + muestra(i, j + 1) * (1 - u) * v + muestra(i + 1, j + 1) * u * v)
    return out


def _desde_open_meteo(lat, lng, http) -> np.ndarray:
    lat, lng = np.asarray(lat).ravel(), np.asarray(lng).ravel()
    out = []
    for k in range(0, len(lat), 100):
        r = http.get(URL_OPEN_METEO, params={
            "latitude": ",".join(f"{v:.6f}" for v in lat[k:k + 100]),
            "longitude": ",".join(f"{v:.6f}" for v in lng[k:k + 100])}, timeout=20)
        r.raise_for_status()
        out.extend(r.json()["elevation"])
    return np.array(out, dtype=float)


def descargar(anillo, paso: float = 5.0, http=None) -> Malla:
    """Descarga alturas para una malla que cubre el terreno (coordenadas Lambert SIRGAS)."""
    http = http or requests.Session()
    x0, y0, dx, nx, ny = malla_para(anillo, paso)
    gx, gy = np.meshgrid(x0 + np.arange(nx) * dx, y0 + np.arange(ny) * dx)
    t = proyeccion._t(proyeccion.SIRGAS_LAMBERT, "EPSG:4326")
    lng, lat = t.transform(gx, gy)
    try:
        z = _desde_aws(lat, lng, http).reshape(ny, nx)
        fuente = "aws"
    except Exception:
        # Open-Meteo limita peticiones: se usa una malla más gruesa (≤1500 puntos).
        x0, y0, dx, nx, ny = malla_para(anillo, max(paso, 10.0), max_nodos=1500)
        gx, gy = np.meshgrid(x0 + np.arange(nx) * dx, y0 + np.arange(ny) * dx)
        lng, lat = t.transform(gx, gy)
        try:
            z = _desde_open_meteo(lat, lng, http).reshape(ny, nx)
        except Exception as e:
            raise ErrorRelieve("No pude descargar las alturas del terreno (ni AWS ni Open-Meteo).") from e
        fuente = "open-meteo"
    if not np.isfinite(z).any():
        raise ErrorRelieve("La fuente de alturas no devolvió datos para esta zona.")
    return Malla(x0, y0, dx, nx, ny, z, fuente)


def intervalo_curvas(desnivel: float, max_lineas: int = 70) -> float:
    for i in (0.5, 1, 2, 5, 10, 20, 25, 50):
        if desnivel / i <= max_lineas:
            return float(max(i, 1))
    return 50.0


def curvas(malla: Malla, intervalo: float, recorte: Polygon | None = None) -> list[dict]:
    """Curvas de nivel. Devuelve [{'cota', 'indice', 'lineas': [[(x,y),...], ...]}]."""
    z = np.where(np.isfinite(malla.z), malla.z, np.nan)
    zmin, zmax = np.nanmin(z), np.nanmax(z)
    gen = contour_generator(malla.xs, malla.ys, z)
    out = []
    cota = math.ceil(zmin / intervalo) * intervalo
    while cota <= zmax:
        lineas = [ln for ln in gen.lines(cota) if len(ln) >= 2]
        if recorte is not None and lineas:
            geom = MultiLineString([LineString(ln) for ln in lineas]).intersection(recorte)
            partes = [geom] if isinstance(geom, LineString) else list(getattr(geom, "geoms", []))
            lineas = [list(g.coords) for g in partes if isinstance(g, LineString) and g.length > 0.5]
        else:
            lineas = [[tuple(p) for p in ln] for ln in lineas]
        if lineas:
            out.append({"cota": round(cota, 2), "indice": abs(cota / (intervalo * 5) - round(cota / (intervalo * 5))) < 1e-6,
                        "lineas": lineas})
        cota += intervalo
    return out


def estadisticas(malla: Malla, anillo) -> dict:
    poly = prep(Polygon(anillo))
    gx, gy = malla.gradiente()
    pend = np.hypot(gx, gy) * 100
    zs, ps = [], []
    for j in range(malla.ny):
        for i in range(malla.nx):
            if np.isfinite(malla.z[j, i]) and poly.contains(Point(malla.x0 + i * malla.dx, malla.y0 + j * malla.dx)):
                zs.append(malla.z[j, i])
                ps.append(pend[j, i])
    if not zs:
        zs = [malla.altura(*p) for p in anillo]
        ps = [math.nan]
    zs, ps = np.array(zs), np.array(ps)
    ok = np.isfinite(ps)
    return {
        "cota_min": round(float(np.nanmin(zs)), 1),
        "cota_max": round(float(np.nanmax(zs)), 1),
        "desnivel": round(float(np.nanmax(zs) - np.nanmin(zs)), 1),
        "pendiente_media": round(float(np.mean(ps[ok])), 1) if ok.any() else None,
        "pct_mas_15": round(float(np.mean(ps[ok] > 15) * 100), 1) if ok.any() else None,
        "pct_mas_30": round(float(np.mean(ps[ok] > 30) * 100), 1) if ok.any() else None,
        "fuente": FUENTES.get(malla.fuente, malla.fuente),
        "resolucion_malla_m": malla.dx,
    }
