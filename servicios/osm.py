"""Calles, ríos y construcciones alrededor del terreno, de OpenStreetMap (todo El Salvador).

Los datos van dentro del bot: al publicar se baja el extracto oficial de El Salvador (Geofabrik) y se arma un paquete
liviano (herramientas/construir_osm.py → datos/osm_sv.npz). Así no depende de servidores de afuera mientras atiende.

Sirve para saber por dónde llega la calle y si pasa una quebrada aunque el plano no traiga colindancias, y para
comparar con lo que dice el plano o el usuario. El mapa puede estar incompleto: "no aparece en el mapa" no quiere decir
"no hay". Datos © colaboradores de OpenStreetMap, licencia ODbL.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

log = logging.getLogger(__name__)

FUENTE = "OpenStreetMap (© colaboradores de OpenStreetMap, licencia ODbL)"
ARCHIVO = os.environ.get("OSM_DATOS") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datos", "osm_sv.npz")
ALCANCE = 300.0          # m alrededor del terreno para calles y ríos
ALCANCE_CONSTR = 500.0   # m para contar construcciones
COLINDA_CALLE = 15.0     # m: una calle a esta distancia o menos se toma como pegada al terreno
COLINDA_AGUA = 30.0      # m: un río o quebrada a esta distancia o menos colinda
CELDA = 100.0            # m, cuadros donde se cuentan las construcciones
METIDO = 5.0             # m: lo que entra menos que esto al terreno se toma como que va por el lindero

LADOS_8 = ["norte", "noreste", "oriente", "sureste", "sur", "suroeste", "poniente", "noroeste"]
CALLES = {
    "motorway": "carretera principal", "motorway_link": "carretera principal", "trunk": "carretera principal",
    "trunk_link": "carretera principal", "primary": "carretera primaria", "primary_link": "carretera primaria",
    "secondary": "carretera secundaria", "secondary_link": "carretera secundaria", "tertiary": "carretera terciaria",
    "tertiary_link": "carretera terciaria", "residential": "calle residencial", "living_street": "calle residencial",
    "unclassified": "calle o camino vecinal", "service": "calle de servicio", "track": "camino de tierra",
    "road": "calle", "construction": "vía en construcción",
}
SENDEROS = {"path": "sendero", "footway": "paso peatonal", "steps": "gradas", "pedestrian": "paso peatonal",
            "cycleway": "ciclovía", "bridleway": "sendero"}
AGUAS = {"river": "río", "stream": "quebrada", "canal": "canal", "drain": "canaleta", "ditch": "zanja"}
# lo que entra al paquete, como clave=valor de OpenStreetMap
TIPOS = ([f"highway={k}" for k in CALLES] + [f"highway={k}" for k in SENDEROS] + [f"waterway={k}" for k in AGUAS]
         + ["natural=water"])

_cache: dict = {}
_candado = threading.Lock()


# ---------------------------------------------------------------------------------------------- el paquete de datos
def empacar(vias, edificios=None, fecha: str | None = None) -> dict:
    """vias: [(tipo 'highway=residential', nombre o None, puntos (N,2) en Lambert)]; edificios: (K,2) en Lambert, un
    punto por construcción. Devuelve el paquete tal como lo usa consultar() (y lo guarda guardar())."""
    codigo = {t: i for i, t in enumerate(TIPOS)}
    coords, offs, tipo, nombre, caja, nombres, idx = [], [0], [], [], [], [], {}
    for t, nom, pts in vias:
        if t not in codigo:
            continue
        xy = np.asarray(LineString(np.asarray(pts, dtype=float)).simplify(1.0).coords, dtype=np.float32)
        if len(xy) < 2:
            continue
        coords.append(xy)
        offs.append(offs[-1] + len(xy))
        tipo.append(codigo[t])
        if nom:
            if nom not in idx:
                idx[nom] = len(nombres)
                nombres.append(nom)
            nombre.append(idx[nom])
        else:
            nombre.append(-1)
        caja.append((xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()))
    celdas, cuenta = np.zeros(0, np.int64), np.zeros(0, np.uint32)
    if edificios is not None and len(edificios):
        e = np.asarray(edificios, dtype=float)
        clave = np.floor(e[:, 0] / CELDA).astype(np.int64) * 100000 + np.floor(e[:, 1] / CELDA).astype(np.int64)
        celdas, cuenta = np.unique(clave, return_counts=True)
        cuenta = cuenta.astype(np.uint32)
    meta = {"fuente": FUENTE, "fecha": fecha, "vias": len(tipo), "construcciones": int(cuenta.sum()), "celda_m": CELDA,
            "tipos": TIPOS}
    return {"coords": np.concatenate(coords) if coords else np.zeros((0, 2), np.float32),
            "offs": np.array(offs, dtype=np.int64), "tipo": np.array(tipo, dtype=np.uint8),
            "nombre": np.array(nombre, dtype=np.int32), "caja": np.array(caja, dtype=np.float32).reshape(-1, 4),
            "nombres": nombres, "celdas": celdas, "cuenta": cuenta, "meta": meta}


def _a_bytes(obj) -> np.ndarray:
    return np.frombuffer(json.dumps(obj, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)


def guardar(paquete: dict, ruta: str) -> None:
    arr = {k: v for k, v in paquete.items() if k not in ("nombres", "meta")}
    np.savez_compressed(ruta, nombres=_a_bytes(paquete["nombres"]), meta=_a_bytes(paquete["meta"]), **arr)


def cargar(ruta: str = ARCHIVO) -> dict | None:
    """El paquete guardado, o None si no está (el bot sigue sin calles ni ríos del mapa)."""
    if not os.path.exists(ruta):
        return None
    with np.load(ruta) as z:
        p = {k: z[k] for k in z.files}
    p["nombres"] = json.loads(p["nombres"].tobytes().decode("utf-8"))
    p["meta"] = json.loads(p["meta"].tobytes().decode("utf-8"))
    return p


def _paquete() -> dict | None:
    with _candado:
        if "p" not in _cache:
            try:
                _cache["p"] = cargar()
            except Exception:  # noqa: BLE001
                log.exception("paquete OSM")
                _cache["p"] = None
            if _cache["p"] is None:
                log.warning("Sin paquete de OpenStreetMap (%s): no habrá calles ni ríos del mapa", ARCHIVO)
        return _cache["p"]


# ---------------------------------------------------------------------------------------------- la consulta
def _lado(az: float) -> str:
    return LADOS_8[int(((az % 360) + 22.5) // 45) % 8]


def _cercanas(p: dict, terreno: Polygon, alcance: float):
    """(tipo, nombre, geometría, distancia) de lo que pasa a menos de `alcance` del terreno."""
    x0, y0, x1, y1 = terreno.bounds
    c = p["caja"]
    if not len(c):
        return
    m = ((c[:, 0] <= x1 + alcance) & (c[:, 2] >= x0 - alcance) & (c[:, 1] <= y1 + alcance) & (c[:, 3] >= y0 - alcance))
    tipos, offs = p["meta"].get("tipos") or TIPOS, p["offs"]
    for i in np.nonzero(m)[0]:
        xy = p["coords"][offs[i]:offs[i + 1]].astype(float)
        t = tipos[int(p["tipo"][i])]
        g = LineString(xy)
        if t == "natural=water" and len(xy) >= 4 and np.allclose(xy[0], xy[-1]):
            g = Polygon(xy).buffer(0)
        d = g.distance(terreno)
        if d <= alcance:
            n = int(p["nombre"][i])
            yield t, (p["nombres"][n] if n >= 0 else None), g, d


def _construcciones(p: dict, terreno: Polygon, alcance: float) -> int:
    """Construcciones en el mapa a menos de `alcance` (por cuadros de 100 m: es un conteo aproximado)."""
    if not len(p["celdas"]):
        return 0
    celda = float(p["meta"].get("celda_m") or CELDA)
    ix, iy = p["celdas"] // 100000, p["celdas"] % 100000
    cx, cy = (ix + 0.5) * celda, (iy + 0.5) * celda
    x0, y0, x1, y1 = terreno.bounds
    r = alcance + celda
    m = (cx >= x0 - r) & (cx <= x1 + r) & (cy >= y0 - r) & (cy <= y1 + r)
    return sum(int(n) for x, y, n in zip(cx[m], cy[m], p["cuenta"][m]) if terreno.distance(Point(x, y)) <= alcance)


def clasificar(cercanas, terreno: Polygon) -> dict:
    """Calles, senderos y aguas con distancia, lado del terreno, si colindan y si lo cruzan."""
    c = terreno.centroid
    calles, senderos, aguas = {}, {}, {}
    for t, nombre, g, d in cercanas:
        clave_osm, valor = t.split("=", 1)
        if clave_osm == "highway" and valor in CALLES:
            tipo, destino, lim = CALLES[valor], calles, COLINDA_CALLE
        elif clave_osm == "highway" and valor in SENDEROS:
            tipo, destino, lim = SENDEROS[valor], senderos, COLINDA_CALLE
        elif clave_osm == "waterway" and valor in AGUAS:
            tipo, destino, lim = AGUAS[valor], aguas, COLINDA_AGUA
        elif t == "natural=water":
            tipo, destino, lim = "cuerpo de agua", aguas, COLINDA_AGUA
        else:
            continue
        # el lado: el tramo de lindero que da a la vía (no solo el punto más cercano, que suele ser una esquina)
        cerca = terreno.boundary.intersection(g.buffer(d + 3.0))
        p_t = cerca.centroid if not cerca.is_empty else nearest_points(terreno, g)[0]
        lado = _lado(math.degrees(math.atan2(p_t.x - c.x, p_t.y - c.y)))
        item = {"tipo": tipo, "nombre": nombre, "distancia_m": round(d, 1), "lado": lado, "colinda": d <= lim}
        # el mapa tiene unos metros de error: solo "cruza" lo que entra más de METIDO m (si no, va por el lindero)
        if d < 0.01 and g.intersects(terreno.buffer(-METIDO)):
            dentro = g.intersection(terreno)
            if isinstance(g, Polygon):
                if dentro.area > 1:
                    item["dentro_m2"] = round(dentro.area)
            elif dentro.length > 1:
                item["cruza_m"] = round(dentro.length, 1)
        clave = (tipo, nombre, lado)
        if clave not in destino or d < destino[clave]["distancia_m"]:
            destino[clave] = item
    orden = lambda dic: sorted(dic.values(), key=lambda i: i["distancia_m"])[:6]  # noqa: E731
    return {"fuente": FUENTE, "calles": orden(calles), "senderos": orden(senderos), "aguas": orden(aguas)}


def consultar(anillo, datos: dict | None = None) -> dict | None:
    """Lo que dice OpenStreetMap alrededor del terreno (anillo en Lambert). None si el bot no tiene el paquete."""
    p = datos if datos is not None else _paquete()
    if not p:
        return None
    terreno = Polygon(anillo).buffer(0)
    out = clasificar(_cercanas(p, terreno, ALCANCE), terreno)
    out["construcciones_500m"] = _construcciones(p, terreno, ALCANCE_CONSTR)
    out["fecha_mapa"] = p["meta"].get("fecha")
    return out


def _texto(i: dict) -> str:
    nombre = f" «{i['nombre']}»" if i.get("nombre") else ""
    if i.get("cruza_m"):
        donde = f"cruza el terreno ({i['cruza_m']:.0f} m)"
    elif i.get("dentro_m2"):
        donde = f"ocupa parte del terreno ({i['dentro_m2']:.0f} m²)"
    elif i["colinda"]:
        donde = "pegada al lado " + i["lado"]
    else:
        donde = f"a {i['distancia_m']:.0f} m hacia el {i['lado']}"
    return f"{i['tipo']}{nombre}: {donde}"


def resumen(osm: dict) -> dict:
    """Lo esencial para el bot y el PDF."""
    out = {"fuente": (osm.get("fuente") or FUENTE) + (f", mapa del {osm['fecha_mapa']}" if osm.get("fecha_mapa") else "")}
    if osm.get("calles"):
        out["calles"] = [_texto(i) for i in osm["calles"][:3]]
    if osm.get("aguas"):
        out["rios_y_quebradas"] = [_texto(i) for i in osm["aguas"][:3]]
    if not osm.get("calles") and not osm.get("aguas"):
        out["nada"] = "en el mapa no aparecen calles ni ríos a menos de 300 m (el mapa puede estar incompleto)"
    n = osm.get("construcciones_500m")
    if n is not None:
        out["construcciones_500m"] = n
    return out
