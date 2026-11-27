"""Lo que dice la OPAMSS de un terreno del AMSS, consultado en su geovisor público (las capas del Esquema Director que
también usa su plataforma NIX): clasificación del suelo, tratamiento y aptitud urbanística, alturas, impermeabilización
y edificabilidad permitidas; vías mayores, menores y proyectadas; zonas de peligro (tectónico, volcánico y por dinámica
superficial) y recarga acuífera.

Solo se consulta y se reporta con su fuente: los números del proyecto los sigue poniendo el motor. Si el servidor no
responde, se dice ("no se pudo consultar"), nunca se inventa.
"""
from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor

import requests
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union

from motor import proyeccion

log = logging.getLogger(__name__)

SERVICIO = "https://geovisor.opamss.org.sv/server/rest/services/NewFolder/Geoportal/MapServer"
FUENTE = "OPAMSS, geovisor del Esquema Director del AMSS (Diario Oficial, 6 de marzo de 2026)"

# capas del servicio público (id → para qué)
USO_SUELO = 50          # Anexo 25: clasificación, tratamiento, aptitud, alturas, impermeabilización, IEN
VIA_MAYOR, VIA_MENOR, VIA_PROYECTADA = 151, 152, 154
TECTONICA, VOLCANICO = 155, 156
RECARGA = 190
PELIGROSIDAD = 290      # Anexo 22: peligrosidad por dinámica superficial

CAMPOS_USO = ["cod_clas", "des_clas", "des_tra", "des_apt", "alt_bas", "alt_amp", "imp_base", "imp_amp", "ien_base",
              "ien_amplia"]
# clasificación (cod_clas): URB suelo urbano, URA urbanizable, RUR rural, NUR no urbanizable
# tratamiento (des_tra): REC, EST, CON, MIT, RAM, RCH, EXP, RED, DAG, CAM (entre paréntesis al final del texto)
ALCANCE_VIAS = 300.0    # m alrededor del terreno para buscar vías
TIEMPO = 12.0           # s por consulta


# ------------------------------------------------------------------ geometría
def _a_wgs(anillo) -> list[list[float]]:
    pts = [list(reversed(proyeccion.a_latlng(x, y))) for x, y in anillo]  # [lng, lat]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _a_lambert(lng: float, lat: float) -> tuple[float, float]:
    return proyeccion.a_lambert(lat, lng)


def _area_firmada(anillo) -> float:
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(anillo, anillo[1:] + anillo[:1])) / 2


def _poligono(geom: dict):
    """Polígono de ESRI JSON (anillos exteriores en sentido horario, huecos al revés) a shapely en Lambert."""
    anillos = [[_a_lambert(*p[:2]) for p in r] for r in (geom or {}).get("rings") or [] if len(r) >= 4]
    if not anillos:
        return None
    exteriores = [Polygon(r) for r in anillos if _area_firmada(r) < 0] or [Polygon(r) for r in anillos]
    huecos = [Polygon(r) for r in anillos if _area_firmada(r) >= 0] if len(exteriores) < len(anillos) else []
    out = []
    for e in exteriores:
        e = e.buffer(0)
        for h in huecos:
            if e.contains(h.representative_point()):
                e = e.difference(h.buffer(0))
        out.append(e)
    return unary_union(out).buffer(0)


def _lineas(geom: dict):
    partes = [[_a_lambert(*p[:2]) for p in r] for r in (geom or {}).get("paths") or [] if len(r) >= 2]
    return MultiLineString(partes) if partes else None


# ------------------------------------------------------------------ consultas
def _consulta(http, capa: int, geometria: dict, tipo: str, campos="*", geometria_salida: bool = False,
              solo_cuenta: bool = False) -> dict:
    params = {"geometry": json.dumps(geometria), "geometryType": tipo, "inSR": 4326, "outSR": 4326,
              "spatialRel": "esriSpatialRelIntersects", "outFields": campos if isinstance(campos, str) else ",".join(campos),
              "returnGeometry": "true" if geometria_salida else "false", "f": "json"}
    if solo_cuenta:
        params["returnCountOnly"] = "true"
    r = http.get(f"{SERVICIO}/{capa}/query", params=params, timeout=TIEMPO)
    r.raise_for_status()
    d = r.json()
    if "error" in d:
        raise RuntimeError(f"capa {capa}: {d['error'].get('message', d['error'])}")
    return d


def _num(v):
    try:
        f = float(str(v).replace("%", "").replace(",", ".").strip())
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _uso_suelo(http, poli_wgs, terreno: Polygon) -> list[dict]:
    d = _consulta(http, USO_SUELO, poli_wgs, "esriGeometryPolygon", CAMPOS_USO, geometria_salida=True)
    zonas: dict[tuple, float] = {}
    attrs: dict[tuple, dict] = {}
    for f in d.get("features") or []:
        a = {k.lower(): v for k, v in (f.get("attributes") or {}).items()}
        g = _poligono(f.get("geometry"))
        area = g.intersection(terreno).area if g is not None else 0.0
        clave = tuple(a.get(k) for k in CAMPOS_USO)
        zonas[clave] = zonas.get(clave, 0.0) + area
        attrs[clave] = a
    total = terreno.area or 1.0
    out = []
    for clave, area in sorted(zonas.items(), key=lambda kv: -kv[1]):
        if area < 0.005 * total and len(zonas) > 1:
            continue
        a = attrs[clave]
        out.append({"codigo": (a.get("cod_clas") or "").strip().upper() or None,
                    "clasificacion": a.get("des_clas"), "tratamiento": a.get("des_tra"), "aptitud": a.get("des_apt"),
                    "altura_base": a.get("alt_bas"), "altura_ampliada": a.get("alt_amp"),
                    "impermeabilizacion_base": a.get("imp_base"), "impermeabilizacion_ampliada": a.get("imp_amp"),
                    "edificabilidad_base": a.get("ien_base"), "edificabilidad_ampliada": a.get("ien_amplia"),
                    "pct": round(100 * area / total, 1)})
    return out


def _vias(http, terreno: Polygon) -> dict:
    x0, y0, x1, y1 = terreno.buffer(ALCANCE_VIAS).bounds
    (la0, lo0), (la1, lo1) = proyeccion.a_latlng(x0, y0), proyeccion.a_latlng(x1, y1)
    caja = {"xmin": lo0, "ymin": la0, "xmax": lo1, "ymax": la1, "spatialReference": {"wkid": 4326}}
    out = {}
    for nombre, capa in (("mayor", VIA_MAYOR), ("menor", VIA_MENOR), ("proyectada", VIA_PROYECTADA)):
        d = _consulta(http, capa, caja, "esriGeometryEnvelope", "*", geometria_salida=True)
        lineas = [g for g in (_lineas(f.get("geometry")) for f in d.get("features") or []) if g is not None]
        if not lineas:
            out[nombre] = None
            continue
        todas = unary_union(lineas)
        out[nombre] = {"distancia_m": round(todas.distance(terreno), 1),
                       "cruza_m": round(todas.intersection(terreno).length, 1)}
    return out


def _presencia(http, capa: int, poli_wgs) -> bool:
    return bool(_consulta(http, capa, poli_wgs, "esriGeometryPolygon", solo_cuenta=True).get("count"))


def _por_area(http, capa: int, poli_wgs, terreno: Polygon, campo_pref: list[str]) -> list[dict]:
    """Clases de una capa dentro del terreno, con su % de área (campo: el primero de `campo_pref` que exista)."""
    d = _consulta(http, capa, poli_wgs, "esriGeometryPolygon", "*", geometria_salida=True)
    clases: dict = {}
    for f in d.get("features") or []:
        a = {k.lower(): v for k, v in (f.get("attributes") or {}).items()}
        campo = next((c for c in campo_pref if c in a and a[c] not in (None, "")), None)
        if campo is None:
            continue
        g = _poligono(f.get("geometry"))
        area = g.intersection(terreno).area if g is not None else 0.0
        clases[a[campo]] = clases.get(a[campo], 0.0) + area
    total = terreno.area or 1.0
    return [{"valor": v, "pct": round(100 * a / total, 1)} for v, a in sorted(clases.items(), key=lambda kv: -kv[1])
            if a >= 0.005 * total]


def consultar(anillo, http=None) -> dict | None:
    """Todo lo que dice el geovisor de la OPAMSS del terreno (anillo en Lambert). None si no respondió nada."""
    http = http or requests.Session()
    terreno = Polygon(anillo).buffer(0)
    poli_wgs = {"rings": [_a_wgs(list(terreno.exterior.coords)[:-1])], "spatialReference": {"wkid": 4326}}
    tareas = {
        "uso_suelo": lambda: _uso_suelo(http, poli_wgs, terreno),
        "vias": lambda: _vias(http, terreno),
        "zona_tectonica": lambda: _presencia(http, TECTONICA, poli_wgs),
        "peligro_volcanico": lambda: _presencia(http, VOLCANICO, poli_wgs),
        "peligrosidad_superficial": lambda: _por_area(http, PELIGROSIDAD, poli_wgs, terreno,
                                                      ["problemati"]),
        "recarga_mm_anio": lambda: _por_area(http, RECARGA, poli_wgs, terreno, ["recarga"]),
    }
    out, fallaron = {"fuente": FUENTE}, []
    with ThreadPoolExecutor(max_workers=len(tareas)) as ex:
        futuros = {k: ex.submit(f) for k, f in tareas.items()}
        for k, fu in futuros.items():
            try:
                out[k] = fu.result()
            except Exception as e:  # noqa: BLE001
                log.warning("OPAMSS %s: %s", k, e)
                fallaron.append(k)
    if len(fallaron) == len(tareas):
        return None
    if fallaron:
        out["no_se_pudo_consultar"] = fallaron
    return out
