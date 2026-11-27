"""Calles, ríos y construcciones alrededor del terreno, de OpenStreetMap (todo el país), por la API pública Overpass.

Sirve para saber por dónde llega la calle y si pasa una quebrada aunque el plano no traiga colindancias, y para
comparar con lo que dice el plano o el usuario. El mapa puede estar incompleto: "no aparece en el mapa" no quiere decir
"no hay". Datos © colaboradores de OpenStreetMap, licencia ODbL.
"""
from __future__ import annotations

import logging
import math

import requests
from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points

from motor import proyeccion

log = logging.getLogger(__name__)

SERVIDORES = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
FUENTE = "OpenStreetMap (© colaboradores de OpenStreetMap, licencia ODbL)"
AGENTE = "Lotes-Bot/1.0 (prefactibilidad de lotificaciones en El Salvador)"
ALCANCE = 300.0          # m alrededor del terreno para calles y ríos
ALCANCE_CONSTR = 500.0   # m para contar construcciones
COLINDA_CALLE = 15.0     # m: una calle a esta distancia o menos se toma como pegada al terreno
COLINDA_AGUA = 30.0      # m: un río o quebrada a esta distancia o menos colinda
TIEMPO = 20.0

LADOS_8 = ["norte", "noreste", "oriente", "sureste", "sur", "suroeste", "poniente", "noroeste"]
CALLES = {
    "motorway": "carretera principal", "trunk": "carretera principal", "primary": "carretera primaria",
    "secondary": "carretera secundaria", "tertiary": "carretera terciaria", "residential": "calle residencial",
    "living_street": "calle residencial", "unclassified": "calle o camino vecinal", "service": "calle de servicio",
    "track": "camino de tierra", "road": "calle", "construction": "vía en construcción",
}
SENDEROS = {"path": "sendero", "footway": "paso peatonal", "steps": "gradas", "pedestrian": "paso peatonal",
            "cycleway": "ciclovía", "bridleway": "sendero"}
AGUAS = {"river": "río", "stream": "quebrada", "canal": "canal", "drain": "canaleta", "ditch": "zanja"}


def _lado(az: float) -> str:
    return LADOS_8[int(((az % 360) + 22.5) // 45) % 8]


def _caja(terreno: Polygon, margen: float) -> str:
    x0, y0, x1, y1 = terreno.buffer(margen).bounds
    (s, w), (n, e) = proyeccion.a_latlng(x0, y0), proyeccion.a_latlng(x1, y1)
    return f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"


def consulta_overpass(terreno: Polygon) -> str:
    b, bc = _caja(terreno, ALCANCE), _caja(terreno, ALCANCE_CONSTR)
    return (f'[out:json][timeout:{int(TIEMPO)}];'
            f'(way["highway"]({b});way["waterway"]({b});way["natural"="water"]({b}););out geom tags;'
            f'way["building"]({bc});out count;')


def interpretar(datos: dict, terreno: Polygon) -> dict:
    """Respuesta de Overpass → calles, senderos y aguas con distancia, lado del terreno y si colindan."""
    c = terreno.centroid
    calles, senderos, aguas = {}, {}, {}
    construcciones = None
    for e in datos.get("elements") or []:
        if e.get("type") == "count":
            t = e.get("tags") or {}
            construcciones = int(t.get("ways", t.get("total", 0)) or 0)
            continue
        if e.get("type") != "way" or not e.get("geometry"):
            continue
        tags = e.get("tags") or {}
        pts = [proyeccion.a_lambert(p["lat"], p["lon"]) for p in e["geometry"] if "lat" in p]
        if len(pts) < 2:
            continue
        g = LineString(pts)
        if tags.get("natural") == "water" and len(pts) >= 4 and pts[0] == pts[-1]:
            g = Polygon(pts).buffer(0)
        d = g.distance(terreno)
        # el lado: el tramo de lindero que da a la vía (no solo el punto más cercano, que suele ser una esquina)
        cerca = terreno.boundary.intersection(g.buffer(d + 3.0))
        p_t = cerca.centroid if not cerca.is_empty else nearest_points(terreno, g)[0]
        lado = _lado(math.degrees(math.atan2(p_t.x - c.x, p_t.y - c.y)))
        dentro = g.intersection(terreno).length if hasattr(g, "length") and d < 0.01 else 0.0
        if "highway" in tags:
            h = tags["highway"]
            if h in CALLES:
                tipo, destino, lim = CALLES[h], calles, COLINDA_CALLE
            elif h in SENDEROS:
                tipo, destino, lim = SENDEROS[h], senderos, COLINDA_CALLE
            else:
                continue
        elif tags.get("waterway") in AGUAS:
            tipo, destino, lim = AGUAS[tags["waterway"]], aguas, COLINDA_AGUA
        elif tags.get("natural") == "water":
            tipo, destino, lim = "cuerpo de agua", aguas, COLINDA_AGUA
        else:
            continue
        item = {"tipo": tipo, "nombre": tags.get("name"), "distancia_m": round(d, 1), "lado": lado,
                "colinda": d <= lim}
        if dentro > 1:
            item["cruza_m"] = round(dentro, 1)
        clave = (tipo, item["nombre"], lado)
        if clave not in destino or d < destino[clave]["distancia_m"]:
            destino[clave] = item
    orden = lambda dic: sorted(dic.values(), key=lambda i: i["distancia_m"])[:6]  # noqa: E731
    return {"fuente": FUENTE, "calles": orden(calles), "senderos": orden(senderos), "aguas": orden(aguas),
            "construcciones_500m": construcciones}


def consultar(anillo, http=None) -> dict | None:
    """Lo que dice OpenStreetMap alrededor del terreno (anillo en Lambert). None si ningún servidor respondió."""
    http = http or requests.Session()
    terreno = Polygon(anillo).buffer(0)
    q = consulta_overpass(terreno)
    for url in SERVIDORES:
        try:
            r = http.post(url, data={"data": q}, timeout=TIEMPO + 5, headers={"User-Agent": AGENTE})
            r.raise_for_status()
            return interpretar(r.json(), terreno)
        except Exception as ex:  # noqa: BLE001
            log.warning("Overpass %s: %s", url, ex)
    return None


def resumen(osm: dict) -> dict:
    """Lo esencial para el bot y el PDF."""
    def txt(i):
        nombre = f" «{i['nombre']}»" if i.get("nombre") else ""
        donde = (f"cruza el terreno ({i['cruza_m']:.0f} m)" if i.get("cruza_m") else
                 "pegada al lado " + i["lado"] if i["colinda"] else f"a {i['distancia_m']:.0f} m hacia el {i['lado']}")
        return f"{i['tipo']}{nombre}: {donde}"
    out = {"fuente": osm.get("fuente")}
    if osm.get("calles"):
        out["calles"] = [txt(i) for i in osm["calles"][:3]]
    if osm.get("aguas"):
        out["rios_y_quebradas"] = [txt(i) for i in osm["aguas"][:3]]
    if not osm.get("calles") and not osm.get("aguas"):
        out["nada"] = "en el mapa no aparecen calles ni ríos a menos de 300 m (el mapa puede estar incompleto)"
    n = osm.get("construcciones_500m")
    if n is not None:
        out["construcciones_500m"] = n
    return out
