"""Lo que se sabe del terreno sin preguntarle al dueño: dónde cae (distrito, municipio, departamento, AMSS), qué tan honda
es la quebrada que el plano menciona, y qué tipo de proyecto le queda. Todo sale como propuesta para confirmar.

Los límites de distritos (datos/distritos_sv.geojson, herramientas/construir_distritos.py) son aproximados: cerca de
un límite se dice "confírmalo"."""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import numpy as np
from shapely import STRtree
from shapely.geometry import Point, Polygon, shape

from . import proyeccion

RUTA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datos", "distritos_sv.geojson")
M_POR_GRADO = 111_000.0


@lru_cache(maxsize=1)
def _datos():
    with open(RUTA, encoding="utf-8") as f:
        fc = json.load(f)
    geoms = [shape(x["geometry"]) for x in fc["features"]]
    return [x["properties"] for x in fc["features"]], geoms, STRtree(geoms)


def donde(x: float, y: float) -> dict | None:
    """Distrito, municipio, departamento y AMSS del punto Lambert (x, y). None si cae fuera del país."""
    if not proyeccion.en_el_salvador(x, y):
        return None
    lat, lng = proyeccion.a_latlng(x, y)
    props, geoms, arbol = _datos()
    p = Point(lng, lat)
    dentro = [int(i) for i in arbol.query(p, predicate="intersects")]
    i = dentro[0] if dentro else int(arbol.nearest(p))
    borde_m = geoms[i].boundary.distance(p) * M_POR_GRADO
    out = dict(props[i])
    out["seguro"] = bool(dentro) and borde_m > 500
    if not out["seguro"]:
        out["nota"] = "cerca del límite entre distritos (el mapa de límites es aproximado): confírmalo"
    return out


def profundidad_al_lado(malla, anillo, lado: str, alcance: float = 30.0) -> dict | None:
    """Qué tan hondo baja el terreno justo afuera del lindero del lado dado (orilla → fondo, en los primeros
    `alcance` metros). Sirve para estimar la profundidad de una quebrada que el plano dice que colinda ahí. Con relieve
    satelital es aproximada."""
    from .lotificacion import LADOS, orilla_hacia

    if lado not in LADOS or malla is None:
        return None
    orilla = orilla_hacia(Polygon(anillo).buffer(0), lado)
    if orilla is None or orilla.is_empty:
        return None
    a = math.radians(LADOS[lado])
    vx, vy = math.sin(a), math.cos(a)
    prof = []
    for ln in getattr(orilla, "geoms", [orilla]):
        n = max(2, int(ln.length / 5))
        for k in range(n + 1):
            q = ln.interpolate(k / n, normalized=True)
            z0 = malla.altura(q.x, q.y)
            zs = [malla.altura(q.x + vx * d, q.y + vy * d) for d in np.arange(5, alcance + 0.1, 5)]
            zs = [z for z in zs if math.isfinite(z)]
            if math.isfinite(z0) and zs:
                prof.append(z0 - min(zs))
    if not prof:
        return None
    tip = float(np.median(prof))
    return {"tipica_m": round(max(0.0, tip) * 2) / 2, "maxima_m": round(max(0.0, max(prof)) * 2) / 2}


def propuesta_proyecto(area_m2: float, pend_media: float | None, amss: bool | None) -> dict:
    """Tipo de proyecto y tamaño de lote que le queda al terreno, como punto de partida (el dueño decide)."""
    p = pend_media or 0.0
    if p >= 25:
        return {"tipo": "campestre", "lote_objetivo_m2": 600,
                "por_que": f"con {p:.0f} % de pendiente conviene lote grande: la casa cabe en terrazas y queda jardín"}
    if area_m2 < 20_000 and not amss:
        return {"tipo": "campestre", "lote_objetivo_m2": 600,
                "por_que": "terreno chico fuera del área metropolitana: pocos lotes grandes se venden mejor"}
    if amss and p < 15:
        return {"tipo": "residencial", "lote_objetivo_m2": 250,
                "por_que": "plano y dentro del área metropolitana: lotes residenciales"}
    return {"tipo": "residencial", "lote_objetivo_m2": 400, "por_que": "lotes residenciales medianos"}
