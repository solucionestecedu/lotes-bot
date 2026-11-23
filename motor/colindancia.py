"""Terrenos colindantes: encontrar el lindero que comparten y unirlos en un solo terreno.

Dos terrenos vecinos describen su lindero común en sus propios cuadros, pero al revés: el mismo
tramo aparece con el rumbo opuesto (N 21°20'17" E en uno, S 21°20'17" O en el otro) y la misma
distancia. Con eso se alinean aunque el plano no traiga coordenadas.
"""
from __future__ import annotations

import math

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .rumbos import formatear_rumbo, rumbo_y_distancia
from .terreno import Terreno

TOL_DIST = 0.05          # m
TOL_ANG = 30 / 3600      # 30 segundos de arco


def tramos_comunes(a: Terreno, b: Terreno) -> list[dict]:
    """Pares de tramos (uno de cada terreno) con rumbo opuesto y misma distancia."""
    out = []
    for i, ta in enumerate(a.tramos):
        for j, tb in enumerate(b.tramos):
            dang = abs(((ta["az"] - tb["az"] - 180) + 180) % 360 - 180)
            if abs(ta["dist"] - tb["dist"]) <= max(TOL_DIST, 0.002 * ta["dist"]) and dang <= TOL_ANG:
                out.append({"tramo_a": i + 1, "tramo_b": j + 1, "distancia": round(ta["dist"], 3)})
    return out


def _mover(t: Terreno, dx: float, dy: float) -> Terreno:
    t2 = Terreno.de_dict(t.a_dict())
    t2.mover(dx, dy)
    return t2


def alinear(a: Terreno, b: Terreno) -> tuple[Terreno, dict] | None:
    """Devuelve b colocado junto a a (si comparten lindero) y el detalle de la colindancia."""
    n_a, n_b = len(a.puntos), len(b.puntos)
    if a.ubicado and b.ubicado:
        pa, pb = Polygon(a.puntos).buffer(0), Polygon(b.puntos).buffer(0)
        dist = pa.distance(pb)
        comun = pa.boundary.intersection(pb.buffer(0.5)).length
        if dist <= 1.0 and comun > 1.0:
            return b, {"modo": "coordenadas", "lindero_comun_m": round(comun, 2),
                       "traslape_m2": round(pa.intersection(pb).area, 2)}
        return None
    mejor = None
    for c in tramos_comunes(a, b):
        i, j = c["tramo_a"] - 1, c["tramo_b"] - 1
        # el tramo i de a va de a[i] a a[i+1]; el j de b va al revés: b[j] = a[i+1], b[j+1] = a[i]
        dx = a.puntos[(i + 1) % n_a][0] - b.puntos[j][0]
        dy = a.puntos[(i + 1) % n_a][1] - b.puntos[j][1]
        b2 = _mover(b, dx, dy)
        pa, pb = Polygon(a.puntos).buffer(0), Polygon(b2.puntos).buffer(0)
        traslape = pa.intersection(pb).area
        comun = pa.boundary.intersection(pb.buffer(0.05)).length
        puntaje = comun - 10 * traslape
        if mejor is None or puntaje > mejor[0]:
            mejor = (puntaje, b2, {"modo": "rumbos", "tramo_a": c["tramo_a"], "tramo_b": c["tramo_b"],
                                   "lindero_comun_m": round(comun, 2), "traslape_m2": round(traslape, 2)})
    if mejor is None:
        return None
    b2, info = mejor[1], mejor[2]
    b2.ubicado = a.ubicado
    b2.sistema_original = a.sistema_original
    return b2, info


def unir(terrenos: list[Terreno], nombre: str = "U") -> Terreno:
    """Une terrenos ya alineados en un solo polígono. Los vértices se renombran U1, U2, ..."""
    polys = [Polygon(t.puntos).buffer(0) for t in terrenos]
    u = unary_union([p.buffer(0.02, join_style=2) for p in polys]).buffer(-0.02, join_style=2)
    if u.geom_type != "Polygon":
        u = max(u.geoms, key=lambda g: g.area)
    u = u.simplify(0.02, preserve_topology=True)
    pts = [tuple(p) for p in list(u.exterior.coords)[:-1]]
    # mismo sentido que los cuadros de los planos (horario)
    if Polygon(pts).exterior.is_ccw:
        pts = pts[::-1]
    tramos = []
    for k in range(len(pts)):
        az, d = rumbo_y_distancia(pts[k], pts[(k + 1) % len(pts)])
        tramos.append({"az": az, "dist": round(d, 3), "rumbo": formatear_rumbo(az)})
    declarada = sum(t.area_declarada for t in terrenos) if all(t.area_declarada for t in terrenos) else None
    t0 = terrenos[0]
    return Terreno(nombres=[f"{nombre}{k + 1}" for k in range(len(pts))], puntos=pts, tramos=tramos, cierre=0.0,
                   perimetro=sum(x["dist"] for x in tramos), area_declarada=declarada,
                   sistema_original=t0.sistema_original, ubicado=all(t.ubicado for t in terrenos),
                   origen=" + ".join(t.origen or "terreno" for t in terrenos))
