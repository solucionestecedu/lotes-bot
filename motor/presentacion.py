"""Imagen de presentación del anteproyecto (tipo plano de ventas), en planta y en vista aérea.

"La IA pinta, el motor mide":
1. Calco: el motor dibuja la geometría exacta (lotes, calle, retorno, portón, área verde, quebrada, casas tipo,
   árboles), sin ningún texto.
2. Un modelo de imagen (Nano Banana) solo lo pinta, con la instrucción de no mover nada ni escribir nada.
3. Se compara la imagen pintada con el calco: si movió linderos o se comió lotes, se descarta.
4. Los linderos de los lotes, sus medidas y los letreros (número y área de cada lote, calle, retorno, área verde,
   quebrada, norte, escala) se ponen aquí encima, con la geometría y los números del motor. Si no hay modelo o
   falla, se entrega el calco con sus letreros.
"""
from __future__ import annotations

import io
import logging
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import linemerge, nearest_points, polylabel, unary_union

from . import modelo3d
from .lotificacion import LADOS, Resultado, _bordes_hacia, _poligonos, proteccion_de_quebrada
from .relieve import Malla, curvas, estadisticas, intervalo_curvas
from .terreno import Terreno

log = logging.getLogger(__name__)

ASPECTOS = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 1.5, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16}
LADO_LARGO = 2048
BANDA_SUP, BANDA_INF = 0.11, 0.12      # fracción del alto para título y simbología
UMBRAL_COINCIDENCIA = 0.25             # correlación mínima de bordes entre el calco y la imagen pintada
UMBRAL_GRIS_EN_VERDE = 0.10            # máx. parte del área verde/protegida que puede verse como asfalto
UMBRAL_GRIS_EN_LOTES = 0.22            # en lotes hay caminos de entrada: un poco más de tolerancia
MIN_GRIS_EN_CALLE = 0.35               # la calle debe seguir viéndose gris
MIN_VERDE_EN_VERDE = 0.40              # el área verde y la protegida deben seguir viéndose verdes

C = {
    "papel": (246, 243, 235), "entorno": (222, 228, 208), "curva": (176, 150, 112), "terreno": (232, 222, 196),
    "lote": (238, 226, 196), "borde_lote": (150, 116, 76), "lindero": (70, 52, 38), "calle": (176, 176, 172),
    "cordon": (238, 238, 232), "verde": (164, 204, 132), "proteccion": (106, 148, 88), "agua": (84, 146, 204),
    "monte": (92, 132, 80), "techo": (196, 96, 62), "cumbrera": (150, 64, 40), "sombra": (60, 60, 50, 70),
    "arbol": (78, 128, 64), "arbol2": (98, 150, 76), "borde_arbol": (48, 88, 44), "camino": (214, 210, 200),
    "porton": (60, 60, 64), "caseta": (236, 232, 222), "acceso": (150, 150, 146),
}

# plano técnico: blanco y negro con acentos suaves (verde, agua), como plano de arquitectura
T = {
    "papel": (255, 255, 255), "entorno": (255, 255, 255), "curva": (150, 150, 150), "terreno": (255, 255, 255),
    "lote": (255, 255, 255), "borde_lote": (25, 25, 25), "lindero": (0, 0, 0), "calle": (226, 226, 226),
    "cordon": (35, 35, 35), "verde": (226, 241, 216), "proteccion": (206, 229, 196), "agua": (70, 130, 200),
    "monte": (206, 229, 196), "techo": (255, 255, 255), "cumbrera": (40, 40, 40), "sombra": (0, 0, 0, 38),
    "arbol": (238, 245, 233), "arbol2": (232, 242, 226), "borde_arbol": (105, 140, 95), "camino": (240, 240, 240),
    "porton": (0, 0, 0), "caseta": (255, 255, 255), "acceso": (214, 214, 214), "alberca": (214, 234, 248),
    "rejilla": (232, 232, 232),
}
C["alberca"] = (96, 176, 222)


@dataclass
class Dibujo:
    vista: str                                   # "planta" | "aerea"
    imagen: Image.Image                          # calco sin textos (RGB)
    aspecto: str
    a_px: Callable[[float, float], tuple]        # coordenada del terreno -> píxel
    linderos_px: list = field(default_factory=list)   # polilíneas en píxeles (para verificar)
    m_por_px: float | None = None                # solo en planta
    zona: tuple = (0, 0, 0, 0)                   # rectángulo del dibujo (sin bandas)
    extra: dict = field(default_factory=dict)    # anclas para letreros


def _fuente(tam: int, negrita: bool = False) -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager

    prop = font_manager.FontProperties(family="DejaVu Sans", weight="bold" if negrita else "normal")
    return ImageFont.truetype(font_manager.findfont(prop), max(8, int(tam)))


def _aspecto(r: float) -> str:
    return min(ASPECTOS, key=lambda k: abs(math.log(ASPECTOS[k] / r)))


def _anillo_de(lot: Resultado) -> list:
    """Contorno del terreno armado con las piezas del prediseño (cuando no se tiene el terreno a mano)."""
    todo = unary_union([l["poly"] for l in lot.lotes] + list(lot.calles) + list(lot.protecciones)
                       + [v["poly"] for v in lot.verdes] + list(lot.remanentes)).buffer(0.05).buffer(-0.05)
    pz = max(_poligonos(todo), key=lambda g: g.area)
    return list(pz.exterior.coords)


def _anillos(g):
    for p in _poligonos(g):
        yield list(p.exterior.coords), [list(i.coords) for i in p.interiors]


def _puntos_dentro(poly, n, rng, sep):
    """Puntos al azar (con semilla) dentro de un polígono, separados al menos `sep`."""
    if poly is None or poly.is_empty or n <= 0:
        return []
    x0, y0, x1, y1 = poly.bounds
    out, intentos = [], 0
    while len(out) < n and intentos < n * 60:
        intentos += 1
        p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        if poly.contains(Point(p)) and all(math.dist(p, q) >= sep for q in out):
            out.append(p)
    return out


# --------------------------------------------------------------------------------------------- planta
def calco_planta(t: Terreno, lot: Resultado, malla: Malla | None = None, ficha: dict | None = None,
                 lado_largo: int = LADO_LARGO, estilo: str = "pintura") -> Dibujo:
    """Trazo exacto de la planta. estilo "pintura": colores para que la IA lo pinte; "tecnico": plano blanco y negro
    de presentación (se entrega tal cual, sin IA)."""
    ficha = ficha or {}
    tec = estilo == "tecnico"
    P = T if tec else C
    prop = Polygon(t.puntos).buffer(0)
    x0, y0, x1, y1 = prop.bounds
    w, h = x1 - x0, y1 - y0
    margen = max(14.0, 0.1 * max(w, h))
    gw, gh = w + 2 * margen, h + 2 * margen
    frac = 1 - BANDA_SUP - BANDA_INF
    aspecto = _aspecto(gw / (gh / frac))
    r = ASPECTOS[aspecto]
    W, H = (lado_largo, round(lado_largo / r)) if r >= 1 else (round(lado_largo * r), lado_largo)
    zona = (0, H * BANDA_SUP, W, H * (1 - BANDA_INF))
    k = min(W / gw, (zona[3] - zona[1]) / gh)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ox, oy = W / 2, (zona[1] + zona[3]) / 2

    S = 2  # se dibuja al doble y se reduce: bordes suaves

    def px(x, y, s=S):
        return ((ox + (x - cx) * k) * s, (oy - (y - cy) * k) * s)

    img = Image.new("RGB", (W * S, H * S), P["papel"])
    d = ImageDraw.Draw(img, "RGBA")
    m2px = k * S  # píxeles (supermuestreo) por metro

    def poli(g, fill=None, outline=None, ancho_m=0.0, min_px=1):
        for ext, hoyos in _anillos(g):
            pts = [px(*p) for p in ext]
            if fill is not None:
                d.polygon(pts, fill=fill)
                for hoyo in hoyos:
                    d.polygon([px(*p) for p in hoyo], fill=P["papel"])
            if outline is not None:
                d.line(pts + [pts[0]], fill=outline, width=max(min_px, int(ancho_m * m2px)), joint="curve")

    def linea(coords, color, ancho_m, min_px=1):
        d.line([px(*p) for p in coords], fill=color, width=max(min_px, int(ancho_m * m2px)), joint="curve")

    rng = random.Random(len(t.puntos) * 7919 + int(prop.area))

    def arbol(q, rr, rng_):
        if tec:  # símbolo de árbol en planta: círculo con copa interior
            d.ellipse([q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr], fill=P["arbol"], outline=P["borde_arbol"],
                      width=max(1, int(rr * 0.12)))
            r2 = rr * 0.45
            d.ellipse([q[0] - r2, q[1] - r2, q[0] + r2, q[1] + r2], outline=P["borde_arbol"], width=max(1, int(rr * 0.08)))
        else:
            _arbol(d, q, rr, rng_)
    dibujo_geo = box(cx - (W / 2) / k, cy - (oy - zona[1]) / k, cx + (W / 2) / k, cy + (zona[3] - oy) / k)

    # entorno (solo en la zona de dibujo) y curvas de nivel
    d.rectangle([0, zona[1] * S, W * S, zona[3] * S], fill=P["entorno"])
    rejilla = None
    if tec:
        gx0, gy0, gx1, gy1 = dibujo_geo.bounds
        paso_g = next(v for v in (10, 20, 25, 50, 100, 200, 500) if (gx1 - gx0) / v <= 12)
        rejilla = (paso_g, gx0, gy0, gx1, gy1)
        for gx in range(int(math.ceil(gx0 / paso_g)), int(gx1 // paso_g) + 1):
            linea([(gx * paso_g, gy0), (gx * paso_g, gy1)], P["rejilla"], 0.0, 2)
        for gy in range(int(math.ceil(gy0 / paso_g)), int(gy1 // paso_g) + 1):
            linea([(gx0, gy * paso_g), (gx1, gy * paso_g)], P["rejilla"], 0.0, 2)
    if malla is not None:
        try:
            st = estadisticas(malla, t.puntos)
            iv = intervalo_curvas(st["desnivel"])
            for c in curvas(malla, iv, recorte=dibujo_geo):
                for ln in c["lineas"]:
                    linea(ln, P["curva"] + ((150,) if c["indice"] else (90,)), 0.25 if c["indice"] else 0.12, 1)
        except Exception:  # noqa: BLE001
            log.exception("curvas para la presentación")

    # calle pública de acceso y quebrada (fuera del terreno, ilustrativas)
    extra: dict = {}
    lado_acc = (ficha.get("acceso_lado") or "").lower()
    calles_u = unary_union(lot.calles) if lot.calles else None
    if lado_acc in LADOS:
        bordes = [LineString(b) for b in _bordes_hacia(prop, lado_acc, tolerancia=35)]
        if bordes and calles_u is not None:  # los que tocan la entrada de la calle, y sus vecinos seguidos
            sel = [b for b in bordes if b.distance(calles_u) < 1.0]
            cambio = True
            while sel and cambio:
                cambio = False
                for b in bordes:
                    if b not in sel and any(b.distance(x) < 0.05 for x in sel):
                        sel.append(b)
                        cambio = True
            bordes = sel or bordes
        if bordes:
            via = linemerge(unary_union(bordes)).buffer(8.0, cap_style=2, join_style=1)
            via = via.difference(prop.buffer(0.6)).intersection(dibujo_geo)
            poli(via, fill=P["acceso"])
            extra["acceso"] = max(_poligonos(via), key=lambda g: g.area, default=None)
    lado_q = None
    for q in ficha.get("quebradas") or []:
        if str(q.get("lado", "")).lower() in LADOS:
            lado_q = str(q["lado"]).lower()
            break
    if lot.protecciones and lado_q:
        bordes = _bordes_hacia(prop, lado_q, tolerancia=50)
        if bordes:
            orilla = linemerge(unary_union([LineString(b) for b in bordes]))
            monte = orilla.buffer(12).difference(prop.buffer(0.4)).intersection(dibujo_geo)
            poli(monte, fill=P["proteccion"])
            for p in _puntos_dentro(monte, int(monte.area / (90 if tec else 40)), rng, 5.0 if tec else 3.5):
                arbol(px(*p), rng.uniform(2.2, 3.4) * m2px, rng)
            agua = []
            for g in getattr(orilla, "geoms", [orilla]):
                try:
                    o1, o2 = g.offset_curve(7.0), g.offset_curve(-7.0)
                    afuera = o1 if prop.distance(o1) > prop.distance(o2) else o2
                    agua.append(afuera)
                except Exception:  # noqa: BLE001
                    continue
            for a in agua:
                if a.length > 2:
                    n = max(8, int(a.length / 2))
                    ondula = []
                    for i in range(n + 1):
                        s_ = a.length * i / n
                        p = a.interpolate(s_)
                        q = a.interpolate(min(a.length, s_ + 0.5))
                        tx, ty = q.x - p.x, q.y - p.y
                        tl = math.hypot(tx, ty) or 1
                        w_ = 1.6 * math.sin(2 * math.pi * s_ / 28.0)
                        ondula.append((p.x - ty / tl * w_, p.y + tx / tl * w_))
                    linea(ondula, P["agua"], 0.7 if tec else 1.8, 2 if tec else 3)
            extra["quebrada"] = max(agua, key=lambda a: a.length) if agua else None

    # terreno, protección, área verde, lotes
    poli(prop, fill=P["terreno"])
    for p in lot.protecciones:
        poli(p, fill=P["proteccion"])
        for q in _puntos_dentro(p, int(p.area / (70 if tec else 28)), rng, 4.5 if tec else 3.0):
            arbol(px(*q), rng.uniform(2.0, 3.2) * m2px, rng)
    for v in lot.verdes:
        poli(v["poly"], fill=P["verde"])
    for e in lot.equipamiento:
        poli(e["poly"], fill=(206, 220, 236))
    for l in lot.lotes:
        poli(l["poly"], fill=P["lote"])

    # calles con cordón
    calles = unary_union(lot.calles) if lot.calles else None
    for c_ in lot.calles:
        poli(c_, fill=P["calle"], outline=P["cordon"], ancho_m=0.35, min_px=2)

    # caminos de entrada y casas tipo (con sombra), árboles en los patios
    casas = []
    for l in lot.lotes:
        esq = modelo3d.huella_casa(l, calles, lot.angulo)
        if esq is None:
            continue
        casa = Polygon(esq)
        casas.append((l, casa))
        if calles is not None:
            frente = LineString([esq[0], esq[1]])
            a, b = nearest_points(frente, calles)
            camino = LineString([(a.x, a.y), (b.x, b.y)]).buffer(1.4, cap_style=2).intersection(l["poly"])
            poli(camino, fill=P["camino"])
    albercas = []
    for l, casa in casas:
        esq = list(casa.exterior.coords)[:4]   # frente izq., frente der., fondo der., fondo izq.
        if l["area"] >= 350:  # alberca detrás de la casa, si cabe
            ux_, uy_ = esq[1][0] - esq[0][0], esq[1][1] - esq[0][1]
            largo = math.hypot(ux_, uy_) or 1
            ux_, uy_ = ux_ / largo, uy_ / largo
            nx_, ny_ = esq[3][0] - esq[0][0], esq[3][1] - esq[0][1]
            prof_c = math.hypot(nx_, ny_) or 1
            nx_, ny_ = nx_ / prof_c, ny_ / prof_c
            pl, pd = min(8.0, largo * 0.6), 3.6
            cx_ = (esq[2][0] + esq[3][0]) / 2 + nx_ * (2.8 + pd / 2)
            cy_ = (esq[2][1] + esq[3][1]) / 2 + ny_ * (2.8 + pd / 2)
            alb = Polygon([(cx_ + sx * ux_ * pl / 2 + sy * nx_ * pd / 2, cy_ + sx * uy_ * pl / 2 + sy * ny_ * pd / 2)
                           for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            if l["poly"].buffer(-1.2).contains(alb) and (calles is None or alb.distance(calles) > 3):
                albercas.append(alb)
                extra.setdefault("albercas", {})[l["id"]] = alb
                poli(alb.buffer(1.0, join_style=2), fill=P["camino"])
                poli(alb, fill=P["alberca"], outline=P["cumbrera"] if tec else (240, 248, 252), ancho_m=0.15, min_px=1)
        sombra = Polygon([(x + 1.0, y - 1.0) for x, y in casa.exterior.coords])
        poli(sombra, fill=P["sombra"])
        poli(casa, fill=P["techo"], outline=P["cumbrera"], ancho_m=0.12, min_px=1)
        # techo a cuatro aguas: cumbrera centrada y limatesas a las esquinas
        m0 = ((esq[0][0] + esq[3][0]) / 2, (esq[0][1] + esq[3][1]) / 2)
        m1 = ((esq[1][0] + esq[2][0]) / 2, (esq[1][1] + esq[2][1]) / 2)
        lg = math.dist(m0, m1) or 1
        pr = math.dist(esq[0], esq[3])
        ex_, ey_ = (m1[0] - m0[0]) / lg, (m1[1] - m0[1]) / lg
        recorte = min(pr / 2, lg / 2)
        r0 = (m0[0] + ex_ * recorte, m0[1] + ey_ * recorte)
        r1 = (m1[0] - ex_ * recorte, m1[1] - ey_ * recorte)
        linea([r0, r1], P["cumbrera"], 0.18, 1)
        for e_, r_ in ((esq[0], r0), (esq[3], r0), (esq[1], r1), (esq[2], r1)):
            linea([e_, r_], P["cumbrera"], 0.1, 1)
    for l in lot.lotes:
        libre = l["poly"].buffer(-1.5)
        for _, casa in casas:
            libre = libre.difference(casa.buffer(2.5))
        for alb in albercas:
            libre = libre.difference(alb.buffer(2.0))
        if calles is not None:
            libre = libre.difference(calles.buffer(6))
        for q in _puntos_dentro(libre, max(1, min(4, int(l["area"] / 350))), rng, 6):
            arbol(px(*q), rng.uniform(1.8, 2.8) * m2px, rng)
    for v in lot.verdes:
        for q in _puntos_dentro(v["poly"].buffer(-1.5), int(v["area"] / 90), rng, 6):
            arbol(px(*q), rng.uniform(2.0, 3.0) * m2px, rng)

    # linderos de lotes y del terreno
    for l in lot.lotes:
        poli(l["poly"], outline=P["borde_lote"], ancho_m=0.22, min_px=2)
    poli(prop, outline=P["lindero"], ancho_m=0.55, min_px=3)

    # portón y caseta (solo dibujo) donde la calle entra al terreno
    tramos = []
    if calles is not None:
        boca = calles.boundary.intersection(prop.exterior.buffer(0.4))
        tramos = [g for g in getattr(boca, "geoms", [boca]) if g.geom_type == "LineString" and g.length > 1]
    if tramos:  # la boca de la calle: la que da al acceso (no la del retorno, que también puede tocar el lindero)
        ent = lot.stats.get("entrada_calle")
        if ent:
            boca = min(tramos, key=lambda g: g.distance(Point(ent)))
        elif extra.get("acceso") is not None:
            boca = min(tramos, key=lambda g: g.distance(extra["acceso"]))
        else:
            boca = max(tramos, key=lambda g: g.length)
        pa, pb = boca.coords[0], boca.coords[-1]
        extra["boca"] = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
    if ficha.get("cerrado") and tramos:
        pa, pb = boca.coords[0], boca.coords[-1]
        mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
        hacia = nearest_points(Point(mid), calles.buffer(-2.5))[1] if not calles.buffer(-2.5).is_empty else calles.centroid
        nx_, ny_ = hacia.x - mid[0], hacia.y - mid[1]
        nl = math.hypot(nx_, ny_) or 1
        nx_, ny_ = nx_ / nl, ny_ / nl
        gate = LineString([(pa[0] + nx_ * 4, pa[1] + ny_ * 4), (pb[0] + nx_ * 4, pb[1] + ny_ * 4)])
        gate = gate.intersection(calles)
        if not gate.is_empty:
            linea(list(getattr(gate, "coords", [])) or list(gate.geoms[0].coords), P["porton"], 0.6, 3)
        ex, ey = pa[0] - pb[0], pa[1] - pb[1]
        el = math.hypot(ex, ey) or 1
        ex, ey = ex / el, ey / el
        bx, by = pa[0] + nx_ * 6.5 - ex * 1.6, pa[1] + ny_ * 6.5 - ey * 1.6
        caseta = Polygon([(bx + sx * 1.3 * ex + sy * 1.6 * nx_, by + sx * 1.3 * ey + sy * 1.6 * ny_)
                          for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        poli(caseta, fill=P["caseta"], outline=P["porton"], ancho_m=0.15, min_px=2)
        extra["porton"] = (bx, by)

    img = img.resize((W, H), Image.LANCZOS)
    a_px = lambda x, y: px(x, y, 1)  # noqa: E731
    # máscara de clases a tamaño final (mismos números que modelo3d.ID_CLASE)
    ids = modelo3d.ID_CLASE
    msk = Image.new("L", (W, H), 0)
    dm = ImageDraw.Draw(msk)

    def pinta(g, n):
        for ext, hoyos in _anillos(g):
            dm.polygon([a_px(*q) for q in ext], fill=n)
            for hoyo in hoyos:
                dm.polygon([a_px(*q) for q in hoyo], fill=0)

    pinta(prop, ids["terreno"])
    for p in lot.protecciones:
        pinta(p, ids["proteccion"])
    for v in lot.verdes:
        pinta(v["poly"], ids["verde"])
    for l in lot.lotes:
        pinta(l["poly"].buffer(-0.6), ids["lote"])
    for c_ in lot.calles:
        pinta(c_.buffer(-0.6), ids["calle"])
    for _, casa in casas:
        pinta(casa.buffer(1.2), ids["techo"])
    extra["mascara"] = np.asarray(msk)
    linderos = [[a_px(*p) for p in l["poly"].exterior.coords] for l in lot.lotes]
    linderos.append([a_px(*p) for p in prop.exterior.coords])
    extra["casas"] = {l["id"]: casa for l, casa in casas}
    if rejilla:
        extra["rejilla"] = rejilla
        extra["ubicado"] = bool(getattr(t, "ubicado", False))
    return Dibujo("tecnica" if tec else "planta", img, aspecto, a_px, linderos, 1 / k, zona, extra)


def _arbol(d: ImageDraw.ImageDraw, q, rr, rng):
    s = rr * 0.35
    d.ellipse([q[0] - rr + s, q[1] - rr + s, q[0] + rr + s, q[1] + rr + s], fill=(40, 50, 30, 60))
    col = C["arbol"] if rng.random() < 0.5 else C["arbol2"]
    d.ellipse([q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr], fill=col, outline=C["borde_arbol"])
    r2 = rr * 0.45
    d.ellipse([q[0] - r2 - rr * 0.2, q[1] - r2 - rr * 0.25, q[0] + r2 - rr * 0.2, q[1] + r2 - rr * 0.25],
              fill=tuple(min(255, c + 22) for c in col))


# --------------------------------------------------------------------------------------------- aérea
def _pads(lot, malla):
    try:
        from .altimetria import plataformas

        return plataformas(lot, malla)
    except Exception:  # noqa: BLE001
        log.exception("plataformas")
        return None


def calco_aereo(t: Terreno, lot: Resultado, malla: Malla, lado_largo: int = LADO_LARGO, casas: bool = True) -> Dibujo:
    escena = modelo3d.construir(t, malla, lot, margen=10.0, casas=casas, plataformas=_pads(lot, malla))
    W, H = lado_largo, round(lado_largo * 9 / 16)
    img, pantalla = modelo3d.render_aereo(escena, t, W, H, 40.0, malla, lineas=True)
    alt = lambda x, y: float(modelo3d._alturas(malla, [x], [y])[0])  # noqa: E731

    def a_px(x, y, z=None):
        return pantalla(x, y, alt(x, y) if z is None else z)

    linderos = []
    for l in lot.lotes:
        pts = list(LineString(l["poly"].exterior.coords).segmentize(2.0).coords)
        linderos.append([a_px(*p) for p in pts])
    msk, _ = modelo3d.render_aereo(escena, t, W, H, 40.0, malla, mascara=True)
    return Dibujo("aerea", img, "16:9", a_px, linderos, None, (0, 0, W, H),
                  {"pantalla": pantalla, "alt": alt, "mascara": np.asarray(msk)[:, :, 0]})


def camaras(t: Terreno, lot: Resultado, malla: Malla, ficha: dict | None = None) -> dict:
    """Cámaras de las vistas de venta: desde el portón mirando la calle hacia el redondel, y desde la quebrada (o el lado
    bajo) mirando el proyecto ladera arriba. Alturas de un dron bajo, para que se lea la terracería."""
    prop = Polygon(t.puntos).buffer(0)
    c = prop.centroid
    zs_ok = malla.z[np.isfinite(malla.z)]
    z_def = float(np.nanmin(zs_ok)) if zs_ok.size else 0.0

    def alt(x, y):
        z = malla.altura(x, y)
        return z if math.isfinite(z) else z_def

    out = {}
    st = lot.stats
    ent, ret = st.get("entrada_calle"), st.get("retorno_centro")
    if ent and ret:
        L = math.dist(ent, ret) or 1
        ux, uy = (ret[0] - ent[0]) / L, (ret[1] - ent[1]) / L
        px_, py_ = ent[0] - ux * 24, ent[1] - uy * 24
        mx, my = ent[0] + ux * 0.55 * L, ent[1] + uy * 0.55 * L
        out["porton"] = {"pos": (px_, py_, alt(ent[0], ent[1]) + 19), "mira": (mx, my, alt(mx, my)), "fov": 60}
    lado = next((q.get("lado") for q in (ficha or {}).get("quebradas") or [] if q.get("lado") in LADOS), None)
    if lado:
        az = math.radians(LADOS[lado])
        gx, gy = math.sin(az), math.cos(az)
    else:
        from .altimetria import resumen

        r = resumen(malla, t.puntos)
        gx, gy = r["punto_bajo"][0] - r["punto_alto"][0], r["punto_bajo"][1] - r["punto_alto"][1]
        n = math.hypot(gx, gy) or 1
        gx, gy = gx / n, gy / n
    ext = max(prop.bounds[2] - prop.bounds[0], prop.bounds[3] - prop.bounds[1])
    px_, py_ = c.x + gx * ext * 0.62, c.y + gy * ext * 0.62
    mx, my = c.x - gx * ext * 0.08, c.y - gy * ext * 0.08
    out["quebrada"] = {"pos": (px_, py_, alt(px_, py_) + 22), "mira": (mx, my, alt(mx, my)), "fov": 60}
    return out


def calco_vista(t: Terreno, lot: Resultado, malla: Malla, camara: dict, lado_largo: int = LADO_LARGO) -> Dibujo:
    escena = modelo3d.construir(t, malla, lot, margen=48.0, plataformas=_pads(lot, malla))
    W, H = lado_largo, round(lado_largo * 9 / 16)
    img, pantalla = modelo3d.render_aereo(escena, t, W, H, 40.0, malla, lineas=True, camara=camara)
    alt = lambda x, y: float(modelo3d._alturas(malla, [x], [y])[0])  # noqa: E731
    return Dibujo("vista", img, "16:9", lambda x, y, z=None: pantalla(x, y, alt(x, y) if z is None else z), [], None,
                  (0, 0, W, H), {"pantalla": pantalla, "alt": alt})


# --------------------------------------------------------------------------------------------- pintar
INSTRUCCION_PLANTA = (
    "You are given an exact top-down site plan of a small country-house (quinta) subdivision in El Salvador. "
    "Repaint it as a polished, professional architectural sales-brochure illustration, seen straight from above "
    "(orthographic). STRICT RULES: keep the composition pixel-aligned with the input. Every property line, lot "
    "division, road edge, roundabout, house footprint, tree and area must stay exactly where it is, with the same "
    "size and shape. Do not add, remove, move, merge or resize any lot, house, road or area. Do not write any text, "
    "letters, numbers, labels, logos or watermarks anywhere. Keep the plain paper bands at the top and at the "
    "bottom empty. Style: soft natural daylight from the upper left with gentle shadows; lots are well-kept lawns; "
    "houses have terracotta clay-tile roofs seen from above; the grey band is the only road (grey asphalt with light "
    "curbs), do not add other roads or tracks; the light green "
    "area is a recreational park lawn with tropical trees; the dark green strip is dense protected ravine forest; "
    "the blue line is a small creek; outside the property, a muted rural landscape. Crisp and realistic, not cartoonish."
)
INSTRUCCION_AEREA = (
    "You are given an exact 3D aerial view of a small country-house (quinta) subdivision on a hillside in El Salvador. "
    "Repaint it as a photorealistic architectural aerial rendering. STRICT RULES: keep the same camera, horizon and "
    "composition, pixel-aligned with the input. Every lot boundary, road, roundabout, house and green area must stay "
    "exactly where it is, with the same size and shape. Do not add, remove, move or resize any lot, house, road or "
    "area. The grey strip ending in a small round turnaround is the ONLY road: do not add any other road, path, "
    "driveway or dirt track, and do not extend it. The light green and dark green areas are lawn and dense forest "
    "and must stay vegetation. Do not write any text, letters, numbers, labels, logos or watermarks. Style: late-morning tropical light, "
    "lush green lawns on the lots, simple modern one-storey houses with terracotta roofs, grey asphalt road, dense "
    "tropical forest in the dark green protected strip, soft blue sky."
)

INSTRUCCION_AMBIENTE = (
    "You are given a simple 3D massing model, seen from the air, of a small gated country-house (quinta) subdivision "
    "on a hillside in El Salvador. Use it as the layout and camera guide and turn it into a premium, photorealistic "
    "real-estate marketing aerial rendering. Keep the same camera angle and framing, the same single access road "
    "ending in its round turnaround, the same number of houses in roughly the same places, the lot boundaries, and "
    "the protected ravine forest along the creek. You may freely improve everything else: varied contemporary one- "
    "and two-storey tropical houses with large windows and terraces, a swimming pool behind each house, lush "
    "landscaped gardens with palms and mature tropical trees, a small gatehouse at the entrance, warm golden-hour "
    "sunlight with long soft shadows, rolling green hills and a hazy sky in the background. Do NOT add other streets, "
    "lots or houses. Do not write any text, letters, numbers, labels, logos or watermarks."
)



INSTRUCCION_PORTON = (
    "You are given a simple 3D massing model of a small gated country-house (quinta) subdivision on a hillside in El "
    "Salvador, seen from a low drone just above the entrance gate, looking along the single access road towards the "
    "round turnaround at the end. Turn it into a premium, photorealistic real-estate marketing photo. Keep the same "
    "camera, the road and its direction, the turnaround, the number of houses on each side and their places: each "
    "house sits right next to the road on its own flat terrace at road level, with a short driveway, and its garden "
    "behind it following the natural slope; between terraces there are grassy embankments and low stone retaining "
    "walls. Improve everything else: elegant gate with stone pillars and a small guard house at the entrance, "
    "contemporary tropical houses, palms and mature trees, warm late-afternoon light. Do NOT add other streets, lots "
    "or houses. Do not write any text, letters, numbers, signs with text, logos or watermarks."
)
INSTRUCCION_QUEBRADA = (
    "You are given a simple 3D massing model of a small country-house (quinta) subdivision on a hillside in El "
    "Salvador, seen from a low drone over the protected ravine at the lower side of the land, looking up the slope at "
    "the project. Turn it into a premium, photorealistic real-estate marketing photo. Keep the same camera and "
    "framing, the single road with its round turnaround, the number of houses and their places on their stepped "
    "terraces, and the slope. In the foreground keep the protected ravine as dense tropical forest with a small rocky "
    "creek and a walking path with benches in the green area next to it. Warm golden-hour light, soft haze in the "
    "background hills. Do NOT add other streets, lots or houses. Do not write any text, letters, numbers, logos or "
    "watermarks."
)
INSTRUCCION_LIMPIOS = (
    "You are given a simple 3D massing model, seen from the air, of a small country-house (quinta) subdivision on a "
    "hillside in El Salvador right after the earthworks and before any house is built. Turn it into a photorealistic "
    "aerial photo of the finished land development: the single paved access road with curbs ending in its round "
    "turnaround, and on each lot a flat graded terrace pad at road level (fresh compacted earth or young grass) next "
    "to the road, with neat grassy embankments and a few low stone retaining walls between terraces, and the rest of "
    "each lot as natural sloping grass. NO houses, no buildings, no pools: only the prepared lots, the road, the "
    "entrance gate, young trees along the road and the protected ravine forest. Keep the same camera, layout, number "
    "of lots and slope. Do not write any text, letters, numbers, logos or watermarks."
)


def guia_ambiente(dib: Dibujo, lot: Resultado) -> str:
    """Datos del trazo que la IA debe respetar en la imagen de ambiente, dichos según la cámara: de qué lado de la
    imagen queda la entrada con el portón y de cuál el redondel, y cuántas casas van a cada lado de la calle."""
    st = lot.stats
    ent, ret = st.get("entrada_calle"), st.get("retorno_centro")
    if st.get("modo") != "calle_central" or not ent or not ret:
        return ""
    (ex, ey), (rx, ry) = dib.a_px(*ent), dib.a_px(*ret)
    lejos, cerca = ("top (far)", "bottom (near the camera)") if ey < ry else ("bottom (near the camera)", "top (far)")
    (nx_, ny_), (fx, fy) = ((ex, ey), (rx, ry)) if ey > ry else ((rx, ry), (ex, ey))  # extremo cercano y lejano
    izq = der = 0
    for l in lot.lotes:
        c = l["poly"].centroid
        x, y = dib.a_px(c.x, c.y)
        # de qué lado de la calle queda el lote en la imagen: mirando del extremo cercano al lejano (hacia arriba),
        # producto cruz positivo = a la derecha
        cruz = (fx - nx_) * (y - ny_) - (fy - ny_) * (x - nx_)
        if cruz > 0:
            der += 1
        else:
            izq += 1
    return (f" LAYOUT TO KEEP: the gated entrance with the gatehouse is at the {lejos} end of the road, where it meets "
            f"the public street; the road ends in the round turnaround at the {cerca} end. Do not swap them. There are "
            f"exactly {izq} houses on the left side of the road and {der} on the right side as seen in the image, "
            f"{izq + der} in total.")


def _gradiente(img: Image.Image, zona, L: int = 512) -> np.ndarray:
    W, H = img.size
    k = L / max(W, H)
    g = img.convert("L").resize((max(8, int(W * k)), max(8, int(H * k))), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1))
    a = np.asarray(g, dtype=np.float32)
    gx, gy = np.zeros_like(a), np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    return np.hypot(gx, gy)[int(zona[1] * k):int(zona[3] * k)]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / den if den > 0 else 0.0


def alinear(base: Image.Image, pintada: Image.Image, zona, L: int = 512) -> tuple[bool, dict, Image.Image]:
    """¿La imagen pintada conserva el dibujo? Compara los bordes (gradiente) con el calco. Busca el mejor
    corrimiento pequeño (hasta ~1 % del ancho) y exige que coincida bien ahí y claramente peor si se corre más:
    si el modelo movió, escaló o inventó formas, no hay un pico claro y se rechaza."""
    A = _gradiente(base, zona, L)
    B = _gradiente(pintada.resize(base.size), zona, L)
    mejor = (-1.0, 0, 0)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            v = _ncc(A, np.roll(np.roll(B, dy, 0), dx, 1))
            if v > mejor[0]:
                mejor = (v, dx, dy)
    s0, bdx, bdy = mejor
    anillo = max(_ncc(A, np.roll(np.roll(B, bdy + ey, 0), bdx + ex, 1))
                 for ex, ey in ((8, 0), (-8, 0), (0, 8), (0, -8), (6, 6), (-6, 6), (6, -6), (-6, -6)))
    info = {"coincidencia": round(s0, 3), "coincidencia_corrida": round(anillo, 3)}
    ok = s0 >= UMBRAL_COINCIDENCIA and s0 >= 1.3 * max(anillo, 0.0)
    if ok and (bdx or bdy):
        k = max(base.size) / L
        info["corrimiento_px"] = (round(bdx * k), round(bdy * k))
        pintada = pintada.resize(base.size).transform(base.size, Image.AFFINE, (1, 0, -bdx * k, 0, 1, -bdy * k),
                                                      resample=Image.BICUBIC, fillcolor=C["papel"])
    return ok, info, pintada.resize(base.size)


def _hsv(img: Image.Image):
    a = np.asarray(img.convert("HSV"), dtype=np.float32)
    return a[..., 0] * 360 / 255, a[..., 1] / 255, a[..., 2] / 255


def medir_clases(img: Image.Image, mascara: np.ndarray) -> dict:
    """Qué tanto de cada zona parece asfalto (gris) o vegetación (verde) en la imagen."""
    h, s, v = _hsv(img.resize((mascara.shape[1], mascara.shape[0])))
    gris = (s < 0.16) & (v > 0.25) & (v < 0.88)
    verde = (h >= 45) & (h <= 175) & (s > 0.15) & (v > 0.08)
    ids = modelo3d.ID_CLASE
    out = {}
    for nombre, cl in (("calle", [ids["calle"]]), ("verde", [ids["verde"], ids["proteccion"]]), ("lotes", [ids["lote"]])):
        m = np.isin(mascara, cl)
        if m.sum() < 50:
            continue
        out[nombre] = {"gris": round(float(gris[m].mean()), 3), "verde": round(float(verde[m].mean()), 3)}
    return out


def clases_ok(medidas: dict) -> tuple[bool, str]:
    """Reglas para rechazar una imagen que inventó cosas: calle encima del área verde o de los lotes, la calle
    convertida en otra cosa, o la zona protegida sin vegetación."""
    v, c, l_ = medidas.get("verde"), medidas.get("calle"), medidas.get("lotes")
    if v and v["gris"] > UMBRAL_GRIS_EN_VERDE:
        return False, "pintó calle o piso donde va área verde o protegida"
    if l_ and l_["gris"] > UMBRAL_GRIS_EN_LOTES:
        return False, "pintó calle o piso encima de los lotes"
    if c and c["gris"] < MIN_GRIS_EN_CALLE:
        return False, "la calle dejó de verse como calle"
    if v and v["verde"] < MIN_VERDE_EN_VERDE:
        return False, "el área verde o protegida dejó de verse verde"
    return True, ""


def pintar(dibujo: Dibujo, pintor: Callable | None, instruccion: str, intentos: int = 2,
           revisar: bool = True) -> tuple[Image.Image, dict]:
    """Devuelve (imagen, info). Si el modelo falla o cambia el dibujo se reintenta; si sigue mal, el calco.
    Con revisar=False (imagen de ambiente) no se exige que respete el trazo: solo se usa como guía."""
    info = {}
    for k in range(intentos):
        img, info = _pintar_una(dibujo, pintor, instruccion, revisar)
        info["intentos"] = k + 1
        if info["ia"] or pintor is None:
            break
    return img, info


def _pintar_una(dibujo: Dibujo, pintor: Callable | None, instruccion: str,
                revisar: bool = True) -> tuple[Image.Image, dict]:
    info = {"ia": False}
    if pintor is None:
        info["motivo"] = "sin modelo de imagen"
        return dibujo.imagen, info
    buf = io.BytesIO()
    dibujo.imagen.save(buf, "PNG")
    try:
        datos = pintor(buf.getvalue(), instruccion, dibujo.aspecto)
    except Exception as e:  # noqa: BLE001
        log.exception("modelo de imagen")
        datos, info["motivo"] = None, f"falló el modelo de imagen: {type(e).__name__}"
    if not datos:
        info.setdefault("motivo", "el modelo de imagen no devolvió imagen")
        return dibujo.imagen, info
    crudo = os.environ.get("PRESENTACION_CRUDO")  # carpeta para guardar lo que devolvió el modelo (pruebas)
    if crudo:
        os.makedirs(crudo, exist_ok=True)
        with open(os.path.join(crudo, f"{dibujo.vista}-modelo.png"), "wb") as f:
            f.write(datos)
        dibujo.imagen.save(os.path.join(crudo, f"{dibujo.vista}-calco.png"))
    try:
        pintada = Image.open(io.BytesIO(datos)).convert("RGB").resize(dibujo.imagen.size, Image.LANCZOS)
    except Exception:  # noqa: BLE001
        info["motivo"] = "la imagen del modelo no se pudo abrir"
        return dibujo.imagen, info
    if not revisar:
        info["ia"] = True
        return pintada, info
    ok, medidas, pintada = alinear(dibujo.imagen, pintada, dibujo.zona)
    info.update(medidas)
    if not ok:
        info["motivo"] = "el modelo movió o cambió el dibujo; se usó el calco exacto"
        return dibujo.imagen, info
    if dibujo.extra.get("mascara") is not None:
        info["clases"] = medir_clases(pintada, dibujo.extra["mascara"])
        ok, por_que = clases_ok(info["clases"])
        if not ok:
            info["motivo"] = f"el modelo {por_que}; se usó el calco exacto"
            return dibujo.imagen, info
    info["ia"] = True
    return pintada, info


# --------------------------------------------------------------------------------------------- letreros
def _texto(d, xy, txt, tam, negrita=False, color=(30, 30, 30), halo=(255, 255, 255), anchor="mm", borde=None,
           ocupado: list | None = None):
    f = _fuente(tam, negrita)
    sw = borde if borde is not None else max(2, int(tam / 7))
    d.text(xy, txt, font=f, fill=color, anchor=anchor, stroke_width=sw, stroke_fill=halo, align="center")
    if ocupado is not None:
        ocupado.append(box(*d.textbbox(xy, txt, font=f, anchor=anchor, stroke_width=sw, align="center")))
    return f


def _caja_girada(xy, w, h, angulo_grados) -> Polygon:
    """Rectángulo w×h centrado en xy y girado (ángulo a la izquierda, como se ve en pantalla), en píxeles."""
    a = math.radians(angulo_grados)
    ux, uy = math.cos(a), -math.sin(a)
    vx, vy = math.sin(a), math.cos(a)
    return Polygon([(xy[0] + sx * ux * w / 2 + sy * vx * h / 2, xy[1] + sx * uy * w / 2 + sy * vy * h / 2)
                    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _texto_girado(img, xy, txt, tam, angulo_grados, negrita=True, color=(30, 30, 30), ocupado: list | None = None):
    f = _fuente(tam, negrita)
    caja = f.getbbox(txt, stroke_width=3)
    w, h = caja[2] - caja[0] + 12, caja[3] - caja[1] + 12
    capa = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(capa).text((w / 2, h / 2), txt, font=f, fill=color + (255,), anchor="mm", stroke_width=3,
                              stroke_fill=(255, 255, 255, 235))
    a = ((angulo_grados + 90) % 180) - 90  # que no quede de cabeza
    if ocupado is not None:
        ocupado.append(_caja_girada(xy, w, h, a))
    capa = capa.rotate(a, expand=True, resample=Image.BICUBIC)
    img.alpha_composite(capa, (int(xy[0] - capa.width / 2), int(xy[1] - capa.height / 2)))


def _tramos(poly, esquina: float = 12.0, junta: float = 0.3) -> list[LineString]:
    """Tramos del contorno de un lote, de esquina a esquina (rectos, o curvos junto al redondel), en metros.
    Une los puntos casi pegados que dejan los recortes y no corta donde el contorno solo dobla poco."""
    pts: list = []
    for q in list(poly.exterior.coords)[:-1]:
        if not pts or math.dist(q, pts[-1]) > junta:
            pts.append((q[0], q[1]))
    while len(pts) > 2 and math.dist(pts[0], pts[-1]) <= junta:
        pts.pop()
    n = len(pts)
    if n < 3:
        return []

    def giro(i):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        g = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(b[1] - a[1], b[0] - a[0]))
        return abs((g + 180) % 360 - 180)

    esq = [i for i in range(n) if giro(i) >= esquina]
    out = []
    for k, i in enumerate(esq):
        j = esq[(k + 1) % len(esq)]
        idx = list(range(i, j + 1)) if j > i else list(range(i, n)) + list(range(0, j + 1))
        out.append(LineString([pts[t % n] for t in idx]))
    return out


def _cota(img, xy, txt, tam, angulo_grados):
    """Medida de un lindero: texto en una pastilla blanca, encima de la línea y a lo largo de ella."""
    f = _fuente(tam, True)
    c = f.getbbox(txt)
    w, h = c[2] - c[0] + tam * 0.9, tam * 1.45
    capa = Image.new("RGBA", (int(w) + 4, int(h) + 4), (0, 0, 0, 0))
    dc = ImageDraw.Draw(capa)
    dc.rounded_rectangle([2, 2, w + 1, h + 1], radius=h / 2, fill=(255, 255, 255, 232), outline=(110, 90, 60, 220), width=1)
    dc.text(((w + 4) / 2, (h + 4) / 2), txt, font=f, fill=(45, 35, 25, 255), anchor="mm")
    capa = capa.rotate(angulo_grados, expand=True, resample=Image.BICUBIC)
    img.alpha_composite(capa, (int(xy[0] - capa.width / 2), int(xy[1] - capa.height / 2)))


def _medida_cota(txt, tam):
    f = _fuente(tam, True)
    c = f.getbbox(txt)
    return c[2] - c[0] + tam * 0.9 + 4, tam * 1.45 + 4


def _fmt(v, dec=0):
    return f"{v:,.{dec}f}"


def _titulo(lot: Resultado, ficha: dict, partes: list | None) -> tuple[str, str]:
    st = lot.stats
    n = st["lotes"]
    nombre = "QUINTAS" if (ficha.get("tipo") == "campestre" or st.get("modo") == "calle_central") else "LOTES"
    areas = [l["area"] for l in lot.lotes]
    rango = f"{_fmt(min(areas))} a {_fmt(max(areas))} m²" if min(areas) < 0.97 * max(areas) else f"~{_fmt(areas[0])} m²"
    t1 = f"ANTEPROYECTO DE {n} {nombre} ({rango})"
    sub = [f"Área total {_fmt(st['area_total'])} m²", f"{st['area_util'] / st['area_total'] * 100:.0f} % vendible"]
    if partes:
        sub.insert(0, " · ".join(f"{p['origen']}: {_fmt(Polygon(p['anillo']).area)} m²" for p in partes))
    return t1, " · ".join(sub)


def rotular_planta(img: Image.Image, dib: Dibujo, lot: Resultado, ficha: dict, proyecto: str,
                   partes: list | None = None, ia: bool = False, estilo: str = "pintura") -> Image.Image:
    tec = estilo == "tecnico"
    img = img.convert("RGBA")
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    m = 1 / dib.m_por_px  # píxeles por metro
    st = lot.stats
    # bandas
    d.rectangle([0, 0, W, dib.zona[1]], fill=(255, 255, 255, 245))
    d.rectangle([0, dib.zona[3], W, H], fill=(255, 255, 255, 245))
    t1, t2 = _titulo(lot, ficha, partes)
    bt = dib.zona[1]
    _texto(d, (W * 0.03, bt * 0.36), t1, bt * 0.27, True, anchor="lm", borde=0)
    _texto(d, (W * 0.03, bt * 0.74), f"{proyecto} · {t2}", bt * 0.17, False, (70, 70, 70), anchor="lm", borde=0)

    ocupado: list = []  # lo que ya tiene letrero (en píxeles): las medidas de los linderos no se le enciman

    # linderos de los lotes (encima de la pintura, exactos): línea blanca con sombra; el del terreno, oscuro.
    # En el plano técnico: líneas negras, y la del terreno más gruesa
    capa = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dl = ImageDraw.Draw(capa)
    ancho = max(2, round(m * 0.4))
    contornos = [[dib.a_px(*q) for q in ext] for l in lot.lotes for ext, _ in _anillos(l["poly"])]
    if tec:
        for pts in contornos:
            dl.line(pts + [pts[0]], fill=(20, 20, 20, 255), width=max(2, round(m * 0.25)), joint="curve")
    else:
        for pts in contornos:
            dl.line(pts + [pts[0]], fill=(30, 24, 16, 120), width=ancho + 3, joint="curve")
        for pts in contornos:
            dl.line(pts + [pts[0]], fill=(255, 255, 255, 235), width=ancho, joint="curve")
    if dib.linderos_px:
        per = list(dib.linderos_px[-1])
        dl.line(per + [per[0]], fill=(0, 0, 0, 255) if tec else (70, 50, 34, 255), width=ancho + (4 if tec else 2),
                joint="curve")
    img.alpha_composite(capa)
    # terrenos unidos: lindero entre ellos (trazo y punto) y el nombre de cada uno afuera
    from .plano_tecnico import _discontinua, terrenos_unidos

    prop_ = Polygon(dib.linderos_px[-1]) if dib.linderos_px else None
    unidos = terrenos_unidos(unary_union([l["poly"] for l in lot.lotes] + list(lot.calles) + [v["poly"] for v in lot.verdes]
                                         + list(lot.protecciones)).buffer(0.05).buffer(-0.05), partes) if partes else []
    for u in unidos:
        for ln in u["linderos"]:
            pts = [dib.a_px(*q) for q in ln.coords]
            _discontinua(d, pts, (255, 255, 255, 230), ancho + 4, 16, 6, punto=True)
            _discontinua(d, pts, (35, 35, 35, 255), max(2, ancho), 16, 6, punto=True)
    if tec:  # marco del plano y coordenadas de la rejilla
        d.rectangle([dib.zona[0] + 3, dib.zona[1], dib.zona[2] - 4, dib.zona[3]], outline=(0, 0, 0), width=3)
        rej = dib.extra.get("rejilla")
        if rej and dib.extra.get("ubicado"):
            paso_g, gx0, gy0, gx1, gy1 = rej
            fg = max(14, int(m * 1.6))
            for gx in range(int(math.ceil(gx0 / paso_g)), int(gx1 // paso_g) + 1):
                x, _ = dib.a_px(gx * paso_g, gy1)
                if dib.zona[0] + 60 < x < dib.zona[2] - 60:
                    _texto(d, (x, dib.zona[1] + fg * 1.1), f"E {gx * paso_g:,.0f}", fg, False, (120, 120, 120), borde=2)
            for gy in range(int(math.ceil(gy0 / paso_g)), int(gy1 // paso_g) + 1):
                _, y = dib.a_px(gx0, gy * paso_g)
                if dib.zona[1] + 60 < y < dib.zona[3] - 60:
                    _texto(d, (dib.zona[0] + 12, y), f"N {gy * paso_g:,.0f}", fg, False, (120, 120, 120), anchor="lm",
                           borde=2)

    # lotes: número y área en una etiqueta
    tam = max(13, min(30, m * 3.0))
    f1, f2 = _fuente(tam, True), _fuente(tam * 0.82, True)
    for l in lot.lotes:
        poly = l["poly"]
        casa = dib.extra.get("casas", {}).get(l["id"])
        libre = poly.buffer(-1.0)
        if casa is not None:
            libre = libre.difference(casa.buffer(1.5))
        alb = dib.extra.get("albercas", {}).get(l["id"])
        if alb is not None:
            libre = libre.difference(alb.buffer(1.5))
        libre = max(_poligonos(libre), key=lambda g: g.area, default=poly)
        try:
            p = polylabel(libre, tolerance=0.5)
        except Exception:  # noqa: BLE001
            p = libre.representative_point()
        x, y = dib.a_px(p.x, p.y)
        t1_, t2_ = f"LOTE {l['id']}", f"{_fmt(l['area'])} m²"
        w = max(d.textlength(t1_, font=f1), d.textlength(t2_, font=f2)) + tam * 1.0
        h = tam * 2.45
        caja_l = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        d.rounded_rectangle(caja_l, radius=tam * 0.35, fill=(255, 255, 255, 238), outline=(70, 50, 34), width=2)
        d.text((x, y - h * 0.2), t1_, font=f1, fill=(30, 30, 30), anchor="mm")
        d.text((x, y + h * 0.24), t2_, font=f2, fill=(70, 55, 40), anchor="mm")
        ocupado.append(box(*caja_l))

    # calle, retorno, portón
    if lot.calles:
        calle = unary_union(lot.calles)
        ancho_c = lot.parametros.get("calle", 8)
        boca = dib.extra.get("boca")
        eje_ang = lot.angulo
        nombre_c = "CALLE CENTRAL" if st.get("modo") == "calle_central" else "CALLE"
        tam_c = max(13, min(30, m * 2.6))
        if boca is not None and st.get("modo") == "calle_central" and calle.geom_type == "Polygon":
            ret = st.get("retorno_centro")
            lejos = Point(ret) if ret else max((Point(c) for c in calle.exterior.coords), key=lambda q: q.distance(Point(boca)))
            espina = LineString([boca, (lejos.x, lejos.y)])
            p = calle.intersection(espina.interpolate(0.42, normalized=True).buffer(ancho_c)).centroid
            ang = math.degrees(math.atan2(lejos.y - boca[1], lejos.x - boca[0]))
        else:
            espina, ang = None, eje_ang
            try:
                p = polylabel(max(_poligonos(calle), key=lambda g: g.area), tolerance=0.5)
            except Exception:  # noqa: BLE001
                p = calle.representative_point()
        _texto_girado(img, dib.a_px(p.x, p.y), f"{nombre_c} · {ancho_c:.2f} m", tam_c, ang, ocupado=ocupado)
        if st.get("retorno_diametro_m") and espina is not None:
            rr = st["retorno_diametro_m"] / 2
            ctr = Point(st["retorno_centro"]) if st.get("retorno_centro") else espina.interpolate(max(0.0, espina.length - rr))
            ux, uy = (lejos.x - boca[0]) / espina.length, (lejos.y - boca[1]) / espina.length
            dd = rr + 14
            candidatos = [(ctr.x - uy * dd, ctr.y + ux * dd), (ctr.x + uy * dd, ctr.y - ux * dd),
                          (ctr.x - ux * dd - uy * dd * 0.6, ctr.y - uy * dd + ux * dd * 0.6),
                          (ctr.x - ux * dd + uy * dd * 0.6, ctr.y - uy * dd - ux * dd * 0.6)]
            try:
                adentro = unary_union([g.buffer(0) for g in [l["poly"] for l in lot.lotes] + list(lot.calles)]).buffer(-2)
                candidatos = [q for q in candidatos if adentro.contains(Point(q))] or candidatos
            except Exception:  # noqa: BLE001  (geometría con astillas: se usan los candidatos tal cual)
                pass
            evitar = []
            for v in sorted(lot.verdes, key=lambda v: -v["area"])[:1]:
                try:
                    evitar.append(polylabel(v["poly"], tolerance=0.5))
                except Exception:  # noqa: BLE001
                    evitar.append(v["poly"].representative_point())
            dest = max(candidatos, key=lambda q: min((Point(q).distance(o) for o in evitar), default=0))
            a, b = dib.a_px(ctr.x, ctr.y), dib.a_px(*dest)
            d.line([a, b], fill=(40, 40, 40, 200), width=2)
            _texto(d, b, f"REDONDEL DE RETORNO\nØ {st['retorno_diametro_m']:.0f} m", max(12, min(24, m * 2.0)), True,
                   ocupado=ocupado)
        if dib.extra.get("porton"):
            bx, by = dib.extra["porton"]
            x, y = dib.a_px(bx, by)
            _texto(d, (x + m * 3.5, y), "PORTÓN Y\nVIGILANCIA", max(11, min(24, m * 2.0)), True, anchor="lm",
                   ocupado=ocupado)

    # área verde y protección
    for v in sorted(lot.verdes, key=lambda v: -v["area"])[:1]:
        if v["area"] > 60:
            try:
                p = polylabel(v["poly"], tolerance=0.5)
            except Exception:  # noqa: BLE001
                p = v["poly"].representative_point()
            x, y = dib.a_px(p.x, p.y)
            _texto(d, (x, y), f"ÁREA VERDE\nRECREATIVA\n{_fmt(st['area_verde'])} m²", max(11, min(26, m * 2.2)), True,
                   color=(28, 80, 28), ocupado=ocupado)
    if lot.protecciones:
        prot = proteccion_de_quebrada(lot.protecciones, _anillo_de(lot), ficha)
        try:
            p = polylabel(prot, tolerance=0.5)
        except Exception:  # noqa: BLE001
            p = prot.representative_point()
        x, y = dib.a_px(p.x, p.y)
        franja = ""
        for q in ficha.get("quebradas") or []:
            if q.get("profundidad_m"):
                franja = f" · franja {1.5 * float(q['profundidad_m']):.0f} m"
                break
        _texto(d, (x, y), f"ZONA PROTEGIDA DE QUEBRADA{franja}", max(11, min(26, m * 2.2)), True,
               color=(30, 90, 30) if tec else (255, 255, 255), halo=(255, 255, 255) if tec else (30, 70, 30),
               ocupado=ocupado)
    if dib.extra.get("quebrada") is not None:
        q = dib.extra["quebrada"].interpolate(0.5, normalized=True)
        x, y = dib.a_px(q.x, q.y)
        _texto(d, (x, y), "Quebrada", max(11, min(24, m * 2.0)), False, color=(20, 70, 140), ocupado=ocupado)
    if dib.extra.get("acceso") is not None:
        a = dib.extra["acceso"]
        p = a.representative_point()
        x, y = dib.a_px(p.x, p.y)
        _texto(d, (x, y), "CALLE DE ACCESO", max(11, min(24, m * 2.0)), True, color=(60, 60, 60), ocupado=ocupado)

    # norte y escala
    zx0, zy0, zx1, zy1 = dib.zona
    R = (zy1 - zy0) * 0.045
    kx, ky = zx1 - R * 2.2, zy0 + R * 2.2
    d.ellipse([kx - R, ky - R, kx + R, ky + R], fill=(255, 255, 255, 220), outline=(40, 40, 40), width=2)
    d.polygon([(kx, ky - R * 0.85), (kx - R * 0.35, ky + R * 0.5), (kx, ky + R * 0.2), (kx + R * 0.35, ky + R * 0.5)],
              fill=(30, 30, 30))
    _texto(d, (kx, ky - R * 1.35), "N", R * 0.6, True)
    paso = next(v for v in (5, 10, 20, 25, 50, 100, 200) if v * m >= W * 0.06)
    x0, y0 = zx0 + W * 0.03, zy1 - (zy1 - zy0) * 0.05
    for i in range(4):
        d.rectangle([x0 + i * paso * m, y0, x0 + (i + 1) * paso * m, y0 + 10], fill=(30, 30, 30) if i % 2 == 0 else (255, 255, 255),
                    outline=(30, 30, 30))
    for i in range(5):
        _texto(d, (x0 + i * paso * m, y0 - 14), f"{i * paso}", 16, False)
    _texto(d, (x0 + 4 * paso * m + 10, y0 + 5), "m", 16, False, anchor="lm")
    ocupado += [box(kx - R, ky - R * 1.8, kx + R, ky + R), box(x0 - 10, y0 - 26, x0 + 4 * paso * m + 30, y0 + 14)]

    # nombre y área de cada terreno unido, afuera de su lado exterior
    for u in unidos:
        for q in u["candidatos"]:
            x, y = dib.a_px(*q)
            if not (dib.zona[0] + 90 < x < dib.zona[2] - 90 and dib.zona[1] + 40 < y < dib.zona[3] - 40):
                continue
            txt = f"{u['nombre']}\n{_fmt(u['area'])} m²"
            f = _fuente(max(14, min(26, m * 2.2)), True)
            caja = box(*d.multiline_textbbox((x, y), txt, font=f, anchor="mm", align="center", stroke_width=3))
            if any(caja.intersects(o) for o in ocupado) or (prop_ is not None and caja.intersects(prop_)):
                continue
            _texto(d, (x, y), txt, max(14, min(26, m * 2.2)), True, (30, 30, 30), ocupado=ocupado)
            break

    # medidas de cada lindero de lote (las compartidas, una sola vez), sin encimarse a los letreros
    tam_m = max(12, min(20, m * 1.7))
    vistos: list = []
    for l in lot.lotes:
        for tr in _tramos(l["poly"]):
            if tr.length < 3 or any(tr.hausdorff_distance(v) < 0.6 for v in vistos):
                continue
            vistos.append(tr)
            txt = f"{tr.length:.2f} m"
            w, h = _medida_cota(txt, tam_m)
            if tr.length * m < w + tam_m * 1.5:
                continue
            for fr in (0.5, 0.38, 0.62, 0.28, 0.72):
                a, b = tr.interpolate(max(0.0, fr * tr.length - 1.0)), tr.interpolate(min(tr.length, fr * tr.length + 1.0))
                ang = ((math.degrees(math.atan2(b.y - a.y, b.x - a.x)) + 90) % 180) - 90
                q = tr.interpolate(fr, normalized=True)
                xy = dib.a_px(q.x, q.y)
                huella = _caja_girada(xy, w, h, ang)
                if not (dib.zona[1] < xy[1] < dib.zona[3]) or any(huella.intersects(o) for o in ocupado):
                    continue
                _cota(img, xy, txt, tam_m, ang)
                ocupado.append(huella.buffer(3))
                break

    # simbología y aviso
    yb = dib.zona[3]
    hb = H - yb
    ids = modelo3d.ID_CLASE
    msk = dib.extra.get("mascara")

    def color_de(cl, defecto):
        if not ia or msk is None:
            return defecto
        m = msk == ids[cl]
        if m.sum() < 30:
            return defecto
        a = np.asarray(img.convert("RGB"))[m]
        return tuple(int(x) for x in np.median(a, axis=0))

    if tec:
        items = [(T["lote"], "Lote"), (T["techo"], "Casa tipo (ilustrativa)"), (T["calle"], "Calle"),
                 (T["verde"], "Área verde recreativa"), (T["proteccion"], "Zona protegida")]
    else:
        items = [(color_de("lote", C["lote"]), "Lote"), (C["techo"], "Casa tipo (ilustrativa)"),
                 (color_de("calle", C["calle"]), "Calle"), (color_de("verde", C["verde"]), "Área verde recreativa"),
                 (color_de("proteccion", C["proteccion"]), "Zona protegida")]
    tl = max(12, int(hb * 0.16))
    while tl > 10:
        f = _fuente(tl)
        largo = sum(tl * 2.6 + d.textlength(txt, font=f) + tl * 1.2 for _, txt in items)
        if largo < W * 0.94:
            break
        tl -= 1
    x = W * 0.03
    for col, txt in items:
        d.rectangle([x, yb + hb * 0.22, x + tl * 1.6, yb + hb * 0.22 + tl * 1.1], fill=col, outline=(60, 60, 60))
        if txt == "Lote" and not tec:
            d.rectangle([x + 3, yb + hb * 0.22 + 3, x + tl * 1.6 - 3, yb + hb * 0.22 + tl * 1.1 - 3], outline=(255, 255, 255),
                        width=2)
        d.text((x + tl * 2.0, yb + hb * 0.22 + tl * 0.55), txt, font=f, fill=(40, 40, 40), anchor="lm")
        x += tl * 2.6 + d.textlength(txt, font=f) + tl * 1.2
    if tec:
        nota = ("Plano de presentación dibujado por el motor con las medidas del prediseño; casas y árboles ilustrativos. "
                "No válido para trámite.")
    else:
        nota = ("Imagen ilustrativa del prediseño" + (" (pintada con IA sobre el trazo exacto)" if ia else "")
                + ". Las medidas y áreas válidas están en la lámina (PDF). No válida para trámite.")
    tn = max(12, int(hb * 0.13))
    while tn > 10 and d.textlength(nota, font=_fuente(tn)) > W * 0.94:
        tn -= 1
    d.text((W * 0.03, yb + hb * 0.68), nota, font=_fuente(tn), fill=(160, 40, 30), anchor="lm")
    return img.convert("RGB")


def rotular_aerea(img: Image.Image, dib: Dibujo, lot: Resultado, ficha: dict, proyecto: str,
                  partes: list | None = None, ia: bool = False) -> Image.Image:
    img = img.convert("RGBA")
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    alt = dib.extra["alt"]
    tam = max(14, int(W * 0.0105))
    for l in lot.lotes:
        p = polylabel(l["poly"], tolerance=0.5) if l["poly"].area > 1 else l["poly"].centroid
        x, y = dib.a_px(p.x, p.y, alt(p.x, p.y) + 7)
        if not (0 <= x < W and 0 <= y < H):
            continue
        txt = f"{l['id']} · {_fmt(l['area'])} m²"
        f = _fuente(tam, True)
        tw = d.textlength(txt, font=f)
        d.rounded_rectangle([x - tw / 2 - 8, y - tam * 0.75, x + tw / 2 + 8, y + tam * 0.75], radius=tam * 0.5,
                            fill=(255, 255, 255, 225), outline=(90, 70, 50), width=2)
        d.text((x, y), txt, font=f, fill=(30, 30, 30), anchor="mm")
    if lot.protecciones:
        prot = proteccion_de_quebrada(lot.protecciones, _anillo_de(lot), ficha)
        p = prot.representative_point()
        x, y = dib.a_px(p.x, p.y, alt(p.x, p.y) + 4)
        if 0 <= x < W and 0 <= y < H:
            _texto(d, (x, y), "ZONA PROTEGIDA DE QUEBRADA", tam, True, color=(255, 255, 255), halo=(30, 70, 30))
    t1, t2 = _titulo(lot, ficha, partes)
    d.rectangle([0, H - H * 0.11, W, H], fill=(255, 255, 255, 230))
    _texto(d, (W * 0.02, H - H * 0.075), t1, H * 0.03, True, anchor="lm", borde=0)
    nota = f"{proyecto} · {t2} · Vista ilustrativa" + (" pintada con IA sobre el trazo exacto" if ia else "") + \
        " · relieve satelital aproximado · no válida para trámite"
    _texto(d, (W * 0.02, H - H * 0.03), nota, H * 0.017, False, (110, 50, 40), anchor="lm", borde=0)
    modelo3d.flecha_norte(d, dib.extra["pantalla"], W - 80, 80, 46)
    return img.convert("RGB")


def rotular_ambiente(img: Image.Image, lot: Resultado, ficha: dict, proyecto: str, partes: list | None = None,
                     vista: str = "") -> Image.Image:
    """Imagen de ambiente (libre, pintada por la IA): solo el título con los números del motor y el aviso de que es
    ilustrativa. No lleva medidas ni números de lote, porque la IA no respeta las posiciones exactas."""
    img = img.convert("RGBA")
    W, H = img.size
    capa = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dc = ImageDraw.Draw(capa)
    alto = int(H * 0.17)
    for i in range(alto):  # degradado oscuro abajo para que el texto se lea sobre la foto
        a = int(200 * (i / alto) ** 1.4)
        dc.line([(0, H - alto + i), (W, H - alto + i)], fill=(10, 14, 10, a))
    img.alpha_composite(capa)
    d = ImageDraw.Draw(img, "RGBA")
    t1, t2 = _titulo(lot, ficha, partes)
    _texto(d, (W * 0.025, H - H * 0.085), t1, H * 0.036, True, (255, 255, 255), anchor="lm", borde=0)
    _texto(d, (W * 0.025, H - H * 0.04), f"{proyecto} · {t2}", H * 0.021, False, (235, 235, 230), anchor="lm", borde=0)
    aviso = (f"{vista.upper() + ' · ' if vista else 'IMAGEN DE AMBIENTE · '}ILUSTRATIVA, NO A ESCALA · las medidas "
             "exactas están en el plano y la lámina")
    ta = H * 0.017
    f = _fuente(int(ta), True)
    tw = d.textlength(aviso, font=f)
    d.rounded_rectangle([W - tw - W * 0.025 - 24, H * 0.025, W - W * 0.025, H * 0.025 + ta * 2.2], radius=ta,
                        fill=(255, 255, 255, 215))
    d.text((W - W * 0.025 - 12, H * 0.025 + ta * 1.1), aviso, font=f, fill=(150, 40, 30), anchor="rm")
    return img.convert("RGB")


# --------------------------------------------------------------------------------------------- todo junto
def generar(carpeta: str, base: str, t: Terreno, lot: Resultado, malla: Malla | None, ficha: dict | None,
            proyecto: str, partes: list | None = None, pintor: Callable | None = None, ubicacion: str = "") -> list[dict]:
    """Imágenes de presentación, en el orden en que se mandan:
    - ambiente: vista aérea pintada libre por la IA sobre el 3D del motor (ilustrativa, sin medidas). Si no hay modelo
      o falla, va la vista 3D exacta del motor con sus letreros (vista "aerea"). Necesita relieve.
    - planta: el trazo exacto pintado por la IA (revisado), con linderos, medidas y letreros del motor.
    - tecnica: lámina de arquitecto en grises (marco, cuadros de áreas y de lotes, simbología, cajetín) dibujada solo
      por el motor (motor/plano_tecnico.py, no usa IA).
    - altimetria: lámina de relieve (pendientes por color, curvas con cota, escurrimiento, perfil de la calle y relieve
      por lote), solo con relieve (motor/plano_altimetria.py, no usa IA).
    Devuelve [{ruta, nombre, titulo, vista, ia, ...}]."""
    ficha = ficha or {}
    calcos = {}
    cams = {}
    if malla is not None:
        try:
            cams = camaras(t, lot, malla, ficha)
        except Exception:  # noqa: BLE001
            log.exception("cámaras")
    for vista, fn in (("ambiente", (lambda: calco_aereo(t, lot, malla)) if malla is not None else None),
                      ("porton", (lambda: calco_vista(t, lot, malla, cams["porton"])) if "porton" in cams else None),
                      ("quebrada", (lambda: calco_vista(t, lot, malla, cams["quebrada"])) if "quebrada" in cams else None),
                      ("limpios", (lambda: calco_aereo(t, lot, malla, casas=False)) if malla is not None else None),
                      ("planta", lambda: calco_planta(t, lot, malla, ficha)),
                      ("tecnica", lambda: None),
                      ("altimetria", (lambda: None) if malla is not None else None),
                      ("corte", (lambda: None) if malla is not None and lot.stats.get("modo") == "calle_central" else None)):
        if fn is None:
            continue
        try:
            calcos[vista] = fn()
        except Exception:  # noqa: BLE001
            log.exception("calco %s", vista)
    if "ambiente" in calcos:
        calcos["ambiente"].vista = "ambiente"
    with ThreadPoolExecutor(max_workers=5) as ex:
        # la planta se revisa contra el trazo (2 intentos); las vistas de venta son libres (1 intento)
        futuros = {}
        if "planta" in calcos:
            futuros["planta"] = ex.submit(pintar, calcos["planta"], pintor, INSTRUCCION_PLANTA, 2)
        for v_, ins_ in (("porton", INSTRUCCION_PORTON), ("quebrada", INSTRUCCION_QUEBRADA), ("limpios", INSTRUCCION_LIMPIOS)):
            if v_ in calcos and calcos[v_] is not None and pintor is not None:
                dib_v = calcos[v_]
                try:
                    from . import altimetria as _alt

                    ins_ += _alt.guia_render(malla, t.puntos, dib_v.a_px, *dib_v.imagen.size)
                except Exception:  # noqa: BLE001
                    log.exception("guía de relieve %s", v_)
                futuros[v_] = ex.submit(pintar, dib_v, pintor, ins_, 1, False)
        if "ambiente" in calcos:
            ins = INSTRUCCION_AMBIENTE + guia_ambiente(calcos["ambiente"], lot)
            try:  # el relieve dicho en términos de la imagen, para que la IA no aplane el terreno
                from . import altimetria

                dib_a = calcos["ambiente"]
                barr = altimetria.barrancos(malla, t.puntos) if ficha.get("respetar_barrancos", True) else []
                ins += altimetria.guia_render(malla, t.puntos, dib_a.a_px, *dib_a.imagen.size, barr)
            except Exception:  # noqa: BLE001
                log.exception("guía de relieve del ambiente")
            futuros["ambiente"] = ex.submit(pintar, calcos["ambiente"], pintor, ins, 1, False)
        pintadas = {v: f.result() for v, f in futuros.items()}
    out = []
    nombres_v = {"porton": ("Vista desde el portón", "Vista desde el portón (ilustrativa)"),
                 "quebrada": ("Vista desde la quebrada", "Vista desde la quebrada (ilustrativa)"),
                 "limpios": ("Lotes listos, antes de construir", "Lotes listos con su terracería (ilustrativa)")}
    for vista in ("ambiente", "porton", "quebrada", "limpios", "planta", "tecnica", "altimetria", "corte"):
        if vista not in calcos:
            continue
        dib = calcos[vista]
        try:
            if vista in nombres_v:
                if vista not in pintadas:
                    continue
                img, info = pintadas[vista]
                if not info["ia"]:  # sin IA no se manda la maqueta: no aporta como imagen de venta
                    continue
                final = rotular_ambiente(img, lot, ficha, proyecto, partes, vista=nombres_v[vista][0])
                titulo = nombres_v[vista][1]
            elif vista == "corte":
                from .plano_corte import generar_corte

                info = {"ia": False}
                final, _ = generar_corte(t, lot, malla, ficha, proyecto, ubicacion)
                titulo = "Corte transversal y terracería por lote"
            elif vista == "ambiente":
                img, info = pintadas[vista]
                if info["ia"]:
                    final = rotular_ambiente(img, lot, ficha, proyecto, partes)
                    titulo = "Imagen de ambiente (ilustrativa, no a escala)"
                else:  # sin IA: la vista 3D exacta del motor con sus letreros
                    vista = "aerea"
                    final = rotular_aerea(img, dib, lot, ficha, proyecto, partes, ia=False)
                    titulo = "Vista aérea de presentación (ilustrativa)"
            elif vista == "planta":
                img, info = pintadas[vista]
                final = rotular_planta(img, dib, lot, ficha, proyecto, partes, ia=info["ia"])
                titulo = "Plano de presentación con medidas (ilustrativo)"
            elif vista == "tecnica":
                from .plano_tecnico import generar_tecnico

                info = {"ia": False}
                final = generar_tecnico(t, lot, malla, ficha, proyecto, partes, ubicacion)
                titulo = "Plano técnico de presentación (lámina)"
            else:
                from .plano_altimetria import generar_altimetria

                info = {"ia": False}
                final = generar_altimetria(t, lot, malla, ficha, proyecto, partes, ubicacion)
                titulo = "Plano altimétrico: pendientes, curvas de nivel y perfil de la calle"
        except Exception:  # noqa: BLE001
            log.exception("presentación %s", vista)
            continue
        ruta = os.path.join(carpeta, f"{base}-presentacion-{vista}.jpg")
        final.save(ruta, "JPEG", quality=90)
        out.append({"ruta": ruta, "nombre": os.path.basename(ruta), "titulo": titulo, "vista": vista, **info})
    return out
