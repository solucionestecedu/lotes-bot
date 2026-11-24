"""Imagen de presentación del anteproyecto (tipo plano de ventas), en planta y en vista aérea.

"La IA pinta, el motor mide":
1. Calco: el motor dibuja la geometría exacta (lotes, calle, retorno, portón, área verde, quebrada, casas tipo,
   árboles), sin ningún texto.
2. Un modelo de imagen (Nano Banana) solo lo pinta, con la instrucción de no mover nada ni escribir nada.
3. Se compara la imagen pintada con el calco: si movió linderos o se comió lotes, se descarta.
4. Los letreros y medidas (número y área de cada lote, calle, retorno, área verde, quebrada, norte, escala) se
   ponen aquí encima, con los números del motor. Si no hay modelo o falla, se entrega el calco con sus letreros.
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
from .lotificacion import LADOS, Resultado, _bordes_hacia, _poligonos
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
                 lado_largo: int = LADO_LARGO) -> Dibujo:
    ficha = ficha or {}
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

    img = Image.new("RGB", (W * S, H * S), C["papel"])
    d = ImageDraw.Draw(img, "RGBA")
    m2px = k * S  # píxeles (supermuestreo) por metro

    def poli(g, fill=None, outline=None, ancho_m=0.0, min_px=1):
        for ext, hoyos in _anillos(g):
            pts = [px(*p) for p in ext]
            if fill is not None:
                d.polygon(pts, fill=fill)
                for hoyo in hoyos:
                    d.polygon([px(*p) for p in hoyo], fill=C["papel"])
            if outline is not None:
                d.line(pts + [pts[0]], fill=outline, width=max(min_px, int(ancho_m * m2px)), joint="curve")

    def linea(coords, color, ancho_m, min_px=1):
        d.line([px(*p) for p in coords], fill=color, width=max(min_px, int(ancho_m * m2px)), joint="curve")

    rng = random.Random(len(t.puntos) * 7919 + int(prop.area))
    dibujo_geo = box(cx - (W / 2) / k, cy - (oy - zona[1]) / k, cx + (W / 2) / k, cy + (zona[3] - oy) / k)

    # entorno (solo en la zona de dibujo) y curvas de nivel
    d.rectangle([0, zona[1] * S, W * S, zona[3] * S], fill=C["entorno"])
    if malla is not None:
        try:
            st = estadisticas(malla, t.puntos)
            iv = intervalo_curvas(st["desnivel"])
            for c in curvas(malla, iv, recorte=dibujo_geo):
                for ln in c["lineas"]:
                    linea(ln, C["curva"] + ((150,) if c["indice"] else (90,)), 0.25 if c["indice"] else 0.12, 1)
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
            poli(via, fill=C["acceso"])
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
            poli(monte, fill=C["proteccion"])
            for p in _puntos_dentro(monte, int(monte.area / 40), rng, 3.5):
                _arbol(d, px(*p), rng.uniform(2.2, 3.4) * m2px, rng)
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
                    linea(ondula, C["agua"], 1.8, 3)
            extra["quebrada"] = max(agua, key=lambda a: a.length) if agua else None

    # terreno, protección, área verde, lotes
    poli(prop, fill=C["terreno"])
    for p in lot.protecciones:
        poli(p, fill=C["proteccion"])
        for q in _puntos_dentro(p, int(p.area / 28), rng, 3.0):
            _arbol(d, px(*q), rng.uniform(2.0, 3.2) * m2px, rng)
    for v in lot.verdes:
        poli(v["poly"], fill=C["verde"])
    for e in lot.equipamiento:
        poli(e["poly"], fill=(206, 220, 236))
    for l in lot.lotes:
        poli(l["poly"], fill=C["lote"])

    # calles con cordón
    calles = unary_union(lot.calles) if lot.calles else None
    for c_ in lot.calles:
        poli(c_, fill=C["calle"], outline=C["cordon"], ancho_m=0.35, min_px=2)

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
            poli(camino, fill=C["camino"])
    for l, casa in casas:
        sombra = Polygon([(x + 1.0, y - 1.0) for x, y in casa.exterior.coords])
        poli(sombra, fill=C["sombra"])
        poli(casa, fill=C["techo"], outline=C["cumbrera"], ancho_m=0.12, min_px=1)
        esq = list(casa.exterior.coords)[:4]
        m0 = ((esq[0][0] + esq[3][0]) / 2, (esq[0][1] + esq[3][1]) / 2)
        m1 = ((esq[1][0] + esq[2][0]) / 2, (esq[1][1] + esq[2][1]) / 2)
        linea([m0, m1], C["cumbrera"], 0.18, 1)
    for l in lot.lotes:
        libre = l["poly"].buffer(-1.5)
        for _, casa in casas:
            libre = libre.difference(casa.buffer(2.5))
        if calles is not None:
            libre = libre.difference(calles.buffer(6))
        for q in _puntos_dentro(libre, max(1, min(4, int(l["area"] / 350))), rng, 6):
            _arbol(d, px(*q), rng.uniform(1.8, 2.8) * m2px, rng)
    for v in lot.verdes:
        for q in _puntos_dentro(v["poly"].buffer(-1.5), int(v["area"] / 90), rng, 6):
            _arbol(d, px(*q), rng.uniform(2.0, 3.0) * m2px, rng)

    # linderos de lotes y del terreno
    for l in lot.lotes:
        poli(l["poly"], outline=C["borde_lote"], ancho_m=0.22, min_px=2)
    poli(prop, outline=C["lindero"], ancho_m=0.55, min_px=3)

    # portón y caseta (solo dibujo) donde la calle entra al terreno
    if ficha.get("cerrado") and calles is not None:
        boca = calles.boundary.intersection(prop.exterior.buffer(0.4))
        tramos = [g for g in getattr(boca, "geoms", [boca]) if g.geom_type == "LineString" and g.length > 1]
        if tramos:
            boca = max(tramos, key=lambda g: g.length)
            pa, pb = boca.coords[0], boca.coords[-1]
            mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
            hacia = nearest_points(Point(mid), calles.buffer(-2.5))[1] if not calles.buffer(-2.5).is_empty else calles.centroid
            nx_, ny_ = hacia.x - mid[0], hacia.y - mid[1]
            nl = math.hypot(nx_, ny_) or 1
            nx_, ny_ = nx_ / nl, ny_ / nl
            gate = LineString([(pa[0] + nx_ * 4, pa[1] + ny_ * 4), (pb[0] + nx_ * 4, pb[1] + ny_ * 4)])
            gate = gate.intersection(calles)
            if not gate.is_empty:
                linea(list(getattr(gate, "coords", [])) or list(gate.geoms[0].coords), C["porton"], 0.6, 3)
            ex, ey = pa[0] - pb[0], pa[1] - pb[1]
            el = math.hypot(ex, ey) or 1
            ex, ey = ex / el, ey / el
            bx, by = pa[0] + nx_ * 6.5 - ex * 1.6, pa[1] + ny_ * 6.5 - ey * 1.6
            caseta = Polygon([(bx + sx * 1.3 * ex + sy * 1.6 * nx_, by + sx * 1.3 * ey + sy * 1.6 * ny_)
                              for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            poli(caseta, fill=C["caseta"], outline=C["porton"], ancho_m=0.15, min_px=2)
            extra["porton"] = (bx, by)
            extra["boca"] = mid

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
    return Dibujo("planta", img, aspecto, a_px, linderos, 1 / k, zona, extra)


def _arbol(d: ImageDraw.ImageDraw, q, rr, rng):
    s = rr * 0.35
    d.ellipse([q[0] - rr + s, q[1] - rr + s, q[0] + rr + s, q[1] + rr + s], fill=(40, 50, 30, 60))
    col = C["arbol"] if rng.random() < 0.5 else C["arbol2"]
    d.ellipse([q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr], fill=col, outline=C["borde_arbol"])
    r2 = rr * 0.45
    d.ellipse([q[0] - r2 - rr * 0.2, q[1] - r2 - rr * 0.25, q[0] + r2 - rr * 0.2, q[1] + r2 - rr * 0.25],
              fill=tuple(min(255, c + 22) for c in col))


# --------------------------------------------------------------------------------------------- aérea
def calco_aereo(t: Terreno, lot: Resultado, malla: Malla, lado_largo: int = LADO_LARGO) -> Dibujo:
    escena = modelo3d.construir(t, malla, lot, margen=10.0)
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


# --------------------------------------------------------------------------------------------- pintar
INSTRUCCION_PLANTA = (
    "You are given an exact top-down site plan of a small country-house (quinta) subdivision in El Salvador. "
    "Repaint it as a polished, professional architectural sales-brochure illustration, seen straight from above "
    "(orthographic). STRICT RULES: keep the composition pixel-aligned with the input. Every property line, lot "
    "division, road edge, roundabout, house footprint, tree and area must stay exactly where it is, with the same "
    "size and shape. Do not add, remove, move, merge or resize any lot, house, road or area. Do not write any text, "
    "letters, numbers, labels, logos or watermarks anywhere. Keep the plain paper bands at the top and at the "
    "bottom empty. Style: soft natural daylight from the upper left with gentle shadows; lots are well-kept lawns; "
    "houses have terracotta clay-tile roofs seen from above; roads are grey asphalt with light curbs; the light green "
    "area is a recreational park lawn with tropical trees; the dark green strip is dense protected ravine forest; "
    "the blue line is a small creek; outside the property, a muted rural landscape. Crisp and realistic, not cartoonish."
)
INSTRUCCION_AEREA = (
    "You are given an exact 3D aerial view of a small country-house (quinta) subdivision on a hillside in El Salvador. "
    "Repaint it as a photorealistic architectural aerial rendering. STRICT RULES: keep the same camera, horizon and "
    "composition, pixel-aligned with the input. Every lot boundary, road, roundabout, house and green area must stay "
    "exactly where it is, with the same size and shape. Do not add, remove, move or resize any lot, house, road or "
    "area. Do not write any text, letters, numbers, labels, logos or watermarks. Style: late-morning tropical light, "
    "lush green lawns on the lots, simple modern one-storey houses with terracotta roofs, grey asphalt road, dense "
    "tropical forest in the dark green protected strip, soft blue sky."
)


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


def pintar(dibujo: Dibujo, pintor: Callable | None, instruccion: str) -> tuple[Image.Image, dict]:
    """Devuelve (imagen, info). Si no hay pintor, falla o mueve el dibujo, regresa el calco."""
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
def _texto(d, xy, txt, tam, negrita=False, color=(30, 30, 30), halo=(255, 255, 255), anchor="mm", borde=None):
    f = _fuente(tam, negrita)
    d.text(xy, txt, font=f, fill=color, anchor=anchor, stroke_width=borde if borde is not None else max(2, int(tam / 7)),
           stroke_fill=halo, align="center")
    return f


def _texto_girado(img, xy, txt, tam, angulo_grados, negrita=True, color=(30, 30, 30)):
    f = _fuente(tam, negrita)
    caja = f.getbbox(txt, stroke_width=3)
    w, h = caja[2] - caja[0] + 12, caja[3] - caja[1] + 12
    capa = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(capa).text((w / 2, h / 2), txt, font=f, fill=color + (255,), anchor="mm", stroke_width=3,
                              stroke_fill=(255, 255, 255, 235))
    a = ((angulo_grados + 90) % 180) - 90  # que no quede de cabeza
    capa = capa.rotate(a, expand=True, resample=Image.BICUBIC)
    img.alpha_composite(capa, (int(xy[0] - capa.width / 2), int(xy[1] - capa.height / 2)))


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
                   partes: list | None = None, ia: bool = False) -> Image.Image:
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

    # lotes: número y área
    tam = max(14, min(40, m * 4.2))
    for l in lot.lotes:
        poly = l["poly"]
        casa = dib.extra.get("casas", {}).get(l["id"])
        libre = poly.buffer(-1.0)
        if casa is not None:
            libre = libre.difference(casa.buffer(1.5))
        libre = max(_poligonos(libre), key=lambda g: g.area, default=poly)
        try:
            p = polylabel(libre, tolerance=0.5)
        except Exception:  # noqa: BLE001
            p = libre.representative_point()
        x, y = dib.a_px(p.x, p.y)
        r = tam * 0.8
        cyc = y - r * 0.9
        d.ellipse([x - r, cyc - r, x + r, cyc + r], fill=(255, 255, 255, 240), outline=(80, 60, 40), width=2)
        _texto(d, (x, cyc), str(l["id"]), tam * 0.95, True, borde=0)
        _texto(d, (x, y + tam * 0.6), f"{_fmt(l['area'])} m²", tam * 0.72, True)

    # calle, retorno, portón
    if lot.calles:
        calle = unary_union(lot.calles)
        ancho_c = lot.parametros.get("calle", 8)
        boca = dib.extra.get("boca")
        eje_ang = lot.angulo
        nombre_c = "CALLE CENTRAL" if st.get("modo") == "calle_central" else "CALLE"
        tam_c = max(13, min(30, m * 2.6))
        if boca is not None:
            lejos = max((Point(c) for c in calle.exterior.coords), key=lambda q: q.distance(Point(boca)))
            espina = LineString([boca, (lejos.x, lejos.y)])
            p = calle.intersection(espina.interpolate(0.42, normalized=True).buffer(ancho_c)).centroid
            ang = math.degrees(math.atan2(lejos.y - boca[1], lejos.x - boca[0]))
        else:
            espina, ang = None, eje_ang
            try:
                p = polylabel(max(_poligonos(calle), key=lambda g: g.area), tolerance=0.5)
            except Exception:  # noqa: BLE001
                p = calle.representative_point()
        _texto_girado(img, dib.a_px(p.x, p.y), f"{nombre_c} · {ancho_c:.2f} m", tam_c, ang)
        if st.get("retorno_diametro_m") and espina is not None:
            rr = st["retorno_diametro_m"] / 2
            ctr = espina.interpolate(max(0.0, espina.length - rr))
            ux, uy = (lejos.x - boca[0]) / espina.length, (lejos.y - boca[1]) / espina.length
            candidatos = [(ctr.x - uy * (rr + 14), ctr.y + ux * (rr + 14)), (ctr.x + uy * (rr + 14), ctr.y - ux * (rr + 14))]
            ocupado = []
            for v in sorted(lot.verdes, key=lambda v: -v["area"])[:1]:
                try:
                    ocupado.append(polylabel(v["poly"], tolerance=0.5))
                except Exception:  # noqa: BLE001
                    ocupado.append(v["poly"].representative_point())
            dest = max(candidatos, key=lambda q: min((Point(q).distance(o) for o in ocupado), default=0))
            a, b = dib.a_px(ctr.x, ctr.y), dib.a_px(*dest)
            d.line([a, b], fill=(40, 40, 40, 200), width=2)
            _texto(d, b, f"REDONDEL DE RETORNO\nØ {st['retorno_diametro_m']:.0f} m", max(12, min(24, m * 2.0)), True)
        if dib.extra.get("porton"):
            bx, by = dib.extra["porton"]
            x, y = dib.a_px(bx, by)
            _texto(d, (x + m * 3.5, y), "PORTÓN Y\nVIGILANCIA", max(11, min(24, m * 2.0)), True, anchor="lm")

    # área verde y protección
    for v in sorted(lot.verdes, key=lambda v: -v["area"])[:1]:
        if v["area"] > 60:
            try:
                p = polylabel(v["poly"], tolerance=0.5)
            except Exception:  # noqa: BLE001
                p = v["poly"].representative_point()
            x, y = dib.a_px(p.x, p.y)
            _texto(d, (x, y), f"ÁREA VERDE\nRECREATIVA\n{_fmt(st['area_verde'])} m²", max(11, min(26, m * 2.2)), True,
                   color=(28, 80, 28))
    if lot.protecciones:
        prot = max(lot.protecciones, key=lambda g: g.area)
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
        _texto(d, (x, y), f"ZONA PROTEGIDA DE QUEBRADA{franja}", max(11, min(26, m * 2.2)), True, color=(255, 255, 255),
               halo=(30, 70, 30))
    if dib.extra.get("quebrada") is not None:
        q = dib.extra["quebrada"].interpolate(0.5, normalized=True)
        x, y = dib.a_px(q.x, q.y)
        _texto(d, (x, y), "Quebrada", max(11, min(24, m * 2.0)), False, color=(20, 70, 140))
    if dib.extra.get("acceso") is not None:
        a = dib.extra["acceso"]
        p = a.representative_point()
        x, y = dib.a_px(p.x, p.y)
        _texto(d, (x, y), "CALLE DE ACCESO", max(11, min(24, m * 2.0)), True, color=(60, 60, 60))

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
        d.text((x + tl * 2.0, yb + hb * 0.22 + tl * 0.55), txt, font=f, fill=(40, 40, 40), anchor="lm")
        x += tl * 2.6 + d.textlength(txt, font=f) + tl * 1.2
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
        prot = max(lot.protecciones, key=lambda g: g.area)
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


# --------------------------------------------------------------------------------------------- todo junto
def generar(carpeta: str, base: str, t: Terreno, lot: Resultado, malla: Malla | None, ficha: dict | None,
            proyecto: str, partes: list | None = None, pintor: Callable | None = None) -> list[dict]:
    """Hace la planta (siempre) y la vista aérea (si hay relieve). Devuelve [{ruta, nombre, titulo, vista, ia, ...}]."""
    ficha = ficha or {}
    trabajos = [("planta", lambda: calco_planta(t, lot, malla, ficha), INSTRUCCION_PLANTA)]
    if malla is not None:
        trabajos.append(("aerea", lambda: calco_aereo(t, lot, malla), INSTRUCCION_AEREA))
    calcos = {}
    for vista, fn, _ in trabajos:
        try:
            calcos[vista] = fn()
        except Exception:  # noqa: BLE001
            log.exception("calco %s", vista)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futuros = {v: ex.submit(pintar, calcos[v], pintor, ins) for v, _, ins in trabajos if v in calcos}
        pintadas = {v: f.result() for v, f in futuros.items()}
    out = []
    for vista, (img, info) in pintadas.items():
        dib = calcos[vista]
        fn = rotular_planta if vista == "planta" else rotular_aerea
        final = fn(img, dib, lot, ficha, proyecto, partes, ia=info["ia"])
        ruta = os.path.join(carpeta, f"{base}-presentacion-{vista}.jpg")
        final.save(ruta, "JPEG", quality=90)
        titulo = ("Plano de presentación (ilustrativo)" if vista == "planta" else "Vista aérea de presentación (ilustrativa)")
        out.append({"ruta": ruta, "nombre": os.path.basename(ruta), "titulo": titulo, "vista": vista, **info})
    return out
