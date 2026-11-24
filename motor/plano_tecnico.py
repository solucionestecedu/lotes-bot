"""Plano técnico de presentación: una lámina de arquitecto en grises, dibujada solo con la geometría del motor.

Lleva marco, columna de datos (título, norte y escala gráfica, cuadro de áreas, cuadro de lotes, simbología y cajetín)
y el dibujo: rejilla de coordenadas, curvas de nivel, calle con eje, lotes con su número, área y la medida de cada
lado, casas tipo con techo a cuatro aguas sombreado, alberca, árboles en línea fina, área verde y franja de protección
de la quebrada con línea verde discontinua y achurado. No usa IA y no pone nombres de colindantes.
"""
from __future__ import annotations

import datetime
import logging
import math
import random

from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import linemerge, nearest_points, polylabel, unary_union

from . import modelo3d
from .lotificacion import LADOS, Resultado, _bordes_hacia, _poligonos
from .relieve import Malla, curvas, estadisticas, intervalo_curvas
from .terreno import Terreno

log = logging.getLogger(__name__)

W_HOJA, H_HOJA = 2048, 1448          # proporción de hoja A (√2), apaisada
S = 2                                 # se dibuja al doble y se reduce: líneas finas y suaves
MARGEN, COLUMNA = 26, 470             # px finales

NEGRO, TINTA, GRIS, GRIS_CLARO = (20, 20, 20), (45, 45, 45), (120, 120, 120), (205, 205, 205)
VERDE, VERDE_TXT, AGUA = (92, 140, 80), (60, 100, 52), (88, 128, 168)
MESES = ["ene.", "feb.", "mar.", "abr.", "may.", "jun.", "jul.", "ago.", "sep.", "oct.", "nov.", "dic."]


def _f(tam: float, negrita: bool = False):
    from .presentacion import _fuente

    return _fuente(max(8, int(round(tam * S))), negrita)


def _m2(v: float, dec: int = 2) -> str:
    return f"{v:,.{dec}f}"


def _discontinua(d: ImageDraw.ImageDraw, pts: list, color, ancho: int, lleno: float, hueco: float, punto: bool = False):
    """Línea discontinua (o de trazo y punto) sobre una polilínea en píxeles."""
    if len(pts) < 2:
        return
    ln = LineString(pts)
    L = ln.length
    s, paso = 0.0, lleno + hueco * (2 if punto else 1)
    while s < L:
        a, b = s, min(L, s + lleno)
        seg = [ln.interpolate(a), *[ln.interpolate(x) for x in (a + (b - a) * 0.5,)], ln.interpolate(b)]
        d.line([(p.x, p.y) for p in seg], fill=color, width=ancho)
        if punto and b + hueco < L:
            q = ln.interpolate(b + hueco)
            r = ancho * 0.6
            d.ellipse([q.x - r, q.y - r, q.x + r, q.y + r], fill=color)
        s += paso


def generar_tecnico(t: Terreno, lot: Resultado, malla: Malla | None, ficha: dict | None, proyecto: str,
                    partes: list | None = None, ubicacion: str = "") -> Image.Image:
    from .presentacion import _anillos, _caja_girada, _fmt, _puntos_dentro, _tramos

    ficha = ficha or {}
    W, H = W_HOJA, H_HOJA
    img = Image.new("RGB", (W * S, H * S), (255, 255, 255))
    d = ImageDraw.Draw(img, "RGBA")
    st = lot.stats

    # ------------------------------------------------------------------ área de dibujo y transformación
    dx0, dy0 = MARGEN + 18, MARGEN + 18
    dx1, dy1 = W - MARGEN - COLUMNA - 18, H - MARGEN - 18
    prop = Polygon(t.puntos).buffer(0)
    x0, y0, x1, y1 = prop.bounds
    mg = max(14.0, 0.09 * max(x1 - x0, y1 - y0))
    k = min((dx1 - dx0) / (x1 - x0 + 2 * mg), (dy1 - dy0) / (y1 - y0 + 2 * mg))  # px finales por metro
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ox, oy = (dx0 + dx1) / 2, (dy0 + dy1) / 2
    geo = box(cx - (ox - dx0) / k, cy - (dy1 - oy) / k, cx + (dx1 - ox) / k, cy + (oy - dy0) / k)

    def px(x, y):
        return ((ox + (x - cx) * k) * S, (oy - (y - cy) * k) * S)

    def a_px(x, y):  # en píxeles finales
        return (ox + (x - cx) * k, oy - (y - cy) * k)

    def ancho(px_final: float) -> int:
        return max(1, int(round(px_final * S)))

    def poli(g, fill=None, outline=None, w=1.0):
        for ext, hoyos in _anillos(g):
            pts = [px(*p) for p in ext]
            if len(pts) < 3:
                continue
            if fill is not None:
                d.polygon(pts, fill=fill)
                for hoyo in hoyos:
                    d.polygon([px(*p) for p in hoyo], fill=(255, 255, 255))
            if outline is not None:
                d.line(pts + [pts[0]], fill=outline, width=ancho(w), joint="curve")

    def linea(coords, color, w=1.0):
        if len(coords) >= 2:
            d.line([px(*p) for p in coords], fill=color, width=ancho(w), joint="curve")

    rng = random.Random(len(t.puntos) * 7919 + int(prop.area))

    def arbol(q, r_m, sombra=True):
        """Árbol en planta: copa de borde ondulado, línea fina, con su sombra."""
        qx, qy = px(*q)
        r = r_m * k * S
        fase = rng.uniform(0, 6.28)
        n = 7 + int(r_m)
        borde = [(qx + r * (1 + 0.07 * math.sin(n * a + fase)) * math.cos(a),
                  qy + r * (1 + 0.07 * math.sin(n * a + fase)) * math.sin(a)) for a in [i * 2 * math.pi / 40 for i in range(40)]]
        if sombra:
            d.polygon([(x + r * 0.35, y + r * 0.35) for x, y in borde], fill=(0, 0, 0, 22))
        d.polygon(borde, fill=(252, 253, 251), outline=(125, 138, 120))
        d.line(borde + [borde[0]], fill=(125, 138, 120), width=ancho(0.9))
        d.ellipse([qx - r * 0.08, qy - r * 0.08, qx + r * 0.08, qy + r * 0.08], fill=(125, 138, 120))

    # ------------------------------------------------------------------ rejilla y curvas
    paso_g = next(v for v in (10, 20, 25, 50, 100, 200, 500) if (geo.bounds[2] - geo.bounds[0]) / v <= 14)
    gx0, gy0, gx1, gy1 = geo.bounds
    for gx in range(int(math.ceil(gx0 / paso_g)), int(gx1 // paso_g) + 1):
        linea([(gx * paso_g, gy0), (gx * paso_g, gy1)], (236, 236, 236), 0.8)
    for gy in range(int(math.ceil(gy0 / paso_g)), int(gy1 // paso_g) + 1):
        linea([(gx0, gy * paso_g), (gx1, gy * paso_g)], (236, 236, 236), 0.8)
    if malla is not None:
        try:
            iv = intervalo_curvas(estadisticas(malla, t.puntos)["desnivel"])
            for c in curvas(malla, iv, recorte=geo.intersection(prop.buffer(max(15.0, mg), join_style=1))):
                for ln in c["lineas"]:
                    linea(ln, (150, 150, 150) if c["indice"] else (200, 200, 200), 1.1 if c["indice"] else 0.7)
        except Exception:  # noqa: BLE001
            log.exception("curvas del plano técnico")

    # ------------------------------------------------------------------ calle de acceso y quebrada (fuera del terreno)
    calles_u = unary_union(lot.calles) if lot.calles else None
    acceso = quebrada = None
    lado_acc = (ficha.get("acceso_lado") or "").lower()
    if lado_acc in LADOS:
        bordes = [LineString(b) for b in _bordes_hacia(prop, lado_acc, tolerancia=35)]
        if bordes and calles_u is not None:
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
            via = via.difference(prop.buffer(0.6)).intersection(geo)
            poli(via, fill=(238, 238, 238), outline=(140, 140, 140), w=0.9)
            acceso = max(_poligonos(via), key=lambda g: g.area, default=None)
    lado_q = next((str(q["lado"]).lower() for q in ficha.get("quebradas") or [] if str(q.get("lado", "")).lower() in LADOS),
                  None)
    if lot.protecciones and lado_q:
        bordes = _bordes_hacia(prop, lado_q, tolerancia=50)
        if bordes:
            orilla = linemerge(unary_union([LineString(b) for b in bordes]))
            cauce = []
            for g in getattr(orilla, "geoms", [orilla]):
                try:
                    o1, o2 = g.offset_curve(7.0), g.offset_curve(-7.0)
                    cauce.append(o1 if prop.distance(o1) > prop.distance(o2) else o2)
                except Exception:  # noqa: BLE001
                    continue
            for a in cauce:
                if a.length > 2:
                    n = max(8, int(a.length / 2))
                    ond = []
                    for i in range(n + 1):
                        s_ = a.length * i / n
                        p, q = a.interpolate(s_), a.interpolate(min(a.length, s_ + 0.5))
                        tx, ty = q.x - p.x, q.y - p.y
                        tl = math.hypot(tx, ty) or 1
                        w_ = 1.4 * math.sin(2 * math.pi * s_ / 26.0)
                        ond.append((p.x - ty / tl * w_, p.y + tx / tl * w_))
                    ond_g = LineString(ond).intersection(geo)
                    for g in getattr(ond_g, "geoms", [ond_g]):
                        if g.geom_type == "LineString" and g.length > 1:
                            for off in (-0.9, 0.9):  # cauce: dos líneas finas
                                try:
                                    gg = g.offset_curve(off)
                                    linea(list(gg.coords), AGUA, 1.0)
                                except Exception:  # noqa: BLE001
                                    pass
            quebrada = max(cauce, key=lambda a: a.length) if cauce else None
            monte = orilla.buffer(12).difference(prop.buffer(0.4)).intersection(geo)
            for p in _puntos_dentro(monte, int(monte.area / 110), rng, 6.0):
                if quebrada is None or Point(p).distance(quebrada) > 2.5:
                    arbol(p, rng.uniform(2.2, 3.4))

    # ------------------------------------------------------------------ terreno: protección, verde, lotes, calles
    for p in lot.protecciones:  # achurado diagonal fino + borde verde discontinuo
        poli(p, fill=(246, 250, 244))
        bx0, by0, bx1, by1 = p.bounds
        paso = 2.2
        for i in range(int((bx1 - bx0 + by1 - by0) / paso) + 2):
            c0 = bx0 - (by1 - by0) + i * paso
            seg = LineString([(c0, by0), (c0 + (by1 - by0), by1)]).intersection(p)
            for g in getattr(seg, "geoms", [seg]):
                if g.geom_type == "LineString":
                    linea(list(g.coords), (200, 222, 192), 0.7)
        for ext, _ in _anillos(p):
            _discontinua(d, [px(*q) for q in ext] + [px(*ext[0])], VERDE, ancho(1.6), 9 * S, 5 * S)
        for q in _puntos_dentro(p.buffer(-1.0), int(p.area / 90), rng, 5.5):
            arbol(q, rng.uniform(2.0, 3.0))
    for v in lot.verdes:
        poli(v["poly"], fill=(240, 247, 236), outline=VERDE, w=1.0)
    for e in lot.equipamiento:
        poli(e["poly"], fill=(236, 240, 248), outline=GRIS, w=1.0)
    for l in lot.lotes:
        poli(l["poly"], fill=(255, 255, 255))
    for c_ in lot.calles:
        poli(c_, fill=(232, 232, 232), outline=(70, 70, 70), w=1.5)

    # eje de la calle central (discontinuo) hasta el redondel
    entrada, retorno = st.get("entrada_calle"), st.get("retorno_centro")
    if st.get("modo") == "calle_central" and entrada and retorno:
        rr = (st.get("retorno_diametro_m") or 18) / 2
        ej = LineString([entrada, retorno])
        if ej.length > rr + 2:
            ej = LineString([ej.interpolate(1.5).coords[0], ej.interpolate(ej.length - rr * 0.6).coords[0]])
            _discontinua(d, [px(*q) for q in ej.coords], (160, 160, 160), ancho(1.0), 10 * S, 7 * S)

    # casas tipo (techo a cuatro aguas sombreado), alberca, entrada de autos y árboles del jardín
    casas, albercas = {}, {}
    luz = (-0.6, -0.8)  # de arriba a la izquierda, en pantalla
    for l in lot.lotes:
        esq = modelo3d.huella_casa(l, calles_u, lot.angulo)
        if esq is None:
            continue
        casa = Polygon(esq)
        casas[l["id"]] = casa
        e0, e1, e2, e3 = esq[:4]                     # frente izq., frente der., fondo der., fondo izq.
        if calles_u is not None:                     # entrada de autos
            a, b = nearest_points(LineString([e0, e1]), calles_u)
            cam = LineString([(a.x, a.y), (b.x, b.y)]).buffer(1.5, cap_style=2).intersection(l["poly"])
            poli(cam, fill=(246, 246, 246), outline=(175, 175, 175), w=0.7)
        ux_, uy_ = e1[0] - e0[0], e1[1] - e0[1]
        largo = math.hypot(ux_, uy_) or 1
        ux_, uy_ = ux_ / largo, uy_ / largo
        nx_, ny_ = e3[0] - e0[0], e3[1] - e0[1]
        prof = math.hypot(nx_, ny_) or 1
        nx_, ny_ = nx_ / prof, ny_ / prof
        if l["area"] >= 350:                         # alberca detrás de la casa, con su andador
            pl, pd = min(8.0, largo * 0.6), 3.6
            ax_ = (e2[0] + e3[0]) / 2 + nx_ * (2.8 + pd / 2)
            ay_ = (e2[1] + e3[1]) / 2 + ny_ * (2.8 + pd / 2)
            alb = Polygon([(ax_ + sx * ux_ * pl / 2 + sy * nx_ * pd / 2, ay_ + sx * uy_ * pl / 2 + sy * ny_ * pd / 2)
                           for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            if l["poly"].buffer(-1.2).contains(alb) and (calles_u is None or alb.distance(calles_u) > 3):
                albercas[l["id"]] = alb
                poli(alb.buffer(1.2, join_style=2), fill=(244, 244, 242), outline=(170, 170, 170), w=0.7)
                poli(alb, fill=(226, 238, 246), outline=(90, 110, 128), w=1.0)
                poli(alb.buffer(-0.45, join_style=2), outline=(160, 186, 206), w=0.6)
        sombra = Polygon([(x + 1.1, y - 1.1) for x, y in casa.exterior.coords])
        poli(sombra, fill=(0, 0, 0, 42))
        m0 = ((e0[0] + e3[0]) / 2, (e0[1] + e3[1]) / 2)
        m1 = ((e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2)
        lg = math.dist(m0, m1) or 1
        rec = min(prof / 2, lg / 2)
        r0 = (m0[0] + (m1[0] - m0[0]) / lg * rec, m0[1] + (m1[1] - m0[1]) / lg * rec)
        r1 = (m1[0] - (m1[0] - m0[0]) / lg * rec, m1[1] - (m1[1] - m0[1]) / lg * rec)
        for agua in ((e0, e1, r1, r0), (e1, e2, r1), (e2, e3, r0, r1), (e3, e0, r0)):
            pts = [px(*q) for q in agua]
            # hacia dónde mira el agua (del caballete al alero), en pantalla, contra la luz
            mx = sum(q[0] for q in pts[:2]) / 2 - sum(q[0] for q in pts[2:]) / len(pts[2:])
            my = sum(q[1] for q in pts[:2]) / 2 - sum(q[1] for q in pts[2:]) / len(pts[2:])
            ml = math.hypot(mx, my) or 1
            luzv = (mx * luz[0] + my * luz[1]) / ml
            tono = int(236 + 18 * luzv)
            d.polygon(pts, fill=(tono, tono, tono))
        poli(casa, outline=(40, 40, 40), w=1.3)
        linea([r0, r1], (70, 70, 70), 0.9)
        for e_, r_ in ((e0, r0), (e3, r0), (e1, r1), (e2, r1)):
            linea([e_, r_], (90, 90, 90), 0.7)
    for l in lot.lotes:
        libre = l["poly"].buffer(-2.0)
        if l["id"] in casas:
            libre = libre.difference(casas[l["id"]].buffer(3.0))
        if l["id"] in albercas:
            libre = libre.difference(albercas[l["id"]].buffer(2.5))
        if calles_u is not None:
            libre = libre.difference(calles_u.buffer(7))
        for q in _puntos_dentro(libre, max(1, min(4, int(l["area"] / 380))), rng, 7):
            arbol(q, rng.uniform(1.8, 2.8))
    for v in lot.verdes:
        for q in _puntos_dentro(v["poly"].buffer(-2.0), int(v["area"] / 110), rng, 7):
            arbol(q, rng.uniform(2.0, 3.0))

    # linderos de lotes y del terreno
    for l in lot.lotes:
        poli(l["poly"], outline=(35, 35, 35), w=1.3)
    poli(prop, outline=NEGRO, w=3.4)

    # portón y caseta
    porton = None
    if calles_u is not None:
        boca = calles_u.boundary.intersection(prop.exterior.buffer(0.4))
        tramos_b = [g for g in getattr(boca, "geoms", [boca]) if g.geom_type == "LineString" and g.length > 1]
        if tramos_b and ficha.get("cerrado"):
            bo = min(tramos_b, key=lambda g: g.distance(Point(entrada))) if entrada else max(tramos_b, key=lambda g: g.length)
            pa, pb = bo.coords[0], bo.coords[-1]
            mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
            interior = calles_u.buffer(-2.5)
            hacia = nearest_points(Point(mid), interior)[1] if not interior.is_empty else calles_u.centroid
            nx_, ny_ = hacia.x - mid[0], hacia.y - mid[1]
            nl = math.hypot(nx_, ny_) or 1
            nx_, ny_ = nx_ / nl, ny_ / nl
            reja = LineString([(pa[0] + nx_ * 4, pa[1] + ny_ * 4), (pb[0] + nx_ * 4, pb[1] + ny_ * 4)]).intersection(calles_u)
            for g in getattr(reja, "geoms", [reja]):
                if g.geom_type == "LineString":
                    linea(list(g.coords), NEGRO, 2.2)
            ex, ey = pa[0] - pb[0], pa[1] - pb[1]
            el = math.hypot(ex, ey) or 1
            ex, ey = ex / el, ey / el
            bx, by = pa[0] + nx_ * 6.5 - ex * 1.6, pa[1] + ny_ * 6.5 - ey * 1.6
            caseta = Polygon([(bx + sx * 1.3 * ex + sy * 1.6 * nx_, by + sx * 1.3 * ey + sy * 1.6 * ny_)
                              for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            poli(caseta, fill=(215, 215, 215), outline=NEGRO, w=1.1)
            porton = (bx, by)

    # ------------------------------------------------------------------ letreros del dibujo (a tamaño final)
    img = img.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(img, "RGBA")
    ocupado: list = []

    def fuente(tam, negrita=False):
        from .presentacion import _fuente

        return _fuente(max(8, int(round(tam))), negrita)

    def texto(xy, txt, tam, negrita=False, color=TINTA, anchor="mm", halo=3, registrar=True):
        f = fuente(tam, negrita)
        d.multiline_text(xy, txt, font=f, fill=color, anchor=anchor, align="center", stroke_width=halo,
                         stroke_fill=(255, 255, 255), spacing=2)
        if registrar:
            ocupado.append(box(*d.multiline_textbbox(xy, txt, font=f, anchor=anchor, align="center", stroke_width=halo,
                                                     spacing=2)))

    def girado(xy, txt, tam, ang, negrita=False, color=TINTA, halo=3):
        f = fuente(tam, negrita)
        c_ = f.getbbox(txt, stroke_width=halo)
        w_, h_ = c_[2] - c_[0] + 6, c_[3] - c_[1] + 6
        a = ((ang + 90) % 180) - 90
        huella = _caja_girada(xy, w_, h_, a)
        if any(huella.intersects(o) for o in ocupado):
            return False
        capa = Image.new("RGBA", (int(w_), int(h_)), (0, 0, 0, 0))
        ImageDraw.Draw(capa).text((w_ / 2, h_ / 2), txt, font=f, fill=color + (255,), anchor="mm", stroke_width=halo,
                                  stroke_fill=(255, 255, 255, 255))
        capa = capa.rotate(a, expand=True, resample=Image.BICUBIC)
        img.paste(capa, (int(xy[0] - capa.width / 2), int(xy[1] - capa.height / 2)), capa)
        ocupado.append(huella)
        return True

    # calle, redondel, portón
    ancho_c = lot.parametros.get("calle", 8)
    if lot.calles:
        nombre_c = "CALLE CENTRAL" if st.get("modo") == "calle_central" else "CALLE"
        if st.get("modo") == "calle_central" and entrada and retorno:
            ej = LineString([entrada, retorno])
            ang = math.degrees(math.atan2(retorno[1] - entrada[1], retorno[0] - entrada[0]))
            for fr in (0.45, 0.35, 0.55, 0.25):
                q = ej.interpolate(fr, normalized=True)
                if girado(a_px(q.x, q.y), f"{nombre_c} · {ancho_c:.2f} m", 14, ang, True):
                    break
            rr = (st.get("retorno_diametro_m") or 18) / 2
            txt_r = f"REDONDEL\nØ {2 * rr:.2f} m"
            if 2 * rr * k >= 70:
                texto(a_px(*retorno), txt_r, 10.5, True, (60, 60, 60), halo=2)
        else:
            try:
                p = polylabel(max(_poligonos(calles_u), key=lambda g: g.area), tolerance=0.5)
            except Exception:  # noqa: BLE001
                p = calles_u.representative_point()
            girado(a_px(p.x, p.y), f"{nombre_c} · {ancho_c:.2f} m", 14, lot.angulo, True)
    # etiquetas de lote: número y área en un recuadro fino
    for l in lot.lotes:
        libre = l["poly"].buffer(-1.0)
        for g in (casas.get(l["id"]), albercas.get(l["id"])):
            if g is not None:
                libre = libre.difference(g.buffer(1.6))
        libre = max(_poligonos(libre), key=lambda g: g.area, default=l["poly"])
        try:
            p = polylabel(libre, tolerance=0.5)
        except Exception:  # noqa: BLE001
            p = libre.representative_point()
        x, y = a_px(p.x, p.y)
        lote_px = Polygon([a_px(*q) for q in l["poly"].exterior.coords]).buffer(4)
        t1, t2 = f"LOTE {l['id']}", f"{_m2(l['area'])} m²"
        f1, f2 = fuente(15, True), fuente(12.5)
        w_ = max(d.textlength(t1, font=f1), d.textlength(t2, font=f2)) + 16
        caja = box(x - w_ / 2, y - 19, x + w_ / 2, y + 19)
        if lote_px.contains(caja) and not any(caja.intersects(o) for o in ocupado):
            d.rectangle(caja.bounds, fill=(255, 255, 255, 240), outline=(40, 40, 40), width=1)
            d.text((x, y - 7), t1, font=f1, fill=NEGRO, anchor="mm")
            d.text((x, y + 9), t2, font=f2, fill=(70, 70, 70), anchor="mm")
            ocupado.append(caja.buffer(2))
        else:  # lote chico: solo el número en un círculo (el área está en el cuadro de lotes)
            num = str(l["id"])
            fn = fuente(12, True)
            r = max(10, d.textlength(num, font=fn) / 2 + 5)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 245), outline=(40, 40, 40), width=1)
            d.text((x, y), num, font=fn, fill=NEGRO, anchor="mm")
            ocupado.append(Point(x, y).buffer(r + 2))

    if porton is not None:
        bx, by = a_px(*porton)
        texto((bx + 14, by), "PORTÓN Y\nCASETA", 10.5, True, (60, 60, 60), anchor="lm", halo=2)
    if acceso is not None:
        p = acceso.representative_point()
        texto(a_px(p.x, p.y), "CALLE DE ACCESO", 11.5, True, (90, 90, 90), halo=2)
    for v in sorted(lot.verdes, key=lambda v: -v["area"])[:1]:
        if v["area"] > 60:
            try:
                p = polylabel(v["poly"], tolerance=0.5)
            except Exception:  # noqa: BLE001
                p = v["poly"].representative_point()
            texto(a_px(p.x, p.y), f"ÁREA VERDE\nRECREATIVA\n{_m2(st['area_verde'])} m²", 11.5, True, VERDE_TXT)
    if lot.protecciones:
        prot = max(lot.protecciones, key=lambda g: g.area)
        try:
            p = polylabel(prot, tolerance=0.5)
        except Exception:  # noqa: BLE001
            p = prot.representative_point()
        franja = next((f" · FRANJA {1.5 * float(q['profundidad_m']):.2f} m" for q in ficha.get("quebradas") or []
                       if q.get("profundidad_m")), "")
        texto(a_px(p.x, p.y), f"ZONA DE PROTECCIÓN DE QUEBRADA{franja}", 11.5, True, VERDE_TXT)
    if quebrada is not None:
        q = quebrada.interpolate(0.5, normalized=True)
        if geo.contains(q):
            texto(a_px(q.x, q.y), "quebrada", 11.5, False, AGUA)

    # medidas de cada lado de cada lote (las compartidas una vez), sobre la línea
    vistos: list = []
    for l in lot.lotes:
        for tr in _tramos(l["poly"]):
            if tr.length < 3 or any(tr.hausdorff_distance(v) < 0.6 for v in vistos):
                continue
            vistos.append(tr)
            txt = f"{tr.length:.2f}"
            if tr.length * k < len(txt) * 7 + 14:
                continue
            for fr in (0.5, 0.36, 0.64, 0.25, 0.75):
                a, b = tr.interpolate(max(0.0, fr * tr.length - 1.0)), tr.interpolate(min(tr.length, fr * tr.length + 1.0))
                ang = math.degrees(math.atan2(b.y - a.y, b.x - a.x))
                q = tr.interpolate(fr, normalized=True)
                if girado(a_px(q.x, q.y), txt, 11, ang, False, (40, 40, 40), halo=3):
                    break

    # coordenadas de la rejilla en el borde del dibujo
    if getattr(t, "ubicado", False):
        fg = fuente(9.5)
        for gx in range(int(math.ceil(gx0 / paso_g)), int(gx1 // paso_g) + 1):
            x, _ = a_px(gx * paso_g, 0)
            if dx0 + 40 < x < dx1 - 40:
                d.text((x, dy0 + 2), f"E {gx * paso_g:,.0f}", font=fg, fill=GRIS, anchor="ma")
        for gy in range(int(math.ceil(gy0 / paso_g)), int(gy1 // paso_g) + 1):
            _, y = a_px(0, gy * paso_g)
            if dy0 + 30 < y < dy1 - 30:
                d.text((dx0 + 3, y - 2), f"N {gy * paso_g:,.0f}", font=fg, fill=GRIS, anchor="ld")

    # ------------------------------------------------------------------ marco y columna de datos
    d.rectangle([MARGEN, MARGEN, W - MARGEN, H - MARGEN], outline=NEGRO, width=3)
    d.rectangle([MARGEN + 7, MARGEN + 7, W - MARGEN - 7, H - MARGEN - 7], outline=NEGRO, width=1)
    cx0 = W - MARGEN - COLUMNA
    d.line([(cx0, MARGEN + 7), (cx0, H - MARGEN - 7)], fill=NEGRO, width=2)
    X0, X1 = cx0 + 22, W - MARGEN - 22
    y = MARGEN + 30

    def regla(yy, grueso=1):
        d.line([(cx0, yy), (W - MARGEN - 7, yy)], fill=NEGRO, width=grueso)

    def envolver(txt, f, ancho_max):
        palabras, lineas, act = txt.split(), [], ""
        for p_ in palabras:
            prueba = (act + " " + p_).strip()
            if d.textlength(prueba, font=f) <= ancho_max:
                act = prueba
            else:
                lineas.append(act)
                act = p_
        return lineas + ([act] if act else [])

    d.text((X0, y), "ANTEPROYECTO DE LOTIFICACIÓN", font=fuente(13, True), fill=GRIS, anchor="la")
    y += 26
    for ln in envolver((proyecto or "Anteproyecto").upper(), fuente(25, True), X1 - X0)[:2]:
        d.text((X0, y), ln, font=fuente(25, True), fill=NEGRO, anchor="la")
        y += 32
    n = st["lotes"]
    nombre = "QUINTAS" if (ficha.get("tipo") == "campestre" or st.get("modo") == "calle_central") else "LOTES"
    areas = [l["area"] for l in lot.lotes]
    rango = f"de {_fmt(min(areas))} a {_fmt(max(areas))} m²" if min(areas) < 0.97 * max(areas) else f"de ~{_fmt(areas[0])} m²"
    d.text((X0, y + 2), f"{n} {nombre} {rango}", font=fuente(16), fill=TINTA, anchor="la")
    y += 26
    if partes and len(partes) > 1:
        txt_p = " · ".join(f"{p_['origen']}: {_m2(Polygon(p_['anillo']).area, 0)} m²" for p_ in partes)
        for ln in envolver(txt_p, fuente(13), X1 - X0)[:2]:
            d.text((X0, y), ln, font=fuente(13), fill=TINTA, anchor="la")
            y += 18
    if ubicacion:
        for ln in envolver(ubicacion, fuente(13), X1 - X0)[:2]:
            d.text((X0, y), ln, font=fuente(13), fill=GRIS, anchor="la")
            y += 18
    y += 14
    regla(y, 2)

    # norte y escala gráfica
    y += 20
    R = 34
    kx, ky = X0 + R + 4, y + R + 14
    d.ellipse([kx - R, ky - R, kx + R, ky + R], outline=NEGRO, width=2)
    d.polygon([(kx, ky - R + 4), (kx - 11, ky + 14), (kx, ky + 6)], fill=NEGRO)
    d.polygon([(kx, ky - R + 4), (kx + 11, ky + 14), (kx, ky + 6)], outline=NEGRO, fill=(255, 255, 255))
    d.text((kx, ky - R - 4), "N", font=fuente(16, True), fill=NEGRO, anchor="md")
    paso = next(v for v in (5, 10, 20, 25, 50, 100, 200) if v * k * 4 >= (X1 - X0) * 0.45)
    while paso * k * 4 > X1 - X0 - 2 * R - 40 and paso > 5:
        paso = {200: 100, 100: 50, 50: 25, 25: 20, 20: 10, 10: 5}.get(paso, 5)
    ex0, ey0 = kx + R + 36, ky - 2
    for i in range(4):
        d.rectangle([ex0 + i * paso * k, ey0, ex0 + (i + 1) * paso * k, ey0 + 8],
                    fill=NEGRO if i % 2 == 0 else (255, 255, 255), outline=NEGRO)
    for i in range(5):
        d.text((ex0 + i * paso * k, ey0 - 5), f"{i * paso}", font=fuente(11), fill=TINTA, anchor="md")
    d.text((ex0, ey0 + 16), "ESCALA GRÁFICA (metros)", font=fuente(11), fill=GRIS, anchor="la")
    y = ky + R + 22
    regla(y)

    # cuadro de áreas
    y += 14
    d.text((X0, y), "CUADRO DE ÁREAS", font=fuente(13, True), fill=NEGRO, anchor="la")
    y += 24
    total = st["area_total"]
    filas = [(f"Lotes ({n})", st["area_util"]), ("Calle y redondel" if st.get("modo") == "calle_central" else "Calles",
                                                  st["area_circulacion"]),
             ("Área verde recreativa", st["area_verde"]), ("Zona de protección de quebrada", st.get("area_proteccion", 0)),
             ("Equipamiento social", st.get("area_equipamiento", 0))]
    filas = [(a, b) for a, b in filas if b and b > 0.5]
    resto = total - sum(b for _, b in filas)
    if resto > 1:
        filas.append(("Otros / remanente", resto))
    cols = (X0, X1 - 150, X1)
    d.text((cols[0], y), "CONCEPTO", font=fuente(10.5, True), fill=GRIS, anchor="la")
    d.text((cols[1], y), "ÁREA (m²)", font=fuente(10.5, True), fill=GRIS, anchor="ra")
    d.text((cols[2], y), "%", font=fuente(10.5, True), fill=GRIS, anchor="ra")
    y += 17
    d.line([(X0, y), (X1, y)], fill=GRIS, width=1)
    for a, b in filas:
        y += 5
        d.text((cols[0], y), a, font=fuente(12.5), fill=TINTA, anchor="la")
        d.text((cols[1], y), _m2(b), font=fuente(12.5), fill=TINTA, anchor="ra")
        d.text((cols[2], y), f"{b / total * 100:.1f}", font=fuente(12.5), fill=TINTA, anchor="ra")
        y += 18
    d.line([(X0, y + 2), (X1, y + 2)], fill=NEGRO, width=1)
    y += 7
    d.text((cols[0], y), "TOTAL DEL TERRENO", font=fuente(12.5, True), fill=NEGRO, anchor="la")
    d.text((cols[1], y), _m2(total), font=fuente(12.5, True), fill=NEGRO, anchor="ra")
    d.text((cols[2], y), "100.0", font=fuente(12.5, True), fill=NEGRO, anchor="ra")
    y += 30
    regla(y)

    # simbología y cajetín se acomodan abajo; el cuadro de lotes usa lo que queda en medio
    alto_simb, alto_caj = 8 * 22 + 40, 150
    y_simb = H - MARGEN - 7 - alto_caj - alto_simb
    y += 14
    d.text((X0, y), "CUADRO DE LOTES", font=fuente(13, True), fill=NEGRO, anchor="la")
    y += 24
    alto_fila = 17
    caben = max(0, int((y_simb - 20 - y - 18) / alto_fila))
    lotes = sorted(lot.lotes, key=lambda l: l["id"] if isinstance(l["id"], int) else str(l["id"]))
    ncol = 1 if len(lotes) <= caben else (2 if len(lotes) <= 2 * caben else 0)
    if ncol == 0:
        d.text((X0, y), f"{n} lotes de {_m2(min(areas))} a {_m2(max(areas))} m²", font=fuente(12.5), fill=TINTA, anchor="la")
        d.text((X0, y + 18), "(el detalle de cada lote está en la lámina PDF)", font=fuente(11), fill=GRIS, anchor="la")
    else:
        anchoc = (X1 - X0 - (16 if ncol == 2 else 0)) / ncol
        for c in range(ncol):
            bx0 = X0 + c * (anchoc + 16)
            sub = lotes[c * caben:(c + 1) * caben] if ncol == 2 else lotes
            yy = y
            d.text((bx0, yy), "LOTE", font=fuente(10.5, True), fill=GRIS, anchor="la")
            d.text((bx0 + anchoc * 0.62, yy), "ÁREA (m²)", font=fuente(10.5, True), fill=GRIS, anchor="ra")
            d.text((bx0 + anchoc, yy), "FRENTE (m)", font=fuente(10.5, True), fill=GRIS, anchor="ra")
            yy += 16
            d.line([(bx0, yy), (bx0 + anchoc, yy)], fill=GRIS, width=1)
            yy += 3
            for l in sub:
                d.text((bx0, yy), f"{l['id']}", font=fuente(12), fill=TINTA, anchor="la")
                d.text((bx0 + anchoc * 0.62, yy), _m2(l["area"]), font=fuente(12), fill=TINTA, anchor="ra")
                fr_ = l.get("frente")
                d.text((bx0 + anchoc, yy), _m2(fr_) if fr_ else "—", font=fuente(12), fill=TINTA, anchor="ra")
                yy += alto_fila

    # simbología
    y = y_simb
    regla(y)
    y += 14
    d.text((X0, y), "SIMBOLOGÍA", font=fuente(13, True), fill=NEGRO, anchor="la")
    y += 26
    mx0, mx1, tx = X0, X0 + 46, X0 + 60

    def muestra(tipo):
        yc = y + 7
        if tipo == "terreno":
            d.line([(mx0, yc), (mx1, yc)], fill=NEGRO, width=4)
        elif tipo == "lote":
            d.line([(mx0, yc), (mx1, yc)], fill=(35, 35, 35), width=1)
        elif tipo == "calle":
            d.rectangle([mx0, yc - 6, mx1, yc + 6], fill=(232, 232, 232), outline=(70, 70, 70))
        elif tipo == "verde":
            d.rectangle([mx0, yc - 6, mx1, yc + 6], fill=(240, 247, 236), outline=VERDE)
        elif tipo == "proteccion":
            d.rectangle([mx0, yc - 6, mx1, yc + 6], fill=(246, 250, 244))
            for i in range(0, 60, 6):
                d.line([(mx0 + i - 12, yc + 6), (mx0 + i, yc - 6)], fill=(200, 222, 192), width=1)
            d.rectangle([mx0 - 14, yc - 8, mx0 - 1, yc + 8], fill=(255, 255, 255))
            d.rectangle([mx1 + 1, yc - 8, mx1 + 14, yc + 8], fill=(255, 255, 255))
            _discontinua(d, [(mx0, yc - 6), (mx1, yc - 6), (mx1, yc + 6), (mx0, yc + 6), (mx0, yc - 6)], VERDE, 2, 7, 4)
        elif tipo == "curva":
            d.line([(mx0, yc + 3), (mx0 + 15, yc - 2), (mx0 + 30, yc + 1), (mx1, yc - 3)], fill=(170, 170, 170), width=1)
        elif tipo == "casa":
            d.polygon([(mx0 + 8, yc - 7), (mx1 - 8, yc - 7), (mx1 - 8, yc + 7), (mx0 + 8, yc + 7)], fill=(236, 236, 236),
                      outline=(40, 40, 40))
            d.line([(mx0 + 15, yc), (mx1 - 15, yc)], fill=(70, 70, 70), width=1)
            for xa, xb in ((mx0 + 8, mx0 + 15), (mx1 - 8, mx1 - 15)):
                d.line([(xa, yc - 7), (xb, yc)], fill=(90, 90, 90), width=1)
                d.line([(xa, yc + 7), (xb, yc)], fill=(90, 90, 90), width=1)
        elif tipo == "arbol":
            d.ellipse([mx0 + 15, yc - 8, mx0 + 31, yc + 8], outline=(125, 138, 120), width=1)
            d.ellipse([mx0 + 22, yc - 1, mx0 + 24, yc + 1], fill=(125, 138, 120))

    for tipo, txt in (("terreno", "Lindero del terreno"), ("lote", "Lindero de lote (medidas en m)"), ("calle", "Calle"),
                      ("verde", "Área verde recreativa"), ("proteccion", "Zona de protección de quebrada"),
                      ("curva", "Curvas de nivel (relieve satelital aproximado)"), ("casa", "Casa tipo y alberca (ilustrativas)"),
                      ("arbol", "Árbol (ilustrativo)")):
        muestra(tipo)
        d.text((tx, y + 7), txt, font=fuente(12), fill=TINTA, anchor="lm")
        y += 22

    # cajetín
    y = H - MARGEN - 7 - alto_caj
    regla(y, 2)
    hoy = datetime.date.today()
    filas_c = [("PROYECTO", proyecto or "Anteproyecto"), ("CONTENIDO", "Plano de lotificación — presentación"),
               ("FECHA", f"{MESES[hoy.month - 1]} {hoy.year}"), ("ESCALA", "Gráfica")]
    yy = y + 10
    for et, val in filas_c:
        d.text((X0, yy), et, font=fuente(10.5, True), fill=GRIS, anchor="la")
        val = envolver(val, fuente(12.5), X1 - X0 - 110)[0]
        d.text((X0 + 110, yy - 1), val, font=fuente(12.5), fill=NEGRO, anchor="la")
        yy += 21
    nota = ("Anteproyecto referencial con las medidas del prediseño. Relieve satelital aproximado. "
            "No válido para trámite: requiere levantamiento topográfico y firma de profesional autorizado.")
    for ln in envolver(nota, fuente(10.5), X1 - X0)[:3]:
        d.text((X0, yy + 2), ln, font=fuente(10.5), fill=GRIS, anchor="la")
        yy += 14
    return img.convert("RGB")
