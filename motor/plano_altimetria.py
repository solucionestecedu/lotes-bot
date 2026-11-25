"""Plano altimétrico: la lámina de relieve de la carpeta técnica, dibujada solo por el motor (no usa IA).

Dibujo: pendientes por color (suave, moderada, fuerte, barranco), curvas de nivel con su cota, flechas de hacia dónde
escurre el agua, cotas en los vértices del terreno, lotes con su número, calle con estaciones cada 20 m y las caídas
fuertes que se dejaron fuera de los lotes. Columna: relieve del terreno, cuadro de pendientes, perfil longitudinal de
la calle, relieve de cada lote, simbología y cajetín. Mismo formato de hoja que el plano técnico.
"""
from __future__ import annotations

import datetime
import logging
import math

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from . import altimetria
from .lotificacion import Resultado
from .plano_tecnico import (COLUMNA, GRIS, H_HOJA, MARGEN, MESES, NEGRO, S, TINTA, VERDE, W_HOJA, _discontinua, _m2,
                            terrenos_unidos)
from .relieve import Malla, curvas, intervalo_curvas
from .terreno import Terreno

log = logging.getLogger(__name__)

CURVA, CURVA_INDICE, CURVA_TXT = (196, 172, 146), (128, 96, 66), (110, 80, 52)
AGUA_F = (52, 96, 150)
ROJO = (178, 52, 40)


def generar_altimetria(t: Terreno, lot: Resultado, malla: Malla, ficha: dict | None, proyecto: str,
                       partes: list | None = None, ubicacion: str = "") -> Image.Image:
    from .presentacion import _anillos, _caja_girada, _fuente

    ficha = ficha or {}
    W, H = W_HOJA, H_HOJA
    img = Image.new("RGB", (W * S, H * S), (255, 255, 255))
    d = ImageDraw.Draw(img, "RGBA")
    st = lot.stats
    prop = Polygon(t.puntos).buffer(0)
    rel = altimetria.resumen(malla, t.puntos)
    barr = altimetria.barrancos(malla, t.puntos) if ficha.get("respetar_barrancos", True) else []

    # ------------------------------------------------------------------ área de dibujo y transformación
    dx0, dy0 = MARGEN + 18, MARGEN + 18
    dx1, dy1 = W - MARGEN - COLUMNA - 18, H - MARGEN - 18
    x0, y0, x1, y1 = prop.bounds
    mg = max(14.0, 0.09 * max(x1 - x0, y1 - y0))
    k = min((dx1 - dx0) / (x1 - x0 + 2 * mg), (dy1 - dy0) / (y1 - y0 + 2 * mg))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ox, oy = (dx0 + dx1) / 2, (dy0 + dy1) / 2
    geo = box(cx - (ox - dx0) / k, cy - (dy1 - oy) / k, cx + (dx1 - ox) / k, cy + (oy - dy0) / k)

    def px(x, y):
        return ((ox + (x - cx) * k) * S, (oy - (y - cy) * k) * S)

    def a_px(x, y):
        return (ox + (x - cx) * k, oy - (y - cy) * k)

    def ancho(v: float) -> int:
        return max(1, int(round(v * S)))

    def poli(g, fill=None, outline=None, w=1.0):
        for ext, hoyos in _anillos(g):
            pts = [px(*p) for p in ext]
            if len(pts) < 3:
                continue
            if fill is not None:
                d.polygon(pts, fill=fill)
            if outline is not None:
                d.line(pts + [pts[0]], fill=outline, width=ancho(w), joint="curve")

    def linea(coords, color, w=1.0):
        if len(coords) >= 2:
            d.line([px(*p) for p in coords], fill=color, width=ancho(w), joint="curve")

    # ------------------------------------------------------------------ pendientes por color (dentro del terreno)
    for a, b, _, col in altimetria.CLASES:
        try:
            for z in altimetria.zonas(malla, prop, a, b, area_min=4.0):
                poli(z, fill=col + (255,))
        except Exception:  # noqa: BLE001
            log.exception("clase de pendiente %s", a)
    # calle y redondel en gris claro encima; protección con achurado fino
    for p in lot.protecciones:
        bx0, by0, bx1, by1 = p.bounds
        for i in range(int((bx1 - bx0 + by1 - by0) / 3.0) + 2):
            c0 = bx0 - (by1 - by0) + i * 3.0
            seg = LineString([(c0, by0), (c0 + (by1 - by0), by1)]).intersection(p)
            for g in getattr(seg, "geoms", [seg]):
                if g.geom_type == "LineString":
                    linea(list(g.coords), (120, 150, 110, 120), 0.6)
    for c_ in lot.calles:
        poli(c_, fill=(238, 238, 238, 235), outline=(90, 90, 90), w=1.3)

    # ------------------------------------------------------------------ curvas de nivel (todo el cuadro)
    iv = intervalo_curvas(rel["desnivel"], max_lineas=40)
    if iv < 1 and rel["desnivel"] > 6:
        iv = 1.0
    cs = []
    try:
        cs = curvas(malla, iv, recorte=geo.intersection(prop.buffer(max(15.0, mg), join_style=1)))
    except Exception:  # noqa: BLE001
        log.exception("curvas")
    for c in cs:
        for ln in c["lineas"]:
            linea(ln, CURVA_INDICE if c["indice"] else CURVA, 1.5 if c["indice"] else 0.8)

    # ------------------------------------------------------------------ lotes, terreno, caídas fuertes
    for l in lot.lotes:
        poli(l["poly"], outline=(40, 40, 40), w=1.1)
    for u in terrenos_unidos(prop, partes):
        for ln in u["linderos"]:
            _discontinua(d, [px(*q) for q in ln.coords], (60, 60, 60), ancho(1.8), 16 * S, 5 * S, punto=True)
    for p in lot.protecciones:
        for ext, _ in _anillos(p):
            _discontinua(d, [px(*q) for q in ext] + [px(*ext[0])], VERDE, ancho(1.5), 9 * S, 5 * S)
    for b_ in barr:
        for ext, _ in _anillos(b_):
            _discontinua(d, [px(*q) for q in ext] + [px(*ext[0])], ROJO, ancho(1.8), 8 * S, 4 * S)
    poli(prop, outline=NEGRO, w=3.2)

    # eje de la calle con estaciones cada 20 m
    entrada, retorno = st.get("entrada_calle"), st.get("retorno_centro")
    eje = None
    if st.get("modo") == "calle_central" and entrada and retorno:
        eje = LineString([entrada, retorno])
        _discontinua(d, [px(*q) for q in eje.coords], (110, 110, 110), ancho(1.0), 10 * S, 6 * S)

    # flechas de escurrimiento (hacia donde baja el agua)
    gx, gy = malla.gradiente()
    paso_f = max(8.0, round(max(x1 - x0, y1 - y0) / 11 / 2) * 2)
    adentro = prop.buffer(-3)
    ocupado_px = [c_.buffer(1.5) for c_ in lot.calles]
    for fx in np.arange(x0 + paso_f / 2, x1, paso_f):
        for fy in np.arange(y0 + paso_f / 2, y1, paso_f):
            q = Point(fx, fy)
            if not adentro.contains(q) or any(o.contains(q) for o in ocupado_px):
                continue
            i, j = int(round((fx - malla.x0) / malla.dx)), int(round((fy - malla.y0) / malla.dx))
            if not (0 <= i < malla.nx and 0 <= j < malla.ny):
                continue
            ax, ay = -gx[j, i], -gy[j, i]
            n_ = math.hypot(ax, ay)
            if not math.isfinite(n_) or n_ < 0.02:
                continue
            L = min(paso_f * 0.55, paso_f * (0.25 + n_ * 0.9))
            ux, uy = ax / n_, ay / n_
            p0 = (fx - ux * L / 2, fy - uy * L / 2)
            p1 = (fx + ux * L / 2, fy + uy * L / 2)
            linea([p0, p1], AGUA_F + (210,), 1.3)
            h = min(2.2, L * 0.35)
            for s_ in (1, -1):
                linea([p1, (p1[0] - ux * h - s_ * uy * h * 0.55, p1[1] - uy * h + s_ * ux * h * 0.55)], AGUA_F + (210,), 1.3)

    # cotas en los vértices del terreno (cruz fina)
    vert = []
    for q in list(prop.exterior.coords)[:-1]:
        if vert and min(math.hypot(q[0] - v[0], q[1] - v[1]) for v in vert) < 9:
            continue
        vert.append(q)
    for q in vert:
        qx, qy = px(*q)
        r_ = 5 * S
        d.line([(qx - r_, qy), (qx + r_, qy)], fill=NEGRO, width=ancho(1.0))
        d.line([(qx, qy - r_), (qx, qy + r_)], fill=NEGRO, width=ancho(1.0))

    # ------------------------------------------------------------------ letreros (a tamaño final)
    img = img.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(img, "RGBA")
    ocupado: list = []

    def fuente(tam, negrita=False):
        return _fuente(max(8, int(round(tam))), negrita)

    def texto(xy, txt, tam, negrita=False, color=TINTA, anchor="mm", halo=3, forzar=False):
        f = fuente(tam, negrita)
        caja = box(*d.multiline_textbbox(xy, txt, font=f, anchor=anchor, align="center", stroke_width=halo, spacing=2))
        if not forzar and any(caja.intersects(o) for o in ocupado):
            return False
        d.multiline_text(xy, txt, font=f, fill=color, anchor=anchor, align="center", stroke_width=halo,
                         stroke_fill=(255, 255, 255), spacing=2)
        ocupado.append(caja)
        return True

    def girado(xy, txt, tam, ang, color=TINTA, halo=3, negrita=False):
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

    # números de lote (círculo) primero: son lo que más se busca
    for l in lot.lotes:
        c = l["poly"].representative_point() if not l["poly"].centroid.within(l["poly"]) else l["poly"].centroid
        x_, y_ = a_px(c.x, c.y)
        r_ = 15
        d.ellipse([x_ - r_, y_ - r_, x_ + r_, y_ + r_], fill=(255, 255, 255), outline=NEGRO, width=2)
        d.text((x_, y_), str(l["id"]), font=fuente(15, True), fill=NEGRO, anchor="mm")
        ocupado.append(box(x_ - r_ - 2, y_ - r_ - 2, x_ + r_ + 2, y_ + r_ + 2))
        rl = l.get("relieve")
        if rl and rl.get("pend_media") is not None:
            texto((x_, y_ + r_ + 12), f"{rl['pend_media']:.0f} %", 12, color=TINTA)

    # estaciones de la calle
    if eje is not None and eje.length > 25:
        for s_ in range(0, int(eje.length) + 1, 20):
            q = eje.interpolate(s_)
            x_, y_ = a_px(q.x, q.y)
            d.ellipse([x_ - 3, y_ - 3, x_ + 3, y_ + 3], fill=NEGRO)
            zq = malla.altura(q.x, q.y)
            texto((x_ + 8, y_), f"0+{s_:03d}" + (f"\n{zq:.1f}" if math.isfinite(zq) else ""), 10.5, anchor="lm",
                  color=(60, 60, 60))

    # cotas de los vértices
    for q in vert:
        zq = malla.altura(*q)
        if math.isfinite(zq):
            x_, y_ = a_px(*q)
            dxv, dyv = x_ - a_px(*prop.centroid.coords[0])[0], y_ - a_px(*prop.centroid.coords[0])[1]
            nv = math.hypot(dxv, dyv) or 1
            texto((x_ + dxv / nv * 26, y_ + dyv / nv * 16), f"{zq:.1f}", 11.5, negrita=True, color=NEGRO)

    # caídas fuertes
    for b_ in barr:
        c = b_.representative_point()
        texto(a_px(c.x, c.y), "CAÍDA FUERTE\nno se lotifica", 11, negrita=True, color=ROJO)

    # cotas de las curvas índice (giradas sobre la curva), y de algunas intermedias si hay espacio
    for c in sorted(cs, key=lambda c: not c["indice"]):
        for ln in sorted(c["lineas"], key=lambda ln: -LineString(ln).length)[:2]:
            L_ = LineString(ln)
            if L_.length * k < 90:
                continue
            for frac in ((0.5,) if not c["indice"] else (0.3, 0.7)):
                q0, q1 = L_.interpolate(frac - 0.02, normalized=True), L_.interpolate(frac + 0.02, normalized=True)
                (ax_, ay_), (bx_, by_) = a_px(q0.x, q0.y), a_px(q1.x, q1.y)
                ang = math.degrees(math.atan2(-(by_ - ay_), bx_ - ax_))
                q = L_.interpolate(frac, normalized=True)
                girado(a_px(q.x, q.y), f"{c['cota']:g}", 11 if c["indice"] else 9.5, ang,
                       color=CURVA_TXT if c["indice"] else (150, 125, 100), negrita=c["indice"])

    # ------------------------------------------------------------------ marco y columna
    d.rectangle([MARGEN, MARGEN, W - MARGEN, H - MARGEN], outline=NEGRO, width=3)
    d.rectangle([MARGEN + 7, MARGEN + 7, W - MARGEN - 7, H - MARGEN - 7], outline=NEGRO, width=1)
    cx0 = W - MARGEN - COLUMNA
    d.line([(cx0, MARGEN + 7), (cx0, H - MARGEN - 7)], fill=NEGRO, width=2)
    X0, X1 = cx0 + 22, W - MARGEN - 22
    y = MARGEN + 30

    def regla(yy, grueso=1):
        d.line([(cx0, yy), (W - MARGEN - 7, yy)], fill=NEGRO, width=grueso)

    def titulo_seccion(txt):
        nonlocal y
        d.text((X0, y), txt, font=fuente(13, True), fill=NEGRO, anchor="la")
        y += 24

    d.text((X0, y), "ANTEPROYECTO DE LOTIFICACIÓN", font=fuente(13, True), fill=GRIS, anchor="la")
    y += 24
    d.text((X0, y), (proyecto or "Anteproyecto").upper()[:34], font=fuente(25, True), fill=NEGRO, anchor="la")
    y += 34
    d.text((X0, y), "PLANO ALTIMÉTRICO · PENDIENTES Y PERFIL", font=fuente(14, True), fill=TINTA, anchor="la")
    y += 22
    if ubicacion:
        d.text((X0, y), ubicacion[:60], font=fuente(12.5), fill=GRIS, anchor="la")
        y += 18
    y += 10
    regla(y, 2)

    # norte y escala
    y += 16
    R = 28
    kx, ky = X0 + R + 4, y + R + 12
    d.ellipse([kx - R, ky - R, kx + R, ky + R], outline=NEGRO, width=2)
    d.polygon([(kx, ky - R + 4), (kx - 9, ky + 11), (kx, ky + 5)], fill=NEGRO)
    d.polygon([(kx, ky - R + 4), (kx + 9, ky + 11), (kx, ky + 5)], outline=NEGRO, fill=(255, 255, 255))
    d.text((kx, ky - R - 3), "N", font=fuente(14, True), fill=NEGRO, anchor="md")
    paso = next(v for v in (5, 10, 20, 25, 50, 100, 200) if v * k * 4 >= (X1 - X0) * 0.45)
    while paso * k * 4 > X1 - X0 - 2 * R - 40 and paso > 5:
        paso = {200: 100, 100: 50, 50: 25, 25: 20, 20: 10, 10: 5}.get(paso, 5)
    ex0, ey0 = kx + R + 34, ky - 2
    for i in range(4):
        d.rectangle([ex0 + i * paso * k, ey0, ex0 + (i + 1) * paso * k, ey0 + 7],
                    fill=NEGRO if i % 2 == 0 else (255, 255, 255), outline=NEGRO)
    for i in range(5):
        d.text((ex0 + i * paso * k, ey0 - 4), f"{i * paso}", font=fuente(10.5), fill=TINTA, anchor="md")
    d.text((ex0, ey0 + 14), "ESCALA GRÁFICA (m)", font=fuente(10.5), fill=GRIS, anchor="la")
    y = ky + R + 16
    regla(y)

    # relieve del terreno
    y += 12
    titulo_seccion("RELIEVE DEL TERRENO")
    datos = [("Cota más alta / más baja", f"{rel['cota_max']:,.1f} / {rel['cota_min']:,.1f} m"),
             ("Desnivel total", f"{rel['desnivel']:,.1f} m"),
             ("Baja hacia el", f"{rel['baja_hacia']} (plano general {rel['pendiente_general']:.0f} %)"),
             ("Pendiente media", f"{rel['pend_media']:.0f} %" if rel["pend_media"] is not None else "—"),
             ("Curvas de nivel", f"cada {iv:g} m (gruesas cada {iv * 5:g} m)")]
    for a_, b_ in datos:
        d.text((X0, y), a_, font=fuente(12), fill=GRIS, anchor="la")
        d.text((X1, y), b_, font=fuente(12), fill=NEGRO, anchor="ra")
        y += 18
    y += 8
    regla(y)

    # cuadro de pendientes
    y += 12
    titulo_seccion("CUADRO DE PENDIENTES")
    for (a_, b_, nombre, col), cl in zip(altimetria.CLASES, rel["clases"]):
        d.rectangle([X0, y + 1, X0 + 30, y + 13], fill=col, outline=(120, 120, 120))
        d.text((X0 + 40, y), nombre, font=fuente(12), fill=TINTA, anchor="la")
        d.text((X1 - 92, y), f"{cl['pct']:.0f} %", font=fuente(12), fill=NEGRO, anchor="ra")
        d.text((X1, y), f"{_m2(cl['m2'], 0)} m²", font=fuente(12), fill=NEGRO, anchor="ra")
        y += 19
    y += 8
    regla(y)

    # perfil longitudinal de la calle
    if eje is not None:
        prf = altimetria.perfil(malla, eje, 1.0)
        if len(prf) >= 3:
            y += 12
            titulo_seccion("PERFIL LONGITUDINAL DE LA CALLE (eje)")
            gx0_, gx1_, gy0_, gy1_ = X0 + 44, X1 - 6, y + 6, y + 150
            ds = [p[0] for p in prf]
            zs = [p[1] for p in prf]
            zmin, zmax = min(zs), max(zs)
            rango_z = max(2.0, zmax - zmin)
            z_lo = math.floor((zmin - 0.1 * rango_z) / 1) * 1
            z_hi = math.ceil((zmax + 0.1 * rango_z) / 1) * 1
            Lx = ds[-1] or 1

            def g(dd, zz):
                return (gx0_ + dd / Lx * (gx1_ - gx0_), gy1_ - (zz - z_lo) / (z_hi - z_lo) * (gy1_ - gy0_))

            paso_z = next(v for v in (0.5, 1, 2, 5, 10, 20, 50) if (z_hi - z_lo) / v <= 6)
            zz = math.ceil(z_lo / paso_z) * paso_z
            while zz <= z_hi:
                yy_ = g(0, zz)[1]
                d.line([(gx0_, yy_), (gx1_, yy_)], fill=(228, 228, 228), width=1)
                d.text((gx0_ - 5, yy_), f"{zz:g}", font=fuente(10), fill=GRIS, anchor="rm")
                zz += paso_z
            for dd in range(0, int(Lx) + 1, 20):
                xx_ = g(dd, z_lo)[0]
                d.line([(xx_, gy0_), (xx_, gy1_)], fill=(236, 236, 236), width=1)
                d.text((xx_, gy1_ + 4), f"0+{dd:03d}", font=fuente(9.5), fill=GRIS, anchor="ma")
            pts = [g(*p) for p in prf]
            d.polygon(pts + [(pts[-1][0], gy1_), (pts[0][0], gy1_)], fill=(240, 232, 220))
            d.line(pts, fill=CURVA_INDICE, width=2)
            d.rectangle([gx0_, gy0_, gx1_, gy1_], outline=(150, 150, 150), width=1)
            from .normas import LIMITES_PENDIENTE_CALLE

            _, media, peor = altimetria.pendiente_eje(malla, eje)
            lim = next((l_ for dd_, l_ in LIMITES_PENDIENTE_CALLE if Lx <= dd_), None)
            d.text((X0, gy1_ + 22), f"Largo {Lx:,.0f} m · desnivel {abs(zs[-1] - zs[0]):.1f} m · pendiente media "
                   f"{media:.1f} %", font=fuente(11.5), fill=TINTA, anchor="la")
            d.text((X0, gy1_ + 39), f"Tramo de 10 m más inclinado: {peor:.1f} %"
                   + (f" (norma: hasta {lim:g} %)" if lim else ""),
                   font=fuente(11.5), fill=ROJO if lim and peor > lim else TINTA, anchor="la")
            d.text((gx0_ - 40, gy0_ - 2), "cota (m)", font=fuente(9.5), fill=GRIS, anchor="ld")
            y = gy1_ + 62
            regla(y)

    # relieve por lote
    y += 12
    titulo_seccion("RELIEVE POR LOTE")
    cols = [("LOTE", X0, "la"), ("COTAS (m)", X0 + 150, "ra"), ("DESNIVEL", X0 + 232, "ra"), ("PEND. MEDIA", X0 + 330, "ra"),
            ("CAE HACIA", X1, "ra")]
    for t_, xx_, an_ in cols:
        d.text((xx_, y), t_, font=fuente(10, True), fill=GRIS, anchor=an_)
    y += 15
    d.line([(X0, y), (X1, y)], fill=GRIS, width=1)
    y += 4
    y_simb = H - MARGEN - 7 - 150 - 22 * 7 - 50
    filas_l = lot.lotes
    alto = 17
    cabe = max(1, int((y_simb - y - 10) / alto))
    for l in filas_l[:cabe]:
        rl = l.get("relieve") or {}
        vals = [str(l["id"]), f"{rl.get('cota_min', 0):,.1f}–{rl.get('cota_max', 0):,.1f}" if rl else "—",
                f"{rl['desnivel']:.1f} m" if rl else "—", f"{rl['pend_media']:.0f} %" if rl.get("pend_media") is not None else "—",
                rl.get("cae_hacia", "—")]
        fuerte = rl.get("pend_media") is not None and rl["pend_media"] > altimetria.PEND_CONSTRUIBLE
        for v, (_, xx_, an_) in zip(vals, cols):
            d.text((xx_, y), v, font=fuente(11.5, fuerte), fill=ROJO if fuerte else TINTA, anchor=an_)
        y += alto
    if len(filas_l) > cabe:
        d.text((X0, y), f"… y {len(filas_l) - cabe} lotes más (ver resumen ejecutivo)", font=fuente(11), fill=GRIS,
               anchor="la")

    # simbología
    y = y_simb
    regla(y)
    y += 12
    titulo_seccion("SIMBOLOGÍA")
    mx0, mx1, tx = X0, X0 + 46, X0 + 60
    items = [("indice", f"Curva de nivel índice (cada {iv * 5:g} m) con su cota"),
             ("curva", f"Curva de nivel intermedia (cada {iv:g} m)"),
             ("flecha", "Hacia dónde baja el terreno (escurrimiento)"),
             ("caida", "Caída fuerte (> 50 %): se deja fuera de los lotes"),
             ("proteccion", "Zona de protección y área no lotificada"),
             ("cruz", "Vértice del terreno con su cota (m)"),
             ("eje", "Eje de la calle con estación y cota cada 20 m")]
    for tipo, txt in items:
        yc = y + 7
        if tipo == "indice":
            d.line([(mx0, yc + 3), (mx0 + 15, yc - 2), (mx0 + 30, yc + 1), (mx1, yc - 3)], fill=CURVA_INDICE, width=2)
        elif tipo == "curva":
            d.line([(mx0, yc + 3), (mx0 + 15, yc - 2), (mx0 + 30, yc + 1), (mx1, yc - 3)], fill=CURVA, width=1)
        elif tipo == "flecha":
            d.line([(mx0 + 8, yc), (mx1 - 8, yc)], fill=AGUA_F, width=2)
            d.line([(mx1 - 8, yc), (mx1 - 15, yc - 5)], fill=AGUA_F, width=2)
            d.line([(mx1 - 8, yc), (mx1 - 15, yc + 5)], fill=AGUA_F, width=2)
        elif tipo == "caida":
            _discontinua(d, [(mx0, yc - 6), (mx1, yc - 6), (mx1, yc + 6), (mx0, yc + 6), (mx0, yc - 6)], ROJO, 2, 7, 4)
        elif tipo == "proteccion":
            _discontinua(d, [(mx0, yc - 6), (mx1, yc - 6), (mx1, yc + 6), (mx0, yc + 6), (mx0, yc - 6)], VERDE, 2, 7, 4)
        elif tipo == "cruz":
            d.line([(mx0 + 18, yc), (mx0 + 28, yc)], fill=NEGRO, width=1)
            d.line([(mx0 + 23, yc - 5), (mx0 + 23, yc + 5)], fill=NEGRO, width=1)
        elif tipo == "eje":
            _discontinua(d, [(mx0, yc), (mx1, yc)], (110, 110, 110), 1, 8, 5)
            d.ellipse([mx0 + 20, yc - 3, mx0 + 26, yc + 3], fill=NEGRO)
        d.text((tx, y + 7), txt, font=fuente(12), fill=TINTA, anchor="lm")
        y += 22

    # cajetín
    y = H - MARGEN - 7 - 150
    regla(y, 2)
    hoy = datetime.date.today()
    from .relieve import FUENTES

    filas_c = [("PROYECTO", (proyecto or "Anteproyecto")[:40]), ("CONTENIDO", "Plano altimétrico — pendientes y perfil"),
               ("FECHA", f"{MESES[hoy.month - 1]} {hoy.year}"), ("ESCALA", "Gráfica")]
    yy = y + 10
    for et, val in filas_c:
        d.text((X0, yy), et, font=fuente(10.5, True), fill=GRIS, anchor="la")
        d.text((X0 + 110, yy - 1), val, font=fuente(12.5), fill=NEGRO, anchor="la")
        yy += 20
    fuente_rel = FUENTES.get(malla.fuente, malla.fuente)
    nota = (f"Relieve: {fuente_rel}. Aproximado para anteproyecto: las caídas cortas se suavizan. Para diseño final y "
            "trámite se sustituye por la altimetría del levantamiento topográfico.")
    palabras, act, lineas_n = nota.split(), "", []
    for p_ in palabras:
        prueba = (act + " " + p_).strip()
        if d.textlength(prueba, font=fuente(10.5)) <= X1 - X0:
            act = prueba
        else:
            lineas_n.append(act)
            act = p_
    lineas_n.append(act)
    for ln in lineas_n[:3]:
        d.text((X0, yy + 2), ln, font=fuente(10.5), fill=GRIS, anchor="la")
        yy += 14
    return img.convert("RGB")
