"""Corte transversal y terracería: la lámina que dice cómo queda cada lote en la ladera, con cotas reales.

Corte A–A' perpendicular a la calle, por donde cruza más casas: terreno natural, calle a su cota, plataforma de cada
casa pegada a la calle a la cota de la calle (acceso del carro a nivel), taludes 1.5:1 hasta el terreno natural,
corte y relleno achurados, casas tipo y cotas. Columna: plano llave con la línea del corte, cuadro de terracería por
lote (cota de plataforma, corte y relleno en m³, alturas máximas), simbología y cajetín. Solo el motor (no usa IA).
"""
from __future__ import annotations

import datetime
import math

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from . import altimetria
from .lotificacion import Resultado
from .plano_tecnico import COLUMNA, GRIS, H_HOJA, MARGEN, MESES, NEGRO, TINTA, VERDE, W_HOJA, _discontinua, _m2
from .relieve import Malla
from .terreno import Terreno

TALUD = 1.5        # horizontal por cada metro vertical
CORTE_C, RELLENO_C = (214, 186, 150), (196, 214, 170)
NATURAL = (128, 96, 66)


def _linea_corte(lot: Resultado, pads: list, prop: Polygon) -> LineString | None:
    """Perpendicular a la calle, en la estación que cruza más casas (y más cerca del centro de la calle)."""
    st = lot.stats
    if not (st.get("entrada_calle") and st.get("retorno_centro")):
        return None
    eje = LineString([st["entrada_calle"], st["retorno_centro"]])
    (ax, ay), (bx, by) = eje.coords[0], eje.coords[-1]
    L = eje.length or 1
    ux, uy = (bx - ax) / L, (by - ay) / L
    nx, ny = -uy, ux
    R = max(prop.bounds[2] - prop.bounds[0], prop.bounds[3] - prop.bounds[1])
    mejor = None
    for s in np.arange(0.15 * L, 0.8 * L, 1.0):
        q = eje.interpolate(s)
        ln = LineString([(q.x - nx * R, q.y - ny * R), (q.x + nx * R, q.y + ny * R)])
        casas = sum(1 for p in pads if ln.intersects(p["casa"].buffer(-0.5)))
        clave = (casas, -abs(s - 0.45 * L))
        if mejor is None or clave > mejor[0]:
            mejor = (clave, ln)
    ln = mejor[1]
    dentro = ln.intersection(prop)
    partes = [g for g in getattr(dentro, "geoms", [dentro]) if g.geom_type == "LineString"]
    if not partes:
        return None
    xs = [c for g in partes for c in g.coords]
    a = min(xs, key=lambda c: (c[0] - ln.coords[0][0]) * nx + (c[1] - ln.coords[0][1]) * ny)
    b = max(xs, key=lambda c: (c[0] - ln.coords[0][0]) * nx + (c[1] - ln.coords[0][1]) * ny)
    m = 8.0
    return LineString([(a[0] - nx * m, a[1] - ny * m), (b[0] + nx * m, b[1] + ny * m)])


def generar_corte(t: Terreno, lot: Resultado, malla: Malla, ficha: dict | None, proyecto: str,
                  ubicacion: str = "") -> tuple[Image.Image, list[dict]]:
    from .presentacion import _anillos, _fuente

    ficha = ficha or {}
    W, H = W_HOJA, H_HOJA
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img, "RGBA")
    prop = Polygon(t.puntos).buffer(0)
    pads = altimetria.terrazas(lot, malla)  # terrazas que siguen el terreno, con muros de contención
    linea = _linea_corte(lot, pads, prop)

    def fuente(tam, negrita=False):
        return _fuente(max(8, int(round(tam))), negrita)

    # ------------------------------------------------------------------ perfil
    dx0, dx1 = MARGEN + 70, W - MARGEN - COLUMNA - 40
    dy0, dy1 = MARGEN + 120, H - MARGEN - 120
    if linea is not None:
        Ltot = linea.length
        paso = 0.25
        n = int(Ltot / paso) + 1
        ss = np.linspace(0, Ltot, n)
        pts = [linea.interpolate(s) for s in ss]
        nat = np.array([malla.altura(p.x, p.y) for p in pts])
        prop_z = nat.copy()
        fijo = np.zeros(n, dtype=bool)
        calles = unary_union(lot.calles) if lot.calles else None
        ent, ret = lot.stats.get("entrada_calle"), lot.stats.get("retorno_centro")
        eje_c = LineString([ent, ret]) if ent and ret else None
        info_pads = []
        terr = []
        for p in pads:
            for k_, tz in enumerate(p["terrazas"]):
                dentro = np.array([tz["poly"].contains(q) for q in pts])
                if dentro.any():
                    prop_z[dentro] = tz["z"]
                    fijo |= dentro
                    idx = np.nonzero(dentro)[0]
                    terr.append((p, tz, ss[idx[0]], ss[idx[-1]]))
                    if k_ == 0:
                        info_pads.append((p, ss[idx[0]], ss[idx[-1]]))
        if calles is not None:
            en_calle = np.array([calles.contains(q) for q in pts])
            if en_calle.any() and eje_c is not None:
                pc = linea.intersection(eje_c)
                zc = malla.altura(pc.x, pc.y) if not pc.is_empty and pc.geom_type == "Point" else float(np.nanmean(nat[en_calle]))
                prop_z[en_calle] = zc
                fijo |= en_calle
        zmin = float(np.nanmin(np.minimum(nat, prop_z)))
        zreal = float(np.nanmax(np.maximum(nat, prop_z)))
        zmax = zreal + 5.0
        kx = (dx1 - dx0) / Ltot
        kz = kx
        exag = 1
        if (zmax - zmin) * kz < (dy1 - dy0) * 0.25:
            exag = 2
            kz = kx * 2
        if (zmax - zmin) * kz > (dy1 - dy0):
            kz = (dy1 - dy0) / (zmax - zmin)
            exag = round(kz / kx, 2)
        base_y = (dy0 + dy1) / 2 + (zmax - zmin) * kz / 2

        def P(s, z):
            return (dx0 + s * kx, base_y - (z - zmin) * kz)

        # rejilla de cotas
        paso_z = next(v for v in (1, 2, 5, 10, 20, 50) if (zmax - zmin) / v <= 10)
        zz = math.ceil(zmin / paso_z) * paso_z
        while zz <= zmax:
            y = P(0, zz)[1]
            d.line([(dx0, y), (dx1, y)], fill=(236, 236, 236), width=1)
            d.text((dx0 - 8, y), f"{zz:g}", font=fuente(11), fill=GRIS, anchor="rm")
            zz += paso_z
        d.text((dx0 - 8, dy0 - 30), "cota (m)", font=fuente(11), fill=GRIS, anchor="rm")
        # tierra bajo el perfil propuesto (primero, para que el relleno quede encima y se vea)
        poly_suelo = [P(ss[i], prop_z[i]) for i in range(n)] + [P(ss[-1], zmin - 2), P(ss[0], zmin - 2)]
        d.polygon(poly_suelo, fill=(246, 241, 232))
        # corte y relleno
        for i in range(n - 1):
            a, b = nat[i], prop_z[i]
            if abs(a - b) < 0.05:
                continue
            col = CORTE_C if a > b else RELLENO_C
            x0_, _ = P(ss[i], 0)
            x1_, _ = P(ss[i + 1], 0)
            ya, yb = P(ss[i], a)[1], P(ss[i], b)[1]
            d.rectangle([x0_, min(ya, yb), x1_ + 0.6, max(ya, yb)], fill=col + (255,))
        # línea propuesta y natural
        d.line([P(ss[i], prop_z[i]) for i in range(n)], fill=NEGRO, width=3)
        # muros de contención: donde el perfil terminado salta de nivel (entre terrazas, a la calle o al terreno)
        muros_c = []
        for i in range(n - 1):
            a_, b_ = prop_z[i], prop_z[i + 1]
            if abs(a_ - b_) >= 0.3:
                (mx, ya_), (_, yb_) = P(ss[i], a_), P(ss[i], b_)
                d.rectangle([mx - 3, min(ya_, yb_), mx + 3, max(ya_, yb_)], fill=(110, 104, 96), outline=NEGRO)
                muros_c.append((mx, min(ya_, yb_), abs(a_ - b_)))
        for mx, ytop, h in muros_c:
            d.text((mx, ytop - 6), f"muro {h:.1f} m", font=fuente(10.5, True), fill=(90, 84, 76), anchor="ms")
        _discontinua(d, [P(ss[i], nat[i]) for i in range(n)], NATURAL, 2, 9, 5)
        # calle
        if calles is not None and en_calle.any():
            idx = np.nonzero(en_calle)[0]
            sa, sb = ss[idx[0]], ss[idx[-1]]
            (xa, ya), (xb, yb) = P(sa, prop_z[idx[0]]), P(sb, prop_z[idx[-1]])
            d.rectangle([xa, ya - 5, xb, ya + 2], fill=(150, 150, 150))
            d.text(((xa + xb) / 2, ya - 22), f"CALLE · {lot.parametros.get('calle', 8):g} m", font=fuente(13, True),
                   fill=TINTA, anchor="ms")
            d.text(((xa + xb) / 2, ya + 22), f"{prop_z[idx[len(idx) // 2]]:.2f}", font=fuente(12, True), fill=NEGRO,
                   anchor="mt")
        # casas y plataformas
        y_rot = P(0, zreal + 4.4 / (kz / kx if kz else 1) * 1.0 + 3.0)[1] - 10
        y_rot = min(y_rot, P(0, zreal)[1] - (4.4 * kz) - 40)
        for p, sa, sb in info_pads:
            (xa, ya), (xb, _) = P(sa, p["z"]), P(sb, p["z"])
            cruza = linea.intersection(p["casa"])
            if not cruza.is_empty and cruza.length > 1:
                ca = linea.project(Point(cruza.coords[0]))
                cb = linea.project(Point(cruza.coords[-1]))
                ca, cb = min(ca, cb), max(ca, cb)
                (hx0, hy0), (hx1, _) = P(ca, p["z"]), P(cb, p["z"])
                alto = 2.8 * kz
                techo = 1.6 * kz
                d.rectangle([hx0, hy0 - alto, hx1, hy0], fill=(250, 250, 248), outline=NEGRO, width=2)
                d.polygon([(hx0 - 6, hy0 - alto), (hx1 + 6, hy0 - alto), ((hx0 + hx1) / 2, hy0 - alto - techo)],
                          fill=(196, 120, 96), outline=NEGRO)
                for k_ in range(1, 3):
                    wx = hx0 + (hx1 - hx0) * k_ / 3
                    d.rectangle([wx - 8, hy0 - alto * 0.7, wx + 8, hy0 - alto * 0.25], fill=(190, 214, 230), outline=GRIS)
            d.text(((xa + xb) / 2, ya + 16), f"terraza casa {p['z']:.2f}", font=fuente(11.5, True), fill=NEGRO, anchor="mt")
            dz_txt = []
            if p["corte_max"] >= 0.3:
                dz_txt.append(f"corte hasta {p['corte_max']:.1f} m")
            if p["relleno_max"] >= 0.3:
                dz_txt.append(f"relleno hasta {p['relleno_max']:.1f} m")
            if dz_txt:
                d.text(((xa + xb) / 2, ya + 32), " · ".join(dz_txt), font=fuente(10.5), fill=TINTA, anchor="mt")
            d.text(((xa + xb) / 2, y_rot), f"LOTE {p['id']}", font=fuente(16, True), fill=NEGRO, anchor="ms")
        for p, tz, sa, sb in terr:  # terrazas de jardín
            if tz["uso"] == "casa":
                continue
            (xa, ya), (xb, _) = P(sa, tz["z"]), P(sb, tz["z"])
            d.text(((xa + xb) / 2, ya + 14), f"jardín {tz['z']:.2f}", font=fuente(10.5), fill=TINTA, anchor="mt")
        # linderos que cruza el corte (lotes y terreno)
        for l in lot.lotes:
            g = linea.intersection(l["poly"].boundary)
            for q in getattr(g, "geoms", [g]):
                if q.geom_type == "Point":
                    s = linea.project(q)
                    x, y = P(s, malla.altura(q.x, q.y))
                    _discontinua(d, [(x, y_rot + 12), (x, y)], (150, 150, 150), 1, 6, 5)
        g = linea.intersection(prop.boundary)
        for q in getattr(g, "geoms", [g]):
            if q.geom_type == "Point":
                s = linea.project(q)
                x, y = P(s, malla.altura(q.x, q.y))
                d.line([(x, y_rot + 4), (x, y)], fill=NEGRO, width=3)
                d.text((x, y + 8), f"{malla.altura(q.x, q.y):.1f}", font=fuente(11.5, True), fill=NEGRO, anchor="mt")
        d.text((dx0, dy1 + 40), "A", font=fuente(30, True), fill=NEGRO, anchor="lm")
        d.text((dx1, dy1 + 40), "A'", font=fuente(30, True), fill=NEGRO, anchor="rm")
        d.text(((dx0 + dx1) / 2, dy1 + 40),
               f"CORTE TRANSVERSAL A–A' · escala horizontal gráfica"
               + (f" · escala vertical ×{exag:g}" if exag != 1 else " · misma escala vertical"),
               font=fuente(15, True), fill=NEGRO, anchor="mm")
        # escala gráfica
        paso_e = next(v for v in (5, 10, 20, 25, 50) if v * kx >= 70)
        ex0, ey0 = dx0, dy1 + 78
        for i in range(4):
            d.rectangle([ex0 + i * paso_e * kx, ey0, ex0 + (i + 1) * paso_e * kx, ey0 + 7],
                        fill=NEGRO if i % 2 == 0 else (255, 255, 255), outline=NEGRO)
        for i in range(5):
            d.text((ex0 + i * paso_e * kx, ey0 - 4), f"{i * paso_e}", font=fuente(10.5), fill=TINTA, anchor="md")
    else:
        d.text(((dx0 + dx1) / 2, (dy0 + dy1) / 2), "Sin calle central: el corte se agrega en una versión próxima",
               font=fuente(18), fill=GRIS, anchor="mm")

    # ------------------------------------------------------------------ marco y columna
    d.rectangle([MARGEN, MARGEN, W - MARGEN, H - MARGEN], outline=NEGRO, width=3)
    d.rectangle([MARGEN + 7, MARGEN + 7, W - MARGEN - 7, H - MARGEN - 7], outline=NEGRO, width=1)
    cx0 = W - MARGEN - COLUMNA
    d.line([(cx0, MARGEN + 7), (cx0, H - MARGEN - 7)], fill=NEGRO, width=2)
    X0, X1 = cx0 + 22, W - MARGEN - 22
    y = MARGEN + 30
    d.text((X0, y), "ANTEPROYECTO DE LOTIFICACIÓN", font=fuente(13, True), fill=GRIS, anchor="la")
    y += 24
    d.text((X0, y), (proyecto or "Anteproyecto").upper()[:34], font=fuente(25, True), fill=NEGRO, anchor="la")
    y += 34
    d.text((X0, y), "CORTE TRANSVERSAL Y TERRACERÍA", font=fuente(14, True), fill=TINTA, anchor="la")
    y += 22
    if ubicacion:
        d.text((X0, y), ubicacion[:60], font=fuente(12.5), fill=GRIS, anchor="la")
        y += 18
    y += 10
    d.line([(cx0, y), (W - MARGEN - 7, y)], fill=NEGRO, width=2)

    # plano llave
    y += 12
    d.text((X0, y), "UBICACIÓN DEL CORTE", font=fuente(13, True), fill=NEGRO, anchor="la")
    y += 22
    kx0, ky0, kx1, ky1 = X0, y, X1, y + 250
    bx0, by0, bx1, by1 = prop.bounds
    kk = min((kx1 - kx0) / (bx1 - bx0 + 20), (ky1 - ky0) / (by1 - by0 + 20))
    ccx, ccy = (bx0 + bx1) / 2, (by0 + by1) / 2

    def K(x, yy):
        return ((kx0 + kx1) / 2 + (x - ccx) * kk, (ky0 + ky1) / 2 - (yy - ccy) * kk)

    for c_ in lot.calles:
        for ext, _ in _anillos(c_):
            d.polygon([K(*q) for q in ext], fill=(225, 225, 225))
    for l in lot.lotes:
        d.line([K(*q) for q in l["poly"].exterior.coords], fill=(120, 120, 120), width=1)
        c = l["poly"].representative_point()
        d.text(K(c.x, c.y), str(l["id"]), font=fuente(11, True), fill=TINTA, anchor="mm")
    for p in pads:
        for tz in p["terrazas"]:
            if tz["poly"].geom_type == "Polygon":
                d.polygon([K(*q) for q in tz["poly"].exterior.coords],
                          fill=(214, 186, 150, 170) if tz["uso"] == "casa" else (196, 214, 170, 150))
    d.line([K(*q) for q in prop.exterior.coords], fill=NEGRO, width=2)
    if linea is not None:
        (la, lb) = linea.coords[0], linea.coords[-1]
        d.line([K(*la), K(*lb)], fill=(178, 52, 40), width=3)
        d.text(K(*la), "A", font=fuente(15, True), fill=(178, 52, 40), anchor="rm")
        d.text(K(*lb), "A'", font=fuente(15, True), fill=(178, 52, 40), anchor="lm")
    y = ky1 + 12
    d.line([(cx0, y), (W - MARGEN - 7, y)], fill=NEGRO, width=1)

    # cuadro de terracería
    y += 12
    d.text((X0, y), "TERRAZAS Y MUROS POR LOTE", font=fuente(13, True), fill=NEGRO, anchor="la")
    y += 22
    cols = [("LOTE", X0, "la"), ("CASA", X0 + 92, "ra"), ("CORTE", X0 + 160, "ra"), ("RELL.", X0 + 222, "ra"),
            ("MURO m²", X0 + 300, "ra"), ("MÁX", X0 + 350, "ra"), ("TERR.", X1, "ra")]
    for t_, xx, an in cols:
        d.text((xx, y), t_, font=fuente(10, True), fill=GRIS, anchor=an)
    y += 15
    d.line([(X0, y), (X1, y)], fill=GRIS, width=1)
    y += 4
    tc = tr = 0
    tm = 0
    for p in pads[:16]:
        vals = [str(p["id"]), f"{p['z']:.1f}", f"{p['corte_m3']:,}", f"{p['relleno_m3']:,}", f"{p['muro_m2']:,}",
                f"{p['muro_max']:.1f}", str(len(p["terrazas"]))]
        alto = p["muro_max"] > altimetria.MURO_ESTRUCTURAL
        for v, (_, xx, an) in zip(vals, cols):
            d.text((xx, y), v, font=fuente(11.5, alto and an == "ra" and v == vals[5]), fill=(178, 52, 40) if (alto and v == vals[5]) else TINTA, anchor=an)
        y += 17
    for p in pads:
        tc += p["corte_m3"]
        tr += p["relleno_m3"]
        tm += p["muro_m2"]
    d.line([(X0, y + 2), (X1, y + 2)], fill=NEGRO, width=1)
    y += 6
    d.text((X0, y), "Total m³ / m²", font=fuente(11.5, True), fill=NEGRO, anchor="la")
    d.text((X0 + 160, y), f"{tc:,}", font=fuente(11.5, True), fill=NEGRO, anchor="ra")
    d.text((X0 + 222, y), f"{tr:,}", font=fuente(11.5, True), fill=NEGRO, anchor="ra")
    d.text((X0 + 300, y), f"{tm:,}", font=fuente(11.5, True), fill=NEGRO, anchor="ra")
    y += 24
    nota = ("Terrazas que siguen el terreno: la de la casa va pegada a la calle y a no más de 1.5 m de su nivel (el carro "
            "entra por una rampa lateral de ≤ 15 %); el jardín va en terrazas escalonadas. Los cambios de nivel se "
            "resuelven con muros de contención (en rojo, más de 3 m: piden diseño estructural). CASA: cota de la "
            "terraza; CORTE y RELL. en m³; MURO: cara de muro en m²; MÁX: altura máxima (m). Volúmenes aproximados.")
    palabras, act, lineas_n = nota.split(), "", []
    for p_ in palabras:
        prueba = (act + " " + p_).strip()
        if d.textlength(prueba, font=fuente(11)) <= X1 - X0:
            act = prueba
        else:
            lineas_n.append(act)
            act = p_
    lineas_n.append(act)
    for ln in lineas_n:
        d.text((X0, y), ln, font=fuente(11), fill=TINTA, anchor="la")
        y += 15

    # simbología
    y = H - MARGEN - 7 - 150 - 22 * 6 - 40
    d.line([(cx0, y), (W - MARGEN - 7, y)], fill=NEGRO, width=1)
    y += 12
    d.text((X0, y), "SIMBOLOGÍA", font=fuente(13, True), fill=NEGRO, anchor="la")
    y += 24
    mx0, mx1, tx = X0, X0 + 46, X0 + 60
    for tipo, txt in (("prop", "Terreno terminado (terrazas y calle)"), ("muro", "Muro de contención"),
                      ("nat", "Terreno natural (relieve)"), ("corte", "Corte"), ("relleno", "Relleno"),
                      ("lote", "Lindero de lote / del terreno")):
        yc = y + 7
        if tipo == "prop":
            d.line([(mx0, yc), (mx1, yc)], fill=NEGRO, width=3)
        elif tipo == "muro":
            d.rectangle([mx0 + 18, yc - 8, mx0 + 26, yc + 8], fill=(110, 104, 96), outline=NEGRO)
        elif tipo == "nat":
            _discontinua(d, [(mx0, yc), (mx1, yc)], NATURAL, 2, 9, 5)
        elif tipo == "corte":
            d.rectangle([mx0, yc - 6, mx1, yc + 6], fill=CORTE_C)
        elif tipo == "relleno":
            d.rectangle([mx0, yc - 6, mx1, yc + 6], fill=RELLENO_C)
        else:
            _discontinua(d, [(mx0 + 10, yc - 7), (mx0 + 10, yc + 7)], (150, 150, 150), 1, 4, 3)
            d.line([(mx0 + 30, yc - 7), (mx0 + 30, yc + 7)], fill=NEGRO, width=3)
        d.text((tx, y + 7), txt, font=fuente(12), fill=TINTA, anchor="lm")
        y += 22

    # cajetín
    y = H - MARGEN - 7 - 150
    d.line([(cx0, y), (W - MARGEN - 7, y)], fill=NEGRO, width=2)
    hoy = datetime.date.today()
    yy = y + 10
    for et, val in (("PROYECTO", (proyecto or "Anteproyecto")[:40]), ("CONTENIDO", "Corte transversal y terracería"),
                    ("FECHA", f"{MESES[hoy.month - 1]} {hoy.year}"), ("ESCALA", "Gráfica")):
        d.text((X0, yy), et, font=fuente(10.5, True), fill=GRIS, anchor="la")
        d.text((X0 + 110, yy - 1), val, font=fuente(12.5), fill=NEGRO, anchor="la")
        yy += 20
    for ln in ("Relieve satelital aproximado: las cotas y volúmenes son de anteproyecto. El diseño final de",
               "terracería se hace con la altimetría del levantamiento topográfico."):
        d.text((X0, yy + 2), ln, font=fuente(10.5), fill=GRIS, anchor="la")
        yy += 14
    return img, pads
