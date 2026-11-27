"""Resumen ejecutivo del anteproyecto (PDF tamaño carta): lo que un dueño o un desarrollador lee primero.

Todo sale de la geometría del motor (no se estima a ojo): cuadro general de áreas por terreno en m² y v², la
verificación de cada plano (lo que dice contra lo que da su cuadro de rumbos), los lotes con su relieve, la calle, el
relieve del terreno, el cumplimiento de normas, el esquema legal a validar y los siguientes pasos.
"""
from __future__ import annotations

import datetime
import logging
import math

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from shapely.geometry import Polygon
from shapely.ops import unary_union

from . import altimetria
from .lotificacion import Resultado
from .rumbos import M2_POR_V2
from .terreno import Terreno

log = logging.getLogger(__name__)

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre",
         "diciembre"]
TINTA = colors.HexColor("#1d1d1d")
GRIS = colors.HexColor("#6f6f6f")
LINEA = colors.HexColor("#c9c9c9")
FONDO = colors.HexColor("#f3f1ec")
ACENTO = colors.HexColor("#2f5d50")
ROJO = colors.HexColor("#a3372a")
AMBAR = colors.HexColor("#9a6a12")
ESTADOS = {"cumple": ("Cumple", ACENTO), "no_cumple": ("No cumple", ROJO), "revisar": ("Revisar", AMBAR),
           "falta_dato": ("Falta dato", GRIS)}


def _fuentes():
    if "DJ" in pdfmetrics.getRegisteredFontNames():
        return
    from matplotlib import font_manager

    for nombre, peso in (("DJ", "normal"), ("DJB", "bold")):
        ruta = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=peso))
        pdfmetrics.registerFont(TTFont(nombre, ruta))


def _n(v, dec=2):
    return f"{v:,.{dec}f}"


def _v2(m2):
    return m2 / M2_POR_V2


def _area_en(geoms, poly: Polygon) -> float:
    g = [x for x in geoms if x is not None and not x.is_empty]
    if not g:
        return 0.0
    return unary_union(g).intersection(poly).area


def recortar_dibujo(ruta: str, salida: str, t: Terreno | None = None) -> str | None:
    """Del plano técnico, solo el terreno y sus letreros (sin la columna, el marco ni la rejilla), para el resumen."""
    from PIL import Image, ImageChops

    from .plano_tecnico import COLUMNA, MARGEN, caja_terreno_px

    try:
        im = Image.open(ruta).convert("RGB")
        W, H = im.size
        if t is not None:
            im = im.crop(caja_terreno_px(t))
        else:
            im = im.crop((MARGEN + 10, MARGEN + 10, W - MARGEN - COLUMNA - 4, H - MARGEN - 10))
            caja = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).convert("L").point(
                lambda v: 255 if v > 40 else 0).getbbox()
            if caja:
                im = im.crop((max(0, caja[0] - 14), max(0, caja[1] - 14), min(im.width, caja[2] + 14),
                              min(im.height, caja[3] + 14)))
        im.save(salida, "JPEG", quality=88)
        return salida
    except Exception:  # noqa: BLE001
        log.exception("recorte del dibujo")
        return None


def cuadro_areas(t: Terreno, lot: Resultado, partes: list | None) -> dict:
    """Áreas por uso y por terreno, calculadas cruzando la geometría del prediseño con cada terreno."""
    prop = Polygon(t.puntos).buffer(0)
    piezas = [(p["origen"], Polygon(p["anillo"]).buffer(0)) for p in (partes or [])] or [("Terreno", prop)]
    usos = [("Lotes", [l["poly"] for l in lot.lotes]), ("Calle y redondel", lot.calles),
            ("Sin lotes (protección y caídas fuertes)", lot.protecciones),
            ("Área verde recreativa", [v["poly"] for v in lot.verdes]),
            ("Equipamiento social", [e["poly"] for e in lot.equipamiento])]
    filas = []
    for nombre, geoms in usos:
        vals = [_area_en(geoms, pz) for _, pz in piezas]
        if sum(vals) > 0.5:
            filas.append((nombre, vals))
    bruta = [pz.area for _, pz in piezas]
    usado = [sum(f[1][i] for f in filas) for i in range(len(piezas))]
    resto = [b - u for b, u in zip(bruta, usado)]
    if sum(resto) > 1.0:
        filas.append(("Remanente / ajustes", resto))
    return {"terrenos": [n for n, _ in piezas], "bruta": bruta, "filas": filas, "total": prop.area}


def generar_resumen(ruta: str, t: Terreno, lot: Resultado, malla, ficha: dict | None, proyecto: str,
                    partes: list | None = None, ubicacion: str = "", cumplimiento: list | None = None,
                    imagen: str | None = None, entrega: dict | None = None) -> dict:
    _fuentes()
    ficha = ficha or {}
    st = lot.stats
    est = {
        "t": ParagraphStyle("t", fontName="DJB", fontSize=19, leading=23, textColor=TINTA),
        "sup": ParagraphStyle("sup", fontName="DJB", fontSize=8.5, leading=11, textColor=ACENTO),
        "sub": ParagraphStyle("sub", fontName="DJ", fontSize=9.5, leading=13, textColor=GRIS),
        "h": ParagraphStyle("h", fontName="DJB", fontSize=11.5, leading=15, textColor=ACENTO, spaceBefore=9, spaceAfter=4),
        "p": ParagraphStyle("p", fontName="DJ", fontSize=9, leading=12.5, textColor=TINTA, alignment=TA_LEFT),
        "pg": ParagraphStyle("pg", fontName="DJ", fontSize=8, leading=10.5, textColor=GRIS),
        "c": ParagraphStyle("c", fontName="DJ", fontSize=8.3, leading=10.5, textColor=TINTA),
        "cb": ParagraphStyle("cb", fontName="DJB", fontSize=8.3, leading=10.5, textColor=TINTA),
        "kn": ParagraphStyle("kn", fontName="DJB", fontSize=17, leading=20, textColor=TINTA),
        "kt": ParagraphStyle("kt", fontName="DJ", fontSize=7.8, leading=10, textColor=GRIS),
    }
    P = lambda txt, s="p": Paragraph(txt, est[s])  # noqa: E731
    W = letter[0] - 36 * mm
    hoy = datetime.date.today()
    story = []

    # ------------------------------------------------------------------ encabezado
    story += [P("RESUMEN EJECUTIVO · ANTEPROYECTO DE LOTIFICACIÓN", "sup"), Spacer(1, 3),
              P(proyecto or "Anteproyecto", "t"), Spacer(1, 2),
              P(" · ".join(x for x in (ubicacion, f"{MESES[hoy.month - 1]} de {hoy.year}") if x), "sub"), Spacer(1, 10)]

    n = st["lotes"]
    areas = [l["area"] for l in lot.lotes]
    calle_w = (lot.parametros or {}).get("calle")
    rel = None
    if malla is not None:
        try:
            rel = altimetria.resumen(malla, t.puntos)
        except Exception:  # noqa: BLE001
            log.exception("relieve del resumen")
    kpis = [(f"{n}", "lotes" + (" (quintas)" if ficha.get("tipo") == "campestre" else "")),
            (_n(st["area_util"], 0), f"m² vendibles · {_n(_v2(st['area_util']), 0)} v²"),
            (_n(sum(areas) / n, 0), f"m² lote promedio ({_n(min(areas), 0)}–{_n(max(areas), 0)})"),
            (f"{st['area_util'] / st['area_total'] * 100:.0f} %", "del terreno se vende")]
    if rel:
        kpis.append((f"{rel['desnivel']:.0f} m", f"de desnivel · baja hacia el {rel['baja_hacia']}"))
    celdas = [[P(a, "kn") for a, _ in kpis], [P(b, "kt") for _, b in kpis]]
    tk = Table(celdas, colWidths=[W / len(kpis)] * len(kpis))
    tk.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), FONDO), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, 0), 8),
                            ("BOTTOMPADDING", (0, 1), (-1, 1), 8), ("LINEAFTER", (0, 0), (-2, -1), 0.6, colors.white)]))
    story += [tk, Spacer(1, 6)]

    # ------------------------------------------------------------------ cuadro general de áreas
    ca = cuadro_areas(t, lot, partes)
    nt = len(ca["terrenos"])
    story.append(P("1. Cuadro general de áreas", "h"))
    enc = ["Uso del suelo"] + [f"{x} (m²)" for x in ca["terrenos"]] + (["Total (m²)"] if nt > 1 else []) + ["Total (v²)", "%"]
    filas = [enc]
    filas.append(["Superficie del terreno (según su cuadro de rumbos)"] + [_n(b) for b in ca["bruta"]] +
                 ([_n(sum(ca["bruta"]))] if nt > 1 else []) + [_n(_v2(sum(ca["bruta"]))), "100 %"])
    for nombre, vals in ca["filas"]:
        tot = sum(vals)
        filas.append([nombre] + [_n(v) for v in vals] + ([_n(tot)] if nt > 1 else []) + [_n(_v2(tot)),
                                                                                           f"{tot / sum(ca['bruta']) * 100:.1f} %"])
    filas = [[P(c, "cb" if i == 0 else "c") if j == 0 else P(c, "cb" if i == 0 else "c") for j, c in enumerate(f)]
             for i, f in enumerate(filas)]
    anchos = [W * 0.34] + [(W * 0.66) / (len(enc) - 1)] * (len(enc) - 1)
    ta = Table(filas, colWidths=anchos, repeatRows=1)
    i_lotes = next((i for i, f in enumerate(ca["filas"], start=2) if f[0] == "Lotes"), None)
    estilo = [("LINEBELOW", (0, 0), (-1, 0), 0.8, TINTA), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINEA),
              ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("BACKGROUND", (0, 1), (-1, 1), FONDO), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if i_lotes:
        estilo.append(("BACKGROUND", (0, i_lotes), (-1, i_lotes), colors.HexColor("#e6efe9")))
    ta.setStyle(TableStyle(estilo))
    story.append(ta)
    if st.get("calle_dentro_de"):
        story.append(Spacer(1, 3))
        story.append(P(f"La calle y el redondel quedan completos dentro del terreno {st['calle_dentro_de']}.", "pg"))

    # ------------------------------------------------------------------ verificación de los planos
    avisos = []
    ver = []
    for p in partes or []:
        dec, cuadro, cierre = p.get("area_declarada"), p.get("area_cuadro"), p.get("cierre")
        if cuadro is None:
            continue
        dif = (cuadro - dec) if dec else None
        ok_cierre = cierre is None or cierre <= 0.30
        ok_area = dif is None or abs(dif) <= max(1.0, 0.005 * dec)
        estado = "Cuadra" if ok_cierre and ok_area else "Revisar"
        ver.append([p["origen"], _n(dec) if dec else "no indica", _n(cuadro), (_n(dif) if dif is not None else "—"),
                    f"{cierre:.2f} m" if cierre is not None else "—", estado])
        if not ok_cierre or not ok_area:
            txt = f"El cuadro de rumbos del {p['origen']}"
            if not ok_cierre:
                txt += f" no cierra ({cierre:.2f} m)"
            if not ok_area:
                txt += (" y" if not ok_cierre else "") + f" da {_n(cuadro)} m², {_n(abs(dif))} m² {'más' if dif > 0 else 'menos'} " \
                       f"que lo que dice el plano ({_n(dec)} m²)"
            avisos.append(txt + ". Las áreas de ese lado son preliminares hasta que el topógrafo corrija el cuadro.")
    if ver:
        story.append(P("2. Verificación de los planos", "h"))
        filas = [["Terreno", "Área que dice el plano (m²)", "Área de su cuadro de rumbos (m²)", "Diferencia (m²)",
                  "Error de cierre", "Estado"]] + ver
        tv = Table([[P(c, "cb" if i == 0 else "c") for c in f] for i, f in enumerate(filas)],
                   colWidths=[W * 0.14, W * 0.19, W * 0.21, W * 0.16, W * 0.15, W * 0.15])
        sv = [("LINEBELOW", (0, 0), (-1, 0), 0.8, TINTA), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINEA),
              ("ALIGN", (1, 1), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
        for i, f in enumerate(ver, start=1):
            if f[-1] != "Cuadra":
                sv.append(("TEXTCOLOR", (5, i), (5, i), ROJO))
        tv.setStyle(TableStyle(sv))
        story.append(tv)
        for a in avisos:
            story += [Spacer(1, 3), P(f"<font color='#a3372a'><b>Atención:</b></font> {a}", "p")]

    # ------------------------------------------------------------------ imagen del plano
    if imagen:
        try:
            from PIL import Image as PILImage

            im = PILImage.open(imagen)
            ancho_img = W
            alto_img = W * im.height / im.width
            if alto_img > 92 * mm:
                alto_img = 92 * mm
                ancho_img = alto_img * im.width / im.height
            story += [Spacer(1, 8), RLImage(imagen, width=ancho_img, height=alto_img)]
        except Exception:  # noqa: BLE001
            log.exception("imagen del resumen")

    # ------------------------------------------------------------------ lotes
    num = 3 if ver else 2
    story.append(P(f"{num}. Lotes", "h"))
    num += 1
    con_rel = any(l.get("relieve") for l in lot.lotes)
    enc = ["Lote", "Terreno", "Área (m²)", "Área (v²)", "Frente (m)"] + (["Pendiente media", "Desnivel", "Cae hacia"]
                                                                       if con_rel else [])
    filas = [enc]
    for l in lot.lotes:
        rl = l.get("relieve") or {}
        filas.append([str(l["id"]), l.get("terreno") or "—", _n(l["area"]), _n(_v2(l["area"])),
                      _n(l["frente"], 1) if l.get("frente") else "—"]
                     + ([f"{rl['pend_media']:.0f} %" if rl.get("pend_media") is not None else "—",
                         f"{rl['desnivel']:.1f} m" if rl else "—", rl.get("cae_hacia", "—")] if con_rel else []))
    filas.append(["Total", "", _n(sum(areas)), _n(_v2(sum(areas))), ""] + (["", "", ""] if con_rel else []))
    anchos_l = ([W * 0.08, W * 0.14, W * 0.14, W * 0.14, W * 0.12, W * 0.14, W * 0.11, W * 0.13] if con_rel
                else [W * 0.12, W * 0.22, W * 0.22, W * 0.22, W * 0.22])
    tl = Table([[P(c, "cb" if i in (0, len(filas) - 1) else "c") for c in f] for i, f in enumerate(filas)],
               colWidths=anchos_l, repeatRows=1)
    sl = [("LINEBELOW", (0, 0), (-1, 0), 0.8, TINTA), ("LINEBELOW", (0, 1), (-1, -2), 0.3, LINEA),
          ("LINEABOVE", (0, -1), (-1, -1), 0.8, TINTA), ("ALIGN", (2, 1), (-2 if con_rel else -1, -1), "RIGHT"),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    for i, l in enumerate(lot.lotes, start=1):
        rl = l.get("relieve") or {}
        if rl.get("pend_media") is not None and rl["pend_media"] > altimetria.PEND_CONSTRUIBLE:
            sl.append(("TEXTCOLOR", (5, i), (5, i), ROJO))
    tl.setStyle(TableStyle(sl))
    story.append(tl)
    if rel:
        story += [Spacer(1, 3), P("Pendiente y desnivel de cada lote salen del relieve (todo el lote, no solo el centro). "
                                 "Arriba de 30 % la casa necesita terrazas con muro o talud.", "pg")]

    # ------------------------------------------------------------------ calle y servicios
    story.append(P(f"{num}. Calle, acceso y servicios", "h"))
    num += 1
    datos = []
    if calle_w:
        datos.append(("Calle interna", f"{calle_w:g} m de ancho, {_n(st.get('largo_calles', 0), 0)} m de largo"
                                       + (f", con redondel de retorno de {st['retorno_diametro_m']:.0f} m" if st.get("retorno_diametro_m") else "")))
        if calle_w >= 6.5:
            datos.append(("Sección propuesta", f"rodaje de {calle_w - 1.5:g} m y aceras de 0.75 m a cada lado, con cordón y "
                                               "cuneta (a confirmar en el diseño final)"))
    if malla is not None and st.get("entrada_calle") and st.get("retorno_centro"):
        from shapely.geometry import LineString

        from .normas import LIMITES_PENDIENTE_CALLE

        eje = LineString([st["entrada_calle"], st["retorno_centro"]])
        largo, media, peor = altimetria.pendiente_eje(malla, eje)
        if math.isfinite(media):
            z0, z1 = malla.altura(*eje.coords[0]), malla.altura(*eje.coords[-1])
            lim = next((l_ for dd_, l_ in LIMITES_PENDIENTE_CALLE if largo <= dd_), None)
            datos.append(("Pendiente de la calle", f"{'baja' if z1 < z0 else 'sube'} {abs(z1 - z0):.1f} m de la entrada al "
                                                   f"redondel: {media:.1f} % en promedio y {peor:.1f} % en su tramo de "
                                                   "10 m más inclinado"
                                                   + (f" (norma: hasta {lim:g} % en calles de hasta "
                                                      f"{next(dd_ for dd_, l_ in LIMITES_PENDIENTE_CALLE if l_ == lim)} m)"
                                                      if lim else "")))
    if ficha.get("cerrado"):
        n_acc = int(st.get("accesos") or 1)
        if n_acc <= 1:
            datos.append(("Acceso", "un solo acceso desde la calle pública, con portón y caseta de vigilancia"))
        else:  # lo mismo que dice el cuadro de normas: no prometer un solo portón si el trazo tiene varias salidas
            datos.append(("Acceso", f"el prediseño tiene {n_acc} salidas a la calle pública; para cerrarlo con un solo "
                                    "portón hay que juntarlas (o poner portón en cada una)"))
    agua = {"anda": "red de ANDA (pedir factibilidad)", "pozo": "pozo propio (pedir permiso y prueba de caudal)"}.get(
        ficha.get("agua"), "por definir (ANDA, pozo o sistema comunitario)")
    datos += [("Agua potable", agua),
              ("Energía eléctrica", "red de la distribuidora de la zona (pedir factibilidad)" if ficha.get("luz") in ("si", True)
               else "por definir con la distribuidora de la zona"),
              ("Aguas lluvias", "cunetas en la calle hacia la parte baja"
               + (f" (el terreno baja hacia el {rel['baja_hacia']})" if rel else "") + "; la descarga se define en el "
               "estudio hidráulico y necesita permiso ambiental"),
              ("Aguas negras", "colector de ANDA si hay factibilidad; si no, fosa séptica por lote o planta de tratamiento")]
    td = Table([[P(a, "cb"), P(b, "c")] for a, b in datos], colWidths=[W * 0.24, W * 0.76])
    td.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.3, LINEA), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(td)

    # ------------------------------------------------------------------ relieve
    if rel:
        story.append(P(f"{num}. Relieve (altimetría)", "h"))
        num += 1
        inclinados = sum(1 for l in lot.lotes if (l.get("relieve") or {}).get("pend_media") is not None
                         and l["relieve"]["pend_media"] > altimetria.PEND_CONSTRUIBLE)
        txt = (f"El terreno va de la cota {rel['cota_max']:,.1f} m a la {rel['cota_min']:,.1f} m ({rel['desnivel']:.1f} m de "
               f"desnivel) y baja hacia el {rel['baja_hacia']}; la pendiente media es de {rel['pend_media']:.0f} %. "
               f"{inclinados} de {n} lotes pasan de 30 %.")
        barr = next((f for f in (cumplimiento or []) if f["regla"].startswith("Caídas fuertes")), None)
        if barr:
            txt += f" Las caídas fuertes (más de 50 %) no se lotificaron: {barr['proyecto']}."
        story.append(P(txt))
        story.append(Spacer(1, 3))
        filas = [["Pendiente", "Área (m²)", "% del terreno"]] + [[c["clase"], _n(c["m2"], 0), f"{c['pct']:.0f} %"]
                                                                  for c in rel["clases"]]
        tr = Table([[P(c, "cb" if i == 0 else "c") for c in f] for i, f in enumerate(filas)],
                   colWidths=[W * 0.4, W * 0.2, W * 0.2])
        sr = [("LINEBELOW", (0, 0), (-1, 0), 0.8, TINTA), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINEA),
              ("ALIGN", (1, 1), (-1, -1), "RIGHT")]
        for i, (_, _, _, col) in enumerate(altimetria.CLASES, start=1):
            sr.append(("BACKGROUND", (0, i), (0, i), colors.Color(*[c / 255 for c in col])))
        tr.setStyle(TableStyle(sr))
        story.append(tr)
        from .relieve import FUENTES

        story += [Spacer(1, 3), P(f"Fuente: {FUENTES.get(malla.fuente, malla.fuente)}. Sirve para anteproyecto; las caídas "
                                  "cortas se suavizan. El levantamiento topográfico con altimetría lo sustituye.", "pg")]

        # -------------------------------------------------------------- terracería: taludes o muros
        cmp_ = altimetria.comparar_terraceria(lot, malla) if lot.lotes else None
        if cmp_:
            story.append(P(f"{num}. Terracería: taludes o muros de contención", "h"))
            num += 1
            ta, mu = cmp_["taludes"], cmp_["muros"]
            gana = mu["area_plana"] - ta["area_plana"]
            alto = mu["lotes_muro_alto"]
            story.append(P(
                f"Dos maneras de acomodar las casas en la ladera. Con taludes, cada casa va sobre una plataforma a la cota "
                f"de la calle y el resto del lote se queda con su caída. Con muros de contención, cada lote se escalona en "
                f"terrazas que siguen el terreno (la de la casa pegada a la calle, a no más de 1.5 m de su nivel) y se "
                f"aprovecha casi todo el lote: {_n(gana, 0)} m² planos más, a cambio de unos {_n(mu['muro_m2'], 0)} m² de "
                f"cara de muro (≈ {_n(mu['muro_largo'], 0)} m de largo, hasta {mu['muro_max']:.1f} m de alto)."
                + (f" Los muros de más de {altimetria.MURO_ESTRUCTURAL:g} m (lote{'s' if len(alto) > 1 else ''} "
                   f"{', '.join(str(i) for i in alto)}) piden diseño estructural." if alto else "")))
            story.append(Spacer(1, 3))
            filas = [["", "Con taludes", "Con muros"],
                     ["Área plana para usar", f"{_n(ta['area_plana'], 0)} m²", f"{_n(mu['area_plana'], 0)} m²"],
                     ["Corte / relleno", f"{_n(ta['corte_m3'], 0)} / {_n(ta['relleno_m3'], 0)} m³",
                      f"{_n(mu['corte_m3'], 0)} / {_n(mu['relleno_m3'], 0)} m³"],
                     ["Terreno ocupado por taludes", f"≈ {_n(ta['talud_m2'], 0)} m²", "—"],
                     ["Muros de contención", "—", f"{_n(mu['muro_m2'], 0)} m² · máx. {mu['muro_max']:.1f} m"],
                     [f"Muro de más de {altimetria.MURO_ESTRUCTURAL:g} m", "—", f"{_n(mu['muro_estructural_m'], 0)} m"]]
            tt = Table([[P(c, "cb" if (i == 0 or j == 0) else "c") for j, c in enumerate(f)] for i, f in enumerate(filas)],
                       colWidths=[W * 0.36, W * 0.27, W * 0.27])
            tt.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, TINTA), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINEA),
                                    ("ALIGN", (1, 1), (-1, -1), "RIGHT")]))
            story.append(tt)
            story += [Spacer(1, 3), P("Volúmenes y muros aproximados con el relieve satelital; el corte transversal de la "
                                      "carpeta técnica muestra las terrazas y los muros. El costo depende del tipo de muro "
                                      "(piedra, gaviones o concreto): se cotiza con el diseño final.", "pg")]

    # ------------------------------------------------------------------ cumplimiento
    if cumplimiento:
        story.append(P(f"{num}. Cumplimiento de normas (revisión preliminar)", "h"))
        num += 1
        filas = [["Regla", "Base", "Pide", "Proyecto", "Estado"]]
        for f in cumplimiento:
            filas.append([f["regla"], f.get("base", ""), f.get("requerido", ""), f.get("proyecto", ""),
                          ESTADOS.get(f["estado"], (f["estado"], GRIS))[0]])
        tc = Table([[P(c, "cb" if i == 0 else "c") for c in f] for i, f in enumerate(filas)],
                   colWidths=[W * 0.25, W * 0.2, W * 0.2, W * 0.22, W * 0.13], repeatRows=1)
        sc = [("LINEBELOW", (0, 0), (-1, 0), 0.8, TINTA), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINEA),
              ("VALIGN", (0, 0), (-1, -1), "TOP")]
        for i, f in enumerate(cumplimiento, start=1):
            sc.append(("TEXTCOLOR", (4, i), (4, i), ESTADOS.get(f["estado"], ("", GRIS))[1]))
        tc.setStyle(TableStyle(sc))
        story.append(tc)

    # ------------------------------------------------------------------ esquema legal y siguientes pasos
    story.append(P(f"{num}. Esquema legal (opciones a validar con notario)", "h"))
    num += 1
    legal = []
    if partes and len(partes) > 1:
        dentro = st.get("calle_dentro_de")
        legal.append("Reunir los terrenos en una sola matrícula y después desmembrar los lotes, la calle y las áreas "
                     "verdes. Es lo más limpio para vender lotes de una sola lotificación.")
        legal.append("Mantener las matrículas separadas y desmembrar los lotes de cada una"
                     + (f"; como la calle va completa dentro del terreno {dentro}, los lotes del otro terreno necesitan "
                        "servidumbre de tránsito sobre ella (o que la calle se done como uso público)." if dentro else "."))
    else:
        legal.append("Desmembrar los lotes, la calle y las áreas verdes de la matrícula del terreno.")
    legal.append("En cualquier caso se tramita como lotificación (permiso de parcelación) antes de vender; vender porciones "
                 "en proindiviso para formar lotes no es una salida válida.")
    for x in legal:
        story.append(P(f"• {x}"))
    story.append(P(f"{num}. Siguientes pasos", "h"))
    pasos = []
    if avisos:
        pasos.append("Pedir al topógrafo que revise el cuadro de rumbos que no cuadra (arriba, en la verificación).")
    hay_quebrada = bool(ficha.get("quebradas"))
    pasos += ["Levantamiento topográfico con altimetría (curvas cada 1 m) para sustituir el relieve satelital y marcar "
              "el borde real " + ("de la quebrada y " if hay_quebrada else "") + "de las caídas fuertes."]
    if hay_quebrada:
        pasos.append("Medir la profundidad real de la quebrada: de ella sale el ancho de la franja de protección.")
    pasos += ["Factibilidades de agua potable, aguas negras y energía eléctrica.",
              "Calificación de lugar, línea de construcción y permiso de parcelación en la oficina que corresponda.",
              "Definir con el notario la figura legal (arriba) y el reglamento de la lotificación."]
    for i, x in enumerate(pasos, 1):
        story.append(P(f"{i}. {x}"))

    # ------------------------------------------------------------------ qué trae la entrega
    if entrega:
        story.append(P("Qué trae esta entrega", "h"))
        for carpeta, items in entrega.items():
            story.append(P(f"<b>{carpeta}:</b> " + "; ".join(items) + "."))

    story += [Spacer(1, 10), P("Anteproyecto referencial generado con las medidas del prediseño. No válido para trámite: "
                               "requiere levantamiento topográfico y firma de profesional autorizado.", "pg")]

    def pie(c, doc):
        c.saveState()
        c.setFont("DJ", 7.5)
        c.setFillColor(GRIS)
        c.drawString(18 * mm, 10 * mm, f"{proyecto or 'Anteproyecto'} · Resumen ejecutivo")
        c.drawRightString(letter[0] - 18 * mm, 10 * mm, f"Página {doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(ruta, pagesize=letter, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title=f"Resumen ejecutivo — {proyecto}", author="Lotes-Bot")
    doc.build(story, onFirstPage=pie, onLaterPages=pie)
    return {"ruta": ruta, "avisos": avisos, "paginas": doc.page}
