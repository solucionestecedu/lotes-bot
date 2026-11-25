"""Carpeta de prefactibilidad: un solo PDF con todo lo que entrega el bot, para abrirlo sin AutoCAD.

Orden: portada (con la imagen de ambiente y las cifras clave) y contenido, resumen ejecutivo, carpeta técnica (plano
técnico, altimétrico, corte y terracería, y la lámina a escala en su tamaño tabloide), carpeta comercial (vistas) y
anexos (los archivos de CAD que van aparte). Con marcadores para saltar a cada parte.
"""
from __future__ import annotations

import datetime
import io
import logging
import os

from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .resumen import ACENTO, GRIS, MESES, TINTA, _fuentes

log = logging.getLogger(__name__)
PAG = landscape(letter)


def _img_reader(ruta: str, lado: int = 2200) -> ImageReader:
    im = Image.open(ruta).convert("RGB")
    if max(im.size) > lado:
        k = lado / max(im.size)
        im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86)
    buf.seek(0)
    return ImageReader(buf)


def _pagina_imagen(c: canvas.Canvas, ruta: str, seccion: str, titulo: str, proyecto: str):
    W, H = PAG
    m = 22
    c.setFillColor(ACENTO)
    c.setFont("DJB", 8.5)
    c.drawString(m, H - m - 4, seccion.upper())
    c.setFillColor(TINTA)
    c.setFont("DJB", 12)
    c.drawString(m, H - m - 19, titulo)
    c.setStrokeColor(colors.HexColor("#c9c9c9"))
    c.line(m, H - m - 26, W - m, H - m - 26)
    ir = _img_reader(ruta)
    iw, ih = ir.getSize()
    caja_w, caja_h = W - 2 * m, H - 2 * m - 46
    k = min(caja_w / iw, caja_h / ih)
    w, h = iw * k, ih * k
    c.drawImage(ir, (W - w) / 2, m + 14 + (caja_h - h) / 2, w, h)
    c.setFont("DJ", 7.5)
    c.setFillColor(GRIS)
    c.drawString(m, m - 6, f"{proyecto} · Carpeta de prefactibilidad")
    c.showPage()


def generar_carpeta(ruta: str, proyecto: str, ubicacion: str, cifras: list[tuple[str, str]], portada: str | None,
                    resumen_pdf: str | None, tecnicas: list[tuple[str, str]], lamina_pdf: str | None,
                    comerciales: list[tuple[str, str]], anexos: list[tuple[str, str]], avisos: list[str] | None = None) -> dict:
    """tecnicas y comerciales: [(ruta de imagen, título)]; anexos: [(nombre de archivo, para qué sirve)]."""
    _fuentes()
    W, H = PAG
    hoy = datetime.date.today()
    n_res = len(PdfReader(resumen_pdf).pages) if resumen_pdf and os.path.exists(resumen_pdf) else 0
    n_lam = len(PdfReader(lamina_pdf).pages) if lamina_pdf and os.path.exists(lamina_pdf) else 0
    # páginas de cada parte (portada y contenido = 2)
    secciones = []
    p = 3
    if n_res:
        secciones.append(("Resumen ejecutivo", p))
        p += n_res
    if tecnicas or n_lam:
        secciones.append(("Carpeta técnica", p))
        p += len(tecnicas) + n_lam
    if comerciales:
        secciones.append(("Carpeta comercial", p))
        p += len(comerciales)
    secciones.append(("Anexos: archivos para AutoCAD y Google Earth", p))

    # ------------------------------------------------------------------ portada y contenido
    buf_a = io.BytesIO()
    c = canvas.Canvas(buf_a, pagesize=PAG)
    if portada:
        ir = _img_reader(portada)
        iw, ih = ir.getSize()
        k = max(W / iw, (H * 0.62) / ih)
        c.drawImage(ir, (W - iw * k) / 2, H - ih * k, iw * k, ih * k)
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H * 0.38, stroke=0, fill=1)
    c.setFillColor(ACENTO)
    c.setFont("DJB", 10)
    c.drawString(40, H * 0.38 - 30, "CARPETA DE PREFACTIBILIDAD · ANTEPROYECTO DE LOTIFICACIÓN")
    c.setFillColor(TINTA)
    c.setFont("DJB", 30)
    c.drawString(40, H * 0.38 - 66, proyecto or "Anteproyecto")
    c.setFont("DJ", 11)
    c.setFillColor(GRIS)
    c.drawString(40, H * 0.38 - 86, " · ".join(x for x in (ubicacion, f"{MESES[hoy.month - 1]} de {hoy.year}") if x))
    x = 40
    for valor, texto in cifras[:5]:
        c.setFillColor(TINTA)
        c.setFont("DJB", 17)
        c.drawString(x, 62, valor)
        c.setFont("DJ", 8)
        c.setFillColor(GRIS)
        c.drawString(x, 48, texto[:34])
        x += (W - 80) / min(5, max(1, len(cifras)))
    c.setFont("DJ", 7.5)
    c.drawString(40, 22, "Anteproyecto referencial. No válido para trámite: requiere levantamiento topográfico y firma de "
                         "profesional autorizado. Las imágenes de ambiente son ilustrativas.")
    c.showPage()
    # contenido
    c.setFillColor(ACENTO)
    c.setFont("DJB", 10)
    c.drawString(60, H - 70, "CONTENIDO")
    y = H - 110
    detalle = {
        "Resumen ejecutivo": "Áreas por terreno, verificación de los planos, lotes, calle, relieve, normas y siguientes pasos",
        "Carpeta técnica": "Plano técnico, plano altimétrico, corte y terracería, lámina a escala (tabloide)",
        "Carpeta comercial": "Vistas de ambiente, desde el portón, desde la quebrada, lotes listos y plano de presentación",
        "Anexos: archivos para AutoCAD y Google Earth": "Qué archivo abrir con qué programa",
    }
    for nombre, pag in secciones:
        c.setFillColor(TINTA)
        c.setFont("DJB", 15)
        c.drawString(60, y, nombre)
        c.drawRightString(W - 60, y, str(pag))
        c.setFont("DJ", 9.5)
        c.setFillColor(GRIS)
        c.drawString(60, y - 16, detalle.get(nombre, ""))
        y -= 52
    if avisos:
        y -= 10
        c.setFillColor(colors.HexColor("#a3372a"))
        c.setFont("DJB", 10)
        c.drawString(60, y, "Atención")
        c.setFont("DJ", 9.5)
        for a in avisos:
            y -= 15
            c.drawString(60, y, a[:150])
    c.showPage()
    c.save()

    # ------------------------------------------------------------------ páginas de imágenes y anexos
    buf_t = io.BytesIO()
    if tecnicas:
        c = canvas.Canvas(buf_t, pagesize=PAG)
        for r, t in tecnicas:
            _pagina_imagen(c, r, "Carpeta técnica", t, proyecto)
        c.save()
    buf_c = io.BytesIO()
    c = canvas.Canvas(buf_c, pagesize=PAG)
    for r, t in comerciales:
        _pagina_imagen(c, r, "Carpeta comercial", t, proyecto)
    # anexos
    c.setFillColor(ACENTO)
    c.setFont("DJB", 10)
    c.drawString(60, H - 70, "ANEXOS · ARCHIVOS QUE VAN APARTE")
    y = H - 110
    for nombre, uso in anexos:
        c.setFillColor(TINTA)
        c.setFont("DJB", 12)
        c.drawString(60, y, nombre)
        c.setFont("DJ", 9.5)
        c.setFillColor(GRIS)
        c.drawString(60, y - 15, uso[:140])
        y -= 44
    c.showPage()
    c.save()

    # ------------------------------------------------------------------ unir con marcadores
    w = PdfWriter()
    marcas = []

    def agregar(buf_o_ruta, marca=None):
        lector = PdfReader(buf_o_ruta)
        if marca and len(lector.pages):
            marcas.append((marca, len(w.pages)))
        for pg in lector.pages:
            w.add_page(pg)

    buf_a.seek(0)
    agregar(buf_a, "Portada y contenido")
    if n_res:
        agregar(resumen_pdf, "Resumen ejecutivo")
    if tecnicas:
        buf_t.seek(0)
        agregar(buf_t, "Carpeta técnica")
    if n_lam:
        agregar(lamina_pdf, "Lámina a escala (tabloide)")
    buf_c.seek(0)
    agregar(buf_c, "Carpeta comercial" if comerciales else "Anexos")
    for titulo, i in marcas:
        w.add_outline_item(titulo, i)
    w.add_metadata({"/Title": f"Carpeta de prefactibilidad — {proyecto}", "/Author": "Lotes-Bot"})
    with open(ruta, "wb") as f:
        w.write(f)
    return {"ruta": ruta, "paginas": len(w.pages)}
