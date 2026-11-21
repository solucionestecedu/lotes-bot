"""Archivos para abrir en otros programas: KML (Google Earth) y DXF (AutoCAD / Civil 3D)."""
from __future__ import annotations

from xml.sax.saxutils import escape

import ezdxf
from shapely.geometry import Polygon

from . import proyeccion
from .lotificacion import Resultado, anillos
from .terreno import Terreno

# KML usa colores aabbggrr
_EST = {
    "terreno": ("ff00ffff", "00000000", 3),
    "lotes": ("ffffffff", "5500a5ff", 1),
    "calles": ("ff808080", "99808080", 1),
    "verde": ("ff00aa00", "8800cc00", 1),
    "curvas": ("ff3d6b8b", "00000000", 1),
    "curvas_indice": ("ff1f3f5f", "00000000", 2),
}


def _ll(pts):
    return " ".join(f"{lng:.8f},{lat:.8f},0" for lat, lng in (proyeccion.a_latlng(x, y) for x, y in pts))


def _poly_kml(nombre, poly: Polygon, estilo, desc=""):
    rs = anillos(poly)
    out = [f"<Placemark><name>{escape(nombre)}</name>"]
    if desc:
        out.append(f"<description>{escape(desc)}</description>")
    out.append(f"<styleUrl>#{estilo}</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>{_ll(rs[0] + rs[0][:1])}</coordinates></LinearRing></outerBoundaryIs>")
    for h in rs[1:]:
        out.append(f"<innerBoundaryIs><LinearRing><coordinates>{_ll(h + h[:1])}</coordinates></LinearRing></innerBoundaryIs>")
    out.append("</Polygon></Placemark>")
    return "".join(out)


def kml(nombre: str, t: Terreno, curvas: list[dict] | None = None, lot: Resultado | None = None) -> str:
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           f"<name>{escape(nombre)}</name>"]
    for k, (linea, relleno, ancho) in _EST.items():
        out.append(f'<Style id="{k}"><LineStyle><color>{linea}</color><width>{ancho}</width></LineStyle>'
                   f"<PolyStyle><color>{relleno}</color></PolyStyle></Style>")
    out.append("<Folder><name>Terreno</name>")
    out.append(_poly_kml("Lindero", Polygon(t.puntos), "terreno", f"Área {t.area:,.2f} m²"))
    for n, p in zip(t.nombres, t.puntos):
        out.append(f"<Placemark><name>{escape(n)}</name><Point><coordinates>{_ll([p])}</coordinates></Point></Placemark>")
    out.append("</Folder>")
    if curvas:
        out.append("<Folder><name>Curvas de nivel (satelitales, aproximadas)</name>")
        for c in curvas:
            for ln in c["lineas"]:
                out.append(f"<Placemark><name>{c['cota']:g} m</name><styleUrl>#{'curvas_indice' if c['indice'] else 'curvas'}</styleUrl>"
                           f"<LineString><tessellate>1</tessellate><coordinates>{_ll(ln)}</coordinates></LineString></Placemark>")
        out.append("</Folder>")
    if lot:
        out.append("<Folder><name>Lotes</name>")
        for l in lot.lotes:
            out.append(_poly_kml(f"Lote {l['id']}", l["poly"], "lotes", f"{l['area']:,.1f} m²"))
        out.append("</Folder><Folder><name>Calles</name>")
        for i, c in enumerate(lot.calles, 1):
            out.append(_poly_kml(f"Calle {i}", c, "calles"))
        out.append("</Folder><Folder><name>Área verde</name>")
        for i, v in enumerate(lot.verdes, 1):
            out.append(_poly_kml(f"Área verde {i}", v["poly"], "verde", f"{v['area']:,.1f} m²"))
        out.append("</Folder>")
    out.append("</Document></kml>\n")
    return "".join(out)


def dxf(ruta: str, t: Terreno, curvas: list[dict] | None = None, lot: Resultado | None = None, alto_texto: float = 1.5) -> str:
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6  # metros
    capas = {"LINDERO": 2, "VERTICES": 7, "CURVAS_INDICE": 30, "CURVAS_INTERMEDIAS": 33, "COTAS": 30,
             "LOTES": 7, "LOTES_TXT": 7, "CALLES": 8, "AREA_VERDE": 3}
    for n, c in capas.items():
        doc.layers.add(n, color=c)
    msp = doc.modelspace()
    msp.add_lwpolyline(t.puntos, close=True, dxfattribs={"layer": "LINDERO"})
    for n, (x, y) in zip(t.nombres, t.puntos):
        msp.add_point((x, y), dxfattribs={"layer": "VERTICES"})
        msp.add_text(n, height=alto_texto, dxfattribs={"layer": "VERTICES"}).set_placement((x + 0.8, y + 0.8))
    for c in curvas or []:
        capa = "CURVAS_INDICE" if c["indice"] else "CURVAS_INTERMEDIAS"
        for ln in c["lineas"]:
            msp.add_lwpolyline(ln, dxfattribs={"layer": capa, "elevation": float(c["cota"])})
            if c["indice"] and len(ln) > 2:
                x, y = ln[len(ln) // 2]
                msp.add_text(f"{c['cota']:g}", height=alto_texto, dxfattribs={"layer": "COTAS"}).set_placement((x, y))
    if lot:
        for l in lot.lotes:
            for r in anillos(l["poly"]):
                msp.add_lwpolyline(r, close=True, dxfattribs={"layer": "LOTES"})
            c = l["poly"].representative_point()
            msp.add_text(f"L{l['id']} {l['area']:.1f} m²", height=alto_texto,
                         dxfattribs={"layer": "LOTES_TXT"}).set_placement((c.x, c.y), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
        for p in lot.calles:
            for r in anillos(p):
                msp.add_lwpolyline(r, close=True, dxfattribs={"layer": "CALLES"})
        for v in lot.verdes:
            for r in anillos(v["poly"]):
                msp.add_lwpolyline(r, close=True, dxfattribs={"layer": "AREA_VERDE"})
    doc.saveas(ruta)
    return ruta
