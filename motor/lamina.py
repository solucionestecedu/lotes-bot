"""Láminas en PDF tamaño tabloide (17×11 in), dibujadas a escala real para imprimir al 100 %.

Hoja 1: plano planialtimétrico (lindero, vértices, curvas, cuadro de construcción, datos, membrete).
Hoja 2: anteproyecto de lotificación (lotes, calles, área verde, cuadro de áreas y de lotes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import PathPatch, Polygon as MplPolygon, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from .lotificacion import Resultado  # noqa: E402
from .proyeccion import NOMBRE_SISTEMA, a_latlng  # noqa: E402
from .relieve import FUENTES, Malla, curvas as calcular_curvas, estadisticas, intervalo_curvas  # noqa: E402
from .rumbos import M2_POR_V2  # noqa: E402
from .terreno import Terreno  # noqa: E402

PT = 72 / 25.4  # puntos tipográficos por mm
HOJA_W, HOJA_H = 431.8, 279.4  # tabloide horizontal (mm)
MARCO = (15.0, 8.0, 423.8, 271.4)  # x0, y0, x1, y1
PANEL_W = 140.0
PX0 = MARCO[2] - PANEL_W            # inicio del panel derecho
VISTA = (27.0, 32.0, PX0 - 10.0, MARCO[3] - 10.0)  # zona de dibujo (mm)
ESCALAS = [100, 200, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 25000, 50000]
TINTA = "#1a1a1a"
GRIS = "#8a8a8a"
COLOR_CURVA = "#a0703c"
COLOR_LOTE = "#f6e3c4"
COLOR_VERDE = "#b9dfa8"
COLOR_CALLE = "#d9d9d9"
COLOR_PROTECCION = "#8fbf7a"
COLOR_EQUIP = "#bcd3ea"
COLOR_ESTADO = {"cumple": "#1e7b34", "no_cumple": "#c0392b", "revisar": "#c77700", "falta_dato": "#666666"}
TEXTO_ESTADO = {"cumple": "✔ Cumple", "no_cumple": "✖ No cumple", "revisar": "! Revisar", "falta_dato": "? Falta dato"}
AVISO = "ANTEPROYECTO PRELIMINAR — NO VÁLIDO PARA TRÁMITE"


@dataclass
class Datos:
    proyecto: str = "Anteproyecto de lotificación"
    ubicacion: str = ""
    propietario: str = ""
    elaboro: str = "Generado automáticamente (Lotes-Bot)"
    fecha: str = field(default_factory=lambda: date.today().strftime("%d/%m/%Y"))


# ---------------------------------------------------------------------------- utilidades
def _escala(ancho_m: float, alto_m: float, vista_w: float, vista_h: float) -> int:
    for e in ESCALAS:
        if ancho_m / e * 1000 <= vista_w and alto_m / e * 1000 <= vista_h:
            return e
    return ESCALAS[-1]


def _paso_lindo(rango: float, objetivo: int) -> float:
    for p in (1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000):
        if rango / p <= objetivo:
            return p
    return 10000


def _t(ax, x, y, s, size=6.0, **kw):
    kw.setdefault("color", TINTA)
    kw.setdefault("va", "center")
    kw.setdefault("ha", "left")
    return ax.text(x, y, s, fontsize=size, **kw)


def _caja(ax, x0, y0, x1, y1, lw=0.6, **kw):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=kw.pop("fill", False), lw=lw, ec=kw.pop("ec", TINTA), **kw))


def _patch(poly: Polygon, **kw):
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        cs = list(ring.coords)
        verts += cs
        codes += [Path.MOVETO] + [Path.LINETO] * (len(cs) - 2) + [Path.CLOSEPOLY]
    return PathPatch(Path(verts, codes), **kw)


def _fmt(v, d=2):
    return f"{v:,.{d}f}"


class _Hoja:
    """Una hoja con marco, membrete y una vista a escala del terreno."""

    def __init__(self, pdf: PdfPages, centro, escala: int, datos: Datos, contenido: str, hoja: str):
        self.fig = plt.figure(figsize=(HOJA_W / 25.4, HOJA_H / 25.4))
        self.pdf = pdf
        self.e = escala
        self.p = self.fig.add_axes([0, 0, 1, 1])
        self.p.set_xlim(0, HOJA_W)
        self.p.set_ylim(0, HOJA_H)
        self.p.axis("off")
        vx0, vy0, vx1, vy1 = VISTA
        self.vista = VISTA
        w_m, h_m = (vx1 - vx0) * escala / 1000, (vy1 - vy0) * escala / 1000
        self.xmin, self.ymin = centro[0] - w_m / 2, centro[1] - h_m / 2
        self.xmax, self.ymax = self.xmin + w_m, self.ymin + h_m
        self.ax = self.fig.add_axes([vx0 / HOJA_W, vy0 / HOJA_H, (vx1 - vx0) / HOJA_W, (vy1 - vy0) / HOJA_H])
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.axis("off")
        self.m = escala / 1000  # metros de terreno por mm de papel
        self._marco()
        self._cuadricula()
        self._norte()
        self._barra()
        self._membrete(datos, contenido, hoja)

    def mm(self, x, y):
        vx0, vy0, _, _ = self.vista
        return vx0 + (x - self.xmin) / self.m, vy0 + (y - self.ymin) / self.m

    def _marco(self):
        x0, y0, x1, y1 = MARCO
        _caja(self.p, x0, y0, x1, y1, lw=1.2)
        _caja(self.p, x0 - 2.5, y0 - 2.5, x1 + 2.5, y1 + 2.5, lw=0.3)
        self.p.plot([PX0, PX0], [y0, y1], color=TINTA, lw=1.0)
        _caja(self.p, *self.vista, lw=0.4, ec=GRIS)

    def _cuadricula(self):
        paso = _paso_lindo(self.xmax - self.xmin, 6)
        vx0, vy0, vx1, vy1 = self.vista
        x = math.ceil(self.xmin / paso) * paso
        while x < self.xmax:
            self.ax.axvline(x, color="#c8c8c8", lw=0.25, zorder=0)
            mx, _ = self.mm(x, self.ymin)
            _t(self.p, mx, vy0 - 2.2, f"E {x:,.0f}", 4.6, ha="center", color="#555")
            _t(self.p, mx, vy1 + 2.2, f"E {x:,.0f}", 4.6, ha="center", color="#555")
            x += paso
        y = math.ceil(self.ymin / paso) * paso
        while y < self.ymax:
            self.ax.axhline(y, color="#c8c8c8", lw=0.25, zorder=0)
            _, my = self.mm(self.xmin, y)
            _t(self.p, vx0 - 2.2, my, f"N {y:,.0f}", 4.6, ha="center", rotation=90, color="#555")
            _t(self.p, vx1 + 2.2, my, f"N {y:,.0f}", 4.6, ha="center", rotation=90, color="#555")
            y += paso

    def _norte(self):
        _, _, vx1, vy1 = self.vista
        cx, cy = vx1 - 12, vy1 - 22
        self.p.add_patch(MplPolygon([(cx, cy + 12), (cx - 4, cy - 6), (cx, cy - 2)], closed=True, fc=TINTA, ec=TINTA, lw=0.5))
        self.p.add_patch(MplPolygon([(cx, cy + 12), (cx + 4, cy - 6), (cx, cy - 2)], closed=True, fc="white", ec=TINTA, lw=0.5))
        _t(self.p, cx, cy + 15.5, "N", 11, ha="center", weight="bold")

    def _barra(self):
        vx0, vy0, _, _ = self.vista
        L = _paso_lindo(80 * self.m, 1)  # longitud total ~≤ 80 mm
        largo = L / self.m
        y = vy0 - 14
        for k in range(5):
            _caja(self.p, vx0 + k * largo / 5, y, vx0 + (k + 1) * largo / 5, y + 2,
                  lw=0.5, fill=True, fc=TINTA if k % 2 == 0 else "white")
        for k in (0, 2.5, 5):
            _t(self.p, vx0 + k * largo / 5, y + 4.2, f"{L * k / 5:g}", 5.5, ha="center")
        _t(self.p, vx0 + largo + 3, y + 1, "metros", 5.5)
        _t(self.p, vx0, y - 4.5, f"ESCALA GRÁFICA      ESCALA 1:{self.e:,}  (imprimir en tabloide 17×11 al 100 %)", 6)

    def _membrete(self, d: Datos, contenido: str, hoja: str):
        x0, y0 = PX0, MARCO[1]
        x1 = MARCO[2]
        y1 = y0 + 68
        _caja(self.p, x0, y0, x1, y1, lw=1.0)
        ys = [y1 - 16, y1 - 26, y1 - 36, y1 - 46]  # líneas horizontales
        for y in ys:
            self.p.plot([x0, x1 if y > ys[0] - 1 else x0 + 88], [y, y], color=TINTA, lw=0.5)
        self.p.plot([x0, x1], [ys[0], ys[0]], color=TINTA, lw=0.5)
        self.p.plot([x0 + 88, x0 + 88], [y0, ys[0]], color=TINTA, lw=0.5)
        _t(self.p, x0 + 2, y1 - 3.2, "PROYECTO", 4.5, color="#555")
        _t(self.p, x0 + 2, y1 - 10, d.proyecto.upper()[:48], 9.5, weight="bold")
        filas = [("UBICACIÓN", d.ubicacion or "—"), ("PROPIETARIO", d.propietario or ""), ("CONTENIDO", contenido)]
        for (k, v), ytop in zip(filas, ys):
            _t(self.p, x0 + 2, ytop - 2.6, k, 4.5, color="#555")
            _t(self.p, x0 + 2, ytop - 6.8, v[:60], 6.4, weight="bold" if k == "CONTENIDO" else "normal")
        # fila inferior: escala, fecha, hoja
        celdas = [("ESCALA", f"1:{self.e:,}"), ("FECHA", d.fecha), ("HOJA", hoja)]
        w = 88 / 3
        for k, (lab, val) in enumerate(celdas):
            cx = x0 + k * w
            if k:
                self.p.plot([cx, cx], [y0 + 10, ys[3]], color=TINTA, lw=0.5)
            _t(self.p, cx + 2, ys[3] - 2.6, lab, 4.5, color="#555")
            _t(self.p, cx + 2, ys[3] - 9, val, 7.5, weight="bold")
        self.p.plot([x0, x0 + 88], [y0 + 10, y0 + 10], color=TINTA, lw=0.5)
        _t(self.p, x0 + 2, y0 + 5, f"ELABORÓ: {d.elaboro}"[:62], 4.8)
        _t(self.p, x0 + 88 + 26, ys[0] - 3, "SELLO Y FIRMA DEL\nPROFESIONAL RESPONSABLE", 4.6, ha="center", va="top", color="#555", linespacing=1.2)
        # aviso
        ay = y1 + 2
        _caja(self.p, x0 + 3, ay, x1 - 3, ay + 7, lw=0.9, ec="#c0392b")
        _t(self.p, (x0 + x1) / 2, ay + 3.5, AVISO, 6.6, ha="center", color="#c0392b", weight="bold")
        self.tope_membrete = ay + 7

    # -------------------------------------------------------------- dibujo en la vista
    def lindero(self, t: Terreno, lw=1.4):
        pts = t.puntos + t.puntos[:1]
        self.ax.plot([p[0] for p in pts], [p[1] for p in pts], color=TINTA, lw=lw, zorder=5, solid_joinstyle="miter")
        cx, cy = t.centroide()
        puestos = []
        for n, (x, y) in zip(t.nombres, t.puntos):
            self.ax.plot([x], [y], marker="o", ms=2.6, mfc="white", mec=TINTA, mew=0.6, zorder=6)
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy) or 1
            off = 3.2 * self.m
            px, py = x + dx / d * off, y + dy / d * off
            for _ in range(4):  # evita que dos etiquetas se encimen
                if all(math.hypot(px - qx, py - qy) > 6.5 * self.m for qx, qy in puestos):
                    break
                px, py = px + dy / d * 2.4 * self.m, py - dx / d * 2.4 * self.m
            puestos.append((px, py))
            _t(self.ax, px, py, n, 5.4, ha="center", weight="bold", zorder=7,
               bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))

    def curvas(self, curvas: list[dict], solo_indice=False, alpha=1.0):
        for c in curvas:
            if solo_indice and not c["indice"]:
                continue
            for ln in c["lineas"]:
                xs, ys = [p[0] for p in ln], [p[1] for p in ln]
                self.ax.plot(xs, ys, color=COLOR_CURVA, lw=0.55 if c["indice"] else 0.25, alpha=alpha, zorder=2)
                if c["indice"]:
                    self._etiqueta_curva(ln, c["cota"], alpha)

    def _etiqueta_curva(self, ln, cota, alpha):
        largo = [0.0]
        for (x1, y1), (x2, y2) in zip(ln, ln[1:]):
            largo.append(largo[-1] + math.hypot(x2 - x1, y2 - y1))
        if largo[-1] < 11 * self.m:
            return
        mitad = largo[-1] / 2
        k = next(i for i in range(1, len(largo)) if largo[i] >= mitad)
        (x1, y1), (x2, y2) = ln[k - 1], ln[k]
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        _t(self.ax, (x1 + x2) / 2, (y1 + y2) / 2, f"{cota:g}", 4.2, ha="center", rotation=ang, color=COLOR_CURVA,
           alpha=alpha, zorder=3, rotation_mode="anchor", bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))

    def recuadro(self, x0, y0, x1, y1):
        """Ejes en blanco encima de la cuadrícula (para perfil o simbología), en mm de papel."""
        a = self.fig.add_axes([x0 / HOJA_W, y0 / HOJA_H, (x1 - x0) / HOJA_W, (y1 - y0) / HOJA_H])
        a.set_facecolor("white")
        return a

    def perfil(self, linea, malla: Malla, caja):
        """Perfil del terreno a lo largo de `linea` (LineString) dibujado en `caja` (mm)."""
        x0, y0, x1, y1 = caja
        (ax_, ay_), (bx_, by_) = linea.coords[0], linea.coords[-1]
        # marca A–A' en planta
        self.ax.plot([ax_, bx_], [ay_, by_], color="#c0392b", lw=0.6, ls=(0, (6, 3)), zorder=6)
        for (x, y), n in (((ax_, ay_), "A"), ((bx_, by_), "A'")):
            _t(self.ax, x, y, n, 7, ha="center", color="#c0392b", weight="bold", zorder=9,
               bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="#c0392b", lw=0.5))
        L = linea.length
        n = max(2, int(L / (malla.dx / 2)))
        ds = [L * k / n for k in range(n + 1)]
        zs = [malla.altura(*linea.interpolate(d).coords[0]) for d in ds]
        pares = [(d, z) for d, z in zip(ds, zs) if math.isfinite(z)]
        if len(pares) < 2:
            return
        _caja(self.p, x0, y0, x1, y1, lw=0.4, ec=GRIS, fill=True, fc="white", zorder=1)
        ancho_mm = L / self.m
        zmin, zmax = min(z for _, z in pares), max(z for _, z in pares)
        paso_z = _paso_lindo(max(zmax - zmin, 1), 6)
        z0 = math.floor(zmin / paso_z) * paso_z
        z1 = math.ceil(zmax / paso_z) * paso_z
        alto_mm = y1 - y0 - 16
        esc_v = next((e for e in ESCALAS if (z1 - z0) / e * 1000 <= alto_mm), ESCALAS[-1])
        alto_real = (z1 - z0) / esc_v * 1000
        gx0 = x0 + 14
        gy0 = y0 + 7
        a = self.recuadro(gx0, gy0, gx0 + min(ancho_mm, x1 - gx0 - 4), gy0 + alto_real)
        a.plot([d for d, _ in pares], [z for _, z in pares], color=TINTA, lw=0.9)
        a.fill_between([d for d, _ in pares], [z for _, z in pares], z0, color="#efe3d0", lw=0)
        a.set_xlim(0, min(ancho_mm, x1 - gx0 - 4) * self.m)
        a.set_ylim(z0, z0 + alto_real * esc_v / 1000)
        a.set_yticks([z0 + k * paso_z for k in range(int(round((z1 - z0) / paso_z)) + 1)])
        a.set_xticks([k * _paso_lindo(L, 8) for k in range(int(L / _paso_lindo(L, 8)) + 1)])
        a.grid(color="#dddddd", lw=0.3)
        a.tick_params(labelsize=4.6, length=1.5, width=0.3, pad=1)
        for sp in a.spines.values():
            sp.set_linewidth(0.4)
        _t(self.p, x0 + 2, y1 - 3, f"PERFIL A–A'   (H 1:{self.e:,}  ·  V 1:{esc_v:,})   distancia y cota en metros, cotas aproximadas",
           5.6, weight="bold")

    def simbologia(self, caja, items):
        x0, y0, x1, y1 = caja
        _caja(self.p, x0, y0, x1, y1, lw=0.4, ec=GRIS, fill=True, fc="white", zorder=1)
        a = self.recuadro(x0 + 0.3, y0 + 0.3, x1 - 0.3, y1 - 0.3)
        a.set_xlim(x0, x1)
        a.set_ylim(y0, y1)
        a.axis("off")
        _t(a, x0 + 3, y1 - 3.5, "SIMBOLOGÍA", 6.2, weight="bold")
        y = y1 - 9
        for nombre, estilo in items:
            if estilo.get("linea"):
                a.plot([x0 + 3, x0 + 11], [y, y], **estilo["linea"])
            else:
                a.add_patch(Rectangle((x0 + 3, y - 1.6), 8, 3.2, **estilo))
            _t(a, x0 + 14, y, nombre, 5.4)
            y -= 5

    def guardar(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def _tabla(ax, x0, y_top, columnas, filas, alto_fila, titulo=None, y_min=0.0, color_ultima=None):
    """columnas: [(titulo, ancho_mm, alineación)]. Dibuja hasta donde quepa; devuelve (y_final, filas_restantes)."""
    fs = max(4.0, min(6.2, alto_fila * 1.55))
    ancho = sum(c[1] for c in columnas)
    y = y_top
    if titulo:
        _t(ax, x0 + ancho / 2, y - 2.6, titulo, 6.6, ha="center", weight="bold")
        y -= 5
    _caja(ax, x0, y - alto_fila, x0 + ancho, y, lw=0.6, fill=True, fc="#eeeeee")
    cx = x0
    for nombre, w, _ in columnas:
        _t(ax, cx + w / 2, y - alto_fila / 2, nombre, fs * 0.92, ha="center", weight="bold")
        cx += w
    y -= alto_fila
    caben = max(0, int((y - y_min) / alto_fila))
    for fila in filas[:caben]:
        cx = x0
        for k, ((nombre, w, al), val) in enumerate(zip(columnas, fila)):
            xx = cx + (w / 2 if al == "center" else (w - 1.2 if al == "right" else 1.2))
            extra = {}
            if color_ultima and k == len(columnas) - 1:
                extra = {"color": color_ultima(fila), "weight": "bold"}
            _t(ax, xx, y - alto_fila / 2, str(val), fs, ha=al, **extra)
            cx += w
        ax.plot([x0, x0 + ancho], [y - alto_fila, y - alto_fila], color="#bbbbbb", lw=0.25)
        y -= alto_fila
    _caja(ax, x0, y, x0 + ancho, y_top - (5 if titulo else 0), lw=0.6)
    cx = x0
    for _, w, _ in columnas[:-1]:
        cx += w
        ax.plot([cx, cx], [y, y_top - (5 if titulo else 0)], color="#999999", lw=0.3)
    return y, filas[caben:]


def _parrafos(ax, x0, y_top, titulo, lineas, ancho, size=5.4, interlineado=3.2):
    _t(ax, x0, y_top - 2.5, titulo, 6.6, weight="bold")
    y = y_top - 7
    import textwrap

    for ln in lineas:
        for k, parte in enumerate(textwrap.wrap(ln, int(ancho / (size * 0.2)))):
            _t(ax, x0 + (0 if k == 0 else 3), y, parte, size)
            y -= interlineado
        y -= 0.6
    return y


def _eje_principal(poly: Polygon):
    """Línea por el centro del terreno, en la dirección de su lado más largo (para el perfil)."""
    from shapely import affinity
    from shapely.geometry import LineString

    r = list(poly.minimum_rotated_rectangle.exterior.coords)
    lados = [(r[i], r[i + 1]) for i in range(4)]
    (ax_, ay_), (bx_, by_) = max(lados, key=lambda l: math.dist(*l))
    if bx_ < ax_:
        ax_, ay_, bx_, by_ = bx_, by_, ax_, ay_
    c = poly.centroid
    L = math.dist((ax_, ay_), (bx_, by_)) * 2
    ux, uy = (bx_ - ax_) / (L / 2), (by_ - ay_) / (L / 2)
    linea = LineString([(c.x - ux * L, c.y - uy * L), (c.x + ux * L, c.y + uy * L)]).intersection(poly)
    if linea.geom_type != "LineString":
        linea = max(getattr(linea, "geoms", []), key=lambda g: g.length)
    return linea


# ---------------------------------------------------------------------------- láminas
def generar(ruta: str, t: Terreno, malla: Malla | None = None, lot: Resultado | None = None,
            datos: Datos | None = None, intervalo: float | None = None, cumplimiento: list | None = None) -> dict:
    datos = datos or Datos()
    poly = Polygon(t.puntos)
    minx, miny, maxx, maxy = poly.bounds
    vw, vh = VISTA[2] - VISTA[0] - 16, VISTA[3] - VISTA[1] - 16
    escala = _escala(maxx - minx, maxy - miny, vw, vh)
    centro = ((minx + maxx) / 2, (miny + maxy) / 2)
    m = escala / 1000
    # Si sobra espacio abajo, se sube la planta y se usa el hueco para el perfil / la simbología.
    libre = (VISTA[3] - VISTA[1]) - (maxy - miny) / m - 22
    caja_inf = None
    if libre >= 62:
        alto_inf = min(88.0, libre - 8)
        caja_inf = (VISTA[0] + 6, VISTA[1] + 6, VISTA[2] - 6, VISTA[1] + 6 + alto_inf)
        cy_obj = (caja_inf[3] + 4 + VISTA[3]) / 2
        centro = (centro[0], centro[1] - (cy_obj - (VISTA[1] + VISTA[3]) / 2) * m)

    st, cs = None, []
    if malla is not None:
        st = estadisticas(malla, t.puntos)
        intervalo = intervalo or intervalo_curvas(st["desnivel"])
        cs = calcular_curvas(malla, intervalo, recorte=poly)

    n_hojas = 1 + (1 if lot else 0)
    filas_cuadro = [(f["est"], f["pv"], f["rumbo"], _fmt(f["dist"]), f["v"], _fmt(f["x"], 3), _fmt(f["y"], 3)) for f in t.cuadro()]
    col_cuadro = [("EST", 11, "center"), ("PV", 11, "center"), ("RUMBO", 33, "center"), ("DIST. (m)", 17, "right"),
                  ("V", 11, "center"), ("X (Este)", 26.5, "right"), ("Y (Norte)", 26.5, "right")]
    hojas_extra = []
    with PdfPages(ruta) as pdf:
        # ---------------- Hoja 1: planialtimétrico
        h = _Hoja(pdf, centro, escala, datos, "PLANO PLANIALTIMÉTRICO" if cs else "PLANO DE POLIGONAL", f"1 / {n_hojas}")
        if cs:
            h.curvas(cs)
        h.lindero(t)
        if cs and caja_inf:
            h.perfil(_eje_principal(poly), malla, caja_inf)
        x0 = PX0 + 3
        y = MARCO[3] - 3
        alto = min(4.4, max(3.0, 112 / (len(filas_cuadro) + 1)))
        y_min_cuadro = MARCO[3] - 3 - 128
        y, resto = _tabla(h.p, x0, y, col_cuadro, filas_cuadro, alto, "CUADRO DE CONSTRUCCIÓN", y_min=y_min_cuadro)
        if resto:
            hojas_extra.append((col_cuadro, resto))
            _t(h.p, x0, y - 3, f"Continúa en hoja anexa ({len(resto)} filas).", 5.2, color="#c0392b")
            y -= 4
        a = t.area
        datos_ter = [
            f"Área: {_fmt(a)} m²  =  {_fmt(a / M2_POR_V2)} v²  =  {a / 10000:,.4f} ha",
            f"Perímetro: {_fmt(t.perimetro)} m",
            "Error de cierre: " + (f"{t.cierre:.3f} m  (precisión 1:{t.precision:,.0f})" if t.cierre > 1e-9 else "0.000 m (polígono por coordenadas)"),
        ]
        if t.area_declarada:
            datos_ter.append(f"Área según plano: {_fmt(t.area_declarada)} m²  (diferencia {(a - t.area_declarada) / t.area_declarada * 100:+.2f} %)")
        if st:
            datos_ter.append(f"Cotas aprox.: {st['cota_min']:,.1f} a {st['cota_max']:,.1f} m  (desnivel {st['desnivel']:,.1f} m)")
            datos_ter.append(f"Pendiente media aprox.: {st['pendiente_media']} %;  área con más de 30 %: {st['pct_mas_30']} %")
        if t.ubicado:
            lat, lng = a_latlng(*t.centroide())
            datos_ter.append(f"Centro aprox.: {lat:.5f}, {lng:.5f}")
        y = _parrafos(h.p, x0, y - 2, "DATOS DEL TERRENO", datos_ter, PANEL_W - 6, 5.6, 3.3)
        notas = [f"1. Coordenadas en {NOMBRE_SISTEMA}, en metros.",
                 "2. Rumbos y distancias según el plano proporcionado; área calculada con las coordenadas."]
        if cs:
            notas.append(f"3. Curvas de nivel a cada {intervalo:g} m (gruesas cada {intervalo * 5:g} m) generadas con "
                         f"{FUENTES.get(malla.fuente, malla.fuente)}. Error vertical de varios metros: solo referenciales.")
        notas.append(f"{len(notas) + 1}. Anteproyecto preliminar. Para trámite se requiere levantamiento topográfico y la firma "
                     "del profesional responsable.")
        _parrafos(h.p, x0, max(y - 1, h.tope_membrete + 34), "NOTAS", notas, PANEL_W - 6, 5.0, 2.9)
        h.guardar()

        # ---------------- Hoja 2: lotificación
        if lot:
            h = _Hoja(pdf, centro, escala, datos, "ANTEPROYECTO DE LOTIFICACIÓN", f"2 / {n_hojas}")
            if cs:
                h.curvas(cs, solo_indice=True, alpha=0.45)
            for c in lot.calles:
                h.ax.add_patch(_patch(c, fc=COLOR_CALLE, ec="#9a9a9a", lw=0.4, zorder=3))
            for v in lot.verdes:
                h.ax.add_patch(_patch(v["poly"], fc=COLOR_VERDE, ec="#4f8a3a", lw=0.4, zorder=3, hatch="..." if v["de_lote"] else None))
            for p in lot.remanentes:
                h.ax.add_patch(_patch(p, fc="#f2f2f2", ec="#bbbbbb", lw=0.3, zorder=3))
            for p in lot.protecciones:
                h.ax.add_patch(_patch(p, fc=COLOR_PROTECCION, ec="#3d6b2e", lw=0.4, zorder=3, hatch="////"))
                c = p.representative_point()
                _t(h.ax, c.x, c.y, "ÁREA VERDE ECOLÓGICA\n(zona de protección)", 4.4, ha="center", color="#1f3d17", zorder=8,
                   bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.75))
            for e in lot.equipamiento:
                h.ax.add_patch(_patch(e["poly"], fc=COLOR_EQUIP, ec="#3e6d99", lw=0.45, zorder=4))
                c = e["poly"].representative_point()
                _t(h.ax, c.x, c.y, "EQ", 5, ha="center", color="#1f3f5f", zorder=8, weight="bold")
            for l in lot.lotes:
                h.ax.add_patch(_patch(l["poly"], fc=COLOR_LOTE, ec="#7a5a2a", lw=0.45, zorder=4))
                c = l["poly"].representative_point()
                lado_mm = math.sqrt(l["area"]) / h.m
                _t(h.ax, c.x, c.y, str(l["id"]), max(3.6, min(7, lado_mm * 0.7)), ha="center", zorder=8, weight="bold")
            for i, v in enumerate(lot.verdes, 1):
                c = v["poly"].representative_point()
                _t(h.ax, c.x, c.y, f"AV-{i}", 4.6, ha="center", color="#2e5e22", zorder=8)
            h.lindero(t, lw=1.2)
            if caja_inf:
                items = [("Lote (número)", dict(fc=COLOR_LOTE, ec="#7a5a2a", lw=0.45)),
                         ("Calle", dict(fc=COLOR_CALLE, ec="#9a9a9a", lw=0.4)),
                         ("Área verde", dict(fc=COLOR_VERDE, ec="#4f8a3a", lw=0.4)),
                         ("Área verde tomada de lotes inclinados", dict(fc=COLOR_VERDE, ec="#4f8a3a", lw=0.4, hatch="...")),
                         ("Lindero del terreno", {"linea": dict(color=TINTA, lw=1.2)})]
                if not any(v["de_lote"] for v in lot.verdes):
                    items = [it for it in items if not it[0].startswith("Área verde tomada")]
                if lot.protecciones:
                    items.insert(4, ("Área verde ecológica: zona de protección de quebrada", dict(fc=COLOR_PROTECCION, ec="#3d6b2e", lw=0.4, hatch="////")))
                if lot.equipamiento:
                    items.insert(4, ("Equipamiento social (EQ)", dict(fc=COLOR_EQUIP, ec="#3e6d99", lw=0.45)))
                if cs:
                    items.append((f"Curva de nivel índice (cada {intervalo * 5:g} m)", {"linea": dict(color=COLOR_CURVA, lw=0.55, alpha=0.6)}))
                h.simbologia((caja_inf[0], caja_inf[1], caja_inf[0] + 78, caja_inf[1] + 9 + 5 * len(items)), items)
            s = lot.stats
            tot = s["area_total"]
            x0 = PX0 + 3
            filas_areas = [
                ("Lotes", s["lotes"], _fmt(s["area_util"]), f"{s['area_util'] / tot * 100:.1f} %"),
                ("Calles", "", _fmt(s["area_circulacion"]), f"{s['area_circulacion'] / tot * 100:.1f} %"),
                ("Área verde recreativa", len(lot.verdes), _fmt(s["area_verde"]), f"{s['area_verde'] / tot * 100:.1f} %"),
                ("Remanentes", len(lot.remanentes), _fmt(s["area_remanente"]), f"{s['area_remanente'] / tot * 100:.1f} %"),
                ("TOTAL", "", _fmt(tot), "100.0 %"),
            ]
            if s.get("area_equipamiento"):
                filas_areas.insert(3, ("Equipamiento social", len(lot.equipamiento), _fmt(s["area_equipamiento"]),
                                       f"{s['area_equipamiento'] / tot * 100:.1f} %"))
            if s.get("area_proteccion"):
                filas_areas.insert(3, ("Área verde ecológica", len(lot.protecciones), _fmt(s["area_proteccion"]),
                                       f"{s['area_proteccion'] / tot * 100:.1f} %"))
            y, _ = _tabla(h.p, x0, MARCO[3] - 3, [("USO", 40, "left"), ("CANT.", 18, "center"), ("ÁREA (m²)", 42, "right"), ("%", 34, "right")],
                          filas_areas, 4.2, "CUADRO DE ÁREAS")
            regla = {"pct_util": f"{lot.parametros['valor_verde']:g} % del área útil",
                     "pct_total": f"{lot.parametros['valor_verde']:g} % del área total",
                     "por_lote": f"{lot.parametros['valor_verde']:g} m² por lote"}[lot.parametros["regla_verde"]]
            pm = lot.parametros
            if pm.get("modo") == "calle_central":
                lados = " · ".join(f"{pl.get('terreno') or pl['lado']}: {pl['lotes']} lotes" for pl in s.get("por_lado", []))
                linea_lote = (f"Calle central de {pm['calle']:g} m con retorno de {s.get('retorno_diametro_m', 0):.0f} m · "
                              f"{lados} · frente mínimo {pm['frente_minimo']:g} m · área mínima {pm['area_minima']:g} m².")
            else:
                linea_lote = (f"Calle {pm['calle']:g} m · lote tipo {pm['frente']:g} × {pm['fondo']:g} m · "
                              f"área mínima {pm['area_minima']:g} m².")
            y = _parrafos(h.p, x0, y - 1, "CRITERIOS USADOS", [
                linea_lote,
                f"Área verde requerida: {regla} = {_fmt(s['area_verde_requerida'])} m²; propuesta {_fmt(s['area_verde'] + s.get('area_proteccion', 0))} m²"
                + (f" ({_fmt(s['area_verde'])} recreativa + {_fmt(s['area_proteccion'])} ecológica)" if s.get("area_proteccion") else "")
                + (f"; {s['lotes_convertidos_a_verde']} lotes pasados a área verde" if s["lotes_convertidos_a_verde"] else "")
                + (f"; {s['lotes_recortados_para_verde']} lotes recortados solo en lo necesario" if s.get("lotes_recortados_para_verde") else "")
                + ". El área verde ecológica debe arborizarse (mín. 400 árboles/ha, RLUC Art. 63).",
                (f"Orientación de calles {lot.angulo:g}°; " if pm.get("modo") != "calle_central" else "")
                + f"pendiente media a lo largo de la calle aprox. {s['pendiente_calles_pct']:.1f} %.",
            ], PANEL_W - 6, 5.2, 3.0)
            if cumplimiento:
                fc = [(f["regla"], f.get("ref", ""), f["requerido"], f["proyecto"], TEXTO_ESTADO.get(f["estado"], f["estado"]))
                      for f in cumplimiento]
                y, _ = _tabla(h.p, x0, y - 1, [("REGLA", 33, "left"), ("BASE", 21, "left"), ("REQUERIDO", 29, "left"),
                                               ("PROYECTO", 27, "left"), ("ESTADO", 24, "left")],
                              fc, 4.0, "CUADRO DE CUMPLIMIENTO (referencial)",
                              color_ultima=lambda fila: next((COLOR_ESTADO[k] for k, v in TEXTO_ESTADO.items() if v == fila[-1]), TINTA))
                _t(h.p, x0, y - 2.6, "LUC: Ley de Urbanismo y Construcción · RLUC: su Reglamento de parcelaciones (D.E. 70/1991) · "
                   "Validar con profesional.", 4.4, color="#555")
                y -= 4
            filas_lotes = [(l["id"], _fmt(l["area"]), _fmt(l["area"] / M2_POR_V2), _fmt(l["frente"], 1),
                            "—" if l["pendiente"] is None else f"{l['pendiente']:.0f}") for l in lot.lotes]
            col_lotes = [("LOTE", 10, "center"), ("m²", 16, "right"), ("v²", 16, "right"), ("FRENTE", 13, "right"), ("PEND.%", 12.5, "right")]
            y_tabla = y - 1
            y_min = h.tope_membrete + 2
            alto = 3.5
            y_fin, resto = _tabla(h.p, x0, y_tabla, col_lotes, filas_lotes, alto, "CUADRO DE LOTES", y_min=y_min)
            if resto:
                y_fin, resto = _tabla(h.p, x0 + 67.5 + 1.5, y_tabla - 5, col_lotes, resto, alto, None, y_min=y_min)
            if resto:
                hojas_extra.append((col_lotes, resto))
            h.guardar()

        # ---------------- hojas anexas (tablas que no cupieron)
        for k, (cols, filas) in enumerate(hojas_extra, 1):
            fig = plt.figure(figsize=(HOJA_W / 25.4, HOJA_H / 25.4))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, HOJA_W)
            ax.set_ylim(0, HOJA_H)
            ax.axis("off")
            _caja(ax, *MARCO, lw=1.2)
            _t(ax, MARCO[0] + 4, MARCO[3] - 6, f"{datos.proyecto.upper()} — ANEXO {k}: continuación de cuadros", 9, weight="bold")
            x = MARCO[0] + 4
            resto = filas
            ancho = sum(c[1] for c in cols) + 6
            while resto and x + ancho < MARCO[2]:
                _, resto = _tabla(ax, x, MARCO[3] - 12, cols, resto, 4.0, None, y_min=MARCO[1] + 4)
                x += ancho
            pdf.savefig(fig)
            plt.close(fig)
    return {"ruta": ruta, "escala": escala, "hojas": n_hojas + len(hojas_extra), "intervalo_curvas": intervalo,
            "estadisticas": st}
