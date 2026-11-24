"""Modelo 3D del anteproyecto: terreno con su relieve, lotes, calles, área verde y casas tipo.

Sirve para dos cosas:
- DXF 3D para AutoCAD / Civil 3D (mallas por capa, en coordenadas Lambert reales).
- Vista aérea en imagen, dibujada aquí mismo (sin programas externos).

Es ilustrativo: el relieve es satelital y las casas son volúmenes tipo, no un diseño arquitectónico.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.ops import nearest_points, unary_union

from .lotificacion import Resultado
from .relieve import Malla
from .terreno import Terreno

# clase -> (capa DXF, color ACI de AutoCAD, color RGB para la imagen)
CLASES = {
    "entorno": ("3D_ENTORNO", 52, (166, 178, 132)),
    "terreno": ("3D_TERRENO", 42, (196, 176, 128)),
    "lote": ("3D_LOTES", 31, (232, 206, 160)),
    "calle": ("3D_CALLES", 8, (150, 150, 150)),
    "verde": ("3D_AREA_VERDE", 94, (104, 160, 82)),
    "proteccion": ("3D_PROTECCION", 96, (70, 120, 60)),
    "equipamiento": ("3D_EQUIPAMIENTO", 151, (176, 200, 226)),
    "muro": ("3D_CASAS_MUROS", 7, (240, 236, 226)),
    "techo": ("3D_CASAS_TECHOS", 14, (170, 72, 52)),
    "base": ("3D_BASE", 33, (176, 140, 104)),
}
ID_CLASE = {c: i for i, c in enumerate(CLASES, start=1)}  # para máscaras de clases (0 = cielo)
ALTO_MUROS = 2.8
ALTO_TECHO = 1.6


@dataclass
class Escena:
    triangulos: dict[str, list] = field(default_factory=dict)   # clase -> lista de triángulos [(x,y,z)*3]
    lineas: dict[str, list] = field(default_factory=dict)       # nombre -> lista de polilíneas [(x,y,z), ...]
    casas: int = 0
    paso: float = 2.0

    def agregar(self, clase, tri):
        self.triangulos.setdefault(clase, []).append(tri)

    def todos(self):
        for clase, tris in self.triangulos.items():
            for t in tris:
                yield clase, t


def _alturas(m: Malla, xs, ys):
    """Altura bilineal vectorizada."""
    fx = np.clip((np.asarray(xs) - m.x0) / m.dx, 0, m.nx - 1.000001)
    fy = np.clip((np.asarray(ys) - m.y0) / m.dx, 0, m.ny - 1.000001)
    i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
    u, v = fx - i, fy - j
    z = m.z
    return (z[j, i] * (1 - u) * (1 - v) + z[j, i + 1] * u * (1 - v) + z[j + 1, i] * (1 - u) * v + z[j + 1, i + 1] * u * v)


def _sobre_terreno(m: Malla, linea, paso=1.5, alza=0.12):
    """Densifica una línea y la pega al relieve."""
    ls = LineString(linea)
    n = max(2, int(ls.length / paso) + 1)
    pts = [ls.interpolate(ls.length * k / (n - 1)) for k in range(n)]
    xs = np.array([p.x for p in pts])
    ys = np.array([p.y for p in pts])
    zs = _alturas(m, xs, ys) + alza
    return list(zip(xs.tolist(), ys.tolist(), zs.tolist()))


def huella_casa(lote: dict, calles, angulo: float) -> list[tuple[float, float]] | None:
    """Planta de la casa tipo de un lote: 4 esquinas (frente izq., frente der., fondo der., fondo izq. según u, v).
    La casa se recarga hacia la calle dejando 3 m de retiro; el ancho va paralelo a la calle."""
    poly = lote["poly"]
    interior = poly.buffer(-1.2)
    if interior.is_empty or interior.area < 30:
        return None
    th = math.radians(angulo)
    ux, uy = math.cos(th), math.sin(th)          # a lo largo de la calle
    vx, vy = -uy, ux                              # hacia el fondo
    c = poly.centroid
    # la casa se recarga hacia la calle (frente), dejando un retiro de 3 m
    if calles is not None and not calles.is_empty:
        p_calle, _ = nearest_points(calles, c)
        dx, dy = p_calle.x - c.x, p_calle.y - c.y
        if dx * vx + dy * vy > 0:
            vx, vy = -vx, -vy                         # v apunta del frente al fondo
    ancho = min(7.0, lote["frente"] * 0.7)
    fondo_lote = lote["area"] / max(lote["frente"], 1)
    prof = min(9.0, fondo_lote * 0.45)
    corrimiento = max(0.0, fondo_lote / 2 - 3.0 - prof / 2)
    centros = [(c.x - vx * corrimiento, c.y - vy * corrimiento)]
    # lotes irregulares (cuñas, lotes con cola): probar pegado a la calle y en el punto más holgado del lote
    if calles is not None and not calles.is_empty:
        p_calle, p_lote = nearest_points(calles, poly)
        for dentro in (3.0 + prof / 2, 3.0 + prof):
            centros.append((p_lote.x + vx * dentro, p_lote.y + vy * dentro))
    try:
        from shapely.ops import polylabel

        q = polylabel(interior if interior.geom_type == "Polygon" else poly, tolerance=0.5)
        centros.append((q.x, q.y))
    except Exception:  # noqa: BLE001
        pass
    for escala in (1.0, 0.85, 0.7):
        w, d = ancho * escala / 2, prof * escala / 2
        for cx, cy in centros:
            esquinas = [(cx + sx * w * ux + sy * d * vx, cy + sx * w * uy + sy * d * vy) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            if Polygon(esquinas).within(interior):
                return esquinas
    return None


def _casa(escena: Escena, m: Malla, lote: dict, calles, angulo: float):
    esquinas = huella_casa(lote, calles, angulo)
    if esquinas is None:
        return
    zs = _alturas(m, [p[0] for p in esquinas], [p[1] for p in esquinas])
    z0, z1 = float(zs.min()) - 0.3, float(zs.max()) + ALTO_MUROS
    base = [(x, y, z0) for x, y in esquinas]
    arriba = [(x, y, z1) for x, y in esquinas]
    for k in range(4):
        a, b = base[k], base[(k + 1) % 4]
        A, B = arriba[k], arriba[(k + 1) % 4]
        escena.agregar("muro", (a, b, B))
        escena.agregar("muro", (a, B, A))
    # techo a dos aguas: la cumbrera corre a lo ancho (paralela a la calle)
    m0 = ((arriba[0][0] + arriba[3][0]) / 2, (arriba[0][1] + arriba[3][1]) / 2, z1 + ALTO_TECHO)
    m1 = ((arriba[1][0] + arriba[2][0]) / 2, (arriba[1][1] + arriba[2][1]) / 2, z1 + ALTO_TECHO)
    escena.agregar("techo", (arriba[0], arriba[1], m1))
    escena.agregar("techo", (arriba[0], m1, m0))
    escena.agregar("techo", (arriba[3], m0, m1))
    escena.agregar("techo", (arriba[3], m1, arriba[2]))
    escena.agregar("muro", (arriba[0], m0, arriba[3]))
    escena.agregar("muro", (arriba[1], arriba[2], m1))
    escena.casas += 1


def construir(t: Terreno, malla: Malla, lot: Resultado | None = None, margen: float = 40.0,
              max_celdas: int = 45000, casas: bool = True) -> Escena:
    """Maqueta: un bloque rectangular de terreno (el predio y sus alrededores) con costados hasta una base."""
    terreno = Polygon(t.puntos).buffer(0)
    tminx, tminy, tmaxx, tmaxy = terreno.bounds
    mx1, my1 = malla.x0 + (malla.nx - 1) * malla.dx, malla.y0 + (malla.ny - 1) * malla.dx
    minx, miny = max(tminx - margen, malla.x0), max(tminy - margen, malla.y0)
    maxx, maxy = min(tmaxx + margen, mx1), min(tmaxy + margen, my1)
    region = box(minx, miny, maxx, maxy)
    paso = max(2.0, math.sqrt(region.area / max_celdas))
    nx = max(2, int(round((maxx - minx) / paso)) + 1)
    ny = max(2, int(round((maxy - miny) / paso)) + 1)
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    gx, gy = np.meshgrid(xs, ys)
    gz = _alturas(malla, gx, gy)
    cx, cy = (gx[:-1, :-1] + gx[1:, 1:]) / 2, (gy[:-1, :-1] + gy[1:, 1:]) / 2
    dentro = np.ones(cx.shape, dtype=bool)
    clase = np.full(cx.shape, "entorno", dtype=object)
    clase[shapely.contains_xy(terreno, cx, cy)] = "terreno"
    if lot is not None:
        for nombre, geoms in (("lote", [l["poly"] for l in lot.lotes]), ("verde", [v["poly"] for v in lot.verdes]),
                              ("proteccion", lot.protecciones), ("equipamiento", [e["poly"] for e in lot.equipamiento]),
                              ("calle", lot.calles)):
            if geoms:
                u = unary_union(geoms)
                clase[shapely.contains_xy(u, cx, cy) & dentro] = nombre
    escena = Escena(paso=paso)
    J, I = np.nonzero(dentro)
    for j, i in zip(J.tolist(), I.tolist()):
        p00 = (gx[j, i], gy[j, i], gz[j, i])
        p10 = (gx[j, i + 1], gy[j, i + 1], gz[j, i + 1])
        p01 = (gx[j + 1, i], gy[j + 1, i], gz[j + 1, i])
        p11 = (gx[j + 1, i + 1], gy[j + 1, i + 1], gz[j + 1, i + 1])
        c = clase[j, i]
        escena.agregar(c, (p00, p10, p11))
        escena.agregar(c, (p00, p11, p01))
    # costados de la maqueta, hasta una base pareja
    zbase = float(np.nanmin(gz)) - max(6.0, 0.08 * float(np.nanmax(gz) - np.nanmin(gz)))
    bordes = [list(zip(gx[0], gy[0], gz[0])), list(zip(gx[-1], gy[-1], gz[-1])),
              list(zip(gx[:, 0], gy[:, 0], gz[:, 0])), list(zip(gx[:, -1], gy[:, -1], gz[:, -1]))]
    for borde in bordes:
        for a, b in zip(borde, borde[1:]):
            a0, b0 = (a[0], a[1], zbase), (b[0], b[1], zbase)
            escena.agregar("base", (a0, b0, b))
            escena.agregar("base", (a0, b, a))
    escena.lineas["lindero"] = [_sobre_terreno(malla, t.puntos + t.puntos[:1], alza=0.2)]
    if lot is not None:
        escena.lineas["lotes"] = [_sobre_terreno(malla, list(l["poly"].exterior.coords)) for l in lot.lotes]
        if casas:
            calles = unary_union(lot.calles) if lot.calles else None
            for l in lot.lotes:
                _casa(escena, malla, l, calles, lot.angulo)
    return escena


# ---------------------------------------------------------------------------- DXF 3D
def dxf(ruta: str, escena: Escena) -> str:
    import ezdxf
    from ezdxf.render import MeshVertexMerger

    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    for clase, (capa, color, _) in CLASES.items():
        doc.layers.add(capa, color=color)
    doc.layers.add("3D_LINDERO", color=1)
    doc.layers.add("3D_BORDES_LOTES", color=250)
    for clase, tris in escena.triangulos.items():
        capa = CLASES[clase][0]
        for k in range(0, len(tris), 9000):  # bloques chicos: cada malla < 32,767 vértices
            mb = MeshVertexMerger()
            for tri in tris[k:k + 9000]:
                mb.add_face(tri)
            mb.render_polyface(msp, dxfattribs={"layer": capa})
    for nombre, capa in (("lindero", "3D_LINDERO"), ("lotes", "3D_BORDES_LOTES")):
        for pl in escena.lineas.get(nombre, []):
            msp.add_polyline3d(pl, dxfattribs={"layer": capa})
    doc.saveas(ruta)
    return ruta


# ---------------------------------------------------------------------------- vista aérea
def render_aereo(escena: Escena, t: Terreno, ancho: int = 1600, alto: int = 1000, elevacion: float = 38.0,
                 malla: Malla | None = None, lineas: bool = True, mascara: bool = False):
    """Vista aérea en perspectiva (algoritmo del pintor), sin textos. Devuelve (imagen PIL, pantalla) donde
    pantalla(x, y, z) da la posición en píxeles de un punto del terreno."""
    from PIL import Image, ImageDraw

    S = 2  # se dibuja al doble y se reduce: bordes suaves
    W, H = ancho * S, alto * S
    terreno = Polygon(t.puntos)
    c = terreno.centroid
    tris = [(cl, tri) for cl, tri in escena.todos()]
    allz = np.array([p[2] for _, tri in tris for p in tri])
    zc = float(np.median(allz))
    # eje largo del terreno y lado desde donde se mira (el lado bajo, para ver la ladera de frente)
    r = list(terreno.minimum_rotated_rectangle.exterior.coords)
    lados = [(r[i], r[i + 1]) for i in range(4)]
    (ax_, ay_), (bx_, by_) = max(lados, key=lambda l: math.dist(*l))
    L = math.dist((ax_, ay_), (bx_, by_)) or 1
    ux, uy = (bx_ - ax_) / L, (by_ - ay_) / L
    vx, vy = -uy, ux
    if malla is not None:
        gx, gy = malla.gradiente()
        mgx, mgy = float(np.nanmean(gx)), float(np.nanmean(gy))
        if vx * (-mgx) + vy * (-mgy) < 0:  # que la cámara quede del lado de abajo
            vx, vy = -vx, -vy
    elif vy > 0:
        vx, vy = -vx, -vy  # por defecto, mirar desde el sur
    bx0, by0, bx1, by1 = (min(p[0] for _, tri in tris for p in tri), min(p[1] for _, tri in tris for p in tri),
                          max(p[0] for _, tri in tris for p in tri), max(p[1] for _, tri in tris for p in tri))
    ext = max(bx1 - bx0, by1 - by0, 60.0)
    dist = ext * 1.5
    el = math.radians(elevacion)
    cam = np.array([c.x + vx * dist * math.cos(el), c.y + vy * dist * math.cos(el), zc + dist * math.sin(el)])
    objetivo = np.array([c.x, c.y, zc])
    f = objetivo - cam
    f /= np.linalg.norm(f)
    der = np.cross(f, [0, 0, 1.0])
    der /= np.linalg.norm(der)
    arr = np.cross(der, f)

    P = np.array([tri for _, tri in tris], dtype=float)          # (n,3,3)
    d = P - cam
    xc, yc, zcam = d @ der, d @ arr, d @ f
    zcam = np.maximum(zcam, 1e-3)
    sx, sy = xc / zcam, yc / zcam
    esc = 0.92 * min(W / (sx.max() - sx.min()), H * 0.82 / (sy.max() - sy.min()))
    ox = W / 2 - esc * (sx.max() + sx.min()) / 2
    oy = H * 0.47 + esc * (sy.max() + sy.min()) / 2
    X, Y = ox + esc * sx, oy - esc * sy

    # sombreado: sol desde el noroeste, 45° de altura
    sol = np.array([-math.cos(math.radians(45)) * math.sin(math.radians(45)),
                    math.cos(math.radians(45)) * math.cos(math.radians(45)), math.sin(math.radians(45))])
    sol /= np.linalg.norm(sol)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(nn == 0, 1, nn)
    horizontal = np.abs(n[:, 2]) > 0.2                      # terreno y techos; los muros son verticales
    n = np.where((n[:, 2:3] < 0) & horizontal[:, None], -n, n)  # normales hacia arriba
    difusa = np.where(horizontal, np.clip(n @ sol, 0, 1), 0.8 * np.abs(n @ sol))
    luz = 0.42 + 0.58 * difusa
    base = np.array([CLASES[cl][2] for cl, _ in tris], dtype=float)
    colores = np.clip(base * luz[:, None], 0, 255).astype(int)
    if mascara:  # cada triángulo con el número de su clase, sin luz ni líneas ni cielo
        colores = np.array([[ID_CLASE[cl], 0, 0] for cl, _ in tris], dtype=int)
        lineas = False
    prof = zcam.mean(axis=1)

    # líneas (bordes de lotes y lindero) como elementos con su propia profundidad
    segs = []
    for nombre, color, grosor in ((("lotes", (120, 92, 52), 2), ("lindero", (178, 34, 34), 4)) if lineas else ()):
        for pl in escena.lineas.get(nombre, []):
            Q = np.array(pl) - cam
            qx, qy, qz = Q @ der, Q @ arr, np.maximum(Q @ f, 1e-3)
            px, py = ox + esc * qx / qz, oy - esc * qy / qz
            for k in range(len(pl) - 1):
                segs.append(((qz[k] + qz[k + 1]) / 2 - 0.5, (px[k], py[k], px[k + 1], py[k + 1]), color, grosor))

    if mascara:
        segs = []
    img = Image.new("RGB", (W, H))
    dib = ImageDraw.Draw(img)
    for yy in (range(0, H, 4) if not mascara else ()):  # cielo
        a = yy / H
        dib.rectangle([0, yy, W, yy + 4], fill=(int(178 + 50 * a), int(206 + 30 * a), int(232 + 15 * a)))
    orden = sorted([(prof[k], 0, k) for k in range(len(tris))] + [(s[0], 1, k) for k, s in enumerate(segs)], reverse=True)
    for _, tipo, k in orden:
        if tipo == 0:
            col = tuple(colores[k])
            dib.polygon([(X[k, 0], Y[k, 0]), (X[k, 1], Y[k, 1]), (X[k, 2], Y[k, 2])], fill=col, outline=col)
        else:
            _, (x1, y1, x2, y2), col, g = segs[k]
            dib.line([(x1, y1), (x2, y2)], fill=col, width=g)
    img = img.resize((ancho, alto), Image.NEAREST if mascara else Image.LANCZOS)

    def pantalla(x, y, z):
        q = np.array([x, y, z]) - cam
        return (ox + esc * (q @ der) / max(q @ f, 1e-3)) / S, (oy - esc * (q @ arr) / max(q @ f, 1e-3)) / S

    pantalla.zc = zc
    pantalla.centro = (c.x, c.y)
    return img, pantalla


def flecha_norte(dib, pantalla, kx: float, ky: float, R: float = 40, fuente_negrita: str | None = None):
    """Rosa del norte para una vista en perspectiva: apunta hacia donde queda el norte en la imagen."""
    from matplotlib import font_manager
    from PIL import ImageFont

    cx_, cy_ = pantalla.centro
    p0, p1 = pantalla(cx_, cy_, pantalla.zc), pantalla(cx_, cy_ + 10, pantalla.zc)
    nx_, ny_ = p1[0] - p0[0], p1[1] - p0[1]
    nl = math.hypot(nx_, ny_) or 1
    nx_, ny_ = nx_ / nl, ny_ / nl
    k = R / 40
    dib.ellipse([kx - R, ky - R, kx + R, ky + R], fill=(255, 255, 255, 210))
    punta = (kx + nx_ * 14 * k, ky + ny_ * 14 * k)
    cola = (kx - nx_ * 24 * k, ky - ny_ * 24 * k)
    izq = (cola[0] - ny_ * 10 * k, cola[1] + nx_ * 10 * k)
    der_ = (cola[0] + ny_ * 10 * k, cola[1] - nx_ * 10 * k)
    dib.polygon([punta, izq, (kx - nx_ * 14 * k, ky - ny_ * 14 * k), der_], fill=(30, 30, 30))
    negrita = fuente_negrita or font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    fN = ImageFont.truetype(negrita, int(18 * k))
    dib.text((kx + nx_ * 27 * k, ky + ny_ * 27 * k), "N", font=fN, fill=(30, 30, 30), anchor="mm")


def vista(ruta: str, escena: Escena, t: Terreno, titulo: str = "", subtitulo: str = "",
          ancho: int = 1600, alto: int = 1000, elevacion: float = 38.0, malla: Malla | None = None) -> str:
    """Vista aérea en perspectiva con título y norte, guardada como JPG."""
    from matplotlib import font_manager
    from PIL import ImageDraw, ImageFont

    img, pantalla = render_aereo(escena, t, ancho, alto, elevacion, malla)
    dib = ImageDraw.Draw(img, "RGBA")
    fuente = font_manager.findfont("DejaVu Sans")
    negrita = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    f1, f2 = ImageFont.truetype(negrita, 30), ImageFont.truetype(fuente, 18)
    flecha_norte(dib, pantalla, ancho - 72, 72, 40, negrita)
    dib.rectangle([0, alto - 92, ancho, alto], fill=(255, 255, 255, 215))
    dib.text((24, alto - 84), titulo or "Anteproyecto de lotificación", font=f1, fill=(30, 30, 30))
    dib.text((24, alto - 40), subtitulo or "Vista 3D ilustrativa", font=f2, fill=(70, 70, 70))
    aviso = "Vista ilustrativa · relieve satelital aproximado · no válida para trámite"
    tw = dib.textlength(aviso, font=f2)
    dib.text((ancho - tw - 24, alto - 40), aviso, font=f2, fill=(160, 40, 30))
    img.save(ruta, "JPEG", quality=90)
    return ruta
