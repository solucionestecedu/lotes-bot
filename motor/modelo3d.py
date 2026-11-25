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
from shapely.geometry import LineString, Point, Polygon, box
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
    "plataforma": ("3D_PLATAFORMAS", 41, (214, 200, 168)),
    "talud": ("3D_TALUDES", 32, (150, 128, 96)),
    "agua": ("3D_QUEBRADA", 5, (64, 124, 186)),
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


def _frente(poly: Polygon, calles):
    """Tramo del lindero del lote que da a la calle (el más largo) o None."""
    if calles is None or calles.is_empty:
        return None
    g = poly.boundary.intersection(calles.buffer(0.6))
    tramos = [x for x in getattr(g, "geoms", [g]) if x.geom_type == "LineString" and x.length > 1.0]
    if not tramos:
        from shapely.ops import linemerge

        try:
            g = linemerge(g)
            tramos = [x for x in getattr(g, "geoms", [g]) if x.geom_type == "LineString" and x.length > 1.0]
        except Exception:  # noqa: BLE001
            tramos = []
    return max(tramos, key=lambda x: x.length) if tramos else None


def huella_casa(lote: dict, calles, angulo: float) -> list[tuple[float, float]] | None:
    """Planta de la casa tipo de un lote: 4 esquinas (frente izq., frente der., fondo der., fondo izq. según u, v).
    La casa va pegada a la calle (3 m de retiro) y el jardín queda al fondo; el ancho va paralelo al frente del lote.
    En lotes sesgados o con cola se busca, a cada profundidad, el centro del lote a esa profundidad, y se queda la
    posición más cercana a la calle donde la casa cabe completa."""
    poly = lote["poly"]
    interior = poly.buffer(-1.2)
    if interior.is_empty or interior.area < 30:
        return None
    fondo_lote = lote["area"] / max(lote["frente"], 1)
    ancho = min(max(7.0, lote["frente"] * 0.5), 14.0, lote["frente"] * 0.7)
    prof = min(max(9.0, fondo_lote * 0.3), 12.0, fondo_lote * 0.45)
    fr = _frente(poly, calles)
    if fr is not None:
        (ax, ay), (bx, by) = fr.coords[0], fr.coords[-1]
        L = math.hypot(bx - ax, by - ay) or 1
        ux, uy = (bx - ax) / L, (by - ay) / L
    else:
        th = math.radians(angulo)
        ux, uy = math.cos(th), math.sin(th)
    vx, vy = -uy, ux                              # hacia el fondo
    base = fr.interpolate(0.5, normalized=True) if fr is not None else poly.centroid
    if fr is not None and not poly.buffer(-0.05).contains(Point(base.x + vx * 2, base.y + vy * 2)):
        vx, vy = -vx, -vy
    if fr is None:
        c = poly.centroid
        if calles is not None and not calles.is_empty:
            p_calle, _ = nearest_points(calles, c)
            if (p_calle.x - c.x) * vx + (p_calle.y - c.y) * vy > 0:
                vx, vy = -vx, -vy
    largo = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1])
    puntos = [base] + ([fr.interpolate(f, normalized=True) for f in (0.3, 0.7, 0.15, 0.85)] if fr is not None else [])
    # orientaciones: a escuadra con el frente y, en lotes en cuña o sesgados, siguiendo el eje del lote
    ejes = [(ux, uy, vx, vy)]
    c = poly.centroid
    ex, ey = c.x - base.x, c.y - base.y
    el = math.hypot(ex, ey)
    if el > 1:
        ex, ey = ex / el, ey / el
        if abs(ex * vx + ey * vy) < 0.985:  # más de ~10° de diferencia
            ejes.append((ey, -ex, ex, ey))
    opciones = []
    for escala in (1.0, 0.85, 0.7, 0.6):
        w, dd = ancho * escala / 2, prof * escala / 2
        mejor = None
        for (ux, uy, vx, vy), p0 in [(e, p) for e in ejes for p in puntos]:
            prof_ = 3.0 + dd
            while prof_ < largo:
                cx, cy = p0.x + vx * prof_, p0.y + vy * prof_
                # centro del lote a esa profundidad (los lotes sesgados se corren hacia un lado)
                corte = LineString([(cx - ux * largo, cy - uy * largo), (cx + ux * largo, cy + uy * largo)]).intersection(interior)
                pedazos = [g for g in getattr(corte, "geoms", [corte]) if g.geom_type == "LineString" and not g.is_empty]
                if pedazos:
                    pz = min(pedazos, key=lambda g: g.distance(Point(cx, cy)))
                    m_ = pz.interpolate(0.5, normalized=True)
                    cx, cy = m_.x, m_.y
                esquinas = [(cx + sx * w * ux + sy * dd * vx, cy + sx * w * uy + sy * dd * vy)
                            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                if Polygon(esquinas).within(interior):
                    dist = Polygon(esquinas).distance(calles) if calles is not None and not calles.is_empty else prof_
                    if mejor is None or dist < mejor[0] - 0.01:
                        mejor = (dist, esquinas)
                    break
                prof_ += 1.0
        if mejor is not None:
            if mejor[0] <= 6.0:  # la más grande que queda pegada a la calle
                return mejor[1]
            opciones.append(mejor)
    # si ninguna cabe junto a la calle (lotes muy angostos al frente), la que quede más cerca
    return min(opciones, key=lambda o: o[0])[1] if opciones else None


def _casa(escena: Escena, m: Malla, lote: dict, calles, angulo: float, pad: dict | None = None):
    esquinas = pad["casa"].exterior.coords[:4] if pad is not None else huella_casa(lote, calles, angulo)
    if esquinas is None:
        return
    if pad is not None:  # sobre su terraza
        z0, z1 = pad["z"], pad["z"] + ALTO_MUROS
    else:
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
              max_celdas: int = 45000, casas: bool = True, plataformas: list | None = None, cauce=None) -> Escena:
    """Maqueta: un bloque rectangular de terreno (el predio y sus alrededores) con costados hasta una base. Con
    `plataformas` (altimetria.plataformas) cada casa queda sobre su terraza a la cota de la calle, con sus taludes."""
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
    if plataformas:
        u_pads = unary_union([p["poly"] for p in plataformas])
        dentro &= ~shapely.contains_xy(u_pads, cx, cy)  # el terreno natural se quita donde va una terraza
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
    pads = {p["id"]: p for p in plataformas or []}
    for p in pads.values():
        _terraza(escena, malla, p, paso)
    if cauce is not None and cauce.length > 2:  # el agua de la quebrada, para que la IA la ponga donde va
        cinta = cauce.buffer(2.2, cap_style=2).buffer(0)
        tris = shapely.delaunay_triangles(cinta.segmentize(max(1.5, paso)))
        for tri in getattr(tris, "geoms", [tris]):
            if tri.is_empty or not tri.representative_point().within(cinta.buffer(0.01)):
                continue
            pts = list(tri.exterior.coords)[:3]
            zs = _alturas(malla, [q[0] for q in pts], [q[1] for q in pts])
            if np.isfinite(zs).all():
                escena.agregar("agua", tuple((q[0], q[1], float(z) + 0.25) for q, z in zip(pts, zs)))
    if lot is not None:
        escena.lineas["lotes"] = [_sobre_terreno(malla, list(l["poly"].exterior.coords)) for l in lot.lotes]
        if casas:
            calles = unary_union(lot.calles) if lot.calles else None
            for l in lot.lotes:
                _casa(escena, malla, l, calles, lot.angulo, pads.get(l["id"]))
    return escena


def _terraza(escena: Escena, m: Malla, pad: dict, paso: float):
    """Terraza plana a su cota y, en el borde, la cara del talud hasta el terreno natural (corte o relleno)."""
    poly = pad["poly"].buffer(0)
    if poly.is_empty:
        return
    z = pad["z"]
    tris = shapely.delaunay_triangles(poly, only_edges=False)
    for tri in getattr(tris, "geoms", [tris]):
        if tri.is_empty or not tri.representative_point().within(poly.buffer(0.01)):
            continue
        a, b, c = list(tri.exterior.coords)[:3]
        escena.agregar("plataforma", ((a[0], a[1], z), (b[0], b[1], z), (c[0], c[1], z)))
    borde = LineString(poly.exterior.coords).segmentize(max(1.0, paso))
    cs = list(borde.coords)
    zs = _alturas(m, [p[0] for p in cs], [p[1] for p in cs])
    for (x0, y0), (x1, y1), g0, g1 in zip(cs, cs[1:], zs, zs[1:]):
        if abs(g0 - z) < 0.05 and abs(g1 - z) < 0.05:
            continue
        a, b = (x0, y0, z), (x1, y1, z)
        A, B = (x0, y0, float(g0)), (x1, y1, float(g1))
        escena.agregar("talud", (a, b, B))
        escena.agregar("talud", (a, B, A))


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
                 malla: Malla | None = None, lineas: bool = True, mascara: bool = False, camara: dict | None = None):
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
    if camara is not None:  # cámara libre (a la altura de un dron bajo): posición, a dónde mira y ángulo de visión
        cam = np.array(camara["pos"], dtype=float)
        objetivo = np.array(camara["mira"], dtype=float)
    f = objetivo - cam
    f /= np.linalg.norm(f)
    der = np.cross(f, [0, 0, 1.0])
    der /= np.linalg.norm(der)
    arr = np.cross(der, f)

    P = np.array([tri for _, tri in tris], dtype=float)          # (n,3,3)
    d = P - cam
    xc, yc, zcam = d @ der, d @ arr, d @ f
    if camara is not None:  # fuera lo que queda detrás de la cámara o pegado a ella
        ok = (zcam > 1.5).all(axis=1)
        P, xc, yc, zcam = P[ok], xc[ok], yc[ok], zcam[ok]
        tris = [tr for tr, k in zip(tris, ok) if k]
    zcam = np.maximum(zcam, 1e-3)
    sx, sy = xc / zcam, yc / zcam
    if camara is not None:
        esc = (W / 2) / math.tan(math.radians(camara.get("fov", 60)) / 2)
        ox, oy = W / 2, H / 2
    else:
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
                if camara is not None and min(qz[k], qz[k + 1]) < 1.5:
                    continue
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
