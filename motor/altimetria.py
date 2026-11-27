"""Altimetría para el diseño: pendientes, zonas de barranco, relieve de cada lote, perfil de la calle y la descripción
del relieve que se le da a la IA del render.

Todo sale de la malla de alturas (satelital o, cuando exista, la del topógrafo). Con relieve satelital (~30 m de origen)
las caídas cortas se suavizan: una quebrada de pocos metros puede verse menos inclinada de lo que es.
"""
from __future__ import annotations

import math

import numpy as np
from contourpy import contour_generator
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from .relieve import Malla

PEND_CONSTRUIBLE = 30.0   # hasta aquí se construye con terrazas normales; arriba pide muros o taludes
PEND_BARRANCO = 50.0      # más que esto es caída fuerte: no se lotifica
CLASES = [  # (desde %, hasta %, nombre, color RGB para la lámina)
    (0.0, 15.0, "suave (0–15 %)", (236, 241, 226)),
    (15.0, 30.0, "moderada (15–30 %)", (246, 228, 182)),
    (30.0, 50.0, "fuerte (30–50 %)", (240, 186, 140)),
    (50.0, math.inf, "barranco (> 50 %)", (214, 128, 112)),
]
RUMBOS = ["este", "noreste", "norte", "noroeste", "oeste", "suroeste", "sur", "sureste"]


def pendientes(malla: Malla) -> np.ndarray:
    """Pendiente en % en cada nodo de la malla."""
    gx, gy = malla.gradiente()
    return np.hypot(gx, gy) * 100


def _pend_en(malla: Malla, x: float, y: float) -> float:
    h = 0.5 * malla.dx
    zx = (malla.altura(x + h, y) - malla.altura(x - h, y)) / (2 * h)
    zy = (malla.altura(x, y + h) - malla.altura(x, y - h)) / (2 * h)
    return math.hypot(zx, zy) * 100


def _muestras(malla: Malla, poly: Polygon, n: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Puntos regulares dentro del polígono con su cota y su pendiente (interpoladas)."""
    paso = max(1.0, min(malla.dx, math.sqrt(poly.area / n)))
    x0, y0, x1, y1 = poly.bounds
    pr = prep(poly)
    xs, ys, zs, ps = [], [], [], []
    for x in np.arange(x0 + paso / 2, x1, paso):
        for y in np.arange(y0 + paso / 2, y1, paso):
            if pr.contains(Point(x, y)):
                z = malla.altura(x, y)
                if math.isfinite(z):
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    ps.append(_pend_en(malla, x, y))
    if not xs:
        c = poly.representative_point()
        xs, ys, zs, ps = [c.x], [c.y], [malla.altura(c.x, c.y)], [_pend_en(malla, c.x, c.y)]
    return np.array(xs), np.array(ys), np.array(zs), np.array(ps)


def _rumbo(dx: float, dy: float) -> str:
    return RUMBOS[int(round(math.degrees(math.atan2(dy, dx)) / 45.0)) % 8]


def caida(xs, ys, zs) -> tuple[float, float, float]:
    """Plano que mejor se ajusta: devuelve (dirección de bajada x, y unitaria, pendiente del plano en %)."""
    if len(xs) < 3:
        return 0.0, 0.0, 0.0
    A = np.c_[xs - xs.mean(), ys - ys.mean(), np.ones(len(xs))]
    a, b, _ = np.linalg.lstsq(A, zs, rcond=None)[0]
    n = math.hypot(a, b)
    return (-a / n, -b / n, n * 100) if n > 1e-9 else (0.0, 0.0, 0.0)


def por_lote(malla: Malla, poly: Polygon) -> dict:
    xs, ys, zs, ps = _muestras(malla, poly)
    ok = np.isfinite(ps)
    bx, by, _ = caida(xs, ys, zs)
    return {
        "cota_min": round(float(zs.min()), 1), "cota_max": round(float(zs.max()), 1),
        "desnivel": round(float(zs.max() - zs.min()), 1),
        "pend_media": round(float(ps[ok].mean()), 1) if ok.any() else None,
        "pend_max": round(float(ps[ok].max()), 1) if ok.any() else None,
        "construible_pct": round(float(np.mean(ps[ok] <= PEND_CONSTRUIBLE) * 100)) if ok.any() else None,
        "cae_hacia": _rumbo(bx, by) if (bx or by) else "plano",
    }


def enriquecer(lot, malla: Malla | None) -> None:
    """Relieve de cada lote (cotas, desnivel, pendiente media y máxima, % construible y hacia dónde cae). La pendiente
    del lote pasa a ser la media de todo el lote, no la del centro."""
    if malla is None:
        return
    for l in lot.lotes:
        try:
            r = por_lote(malla, l["poly"])
        except Exception:  # noqa: BLE001
            continue
        l["relieve"] = r
        if r["pend_media"] is not None:
            l["pendiente"] = r["pend_media"]


def zonas(malla: Malla, recorte: Polygon, desde: float, hasta: float = math.inf, area_min: float = 40.0) -> list[Polygon]:
    """Polígonos donde la pendiente está entre `desde` y `hasta` (%), recortados a `recorte`."""
    p = pendientes(malla)
    p = np.where(np.isfinite(p), p, 0.0)
    gen = contour_generator(malla.xs, malla.ys, p, fill_type="OuterOffset")
    pts_l, offs_l = gen.filled(desde, min(hasta, 1e6))
    polys = []
    for pts, offs in zip(pts_l, offs_l):
        anillos = [pts[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
        if not anillos or len(anillos[0]) < 4:
            continue
        g = Polygon(anillos[0], [a for a in anillos[1:] if len(a) >= 4]).buffer(0)
        if not g.is_empty:
            polys.append(g)
    if not polys:
        return []
    g = unary_union(polys).intersection(recorte)
    return [q for q in getattr(g, "geoms", [g]) if q.geom_type == "Polygon" and q.area >= area_min]


def barrancos(malla: Malla, anillo, umbral: float = PEND_BARRANCO, area_min: float = 60.0, tope: float = 0.20,
              partir: bool = False) -> list[Polygon]:
    """Caídas fuertes que no se lotifican: zonas con más de `umbral` % pegadas al borde del terreno (el barranco de una
    quebrada, un talud). Se suavizan 2 m. No se toman las que partirían el terreno en pedazos ni más del `tope` del
    área: en un terreno empinado de punta a punta las pendientes fuertes son parte de la ladera (se resuelven con
    terrazas y lotes más grandes), no un barranco. Con `partir` sí se toman las que cruzan el terreno de lado a lado
    (en una sola fila de lotes la calle las cruza y los lotes quedan antes y después)."""
    terreno = Polygon(anillo).buffer(0)
    borde = terreno.exterior.buffer(3.0)
    cands = []
    for z in zonas(malla, terreno, umbral, area_min=area_min):
        if z.intersects(borde):
            z = z.buffer(2.0, join_style=1).buffer(-2.0, join_style=1).intersection(terreno)
            cands += [q for q in getattr(z, "geoms", [z]) if q.geom_type == "Polygon" and q.area >= area_min
                      and q.intersects(terreno.exterior.buffer(1.0))]  # que de verdad llegue al lindero
    out, quitado = [], 0.0
    hoyos0 = len(terreno.interiors)
    for z in sorted(cands, key=lambda q: -q.area):
        if quitado + z.area > tope * terreno.area:
            continue
        resto = terreno.difference(unary_union(out + [z]))
        piezas = sorted((q for q in getattr(resto, "geoms", [resto]) if q.geom_type == "Polygon"), key=lambda q: -q.area)
        if not piezas or (not partir and piezas[0].area < 0.97 * resto.area):  # dejaría pedazos sueltos
            continue
        if len(piezas[0].interiors) > hoyos0:  # dejaría un hoyo adentro de los lotes
            continue
        out.append(z)
        quitado += z.area
    return out


def perfil(malla: Malla, linea: LineString, paso: float = 2.0) -> list[tuple[float, float]]:
    """Perfil longitudinal: [(distancia desde el inicio, cota)]."""
    n = max(2, int(linea.length / paso))
    out = []
    for i in range(n + 1):
        q = linea.interpolate(i / n, normalized=True)
        z = malla.altura(q.x, q.y)
        if math.isfinite(z):
            out.append((round(linea.length * i / n, 2), round(z, 2)))
    return out


def pendiente_eje(malla: Malla, linea: LineString, tramo: float = 10.0) -> tuple[float, float, float]:
    """Pendiente de una calle por su eje: (largo, pendiente media %, pendiente del tramo de `tramo` m más inclinado %).
    Es la misma medida que usa la revisión de normas."""
    L = linea.length
    if L < 5:
        return L, math.nan, math.nan
    n = max(2, int(L / tramo) + 1)
    zs = [malla.altura(*linea.interpolate(L * k / (n - 1)).coords[0]) for k in range(n)]
    if not all(math.isfinite(z) for z in zs):
        return L, math.nan, math.nan
    paso = L / (n - 1)
    return L, abs(zs[-1] - zs[0]) / L * 100, max(abs(zs[k + 1] - zs[k]) / paso * 100 for k in range(n - 1))


def resumen(malla: Malla, anillo) -> dict:
    """Relieve del terreno completo: cotas, desnivel, hacia dónde baja, pendiente media y reparto por clases."""
    poly = Polygon(anillo).buffer(0)
    xs, ys, zs, ps = _muestras(malla, poly, n=600)
    ok = np.isfinite(ps)
    bx, by, plano = caida(xs, ys, zs)
    clases = []
    for a, b, nombre, _ in CLASES:
        frac = float(np.mean((ps[ok] >= a) & (ps[ok] < b))) if ok.any() else 0.0
        clases.append({"clase": nombre, "pct": round(frac * 100, 1), "m2": round(frac * poly.area, 1)})
    alto = (xs[zs >= np.percentile(zs, 90)].mean(), ys[zs >= np.percentile(zs, 90)].mean())
    bajo = (xs[zs <= np.percentile(zs, 10)].mean(), ys[zs <= np.percentile(zs, 10)].mean())
    return {
        "cota_min": round(float(zs.min()), 1), "cota_max": round(float(zs.max()), 1),
        "desnivel": round(float(zs.max() - zs.min()), 1),
        "pend_media": round(float(ps[ok].mean()), 1) if ok.any() else None,
        "baja_hacia": _rumbo(bx, by) if (bx or by) else "plano", "pendiente_general": round(plano, 1),
        "clases": clases, "punto_alto": alto, "punto_bajo": bajo,
    }


def _zona_imagen(x: float, y: float, caja: tuple) -> str:
    """Dónde cae un punto dentro del terreno tal como se ve en la imagen (caja = x0, y0, x1, y1 del terreno en px)."""
    x0, y0, x1, y1 = caja
    fy = (y - y0) / max(1e-6, y1 - y0)
    fx = (x - x0) / max(1e-6, x1 - x0)
    v = "far end (top of the image)" if fy < 0.36 else "near end (bottom of the image)" if fy > 0.64 else "middle"
    h = "left side" if fx < 0.36 else "right side" if fx > 0.64 else ""
    if v == "middle":
        return f"middle, {h}" if h else "middle"
    return f"{v}, {h}" if h else v


def guia_render(malla: Malla | None, anillo, a_px, W: float, H: float, barr: list | None = None) -> str:
    """El relieve dicho en términos de la imagen, para que la IA no aplane el terreno: de qué parte de la imagen a cuál
    baja, cuántos metros, qué tan inclinado y dónde están las caídas fuertes."""
    if malla is None:
        return ""
    try:
        r = resumen(malla, anillo)
    except Exception:  # noqa: BLE001
        return ""
    if r["desnivel"] < 3:
        return (f" TERRAIN: the land is almost flat (about {r['desnivel']:.0f} m of difference); keep it gently "
                "undulating, not perfectly flat.")
    pts = [a_px(*q) for q in Polygon(anillo).exterior.coords]
    caja = (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
    alto = _zona_imagen(*a_px(*r["punto_alto"]), caja)
    bajo = _zona_imagen(*a_px(*r["punto_bajo"]), caja)
    pm = r["pend_media"] or 0
    tipo = ("a gentle slope" if pm < 12 else "a clear hillside slope" if pm < 25 else
            "a steep hillside" if pm < 45 else "a very steep hillside")
    txt = (f" TERRAIN, KEEP IT: the land is NOT flat. It is {tipo}: the highest part of the property is at its {alto} and "
           f"it goes down about {r['desnivel']:.0f} m towards its {bajo} (average slope about {pm:.0f} %). The road runs "
           "with the slope, houses sit on stepped terraces with low stone retaining walls and short embankments, and "
           "gardens follow the fall of the land. Do not flatten, level or tilt the ground differently.")
    fuertes = [b for b in (barr or []) if b.area > 30]
    if fuertes:
        c = unary_union(fuertes).centroid
        txt += (f" At the {_zona_imagen(*a_px(c.x, c.y), caja)} of the property there is a steep drop (ravine bank): show "
                "it as a sharp forested slope going down, with no houses on it.")
    return txt


# ------------------------------------------------------------------ terracería por lote
def plataformas(lot, malla: Malla, retiro_fondo: float = 3.0, margen_lado: float = 2.0) -> list[dict]:
    """Plataforma de cada lote para la casa pegada a la calle: va del borde de la calle hasta 3 m detrás de la casa, con
    2 m a cada lado, a la cota de la calle frente al lote (acceso del carro a nivel). El resto del lote (jardín) se
    deja con su caída natural. Devuelve por lote la plataforma, su cota y el corte y relleno aproximados (m³, sin
    taludes) y la altura máxima de corte y de relleno."""
    from shapely.ops import nearest_points

    from .modelo3d import _frente, huella_casa

    calles = unary_union(lot.calles) if lot.calles else None
    out = []
    for l in lot.lotes:
        esq = huella_casa(l, calles, lot.angulo)
        if esq is None:
            continue
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = esq
        ux, uy = x1 - x0, y1 - y0
        w = math.hypot(ux, uy) or 1
        ux, uy = ux / w, uy / w
        vx, vy = x3 - x0, y3 - y0
        dpr = math.hypot(vx, vy) or 1
        vx, vy = vx / dpr, vy / dpr
        casa = Polygon(esq)
        frente_casa = LineString([(x0, y0), (x1, y1)])
        # hasta la calle: el frente de la casa se proyecta al borde de la calle más cercano
        if calles is not None:
            p_c, p_h = nearest_points(calles, frente_casa)
            hasta = max(0.0, p_c.distance(p_h)) + 0.5
        else:
            hasta = 3.0
        a = (x0 - ux * margen_lado - vx * hasta, y0 - uy * margen_lado - vy * hasta)
        b = (x1 + ux * margen_lado - vx * hasta, y1 + uy * margen_lado - vy * hasta)
        c = (x2 + ux * margen_lado + vx * retiro_fondo, y2 + uy * margen_lado + vy * retiro_fondo)
        d = (x3 - ux * margen_lado + vx * retiro_fondo, y3 - uy * margen_lado + vy * retiro_fondo)
        pad = Polygon([a, b, c, d]).intersection(l["poly"])
        if pad.is_empty:
            continue
        if pad.geom_type != "Polygon":
            pad = max(getattr(pad, "geoms", [pad]), key=lambda g: g.area)
        # cota: la de la calle frente a la casa (eje de la calle en el punto más cercano)
        if calles is not None:
            pc, _ = nearest_points(calles, casa)
            z = malla.altura(pc.x, pc.y)
        else:
            z = float(np.nanmean(_muestras(malla, pad)[2]))
        xs, ys, zs, _ = _muestras(malla, pad, n=200)
        celda = pad.area / max(1, len(zs))
        dz = zs - z
        out.append({"id": l["id"], "poly": pad, "casa": casa, "z": round(float(z), 2),
                    "corte_m3": round(float(np.sum(np.clip(dz, 0, None)) * celda)),
                    "relleno_m3": round(float(np.sum(np.clip(-dz, 0, None)) * celda)),
                    "corte_max": round(float(max(0.0, dz.max())), 1), "relleno_max": round(float(max(0.0, -dz.min())), 1),
                    "eje": (ux, uy, vx, vy)})
    return out


# ------------------------------------------------------------------ terrazas con muros de contención
RAMPA_MAX = 1.5      # m que puede subir o bajar la terraza de la casa respecto a la calle (rampa lateral de ≤ 15 %)
ESCALON_JARDIN = 2.5  # desnivel natural máximo dentro de una terraza de jardín antes de partirla en otra
MURO_ESTRUCTURAL = 3.0  # muros más altos que esto piden diseño estructural
ANCHO_MIN_TERRAZA = 3.0  # m: en ladera muy fuerte el jardín se hace en bancales, pero ninguno más angosto que esto


def _banda(poly: Polygon, base, vx, vy, s0: float, s1: float, largo: float, w0: float | None = None,
           w1: float | None = None):
    """Parte del lote entre las profundidades s0 y s1 (medidas desde `base` en la dirección v) y, si se dan, entre
    w0 y w1 a lo ancho (dirección u, a lo largo de la calle). Puede salir en varias piezas."""
    ux, uy = vy, -vx
    bx, by = base
    a, b = (-largo, largo) if w0 is None else (w0, w1)
    caja = Polygon([(bx + vx * s0 + ux * a, by + vy * s0 + uy * a), (bx + vx * s0 + ux * b, by + vy * s0 + uy * b),
                    (bx + vx * s1 + ux * b, by + vy * s1 + uy * b), (bx + vx * s1 + ux * a, by + vy * s1 + uy * a)])
    g = poly.intersection(caja)
    if g.is_empty:
        return g
    piezas = [q for q in getattr(g, "geoms", [g]) if q.geom_type == "Polygon" and q.area > 0.5]
    return unary_union(piezas) if piezas else Polygon()


def _partes(g):
    return [q for q in getattr(g, "geoms", [g]) if q.geom_type == "Polygon" and not q.is_empty]


def _cortes_ancho(malla, poly, base, vx, vy, s0, s1, w0, w1, largo) -> list[float]:
    """Dónde partir a lo ancho una franja lateral para que cada pedazo no tenga más de ESCALON_JARDIN de caída."""
    if w1 - w0 < 3.0:
        return [w0, w1]
    g = _banda(poly, base, vx, vy, s0, s1, largo, w0, w1)
    if g.is_empty or g.area < 6:
        return [w0, w1]
    zs = _muestras(malla, g, n=60)[2]
    caida = float(np.nanmax(zs) - np.nanmin(zs)) if len(zs) else 0.0
    k = int(min(6, max(1, math.ceil(caida / ESCALON_JARDIN)), max(1, (w1 - w0) // ANCHO_MIN_TERRAZA)))
    return [w0 + (w1 - w0) * i / k for i in range(k + 1)]


def terrazas(lot, malla: Malla) -> list[dict]:
    """Terracería que sigue el terreno, con muros de contención en lugar de taludes (no se pierde terreno en taludes).
    Cada lote se parte en una cuadrícula que escalona en los dos sentidos:
    - a lo hondo: la fila de la casa (desde la calle hasta 3 m detrás de la casa) y el jardín en filas escalonadas;
    - a lo ancho (a lo largo de la calle): la franja de la casa (la casa y 2 m a cada lado) y, a sus lados, franjas que
      bajan con la calle, para que dos lotes vecinos se encuentren cerca del terreno natural y no con un muro alto.
    La terraza de la casa va a la cota que menos tierra mueve, sin alejarse más de 1.5 m de la calle (rampa lateral
    de ≤ 15 %); las demás, a la cota media de su terreno. Los muros van donde dos terrazas, o una terraza y la calle o
    el terreno de al lado, quedan a distinta altura. Devuelve por lote las terrazas, los muros y las cantidades."""
    from shapely import STRtree
    from shapely.ops import nearest_points

    from .modelo3d import huella_casa

    calles = unary_union(lot.calles) if lot.calles else None
    previo = []
    for l in lot.lotes:
        esq = huella_casa(l, calles, lot.angulo)
        if esq is None:
            continue
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = esq
        vx, vy = x3 - x0, y3 - y0
        prof = math.hypot(vx, vy) or 1
        vx, vy = vx / prof, vy / prof
        ux, uy = vy, -vx
        base = ((x0 + x1) / 2, (y0 + y1) / 2)     # frente de la casa
        casa = Polygon(esq)
        poly = l["poly"]
        largo = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) * 2
        coords = list(poly.exterior.coords)
        prof_lote = max((q[0] - base[0]) * vx + (q[1] - base[1]) * vy for q in coords)
        ws = [(q[0] - base[0]) * ux + (q[1] - base[1]) * uy for q in coords]
        wc = [(q[0] - base[0]) * ux + (q[1] - base[1]) * uy for q in esq]
        wa, wb = min(wc) - 2.0, max(wc) + 2.0           # franja de la casa
        wmin, wmax = min(ws) - 1, max(ws) + 1
        if wa - wmin < 4.0:                              # franjas laterales angostas: se quedan con la casa
            wa = wmin
        if wmax - wb < 4.0:
            wb = wmax
        corte_casa = prof + 3.0
        # filas a lo hondo
        filas = [(-largo, corte_casa)]
        if prof_lote > corte_casa + 4:
            resto = _banda(poly, base, vx, vy, corte_casa, prof_lote + 1, largo)
            if not resto.is_empty and resto.area > 25:
                zs_r = _muestras(malla, resto, n=150)[2]
                caida = float(np.nanmax(zs_r) - np.nanmin(zs_r))
                k = int(min(12, max(1, math.ceil(caida / ESCALON_JARDIN)),
                            max(1, (prof_lote + 1 - corte_casa) // ANCHO_MIN_TERRAZA)))
                paso = (prof_lote + 1 - corte_casa) / k
                filas += [(corte_casa + i * paso, corte_casa + (i + 1) * paso) for i in range(k)]
            else:
                filas[0] = (-largo, prof_lote + 1)
        else:
            filas[0] = (-largo, prof_lote + 1)
        # columnas a lo ancho: izquierda (partida según su caída), casa, derecha
        izq = _cortes_ancho(malla, poly, base, vx, vy, -largo, corte_casa, wmin, wa, largo) if wa > wmin else [wmin]
        der = _cortes_ancho(malla, poly, base, vx, vy, -largo, corte_casa, wb, wmax, largo) if wmax > wb else [wmax]
        columnas = list(zip(izq, izq[1:])) + [(wa, wb)] + list(zip(der, der[1:]))
        i_casa = len(izq) - 1
        if calles is not None:
            pc, _ = nearest_points(calles, casa)
            z_calle = malla.altura(pc.x, pc.y)
        else:
            z_calle = float(np.nanmedian(_muestras(malla, poly)[2]))
        lista = []
        for r_, (s0, s1) in enumerate(filas):
            fila = []
            for c_, (w0, w1) in enumerate(columnas):
                g = _banda(poly, base, vx, vy, s0, s1, largo, w0, w1)
                if g.is_empty or g.area < 2.0:
                    continue
                zn = _muestras(malla, g, n=80)[2]
                if not len(zn):
                    continue
                z = float(np.nanmedian(zn))
                es_casa = r_ == 0 and c_ == i_casa
                if es_casa:
                    z = min(max(z, z_calle - RAMPA_MAX), z_calle + RAMPA_MAX)
                tz = {"poly": g, "z": round(z, 2), "uso": "casa" if es_casa else ("jardín" if r_ else "lateral")}
                if fila and abs(fila[-1]["z"] - tz["z"]) < 0.3 and "casa" not in (fila[-1]["uso"], tz["uso"]):
                    a_, b_ = fila[-1]["poly"].area, g.area   # casi al mismo nivel: una sola terraza
                    fila[-1] = {**fila[-1], "poly": unary_union([fila[-1]["poly"], g]).buffer(0),
                                "z": round((fila[-1]["z"] * a_ + tz["z"] * b_) / (a_ + b_), 2)}
                else:
                    fila.append(tz)
            lista += fila
        t1 = next((tz for tz in lista if tz["uso"] == "casa"), None)
        if t1 is None:
            continue
        lista.remove(t1)
        lista.insert(0, t1)
        previo.append({"id": l["id"], "casa": casa, "z": t1["z"], "z_calle": round(float(z_calle), 2), "poly": t1["poly"],
                       "terrazas": lista, "eje": (vy, -vx, vx, vy), "lote": poly})
    # muros: se recorre el borde de cada terraza y se compara con lo que hay del otro lado
    todas = [(p["id"], k, tz) for p in previo for k, tz in enumerate(p["terrazas"])]
    arbol = STRtree([tz["poly"].buffer(0.05) for _, _, tz in todas])
    calles_b = calles.buffer(0.05) if calles is not None else None

    def nivel_en(x, y, yo):
        q = Point(x, y)
        for i_ in arbol.query(q, predicate="intersects"):
            i, k, tz = todas[int(i_)]
            if (i, k) != yo:
                return tz["z"], (i, k)
        if calles_b is not None and calles_b.contains(q):
            return malla.altura(x, y), "calle"
        return malla.altura(x, y), "terreno"

    for p in previo:
        muros = []
        for k, tz in enumerate(p["terrazas"]):
            g = tz["poly"]
            for pieza in _partes(g):
                borde = LineString(pieza.exterior.coords)
                n = max(4, int(borde.length / 1.0))
                for j in range(n):
                    q0 = borde.interpolate(j / n, normalized=True)
                    q1 = borde.interpolate(min(1.0, (j + 0.5) / n), normalized=True)
                    tx, ty = q1.x - q0.x, q1.y - q0.y
                    tl = math.hypot(tx, ty) or 1
                    nx, ny = ty / tl, -tx / tl
                    if pieza.contains(Point(q0.x + nx * 0.6, q0.y + ny * 0.6)):
                        nx, ny = -nx, -ny
                    zo, quien = nivel_en(q0.x + nx * 0.6, q0.y + ny * 0.6, (p["id"], k))
                    if not math.isfinite(zo):
                        continue
                    h = tz["z"] - zo
                    if abs(h) < 0.3:
                        continue
                    if isinstance(quien, tuple) and (quien[0], quien[1]) < (p["id"], k):
                        continue  # el muro entre dos terrazas se cuenta una vez
                    tipo = ("calle" if quien == "calle" else "lindero" if quien == "terreno" else
                            "entre terrazas" if quien[0] == p["id"] else "entre lotes")
                    muros.append({"x": q0.x, "y": q0.y, "h": round(abs(h), 2), "tipo": tipo, "largo": borde.length / n})
        corte = relleno = 0.0
        cmax = rmax = 0.0
        for tz in p["terrazas"]:
            xs, ys, zs, _ = _muestras(malla, tz["poly"], n=150)
            if not len(zs):
                continue
            celda = tz["poly"].area / max(1, len(zs))
            dz = zs - tz["z"]
            corte += float(np.sum(np.clip(dz, 0, None)) * celda)
            relleno += float(np.sum(np.clip(-dz, 0, None)) * celda)
            cmax = max(cmax, float(dz.max()))
            rmax = max(rmax, float(-dz.min()))
        cara = sum(m["h"] * m["largo"] for m in muros)
        p.update({"muros": muros, "corte_m3": round(corte), "relleno_m3": round(relleno),
                  "corte_max": round(max(0.0, cmax), 1), "relleno_max": round(max(0.0, rmax), 1),
                  "area_plana": round(sum(tz["poly"].area for tz in p["terrazas"])),
                  "muro_m2": round(cara), "muro_max": round(max((m["h"] for m in muros), default=0.0), 1),
                  "muro_largo": round(sum(m["largo"] for m in muros)),
                  "muro_estructural_m": round(sum(m["largo"] for m in muros if m["h"] > MURO_ESTRUCTURAL)),
                  "rampa": round(p["z"] - p["z_calle"], 2)})
    return previo


def comparar_terraceria(lot, malla: Malla) -> dict | None:
    """Las dos maneras de acomodar las casas en la ladera, para todo el proyecto:
    - taludes: plataforma de la casa a la cota de la calle y taludes de tierra 1.5:1 hasta el terreno natural
      (el resto del lote queda con su caída);
    - muros: terrazas que siguen el terreno con muros de contención (altimetria.terrazas).
    Da área plana, tierra que se mueve, terreno que se comen los taludes y metros de muro."""
    try:
        pads = plataformas(lot, malla)
        terr = terrazas(lot, malla)
    except Exception:  # noqa: BLE001
        return None
    if not pads or not terr:
        return None
    calles = unary_union(lot.calles).buffer(0.5) if lot.calles else None
    talud_m2 = 0.0
    for p in pads:  # huella de los taludes: ancho 1.5 × altura en cada metro del borde que no da a la calle
        borde = LineString(p["poly"].exterior.coords).segmentize(1.0)
        cs = list(borde.coords)
        xs = [(a[0] + b[0]) / 2 for a, b in zip(cs, cs[1:])]
        ys = [(a[1] + b[1]) / 2 for a, b in zip(cs, cs[1:])]
        zs = [malla.altura(x, y) for x, y in zip(xs, ys)]
        for (a, b), x, y, z in zip(zip(cs, cs[1:]), xs, ys, zs):
            if not math.isfinite(z) or (calles is not None and calles.contains(Point(x, y))):
                continue
            talud_m2 += 1.5 * abs(p["z"] - float(z)) * math.hypot(b[0] - a[0], b[1] - a[1])
    area_lotes = sum(l["poly"].area for l in lot.lotes)
    t = {"area_plana": round(sum(p["poly"].area for p in pads)), "corte_m3": sum(p["corte_m3"] for p in pads),
         "relleno_m3": sum(p["relleno_m3"] for p in pads), "talud_m2": round(talud_m2),
         "corte_max": max(p["corte_max"] for p in pads), "relleno_max": max(p["relleno_max"] for p in pads)}
    m = {"area_plana": sum(p["area_plana"] for p in terr), "corte_m3": sum(p["corte_m3"] for p in terr),
         "relleno_m3": sum(p["relleno_m3"] for p in terr), "muro_m2": sum(p["muro_m2"] for p in terr),
         "muro_largo": sum(p["muro_largo"] for p in terr), "muro_max": max(p["muro_max"] for p in terr),
         "muro_estructural_m": sum(p["muro_estructural_m"] for p in terr),
         "lotes_muro_alto": [p["id"] for p in terr if p["muro_max"] > MURO_ESTRUCTURAL]}
    return {"area_lotes": round(area_lotes), "taludes": t, "muros": m}
