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


def barrancos(malla: Malla, anillo, umbral: float = PEND_BARRANCO, area_min: float = 60.0, tope: float = 0.20) -> list[Polygon]:
    """Caídas fuertes que no se lotifican: zonas con más de `umbral` % pegadas al borde del terreno (el barranco de una
    quebrada, un talud). Se suavizan 2 m. No se toman las que partirían el terreno en pedazos ni más del `tope` del
    área: en un terreno empinado de punta a punta las pendientes fuertes son parte de la ladera (se resuelven con
    terrazas y lotes más grandes), no un barranco."""
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
        if not piezas or piezas[0].area < 0.97 * resto.area:  # dejaría pedazos sueltos
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
