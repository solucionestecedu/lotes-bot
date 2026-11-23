"""Prediseño automático de lotes (anteproyecto).

Idea: se prueban 36 orientaciones de calles (cada 5°) y varias posiciones de la calle principal;
se cuentan lotes y se penaliza ir contra la pendiente. Con la mejor opción se trazan manzanas de
doble fondo con calle al centro, lotes de frente fijo, y el área verde se completa convirtiendo
los lotes más inclinados. Todo en metros (Lambert).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

from .relieve import Malla


class ErrorLotificacion(ValueError):
    pass


@dataclass
class Parametros:
    calle: float = 8.0          # ancho de calle (m)
    frente: float = 10.0        # frente de lote (m)
    fondo: float = 25.0         # fondo de lote (m)
    area_minima: float = 200.0  # m²
    regla_verde: str = "pct_util"   # pct_util | pct_total | por_lote
    valor_verde: float = 10.0       # % o m² por lote
    angulo: float | None = None     # grados; None = automático
    entrada: tuple[float, float] | None = None  # punto (x, y) de acceso
    verde_minimo: float = 100.0     # remanente mínimo para contar como área verde
    pct_recreativa: float = 60.0    # parte del área verde que debe ser recreativa (RLUC Art. 59)
    equipamiento_m2: float = 0.0    # terreno para equipamiento social (escuela, casa comunal)

    @classmethod
    def de_dict(cls, d: dict | None) -> "Parametros":
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, tuple(v) if k == "entrada" else (v if k == "regla_verde" else float(v)))
        return p


def area_verde_requerida(regla: str, valor: float, n_lotes: int, area_util: float, area_total: float) -> float:
    if regla == "por_lote":
        return valor * n_lotes
    if regla == "pct_total":
        return valor / 100 * area_total
    return valor / 100 * area_util


def _poligonos(g) -> list[Polygon]:
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon) and not p.is_empty]


def _mayor(g) -> Polygon | None:
    ps = _poligonos(g)
    return max(ps, key=lambda p: p.area) if ps else None


def _barrido(anillo, y):
    xs = []
    n = len(anillo)
    for i in range(n):
        a, b = anillo[i], anillo[(i + 1) % n]
        if (a[1] <= y) != (b[1] <= y):
            xs.append(a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]))
    xs.sort()
    return [(xs[k], xs[k + 1]) for k in range(0, len(xs) - 1, 2)]


def _inter_iv(A, B):
    return [(max(a[0], b[0]), min(a[1], b[1])) for a in A for b in B if min(a[1], b[1]) > max(a[0], b[0])]


def _restar_iv(A, l, r):
    out = []
    for a in A:
        if a[1] <= l or a[0] >= r:
            out.append(a)
        else:
            if a[0] < l:
                out.append((a[0], l))
            if a[1] > r:
                out.append((r, a[1]))
    return out


@dataclass
class _Lote:
    poly: Polygon
    area: float
    frente: float
    fila: int
    x: float
    pendiente: float = math.nan


@dataclass
class Resultado:
    angulo: float
    lotes: list[dict]
    calles: list[Polygon]
    verdes: list[dict]
    remanentes: list[Polygon]
    stats: dict
    parametros: dict = field(default_factory=dict)
    protecciones: list[Polygon] = field(default_factory=list)
    equipamiento: list[dict] = field(default_factory=list)


LADOS = {"norte": 0, "noreste": 45, "este": 90, "oriente": 90, "sureste": 135, "sur": 180, "suroeste": 225,
         "oeste": 270, "poniente": 270, "noroeste": 315}


def _bordes_hacia(poly: Polygon, lado: str, tolerancia: float = 50.0):
    """Tramos del lindero cuya cara exterior mira hacia `lado` (norte, sur, oriente...)."""
    from shapely.geometry.polygon import orient

    objetivo = LADOS[lado.strip().lower()]
    pts = list(orient(poly, 1.0).exterior.coords)  # antihorario
    out = []
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if math.hypot(dx, dy) < 0.5:
            continue
        az = math.degrees(math.atan2(dy, -dx)) % 360  # normal exterior (dy, -dx) → azimut
        if abs((az - objetivo + 180) % 360 - 180) <= tolerancia:
            out.append((a, b))
    return out


def punto_de_acceso(anillo, lado: str) -> tuple[float, float] | None:
    """Mitad del tramo más largo del lindero que da hacia `lado`."""
    bordes = _bordes_hacia(Polygon(anillo).buffer(0), lado)
    if not bordes:
        return None
    a, b = max(bordes, key=lambda e: math.dist(*e))
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def franja_de_proteccion(anillo, lado: str, ancho: float) -> Polygon | None:
    """Franja de `ancho` metros hacia adentro del terreno, a lo largo del lindero que da hacia `lado`."""
    from shapely.geometry import LineString

    poly = Polygon(anillo).buffer(0)
    bordes = _bordes_hacia(poly, lado)
    if not bordes or ancho <= 0:
        return None
    franja = unary_union([LineString(e).buffer(ancho, cap_style=2) for e in bordes]).intersection(poly)
    return _mayor(franja) if franja.geom_type != "Polygon" else franja


def lotificar(anillo, P: Parametros | None = None, malla: Malla | None = None, excluir: list | None = None) -> Resultado:
    P = P or Parametros()
    W, F, D = P.calle, P.frente, P.fondo
    M = 2 * D + W
    terreno_total = Polygon(anillo).buffer(0)
    if not isinstance(terreno_total, Polygon):
        terreno_total = _mayor(terreno_total)
    protecciones, piezas_sueltas = [], []
    terreno = terreno_total
    if excluir:
        zona = unary_union([e for e in excluir if e is not None and not e.is_empty]).intersection(terreno_total)
        protecciones = _poligonos(zona)
        piezas = sorted(_poligonos(terreno_total.difference(zona)), key=lambda p: -p.area)
        terreno = piezas[0] if piezas else None
        piezas_sueltas = piezas[1:]
    if terreno is None or terreno.area < P.area_minima:
        raise ErrorLotificacion("El terreno es más chico que un lote.")
    C = terreno.centroid

    # Pendientes dentro del terreno (fracción, no %).
    dentro = None
    if malla is not None:
        gx, gy = malla.gradiente()
        pr = prep(terreno)
        dentro = np.array([(gx[j, i], gy[j, i]) for j in range(malla.ny) for i in range(malla.nx)
                           if np.isfinite(gx[j, i]) and pr.contains(Point(malla.x0 + i * malla.dx, malla.y0 + j * malla.dx))])
        if not len(dentro):
            dentro = None

    def pendiente_dir(th):
        if dentro is None:
            return 0.0
        return float(np.mean(np.abs(dentro[:, 0] * math.cos(th) + dentro[:, 1] * math.sin(th))))

    def pendiente_en(x, y):
        if malla is None:
            return math.nan
        h = 0.5 * malla.dx
        dzdx = (malla.altura(x + h, y) - malla.altura(x - h, y)) / (2 * h)
        dzdy = (malla.altura(x, y + h) - malla.altura(x, y - h)) / (2 * h)
        return math.hypot(dzdx, dzdy)

    def rotar(g, deg):  # al sistema donde las calles son horizontales
        return affinity.rotate(g, -deg, origin=C)

    def desrotar(g, deg):
        return affinity.rotate(g, deg, origin=C)

    def conteo(R, bb, o, xs):
        """Conteo rápido con líneas de barrido: tramos que cubren todo el fondo del lote."""
        anillo_r = list(R.exterior.coords)[:-1]
        n = 0
        s = bb[1] + o - M
        while s - D < bb[3]:
            for y0, y1 in ((s - D, s), (s + W, s + W + D)):
                if y1 <= bb[1] or y0 >= bb[3]:
                    continue
                iv = _inter_iv(_barrido(anillo_r, y0 + 0.05), _barrido(anillo_r, y1 - 0.05))
                iv = _restar_iv(iv, xs - W / 2, xs + W / 2)
                n += sum(int((r - l) / F + 1e-9) for l, r in iv)
            s += M
        return n

    # ---- búsqueda de orientación
    angulos = [P.angulo] if P.angulo is not None else [k * 5.0 for k in range(36)]
    mejor = None
    for deg in angulos:
        th = math.radians(deg)
        R = rotar(terreno, deg)
        bb = R.bounds
        sl = pendiente_dir(th)
        factor = 1 - min(0.6, max(0.0, sl - 0.08) * 3)
        offs = [f * M for f in (0, 0.2, 0.4, 0.6, 0.8)]
        cands = []
        if P.entrada:
            e = rotar(Point(P.entrada), deg)
            cands += [(o, e.x) for o in offs]
            oE = ((e.y - W / 2 - bb[1]) % M + M) % M
            cands += [(oE, bb[0] + f * (bb[2] - bb[0])) for f in (0.25, 0.5, 0.75)]
        else:
            cands = [(o, bb[0] + f * (bb[2] - bb[0])) for f in (0.25, 0.5, 0.75) for o in offs]
        for o, xs in cands:
            n = conteo(R, bb, o, xs)
            score = n * factor
            if mejor is None or score > mejor["score"] + 1e-9:
                mejor = {"deg": deg, "o": o, "xs": xs, "n": n, "score": score, "pend": sl}

    # ---- trazo con la mejor combinación
    deg = mejor["deg"]
    R = rotar(terreno, deg)
    bb = R.bounds
    xs = mejor["xs"]
    entrada_r = rotar(Point(P.entrada), deg) if P.entrada else None
    calles, lotes, ys_calle = [], [], []
    fila = 0
    s = bb[1] + mejor["o"] - M
    while s - D < bb[3]:
        franja_calle = R.intersection(box(bb[0] - 5, s, bb[2] + 5, s + W))
        en_banda = []
        for y0, y1, abajo in ((s - D, s, True), (s + W, s + W + D, False)):
            fila += 1
            if y1 <= bb[1] or y0 >= bb[3]:
                continue
            for lado in (-1, 1):
                x = xs + lado * W / 2
                prev = None
                while (x < bb[2]) if lado > 0 else (x > bb[0]):
                    xa, xb = x, x + lado * F
                    xl, xr = min(xa, xb), max(xa, xb)
                    x = xb
                    pieza = _mayor(R.intersection(box(xl, y0, xr, y1)))
                    if pieza is None:
                        prev = None
                        continue
                    a = pieza.area
                    ft = 0.3
                    fy = y1 - ft if abajo else y0
                    frente = pieza.intersection(box(xl - 0.01, fy, xr + 0.01, fy + ft)).area / ft
                    if a >= P.area_minima and frente >= 0.6 * F:
                        prev = _Lote(pieza, a, frente, fila, (xl + xr) / 2)
                        en_banda.append(prev)
                    elif prev is not None and 0.5 < a < 0.6 * P.area_minima and frente > 0.5:
                        u = unary_union([prev.poly, pieza])
                        if isinstance(u, Polygon):
                            prev.poly, prev.area, prev.frente = u, u.area, prev.frente + frente
                        else:
                            prev = None
                    else:
                        prev = None
        if en_banda and not franja_calle.is_empty:
            calles += _poligonos(franja_calle)
            lotes += en_banda
            ys_calle.append(s)
        s += M
    if not lotes:
        raise ErrorLotificacion("No cupo ningún lote con estas medidas. Prueba lotes más chicos o calles más angostas.")

    # calle principal (une las secundarias y sale al lindero)
    y_lo, y_hi = min(ys_calle), max(ys_calle) + W
    if entrada_r is not None:
        if entrada_r.y < y_lo:
            y_lo = bb[1] - 5
        if entrada_r.y > y_hi:
            y_hi = bb[3] + 5
    else:
        y_lo = bb[1] - 5
    espina = R.intersection(box(xs - W / 2, y_lo, xs + W / 2, y_hi))
    circ = unary_union(calles + _poligonos(espina))
    # los lotes no pueden pisar la calle principal
    for l in lotes:
        l.poly = _mayor(l.poly.difference(circ)) or l.poly
        l.area = l.poly.area
    lotes = [l for l in lotes if l.area >= 0.95 * P.area_minima]
    if not lotes:
        raise ErrorLotificacion("No cupo ningún lote con estas medidas. Prueba lotes más chicos o calles más angostas.")

    resto = R.difference(unary_union([circ] + [l.poly for l in lotes]))
    verdes, remanentes = [], []
    for p in _poligonos(resto):
        if p.area >= P.verde_minimo:
            verdes.append({"poly": p, "area": p.area, "de_lote": False})
        elif p.area > 0.01:
            remanentes.append(p)

    for l in lotes:
        c = desrotar(l.poly.centroid, deg)
        l.pendiente = pendiente_en(c.x, c.y)

    area_total = terreno_total.area
    area_prot = sum(p.area for p in protecciones)
    for p in piezas_sueltas:
        (verdes.append({"poly": p, "area": p.area, "de_lote": False}) if p.area >= P.verde_minimo else remanentes.append(p))

    def prioridad(l):
        if math.isfinite(l.pendiente):
            return l.pendiente
        return abs(l.area - F * D) / (F * D) + abs(l.x - xs) / 1e6

    convertidos = 0
    while lotes:
        util = sum(l.area for l in lotes)
        req = area_verde_requerida(P.regla_verde, P.valor_verde, len(lotes), util, area_total)
        meta = max(req - area_prot, req * P.pct_recreativa / 100)  # la protección cuenta como área verde ecológica
        if sum(v["area"] for v in verdes) >= meta - 1e-6:
            break
        k = max(range(len(lotes)), key=lambda i: prioridad(lotes[i]))
        l = lotes.pop(k)
        verdes.append({"poly": l.poly, "area": l.area, "de_lote": True})
        convertidos += 1

    # equipamiento social: los lotes más planos que queden
    equip = []
    while lotes and P.equipamiento_m2 > 0 and sum(e.area for e in equip) < P.equipamiento_m2 - 1e-6:
        k = min(range(len(lotes)), key=lambda i: lotes[i].pendiente if math.isfinite(lotes[i].pendiente) else abs(lotes[i].x - xs))
        equip.append(lotes.pop(k))

    lotes.sort(key=lambda l: (l.fila, l.x))
    salida = []
    for i, l in enumerate(lotes, start=1):
        salida.append({"id": i, "poly": desrotar(l.poly, deg), "area": l.area, "frente": l.frente,
                       "pendiente": l.pendiente * 100 if math.isfinite(l.pendiente) else None})
    area_util = sum(l["area"] for l in salida)
    area_verde = sum(v["area"] for v in verdes)
    area_circ = circ.area
    areas = [l["area"] for l in salida]
    req = area_verde_requerida(P.regla_verde, P.valor_verde, len(salida), area_util, area_total)
    if not salida:
        raise ErrorLotificacion("Después de dejar área verde y equipamiento no quedaron lotes. Revisa las medidas.")
    return Resultado(
        angulo=deg,
        lotes=salida,
        calles=[desrotar(p, deg) for p in _poligonos(circ)],
        verdes=[{"poly": desrotar(v["poly"], deg), "area": v["area"], "de_lote": v["de_lote"]} for v in verdes],
        remanentes=[desrotar(p, deg) for p in remanentes],
        stats={
            "area_total": area_total, "area_util": area_util, "area_circulacion": area_circ,
            "area_verde": area_verde, "area_verde_requerida": req, "area_remanente": sum(p.area for p in remanentes),
            "lotes": len(salida), "lotes_convertidos_a_verde": convertidos,
            "largo_calles": area_circ / W, "lote_promedio": area_util / len(salida),
            "lote_min": min(areas), "lote_max": max(areas), "pendiente_calles_pct": mejor["pend"] * 100,
            "area_proteccion": area_prot, "area_equipamiento": sum(e.area for e in equip),
            "calles_sin_salida": P.entrada is not None,
        },
        parametros={"calle": W, "frente": F, "fondo": D, "area_minima": P.area_minima,
                    "regla_verde": P.regla_verde, "valor_verde": P.valor_verde},
        protecciones=protecciones,
        equipamiento=[{"poly": desrotar(e.poly, deg), "area": e.area} for e in equip],
    )


def anillos(poly: Polygon) -> list[list[tuple[float, float]]]:
    return [list(poly.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in poly.interiors]
