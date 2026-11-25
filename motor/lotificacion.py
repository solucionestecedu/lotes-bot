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
    lado_acceso: str | None = None  # lado del lindero que da a la calle pública (norte, sur, oriente...)
    verde_minimo: float = 100.0     # remanente mínimo para contar como área verde
    pct_recreativa: float = 60.0    # parte del área verde que debe ser recreativa (RLUC Art. 59)
    equipamiento_m2: float = 0.0    # terreno para equipamiento social (escuela, casa comunal)
    aprovechar: bool = True         # el área verde que sobra se vuelve lote o se suma a un lote vecino

    @classmethod
    def de_dict(cls, d: dict | None) -> "Parametros":
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None:
                if k == "entrada":
                    setattr(p, k, tuple(v))
                elif k in ("regla_verde", "lado_acceso"):
                    setattr(p, k, v)
                elif k == "aprovechar":
                    setattr(p, k, bool(v))
                else:
                    setattr(p, k, float(v))
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


def _frente(poly, y0, y1, abajo, x0, x1):
    ft = 0.3
    fy = y1 - ft if abajo else y0
    return poly.intersection(box(x0 - 0.01, fy, x1 + 0.01, fy + ft)).area / ft


def _cortar_por_area(zona, y0, y1, abajo, meta, minima, frente_min, fila):
    """Corta una franja (entre la calle y el fondo) en lotes de área parecida a `meta`, de izquierda a derecha."""
    minx, _, maxx, _ = zona.bounds
    if zona.area < minima:
        return []
    n = max(1, int(zona.area / meta + 0.35))  # cuántos lotes caben con área cercana a la meta
    objetivo = zona.area / n
    cortes = [minx]
    for _ in range(n - 1):
        lo, hi = cortes[-1], maxx
        for _ in range(30):  # bisección: dónde cortar para juntar el área objetivo
            mid = (lo + hi) / 2
            if zona.intersection(box(cortes[-1], y0 - 1, mid, y1 + 1)).area < objetivo:
                lo = mid
            else:
                hi = mid
        cortes.append(hi)
    cortes.append(maxx)
    out = []
    for a, b in zip(cortes, cortes[1:]):
        pieza = _mayor(zona.intersection(box(a, y0 - 1, b, y1 + 1)))
        if pieza is None:
            continue
        fr = _frente(pieza, y0, y1, abajo, a, b)
        if pieza.area >= minima and fr >= frente_min:
            out.append(_Lote(pieza, pieza.area, fr, fila, (a + b) / 2))
        elif out:  # muy chico o sin frente: se suma al lote anterior
            u = unary_union([out[-1].poly, pieza])
            if isinstance(u, Polygon):
                out[-1].poly, out[-1].area = u, u.area
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


def orilla_hacia(poly: Polygon, lado: str, tolerancia: float = 50.0, puente: float = 25.0):
    """Orilla de la quebrada: la cadena del lindero que mira hacia `lado`, sin cortes. Además de los tramos que miran
    hacia allá entran los que quedan entre dos de ellos si son cortos o no miran al revés: la quebrada sigue su curso
    aunque el lindero quiebre. Es la única fuente para la franja de protección, el cauce dibujado y el 3D."""
    from shapely.geometry import LineString
    from shapely.geometry.polygon import orient
    from shapely.ops import linemerge

    objetivo = LADOS[lado.strip().lower()]
    pts = list(orient(poly, 1.0).exterior.coords)
    tramos = [(a, b) for a, b in zip(pts, pts[1:]) if math.dist(a, b) >= 0.05]
    n = len(tramos)
    if not n:
        return None

    def desvio(a, b):
        az = math.degrees(math.atan2(b[1] - a[1], -(b[0] - a[0]))) % 360
        return abs((az - objetivo + 180) % 360 - 180)

    ok = [desvio(a, b) <= tolerancia for a, b in tramos]
    if not any(ok):
        return None
    cambio = True
    while cambio:
        cambio = False
        for i, (a, b) in enumerate(tramos):
            if not ok[i] and ok[(i - 1) % n] and ok[(i + 1) % n] and (math.dist(a, b) <= puente or desvio(a, b) <= 90):
                ok[i] = cambio = True
    orilla = unary_union([LineString(tramos[i]) for i in range(n) if ok[i]])
    return orilla if orilla.geom_type == "LineString" else linemerge(orilla)  # un solo tramo recto: ya es la orilla


def franja_de_proteccion(anillo, lado: str, ancho: float):
    """Franja de `ancho` metros hacia adentro del terreno, a lo largo de toda la orilla de la quebrada."""
    poly = Polygon(anillo).buffer(0)
    orilla = orilla_hacia(poly, lado)
    if orilla is None or orilla.is_empty or ancho <= 0:
        return None
    franja = orilla.buffer(ancho, cap_style=2).intersection(poly).buffer(0)
    if franja.is_empty:
        return None
    if franja.geom_type != "Polygon":  # se queda todo lo que toca la orilla (antes solo el pedazo más grande)
        piezas = [g for g in getattr(franja, "geoms", [franja]) if g.geom_type == "Polygon" and g.area > 1.0]
        franja = unary_union(piezas)
    return franja


def cauce_quebrada(anillo, ficha: dict | None, separacion: float = 7.0):
    """Línea del cauce de la quebrada: el lindero del lado de la quebrada corrido `separacion` m hacia afuera.
    None si la ficha no dice de qué lado está."""
    from shapely.geometry import LineString
    from shapely.ops import linemerge

    lado = next((str(q.get("lado", "")).lower() for q in (ficha or {}).get("quebradas") or []
                 if str(q.get("lado", "")).lower() in LADOS), None)
    if not lado:
        return None
    poly = Polygon(anillo).buffer(0)
    orilla = orilla_hacia(poly, lado)
    if orilla is None or orilla.is_empty:
        return None
    mejor = None
    for g in getattr(orilla, "geoms", [orilla]):
        try:
            o1, o2 = g.offset_curve(separacion), g.offset_curve(-separacion)
        except Exception:  # noqa: BLE001
            continue
        o = o1 if poly.distance(o1) > poly.distance(o2) else o2
        if mejor is None or o.length > mejor.length:
            mejor = o
    return mejor


def proteccion_de_quebrada(protecciones: list, anillo, ficha: dict | None) -> Polygon | None:
    """De las zonas no lotificadas, la que es la franja de la quebrada: la que más lindero comparte con el lado de la
    quebrada (las demás son caídas fuertes). Sin lado conocido, la más grande."""
    if not protecciones:
        return None
    lados = [q.get("lado") for q in (ficha or {}).get("quebradas") or [] if q.get("lado")]
    if lados:
        from shapely.geometry import LineString

        poly = Polygon(anillo).buffer(0)
        bordes = [LineString(e) for lado in lados for e in _bordes_hacia(poly, lado)]
        if bordes:
            borde = unary_union(bordes).buffer(0.5)
            mejor = max(protecciones, key=lambda g: g.intersection(borde).area)
            if mejor.intersection(borde).area > 0:
                return mejor
    return max(protecciones, key=lambda g: g.area)


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

    # lindero que da a la calle pública: las calles deben llegar ahí (o al punto de entrada)
    acceso_geom = None
    if P.lado_acceso and P.lado_acceso.lower() in LADOS:
        from shapely.geometry import LineString, MultiLineString

        bordes = _bordes_hacia(terreno_total, P.lado_acceso, tolerancia=35)
        if bordes:
            acceso_geom = MultiLineString([LineString(e) for e in bordes])

    def conectividad(circ, R_deg):
        """(piezas de calle sin salida a la calle pública, número de piezas que tocan la calle pública)."""
        piezas = _poligonos(circ)
        if not piezas or (acceso_geom is None and not P.entrada):
            return 0, 0
        destinos = []
        if acceso_geom is not None:
            destinos.append(rotar(acceso_geom, R_deg))
        if P.entrada:
            destinos.append(rotar(Point(P.entrada), R_deg))
        con = {i for i, g in enumerate(piezas) if any(g.distance(d) < 1.0 for d in destinos)}
        directas = len(con)
        cambio = True
        while cambio:
            cambio = False
            for i, g in enumerate(piezas):
                if i not in con and any(g.distance(piezas[j]) < 0.5 for j in con):
                    con.add(i)
                    cambio = True
        return len(piezas) - len(con), directas

    # ---- búsqueda: orientación, posición de calles y de la calle principal
    angulos = [P.angulo] if P.angulo is not None else [k * 5.0 for k in range(36)]
    candidatos = []
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
            cands += [(o, e.x, False) for o in offs]
            oE = ((e.y - W / 2 - bb[1]) % M + M) % M
            cands += [(oE, bb[0] + f * (bb[2] - bb[0]), False) for f in (0.25, 0.5, 0.75)]
            cands.append((oE, bb[0] - 2 * W, True))  # una sola calle que entra desde el acceso (sin calle principal)
        else:
            cands = [(o, bb[0] + f * (bb[2] - bb[0]), False) for f in (0.25, 0.5, 0.75) for o in offs]
        for o, xs, unica in cands:
            n = conteo(R, bb, o, xs)
            candidatos.append({"deg": deg, "o": o, "xs": xs, "unica": unica, "score": n * factor, "pend": sl,
                               "factor": factor})

    def trazar(c):
        deg, xs = c["deg"], c["xs"]
        R = rotar(terreno, deg)
        bb = R.bounds
        entrada_r = rotar(Point(P.entrada), deg) if P.entrada else None
        calles, lotes, ys_calle = [], [], []
        # bandas de calle que tocan el terreno
        bandas = []
        s = bb[1] + c["o"] - M
        while s - D < bb[3]:
            if not (c["unica"] and not (s - 0.5 <= entrada_r.y <= s + W + 0.5)):
                if not R.intersection(box(bb[0] - 5, s, bb[2] + 5, s + W)).is_empty:
                    bandas.append(s)
            s += M
        meta = max(P.area_minima, F * D)  # área objetivo de cada lote
        espina_x = None if c["unica"] else (xs - W / 2, xs + W / 2)
        fila = 0
        for k, s in enumerate(bandas):
            franja_calle = R.intersection(box(bb[0] - 5, s, bb[2] + 5, s + W))
            en_banda = []
            # la primera y la última banda se estiran hacia el lindero para no dejar tiras sueltas
            abajo_y0 = s - (1.6 * D if k == 0 else D)
            arriba_y1 = s + W + (1.6 * D if k == len(bandas) - 1 else D)
            for y0, y1, abajo in ((abajo_y0, s, True), (s + W, arriba_y1, False)):
                fila += 1
                zona = R.intersection(box(bb[0] - 1, y0, bb[2] + 1, y1))
                if espina_x:
                    zona = zona.difference(box(espina_x[0], y0 - 1, espina_x[1], y1 + 1))
                for pieza in sorted(_poligonos(zona), key=lambda g: g.bounds[0]):
                    en_banda += _cortar_por_area(pieza, y0, y1, abajo, meta, P.area_minima, 0.6 * F, fila)
            if en_banda and not franja_calle.is_empty:
                calles += _poligonos(franja_calle)
                lotes += en_banda
                ys_calle.append(s)
        if not lotes:
            return None
        # calle principal (une las calles y sale al lindero)
        y_lo, y_hi = min(ys_calle), max(ys_calle) + W
        if entrada_r is not None:
            if entrada_r.y < y_lo:
                y_lo = bb[1] - 5
            if entrada_r.y > y_hi:
                y_hi = bb[3] + 5
        else:
            y_lo = bb[1] - 5
        espina = R.intersection(box(xs - W / 2, y_lo, xs + W / 2, y_hi)) if not c["unica"] else Polygon()
        circ = unary_union(calles + _poligonos(espina))
        for l in lotes:  # los lotes no pueden pisar la calle principal
            l.poly = _mayor(l.poly.difference(circ)) or l.poly
            l.area = l.poly.area
        lotes = [l for l in lotes if l.area >= 0.95 * P.area_minima]
        if not lotes:
            return None
        sueltas, accesos = conectividad(circ, deg)
        return {"R": R, "lotes": lotes, "circ": circ, "calles": calles, "sueltas": sueltas, "accesos": accesos}

    # se trazan las mejores opciones del conteo rápido y se queda la que da más lotes
    candidatos.sort(key=lambda c: -c["score"])
    mejor, mejor_traz, vistos = None, None, set()
    for c in candidatos:
        clave = (c["deg"], round(c["o"], 1), round(c["xs"], 1), c["unica"])
        if clave in vistos:
            continue
        vistos.add(clave)
        if len(vistos) > 24 and mejor is not None and mejor[0][0] > -50:
            break
        tz = trazar(c)
        if tz is None:
            continue
        # una calle sin salida a la calle pública deja lotes sin acceso: se castiga fuerte
        valor = (len(tz["lotes"]) * c["factor"] - 100 * tz["sueltas"], -tz["circ"].area)
        if mejor is None or valor > mejor[0]:
            mejor, mejor_traz = (valor, c), tz
    if mejor is None:
        raise ErrorLotificacion("No cupo ningún lote con estas medidas. Prueba lotes más chicos o calles más angostas.")
    mejor = mejor[1]
    deg, xs = mejor["deg"], mejor["xs"]
    R, lotes, circ, calles = mejor_traz["R"], mejor_traz["lotes"], mejor_traz["circ"], mejor_traz["calles"]
    calles_sueltas, accesos = mejor_traz["sueltas"], mejor_traz["accesos"]

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

    calle_b = circ.buffer(0.3)
    prot_r = unary_union([rotar(p, deg) for p in protecciones]) if protecciones else None

    def recortar(l, falta, frente_pedazo=3.0):
        """Saca `falta` m² de un lado del lote (corte paralelo al fondo); el pedazo queda con frente a la calle,
        y el resto sigue siendo un lote válido. Devuelve (pedazo, resto) o None."""
        minx, miny, maxx, maxy = l.poly.bounds
        mejor = None
        for lado in (-1, 1):
            lo, hi = minx, maxx
            for _ in range(32):
                mid = (lo + hi) / 2
                caja = box(minx - 1, miny - 1, mid, maxy + 1) if lado < 0 else box(mid, miny - 1, maxx + 1, maxy + 1)
                a = l.poly.intersection(caja).area
                if (a < falta) == (lado < 0):
                    lo = mid
                else:
                    hi = mid
            corte = (lo + hi) / 2
            caja = box(minx - 1, miny - 1, corte, maxy + 1) if lado < 0 else box(corte, miny - 1, maxx + 1, maxy + 1)
            pedazo, resto = _mayor(l.poly.intersection(caja)), _mayor(l.poly.difference(caja))
            if pedazo is None or resto is None or resto.area < P.area_minima:
                continue
            fr_resto = resto.boundary.intersection(calle_b).length
            if fr_resto < 0.6 * F or pedazo.boundary.intersection(calle_b).length < frente_pedazo:
                continue
            junto_quebrada = prot_r is not None and pedazo.distance(prot_r) < 1.0
            clave = (junto_quebrada, -abs(pedazo.area - falta))
            if mejor is None or clave > mejor[0]:
                mejor = (clave, pedazo, resto, fr_resto)
        return None if mejor is None else mejor[1:]

    convertidos, recortados = 0, 0
    while lotes:
        util = sum(l.area for l in lotes)
        req = area_verde_requerida(P.regla_verde, P.valor_verde, len(lotes), util, area_total)
        meta = max(req - area_prot, req * P.pct_recreativa / 100)  # la protección cuenta como área verde ecológica
        falta = meta - sum(v["area"] for v in verdes)
        if falta <= 1e-6:
            break
        # primero: sacar justo lo que falta de un lote grande o inclinado, sin perder el lote
        if P.aprovechar:
            orden = sorted(lotes, key=lambda l: (-(l.pendiente if math.isfinite(l.pendiente) else 0), -l.area))
            hecho = False
            for l in orden:
                if l.area - falta * 1.02 < P.area_minima:
                    continue
                r_ = recortar(l, falta * 1.02)
                if r_:
                    pedazo, resto, fr = r_
                    l.poly, l.area, l.frente = resto, resto.area, fr
                    verdes.append({"poly": pedazo, "area": pedazo.area, "de_lote": True})
                    recortados += 1
                    hecho = True
                    break
            if hecho:
                continue
        # si no se puede recortar: el lote que mejor cubre lo que falta (el más inclinado si hay relieve)
        if P.aprovechar and not any(math.isfinite(l.pendiente) for l in lotes):
            cubren = [i for i in range(len(lotes)) if lotes[i].area >= falta]
            k = min(cubren, key=lambda i: lotes[i].area) if cubren else max(range(len(lotes)), key=lambda i: lotes[i].area)
        else:
            k = max(range(len(lotes)), key=lambda i: prioridad(lotes[i]))
        l = lotes.pop(k)
        verdes.append({"poly": l.poly, "area": l.area, "de_lote": True})
        convertidos += 1

    # ---- máximo aprovechamiento: lo que sobra de área verde se vende (como lote nuevo o sumado a un vecino)
    def meta_verde(lista):
        util = sum(l.area for l in lista)
        req = area_verde_requerida(P.regla_verde, P.valor_verde, len(lista), util, area_total)
        return max(req - area_prot, req * P.pct_recreativa / 100)

    lotes_extra, sumado_m2 = 0, 0.0
    if P.aprovechar and lotes:
        lote_tipo = max(P.area_minima, F * D)
        # primero se sueltan las piezas que menos sirven de parque: lejos de la quebrada y chicas
        sueltas = sorted([v for v in verdes if not v["de_lote"]],
                         key=lambda v: (prot_r is not None and v["poly"].distance(prot_r) < 1.0, v["area"]))
        for v in sueltas:
            pieza = v["poly"]
            resto = sum(x["area"] for x in verdes if x is not v)
            frente_calle = pieza.boundary.intersection(calle_b).length
            nuevo = None
            if P.area_minima <= pieza.area <= 2.5 * lote_tipo and frente_calle >= 0.6 * F:
                c = pieza.centroid
                cw = desrotar(c, deg)
                nuevo = _Lote(pieza, pieza.area, frente_calle, 999, c.x, pendiente_en(cw.x, cw.y))
                if resto >= meta_verde(lotes + [nuevo]) - 1e-6:
                    lotes.append(nuevo)
                    verdes.remove(v)
                    lotes_extra += 1
                    continue
            # si no da para lote propio, se suma al lote vecino con el que comparte más lindero
            vecinos = [(l.poly.boundary.intersection(pieza.buffer(0.1)).length, l) for l in lotes]
            vecinos = [(lg, l) for lg, l in vecinos if lg >= 2.0]
            if not vecinos:
                continue
            _, l = max(vecinos, key=lambda x: x[0])
            u = unary_union([l.poly, pieza])
            if not isinstance(u, Polygon) or u.area > 2.0 * lote_tipo:
                continue
            area_antes = l.area
            l.poly, l.area = u, u.area
            if resto >= meta_verde(lotes) - 1e-6:
                verdes.remove(v)
                sumado_m2 += pieza.area
            else:  # no alcanza el área verde: se deshace
                l.poly = _mayor(l.poly.difference(pieza)) or l.poly
                l.area = area_antes

    # lotes que quedaron del doble de tamaño: se parten en dos si ambos quedan con frente y área mínima
    partidos = 0
    if P.aprovechar and lotes:
        for l in sorted(list(lotes), key=lambda x: -x.area):
            if l.area < max(2 * P.area_minima, 1.6 * lote_tipo):
                continue
            r_ = recortar(l, l.area / 2, frente_pedazo=0.6 * F)
            if not r_:
                continue
            pedazo, resto, fr = r_
            c = pedazo.centroid
            cw = desrotar(c, deg)
            nuevo = _Lote(pedazo, pedazo.area, pedazo.boundary.intersection(calle_b).length, l.fila, c.x,
                          pendiente_en(cw.x, cw.y))
            antes = (l.poly, l.area, l.frente)
            l.poly, l.area, l.frente = resto, resto.area, fr
            if sum(v["area"] for v in verdes) >= meta_verde(lotes + [nuevo]) - 1e-6:
                lotes.append(nuevo)
                partidos += 1
            else:
                l.poly, l.area, l.frente = antes

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
            "calles_desconectadas": calles_sueltas, "accesos": accesos,
            "lotes_de_area_sobrante": lotes_extra, "area_sobrante_sumada_a_lotes": sumado_m2,
            "lotes_recortados_para_verde": recortados, "lotes_partidos": partidos,
        },
        parametros={"calle": W, "frente": F, "fondo": D, "area_minima": P.area_minima,
                    "regla_verde": P.regla_verde, "valor_verde": P.valor_verde},
        protecciones=protecciones,
        equipamiento=[{"poly": desrotar(e.poly, deg), "area": e.area} for e in equip],
    )


def anillos(poly: Polygon) -> list[list[tuple[float, float]]]:
    return [list(poly.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in poly.interiors]
