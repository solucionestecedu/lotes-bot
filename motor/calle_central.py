"""Prediseño con una sola calle central (quintas o pocos lotes).

Una calle entra desde el lado de la calle pública siguiendo un eje (el lindero común cuando se unieron dos
terrenos, o una línea perpendicular al acceso). A cada lado se reparten N lotes de igual área con frente a la
calle. Al fondo queda el retorno, el área verde y la zona de protección de la quebrada. Todo en metros (Lambert).
"""
from __future__ import annotations

import math

from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box  # noqa: F401
from shapely.ops import linemerge, unary_union

from .lotificacion import (LADOS, ErrorLotificacion, Parametros, Resultado, _mayor, _poligonos, area_verde_requerida,
                           punto_de_acceso)
from .relieve import Malla

LADOS_8 = ["norte", "noreste", "oriente", "sureste", "sur", "suroeste", "poniente", "noroeste"]


def _lado_de_azimut(az: float) -> str:
    return LADOS_8[int(((az % 360) + 22.5) // 45) % 8]


def bordes_largos(terreno: Polygon) -> list[str]:
    """Hacia dónde miran los dos linderos largos del terreno (en una franja de oriente a poniente: norte y sur)."""
    c = list(terreno.minimum_rotated_rectangle.exterior.coords)
    l1, l2 = math.dist(c[0], c[1]), math.dist(c[1], c[2])
    a, b = (c[0], c[1]) if l1 >= l2 else (c[1], c[2])
    az = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))  # azimut del lado largo
    return [_lado_de_azimut(az - 90), _lado_de_azimut(az + 90)]


def es_angosto(terreno: Polygon, ancho_max: float = 60.0) -> bool:
    """Franja: tan angosta que no caben lotes a los dos lados de una calle (ancho típico < `ancho_max` m)."""
    c = list(terreno.minimum_rotated_rectangle.exterior.coords)
    l1, l2 = math.dist(c[0], c[1]), math.dist(c[1], c[2])
    largo = max(l1, l2)
    return largo > 0 and terreno.area / largo < ancho_max and largo >= 3 * terreno.area / largo


def eje_de_borde(terreno: Polygon, lado: str, ancho_calle: float) -> list[tuple[float, float]] | None:
    """Eje de una calle pegada al lindero que mira hacia `lado`: la cadena más larga de tramos seguidos que miran hacia
    allá (sin doblar las esquinas del terreno), corrida media calle hacia adentro."""
    from shapely.geometry.polygon import orient

    objetivo = LADOS[lado]
    pts = list(orient(terreno, 1.0).exterior.coords)
    tramos = [(a, b) for a, b in zip(pts, pts[1:]) if math.dist(a, b) >= 0.05]
    if not tramos:
        return None

    def mira(a, b):
        az = math.degrees(math.atan2(b[1] - a[1], -(b[0] - a[0]))) % 360
        return abs((az - objetivo + 180) % 360 - 180) <= 50

    ok = [mira(a, b) for a, b in tramos]
    if not any(ok):
        return None
    n = len(tramos)
    inicio = next((i for i in range(n) if not ok[i]), None)  # arrancar después de un tramo que no mira
    corridas, actual = [], []
    for k in range(n):
        i = (k + (inicio + 1 if inicio is not None else 0)) % n
        if ok[i]:
            actual.append(tramos[i])
        elif actual:
            corridas.append(actual)
            actual = []
    if actual:
        corridas.append(actual)
    cadena = max(corridas, key=lambda c: sum(math.dist(a, b) for a, b in c))
    orilla = LineString([cadena[0][0]] + [b for _, b in cadena])
    if orilla.length < 2 * ancho_calle:
        return None
    for d in (ancho_calle / 2, -ancho_calle / 2):
        e = orilla.offset_curve(d, join_style=2, mitre_limit=3.0)
        if e.geom_type == "MultiLineString":
            e = max(e.geoms, key=lambda g: g.length)
        if e.geom_type == "LineString" and e.length > 1 and terreno.contains(e.interpolate(0.5, normalized=True)):
            return [tuple(c) for c in e.coords]
    return None


def lotificar_un_lado(anillo, P: Parametros | None = None, malla: Malla | None = None, excluir: list | None = None,
                      borde: str | None = None, lotes: tuple | list | None = None,
                      radio_retorno: float | None = None, cruzables: list | None = None) -> Resultado:
    """Lotes de un solo lado: la calle va pegada a un lindero largo desde la entrada y los lotes quedan en una sola
    fila del otro lado. Para franjas angostas o cuando lo piden. Sin `borde` se prueban los dos linderos largos y gana
    el que deja más lotes y más área vendible; `lotes` = [mínimo, máximo] de lotes (sin él, los que quepan con el área
    mínima). `cruzables`: caídas fuertes que la calle puede cruzar para llegar a la entrada, pero sin lotes."""
    P = P or Parametros()
    terreno = Polygon(anillo).buffer(0)
    if not isinstance(terreno, Polygon):
        terreno = _mayor(terreno)
    c = list(terreno.minimum_rotated_rectangle.exterior.coords)
    largo = max(math.dist(c[0], c[1]), math.dist(c[1], c[2]))
    candidatos = [borde.strip().lower()] if borde else bordes_largos(terreno)
    mejor, error = None, None
    for lado in candidatos:
        if lado not in LADOS:
            continue
        eje = eje_de_borde(terreno, lado, P.calle)
        if eje is None:
            error = ErrorLotificacion(f"No hay un lindero largo hacia el {lado} para pegar la calle.")
            continue
        if lotes:
            rango = lotes
        else:
            libre = max(terreno.area - P.calle * largo, P.area_minima)
            rango = (1, max(1, min(40, int(libre / max(P.area_minima, 1.0)))))
        try:
            r = _calle_central(anillo, P, malla, excluir, eje, rango, None, radio_retorno, "lindero", True,
                               un_lado=True, cruzables=cruzables)
        except ErrorLotificacion as ex:
            error = ex
            continue
        r.stats["calle_pegada_a"] = lado
        r.parametros["calle_pegada_a"] = lado
        clave = (r.stats["lotes"], r.stats["area_util"])
        if mejor is None or clave > mejor[0]:
            mejor = (clave, r)
    if mejor is None:
        raise error or ErrorLotificacion("No cupo la calle pegada al lindero en este terreno.")
    return mejor[1]


def lindero_comun(a_anillo, b_anillo) -> list[tuple[float, float]] | None:
    """Línea que comparten dos terrenos ya alineados (de punta a punta)."""
    pa, pb = Polygon(a_anillo).buffer(0), Polygon(b_anillo).buffer(0)
    g = pa.boundary.intersection(pb.buffer(0.05))
    lineas = [x for x in getattr(g, "geoms", [g]) if x.geom_type == "LineString" and x.length > 0.5]
    if not lineas:
        return None
    m = linemerge(MultiLineString(lineas)) if len(lineas) > 1 else lineas[0]
    if m.geom_type != "LineString":
        m = max(m.geoms, key=lambda x: x.length)
    return [tuple(c) for c in m.coords]


def eje_por_defecto(terreno: Polygon, P: Parametros) -> list[tuple[float, float]]:
    """Sin lindero común: la calle entra perpendicular al lindero de acceso; si no hay acceso, por el eje largo."""
    lado = (P.lado_acceso or "").lower()
    inicio = P.entrada or (punto_de_acceso(list(terreno.exterior.coords)[:-1], lado) if lado in LADOS else None)
    if inicio is not None and lado in LADOS:
        az = math.radians(LADOS[lado] + 180)  # hacia adentro
        return [tuple(inicio), (inicio[0] + 1000 * math.sin(az), inicio[1] + 1000 * math.cos(az))]
    rect = terreno.minimum_rotated_rectangle
    c = list(rect.exterior.coords)
    l1, l2 = math.dist(c[0], c[1]), math.dist(c[1], c[2])
    a, b = (c[0], c[1]) if l1 >= l2 else (c[1], c[2])
    ux, uy = (b[0] - a[0]) / max(l1, l2), (b[1] - a[1]) / max(l1, l2)
    cx, cy = terreno.centroid.x, terreno.centroid.y
    if inicio is not None:
        cx, cy = inicio
    return [(cx - 1000 * ux, cy - 1000 * uy), (cx + 1000 * ux, cy + 1000 * uy)]


def ejes_candidatos(terreno: Polygon, P: Parametros) -> list[list[tuple[float, float]]]:
    """Sin lindero común se prueban dos: perpendicular al acceso y a lo largo del terreno."""
    out = []
    lado = (P.lado_acceso or "").lower()
    if lado in LADOS or P.entrada:
        out.append(eje_por_defecto(terreno, P))
    rect = terreno.minimum_rotated_rectangle
    c = list(rect.exterior.coords)
    l1, l2 = math.dist(c[0], c[1]), math.dist(c[1], c[2])
    a, b = (c[0], c[1]) if l1 >= l2 else (c[1], c[2])
    ux, uy = (b[0] - a[0]) / max(l1, l2), (b[1] - a[1]) / max(l1, l2)
    cx, cy = terreno.centroid.x, terreno.centroid.y
    out.append([(cx - 1000 * ux, cy - 1000 * uy), (cx + 1000 * ux, cy + 1000 * uy)])
    return out


def lotificar_calle_central(anillo, P: Parametros | None = None, malla: Malla | None = None, excluir: list | None = None,
                            eje: list | None = None, lotes_por_lado=(4, 5), partes: list | None = None,
                            radio_retorno: float | None = None, calle_sobre: str = "lindero",
                            equilibrar: bool = True) -> Resultado:
    """Calle central sobre `eje` (p. ej. el lindero común). Sin eje se prueban los candidatos y gana el que deja
    más lotes dentro del rango pedido y más área vendible."""
    if eje is not None:
        return _calle_central(anillo, P, malla, excluir, eje, lotes_por_lado, partes, radio_retorno, calle_sobre,
                              equilibrar)
    P = P or Parametros()
    terreno = Polygon(anillo).buffer(0)
    mejor, error = None, None
    for e in ejes_candidatos(_mayor(terreno) if terreno.geom_type != "Polygon" else terreno, P):
        try:
            r = _calle_central(anillo, P, malla, excluir, e, lotes_por_lado, partes, radio_retorno, "lindero",
                               equilibrar)
        except ErrorLotificacion as ex:
            error = ex
            continue
        nmin = min(lotes_por_lado)
        clave = (sum(1 for pl in r.stats["por_lado"] if pl["lotes"] >= nmin), r.stats["lotes"], r.stats["area_util"])
        if mejor is None or clave > mejor[0]:
            mejor = (clave, r)
    if mejor is None:
        raise error or ErrorLotificacion("No cupo la calle central en este terreno.")
    return mejor[1]


def _equilibrar_lado(lotes: list, y_eje, calle_b, frente_min: float, area_min: float,
                     max_grados: float = 35.0) -> list | None:
    """Reparte parejo el área de los lotes de un lado de la calle girando los cortes: cada corte sigue saliendo del
    mismo punto en la calle (el frente de cada lote no cambia) y su otro extremo se corre hasta que todos los lotes
    queden de igual área, sin inclinarse más de `max_grados` respecto a la perpendicular a la calle.
    Trabaja en el marco girado (calle a lo largo de +x). Devuelve los lotes nuevos o None si no conviene."""
    U = unary_union(lotes).buffer(0.01, join_style=2).buffer(-0.01, join_style=2)
    if not isinstance(U, Polygon) or len(lotes) < 2:
        return None
    viejas = [l.area for l in lotes]
    if max(viejas) < 1.06 * min(viejas):
        return None
    n = len(lotes)
    objetivo = U.area / n
    c = U.centroid
    signo = 1 if c.y >= y_eje(c.x) else -1
    x0, y0, x1, y1 = U.bounds
    y_lejos, y_cerca = (y1 + 5, y0 - 5) if signo > 0 else (y0 - 5, y1 + 5)
    frentes = []
    for a, b in zip(lotes, lotes[1:]):
        comun = a.buffer(0.05).intersection(b.buffer(0.05))
        pts = [q for g in _poligonos(comun) for q in g.exterior.coords]
        if not pts:
            return None
        frentes.append(min(pts, key=lambda q: abs(q[1] - y_eje(q[0]))))

    def antes(F, bx):
        def x_en(y):
            return F[0] + (bx - F[0]) * (y - F[1]) / (y_lejos - F[1])
        return Polygon([(x0 - 50, y_cerca), (x_en(y_cerca), y_cerca), (x_en(y_lejos), y_lejos), (x0 - 50, y_lejos)])

    tan = math.tan(math.radians(max_grados))
    cortes, b_prev = [], -math.inf
    for k, F in enumerate(frentes, start=1):
        dmax = abs(y_lejos - F[1]) * tan
        lo, hi = max(F[0] - dmax, b_prev + 0.5), F[0] + dmax
        if lo >= hi:
            return None
        for _ in range(50):
            m = (lo + hi) / 2
            if U.intersection(antes(F, m)).area < k * objetivo:
                lo = m
            else:
                hi = m
        b_prev = (lo + hi) / 2
        cortes.append(antes(F, b_prev))
    nuevos, hecho = [], Polygon()
    for zona in cortes + [None]:
        hasta = U if zona is None else U.intersection(zona)
        partes_ = sorted(_poligonos(hasta.difference(hecho)), key=lambda g: -g.area)
        if not partes_ or sum(g.area for g in partes_[1:]) > 1.0:
            return None
        nuevos.append(partes_[0])
        hecho = hasta
    if any(l.boundary.intersection(calle_b).length < frente_min or l.area < area_min for l in nuevos):
        return None
    import shapely

    nuevos = [_mayor(shapely.set_precision(g, 0.001).buffer(0)) for g in nuevos]  # coordenadas limpias, sin astillas
    if any(g is None or not isinstance(g, Polygon) for g in nuevos):
        return None
    nuevas = [l.area for l in nuevos]
    if max(nuevas) - min(nuevas) >= max(viejas) - min(viejas) - 1.0:
        return None
    return nuevos


def _calle_central(anillo, P, malla, excluir, eje, lotes_por_lado, partes, radio_retorno, calle_sobre="lindero",
                   equilibrar: bool = True, un_lado: bool = False, cruzables: list | None = None) -> Resultado:
    P = P or Parametros()
    W = P.calle
    nmin, nmax = sorted(int(x) for x in (list(lotes_por_lado) * 2)[:2])
    nmin = max(1, nmin)
    r_ret = radio_retorno or max(W, 9.0)
    frente_min = 0.6 * P.frente

    terreno_total = Polygon(anillo).buffer(0)
    if not isinstance(terreno_total, Polygon):
        terreno_total = _mayor(terreno_total)
    protecciones, sueltas = [], []
    Z = terreno_total
    if excluir:
        zona = unary_union([e for e in excluir if e is not None and not e.is_empty]).intersection(terreno_total)
        protecciones = _poligonos(zona)
        piezas = sorted(_poligonos(terreno_total.difference(zona)), key=lambda p: -p.area)
        Z = piezas[0] if piezas else None
        sueltas = piezas[1:]
    if Z is None or Z.area < P.area_minima:
        raise ErrorLotificacion("El terreno es más chico que un lote.")
    area_total = terreno_total.area
    area_prot = sum(p.area for p in protecciones)
    # zonas que la calle sí puede cruzar pero donde no van lotes (caídas fuertes del relieve en una franja: la calle
    # tiene que pasar para llegar a la entrada)
    cruz = None
    if cruzables:
        cruz = unary_union([c for c in cruzables if c is not None and not c.is_empty]).intersection(Z)
        cruz = None if cruz.is_empty else cruz

    # ---- eje: se ordena para que empiece en el lado de la calle pública
    pts = [tuple(p) for p in eje]
    lado = (P.lado_acceso or "").lower()
    if lado in LADOS:  # el extremo que queda más hacia el lado de la calle pública
        az = math.radians(LADOS[lado])
        hacia = lambda q: q[0] * math.sin(az) + q[1] * math.cos(az)  # noqa: E731
        if hacia(pts[-1]) > hacia(pts[0]):
            pts.reverse()
    elif P.entrada:
        if Point(pts[-1]).distance(Point(P.entrada)) < Point(pts[0]).distance(Point(P.entrada)):
            pts.reverse()
    elif protecciones:
        prot = unary_union(protecciones)
        if Point(pts[0]).distance(prot) < Point(pts[-1]).distance(prot):
            pts.reverse()
    # calle_sobre: "lindero" = el eje es el lindero (media calle en cada terreno); "mayor"/"menor" = toda la calle
    # dentro del terreno más grande o más chico, pegada al lindero (la tierra de la calle la pone un solo dueño)
    dentro_de = None
    if calle_sobre in ("mayor", "menor") and partes and len(partes) >= 2 and len(pts) >= 2:
        polys = sorted(((Polygon(p["anillo"]).buffer(0), p["origen"]) for p in partes), key=lambda x: x[0].area)
        destino, dentro_de = polys[-1] if calle_sobre == "mayor" else polys[0]
        linea_eje = LineString(pts)
        cerca = linea_eje.interpolate(0.5, normalized=True)
        c_ = destino.centroid
        a_, b_ = pts[0], pts[-1]
        izquierda = (b_[0] - a_[0]) * (c_.y - a_[1]) - (b_[1] - a_[1]) * (c_.x - a_[0]) > 0
        movida = linea_eje.offset_curve(W / 2 if izquierda else -W / 2, join_style=2)
        if movida.geom_type == "LineString" and movida.length > 1 and movida.distance(cerca) < W:
            pts = [tuple(c) for c in movida.coords]
            if math.dist(pts[0], a_) > math.dist(pts[-1], a_):
                pts.reverse()
        else:
            dentro_de = None
    ux, uy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
    largo = math.hypot(ux, uy)
    if largo < 1:
        raise ErrorLotificacion("El eje de la calle es demasiado corto.")
    ux, uy = ux / largo, uy / largo
    ext = LineString([(pts[0][0] - 1000 * ux, pts[0][1] - 1000 * uy)] + pts[1:-1] +
                     [(pts[-1][0] + 1000 * ux, pts[-1][1] + 1000 * uy)])

    # marco girado: el eje corre en +x desde el acceso
    deg = math.degrees(math.atan2(uy, ux))
    C = terreno_total.centroid

    def rot(g):
        return affinity.rotate(g, -deg, origin=C)

    def desrot(g):
        return affinity.rotate(g, deg, origin=C)

    Zr, Tr = rot(Z), rot(terreno_total)
    cruz_r = rot(cruz) if cruz is not None else None
    # con la calle dentro de un solo terreno, el redondel también se corre hacia ese terreno
    corrimiento = 0.0
    if dentro_de is not None:
        c_dest = rot(next(Polygon(p["anillo"]).buffer(0) for p in partes if p["origen"] == dentro_de).centroid)
    linea = ext.intersection(terreno_total)
    piezas_l = [g for g in getattr(linea, "geoms", [linea]) if g.geom_type == "LineString" and g.length > 1]
    if not piezas_l:
        raise ErrorLotificacion("El eje de la calle no pasa por el terreno.")
    ref = LineString(pts) if len(pts) > 1 else ext
    linea = max(piezas_l, key=lambda g: (g.distance(ref) < 1.0, g.length))
    linea_r = rot(linea)
    xs_l = sorted(linea_r.coords, key=lambda c: c[0])
    x_ini = xs_l[0][0]
    dentro = linea_r.intersection(Zr)
    tramos_z = [g for g in getattr(dentro, "geoms", [dentro]) if g.geom_type == "LineString" and g.length > 0.5]
    if not tramos_z:
        raise ErrorLotificacion("La calle no cabe fuera de la zona de protección.")
    tramo_z = min(tramos_z, key=lambda g: min(c[0] for c in g.coords))
    x_fin = max(c[0] for c in tramo_z.coords)

    def y_eje(x):
        cs = xs_l
        if x <= cs[0][0]:
            return cs[0][1]
        for (x0, y0), (x1, y1) in zip(cs, cs[1:]):
            if x0 <= x <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return cs[-1][1]

    if dentro_de is not None:
        corrimiento = max(0.0, r_ret - W / 2) * (1 if c_dest.y > y_eje(c_dest.x) else -1)
    elif un_lado:  # calle pegada al lindero: el redondel se corre hacia los lotes para caber completo
        c_t = Zr.representative_point()
        corrimiento = max(0.0, r_ret - W / 2) * (1 if c_t.y > y_eje(c_t.x) else -1)

    def calle_hasta(x_r):
        """Calle del acceso hasta x_r con un redondel de retorno centrado ahí."""
        coords = [(x_ini - 2, y_eje(x_ini))] + [c for c in xs_l if x_ini < c[0] < x_r] + [(x_r, y_eje(x_r))]
        tira = LineString(coords).buffer(W / 2, cap_style=2)
        redondel = Point(x_r, y_eje(x_r) + corrimiento).buffer(r_ret, quad_segs=12)
        return _mayor(unary_union([tira, redondel]).intersection(Zr))

    if cruz_r is not None:  # lo que la calle no ocupa de las caídas fuertes cuenta como área verde ecológica
        tira = LineString([(x_ini - 2, y_eje(x_ini))] + [c for c in xs_l if c[0] > x_ini]).buffer(W / 2, cap_style=2)
        area_prot += cruz_r.difference(tira).area

    cache_r: dict = {}
    # pegado al lindero, el redondel queda tangente: se le perdona un borde mínimo que cae fuera (se recorta)
    tolerancia_r = 0.03 * math.pi * r_ret ** 2 if un_lado else 0.5

    def x_redondel(x_r):
        """Corre el redondel hacia la entrada hasta que quepa completo (sin morder la franja de protección, una caída
        fuerte ni el lindero): un retorno recortado no sirve para dar la vuelta."""
        clave = round(x_r, 1)
        if clave not in cache_r:
            x = x_r
            while x > x_ini + r_ret:
                circulo = Point(x, y_eje(x) + corrimiento).buffer(r_ret, quad_segs=12)
                if circulo.difference(Zr).area < tolerancia_r:
                    break
                x -= 0.5
            cache_r[clave] = max(x, x_ini + r_ret)
        return cache_r[clave]

    def armar(x_corte):
        ideal = max(x_ini + r_ret, min(x_corte + r_ret, x_fin - r_ret))
        x_r = x_redondel(ideal)
        x_corte = x_corte - (ideal - x_r)  # si el redondel se recorre, los lotes terminan antes de él
        calle = calle_hasta(x_r)
        libre = Zr.difference(calle.buffer(0.01))
        if cruz_r is not None:
            libre = libre.difference(cruz_r)
        zona_lotes = libre.intersection(box(Tr.bounds[0] - 5, Tr.bounds[1] - 5, x_corte, Tr.bounds[3] + 5))
        zona_verde = libre.difference(box(Tr.bounds[0] - 5, Tr.bounds[1] - 5, x_corte, Tr.bounds[3] + 5))
        return calle, zona_lotes, zona_verde

    equip_m2 = max(0.0, float(P.equipamiento_m2 or 0.0))

    def meta_verde(n_lotes, util):
        """Lo que hay que dejar junto al retorno: el área verde que falta y el terreno del equipamiento social."""
        req = area_verde_requerida(P.regla_verde, P.valor_verde, n_lotes, util, area_total)
        return (0.0 if req <= 0 else max(req - area_prot, req * P.pct_recreativa / 100)) + equip_m2

    # ---- dónde terminan los lotes: lo más al fondo posible dejando el área verde que pide la norma,
    # y que esa área verde dé a la calle (si no, nadie puede entrar al parque)
    n_est = nmax if un_lado else 2 * nmax

    def suficiente(x):
        calle_, zl, zv = armar(x)
        meta = meta_verde(n_est, zl.area)
        if meta <= 0:
            return True
        utiles = [g for g in _poligonos(zv) if g.area >= P.verde_minimo]
        frente_v = sum(g.boundary.intersection(calle_.buffer(0.3)).length for g in utiles)
        return sum(g.area for g in utiles) >= meta and frente_v >= 6.0

    lo, hi = x_ini + 10, x_fin
    if suficiente(hi):
        x_corte = hi
    elif un_lado:  # en una franja larga la prueba no es pareja: se recorre desde el fondo hacia la entrada
        x_corte, x = lo, hi
        while x > lo:
            x -= 2.0
            if suficiente(x):
                x_corte = x
                break
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            if suficiente(mid):
                lo = mid
            else:
                hi = mid
        x_corte = lo
    calle, zona_lotes, zona_verde = armar(x_corte)
    calle_b = calle.buffer(0.3)

    # ---- lados de la calle
    partes_r = [(p["origen"], rot(Polygon(p["anillo"]).buffer(0))) for p in (partes or [])]
    lados = {}
    remanentes, verdes = [], []
    for pieza in _poligonos(zona_lotes):
        c = pieza.centroid
        clave = "izquierda" if c.y >= y_eje(c.x) else "derecha"
        if pieza.boundary.intersection(calle_b).length < frente_min or pieza.area < P.area_minima:
            (verdes if pieza.area >= P.verde_minimo else remanentes).append(pieza)
            continue
        if un_lado:  # una sola fila: cada tramo entre caídas fuertes lleva sus propios lotes
            lados[f"{clave}#{len(lados)}"] = pieza
            continue
        if clave not in lados or pieza.area > lados[clave].area:
            if clave in lados:
                verdes.append(lados[clave])
            lados[clave] = pieza
        else:
            verdes.append(pieza)

    def rebanar(pieza, cortes_internos):
        x0, y0, x1, y1 = pieza.bounds
        cortes = [x0 - 1] + list(cortes_internos) + [x1 + 1]
        lotes, sobras = [], []
        for a, b in zip(cortes, cortes[1:]):
            ps = sorted(_poligonos(pieza.intersection(box(a, y0 - 2, b, y1 + 2))), key=lambda p: -p.area)
            if not ps:
                return None
            lotes.append(ps[0])
            sobras += ps[1:]
        return lotes, sobras

    def cortes_igual_area(pieza, n):
        """Cortes perpendiculares a la calle para n lotes de igual área, desde el acceso."""
        x0, y0, x1, y1 = pieza.bounds
        objetivo = pieza.area / n
        cortes, a0 = [], x0 - 1
        for k in range(1, n):
            a, b = a0, x1
            for _ in range(40):
                m = (a + b) / 2
                if pieza.intersection(box(x0 - 2, y0 - 2, m, y1 + 2)).area < k * objetivo:
                    a = m
                else:
                    b = m
            a0 = (a + b) / 2
            cortes.append(a0)
        return cortes

    def cortes_igual_frente(pieza, n):
        """Cortes para que todos tengan el mismo frente a la calle (las áreas varían)."""
        contacto = pieza.boundary.intersection(calle_b)
        if contacto.is_empty:
            return None
        c0, _, c1, _ = contacto.bounds
        return [c0 + k * (c1 - c0) / n for k in range(1, n)]

    def evaluar(pieza, n):
        """Prefiere lotes de igual área; si alguno se queda sin frente suficiente, mueve los cortes hacia
        frentes iguales hasta que todos cumplan frente y área mínima."""
        ca = cortes_igual_area(pieza, n)
        cf = cortes_igual_frente(pieza, n) or ca
        mejor = None
        for t in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
            cortes = sorted(t * a + (1 - t) * f for a, f in zip(ca, cf))
            r_ = rebanar(pieza, cortes)
            if r_ is None:
                continue
            lotes, sobras = r_
            frentes = [l.boundary.intersection(calle_b).length for l in lotes]
            ok = all(f >= frente_min for f in frentes) and all(l.area >= P.area_minima for l in lotes)
            o = {"lotes": lotes, "sobras": sobras, "frentes": frentes, "ok": ok, "n": n,
                 "reparto": "igual_area" if t == 1.0 else "igual_frente" if t == 0.0 else "mixto"}
            if ok:
                return o
            mejor = mejor or o
        return mejor

    por_lado, lotes_out = [], []
    if un_lado:  # desde la entrada hacia el fondo
        orden = sorted(lados.items(), key=lambda kv: kv[1].bounds[0])
    else:
        orden = [(k, lados[k]) for k in ("izquierda", "derecha") if k in lados]
    for clave_l, pieza in orden:
        clave = clave_l.split("#")[0]
        origen = None
        if partes_r:
            origen = max(partes_r, key=lambda pr: pr[1].intersection(pieza).area)[0]
        opciones = {n: evaluar(pieza, n) for n in range(nmax, 0, -1)}
        elegido = next((opciones[n] for n in range(nmax, nmin - 1, -1) if opciones[n] and opciones[n]["ok"]), None)
        nota = None
        if elegido is None:
            elegido = next((o for n, o in opciones.items() if n < nmin and o and o["ok"]), None)
            nota = f"no caben {nmin} lotes con frente de {frente_min:.0f} m o más" if elegido else None
        if elegido is None:
            verdes.append(pieza)
            continue
        alternativas = []
        for n in range(nmin, nmax + 1):
            o = opciones.get(n)
            if o and o["ok"] and n != elegido["n"]:
                ar = [l.area for l in o["lotes"]]
                alternativas.append({"lotes": n, "lote_m2": round(sum(ar) / n, 1),
                                     "frente_min_m": round(min(o["frentes"]), 1)})
        if elegido["n"] < nmax:
            o = opciones.get(nmax)
            if o:
                motivo = []
                if min(o["frentes"]) < frente_min:
                    motivo.append(f"un lote quedaría con {min(o['frentes']):.1f} m de frente (mínimo {frente_min:.0f} m)")
                if min(l.area for l in o["lotes"]) < P.area_minima:
                    motivo.append(f"un lote quedaría de {min(l.area for l in o['lotes']):.0f} m² (mínimo {P.area_minima:.0f} m²)")
                nota = f"con {nmax} no cabe: " + " y ".join(motivo) if motivo else nota
        if un_lado and elegido["n"] >= nmin:
            nota = None  # en una fila el máximo es un tope, no lo que pidieron
        for s_ in elegido["sobras"]:
            (verdes if s_.area >= P.verde_minimo else remanentes).append(s_)
        ids = []
        for l, f in zip(elegido["lotes"], elegido["frentes"]):
            lotes_out.append({"poly": l, "area": l.area, "frente": f, "lado": clave, "origen": origen})
            ids.append(len(lotes_out))
        por_lado.append({"lado": clave, "terreno": origen, "lotes": elegido["n"], "reparto": elegido["reparto"],
                         "lote_m2": round(pieza.area / elegido["n"], 1),
                         "frente_min_m": round(min(elegido["frentes"]), 1), "ids": ids, "alternativas": alternativas,
                         "nota": nota})
    if not lotes_out:
        raise ErrorLotificacion("No cupo ningún lote a los lados de la calle con estas medidas.")

    for p in _poligonos(zona_verde):
        (verdes if p.area >= P.verde_minimo else remanentes).append(p)
    for p in sueltas:
        (verdes.append(rot(p)) if p.area >= P.verde_minimo else remanentes.append(rot(p)))

    # ---- máximo aprovechamiento: el área verde que sobra (sobre todo la que no da a la calle) se suma
    # al lote vecino; la que se queda es la que rodea el retorno, para que el parque tenga entrada.
    x_r = max(x_ini + r_ret, min(x_corte + r_ret, x_fin - r_ret))
    centro_ret = Point(x_r, y_eje(x_r) + corrimiento)
    sumado = 0.0
    if P.aprovechar:
        def meta_actual():
            return meta_verde(len(lotes_out), sum(l["poly"].area for l in lotes_out))

        def dar(pieza, es_verde=True):
            """Suma `pieza` al lote con el que comparte más lindero, si el área verde sigue alcanzando."""
            nonlocal sumado
            vecinos = [(l["poly"].boundary.intersection(pieza.buffer(0.1)).length, l) for l in lotes_out]
            vecinos = [(lg, l) for lg, l in vecinos if lg >= 3.0]
            if not vecinos:
                return False
            _, l = max(vecinos, key=lambda x: x[0])
            u = unary_union([l["poly"], pieza]).buffer(0.01, join_style=2).buffer(-0.01, join_style=2)
            if not isinstance(u, Polygon):
                return False
            antes = l["poly"]
            l["poly"] = u
            if sum(v.area for v in verdes) - (pieza.area if es_verde else 0.0) >= meta_actual() - 1e-6:
                sumado += pieza.area
                return True
            l["poly"] = antes
            return False

        sin_entrada = sorted([v for v in verdes if v.boundary.intersection(calle_b).length < 1.0], key=lambda v: v.area)
        for v in sin_entrada:
            if dar(v):
                verdes.remove(v)
        # la pieza con entrada: se deja solo lo que rodea el retorno
        for v in sorted([v for v in verdes if v.boundary.intersection(calle_b).length >= 1.0], key=lambda v: -v.area):
            otras = sum(x.area for x in verdes if x is not v)
            util_si = sum(l["poly"].area for l in lotes_out) + v.area  # lo más que podría crecer el área útil
            guardar = meta_verde(len(lotes_out), util_si) * 1.02 - otras
            if guardar < P.verde_minimo:
                if dar(v):
                    verdes.remove(v)
                continue
            if v.area - guardar < P.verde_minimo:
                continue
            lo_r, hi_r = r_ret, 400.0
            for _ in range(40):
                m = (lo_r + hi_r) / 2
                if v.intersection(centro_ret.buffer(m, quad_segs=24)).area < guardar:
                    lo_r = m
                else:
                    hi_r = m
            queda = _mayor(v.intersection(centro_ret.buffer(hi_r, quad_segs=24)))
            if queda is None or queda.boundary.intersection(calle_b).length < 1.0:
                continue
            sobran = [g for g in _poligonos(v.difference(queda)) if g.area > 1.0]
            verdes.remove(v)
            verdes.append(queda)
            for g in sorted(sobran, key=lambda g: -g.area):
                if not dar(g, es_verde=False):
                    (verdes if g.area >= P.verde_minimo else remanentes).append(g)
        for l in lotes_out:
            l["area"] = l["poly"].area
    # ---- equipamiento social: del área que queda junto al retorno, la parte más cercana a él (con entrada)
    equip = []
    if equip_m2 > 0 and verdes:
        con_calle = [v for v in verdes if v.boundary.intersection(calle_b).length >= 1.0] or list(verdes)
        v = min(con_calle, key=lambda g: g.distance(centro_ret))
        if v.area > equip_m2 + 1.0:
            lo_r, hi_r = 0.0, 400.0
            for _ in range(40):
                m_ = (lo_r + hi_r) / 2
                if v.intersection(centro_ret.buffer(m_, quad_segs=24)).area < equip_m2:
                    lo_r = m_
                else:
                    hi_r = m_
            e_ = _mayor(v.intersection(centro_ret.buffer(hi_r, quad_segs=24)))
            resto = [g for g in _poligonos(v.difference(e_)) if g.area > 1.0] if e_ is not None else [v]
        else:
            e_, resto = v, []
        if e_ is not None:
            verdes.remove(v)
            equip.append(e_)
            for g in resto:
                (verdes if g.area >= P.verde_minimo else remanentes).append(g)
    # ---- reparto parejo: la tierra que recibió el lote del fondo (detrás del área verde) se reparte entre los lotes
    # de su lado girando los cortes; el frente de cada lote a la calle no cambia
    if equilibrar:
        for pl in por_lado:
            idx = [i - 1 for i in pl["ids"]]
            nuevos = _equilibrar_lado([lotes_out[i]["poly"] for i in idx], y_eje, calle_b, frente_min, P.area_minima)
            if nuevos:
                for i, poly in zip(idx, nuevos):
                    lotes_out[i]["poly"] = poly
                    lotes_out[i]["area"] = poly.area
                    lotes_out[i]["frente"] = poly.boundary.intersection(calle_b).length
                pl["reparto"] = "parejo"
    for pl in por_lado:
        ar = [lotes_out[i - 1]["area"] for i in pl["ids"]]
        pl["lote_m2"] = round(sum(ar) / len(ar), 1)
        pl["areas_m2"] = [round(a, 1) for a in ar]

    # pendientes
    def pendiente_en(x, y):
        if malla is None:
            return math.nan
        h = 0.5 * malla.dx
        dzdx = (malla.altura(x + h, y) - malla.altura(x - h, y)) / (2 * h)
        dzdy = (malla.altura(x, y + h) - malla.altura(x, y - h)) / (2 * h)
        return math.hypot(dzdx, dzdy)

    pend_calle = 0.0
    eje_calle = desrot(LineString([(x_ini, y_eje(x_ini))] + [c for c in xs_l if x_ini < c[0] < x_corte] +
                                  [(x_corte, y_eje(x_corte))]))
    if malla is not None and eje_calle.length > 4:
        n_m = max(2, int(eje_calle.length / 2))
        zs = [malla.altura(*eje_calle.interpolate(i / n_m, normalized=True).coords[0]) for i in range(n_m + 1)]
        paso = eje_calle.length / n_m
        difs = [abs(b - a) / paso for a, b in zip(zs, zs[1:]) if math.isfinite(a) and math.isfinite(b)]
        pend_calle = sum(difs) / len(difs) if difs else 0.0

    salida = []
    for i, l in enumerate(lotes_out, start=1):
        poly = desrot(l["poly"])
        c = poly.centroid
        p_ = pendiente_en(c.x, c.y)
        salida.append({"id": i, "poly": poly, "area": l["area"], "frente": l["frente"],
                       "pendiente": p_ * 100 if math.isfinite(p_) else None, "terreno": l["origen"], "lado": l["lado"]})
    area_util = sum(l["area"] for l in salida)
    area_verde = sum(v.area for v in verdes)
    areas = [l["area"] for l in salida]
    req = area_verde_requerida(P.regla_verde, P.valor_verde, len(salida), area_util, area_total)
    if cruz is not None:  # lo que la calle no ocupa de las caídas fuertes queda como zona sin lotes
        protecciones = protecciones + [g for g in _poligonos(cruz.difference(desrot(calle))) if g.area >= 1.0]
        area_prot = sum(p.area for p in protecciones)
    return Resultado(
        angulo=round(deg % 180, 1),
        lotes=salida,
        calles=[desrot(calle)],
        verdes=[{"poly": desrot(v), "area": v.area, "de_lote": False} for v in verdes],
        remanentes=[desrot(p) for p in remanentes],
        stats={
            "area_total": area_total, "area_util": area_util, "area_circulacion": calle.area,
            "area_verde": area_verde, "area_verde_requerida": req, "area_remanente": sum(p.area for p in remanentes),
            "lotes": len(salida), "lotes_convertidos_a_verde": 0,
            "largo_calles": eje_calle.length + r_ret, "lote_promedio": area_util / len(salida),
            "lote_min": min(areas), "lote_max": max(areas), "pendiente_calles_pct": pend_calle * 100,
            "area_proteccion": area_prot, "area_equipamiento": sum(e.area for e in equip),
            "calles_sin_salida": True, "calles_desconectadas": 0, "accesos": 1,
            "lotes_de_area_sobrante": 0, "area_sobrante_sumada_a_lotes": sumado,
            "lotes_recortados_para_verde": 0, "lotes_partidos": 0,
            "modo": "calle_central", "retorno_diametro_m": 2 * r_ret, "por_lado": por_lado,
            "calle_dentro_de": dentro_de, "un_lado": un_lado,
            "entrada_calle": [round(c, 3) for c in desrot(Point(x_ini, y_eje(x_ini))).coords[0]],
            # boca de la calle: su sección en la entrada (pegada al lindero, todo su costado toca el lindero y no sirve)
            "boca_calle": [[round(c, 3) for c in desrot(Point(x_ini, y_eje(x_ini) + k * W / 2)).coords[0]]
                           for k in (-1, 1)],
            "retorno_centro": [round(c, 3) for c in desrot(centro_ret).coords[0]],
        },
        parametros={"calle": W, "frente_minimo": round(frente_min, 1), "area_minima": P.area_minima,
                    "regla_verde": P.regla_verde, "valor_verde": P.valor_verde, "modo": "calle_central",
                    "lotes_por_lado": [nmin, nmax], "un_lado": un_lado},
        protecciones=protecciones,
        equipamiento=[{"poly": desrot(e), "area": e.area} for e in equip],
    )
