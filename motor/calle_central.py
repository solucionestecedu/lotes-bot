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

from .lotificacion import (LADOS, ErrorLotificacion, Parametros, Resultado, _bordes_hacia, _mayor, _poligonos,
                           area_verde_requerida, punto_de_acceso)
from .relieve import Malla


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
                            radio_retorno: float | None = None, calle_sobre: str = "lindero") -> Resultado:
    """Calle central sobre `eje` (p. ej. el lindero común). Sin eje se prueban los candidatos y gana el que deja
    más lotes dentro del rango pedido y más área vendible."""
    if eje is not None:
        return _calle_central(anillo, P, malla, excluir, eje, lotes_por_lado, partes, radio_retorno, calle_sobre)
    P = P or Parametros()
    terreno = Polygon(anillo).buffer(0)
    mejor, error = None, None
    for e in ejes_candidatos(_mayor(terreno) if terreno.geom_type != "Polygon" else terreno, P):
        try:
            r = _calle_central(anillo, P, malla, excluir, e, lotes_por_lado, partes, radio_retorno, "lindero")
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


def _calle_central(anillo, P, malla, excluir, eje, lotes_por_lado, partes, radio_retorno, calle_sobre="lindero") -> Resultado:
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

    def calle_hasta(x_r):
        """Calle del acceso hasta x_r con un redondel de retorno centrado ahí."""
        coords = [(x_ini - 2, y_eje(x_ini))] + [c for c in xs_l if x_ini < c[0] < x_r] + [(x_r, y_eje(x_r))]
        tira = LineString(coords).buffer(W / 2, cap_style=2)
        redondel = Point(x_r, y_eje(x_r) + corrimiento).buffer(r_ret, quad_segs=12)
        return _mayor(unary_union([tira, redondel]).intersection(Zr))

    def armar(x_corte):
        x_r = max(x_ini + r_ret, min(x_corte + r_ret, x_fin - r_ret))
        calle = calle_hasta(x_r)
        libre = Zr.difference(calle.buffer(0.01))
        zona_lotes = libre.intersection(box(Tr.bounds[0] - 5, Tr.bounds[1] - 5, x_corte, Tr.bounds[3] + 5))
        zona_verde = libre.difference(box(Tr.bounds[0] - 5, Tr.bounds[1] - 5, x_corte, Tr.bounds[3] + 5))
        return calle, zona_lotes, zona_verde

    def meta_verde(n_lotes, util):
        req = area_verde_requerida(P.regla_verde, P.valor_verde, n_lotes, util, area_total)
        return 0.0 if req <= 0 else max(req - area_prot, req * P.pct_recreativa / 100)

    # ---- dónde terminan los lotes: lo más al fondo posible dejando el área verde que pide la norma,
    # y que esa área verde dé a la calle (si no, nadie puede entrar al parque)
    n_est = 2 * nmax

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
    for clave in ("izquierda", "derecha"):
        pieza = lados.get(clave)
        if pieza is None:
            continue
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
            "area_proteccion": area_prot, "area_equipamiento": 0.0,
            "calles_sin_salida": True, "calles_desconectadas": 0, "accesos": 1,
            "lotes_de_area_sobrante": 0, "area_sobrante_sumada_a_lotes": sumado,
            "lotes_recortados_para_verde": 0, "lotes_partidos": 0,
            "modo": "calle_central", "retorno_diametro_m": 2 * r_ret, "por_lado": por_lado,
            "calle_dentro_de": dentro_de,
        },
        parametros={"calle": W, "frente_minimo": round(frente_min, 1), "area_minima": P.area_minima,
                    "regla_verde": P.regla_verde, "valor_verde": P.valor_verde, "modo": "calle_central",
                    "lotes_por_lado": [nmin, nmax]},
        protecciones=protecciones,
        equipamiento=[],
    )
