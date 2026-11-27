"""Revisión de normas de parcelación habitacional en El Salvador (referencial).

Fuentes revisadas (nov 2026):
- Ley de Urbanismo y Construcción (LUC), Art. 2 lits. e) y g).
- Reglamento a la LUC en lo relativo a Parcelaciones y Urbanizaciones Habitacionales (D.E. 70/1991, "RLUC"):
  Arts. 44, 49, 50, 52, 55, 58, 59, 61, 65, 76, 84, 87.
- Reglamento de la LDOTAMSS (OPAMSS), Arts. 64 y 66, para el Área Metropolitana de San Salvador.

Los anexos del RLUC con anchos de vías y retornos no están en el texto público consultado, así que esas
reglas se marcan para revisar. Todo resultado es una ayuda para el profesional, no un dictamen.
"""
from __future__ import annotations

import math

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

ESTADOS = {"cumple": "Cumple", "no_cumple": "No cumple", "revisar": "Revisar", "falta_dato": "Falta dato"}
ZONAS = {
    "urbano": "zona urbana o urbanizable (fuera del AMSS)",
    "rural": "fuera de centros poblados",
    "amss": "Área Metropolitana de San Salvador (OPAMSS)",
    "reserva_ecologica": "zona de reserva ecológica",
}
LIMITES_PENDIENTE_CALLE = [(100, 18.0), (300, 12.0), (500, 8.0)]  # RLUC Art. 84


def area_verde_requerida(zona: str, n_lotes: int, area_util: float) -> tuple[float, str]:
    if zona == "rural":
        return 12.5 * n_lotes, "LUC Art. 2 e): 12.5 m² por lote fuera de centros poblados"
    if zona == "amss":
        return 0.10 * area_util, "Reglamento LDOTAMSS Art. 64: 10 % del área útil"
    return 0.10 * area_util, "LUC Art. 2 e) y RLUC Art. 55: 10 % del área útil"


def equipamiento_requerido(zona: str, n_lotes: int, tipo: str = "residencial") -> tuple[float, str, bool]:
    """(m² requeridos, base, obligatorio_claro)."""
    if zona == "amss":
        por_lote = 6.4 if tipo == "interes_social" else 8.0
        return por_lote * n_lotes, f"Reglamento LDOTAMSS Art. 66: {por_lote:g} m² por vivienda", True
    if n_lotes >= 80:
        return 8.0 * n_lotes, "RLUC Art. 65: 8 m² por lote en parcelaciones de 80 lotes o más", True
    return 8.0 * n_lotes, ("RLUC Art. 65 lo exige desde 80 lotes; la LUC Art. 2 g) pide 8 m² por lote para escuela: "
                           "confirmar con la DOT"), False


def _ref(base: str) -> str:
    """Referencia corta para tablas: 'RLUC 84', 'LUC 2 e)', 'OPAMSS 64'."""
    import re

    out = []
    for ley, art in re.findall(r"(LUC|RLUC|LDOTAMSS)\s+Art\.\s*(\d+(?:\s*[a-z]\))?)", base):
        out.append(f"{'OPAMSS' if ley == 'LDOTAMSS' else ley} {art}")
    return ", ".join(dict.fromkeys(out)) or base[:14]


def _fila(regla, base, requerido, proyecto, estado, nota=""):
    return {"regla": regla, "base": base, "ref": _ref(base), "requerido": requerido, "proyecto": proyecto,
            "estado": estado, "nota": nota}


def _pendiente_calle(poly: Polygon, altura) -> tuple[float, float]:
    """Largo y pendiente máxima (en %) a lo largo del eje de una calle."""
    r = list(poly.minimum_rotated_rectangle.exterior.coords)
    lados = sorted([(r[i], r[i + 1]) for i in range(4)], key=lambda l: -math.dist(*l))
    (a1, a2), (b1, b2) = lados[0], lados[1]
    # eje: de la mitad de un lado corto a la mitad del otro
    cortos = sorted([(r[i], r[i + 1]) for i in range(4)], key=lambda l: math.dist(*l))[:2]
    p = ((cortos[0][0][0] + cortos[0][1][0]) / 2, (cortos[0][0][1] + cortos[0][1][1]) / 2)
    q = ((cortos[1][0][0] + cortos[1][1][0]) / 2, (cortos[1][0][1] + cortos[1][1][1]) / 2)
    eje = LineString([p, q])
    largo = eje.length
    if altura is None or largo < 5:
        return largo, math.nan
    n = max(2, int(largo / 10) + 1)
    zs = [altura(*eje.interpolate(largo * k / (n - 1)).coords[0]) for k in range(n)]
    tramo = largo / (n - 1)
    pend = max(abs(zs[k + 1] - zs[k]) / tramo * 100 for k in range(n - 1))
    return largo, pend


def revisar(lot, terreno, malla=None, ficha: dict | None = None) -> list[dict]:
    """Devuelve el cuadro de cumplimiento del prediseño."""
    ficha = ficha or {}
    zona = ficha.get("zona")
    filas = []
    s = lot.stats
    n = s["lotes"]
    area_util = s["area_util"]
    verde_rec = s["area_verde"]
    proteccion = s.get("area_proteccion", 0.0)

    # 1. Área verde total
    if not zona:
        filas.append(_fila("Área verde total", "LUC Art. 2 e)", "depende de la zona", f"{verde_rec + proteccion:,.0f} m²",
                           "falta_dato", "Falta saber si el terreno está en el AMSS, en zona urbana o fuera de centros poblados."))
    elif zona == "reserva_ecologica":
        filas.append(_fila("Uso en reserva ecológica", "RLUC Art. 44", "lote ≥ 4,000 m², techar ≤ 10 %",
                           f"lote mínimo {s['lote_min']:,.0f} m²", "cumple" if s["lote_min"] >= 4000 else "no_cumple",
                           "Densidad máxima de 10 habitantes por hectárea."))
    else:
        req, base = area_verde_requerida(zona, n, area_util)
        total = verde_rec + proteccion
        estado = "cumple" if total >= req - 0.5 else "no_cumple"
        nota = ""
        if zona in ("urbano", "rural") and area_util <= 5000:
            nota = "Con área útil de 5,000 m² o menos, el RLUC Art. 55 exime de donar zona verde; confirmar con la DOT."
            if estado == "no_cumple":
                estado = "revisar"
        if proteccion:
            nota = (nota + " " if nota else "") + "Incluye zona de protección como área verde ecológica (RLUC Arts. 52 y 61)."
        filas.append(_fila("Área verde total", base, f"{req:,.0f} m²", f"{total:,.0f} m²", estado, nota))
        # 2. Parte recreativa
        pct = 50 if zona == "amss" else 60
        base_r = "Reglamento LDOTAMSS Art. 64: ≥ 50 % del área verde" if zona == "amss" else "RLUC Art. 59: 60 % del área verde total"
        req_r = req * pct / 100
        filas.append(_fila("Área verde recreativa", base_r, f"{req_r:,.0f} m²", f"{verde_rec:,.0f} m²",
                           "cumple" if verde_rec >= req_r - 0.5 else "no_cumple",
                           "Debe quedar equipada para uso recreativo (parque, juegos, senderos)."))

    # 3. Distancia al área verde recreativa
    verdes = [v["poly"].buffer(0) for v in lot.verdes if v["area"] >= 50]
    if verdes and lot.lotes:
        try:
            u = unary_union(verdes)
        except Exception:  # geometría con detalles mínimos inválidos: se mide contra cada pieza
            u = None
        if u is None:
            lejos = max(min(l["poly"].centroid.distance(v) for v in verdes) for l in lot.lotes)
        else:
            lejos = max(l["poly"].centroid.distance(u) for l in lot.lotes)
        filas.append(_fila("Distancia al área verde", "RLUC Art. 58", "≤ 400 m", f"{lejos:,.0f} m",
                           "cumple" if lejos <= 400 else "no_cumple"))

    # 4. Equipamiento social
    if zona:
        req_e, base_e, claro = equipamiento_requerido(zona, n, ficha.get("tipo", "residencial"))
        eq = s.get("area_equipamiento", 0.0)
        if eq >= req_e - 0.5:
            estado = "cumple"
        else:
            estado = "no_cumple" if claro else "revisar"
        filas.append(_fila("Equipamiento social", base_e, f"{req_e:,.0f} m²", f"{eq:,.0f} m²", estado))

    # 5. Pendiente de calles
    if malla is not None and lot.calles:
        peor = None
        ent, ret = lot.stats.get("entrada_calle"), lot.stats.get("retorno_centro")
        ejes = None
        if lot.stats.get("modo") == "calle_central" and ent and ret:  # el eje real de la calle, no la caja que la envuelve
            from .altimetria import pendiente_eje

            largo_, _, pend_ = pendiente_eje(malla, LineString([ent, ret]))
            ejes = [(largo_, pend_)]
        for largo, pend in ejes or [_pendiente_calle(c, malla.altura) for c in lot.calles]:
            if not math.isfinite(pend):
                continue
            lim = next((l for d, l in LIMITES_PENDIENTE_CALLE if largo <= d), None)
            exceso = (pend - lim) if lim is not None else math.inf
            if peor is None or exceso > peor[0]:
                peor = (exceso, largo, pend, lim)
        if peor:
            _, largo, pend, lim = peor
            if lim is None:
                filas.append(_fila("Pendiente de calles", "RLUC Art. 84", "calle > 500 m: diseño especial",
                                   f"{pend:.1f} % en {largo:,.0f} m", "revisar"))
            else:
                filas.append(_fila("Pendiente de calles", "RLUC Art. 84", f"≤ {lim:g} % (tramo de {largo:,.0f} m)",
                                   f"{pend:.1f} % (peor calle)", "cumple" if pend <= lim else "no_cumple",
                                   "Máximos: 18 % hasta 100 m, 12 % hasta 300 m, 8 % hasta 500 m. Pendientes "
                                   "satelitales aproximadas."))

    # 6. Largo de bloque
    if lot.calles:
        largo_max = max(_pendiente_calle(c, None)[0] for c in lot.calles)
        filas.append(_fila("Largo de bloque", "RLUC Art. 76", "≤ 500 m", f"{largo_max:,.0f} m",
                           "cumple" if largo_max <= 500 else "no_cumple"))

    # 7. Quebradas y ríos
    for z in ficha.get("protecciones") or []:
        if z.get("tipo") == "rio":
            filas.append(_fila(f"Protección de río ({z.get('lado', '')})", "RLUC Art. 50 b)",
                               "estudio hidrológico", f"franja de {z.get('ancho', 0):g} m", "revisar",
                               "El ancho lo fija un estudio de crecidas; como referencia se usan 50 m desde el eje."))
        elif z.get("profundidad"):
            ancho = 1.5 * float(z["profundidad"])
            filas.append(_fila(f"Protección de quebrada ({z.get('lado', '')})", "RLUC Art. 50 a)",
                               f"1.5 × {float(z['profundidad']):g} m = {ancho:g} m", f"franja de {z.get('ancho', ancho):g} m",
                               "cumple", "Medir la profundidad real en campo; solo se reduce con obras (Art. 51)."))
        else:
            filas.append(_fila(f"Protección de quebrada ({z.get('lado', '')})", "RLUC Art. 50 a)",
                               "1.5 × profundidad", "sin franja", "falta_dato", "Falta la profundidad de la quebrada."))
    if ficha.get("quebrada_sin_datos"):
        filas.append(_fila("Quebrada colindante", "RLUC Art. 50", "franja de protección", "sin definir", "falta_dato",
                           "El plano menciona una quebrada: falta saber de qué lado y qué tan profunda es."))

    # 8. Lotes muy inclinados (criterio de diseño)
    inclinados = sum(1 for l in lot.lotes if l["pendiente"] is not None and l["pendiente"] > 30)
    if malla is not None:
        filas.append(_fila("Lotes con pendiente > 30 %", "RLUC Art. 49 (taludes y muros)", "terrazas con muro o talud 1.5:1",
                           f"{inclinados} de {n}", "cumple" if inclinados == 0 else "revisar",
                           "Cortes o rellenos de más de 1 m entre lotes requieren muro o talud."))

    # 8b. Caídas fuertes que se dejaron fuera de los lotes (criterio de diseño)
    if malla is not None and ficha.get("barrancos_m2"):
        filas.append(_fila("Caídas fuertes (pendiente > 50 %)", "Criterio de diseño", "no lotificar",
                           f"{ficha['barrancos_m2']:,.0f} m² fuera de los lotes", "cumple",
                           "Se dejan como área verde o de protección. Con relieve satelital el borde es aproximado: "
                           "el topógrafo debe marcar el borde real del barranco."))

    # 9. Acceso
    if lot.stats.get("calles_desconectadas"):
        filas.append(_fila("Acceso a todas las calles", "Criterio de diseño", "todas conectadas a la calle pública",
                           f"{lot.stats['calles_desconectadas']} sin conexión", "no_cumple",
                           "Hay lotes que dan a una calle que no llega a la calle pública."))
    if ficha.get("cerrado") and lot.stats.get("accesos", 0) > 1:
        filas.append(_fila("Un solo acceso (proyecto cerrado)", "Criterio de diseño", "1 acceso con portón",
                           f"{lot.stats['accesos']} accesos", "revisar",
                           "Para un proyecto cerrado conviene unir las calles por dentro y dejar una sola entrada."))

    # 11. Figura de venta
    if ficha.get("figura") == "proindiviso":
        filas.append(_fila("Figura de venta", "Ley Esp. de Lotificaciones Art. 25", "lotificación o condominio con permiso",
                           "venta en proindiviso", "no_cumple",
                           "Vender porciones en proindiviso para formar lotes se presume lotificación, y la Ley de "
                           "Delimitación de Derechos Proindivisos obliga al notario y al registrador a detenerla."))
    elif ficha.get("figura") == "condominio":
        filas.append(_fila("Área verde en condominio", "Reglamento LDOTAMSS Art. 65" if zona == "amss" else "Por confirmar con la DOT",
                           "área común de los condóminos", "área común", "cumple" if zona == "amss" else "revisar",
                           "En condominio el área verde no se dona: queda como área común y la mantienen los dueños."))

    # 10. Calles sin salida
    if lot.stats.get("calles_sin_salida"):
        d_ret = lot.stats.get("retorno_diametro_m")
        filas.append(_fila("Calles sin salida", "RLUC Art. 87", "plazoleta de retorno",
                           f"redondel de {d_ret:.0f} m" if d_ret else "por definir", "revisar",
                           "Los retornos (redondel, T o L) solo se permiten en vías de reparto y accesos rodados."))

    # 12. Lo que dice la OPAMSS del lugar (Esquema Director, consultado en su geovisor)
    if ficha.get("opamss"):
        filas += filas_opamss(lot, ficha["opamss"])
    return filas


ED = "Esquema Director AMSS"
TRATAMIENTOS_SENSIBLES = {
    "CAM": "conservación ambiental: el Esquema Director protege esta zona y casi no admite urbanizar",
    "RAM": "recuperación ambiental: zona a recuperar, no a urbanizar",
    "MIT": "mitigación: zona con riesgo que pide obras de mitigación antes de construir",
    "DAG": "desarrollo agropecuario: zona para producción, no para lotes habitacionales",
}


def _codigo_tratamiento(texto) -> str | None:
    import re

    m = re.search(r"\((\w+)\)+\s*$", str(texto or ""))
    return m.group(1).upper() if m else None


def _casas(lot) -> list[float]:
    """Área de la casa tipo (un nivel) de cada lote, como la dibuja el motor."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from .modelo3d import huella_casa

    calles = unary_union(lot.calles) if lot.calles else None
    out = []
    for l in lot.lotes:
        try:
            esq = huella_casa(l, calles, lot.angulo)
        except Exception:  # noqa: BLE001
            esq = None
        out.append(Polygon(esq).area if esq else 0.0)
    return out


def filas_opamss(lot, op: dict) -> list[dict]:
    """Filas del cuadro con lo que dice el geovisor de la OPAMSS: clasificación y tratamiento del suelo, alturas,
    impermeabilización y edificabilidad, vías proyectadas y peligros. Fuente en cada fila."""
    filas = []
    zonas = [z for z in op.get("uso_suelo") or [] if z.get("pct", 0) > 0]
    if zonas:
        txt = " · ".join(f"{z['pct']:.0f} % {z.get('clasificacion') or z.get('codigo')}" for z in zonas)
        nur = sum(z["pct"] for z in zonas if z.get("codigo") == "NUR")
        rur = sum(z["pct"] for z in zonas if z.get("codigo") == "RUR")
        if nur >= 5:
            estado, nota = "no_cumple", (f"El {nur:.0f} % del terreno es suelo no urbanizable: ahí no se puede lotificar. "
                                         "Confirmarlo con una calificación de lugar en la OPAMSS.")
        elif rur >= 5:
            estado, nota = "revisar", ("En suelo rural el Esquema Director limita los usos: pedir calificación de lugar a "
                                       "la OPAMSS para saber si admite lotificación y con qué densidad.")
        else:
            estado, nota = "cumple", ""
        filas.append(_fila("Clasificación del suelo", f"{ED}, Anexo 25", "suelo urbano o urbanizable", txt, estado, nota))

        sens = [(z, _codigo_tratamiento(z.get("tratamiento"))) for z in zonas]
        sens = [(z, c) for z, c in sens if c in TRATAMIENTOS_SENSIBLES and z["pct"] >= 5]
        trat = " · ".join(f"{z['pct']:.0f} % {z.get('tratamiento')}" for z in zonas if z.get("tratamiento"))
        if trat:
            filas.append(_fila("Tratamiento urbanístico", f"{ED}, Anexo 26", "compatible con lotificación", trat,
                               "revisar" if sens else "cumple",
                               " ".join(f"{z['pct']:.0f} % en {TRATAMIENTOS_SENSIBLES[c]}." for z, c in sens)))

        z0 = zonas[0]  # la zona con más área manda en las cifras por lote
        casas = _casas(lot)
        areas = [l["area"] for l in lot.lotes]
        if z0.get("altura_base"):
            amp = f" ({z0['altura_ampliada']} con ampliación)" if z0.get("altura_ampliada") else ""
            filas.append(_fila("Altura máxima", f"{ED}, Anexo 32", f"{z0['altura_base']} niveles{amp}",
                               "casa tipo de 1 nivel", "cumple" if float(z0["altura_base"]) >= 1 else "revisar"))
        if z0.get("impermeabilizacion_base") and areas:
            lim = float(z0["impermeabilizacion_base"])
            pcts = [100 * c / a for c, a in zip(casas, areas) if a > 0 and c > 0]
            peor = max(pcts) if pcts else 0.0
            amp = (f" ({z0['impermeabilizacion_ampliada']} % con ampliación)"
                   if z0.get("impermeabilizacion_ampliada") else "")
            filas.append(_fila("Impermeabilización por lote", f"{ED}, Anexo 33", f"hasta {lim:g} % del lote{amp}",
                               f"casa tipo: hasta {peor:.0f} % del lote", "cumple" if peor <= lim else "revisar",
                               f"En un lote de {sum(areas) / len(areas):,.0f} m² se pueden techar o pavimentar hasta "
                               f"{lim / 100 * sum(areas) / len(areas):,.0f} m² (casa, cochera, patios)."))
        if z0.get("edificabilidad_base") and areas:
            ien = float(z0["edificabilidad_base"])
            usado = max((c / a for c, a in zip(casas, areas) if a > 0), default=0.0)
            filas.append(_fila("Edificabilidad (IEN)", f"{ED}, Anexo 34", f"hasta {ien:g} × área del lote",
                               f"casa tipo: {usado:.2f} × área", "cumple" if usado <= ien else "revisar",
                               f"En el lote promedio caben hasta {ien * sum(areas) / len(areas):,.0f} m² de construcción."))

    vias = op.get("vias") or {}
    pr = vias.get("proyectada")
    if pr and pr.get("cruza_m", 0) > 1:
        filas.append(_fila("Vía proyectada", f"{ED}, Anexo 28", "respetar el derecho de vía",
                           f"cruza el terreno ({pr['cruza_m']:,.0f} m)", "revisar",
                           "El Esquema Director proyecta una vía que pasa por el terreno: puede quitar lotes o cambiar el "
                           "trazo. Pedir a la OPAMSS el derecho de vía."))
    elif pr and pr.get("distancia_m") is not None and pr["distancia_m"] <= 30:
        filas.append(_fila("Vía proyectada", f"{ED}, Anexo 28", "respetar el derecho de vía",
                           f"a {pr['distancia_m']:,.0f} m del terreno", "revisar",
                           "Hay una vía proyectada pegada al terreno: confirmar su ancho con la OPAMSS."))

    peligros = []
    if op.get("zona_tectonica"):
        peligros.append("zona de mayor densidad tectónica")
    if op.get("peligro_volcanico"):
        peligros.append("zona de mayor peligro volcánico")
    altas = [p for p in op.get("peligrosidad_superficial") or []
             if str(p.get("valor", "")).strip().lower() in ("alta", "muy alta") and p.get("pct", 0) >= 5]
    for p in altas:
        peligros.append(f"{p['pct']:.0f} % con peligrosidad {str(p['valor']).lower()} por dinámica superficial")
    if peligros:
        filas.append(_fila("Peligros del lugar", f"{ED}, Anexos 22 y 28", "estudios según el peligro",
                           "; ".join(peligros), "revisar",
                           "Pedir estudio geotécnico (suelos y estabilidad de taludes) y diseño sísmico; en zonas de "
                           "dinámica superficial, obras de drenaje y protección de laderas."))
    elif op.get("uso_suelo") is not None and op.get("peligrosidad_superficial") is not None:
        filas.append(_fila("Peligros del lugar", f"{ED}, Anexos 22 y 28", "sin peligro alto", "no se marca peligro alto",
                           "cumple"))

    falta = op.get("no_se_pudo_consultar") or []
    if falta:
        filas.append(_fila("Datos de la OPAMSS", "Geovisor OPAMSS", "consulta completa",
                           "no respondió: " + ", ".join(f.replace("_", " ") for f in falta), "falta_dato",
                           "El servidor de la OPAMSS no contestó esa parte; volver a consultar."))
    for f in filas:
        f["fuente"] = op.get("fuente")
    return filas


def resumen(filas: list[dict]) -> dict:
    cuenta = {k: sum(1 for f in filas if f["estado"] == k) for k in ESTADOS}
    return {"cumple": cuenta["cumple"], "no_cumple": cuenta["no_cumple"], "revisar": cuenta["revisar"],
            "falta_dato": cuenta["falta_dato"]}
