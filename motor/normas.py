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
        for c in lot.calles:
            largo, pend = _pendiente_calle(c, malla.altura)
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
        filas.append(_fila("Calles sin salida", "RLUC Art. 87", "plazoleta de retorno", "por definir", "revisar",
                           "Los retornos (redondel, T o L) solo se permiten en vías de reparto y accesos rodados."))
    return filas


def resumen(filas: list[dict]) -> dict:
    cuenta = {k: sum(1 for f in filas if f["estado"] == k) for k in ESTADOS}
    return {"cumple": cuenta["cumple"], "no_cumple": cuenta["no_cumple"], "revisar": cuenta["revisar"],
            "falta_dato": cuenta["falta_dato"]}
