"""Diseñador: varias maneras de lotificar el mismo terreno, medidas por el motor, y un modelo que piensa elige y explica.

"La IA piensa, el motor mide": el motor arma las variantes con las palancas que sabe mover (lotes por lado, en qué
terreno va la calle, tamaño de lote, calle central o retícula) y mide cada una (lotes, área vendible, pendiente de la
calle, muros, tierra que se mueve, normas y la revisión geométrica). El modelo no inventa números ni geometría: lee
la tabla y recomienda dos o tres con sus pros y contras. Sin modelo, se ordenan con una regla fija.
"""
from __future__ import annotations

import json
import logging
import math

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

log = logging.getLogger(__name__)


def variantes(lotes_param: dict, ficha: dict, dos_terrenos: bool, modo: str, rango,
              medidas: dict | None = None) -> list[dict]:
    """Las variantes a medir: la actual y cambios de una sola palanca, sin repetir. Cada una es {id, cambio, param}.
    `medidas` = frente, fondo y área mínima con que se diseña hoy (de la ficha): los cambios de tamaño parten de ahí."""
    base = dict(lotes_param or {})
    hoy = {**(medidas or {}), **{k: v for k, v in base.items() if k in ("frente", "fondo", "area_minima") and v}}
    out = [{"id": "A", "cambio": "el prediseño actual", "param": base}]
    vistos = {json.dumps(base, sort_keys=True, default=str)}

    def agregar(cambio, **kw):
        p = {**base, **kw}
        clave = json.dumps(p, sort_keys=True, default=str)
        if clave in vistos:
            return
        vistos.add(clave)
        out.append({"id": chr(ord("A") + len(out)), "cambio": cambio, "param": p})

    if modo == "calle_central":
        a, b = (list(rango) + [None, None])[:2] if rango else (4, 5)
        b = b or a
        if a > 2:
            agregar(f"menos lotes y más grandes ({a - 1}–{b - 1} por lado)", lotes_por_lado=[a - 1, b - 1])
        agregar(f"más lotes ({a + 1}–{b + 1} por lado)", lotes_por_lado=[a + 1, b + 1])
        if dos_terrenos:
            actual = base.get("calle_sobre") or ficha.get("calle_sobre") or "terreno_mas_grande"
            for sobre, txt in (("terreno_mas_grande", "la calle dentro del terreno más grande"),
                               ("terreno_mas_chico", "la calle dentro del terreno más chico"),
                               ("centro_del_lindero", "la calle centrada sobre el lindero común")):
                if sobre != actual:
                    agregar(txt, calle_sobre=sobre)
        if not (lotes_param or {}).get("distribucion") and not ficha.get("distribucion"):
            agregar("retícula de calles en lugar de una calle central", distribucion="automatica")
    else:
        fr = float(hoy.get("frente") or 10)
        fo = float(hoy.get("fondo") or 25)
        am = float(hoy.get("area_minima") or fr * fo * 0.9)
        agregar(f"lotes más grandes (~{am * 1.25 / 0.9:,.0f} m²)", frente=round(fr * 1.12, 1), fondo=round(fo * 1.12, 1),
                area_minima=round(am * 1.25))
        agregar(f"lotes más chicos (~{am * 0.8 / 0.9:,.0f} m²)", frente=round(fr * 0.9, 1), fondo=round(fo * 0.9, 1),
                area_minima=round(am * 0.8))
        agregar("una sola calle central con lotes a los dos lados", distribucion="calle_central")
    return out[:7]


def medir(t, lot, malla, ficha: dict, filas: list) -> dict:
    """Las cifras que deciden: lotes, área vendible, calle, muros, tierra, normas y revisión geométrica."""
    from . import altimetria, revisor

    st = lot.stats
    m = {"lotes": st["lotes"], "area_vendible_m2": round(st["area_util"]),
         "vendible_pct": round(st["area_util"] / st["area_total"] * 100, 1),
         "lote_min_m2": round(st["lote_min"]), "lote_max_m2": round(st["lote_max"]),
         "no_cumple": [f["regla"] for f in filas or [] if f.get("estado") == "no_cumple"]}
    calle = next((f for f in filas or [] if "endiente" in f.get("regla", "") and "calle" in f.get("regla", "").lower()), None)
    if calle:
        m["pendiente_calle"] = f"{calle.get('proyecto', '')} ({calle.get('estado', '')})"
    terr = None
    if malla is not None:
        try:
            terr = altimetria.terrazas(lot, malla)
            m.update(muros_m2=sum(p["muro_m2"] for p in terr), muro_max_m=max(p["muro_max"] for p in terr),
                     tierra_m3=sum(p["corte_m3"] + p["relleno_m3"] for p in terr),
                     lotes_muro_alto=[p["id"] for p in terr if p["muro_max"] > altimetria.MURO_ESTRUCTURAL])
            pend = [l["relieve"]["pend_media"] for l in lot.lotes if (l.get("relieve") or {}).get("pend_media") is not None]
            if pend:
                m["lotes_de_mas_de_30_pct"] = sum(1 for p in pend if p > altimetria.PEND_CONSTRUIBLE)
        except Exception:  # noqa: BLE001
            log.exception("medir terrazas")
    try:
        ch = revisor.revisar(t, lot, malla, ficha, terr)
        m["fallas_graves"] = [c["detalle"] for c in revisor.fallas(ch)]
    except Exception:  # noqa: BLE001
        log.exception("medir revisión")
    return m


def puntaje(m: dict) -> tuple:
    """Orden sin modelo: primero lo que no tiene fallas graves ni normas incumplidas, luego más área vendible por
    lote útil, luego menos muro."""
    return (-len(m.get("fallas_graves") or []), -len(m.get("no_cumple") or []), m["area_vendible_m2"],
            m["lotes"], -m.get("muros_m2", 0))


INSTRUCCION_DISENADOR = """Eres un urbanista y desarrollador inmobiliario con mucha obra en laderas de El Salvador. Un
motor geométrico ya armó y midió varias maneras de lotificar el mismo terreno. Las cifras son exactas: no las
recalcules ni inventes otras. Tu trabajo es pensar como el dueño del proyecto y su arquitecto y elegir.

Terreno y proyecto:
{contexto}

Variantes medidas (JSON):
{variantes}

Criterios, en este orden: (1) nada con fallas graves ni normas incumplidas, salvo que todas las tengan; (2) que se
pueda construir y vender: lotes que no queden demasiado inclinados, muros razonables (más de 3 m pide diseño
estructural y encarece), calle que cumpla pendiente; (3) área vendible y número de lotes; (4) que respete lo que el
cliente pidió. Recomienda dos o tres variantes, de la mejor a la peor, con un título corto, por qué conviene y su
contra principal, en frases cortas y en español sencillo. Si ninguna sirve, dilo en "advertencias"."""

ESQUEMA_DISENADOR = {
    "type": "object",
    "properties": {
        "recomendadas": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "titulo": {"type": "string"}, "por_que": {"type": "string"},
            "contra": {"type": "string"}}, "required": ["id", "titulo", "por_que", "contra"]}},
        "advertencias": {"type": "array", "items": {"type": "string"}}},
    "required": ["recomendadas"],
}


def recomendar(pensar, medidas: list[dict], contexto: str) -> dict:
    """`pensar(texto, esquema) -> dict` (modelo que piensa). Sin modelo o si falla, la regla fija."""
    ids = {v["id"] for v in medidas}
    if pensar is not None:
        try:
            tabla = [{"id": v["id"], "cambio": v["cambio"], **v["medidas"]} for v in medidas]
            r = pensar(INSTRUCCION_DISENADOR.format(contexto=contexto, variantes=json.dumps(tabla, ensure_ascii=False)),
                       ESQUEMA_DISENADOR)
            rec = [x for x in (r or {}).get("recomendadas") or [] if x.get("id") in ids][:3]
            if rec:
                return {"recomendadas": rec, "advertencias": (r or {}).get("advertencias") or [], "por": "modelo"}
        except Exception:  # noqa: BLE001
            log.exception("diseñador")
    orden = sorted(medidas, key=lambda v: puntaje(v["medidas"]), reverse=True)[:3]
    return {"recomendadas": [{"id": v["id"], "titulo": v["cambio"], "por_que": "", "contra": ""} for v in orden],
            "advertencias": [], "por": "regla"}


# ---------------------------------------------------------------------------- imagen comparativa
def imagen(t, opciones: list[tuple[str, str, object, dict]], ruta: str) -> str:
    """Las variantes recomendadas lado a lado (planta simple + cifras), para verlas en el celular.
    `opciones` = [(id, titulo, Resultado, medidas)]. El dibujo se gira para que el lado largo del terreno quede
    horizontal (con flecha de norte); si el terreno es muy alargado, las opciones van una debajo de otra."""
    from shapely import affinity

    from .presentacion import _fuente

    def fuente(tam, negrita=False):
        return _fuente(int(tam), negrita)

    terreno = Polygon(t.puntos).buffer(0)
    rect = terreno.minimum_rotated_rectangle
    cs = list(rect.exterior.coords)
    lados = [(math.hypot(b[0] - a[0], b[1] - a[1]), math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
             for a, b in zip(cs, cs[1:])]
    largo, ang = max(lados)
    corto = min(l for l, _ in lados) or 1
    ang = ((ang + 90) % 180) - 90                       # girar lo menos posible
    centro = terreno.centroid

    def gir(g):
        return affinity.rotate(g, -ang, origin=centro)

    ter = gir(terreno)
    minx, miny, maxx, maxy = ter.bounds
    n = len(opciones)
    filas_layout = largo / corto > 2.2
    if filas_layout:
        W, alto_fila = 2000, 360
        H = 30 + alto_fila * n
    else:
        W, H = 760 * n, 980
    im = Image.new("RGB", (W, H), (250, 249, 245))
    d = ImageDraw.Draw(im)
    for k, (id_, titulo, lot, m) in enumerate(opciones):
        if filas_layout:
            y0 = 30 + alto_fila * k
            caja = (640, y0 + 30, W - 90, y0 + alto_fila - 30)
            xt, yt, yc = 40, y0 + 6, y0 + 112
        else:
            x0 = 760 * k
            caja = (x0 + 40, 150, x0 + 720, 700)
            xt, yt, yc = x0 + 40, 30, 730
        esc = min((caja[2] - caja[0]) / (maxx - minx), (caja[3] - caja[1]) / (maxy - miny))
        ox = caja[0] + ((caja[2] - caja[0]) - (maxx - minx) * esc) / 2
        oy = caja[1] + ((caja[3] - caja[1]) - (maxy - miny) * esc) / 2

        def P(x, y):
            return ox + (x - minx) * esc, oy + (maxy - y) * esc

        def poli(g, **kw):
            for p in getattr(gir(g), "geoms", [gir(g)]):
                if p.geom_type == "Polygon" and not p.is_empty:
                    d.polygon([P(*q) for q in p.exterior.coords], **kw)

        d.text((xt, yt), f"Opción {k + 1}", font=fuente(34, True), fill=(30, 30, 30))
        d.text((xt, yt + 50), titulo[:48], font=fuente(22), fill=(80, 80, 80))
        poli(terreno, fill=(236, 232, 220), outline=(40, 40, 40), width=3)
        for p in lot.protecciones:
            poli(p, fill=(150, 190, 140))
        for v in lot.verdes:
            poli(v["poly"], fill=(176, 208, 150))
        for c in lot.calles:
            poli(c, fill=(160, 160, 160))
        tam = max(14, min(24, esc * 6))
        for l in lot.lotes:
            poli(l["poly"], fill=(232, 210, 170), outline=(90, 70, 50), width=2)
            c = gir(l["poly"]).representative_point()
            d.text(P(c.x, c.y), str(l["id"]), font=fuente(tam, True), fill=(60, 45, 30), anchor="mm")
        # flecha de norte (girada con el dibujo)
        fx, fy = (caja[2] + 45, caja[1] + 40) if filas_layout else (caja[2] - 20, caja[1] - 20)
        a = math.radians(-ang)
        nx, ny = -math.sin(a), -math.cos(a)                # el norte en pantalla (y crece hacia abajo)
        d.line([(fx - nx * 22, fy - ny * 22), (fx + nx * 22, fy + ny * 22)], fill=(40, 40, 40), width=3)
        d.polygon([(fx + nx * 30, fy + ny * 30), (fx + nx * 14 - ny * 8, fy + ny * 14 + nx * 8),
                   (fx + nx * 14 + ny * 8, fy + ny * 14 - nx * 8)], fill=(40, 40, 40))
        d.text((fx + nx * 46, fy + ny * 46), "N", font=fuente(20, True), fill=(40, 40, 40), anchor="mm")
        filas = [(f"{m['lotes']}", "lotes"), (f"{m['area_vendible_m2']:,} m²", f"vendibles ({m['vendible_pct']:.0f} %)"),
                 (f"{m['lote_min_m2']:,}–{m['lote_max_m2']:,} m²", "por lote")]
        if "muros_m2" in m:
            filas.append((f"{m['muros_m2']:,} m²", f"de muro (máx. {m['muro_max_m']:.1f} m)"))
        if m.get("fallas_graves") or m.get("no_cumple"):
            txt = "; ".join((m.get("fallas_graves") or []) + (m.get("no_cumple") or []))
            tope = 58 if filas_layout else 40
            if len(txt) > tope:
                txt = txt[:tope].rsplit(" ", 1)[0].rstrip(":;,") + "…"
            filas.append(("revisar", txt))
        y = yc
        for a_, b_ in filas:
            d.text((xt, y), a_, font=fuente(26 if not filas_layout else 24, True), fill=(30, 30, 30))
            d.text((xt + 250, y + 4), b_, font=fuente(21 if not filas_layout else 19), fill=(90, 90, 90))
            y += 44 if not filas_layout else 40
        if k and not filas_layout:
            d.line([(760 * k, 30), (760 * k, H - 30)], fill=(210, 205, 195), width=2)
        if k and filas_layout:
            d.line([(30, 30 + alto_fila * k), (W - 30, 30 + alto_fila * k)], fill=(210, 205, 195), width=2)
    im.save(ruta, "JPEG", quality=90)
    return ruta


def contexto(t, malla, ficha: dict, lotes_param: dict) -> str:
    from . import altimetria

    lineas = [f"- área del terreno: {Polygon(t.puntos).area:,.0f} m²"]
    if malla is not None:
        try:
            r = altimetria.resumen(malla, t.puntos)
            lineas.append(f"- relieve: {r['desnivel']:.0f} m de desnivel, pendiente media {r['pend_media']:.0f} %, baja hacia "
                          f"el {r['baja_hacia']}")
        except Exception:  # noqa: BLE001
            pass
    q = [f"{x.get('tipo')} al {x.get('lado')}" for x in ficha.get("quebradas") or [] if x.get("lado")]
    if q:
        lineas.append("- colinda con " + ", ".join(q) + " (franja protegida)")
    for k, txt in (("acceso_lado", "acceso por el"), ("tipo", "tipo de proyecto"), ("zona", "zona"),
                   ("lote_objetivo_m2", "lote buscado (m²)")):
        if ficha.get(k):
            lineas.append(f"- {txt}: {ficha[k]}")
    if lotes_param:
        lineas.append("- lo que pidió el cliente: " + json.dumps(lotes_param, ensure_ascii=False, default=str))
    return "\n".join(lineas)


def _r(v):
    return round(v, 1) if isinstance(v, float) and math.isfinite(v) else v
