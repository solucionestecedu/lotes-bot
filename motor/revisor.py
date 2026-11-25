"""Revisor automático antes de entregar.

Dos capas, con papeles distintos:
- la geometría la revisa el motor con reglas exactas (`revisar`): lotes dentro del terreno y sin traslapes, la franja
  de la quebrada cubre toda la orilla y ningún lote ni calle se mete en ella, lotes sin huecos, con salida a la calle,
  con área y ancho mínimos, casas al frente, muros razonables. Lo grave se avisa al usuario.
- las láminas y las vistas las revisa un modelo que piensa (`revisar_laminas`), con una lista de chequeo y las
  imágenes: textos encimados, letreros en la pieza equivocada, vistas que no respetan el trazo (número de casas, calle
  recta, quebrada del lado correcto). La IA solo señala; lo que dice va como "para revisar", nunca cambia números.
"""
from __future__ import annotations

import io
import logging
import math

from shapely.geometry import Polygon
from shapely.ops import unary_union

log = logging.getLogger(__name__)

ANCHO_MINIMO = 7.0        # m: un lote más angosto que esto en alguna parte no sirve para una casa
MURO_MAXIMO = 6.0         # m: un muro más alto es una obra mayor; se avisa como grave


def _chequeo(clave: str, ok: bool, detalle: str, grave: bool = True) -> dict:
    return {"chequeo": clave, "ok": bool(ok), "grave": bool(grave and not ok), "detalle": detalle}


def _ids(xs) -> str:
    xs = list(xs)
    return ", ".join(str(x) for x in xs[:12]) + ("…" if len(xs) > 12 else "")


def revisar(t, lot, malla=None, ficha: dict | None = None, terr: list | None = None) -> list[dict]:
    """Revisa la geometría del prediseño. Devuelve un chequeo por regla: {chequeo, ok, grave, detalle}."""
    from .lotificacion import LADOS, orilla_hacia

    out: list[dict] = []
    if lot is None or not lot.lotes:
        return out
    terreno = Polygon(t.puntos).buffer(0)
    lotes = lot.lotes
    calles = unary_union(lot.calles).buffer(0) if lot.calles else None
    prot = unary_union(lot.protecciones).buffer(0) if lot.protecciones else None
    P = lot.parametros or {}

    # 1. dentro del terreno y sin traslapes
    fuera = [l["id"] for l in lotes if l["poly"].difference(terreno.buffer(0.3)).area > 1.0]
    out.append(_chequeo("lotes_dentro_del_terreno", not fuera,
                        f"lotes que se salen del terreno: {_ids(fuera)}" if fuera else "todos los lotes dentro del terreno"))
    tras = []
    for i, a in enumerate(lotes):
        for b in lotes[i + 1:]:
            if a["poly"].intersection(b["poly"]).area > 1.0:
                tras.append(f"{a['id']}-{b['id']}")
    if calles is not None:
        tras += [f"{l['id']}-calle" for l in lotes if l["poly"].intersection(calles).area > 1.0]
    out.append(_chequeo("sin_traslapes", not tras,
                        f"se traslapan: {_ids(tras)}" if tras else "lotes y calle sin traslapes"))
    if calles is not None:
        c_fuera = calles.difference(terreno.buffer(0.3)).area
        salida = calles.intersection(terreno.exterior.buffer(0.6)).length if calles.geom_type == "Polygon" else \
            sum(g.intersection(terreno.exterior.buffer(0.6)).length for g in calles.geoms)
        out.append(_chequeo("calle_llega_al_acceso", salida >= 3.0,
                            "la calle llega al lindero (salida a la calle pública)" if salida >= 3.0
                            else "la calle no llega al lindero: no tiene salida a la calle pública"))
        out.append(_chequeo("calle_dentro_del_terreno", c_fuera <= 2.0,
                            f"la calle se sale del terreno {c_fuera:,.0f} m² (redondel o tramo cortado)" if c_fuera > 2.0
                            else "la calle y el redondel caben completos en el terreno"))

    # 2. quebrada: la franja cubre toda la orilla y nada se mete en ella
    lados = [str(q.get("lado", "")).lower() for q in (ficha or {}).get("quebradas") or []
             if str(q.get("lado", "")).lower() in LADOS and q.get("tipo") in ("quebrada", "rio", "talud", None)]
    for lado in lados:
        orilla = orilla_hacia(terreno, lado)
        if orilla is None or orilla.is_empty:
            continue
        cubierta = orilla.intersection(prot.buffer(0.5)).length / orilla.length if prot is not None else 0.0
        out.append(_chequeo("franja_cubre_la_orilla", cubierta >= 0.95,
                            f"la franja de protección cubre el {cubierta * 100:.0f} % de la orilla de la quebrada ({lado})"))
        tocan = [l["id"] for l in lotes if l["poly"].intersection(orilla.buffer(0.5)).length > 0.5
                 or l["poly"].intersection(orilla.buffer(0.5)).area > 1.0]
        out.append(_chequeo("lotes_fuera_de_la_quebrada", not tocan,
                            f"lotes que llegan a la orilla de la quebrada: {_ids(tocan)}" if tocan
                            else "ningún lote llega a la orilla de la quebrada"))
    if prot is not None:
        dentro = [l["id"] for l in lotes if l["poly"].intersection(prot).area > 1.0]
        if calles is not None and calles.intersection(prot).area > 2.0:
            dentro.append("calle")
        out.append(_chequeo("proteccion_respetada", not dentro,
                            f"se meten en la zona protegida: {_ids(dentro)}" if dentro else "nada se mete en la zona protegida"))

    # 3. forma de cada lote
    huecos = [l["id"] for l in lotes if l["poly"].geom_type != "Polygon" or len(l["poly"].interiors) > 0]
    out.append(_chequeo("lotes_sin_huecos", not huecos,
                        f"lotes partidos o con huecos: {_ids(huecos)}" if huecos else "lotes enteros, sin huecos"))
    if calles is not None:
        borde = calles.buffer(0.5)
        sin = [l["id"] for l in lotes if l["poly"].boundary.intersection(borde).length < 3.0]
        out.append(_chequeo("lotes_con_salida_a_la_calle", not sin,
                            f"lotes sin frente a la calle: {_ids(sin)}" if sin else "todos los lotes dan a la calle"))
    amin = P.get("area_minima")
    if amin:
        chicos = [l["id"] for l in lotes if l["poly"].area < 0.95 * amin]
        out.append(_chequeo("area_minima", not chicos,
                            f"lotes de menos de {amin:,.0f} m²: {_ids(chicos)}" if chicos else f"todos de {amin:,.0f} m² o más"))
    angostos = []
    for l in lotes:
        try:
            import shapely

            circ = shapely.maximum_inscribed_circle(l["poly"], 0.25)
            if circ.length * 2 < ANCHO_MINIMO:
                angostos.append(l["id"])
        except Exception:  # noqa: BLE001
            pass
    out.append(_chequeo("ancho_de_lote", not angostos,
                        f"lotes de menos de {ANCHO_MINIMO:g} m de ancho: {_ids(angostos)}" if angostos
                        else f"todos los lotes miden {ANCHO_MINIMO:g} m de ancho o más"))

    # 4. casas al frente y muros
    if calles is not None:
        from .modelo3d import huella_casa

        lejos = []
        for l in lotes:
            try:
                esq = huella_casa(l, calles, lot.angulo)
            except Exception:  # noqa: BLE001 - un lote roto ya lo marcan los otros chequeos
                esq = None
            if esq is None:
                continue
            casa = Polygon(esq)
            if casa.distance(calles) > 6.5 or casa.difference(l["poly"].buffer(0.5)).area > 1.0:
                lejos.append(l["id"])
        out.append(_chequeo("casas_al_frente", not lejos,
                            f"casas lejos de la calle o fuera del lote: {_ids(lejos)}" if lejos
                            else "cada casa pegada a la calle, dentro de su lote", grave=False))
    if terr:
        altos = [p["id"] for p in terr if p["muro_max"] > MURO_MAXIMO]
        medios = [p["id"] for p in terr if 3.0 < p["muro_max"] <= MURO_MAXIMO]
        if altos:
            out.append(_chequeo("muros", False, f"muros de más de {MURO_MAXIMO:g} m en los lotes {_ids(altos)}: "
                                "conviene otra terracería o mover la casa"))
        else:
            out.append(_chequeo("muros", True, "muros de hasta {:.1f} m".format(max(p["muro_max"] for p in terr))
                                + (f"; los lotes {_ids(medios)} llevan muros de más de 3 m (diseño estructural)" if medios else "")))
    return out


def fallas(chequeos: list[dict], graves: bool = True) -> list[dict]:
    return [c for c in chequeos if not c["ok"] and (c["grave"] or not graves)]


# ---------------------------------------------------------------------------- revisión de láminas con IA
INSTRUCCION_REVISOR = """Eres un revisor experto de anteproyectos de lotificación en El Salvador (arquitecto urbanista y
topógrafo). Antes de que el cliente reciba la entrega, revisa estas láminas e imágenes como lo haría un jefe de taller
exigente. Los números y el trazo los hizo un motor geométrico que ya está verificado: NO recalcules áreas ni medidas.
Tu trabajo es encontrar lo que se ve mal o lo que contradice el proyecto.

Datos del proyecto (son la verdad):
{contexto}

Lista de chequeo:
1. Planos (técnico, altimétrico, corte): textos encimados, cortados o ilegibles; letreros puestos en la pieza
   equivocada (por ejemplo "quebrada" en una zona que no está del lado de la quebrada); lotes sin número o números
   repetidos; la calle o el redondel cortados; la franja de protección de la quebrada que no sigue toda la orilla del
   lado de la quebrada; cotas o cuadros que contradicen el dibujo.
2. Corte transversal: casas enterradas o flotando respecto a su terraza; muros sin su altura; el perfil del terreno
   terminado sin sentido.
3. Vistas pintadas (ambiente, portón, quebrada, lotes listos): número de casas distinto al de lotes con casa, más de
   una calle o calle con curvas cuando el trazo es recto, casas lejos de la calle, la quebrada del lado equivocado,
   portón fuera de la entrada, casas en la vista de lotes listos, letras, números o marcas de agua inventadas.
Señala solo problemas reales y concretos que se ven en la imagen. Si una lámina está bien, no la menciones.
Gravedad: "alta" = el cliente lo notaría o lo induce a error; "media" = se ve descuidado; "baja" = detalle menor.
Responde en español, en frases cortas. En "id" pon el identificador que va entre corchetes en el título de la lámina."""

def esquema_revisor(ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"problemas": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string", "enum": list(ids)}, "problema": {"type": "string"},
                           "gravedad": {"type": "string", "enum": ["alta", "media", "baja"]}},
            "required": ["id", "problema", "gravedad"]}}},
        "required": ["problemas"],
    }


def _jpg(ruta: str, lado: int = 1400) -> bytes:
    from PIL import Image

    im = Image.open(ruta).convert("RGB")
    im.thumbnail((lado, lado))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return buf.getvalue()


def contexto_proyecto(lot, ficha: dict | None, chequeos: list[dict] | None = None) -> str:
    q = [f"{x.get('tipo', 'quebrada')} al {x.get('lado')}" for x in (ficha or {}).get("quebradas") or [] if x.get("lado")]
    lineas = [f"- {len(lot.lotes)} lotes, numerados del {min(l['id'] for l in lot.lotes)} al {max(l['id'] for l in lot.lotes)}",
              f"- {len(lot.calles)} calle de acceso" + (" recta con redondel al final" if (lot.parametros or {}).get("modo") == "calle_central" else ""),
              "- quebrada: " + (", ".join(q) if q else "no colinda"),
              "- la vista 'lotes listos' NO lleva casas; las demás llevan una casa por lote"]
    if chequeos:
        lineas.append("- revisión geométrica del motor: " + "; ".join(c["detalle"] for c in chequeos if not c["ok"])
                      if fallas(chequeos, graves=False) else "- revisión geométrica del motor: sin fallas")
    return "\n".join(lineas)


def revisar_laminas(revisor, imagenes: list[tuple[str, str, str]], contexto: str) -> list[dict] | None:
    """`revisor(partes, instruccion, esquema) -> dict`; `imagenes` = [(id, titulo, ruta)].
    Devuelve [{id, problema, gravedad}] de lo más grave a lo menos (como mucho 12), o None si no se pudo revisar
    (sin revisor o falló el modelo): "no se revisó" nunca se reporta como "sin problemas"."""
    if revisor is None or not imagenes:
        return None
    partes, ids = [], []
    for id_, titulo, ruta in imagenes:
        if not str(ruta).lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            partes.append((f"[{id_}] {titulo}", _jpg(ruta)))
            ids.append(id_)
        except Exception:  # noqa: BLE001
            log.exception("revisor: no se pudo leer %s", ruta)
    if not partes:
        return None
    try:
        r = revisor(partes, INSTRUCCION_REVISOR.format(contexto=contexto), esquema_revisor(ids))
    except Exception:  # noqa: BLE001
        log.exception("revisor de láminas")
        return None
    orden = {"alta": 0, "media": 1, "baja": 2}
    probs = [p for p in (r or {}).get("problemas") or [] if p.get("problema") and p.get("id") in ids]
    return sorted(probs, key=lambda p: orden.get(p.get("gravedad"), 3))[:12]


def correccion(problemas: list[dict]) -> str:
    """Lo que vio el revisor, como corrección para volver a pintar una vista."""
    return ("\n\nCORRECTIONS: a reviewer found these problems in a previous attempt of this image. Fix them and keep "
            "everything else (camera, layout, number of houses, road, ravine side): "
            + "; ".join(p["problema"] for p in problemas) + ".")


def avisos_de_revision(chequeos: list[dict]) -> list[str]:
    """Lo grave de la geometría, en una línea para la carpeta y para el usuario."""
    return [f"Revisión automática: {c['detalle']}." for c in fallas(chequeos)]
