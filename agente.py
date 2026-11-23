"""El agente: conversa, decide qué herramienta usar y arma las respuestas.

Regla de oro: la IA lee y conversa; la matemática verifica. Ningún número que llegue al usuario
sale del modelo de lenguaje: todos salen de las herramientas (motor/).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import unicodedata
import uuid
from dataclasses import dataclass, field

import config
from motor import colindancia, exportar, lamina, modelo3d, normas, proyeccion, relieve
from motor.lotificacion import (LADOS, ErrorLotificacion, Parametros, franja_de_proteccion, lotificar,
                                punto_de_acceso)
from motor.rumbos import M2_POR_V2, ErrorRumbo, leer_rumbo, poligonal
from motor.terreno import Terreno, desde_coordenadas, desde_texto_rumbos, desde_tramos, verificar_con_coordenadas
from servicios.llm import LLM

log = logging.getLogger("lotes-bot.agente")
MAX_PASOS = 8
MAX_HISTORIAL = 24

# ---------------------------------------------------------------------------- sesión
@dataclass
class Sesion:
    id: str
    historial: list[dict] = field(default_factory=list)       # solo texto usuario/modelo
    terreno: dict | None = None
    conocidas: dict = field(default_factory=dict)              # coordenadas escritas en el plano {punto: [x, y]} (SIRGAS)
    malla: dict | None = None
    lotes_param: dict | None = None
    lotes_stats: dict | None = None
    proyecto: dict = field(default_factory=dict)               # nombre, ubicacion, propietario
    pin: list | None = None                                    # [lat, lng] compartido por el usuario
    adjuntos: dict = field(default_factory=dict)               # id -> {"mime","nombre","datos"} (solo en memoria)
    candidatos: list = field(default_factory=list)             # varios terrenos leídos de un plano (dicts de Terreno)
    ficha: dict = field(default_factory=dict)                  # ficha del proyecto: zona, tipo, lote, acceso, quebradas...
    cumplimiento: list = field(default_factory=list)           # último cuadro de cumplimiento
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def get_terreno(self) -> Terreno | None:
        return Terreno.de_dict(self.terreno) if self.terreno else None

    def set_terreno(self, t: Terreno | None):
        self.terreno = t.a_dict() if t else None
        self.malla = None
        self.lotes_param = None
        self.lotes_stats = None
        self.cumplimiento = []

    def get_malla(self):
        return relieve.Malla.de_dict(self.malla) if self.malla else None


class MemoriaSesiones:
    """Sesiones en memoria. Para el piloto (una instancia). Se pierden si el servidor se reinicia."""

    def __init__(self):
        self._s: dict[str, Sesion] = {}
        self._l = threading.Lock()

    def obtener(self, sid: str) -> Sesion:
        with self._l:
            if sid not in self._s:
                self._s[sid] = Sesion(sid)
            return self._s[sid]

    def borrar(self, sid: str):
        with self._l:
            self._s.pop(sid, None)


# ---------------------------------------------------------------------------- herramientas
def _obj(props: dict, requeridos: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": requeridos or []}


HERRAMIENTAS = [
    {"nombre": "leer_plano", "descripcion": "Lee fotos o PDF de planos topográficos (cuadros de rumbos y distancias y/o "
     "coordenadas) y verifica matemáticamente que cada poligonal cierre. Si mandan varias fotos (por ejemplo una por "
     "lote), pásalas todas en 'adjuntos'. Si ya hay un terreno cargado y mandan OTRO terreno para trabajarlos juntos, usa "
     "agregar=true.",
     "parametros": _obj({"adjuntos": {"type": "array", "items": {"type": "string"},
                                      "description": "ids de los adjuntos, por ejemplo ['a1', 'a2']"},
                         "agregar": {"type": "boolean", "description": "true para sumar estos terrenos a los que ya están"}},
                        ["adjuntos"])},
    {"nombre": "cargar_rumbos", "descripcion": "Carga el terreno a partir de rumbos y distancias escritos como texto, uno por línea.",
     "parametros": _obj({
         "texto": {"type": "string", "description": "Rumbos y distancias, uno por línea: S 82°40'17\" E 82.87"},
         "x_inicio": {"type": "number", "description": "Coordenada X (Este) del primer vértice, si se conoce"},
         "y_inicio": {"type": "number", "description": "Coordenada Y (Norte) del primer vértice, si se conoce"},
         "sistema": {"type": "string", "enum": ["SIRGAS", "NAD27"], "description": "Sistema de las coordenadas; por defecto SIRGAS"},
         "area_declarada_m2": {"type": "number", "description": "Área que dice el plano o la escritura, en m²"},
     }, ["texto"])},
    {"nombre": "cargar_coordenadas", "descripcion": "Carga el terreno a partir de una lista de vértices con coordenadas Lambert.",
     "parametros": _obj({
         "puntos": {"type": "array", "items": _obj({"punto": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"}},
                                                   ["x", "y"])},
         "sistema": {"type": "string", "enum": ["SIRGAS", "NAD27"]},
         "area_declarada_m2": {"type": "number"},
     }, ["puntos"])},
    {"nombre": "corregir_tramo", "descripcion": "Corrige el rumbo y/o la distancia de un tramo (numerado desde 1) y vuelve a "
     "verificar el cierre. Úsala cuando el usuario confirme el dato correcto del plano.",
     "parametros": _obj({"numero": {"type": "integer"}, "rumbo": {"type": "string", "description": "por ejemplo N 21°59'09\" O"},
                         "distancia": {"type": "number"}}, ["numero"])},
    {"nombre": "ubicar_terreno", "descripcion": "Ubica el terreno en el mapa cuando el plano no trae coordenadas: con un punto "
     "lat/lng (pin de WhatsApp) o con coordenadas Lambert X/Y.",
     "parametros": _obj({
         "lat": {"type": "number"}, "lng": {"type": "number"},
         "x": {"type": "number"}, "y": {"type": "number"},
         "referencia": {"type": "string", "enum": ["vertice1", "centro"],
                        "description": "si el punto es el primer vértice o el centro aproximado del terreno"},
     })},
    {"nombre": "elegir_terreno", "descripcion": "Cuando el plano trae varios terrenos: elige con cuál trabajar ('1', '2', ...) "
     "o 'unir' para juntarlos en un solo proyecto (deben ser colindantes; se alinean por su lindero común).",
     "parametros": _obj({"opcion": {"type": "string", "description": "'1', '2', ... o 'unir'"}}, ["opcion"])},
    {"nombre": "guardar_ficha", "descripcion": "Guarda los datos del proyecto que el usuario va contestando. Llama con lo que "
     "sepas; los demás campos se conservan. Devuelve lo que todavía falta preguntar.",
     "parametros": _obj({
         "zona": {"type": "string", "enum": ["amss", "urbano", "rural", "reserva_ecologica"],
                  "description": "amss = San Salvador y municipios vecinos (OPAMSS); urbano = ciudad o centro poblado de otro "
                                 "municipio; rural = fuera de centros poblados"},
         "tipo": {"type": "string", "enum": ["interes_social", "residencial", "campestre", "otro"]},
         "lote_objetivo_m2": {"type": "number", "description": "tamaño de lote que quieren vender"},
         "frente_m": {"type": "number", "description": "frente de lote deseado, si lo dicen"},
         "acceso_lado": {"type": "string", "enum": ["norte", "noreste", "oriente", "sureste", "sur", "suroeste", "poniente", "noroeste"], "description": "lado por donde llega la calle pública"},
         "cerrado": {"type": "boolean", "description": "proyecto cerrado con portón y vigilancia"},
         "quebradas": {"type": "array", "description": "quebradas, ríos o taludes que colindan", "items": _obj({
             "tipo": {"type": "string", "enum": ["quebrada", "rio", "talud"]},
             "lado": {"type": "string", "enum": ["norte", "noreste", "oriente", "sureste", "sur", "suroeste", "poniente", "noroeste"]},
             "profundidad_m": {"type": "number", "description": "profundidad aproximada (para quebradas y taludes)"}},
             ["tipo", "lado"])},
         "sin_quebradas": {"type": "boolean", "description": "true si el usuario confirma que no colinda con quebradas ni ríos"},
         "agua": {"type": "string", "enum": ["anda", "pozo", "no_hay", "no_se"]},
         "luz": {"type": "string", "enum": ["si", "no", "no_se"]},
         "notas": {"type": "string"},
     })},
    {"nombre": "obtener_relieve", "descripcion": "Descarga alturas satelitales del terreno ubicado y calcula desnivel, "
     "pendientes y curvas de nivel (aproximadas).", "parametros": _obj({})},
    {"nombre": "predisenar", "descripcion": "Prediseña la lotificación automáticamente con la ficha del proyecto (zona, lote, "
     "acceso, quebradas) y revisa las normas: devuelve el cuadro de cumplimiento. Los parámetros son opcionales y mandan "
     "sobre la ficha.",
     "parametros": _obj({
         "calle": {"type": "number", "description": "ancho de calle en m (defecto 8)"},
         "frente": {"type": "number", "description": "frente de lote en m (defecto 10)"},
         "fondo": {"type": "number", "description": "fondo de lote en m (defecto 25)"},
         "area_minima": {"type": "number", "description": "área mínima de lote en m² (defecto 200)"},
         "regla_verde": {"type": "string", "enum": ["pct_util", "pct_total", "por_lote"],
                         "description": "pct_util = % del área útil (defecto, 10 %); por_lote = m² por lote"},
         "valor_verde": {"type": "number"},
         "angulo": {"type": "number", "description": "orientación de calles en grados; omitir para automático"},
     })},
    {"nombre": "generar_entregables", "descripcion": "Genera la lámina en PDF (planialtimétrico y, si hay prediseño, la "
     "lotificación), el KML para Google Earth, el DXF 2D para AutoCAD/Civil 3D y, si ya hay relieve, una vista 3D en "
     "imagen y un modelo 3D (DXF) con terreno, lotes, calles y casas tipo para renderizar en AutoCAD. Se los envía al usuario.",
     "parametros": _obj({"nombre_proyecto": {"type": "string"}, "ubicacion": {"type": "string",
                         "description": "municipio y departamento"}, "propietario": {"type": "string"}})},
]

_ESQ_TERRENO = {
    "type": "object",
    "properties": {
        "nombre": {"type": "string", "description": "cómo lo llama el plano: Lote A, Porción 1, Área 2, Inmueble general..."},
        "tramos": {"type": "array", "description": "cuadro de rumbos y distancias de ESTE terreno, en orden, tal como está escrito",
                   "items": {"type": "object", "properties": {
                       "de": {"type": "string"}, "a": {"type": "string"},
                       "rumbo": {"type": "string", "description": "copiado exacto, ej. S 82°40'17.03\" E (O = oeste)"},
                       "distancia": {"type": "number"}},
                       "required": ["rumbo", "distancia"]}},
        "coordenadas": {"type": "array", "description": "vértices de ESTE terreno con coordenadas X (Este) y Y (Norte), si las hay",
                        "items": {"type": "object", "properties": {"punto": {"type": "string"}, "x": {"type": "number"},
                                                                   "y": {"type": "number"}}, "required": ["x", "y"]}},
        "area_declarada": {"type": "number", "description": "área de este terreno escrita en el plano"},
        "unidad_area": {"type": "string", "enum": ["m2", "v2", "ha"]},
        "colindancias": {"type": "array", "description": "con qué colinda cada lado, SIN nombres de personas", "items": {
            "type": "object", "properties": {
                "lado": {"type": "string", "enum": ["norte", "noreste", "oriente", "sureste", "sur", "suroeste", "poniente", "noroeste"]},
                "tipo": {"type": "string", "enum": ["calle", "quebrada", "rio", "propiedad", "otro"]},
                "descripcion": {"type": "string", "description": "ej. 'calle vecinal', 'quebrada El Zope'; nunca nombres de personas"}},
            "required": ["lado", "tipo"]}},
    },
    "required": ["tramos", "coordenadas"],
}

ESQUEMA_PLANO = {
    "type": "object",
    "properties": {
        "es_plano": {"type": "boolean", "description": "true si la imagen contiene un cuadro de rumbos/distancias o de coordenadas"},
        "terrenos": {"type": "array", "description": "un elemento por cada terreno, lote o porción que tenga su propio cuadro",
                     "items": _ESQ_TERRENO},
        "sistema": {"type": "string", "enum": ["SIRGAS", "NAD27", "desconocido"],
                    "description": "SIRGAS-ES2007 o NAD27/Ocotepeque si el plano lo dice"},
        "dudas": {"type": "string", "description": "números que no se leen bien, en una frase"},
    },
    "required": ["es_plano", "terrenos"],
}

INSTRUCCION_PLANO = (
    "Eres un dibujante topográfico de El Salvador. Transcribe EXACTAMENTE el cuadro de rumbos y distancias de este plano "
    "(estación, punto visado, rumbo con grados, minutos y segundos con sus decimales, y distancia en metros), en el orden en "
    "que aparece, y la tabla de coordenadas si existe. En los rumbos escribe E para este y O para oeste. No calcules, no "
    "redondees, no inventes: si un dígito no se lee, pon tu mejor lectura y menciónalo en 'dudas'. Si el plano trae cuadros "
    "de VARIOS terrenos (por ejemplo Lote A y Lote B, o Área 1 y Área 2), pon cada uno por separado en 'terrenos': nunca "
    "mezcles sus tramos. Anota con qué colinda cada lado (calle, quebrada, río u otra propiedad) si el plano lo dice. No "
    "transcribas nombres de personas ni números de documentos de identidad."
)

SISTEMA = """Eres Lotes-Bot, asistente de anteproyectos de lotificación en El Salvador, por WhatsApp.
Hablas como un proyectista con experiencia, en español sencillo y cercano, con mensajes cortos (WhatsApp: *negritas* con
asteriscos, sin tablas ni #). Preguntas antes de suponer.

FLUJO:
1. PLANO. Foto o PDF → leer_plano. Si lo escriben a mano → cargar_rumbos o cargar_coordenadas.
   Si el plano trae VARIOS terrenos: di cuántos son, el área de cada uno, si cierran y si son colindantes; pregunta si
   trabajas con uno o si los unes en un solo proyecto → elegir_terreno.
2. VERIFICAR. Si la poligonal no cierra o el área no coincide, NO avances: explica en simple, muestra el tramo sospechoso
   y pide confirmarlo viendo el plano → corregir_tramo.
3. UBICAR si hace falta (pin de WhatsApp o coordenada del vértice 1) → ubicar_terreno. Luego obtener_relieve.
4. FICHA DEL PROYECTO → guardar_ficha. Antes de diseñar pregunta lo que falte, máximo 3 preguntas por mensaje, en este
   orden de importancia:
   a) Zona: ¿está en el AMSS (San Salvador y municipios vecinos), en la zona urbana de otro municipio, o fuera de
      centros poblados? Cambia el área verde que pide la ley.
   b) Tipo de proyecto y tamaño de lote que quieren vender (interés social ~150 m², residencial ~250-400 m², campestre
      600 m² o más).
   c) Acceso: ¿por qué lado llega la calle pública?
   d) Quebradas, ríos o taludes: ¿colinda con alguno, de qué lado y qué tan profundo más o menos? La ley pide una franja
      de protección de 1.5 veces la profundidad de la quebrada.
   e) ¿Cerrado con portón o abierto? ¿Hay agua (ANDA o pozo) y luz?
   Si el plano ya dice las colindancias (calle, quebrada), úsalas y solo confírmalas. Si el usuario dice "usa lo
   estándar", sigue con lo que tengas y di qué supusiste.
5. PREDISEÑO → predisenar. Resume: número de lotes, tamaño promedio, área verde, y el cuadro de cumplimiento en corto
   (qué cumple, qué no, qué falta revisar, con su artículo). Si algo no cumple, propone cómo arreglarlo y pregunta antes
   de rehacer.
6. ENTREGABLES → generar_entregables: vista 3D, lámina PDF con el cuadro de cumplimiento, KML, DXF 2D y DXF 3D. Pide
   nombre del proyecto y municipio si no los tienes.

REGLAS:
- Nunca inventes ni calcules números: todos salen de las herramientas. Si no hay dato, dilo.
- La revisión de normas es referencial (Ley de Urbanismo y Construcción, su Reglamento de parcelaciones habitacionales
  y, en el AMSS, el Reglamento de OPAMSS): dilo al presentarla y que la valida un profesional.
- Siempre que entregues resultados, recuerda en una línea que es un *anteproyecto*: no sirve para trámite; la DOT u
  OPAMSS piden levantamiento topográfico y firma de un profesional autorizado.
- No pidas ni repitas DUI, NIT ni nombres de personas que aparezcan en los planos.
- Las curvas de nivel y pendientes son satelitales (error de metros): son orientativas.
- Nunca más de 3 preguntas en un mensaje.
"""


def _r(v, d=2):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(v, d)


class Agente:
    def __init__(self, llm: LLM, archivos_dir: str | None = None):
        self.llm = llm
        self.dir = archivos_dir or config.ARCHIVOS_DIR
        os.makedirs(self.dir, exist_ok=True)

    # ------------------------------------------------------------------ estado para el modelo
    def _estado(self, s: Sesion) -> str:
        partes = []
        t = s.get_terreno()
        if t:
            r = t.resumen()
            partes.append(f"Terreno cargado: {r['vertices']} vértices, {r['area_m2']} m², cierre {r['error_cierre_m']} m, "
                          f"{'cierra bien' if r['cierra_bien'] else 'NO CIERRA'}, {'ubicado' if t.ubicado else 'SIN UBICAR'}.")
        else:
            partes.append("Aún no hay terreno cargado.")
        if s.malla:
            partes.append("Relieve: listo.")
        if s.lotes_stats:
            partes.append(f"Prediseño: {s.lotes_stats['lotes']} lotes.")
        if s.candidatos and not s.terreno:
            partes.append(f"El plano trae {len(s.candidatos)} terrenos: falta que el usuario elija uno o unirlos.")
        if s.ficha:
            partes.append("Ficha del proyecto: " + json.dumps(s.ficha, ensure_ascii=False))
            falta = self._faltantes(s)
            if falta:
                partes.append("Falta preguntar: " + ", ".join(falta) + ".")
        else:
            partes.append("Ficha del proyecto: vacía (zona, tipo y lote, acceso, quebradas, servicios).")
        if s.cumplimiento:
            partes.append("Cumplimiento: " + json.dumps(normas.resumen(s.cumplimiento), ensure_ascii=False))
        if s.proyecto:
            partes.append("Datos del proyecto: " + json.dumps(s.proyecto, ensure_ascii=False))
        if s.pin:
            partes.append(f"Ubicación compartida por el usuario: {s.pin}.")
        if s.adjuntos:
            partes.append("Adjuntos disponibles: " + ", ".join(f"{k} ({v['nombre']})" for k, v in s.adjuntos.items()))
        return SISTEMA + "\nESTADO ACTUAL: " + " ".join(partes)

    # ------------------------------------------------------------------ turno
    def atender(self, s: Sesion, texto: str = "", adjuntos: list[dict] | None = None,
                ubicacion: tuple[float, float] | None = None) -> list[dict]:
        """Procesa un mensaje del usuario. Devuelve [{'tipo':'texto','texto'} | {'tipo':'archivo','ruta','nombre','titulo'}]."""
        with s.lock:
            return self._atender(s, texto, adjuntos or [], ubicacion)

    def _atender(self, s, texto, adjuntos, ubicacion):
        entrada = (texto or "").strip()
        for a in adjuntos:
            aid = f"a{len(s.adjuntos) + 1}"
            s.adjuntos[aid] = a
            entrada += f"\n[adjunto: {aid}, {a.get('nombre') or a['mime']}]"
        if ubicacion:
            s.pin = [round(ubicacion[0], 6), round(ubicacion[1], 6)]
            entrada += f"\n[ubicación compartida: {s.pin[0]}, {s.pin[1]}]"
        entrada = entrada.strip() or "(mensaje vacío)"
        mensajes = list(s.historial[-MAX_HISTORIAL:]) + [{"rol": "usuario", "texto": entrada}]
        salidas: list[dict] = []
        texto_final = ""
        for _ in range(MAX_PASOS):
            try:
                resp = self.llm.turno(self._estado(s), mensajes, HERRAMIENTAS)
            except Exception:
                log.exception("fallo del modelo")
                texto_final = "Tuve un problema para pensar la respuesta. ¿Me lo repites en un momento?"
                break
            if not resp.llamadas:
                texto_final = resp.texto or "Listo."
                break
            mensajes.append({"rol": "modelo", "llamadas": resp.llamadas, "crudo": resp.crudo})
            for c in resp.llamadas:
                resultado = self.ejecutar(s, c["nombre"], c.get("args") or {}, salidas)
                mensajes.append({"rol": "herramienta", "nombre": c["nombre"], "resultado": resultado})
        else:
            texto_final = "Me enredé con tantos pasos. ¿Qué necesitas que haga ahora?"
        s.historial += [{"rol": "usuario", "texto": entrada}, {"rol": "modelo", "texto": texto_final}]
        s.historial = s.historial[-MAX_HISTORIAL:]
        return [{"tipo": "texto", "texto": texto_final}] + salidas

    def ejecutar(self, s: Sesion, nombre: str, args: dict, salidas: list) -> dict:
        f = getattr(self, "h_" + nombre, None)
        if f is None:
            return {"ok": False, "error": f"herramienta desconocida: {nombre}"}
        try:
            return f(s, salidas=salidas, **args)
        except (ErrorRumbo, ErrorLotificacion, relieve.ErrorRelieve, ValueError) as e:
            return {"ok": False, "error": str(e)}
        except TypeError as e:
            return {"ok": False, "error": f"parámetros inválidos: {e}"}
        except Exception as e:  # noqa: BLE001
            log.exception("fallo en herramienta %s", nombre)
            return {"ok": False, "error": f"falló {nombre}: {type(e).__name__}"}

    # ------------------------------------------------------------------ herramientas: terreno
    def _informe_terreno(self, s: Sesion, t: Terreno, extra: dict | None = None) -> dict:
        s.set_terreno(t)
        r = {"ok": True, "resumen": t.resumen(), "problemas": t.problemas(),
             "tramos": [f"{i}. {x.get('rumbo')} {x['dist']:.2f} m" for i, x in enumerate(t.tramos, 1)]}
        sos = t.sospechosos()
        if sos:
            r["tramos_sospechosos"] = [f"Tramo {x['tramo']}: se leyó {x['distancia_leida']} m; con {x['distancia_que_cerraria']:.2f} m "
                                       f"cerraría (área {x['area_resultante']} m²)" for x in sos]
        elif not t.cierra_bien():
            r["nota"] = ("Ningún cambio de una sola distancia explica el error: puede ser un rumbo mal leído, un tramo "
                         "faltante o un error del plano. Pide revisar el cuadro completo.")
        if extra:
            r.update(extra)
        return r

    def h_leer_plano(self, s: Sesion, adjuntos: list | None = None, adjunto: str | None = None, agregar: bool = False,
                     salidas=None) -> dict:
        ids = [str(x) for x in (adjuntos or [])] + ([adjunto] if adjunto else [])
        if not ids:
            return {"ok": False, "error": "falta indicar qué adjunto leer"}
        if not self.llm.puede_ver:
            return {"ok": False, "error": "este modo no puede leer imágenes; pide los rumbos escritos"}
        junto = {"es_plano": False, "terrenos": [], "sistema": "desconocido", "dudas": ""}
        for aid in ids:
            a = s.adjuntos.get(aid)
            if a is None:
                return {"ok": False, "error": f"no existe el adjunto {aid}"}
            d = self.llm.leer_imagen(a["datos"], a["mime"], INSTRUCCION_PLANO, ESQUEMA_PLANO)
            terrenos = d.get("terrenos")
            if terrenos is None and (d.get("tramos") or d.get("coordenadas")):
                terrenos = [{k: d.get(k) for k in ("tramos", "coordenadas", "area_declarada", "unidad_area", "colindancias")}]
            for k, t in enumerate(terrenos or [], 1):
                t.setdefault("nombre", (a.get("nombre") or aid) + (f" ({k})" if len(terrenos) > 1 else ""))
                junto["terrenos"].append(t)
            junto["es_plano"] = junto["es_plano"] or bool(d.get("es_plano"))
            if d.get("sistema") in ("SIRGAS", "NAD27"):
                junto["sistema"] = d["sistema"]
            if d.get("dudas"):
                junto["dudas"] = (junto["dudas"] + " " + d["dudas"]).strip()
        previos = []
        if agregar:
            previos = [Terreno.de_dict(c) for c in s.candidatos] or ([s.get_terreno()] if s.terreno else [])
        return self._desde_lectura(s, junto, origen=", ".join(ids), previos=previos)

    def _terreno_de_lectura(self, d: dict, sistema: str, origen: str) -> tuple[Terreno, dict, dict]:
        """Arma un terreno con lo que leyó la IA y lo verifica. Devuelve (terreno, extra, coordenadas conocidas)."""
        area = d.get("area_declarada")
        unidad = d.get("unidad_area") or "m2"
        if area:
            area = area * M2_POR_V2 if unidad == "v2" else (area * 10000 if unidad == "ha" else area)
        tramos, errores = [], []
        for k, tr in enumerate(d.get("tramos") or [], 1):
            try:
                x = leer_rumbo(f"{tr['rumbo']} {tr['distancia']}")
                x.update(de=tr.get("de") or "", a=tr.get("a") or "")
                tramos.append(x)
            except (ErrorRumbo, KeyError, TypeError) as e:
                errores.append(f"tramo {k}: {e}")
        conocidas = {}
        for c in d.get("coordenadas") or []:
            x, y = float(c["x"]), float(c["y"])
            if not proyeccion.en_el_salvador(x, y) and proyeccion.en_el_salvador(y, x):
                x, y = y, x  # columnas invertidas
            if sistema == "NAD27":
                x, y = proyeccion.nad27_a_sirgas(x, y)
            conocidas[str(c.get("punto") or f"P{len(conocidas) + 1}")] = (x, y)
        fuera = [k for k, (x, y) in conocidas.items() if not proyeccion.en_el_salvador(x, y)]
        if len(tramos) >= 3:
            nombres = [t["de"] for t in tramos] if all(t["de"] for t in tramos) else None
            t = desde_tramos(tramos, None, nombres, "SIRGAS", area, origen)
            t = verificar_con_coordenadas(t, {k: v for k, v in conocidas.items() if k not in fuera})
        elif len(conocidas) >= 3:
            t = desde_coordenadas([(k, x, y) for k, (x, y) in conocidas.items()], "SIRGAS", area, origen)
        else:
            raise ValueError("leí menos de 3 tramos o coordenadas; manda una foto más cercana y derecha del cuadro")
        extra = {}
        if errores:
            extra["errores_lectura"] = errores
        if fuera:
            extra["coordenadas_fuera_de_el_salvador"] = fuera
        if d.get("colindancias"):
            extra["colindancias"] = [{k: c.get(k) for k in ("lado", "tipo", "descripcion") if c.get(k)} for c in d["colindancias"]]
        return t, extra, conocidas

    def _desde_lectura(self, s: Sesion, d: dict, origen: str = "", previos: list | None = None) -> dict:
        crudos = d.get("terrenos")
        if crudos is None and (d.get("tramos") or d.get("coordenadas")):  # formato de un solo terreno
            crudos = [{k: d.get(k) for k in ("tramos", "coordenadas", "area_declarada", "unidad_area", "colindancias")}]
        crudos = [c for c in (crudos or []) if (c.get("tramos") or c.get("coordenadas"))]
        if not crudos:
            return {"ok": False, "error": "no encontré un cuadro de rumbos ni de coordenadas en el archivo"}
        sistema = d.get("sistema") if d.get("sistema") in ("SIRGAS", "NAD27") else "SIRGAS"
        comunes = {"sistema_detectado": d.get("sistema") or "desconocido"}
        if d.get("dudas"):
            comunes["dudas_de_lectura"] = d["dudas"]
        if sistema == "NAD27":
            comunes["nota_sistema"] = "Coordenadas NAD27 convertidas a SIRGAS-ES2007."

        previos = previos or []
        if len(crudos) == 1 and not previos:
            try:
                t, extra, conocidas = self._terreno_de_lectura(crudos[0], sistema, origen)
            except ValueError as e:
                return {"ok": False, "error": str(e)}
            s.candidatos = []
            s.conocidas = {k: list(v) for k, v in conocidas.items()}
            self._colindancias_a_ficha(s, extra.get("colindancias"))
            return self._informe_terreno(s, t, {**comunes, **extra})

        # varios terrenos (en el mismo plano, en varias fotos, o sumados a los que ya estaban)
        terrenos, informes = [], []
        for t in previos:
            terrenos.append(t)
            informes.append({"numero": len(terrenos), "nombre": t.origen or f"Terreno {len(terrenos)}",
                             "resumen": t.resumen(), "problemas": t.problemas(), "ya_estaba": True})
        for k, c in enumerate(crudos, 1):
            nombre = c.get("nombre") or f"Terreno {k}"
            try:
                t, extra, _ = self._terreno_de_lectura(c, sistema, nombre)
            except ValueError as e:
                informes.append({"numero": k, "nombre": nombre, "error": str(e)})
                continue
            t.origen = nombre
            terrenos.append(t)
            inf = {"numero": len(terrenos), "nombre": nombre, "resumen": t.resumen(), "problemas": t.problemas()}
            sos = t.sospechosos()
            if sos:
                inf["tramos_sospechosos"] = [f"Tramo {x['tramo']}: se leyó {x['distancia_leida']} m; con "
                                             f"{x['distancia_que_cerraria']:.2f} m cerraría" for x in sos]
            if extra:
                inf.update(extra)
            informes.append(inf)
        if not terrenos:
            return {"ok": False, "error": "no pude armar ninguno de los terrenos", "detalle": informes}
        s.set_terreno(None)
        s.candidatos = [t.a_dict() for t in terrenos]
        s.conocidas = {}
        cols = []
        for i in range(len(terrenos)):
            for j in range(i + 1, len(terrenos)):
                r = colindancia.alinear(terrenos[i], terrenos[j])
                if r:
                    info = r[1]
                    txt = f"Terreno {i + 1} y {j + 1} son colindantes: comparten {info['lindero_comun_m']} m de lindero"
                    if info.get("tramo_a"):
                        txt += f" (tramo {info['tramo_a']} del {i + 1} = tramo {info['tramo_b']} del {j + 1}, al revés)"
                    cols.append(txt)
        todos_col = [x.get("colindancias") for x in informes if x.get("colindancias")]
        for c in todos_col:
            self._colindancias_a_ficha(s, c)
        return {"ok": True, "varios_terrenos": informes, "colindantes": cols or ["No encontré un lindero común entre ellos."],
                "siguiente": "Muéstrale al usuario los terrenos y pregúntale si trabaja con uno (elegir_terreno '1', '2'...) "
                             "o si los une en un solo proyecto (elegir_terreno 'unir').", **comunes}

    def _colindancias_a_ficha(self, s: Sesion, cols):
        """Lo que el plano dice de las colindancias se guarda como pista para la ficha (el usuario lo confirma)."""
        if not cols:
            return
        pistas = s.ficha.setdefault("del_plano", [])
        for c in cols:
            if c not in pistas:
                pistas.append(c)

    def h_elegir_terreno(self, s: Sesion, opcion: str, salidas=None) -> dict:
        if not s.candidatos:
            return {"ok": False, "error": "no hay varios terrenos cargados"}
        cands = [Terreno.de_dict(c) for c in s.candidatos]
        op = str(opcion).strip().lower()
        if op.isdigit():
            k = int(op) - 1
            if not 0 <= k < len(cands):
                return {"ok": False, "error": f"elige entre 1 y {len(cands)} o 'unir'"}
            return self._informe_terreno(s, cands[k], {"elegido": cands[k].origen})
        if not op.startswith("unir"):
            return {"ok": False, "error": "opción no válida: usa '1', '2'... o 'unir'"}
        colocados = [cands[0]]
        pendientes = cands[1:]
        while pendientes:
            avance = False
            for b in list(pendientes):
                for a in colocados:
                    r = colindancia.alinear(a, b)
                    if r:
                        colocados.append(r[0])
                        pendientes.remove(b)
                        avance = True
                        break
            if not avance:
                return {"ok": False, "error": "no encontré el lindero común de: " + ", ".join(b.origen for b in pendientes)}
        u = colindancia.unir(colocados)
        problemas = [f"{t.origen}: {p}" for t in cands for p in t.problemas()]
        return self._informe_terreno(s, u, {"unidos": [t.origen for t in cands],
                                            "areas_por_terreno_m2": [round(t.area, 2) for t in cands],
                                            "problemas_de_los_originales": problemas})

    # ------------------------------------------------------------------ ficha del proyecto
    def _faltantes(self, s: Sesion) -> list[str]:
        f = s.ficha
        falta = []
        if not f.get("zona"):
            falta.append("zona (AMSS, urbana o fuera de centros poblados)")
        if not f.get("lote_objetivo_m2") and not f.get("tipo"):
            falta.append("tipo de proyecto y tamaño de lote")
        if not f.get("acceso_lado"):
            falta.append("por qué lado llega la calle pública")
        quebradas_plano = [c for c in f.get("del_plano", []) if c.get("tipo") in ("quebrada", "rio")]
        if not f.get("quebradas") and not f.get("sin_quebradas"):
            falta.append("si colinda con quebradas o ríos" + (" (el plano menciona una)" if quebradas_plano else ""))
        for q in f.get("quebradas") or []:
            if q.get("tipo") in ("quebrada", "talud") and not q.get("profundidad_m"):
                falta.append(f"profundidad de la {q['tipo']} del lado {q.get('lado')}")
        if "cerrado" not in f:
            falta.append("si es cerrado con portón")
        if not f.get("agua"):
            falta.append("agua y luz")
        return falta

    def h_guardar_ficha(self, s: Sesion, salidas=None, **datos) -> dict:
        campos = {"zona", "tipo", "lote_objetivo_m2", "frente_m", "acceso_lado", "cerrado", "quebradas", "sin_quebradas",
                  "agua", "luz", "notas"}
        for k, v in datos.items():
            if k in campos and v is not None:
                if k in ("acceso_lado",) and str(v).lower() not in LADOS:
                    return {"ok": False, "error": f"lado no reconocido: {v}"}
                s.ficha[k] = v
        if s.ficha.get("quebradas"):
            s.ficha["sin_quebradas"] = False
        return {"ok": True, "ficha": {k: v for k, v in s.ficha.items() if k != "del_plano"},
                "falta_preguntar": self._faltantes(s)}

    def _parametros(self, s: Sesion) -> tuple[Parametros, list, dict]:
        """Parámetros del prediseño: la ficha pone los valores base y lo que pidió el usuario en predisenar manda."""
        f = s.ficha
        base: dict = {}
        zona = f.get("zona")
        if zona == "rural":
            base.update(regla_verde="por_lote", valor_verde=12.5)
        elif zona in ("urbano", "amss", "reserva_ecologica"):
            base.update(regla_verde="pct_util", valor_verde=10.0)
        if zona == "amss":
            base["pct_recreativa"] = 50.0
        lote = f.get("lote_objetivo_m2") or {"interes_social": 150, "residencial": 250, "campestre": 700}.get(f.get("tipo"))
        if zona == "reserva_ecologica":
            lote = max(lote or 0, 4000)
        if lote:
            frente = f.get("frente_m") or (8 if lote <= 160 else 10 if lote <= 260 else 12 if lote <= 400 else
                                           15 if lote <= 600 else 20 if lote <= 1000 else 25 if lote <= 2000 else 40)
            base.update(frente=float(frente), fondo=round(lote / frente, 1), area_minima=round(lote * 0.9))
        base.update(s.lotes_param or {})
        t = s.get_terreno()
        if f.get("acceso_lado") and "entrada" not in (s.lotes_param or {}):
            p = punto_de_acceso(t.puntos, f["acceso_lado"])
            if p:
                base["entrada"] = p
            base["lado_acceso"] = f["acceso_lado"]
        excluir, prot = [], []
        for q in f.get("quebradas") or []:
            if q.get("tipo") == "rio":
                ancho = float(q.get("ancho_m") or 50.0)
            elif q.get("profundidad_m"):
                ancho = 1.5 * float(q["profundidad_m"])
            else:
                prot.append({"tipo": q.get("tipo"), "lado": q.get("lado")})
                continue
            franja = franja_de_proteccion(t.puntos, q["lado"], ancho)
            if franja is not None:
                excluir.append(franja)
            prot.append({"tipo": q.get("tipo"), "lado": q.get("lado"), "profundidad": q.get("profundidad_m"),
                         "ancho": round(ancho, 2)})
        ficha_normas = {"zona": zona, "tipo": f.get("tipo"), "protecciones": prot, "cerrado": f.get("cerrado"),
                        "quebrada_sin_datos": (any(c.get("tipo") in ("quebrada", "rio") for c in f.get("del_plano", []))
                                               and not f.get("quebradas") and not f.get("sin_quebradas"))}
        return Parametros.de_dict(base), excluir, ficha_normas

    def _lotificar(self, s: Sesion):
        """Prediseño + revisión de normas. Si la norma pide equipamiento social, se reserva y se rehace."""
        t = s.get_terreno()
        m = s.get_malla()
        P, excluir, fn = self._parametros(s)
        r = lotificar(t.puntos, P, m, excluir)
        if fn.get("zona"):
            req, _, claro = normas.equipamiento_requerido(fn["zona"], r.stats["lotes"], fn.get("tipo") or "residencial")
            if claro and r.stats["area_equipamiento"] < req - 0.5:
                P.equipamiento_m2 = req
                r = lotificar(t.puntos, P, m, excluir)
        filas = normas.revisar(r, t, m, fn)
        s.cumplimiento = filas
        return r, filas

    def h_cargar_rumbos(self, s: Sesion, texto: str, x_inicio=None, y_inicio=None, sistema="SIRGAS",
                        area_declarada_m2=None, salidas=None) -> dict:
        inicio = (float(x_inicio), float(y_inicio)) if x_inicio is not None and y_inicio is not None else None
        t, errores = desde_texto_rumbos(texto, inicio=inicio, sistema=sistema or "SIRGAS", area_declarada=area_declarada_m2)
        if t is None:
            return {"ok": False, "error": "no pude leer los rumbos", "errores": errores}
        s.conocidas = {}
        return self._informe_terreno(s, t, {"errores_lectura": errores} if errores else None)

    def h_cargar_coordenadas(self, s: Sesion, puntos: list, sistema="SIRGAS", area_declarada_m2=None, salidas=None) -> dict:
        if len(puntos) < 3:
            return {"ok": False, "error": "se necesitan al menos 3 vértices"}
        t = desde_coordenadas([(p.get("punto") or str(i), p["x"], p["y"]) for i, p in enumerate(puntos, 1)],
                              sistema or "SIRGAS", area_declarada_m2)
        s.conocidas = {}
        return self._informe_terreno(s, t)

    def h_corregir_tramo(self, s: Sesion, numero: int, rumbo: str | None = None, distancia: float | None = None,
                         salidas=None) -> dict:
        t = s.get_terreno()
        if not t or not t.tramos:
            return {"ok": False, "error": "no hay rumbos cargados"}
        i = int(numero) - 1
        if not 0 <= i < len(t.tramos):
            return {"ok": False, "error": f"el tramo debe ir de 1 a {len(t.tramos)}"}
        viejo = t.tramos[i]
        nuevo = leer_rumbo(f"{rumbo or viejo['rumbo']} {distancia if distancia is not None else viejo['dist']}")
        nuevo.update(de=viejo.get("de", ""), a=viejo.get("a", ""))
        tramos = list(t.tramos)
        tramos[i] = nuevo
        pg = poligonal(t.puntos[0], tramos)
        t2 = Terreno(nombres=t.nombres, puntos=pg["puntos"], tramos=tramos, cierre=pg["cierre"], perimetro=pg["perimetro"],
                     area_declarada=t.area_declarada, sistema_original=t.sistema_original, ubicado=t.ubicado, origen=t.origen)
        if s.conocidas:
            t2 = verificar_con_coordenadas(t2, {k: tuple(v) for k, v in s.conocidas.items()})
        return self._informe_terreno(s, t2, {"cambio": f"tramo {numero}: {viejo['rumbo']} {viejo['dist']} → {nuevo['rumbo']} {nuevo['dist']}"})

    def h_ubicar_terreno(self, s: Sesion, lat=None, lng=None, x=None, y=None, referencia="centro", salidas=None) -> dict:
        if lat is not None and lng is not None:
            x, y = proyeccion.a_lambert(float(lat), float(lng))
        if x is None or y is None:
            return {"ok": False, "error": "falta la ubicación (lat/lng o X/Y)"}
        x, y = float(x), float(y)
        if not proyeccion.en_el_salvador(x, y):
            return {"ok": False, "error": "ese punto no cae en El Salvador"}
        t = s.get_terreno()
        if t is None:
            s.pin = list(proyeccion.a_latlng(x, y))
            return {"ok": True, "nota": "Guardé la ubicación; falta cargar el plano."}
        t.ubicar_en(x, y, referencia or "centro")
        s.set_terreno(t)
        lat_c, lng_c = proyeccion.a_latlng(*t.centroide())
        return {"ok": True, "centro_latlng": [round(lat_c, 6), round(lng_c, 6)],
                "nota": "Ubicación aproximada: con un pin de celular el error es de varios metros. Las coordenadas "
                        "oficiales salen del levantamiento."}

    # ------------------------------------------------------------------ herramientas: relieve y lotes
    def h_obtener_relieve(self, s: Sesion, salidas=None) -> dict:
        t = s.get_terreno()
        if t is None:
            return {"ok": False, "error": "primero hay que cargar el terreno"}
        if not t.ubicado:
            return {"ok": False, "error": "el terreno no está ubicado; pide la ubicación"}
        m = relieve.descargar(t.puntos)
        s.malla = m.a_dict()
        s.lotes_param = s.lotes_stats = None
        st = relieve.estadisticas(m, t.puntos)
        return {"ok": True, "estadisticas": st, "intervalo_curvas_m": relieve.intervalo_curvas(st["desnivel"])}

    def h_predisenar(self, s: Sesion, salidas=None, **params) -> dict:
        t = s.get_terreno()
        if t is None:
            return {"ok": False, "error": "primero hay que cargar el terreno"}
        validos = {k: v for k, v in params.items() if k in Parametros.__dataclass_fields__ and v is not None}
        base = dict(s.lotes_param or {})
        base.update(validos)
        s.lotes_param = base
        r, filas = self._lotificar(s)
        st = r.stats
        s.lotes_stats = {k: _r(v) for k, v in st.items()}
        inclinados = sum(1 for l in r.lotes if l["pendiente"] is not None and l["pendiente"] > 30)
        out = {"ok": True, "lotes": st["lotes"], "area_util_m2": _r(st["area_util"]),
               "area_calles_m2": _r(st["area_circulacion"]), "area_verde_m2": _r(st["area_verde"]),
               "area_verde_requerida_m2": _r(st["area_verde_requerida"]),
               "lotes_pasados_a_area_verde": st["lotes_convertidos_a_verde"],
               "lote_promedio_m2": _r(st["lote_promedio"]), "lote_min_m2": _r(st["lote_min"]), "lote_max_m2": _r(st["lote_max"]),
               "aprovechamiento_pct": _r(st["area_util"] / st["area_total"] * 100, 1),
               "parametros": r.parametros, "orientacion_calles_grados": r.angulo,
               "area_proteccion_m2": _r(st["area_proteccion"]), "area_equipamiento_m2": _r(st["area_equipamiento"]),
               "cumplimiento": [{k: f[k] for k in ("regla", "base", "requerido", "proyecto", "estado", "nota") if f.get(k)}
                                for f in filas],
               "resumen_cumplimiento": normas.resumen(filas),
               "supuestos": self._faltantes(s)}
        if s.malla:
            out["lotes_con_pendiente_mayor_30pct"] = inclinados
            out["pendiente_media_de_calles_pct"] = _r(st["pendiente_calles_pct"], 1)
        else:
            out["nota"] = "Sin relieve: el prediseño no consideró pendientes."
        return out

    def h_generar_entregables(self, s: Sesion, nombre_proyecto=None, ubicacion=None, propietario=None, salidas=None) -> dict:
        t = s.get_terreno()
        if t is None:
            return {"ok": False, "error": "primero hay que cargar el terreno"}
        for k, v in (("nombre", nombre_proyecto), ("ubicacion", ubicacion), ("propietario", propietario)):
            if v:
                s.proyecto[k] = v
        nombre = s.proyecto.get("nombre") or "Anteproyecto de lotificación"
        datos = lamina.Datos(proyecto=nombre, ubicacion=s.proyecto.get("ubicacion", ""),
                             propietario=s.proyecto.get("propietario", ""))
        m = s.get_malla()
        lot, filas = self._lotificar(s) if s.lotes_param is not None else (None, [])
        carpeta = os.path.join(self.dir, uuid.uuid4().hex)
        os.makedirs(carpeta, exist_ok=True)
        base = _slug(nombre)
        pdf = os.path.join(carpeta, f"{base}.pdf")
        info = lamina.generar(pdf, t, m, lot, datos, cumplimiento=filas)
        cs = None
        if m is not None:
            from shapely.geometry import Polygon

            cs = relieve.curvas(m, info["intervalo_curvas"], recorte=Polygon(t.puntos).buffer(20))
        archivos = [(pdf, "Lámina en PDF (imprimir en tabloide al 100 %)")]
        if t.ubicado:
            kml = os.path.join(carpeta, f"{base}.kml")
            with open(kml, "w", encoding="utf-8") as f:
                f.write(exportar.kml(nombre, t, cs, lot))
            archivos.append((kml, "Para abrir en Google Earth"))
        dxf = os.path.join(carpeta, f"{base}.dxf")
        exportar.dxf(dxf, t, cs, lot)
        archivos.append((dxf, "Plano 2D para AutoCAD / Civil 3D (coordenadas Lambert SIRGAS)"))
        if m is not None and t.ubicado:
            escena = modelo3d.construir(t, m, lot)
            jpg = os.path.join(carpeta, f"{base}-vista-3d.jpg")
            sub = (f"{len(lot.lotes)} lotes · " if lot else "") + (s.proyecto.get("ubicacion") or "")
            modelo3d.vista(jpg, escena, t, nombre, sub.strip(" ·"), malla=m)
            archivos.insert(0, (jpg, "Vista 3D ilustrativa del anteproyecto"))
            dxf3d = os.path.join(carpeta, f"{base}-3d.dxf")
            modelo3d.dxf(dxf3d, escena)
            archivos.append((dxf3d, "Modelo 3D para AutoCAD: terreno, lotes, calles y casas tipo (vista Realista → Render)"))
        for ruta, titulo in archivos:
            salidas.append({"tipo": "archivo", "ruta": ruta, "nombre": os.path.basename(ruta), "titulo": titulo,
                            "imagen": ruta.endswith(".jpg")})
        return {"ok": True, "archivos_enviados": [os.path.basename(a) for a, _ in archivos], "escala": f"1:{info['escala']:,}",
                "hojas": info["hojas"], "nota": "Los archivos se envían automáticamente después de tu mensaje."}


def _slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9]+", "-", t).strip("-").lower()
    return (t or "anteproyecto")[:50]
