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

from shapely.geometry import Polygon

import config
from motor import (altimetria, colindancia, exportar, lamina, modelo3d, normas, presentacion, proyeccion, relieve,
                   territorio)
from motor.calle_central import es_angosto, lindero_comun, lotificar_calle_central, lotificar_un_lado
from motor.lotificacion import (LADOS, ErrorLotificacion, Parametros, franja_de_proteccion, lotificar,
                                punto_de_acceso)
from motor.rumbos import M2_POR_V2, ErrorRumbo, leer_rumbo, poligonal
from motor.terreno import Terreno, desde_coordenadas, desde_texto_rumbos, desde_tramos, verificar_con_coordenadas
from servicios import opamss as geovisor_opamss
from servicios import osm as mapa_osm
from servicios.llm import LLM

log = logging.getLogger("lotes-bot.agente")
MAX_PASOS = 8
MAX_HISTORIAL = 24

# ---------------------------------------------------------------------------- sesión
def resumen_opamss(op: dict) -> dict:
    """Lo esencial del geovisor de la OPAMSS para que el bot se lo diga al usuario (con su fuente)."""
    out = {"fuente": op.get("fuente")}
    if op.get("sin_cobertura"):
        out["sin_datos"] = ("el Esquema Director de la OPAMSS no tiene datos de este terreno (puede ser un distrito que "
                            "entró al AMSS después); hay que pedir calificación de lugar")
        return out
    zonas = op.get("uso_suelo") or []
    if zonas:
        out["clasificacion_del_suelo"] = [f"{z['pct']:.0f} % {z.get('clasificacion') or z.get('codigo')}" +
                                          (f" ({z['tratamiento']})" if z.get("tratamiento") else "") for z in zonas]
        z0 = next((z for z in zonas if z.get("codigo") != "NUR"), zonas[0])
        if any(z.get("codigo") == "NUR" and z["pct"] >= 5 for z in zonas):
            out["alerta"] = "parte del terreno es suelo NO urbanizable: ahí no se puede lotificar"
        limites = []
        if z0.get("altura_base"):
            limites.append(f"altura hasta {z0['altura_base']} niveles")
        if z0.get("impermeabilizacion_base"):
            limites.append(f"impermeabilizar hasta el {z0['impermeabilizacion_base']} % de cada lote")
        if z0.get("edificabilidad_base"):
            limites.append(f"edificabilidad hasta {z0['edificabilidad_base']} × el área del lote")
        if limites:
            out["limites"] = limites
    v = (op.get("vias") or {})
    if v.get("proyectada") and v["proyectada"].get("cruza_m", 0) > 1:
        out["via_proyectada"] = f"una vía proyectada del Esquema Director cruza el terreno ({v['proyectada']['cruza_m']:.0f} m)"
    cercanas = [f"vía {k} a {d['distancia_m']:.0f} m" for k, d in v.items() if d and k != "proyectada"
                and d.get("distancia_m") is not None]
    if cercanas:
        out["vias_cercanas"] = cercanas
    peligros = [n for k, n in (("zona_tectonica", "zona de mayor densidad tectónica"),
                               ("peligro_volcanico", "zona de mayor peligro volcánico")) if op.get(k)]
    peligros += [f"{p['pct']:.0f} % peligrosidad {str(p['valor']).lower()} por dinámica superficial"
                 for p in op.get("peligrosidad_superficial") or [] if str(p.get("valor", "")).lower() in ("alta", "muy alta")]
    if peligros:
        out["peligros"] = peligros
    if op.get("no_se_pudo_consultar"):
        out["sin_respuesta"] = op["no_se_pudo_consultar"]
    return out


AVISO_PIN = ("Para ubicar el terreno necesito coordenadas, no la ubicación del teléfono (no es exacta). Mándame la X y "
             "la Y (Lambert) de cualquier mojón, como vienen en el plano o te las da el topógrafo, por ejemplo: "
             "Mo. 1  X 473280.68  Y 289049.93")


def formato_whatsapp(t: str) -> str:
    """El modelo a veces escribe Markdown: WhatsApp usa *negritas* con un asterisco, viñetas como texto y no tiene
    separadores ni títulos con #."""
    if not t:
        return t
    t = re.sub(r"\*\*(.+?)\*\*", r"*\1*", t, flags=re.S)
    t = re.sub(r"__(.+?)__", r"_\1_", t, flags=re.S)
    t = re.sub(r"(?m)^(\s*)[*-] (?=\S)", r"\1• ", t)
    t = re.sub(r"(?m)^\s*(-{3,}|\*{3,}|_{3,})\s*$\n?", "", t)
    t = re.sub(r"(?m)^#{1,6}\s*(.+)$", r"*\1*", t)
    t = re.sub(r"\*{2,}", "*", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


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
    adjuntos: dict = field(default_factory=dict)               # id -> {"mime","nombre","datos"} (solo en memoria)
    candidatos: list = field(default_factory=list)             # varios terrenos leídos de un plano (dicts de Terreno)
    ficha: dict = field(default_factory=dict)                  # ficha del proyecto: zona, tipo, lote, acceso, quebradas...
    cumplimiento: list = field(default_factory=list)           # último cuadro de cumplimiento
    partes: list = field(default_factory=list)                 # terrenos unidos: [{"origen", "anillo"}] relativos al centroide
    lindero: list | None = None                                # lindero común de los terrenos unidos (relativo al centroide)
    opamss: dict | None = None                                 # lo que dice el geovisor de la OPAMSS del terreno
    osm: dict | None = None                                    # calles, ríos y construcciones alrededor (OpenStreetMap)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def get_terreno(self) -> Terreno | None:
        return Terreno.de_dict(self.terreno) if self.terreno else None

    def set_terreno(self, t: Terreno | None):
        self.terreno = t.a_dict() if t else None
        self.malla = None
        self.lotes_param = None
        self.lotes_stats = None
        self.cumplimiento = []
        self.opamss = None
        self.osm = None

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
    {"nombre": "ubicar_terreno", "descripcion": "Ubica el terreno en el mapa cuando el plano no trae coordenadas (solo rumbos "
     "y distancias): con las coordenadas de un mojón que el usuario escribe (X/Y Lambert del plano o del topógrafo; o "
     "latitud y longitud si así las tiene). Nunca con la ubicación del teléfono (pin). Basta un mojón; si da más, van en "
     "'puntos' y se comprueban.",
     "parametros": _obj({
         "x": {"type": "number", "description": "X (Este) Lambert del mojón"},
         "y": {"type": "number", "description": "Y (Norte) Lambert del mojón"},
         "lat": {"type": "number", "description": "solo si el usuario escribe latitud y longitud de un mojón"},
         "lng": {"type": "number", "description": "solo si el usuario escribe latitud y longitud de un mojón"},
         "referencia": {"type": "string", "description": "el mojón de esas coordenadas, como lo dice el usuario ('Mo. 1', "
                        "'5')"},
         "sistema": {"type": "string", "enum": ["SIRGAS", "NAD27"], "description": "de las X/Y; por defecto SIRGAS"},
         "puntos": {"type": "array", "description": "otros mojones con su X/Y para comprobar la ubicación",
                    "items": {"type": "object", "properties": {"mojon": {"type": "string"}, "x": {"type": "number"},
                                                                "y": {"type": "number"}}}},
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
         "quebrada_confirmada": {"type": "boolean", "description": "true si, después del aviso del relieve "
                                 "(revisar_quebrada), el usuario confirma que sí colinda con esa quebrada"},
         "figura": {"type": "string", "enum": ["lotificacion", "condominio", "proindiviso", "no_se"],
                    "description": "cómo se va a vender: lotes con permiso de lotificación, condominio, o en proindiviso"},
         "distribucion": {"type": "string", "enum": ["automatica", "calle_central", "un_lado"],
                          "description": "calle_central = una sola calle por el medio (si se unieron dos terrenos, sobre el "
                                         "lindero común) con lotes a los dos lados, retorno y área verde al fondo; para quintas "
                                         "o pocos lotes, o cuando piden 'una calle en medio'. un_lado = calle pegada a un "
                                         "lindero largo desde la entrada y una sola fila de lotes del otro lado; para "
                                         "terrenos angostos (franjas) o cuando piden 'de un solo lado' o 'una sola fila'"},
         "calle_pegada_a": {"type": "string", "enum": ["norte", "noreste", "oriente", "sureste", "sur", "suroeste",
                                                       "poniente", "noroeste"],
                            "description": "con distribucion=un_lado: el lindero largo al que va pegada la calle, si el "
                                           "usuario lo dice; si no, el motor prueba los dos y escoge"},
         "calle_sobre": {"type": "string", "enum": ["terreno_mas_grande", "terreno_mas_chico", "centro_del_lindero"],
                         "description": "con dos terrenos unidos y calle central: en qué terreno va la calle, pegada al "
                                        "lindero común (por defecto en el más grande)"},
         "lotes_por_lado": {"type": "array", "items": {"type": "integer"},
                            "description": "[mínimo, máximo] de lotes a cada lado de la calle central (cuando se unieron dos "
                                           "terrenos, cada lado es un terreno). '4 o 5 lotes por área' = [4, 5]. No calcules "
                                           "tamaños: el motor reparte"},
         "agua": {"type": "string", "enum": ["anda", "pozo", "no_hay", "no_se"]},
         "luz": {"type": "string", "enum": ["si", "no", "no_se"]},
         "notas": {"type": "string"},
         "usar_sugerido": {"type": "boolean", "description": "true cuando el usuario confirma lo que el bot encontró solo "
                           "(lugar y zona, acceso y quebradas del plano, tipo y lote propuestos): se guarda todo de una vez; "
                           "lo que el usuario corrija va en los demás campos y manda"},
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
         "distribucion": {"type": "string", "enum": ["automatica", "calle_central", "un_lado"],
                          "description": "calle_central = una sola calle por el medio (si se unieron dos terrenos, sobre el "
                                         "lindero común) con lotes a los dos lados, retorno y área verde al fondo; para quintas "
                                         "o pocos lotes, o cuando piden 'una calle en medio'. un_lado = calle pegada a un "
                                         "lindero largo desde la entrada y una sola fila de lotes del otro lado; para "
                                         "terrenos angostos (franjas) o cuando piden 'de un solo lado' o 'una sola fila'"},
         "calle_pegada_a": {"type": "string", "enum": ["norte", "noreste", "oriente", "sureste", "sur", "suroeste",
                                                       "poniente", "noroeste"],
                            "description": "con distribucion=un_lado: el lindero largo al que va pegada la calle, si el "
                                           "usuario lo dice; si no, el motor prueba los dos y escoge"},
         "calle_sobre": {"type": "string", "enum": ["terreno_mas_grande", "terreno_mas_chico", "centro_del_lindero"],
                         "description": "con dos terrenos unidos y calle central: en qué terreno va la calle, pegada al "
                                        "lindero común (por defecto en el más grande)"},
         "lotes_por_lado": {"type": "array", "items": {"type": "integer"},
                            "description": "[mínimo, máximo] de lotes a cada lado de la calle central (cuando se unieron dos "
                                           "terrenos, cada lado es un terreno). '4 o 5 lotes por área' = [4, 5]. No calcules "
                                           "tamaños: el motor reparte"},
         "aprovechar": {"type": "boolean", "description": "máximo aprovechamiento (por defecto sí): el área verde se saca "
                        "solo de lo que falta y de la tierra menos vendible; lo que sobra se vuelve lote o se suma a lotes"},
     })},
    {"nombre": "proponer_alternativas", "descripcion": "Arma varias maneras de lotificar el terreno (más o menos lotes, en "
     "qué terreno va la calle, calle central o retícula), el motor mide cada una (lotes, área vendible, muros, tierra, "
     "normas y revisión) y un modelo que piensa recomienda dos o tres con pros y contras. Manda una imagen con las "
     "opciones lado a lado. Úsala cuando pidan opciones, alternativas, 'qué me conviene' o 'otra forma', o cuando el "
     "prediseño tenga fallas o muros muy altos. Para aplicar la que elijan, llama predisenar con sus 'parametros'.",
     "parametros": _obj({})},
    {"nombre": "generar_entregables", "descripcion": "Genera la lámina en PDF (planialtimétrico y, si hay prediseño, la "
     "lotificación), el KML para Google Earth, el DXF 2D para AutoCAD/Civil 3D y, si ya hay relieve, una vista 3D en "
     "imagen y un modelo 3D (DXF) con terreno, lotes, calles y casas tipo para renderizar en AutoCAD. Se los envía al usuario.",
     "parametros": _obj({"nombre_proyecto": {"type": "string"}, "ubicacion": {"type": "string",
                         "description": "municipio y departamento"},
                         "propietario": {"type": "string", "description": "solo si el usuario lo da por su cuenta: no lo "
                                         "pidas (no transcribimos nombres de personas)"}})},
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
3. UBICAR. Si el plano trae coordenadas Lambert, ya queda ubicado: no pidas nada. Si el estado dice SIN UBICAR (el
   plano solo trae rumbos y distancias, o sus coordenadas son locales), pide en un mensaje corto un punto escrito a mano:
   "Con el cuadro ya tengo la forma exacta del terreno, pero no dónde está, y sin eso no sale el relieve. Mándame la X y
   la Y (Lambert) de cualquier mojón, como vienen en el plano o te las da el topógrafo, por ejemplo: Mo. 1  X 473280.68
   Y 289049.93." Usa los nombres de mojón del plano. → ubicar_terreno con referencia = ese mojón (si dan varios, los
   demás en 'puntos'). NO pidas ni uses la ubicación del teléfono (pin): no es exacta. Si no tienen las coordenadas,
   que se las pidan al topógrafo. Luego obtener_relieve.
4. FICHA DEL PROYECTO → guardar_ficha. Primero lo que el bot ya averiguó: obtener_relieve trae "encontrado" (distrito,
   municipio y si está en el AMSS; acceso y quebradas que dice el plano, con su profundidad aproximada; tipo y lote
   propuestos) y "preguntar_solo". Manda UN mensaje corto: lo encontrado como propuesta ("Esto es lo que encontré: …
   ¿algo está mal?") y, en la misma tanda, solo lo de "preguntar_solo". Si confirma → guardar_ficha con usar_sugerido=true
   y lo que haya contestado; si corrige algo, ese dato va en su campo y manda. No vuelvas a preguntar lo que ya se
   averiguó. Si no hubo relieve (no se pudo ubicar), pregunta lo que falte, máximo 3 preguntas por mensaje, en este
   orden de importancia:
   a) Zona: ¿está en el AMSS (San Salvador y municipios vecinos), en la zona urbana de otro municipio, o fuera de
      centros poblados? Cambia el área verde que pide la ley.
   b) Tipo de proyecto y tamaño de lote que quieren vender (interés social ~150 m², residencial ~250-400 m², campestre
      600 m² o más).
   c) Acceso: ¿por qué lado llega la calle pública?
   d) Quebradas, ríos o taludes: ¿colinda con alguno, de qué lado y qué tan profundo más o menos? La ley pide una franja
      de protección de 1.5 veces la profundidad de la quebrada.
   e) ¿Cerrado con portón o abierto? ¿Cómo se va a vender: lotificación con permiso o condominio? ¿Hay agua (ANDA o
      pozo) y luz?
   Si el terreno es una franja angosta, o piden lotes "de un solo lado" o "en una sola fila" → distribucion=un_lado: la
   calle va pegada a un lindero largo desde la entrada y los lotes quedan en una fila del otro lado (en franjas el motor
   ya lo hace solo). Si dicen de qué lado va la calle o dónde está la entrada, guarda calle_pegada_a y acceso_lado.
   Si piden quintas, pocos lotes ("4 o 5 lotes por área") o "una calle en medio" → guarda distribucion=calle_central y
   lotes_por_lado=[mín, máx]; tipo campestre. Con dos terrenos unidos la calle va sobre el lindero común y cada lado es
   un terreno; por defecto la calle va completa dentro del terreno más grande, pegada al lindero (calle_sobre). No
   calcules el tamaño de lote tú: el motor reparte y te dice cuánto queda.
   Lo que viene "según el mapa (OpenStreetMap)" es pista, no dato del plano: pídele que lo confirme. Si "acceso_mapa"
   dice que el plano y el mapa no coinciden, pregúntalo.
   Si "encontrado" trae "opamss", dile en 2 o 3 líneas lo que dice la OPAMSS del lugar ("según la OPAMSS"): la
   clasificación del suelo, los límites (altura, impermeabilización, edificabilidad) y si hay peligros o una vía
   proyectada. Si trae "alerta" (suelo no urbanizable), dilo primero y claro.
   Si guardar_ficha trae "revisar_quebrada", NO prediseñes todavía: dile al usuario en una línea lo que muestra el
   relieve y pregúntale si de verdad colinda con esa quebrada.
   Si el plano ya dice las colindancias (calle, quebrada), úsalas y solo confírmalas. Si el usuario dice "usa lo
   estándar", sigue con lo que tengas y di qué supusiste.
5. PREDISEÑO → predisenar. Resume: número de lotes, tamaño promedio, área verde, y el cuadro de cumplimiento en corto
   (qué cumple, qué no, qué falta revisar, con su artículo). Con calle central di cuántos lotes quedan de cada lado (o de
   cada terreno), su área y frente, y la alternativa que traiga "alternativas" (por ejemplo con 4 en vez de 5). Si algo no cumple, propone cómo arreglarlo y pregunta antes
   de rehacer.
6. ENTREGABLES → generar_entregables. Lo primero que llega es la CARPETA DE PREFACTIBILIDAD: un solo PDF con todo
   (portada, resumen ejecutivo, carpeta técnica con la lámina a escala, carpeta comercial y anexos), para abrirlo sin
   AutoCAD. Después llegan las vistas (para reenviarlas) y los archivos DXF y KML. Lo que trae el PDF, en orden:
   a) RESUMEN EJECUTIVO (PDF): cuadro general de áreas por terreno en m² y v², verificación de cada plano, lotes con su
      relieve, calle y servicios, relieve, cumplimiento, opciones legales a validar con notario y siguientes pasos.
   b) CARPETA TÉCNICA: plano técnico (planimetría con medidas), plano altimétrico (pendientes, curvas de nivel, perfil
      de la calle y relieve por lote), lámina PDF con el cuadro de cumplimiento, DXF 2D y 3D y KML.
   b1) Al entregar: no pidas el nombre del propietario. Antes de generar, basta el nombre del proyecto (si no lo dan,
      usa uno sencillo con el lugar). Después, di en 2 o 3 líneas qué llegó y lo importante de la revisión; no describas
      el contenido de los archivos con detalles que no vengan en el resultado de la herramienta (número de hojas,
      escalas, capas): los archivos ya llegan con su título.
   b2) La carpeta técnica también trae el CORTE TRANSVERSAL con la terracería: cada lote escalonado en terrazas que
      siguen el terreno, la de la casa pegada a la calle y a no más de 1.5 m de su nivel (el carro entra por una rampa
      de ≤ 15 %), y los cambios de nivel resueltos con MUROS DE CONTENCIÓN (altura de cada muro; los de más de 3 m
      piden diseño estructural), con corte, relleno y m² de muro por lote (aproximados con relieve satelital). El
      resumen compara las dos maneras: taludes (plataforma y el resto del lote con su caída) contra muros (se
      aprovecha casi todo el lote a cambio de los m² de muro). Si preguntan cuál conviene, explica ese cambio; el
      costo de los muros se cotiza con el diseño final.
   c) CARPETA COMERCIAL (pintadas con IA sobre el 3D del motor, ilustrativas y no a escala, respetando hacia dónde baja
      el terreno): vista aérea de ambiente, vista desde el portón, vista desde la quebrada, lotes listos con su
      terracería (sin casas), y el plano de presentación con medidas.
   Pide nombre del proyecto y municipio si no los tienes. Di en una línea que la imagen de ambiente es solo para dar
   idea del proyecto y que las medidas y áreas están en el resumen y la carpeta técnica, que son del motor. Si el
   resumen trae "avisos" (un plano cuyo cuadro no cuadra con su área), dilo primero y claro. Si la planta tiene
   "pintada_con_ia" en false, solo menciona que salió con el trazo del motor.
   REVISIÓN: antes de mandar, el motor revisa la geometría y un revisor con IA revisa las láminas (las vistas con un
   problema grave se repintan solas una vez). El resultado viene en "revision". Si "geometria" trae algo grave, dilo
   primero, en una línea, y ofrece corregirlo (volver a prediseñar). De "laminas" menciona solo lo de gravedad "alta"
   que siga ahí, en pocas palabras y sin dramatizar; lo demás no lo cuentes salvo que pregunten.
   RELIEVE: con relieve, el prediseño deja fuera de los lotes las caídas fuertes (más de 50 %: barrancos, taludes de
   quebrada) y da pendiente y desnivel de cada lote. Si el usuario quiere lotificar también esas caídas, guarda en la
   ficha respetar_barrancos=false y vuelve a prediseñar.

MÁXIMO APROVECHAMIENTO (lo que buscan los desarrolladores):
- predisenar ya lo hace por defecto: pone el área verde en la tierra menos vendible (la franja de la quebrada cuenta como
  área verde ecológica, hasta el 40 % del área verde total; el 60 % debe ser recreativa), recorta de un lote solo lo que
  falta en vez de sacrificar lotes completos, y lo que sobra lo suma a lotes o lo vuelve lote. Explica estas decisiones
  con los números de "aprovechamiento".
- Palancas que puedes explicar: el tamaño y frente de lote, el lado de acceso, unir terrenos colindantes, y la zona (fuera
  de centros poblados la ley pide 12.5 m² por lote en vez del 10 % del área útil; pero la zona la define la autoridad,
  no se escoge).
- Vender en proindiviso para evitar la lotificación NO es una salida: la ley lo presume lotificación y el notario y el
  registrador deben detenerlo. Si lo proponen, dilo claro y ofrece las figuras formales (lotificación o condominio).
- Nunca sugieras evadir normas.

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
            falta = self._faltantes(s, sugerido=bool(s.ficha.get("sugerido")))
            if falta:
                partes.append("Falta preguntar: " + ", ".join(falta) + ".")
            if s.ficha.get("sugerido") and not s.ficha["sugerido"].get("confirmado"):
                partes.append("Hay datos averiguados por el bot en ficha.sugerido: confírmalos de una vez, no los preguntes.")
        else:
            partes.append("Ficha del proyecto: vacía (zona, tipo y lote, acceso, quebradas, servicios).")
        if s.cumplimiento:
            partes.append("Cumplimiento: " + json.dumps(normas.resumen(s.cumplimiento), ensure_ascii=False))
        if s.proyecto:
            partes.append("Datos del proyecto: " + json.dumps(s.proyecto, ensure_ascii=False))
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
        if ubicacion:  # el pin del teléfono no sirve para ubicar el terreno: se piden coordenadas
            if not entrada:
                s.historial += [{"rol": "usuario", "texto": "(compartió su ubicación de WhatsApp)"},
                                {"rol": "modelo", "texto": AVISO_PIN}]
                s.historial = s.historial[-MAX_HISTORIAL:]
                return [{"tipo": "texto", "texto": AVISO_PIN}]
            entrada += ("\n[compartió su ubicación de WhatsApp: no sirve para ubicar el terreno; si falta ubicarlo, pide la "
                        "X y la Y de un mojón]")
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
        return [{"tipo": "texto", "texto": formato_whatsapp(texto_final)}] + salidas

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
        s.partes, s.lindero = [], None
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
        if not t.ubicado:
            extra["falta_ubicar"] = (("las coordenadas del plano no son Lambert de El Salvador (parecen locales)" if fuera
                                      else "el plano solo trae rumbos y distancias")
                                     + ": pide la X y la Y (Lambert) de un mojón escritas a mano")
            extra["mojones"] = t.nombres[:30]
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
                t, extra, conocidas = self._terreno_de_lectura(crudos[0], sistema, crudos[0].get("nombre") or "Terreno 1")
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
            nombre = c.get("nombre") or f"Terreno {len(terrenos) + 1}"
            if any(x.origen == nombre for x in terrenos):  # dos fotos sin nombre o con el mismo nombre
                nombre = f"{nombre} ({len(terrenos) + 1})"
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
        r = self._informe_terreno(s, u, {"unidos": [t.origen for t in cands],
                                         "areas_por_terreno_m2": [round(t.area, 2) for t in cands],
                                         "problemas_de_los_originales": problemas})
        cx, cy = u.centroide()
        orig = {c.origen: c for c in cands}
        s.partes = [{"origen": t.origen, "anillo": [[x - cx, y - cy] for x, y in t.puntos],
                     # para el resumen ejecutivo: lo que dice el plano contra lo que da su cuadro de rumbos
                     "area_declarada": orig.get(t.origen, t).area_declarada, "cierre": round(orig.get(t.origen, t).cierre, 3),
                     "area_cuadro": round(orig.get(t.origen, t).area, 2)} for t in colocados]
        lc = lindero_comun(colocados[0].puntos, colocados[1].puntos) if len(colocados) == 2 else None
        s.lindero = [[x - cx, y - cy] for x, y in lc] if lc else None
        if lc:
            r["lindero_comun_m"] = round(sum(math.dist(a, b) for a, b in zip(lc, lc[1:])), 2)
        return r

    # ------------------------------------------------------------------ ficha del proyecto
    def _faltantes(self, s: Sesion, sugerido: bool = False) -> list[str]:
        """Lo que falta preguntar. Con `sugerido`, lo que el bot ya averiguó no cuenta (se confirma de una vez)."""
        f = dict(s.ficha)
        if sugerido and f.get("sugerido"):
            for k, v in f["sugerido"].get("ficha", {}).items():
                f.setdefault(k, v)
        falta = []
        if not f.get("zona"):
            falta.append("zona: ¿está en el casco urbano (o un centro poblado) o fuera?" if sugerido
                         else "zona (AMSS, urbana o fuera de centros poblados)")
        if not f.get("lote_objetivo_m2") and not f.get("tipo") and not f.get("lotes_por_lado"):
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
        if not f.get("figura"):
            falta.append("cómo se va a vender (lotificación o condominio)")
        if not f.get("agua"):
            falta.append("agua y luz")
        return falta

    def h_guardar_ficha(self, s: Sesion, salidas=None, usar_sugerido=False, **datos) -> dict:
        campos = {"zona", "tipo", "lote_objetivo_m2", "frente_m", "acceso_lado", "cerrado", "quebradas", "sin_quebradas",
                  "figura", "agua", "luz", "notas", "distribucion", "lotes_por_lado", "calle_sobre", "calle_pegada_a",
                  "quebrada_confirmada"}
        if usar_sugerido and s.ficha.get("sugerido"):
            for k, v in s.ficha["sugerido"].get("ficha", {}).items():
                if datos.get(k) is None and (k not in s.ficha or k == "sin_quebradas"):
                    datos[k] = v
            s.ficha["sugerido"]["confirmado"] = True
        for k, v in datos.items():
            if k in campos and v is not None:
                if k in ("acceso_lado", "calle_pegada_a") and str(v).lower() not in LADOS:
                    return {"ok": False, "error": f"lado no reconocido: {v}"}
                if k == "lotes_por_lado":
                    try:
                        v = sorted(int(x) for x in (v if isinstance(v, (list, tuple)) else [v]))
                    except (TypeError, ValueError):
                        return {"ok": False, "error": "lotes_por_lado debe ser [mínimo, máximo]"}
                    v = [max(1, v[0]), max(1, v[-1])]
                s.ficha[k] = v
        if s.ficha.get("quebradas"):
            s.ficha["sin_quebradas"] = False
        out = {"ok": True, "ficha": {k: v for k, v in s.ficha.items() if k not in ("del_plano", "sugerido")},
               "falta_preguntar": self._faltantes(s)}
        revisar = self._revisar_quebradas(s)
        if revisar:
            out["revisar_quebrada"] = revisar
        return out

    def _revisar_quebradas(self, s: Sesion) -> list[str]:
        """Las quebradas que dijo el usuario contra el relieve: si de ese lado no se ve un cauce, hay que preguntar
        antes de quitarle terreno con la franja de protección. Las que dice el plano no se cuestionan."""
        m, t = s.get_malla(), s.get_terreno()
        if m is None or t is None or s.ficha.get("quebrada_confirmada"):
            return []
        del_plano = {str(c.get("lado", "")).lower() for c in s.ficha.get("del_plano") or []
                     if c.get("tipo") in ("quebrada", "rio")}
        out = []
        for q in s.ficha.get("quebradas") or []:
            lado = str(q.get("lado") or "").lower()
            if q.get("tipo") not in ("quebrada", "rio") or lado not in LADOS or lado in del_plano:
                continue
            try:
                c = territorio.cauce_al_lado(m, t.puntos, lado)
            except Exception:  # noqa: BLE001
                log.exception("cauce al lado")
                continue
            if not c or c["veredicto"] == "cauce":
                continue
            if any(a["lado"] == lado and a["distancia_m"] <= 50 and a["tipo"] in ("quebrada", "río")
                   for a in (s.osm or {}).get("aguas") or []):
                continue  # el mapa (OpenStreetMap) la muestra de ese lado
            que = {"ladera": f"del lado {lado} el terreno solo sigue bajando (ladera), sin el fondo de una quebrada",
                   "sube": f"del lado {lado} el terreno sube: ahí no se ve una quebrada",
                   "plano": f"del lado {lado} el terreno está casi parejo: no se ve una quebrada"}[c["veredicto"]]
            out.append(f"El relieve satelital no muestra un cauce junto al lindero {lado} (en los primeros "
                       f"{c['alcance_m']:.0f} m {que}). Antes de prediseñar, pregúntale al usuario si de verdad colinda "
                       f"con una {q['tipo']} de ese lado: la franja de protección le quita terreno. Si confirma, guarda "
                       f"quebrada_confirmada=true; si no, quítala (sin_quebradas=true).")
        return out

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
        # caídas fuertes (barrancos, taludes de quebrada) según el relieve: no se lotifican
        barr_m2, barr_zonas = 0.0, []
        m = s.get_malla()
        if m is not None and f.get("respetar_barrancos", True):
            try:
                barr = altimetria.barrancos(m, t.puntos)
            except Exception:  # noqa: BLE001
                log.exception("barrancos")
                barr = []
            if barr:
                from shapely.ops import unary_union

                previo = unary_union(excluir) if excluir else None
                zona_b = unary_union(barr)
                barr_m2 = (zona_b.difference(previo) if previo is not None else zona_b).area  # lo que no cubría la franja
                excluir.extend(barr)
                barr_zonas = list(barr)
        ficha_normas = {"zona": zona, "tipo": f.get("tipo"), "protecciones": prot, "cerrado": f.get("cerrado"),
                        "barrancos_m2": barr_m2, "barrancos_zonas": [b for b in excluir if any(b is x for x in barr_zonas)],
                        "figura": f.get("figura"), "opamss": s.opamss,
                        "quebrada_sin_datos": (any(c.get("tipo") in ("quebrada", "rio") for c in f.get("del_plano", []))
                                               and not f.get("quebradas") and not f.get("sin_quebradas"))}
        return Parametros.de_dict(base), excluir, ficha_normas

    def _modo(self, s: Sesion) -> tuple[str, list | None]:
        lp, f = s.lotes_param or {}, s.ficha
        pedido = lp.get("distribucion") or f.get("distribucion")
        modo = pedido or "automatica"
        rango = lp.get("lotes_por_lado") or f.get("lotes_por_lado")
        if rango and modo == "automatica" and not pedido:
            modo = "calle_central"  # pedir N lotes por lado ya implica la calle al centro
        if not pedido and not rango and len(s.partes) <= 1:
            t = s.get_terreno()
            if t is not None and es_angosto(Polygon(t.puntos).buffer(0)):
                modo = "un_lado"  # en una franja no caben lotes a los dos lados: una fila con la calle pegada al lindero
        return modo, rango

    @staticmethod
    def _falta_equipamiento(r, P: Parametros, fn: dict) -> bool:
        """Si la norma pide equipamiento social y no se dejó, lo pone en P (para rehacer) y dice True."""
        if not fn.get("zona"):
            return False
        req, _, claro = normas.equipamiento_requerido(fn["zona"], r.stats["lotes"], fn.get("tipo") or "residencial")
        if claro and r.stats.get("area_equipamiento", 0.0) < req - 0.5:
            P.equipamiento_m2 = req
            return True
        return False

    def _lotificar(self, s: Sesion):
        """Prediseño + revisión de normas. Si la norma pide equipamiento social, se reserva y se rehace."""
        t = s.get_terreno()
        m = s.get_malla()
        P, excluir, fn = self._parametros(s)
        modo, rango = self._modo(s)
        if modo == "un_lado":
            barr = fn.get("barrancos_zonas") or []
            legales = [e for e in excluir if not any(e is b for b in barr)]
            if m is not None and s.ficha.get("respetar_barrancos", True):
                try:  # en una fila la calle cruza las caídas fuertes: también cuentan las que van de lado a lado
                    barr = altimetria.barrancos(m, t.puntos, partir=True)
                except Exception:  # noqa: BLE001
                    log.exception("barrancos")
                if barr:
                    from shapely.ops import unary_union

                    zona_b = unary_union(barr)
                    fn["barrancos_m2"] = (zona_b.difference(unary_union(legales)) if legales else zona_b).area
            borde = (s.lotes_param or {}).get("calle_pegada_a") or s.ficha.get("calle_pegada_a")
            r = lotificar_un_lado(t.puntos, P, m, legales, borde=borde, lotes=rango, cruzables=barr)
            if self._falta_equipamiento(r, P, fn):
                r = lotificar_un_lado(t.puntos, P, m, legales, borde=borde, lotes=rango, cruzables=barr)
            altimetria.enriquecer(r, m)
            filas = normas.revisar(r, t, m, fn)
            s.cumplimiento = filas
            return r, filas
        if modo == "calle_central":
            cx, cy = t.centroide()
            eje = [(x + cx, y + cy) for x, y in s.lindero] if s.lindero else None
            partes = [{"origen": p["origen"], "anillo": [(x + cx, y + cy) for x, y in p["anillo"]]} for p in s.partes]
            sobre = (s.lotes_param or {}).get("calle_sobre") or s.ficha.get("calle_sobre") or "terreno_mas_grande"
            sobre = {"terreno_mas_grande": "mayor", "terreno_mas_chico": "menor"}.get(sobre, "lindero")
            r = lotificar_calle_central(t.puntos, P, m, excluir, eje=eje, lotes_por_lado=rango or (4, 5), partes=partes,
                                        calle_sobre=sobre)
            if self._falta_equipamiento(r, P, fn):
                r = lotificar_calle_central(t.puntos, P, m, excluir, eje=eje, lotes_por_lado=rango or (4, 5),
                                            partes=partes, calle_sobre=sobre)
            altimetria.enriquecer(r, m)
            filas = normas.revisar(r, t, m, fn)
            s.cumplimiento = filas
            return r, filas
        r = lotificar(t.puntos, P, m, excluir)
        if fn.get("zona"):
            req, _, claro = normas.equipamiento_requerido(fn["zona"], r.stats["lotes"], fn.get("tipo") or "residencial")
            if claro and r.stats["area_equipamiento"] < req - 0.5:
                P.equipamiento_m2 = req
                r = lotificar(t.puntos, P, m, excluir)
        altimetria.enriquecer(r, m)
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

    def h_ubicar_terreno(self, s: Sesion, lat=None, lng=None, x=None, y=None, referencia=None, sistema="SIRGAS",
                         puntos=None, salidas=None) -> dict:
        geo = (x is None or y is None) and lat is not None and lng is not None  # latitud/longitud escritas de un mojón
        if geo:
            x, y = proyeccion.a_lambert(float(lat), float(lng))
        if x is None or y is None:
            return {"ok": False, "error": "faltan las coordenadas: la X y la Y (Lambert) de un mojón"}
        x, y = float(x), float(y)
        if not geo:
            if not proyeccion.en_el_salvador(x, y) and proyeccion.en_el_salvador(y, x):
                x, y = y, x  # escribieron primero la Y
            if sistema == "NAD27":
                x, y = proyeccion.nad27_a_sirgas(x, y)
        if not proyeccion.en_el_salvador(x, y):
            return {"ok": False, "error": "ese punto no cae en El Salvador" + (
                "" if geo else ": no parecen coordenadas Lambert. Muchos topógrafos usan coordenadas locales (que "
                "arrancan en 1000, 1000 o parecido) para dibujar; esas no dicen dónde está el terreno. Pide la X y la Y "
                "Lambert de un mojón (en El Salvador la X va de 380,000 a 640,000 y la Y de 225,000 a 370,000).")}
        t = s.get_terreno()
        if t is None:
            return {"ok": False, "error": "primero hace falta el plano (el cuadro de rumbos); después las coordenadas "
                                          "de un mojón"}
        ref = (referencia or "").strip()
        if not ref:
            return {"ok": False, "error": "falta decir de qué mojón son esas coordenadas", "mojones": t.nombres[:30],
                    "nota": "Pregunta a qué mojón del cuadro corresponden."}
        if t.indice_vertice(ref) is None:
            return {"ok": False, "error": f"no encuentro el mojón '{referencia}' en el plano",
                    "mojones": t.nombres[:30], "nota": "Pregunta a qué mojón del cuadro corresponde ese punto."}
        t.ubicar_en(x, y, ref)
        out = {"ok": True}

        def mojon(i):
            n = str(t.nombres[i])
            return f"Mo. {n}" if n.isdigit() else n

        comprobados, lejos = [], []
        for p in puntos or []:
            i = t.indice_vertice(p.get("mojon"))
            if i is None or p.get("x") is None or p.get("y") is None:
                continue
            px, py = float(p["x"]), float(p["y"])
            if not proyeccion.en_el_salvador(px, py) and proyeccion.en_el_salvador(py, px):
                px, py = py, px
            if sistema == "NAD27":
                px, py = proyeccion.nad27_a_sirgas(px, py)
            d = math.hypot(t.puntos[i][0] - px, t.puntos[i][1] - py)
            comprobados.append({"mojon": mojon(i), "diferencia_m": round(d, 2)})
            if d > 2.0:
                lejos.append(mojon(i))
        s.set_terreno(t)  # borra el relieve y el prediseño de la ubicación anterior
        lat_c, lng_c = proyeccion.a_latlng(*t.centroide())
        out["centro_latlng"] = [round(lat_c, 6), round(lng_c, 6)]
        out["nota"] = f"Ubicado con la coordenada del {mojon(t.indice_vertice(ref))}."
        if comprobados:
            out["comprobacion"] = comprobados
        if lejos:
            out["aviso"] = (f"Los mojones {', '.join(lejos)} no cuadran (más de 2 m): revisa que las coordenadas estén bien "
                            "escritas y que cada una sea del mojón que se dice.")
        return out

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
        out = {"ok": True, "estadisticas": st, "intervalo_curvas_m": relieve.intervalo_curvas(st["desnivel"])}
        try:
            enc = self._averiguar(s, t, m, st)
        except Exception:  # noqa: BLE001 - lo que no se averigua se pregunta
            log.exception("averiguar contexto")
            enc = None
        if enc:
            out["encontrado"] = enc["encontrado"]
            out["preguntar_solo"] = self._faltantes(s, sugerido=True)
            out["nota"] = ("Manda UN solo mensaje: lo que encontraste (dilo como propuesta y pide que lo corrija si algo "
                           "está mal) y, en la misma tanda, solo lo que hay en 'preguntar_solo'. Si confirma, guardar_ficha "
                           "con usar_sugerido=true más lo que conteste.")
        return out

    def _averiguar(self, s: Sesion, t, m, st: dict) -> dict | None:
        """Lo que se puede saber sin preguntar: dónde cae (y si está en el AMSS, con lo que dice la OPAMSS), el acceso y
        las quebradas que dicen el plano y el mapa (OpenStreetMap), con la profundidad aproximada por el relieve, y el
        tipo de proyecto que le queda. Queda en la ficha como 'sugerido' hasta que el usuario lo confirme."""
        from concurrent.futures import ThreadPoolExecutor

        enc, ficha = {}, {}
        lugar = territorio.donde(*t.centroide())
        amss = bool(lugar and lugar["amss"])
        with ThreadPoolExecutor(max_workers=2) as ex:  # la OPAMSS y el mapa a la vez
            f_op = ex.submit(geovisor_opamss.consultar, t.puntos) if amss else None
            f_osm = ex.submit(mapa_osm.consultar, t.puntos)
            for nombre, fut in (("opamss", f_op), ("osm", f_osm)):
                try:
                    setattr(s, nombre, fut.result() if fut else None)
                except Exception:  # noqa: BLE001
                    log.exception("consulta %s", nombre)
                    setattr(s, nombre, None)
        if lugar:
            enc["lugar"] = lugar
            if amss:
                ficha["zona"] = "amss"
                enc["zona"] = ("dentro del Área Metropolitana de San Salvador: el permiso va por la OPAMSS y aplican sus "
                               "normas (área verde 10 % del área útil)")
                enc["opamss"] = (resumen_opamss(s.opamss) if s.opamss else
                                 "no se pudo consultar el geovisor de la OPAMSS (se puede volver a intentar)")
        osm = s.osm or {}
        if s.osm:
            enc["mapa"] = mapa_osm.resumen(s.osm)
        plano = s.ficha.get("del_plano") or []
        calles = [c["lado"] for c in plano if c.get("tipo") == "calle" and c.get("lado") in LADOS]
        calle_mapa = next((c for c in osm.get("calles") or [] if c["colinda"]), None)
        if calles:
            ficha["acceso_lado"] = calles[0]
            enc["acceso"] = f"la calle pública llega por el {calles[0]} (lo dice el plano)"
            if calle_mapa and calle_mapa["lado"] != calles[0] and not any(
                    c["colinda"] and c["lado"] == calles[0] for c in osm.get("calles") or []):
                enc["acceso_mapa"] = (f"ojo: el plano dice que la calle llega por el {calles[0]}, pero en el mapa la calle "
                                      f"pegada al terreno está al {calle_mapa['lado']} ({calle_mapa['tipo']}); confírmalo")
        elif calle_mapa:
            ficha["acceso_lado"] = calle_mapa["lado"]
            enc["acceso"] = (f"según el mapa (OpenStreetMap), una {calle_mapa['tipo']} pasa pegada al lado "
                             f"{calle_mapa['lado']}: por ahí sería el acceso (confírmalo)")
        qs = []
        for c in plano:
            if c.get("tipo") in ("quebrada", "rio") and c.get("lado") in LADOS:
                qs.append({"tipo": c["tipo"], "lado": c["lado"], "origen": "plano"})
        lados_q = {q["lado"] for q in qs}
        for a in osm.get("aguas") or []:
            if a["colinda"] and a["tipo"] in ("quebrada", "río") and a["lado"] not in lados_q:
                qs.append({"tipo": "rio" if a["tipo"] == "río" else "quebrada", "lado": a["lado"], "origen": "mapa",
                           **({"nombre": a["nombre"]} if a.get("nombre") else {})})
                lados_q.add(a["lado"])
        for q in qs:
            p = territorio.profundidad_al_lado(m, t.puntos, q["lado"]) if q["tipo"] == "quebrada" else None
            if p and p["tipica_m"] >= 1.0:
                q["profundidad_m"] = max(p["tipica_m"], 1.0)
                q["profundidad_aprox"] = f"unos {p['tipica_m']:g} m (hasta {p['maxima_m']:g} m), medida con el relieve satelital"
        if qs:
            ficha["quebradas"] = [{k: v for k, v in q.items() if k in ("tipo", "lado", "profundidad_m")} for q in qs]
            enc["quebradas"] = [dict(q, origen="lo dice el plano" if q["origen"] == "plano" else
                                     "aparece en el mapa (OpenStreetMap); confírmalo") for q in qs]
        elif plano:
            ficha["sin_quebradas"] = True
            enc["quebradas"] = "el plano no menciona quebradas ni ríos en las colindancias" + (
                " y en el mapa tampoco aparece ninguna pegada al terreno" if s.osm else "")
        prop = territorio.propuesta_proyecto(t.area, st.get("pendiente_media"), amss)
        ficha.update(tipo=prop["tipo"], lote_objetivo_m2=prop["lote_objetivo_m2"])
        enc["propuesta"] = prop
        s.ficha["sugerido"] = {"encontrado": enc, "ficha": ficha}
        return s.ficha["sugerido"]

    def h_predisenar(self, s: Sesion, salidas=None, **params) -> dict:
        t = s.get_terreno()
        if t is None:
            return {"ok": False, "error": "primero hay que cargar el terreno"}
        validos = {k: v for k, v in params.items()
                   if (k in Parametros.__dataclass_fields__
                       or k in ("distribucion", "lotes_por_lado", "calle_sobre", "calle_pegada_a"))
                   and v is not None}
        base = dict(s.lotes_param or {})
        base.update(validos)
        s.lotes_param = base
        r, filas = self._lotificar(s)
        st = r.stats
        s.lotes_stats = {k: (_r(v) if isinstance(v, (int, float)) else v) for k, v in st.items()}
        inclinados = sum(1 for l in r.lotes if l["pendiente"] is not None and l["pendiente"] > 30)
        out = {"ok": True, "lotes": st["lotes"], "area_util_m2": _r(st["area_util"]),
               "area_calles_m2": _r(st["area_circulacion"]), "area_verde_m2": _r(st["area_verde"]),
               "area_verde_requerida_m2": _r(st["area_verde_requerida"]),
               "lotes_pasados_a_area_verde": st["lotes_convertidos_a_verde"],
               "lote_promedio_m2": _r(st["lote_promedio"]), "lote_min_m2": _r(st["lote_min"]), "lote_max_m2": _r(st["lote_max"]),
               "aprovechamiento_pct": _r(st["area_util"] / st["area_total"] * 100, 1),
               "parametros": r.parametros, "orientacion_calles_grados": r.angulo,
               "area_proteccion_m2": _r(st["area_proteccion"]), "area_equipamiento_m2": _r(st["area_equipamiento"]),
               "aprovechamiento": {"vendible_pct": _r(st["area_util"] / st["area_total"] * 100, 1),
                                   "lotes_recortados_para_area_verde": st.get("lotes_recortados_para_verde", 0),
                                   "lotes_completos_pasados_a_verde": st["lotes_convertidos_a_verde"],
                                   "lotes_nuevos_de_area_sobrante": st.get("lotes_de_area_sobrante", 0),
                                   "m2_sobrantes_sumados_a_lotes": _r(st.get("area_sobrante_sumada_a_lotes", 0)),
                                   "lotes_grandes_partidos": st.get("lotes_partidos", 0)},
               "cumplimiento": [{k: f[k] for k in ("regla", "base", "requerido", "proyecto", "estado", "nota") if f.get(k)}
                                for f in filas],
               "resumen_cumplimiento": normas.resumen(filas),
               "supuestos": self._faltantes(s)}
        if st.get("modo") == "calle_central":
            out["distribucion"] = "un_lado" if st.get("un_lado") else "calle_central"
            if st.get("un_lado"):
                out["calle_pegada_a"] = st.get("calle_pegada_a")
                out["nota_un_lado"] = ("una sola fila de lotes; la calle va pegada al lindero " +
                                       str(st.get("calle_pegada_a")) + " desde la entrada; si la entrada o la calle van "
                                       "del otro lado, se puede cambiar con calle_pegada_a")
            out["por_lado"] = st["por_lado"]
            out["retorno_diametro_m"] = _r(st["retorno_diametro_m"], 1)
            if st.get("calle_dentro_de"):
                out["calle_dentro_de"] = st["calle_dentro_de"]
            out["lotes_detalle"] = [{"lote": l["id"], "terreno": l.get("terreno"), "area_m2": _r(l["area"], 1),
                                     "frente_m": _r(l["frente"], 1),
                                     **({"pendiente_media_pct": l["relieve"]["pend_media"],
                                         "desnivel_m": l["relieve"]["desnivel"], "cae_hacia": l["relieve"]["cae_hacia"]}
                                        if l.get("relieve") else {})} for l in r.lotes]
            out.pop("orientacion_calles_grados", None)
        if s.malla:
            out["lotes_con_pendiente_mayor_30pct"] = inclinados
            barr = next((f_ for f_ in filas if f_["regla"].startswith("Caídas fuertes")), None)
            if barr:
                out["barrancos_fuera_de_lotes"] = barr["proyecto"]
            out["pendiente_media_de_calles_pct"] = _r(st["pendiente_calles_pct"], 1)
        else:
            out["nota"] = "Sin relieve: el prediseño no consideró pendientes."
        return out

    def h_proponer_alternativas(self, s: Sesion, salidas=None) -> dict:
        t = s.get_terreno()
        if t is None:
            return {"ok": False, "error": "primero hay que cargar el terreno"}
        from motor import disenador

        m = s.get_malla()
        modo, rango = self._modo(s)
        original, cumplimiento = s.lotes_param, s.cumplimiento  # medir variantes no cambia el prediseño vigente
        medidas, resultados = [], {}
        try:
            try:
                P0 = self._parametros(s)[0]
                hoy = {"frente": P0.frente, "fondo": P0.fondo, "area_minima": P0.area_minima}
            except Exception:  # noqa: BLE001
                hoy = None
            from motor.calle_central import bordes_largos

            poly = Polygon(t.puntos).buffer(0)
            una = len(s.partes) <= 1
            for v in disenador.variantes(s.lotes_param or {}, s.ficha, len(s.partes) > 1 and bool(s.lindero), modo, rango,
                                         medidas=hoy, angosto=una and es_angosto(poly),
                                         bordes=bordes_largos(poly) if una else None,
                                         borde_actual=(s.lotes_stats or {}).get("calle_pegada_a")):
                s.lotes_param = v["param"]
                try:
                    r, filas = self._lotificar(s)
                except Exception as e:  # noqa: BLE001 - una variante que no cabe no tumba las demás
                    log.info("variante %s no cupo: %s", v["id"], e)
                    continue
                medidas.append({**v, "medidas": disenador.medir(t, r, m, s.ficha, filas)})
                resultados[v["id"]] = r
        finally:
            s.lotes_param, s.cumplimiento = original, cumplimiento
        if not medidas:
            return {"ok": False, "error": "ninguna variante cupo en el terreno"}
        rec = disenador.recomendar(getattr(self.llm, "pensar", None), medidas,
                                   disenador.contexto(t, m, s.ficha, s.lotes_param or {}))
        por_id = {v["id"]: v for v in medidas}
        opciones = []
        for k, x in enumerate(rec["recomendadas"], 1):
            v = por_id[x["id"]]
            opciones.append({"opcion": k, "titulo": x["titulo"], "por_que": x["por_que"], "contra": x["contra"],
                             "cambio": v["cambio"], "medidas": v["medidas"],
                             "parametros": {kk: vv for kk, vv in v["param"].items() if kk != "entrada"}})
        try:
            carpeta = os.path.join(self.dir, uuid.uuid4().hex)
            os.makedirs(carpeta, exist_ok=True)
            ruta = disenador.imagen(t, [(x["id"], x["titulo"], resultados[x["id"]], por_id[x["id"]]["medidas"])
                                        for x in rec["recomendadas"]], os.path.join(carpeta, "opciones.jpg"))
            if salidas is not None:
                salidas.append({"tipo": "archivo", "ruta": ruta, "nombre": "opciones.jpg", "imagen": True,
                                "titulo": "Opciones de lotificación lado a lado"})
        except Exception:  # noqa: BLE001
            log.exception("imagen de opciones")
        return {"ok": True, "opciones": opciones, "advertencias": rec["advertencias"], "eligio": rec["por"],
                "variantes_medidas": len(medidas),
                "nota": "Presenta las opciones en pocas líneas (número, título, por qué, contra y 2-3 cifras) y pregunta "
                        "cuál quiere. Para aplicarla llama predisenar con sus 'parametros'. No inventes cifras."}

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
        # imagen de presentación: el motor traza, el modelo de imagen solo pinta, los letreros los pone el motor
        presentacion_ = []
        chequeos = []
        if lot is not None:
            # revisión geométrica del motor (reglas exactas) antes de dibujar nada
            from motor import revisor

            try:
                terr = altimetria.terrazas(lot, m) if (m is not None and t.ubicado) else None
                chequeos = revisor.revisar(t, lot, m, s.ficha, terr)
            except Exception:  # noqa: BLE001
                log.exception("revisión geométrica")
            cx, cy = t.centroide()
            partes = [{**p, "anillo": [(x + cx, y + cy) for x, y in p["anillo"]]} for p in s.partes]
            try:
                presentacion_ = presentacion.generar(carpeta, base, t, lot, m if t.ubicado else None, s.ficha, nombre,
                                                     partes or None, getattr(self.llm, "pintar", None),
                                                     s.proyecto.get("ubicacion", ""),
                                                     revisor=getattr(self.llm, "revisar", None),
                                                     contexto=revisor.contexto_proyecto(lot, s.ficha, chequeos))
            except Exception:  # noqa: BLE001
                log.exception("imagen de presentación")
        pdf = os.path.join(carpeta, f"{base}.pdf")
        info = lamina.generar(pdf, t, m, lot, datos, cumplimiento=filas,
                              imagenes=[(p["ruta"], p["titulo"]) for p in presentacion_ if p["vista"] == "planta"])
        cs = None
        if m is not None:
            cs = relieve.curvas(m, info["intervalo_curvas"], recorte=Polygon(t.puntos).buffer(20))
        archivos = [(pdf, "Lámina en PDF (imprimir en tabloide al 100 %)")]
        if t.ubicado:
            kml = os.path.join(carpeta, f"{base}.kml")
            with open(kml, "w", encoding="utf-8") as f:
                f.write(exportar.kml(nombre, t, cs, lot))
            archivos.append((kml, "Ubicación para Google Earth (KML)"))
        dxf = os.path.join(carpeta, f"{base}.dxf")
        exportar.dxf(dxf, t, cs, lot)
        archivos.append((dxf, "Plano 2D para AutoCAD / Civil 3D (coordenadas Lambert SIRGAS)"))
        if m is not None and t.ubicado:
            escena = modelo3d.construir(t, m, lot)
            if not any(p["vista"] in ("aerea", "ambiente") for p in presentacion_):  # ya hay vista aérea
                jpg = os.path.join(carpeta, f"{base}-vista-3d.jpg")
                sub = (f"{len(lot.lotes)} lotes · " if lot else "") + (s.proyecto.get("ubicacion") or "")
                modelo3d.vista(jpg, escena, t, nombre, sub.strip(" ·"), malla=m)
                archivos.insert(0, (jpg, "Vista 3D ilustrativa del anteproyecto"))
            dxf3d = os.path.join(carpeta, f"{base}-3d.dxf")
            modelo3d.dxf(dxf3d, escena)
            archivos.append((dxf3d, "Modelo 3D para AutoCAD: terreno, lotes, calles y casas tipo (vista Realista → Render)"))
        # tres partes: resumen ejecutivo, carpeta técnica y carpeta comercial
        comercial = [(p["ruta"], p["titulo"]) for p in presentacion_
                     if p["vista"] in ("ambiente", "aerea", "porton", "quebrada", "limpios", "planta")]
        tecnica = [(p["ruta"], p["titulo"]) for p in presentacion_ if p["vista"] in ("tecnica", "altimetria", "corte")]
        for ruta, titulo in archivos:
            (comercial if ruta.endswith("-vista-3d.jpg") else tecnica).append((ruta, titulo))
        archivos = []
        avisos = []
        if lot is not None:
            try:
                from motor.resumen import generar_resumen, recortar_dibujo

                tec = next((p["ruta"] for p in presentacion_ if p["vista"] == "tecnica"), None)
                img_res = recortar_dibujo(tec, os.path.join(carpeta, "_dibujo.jpg"), t) if tec else None
                entrega = {"Carpeta técnica": [_nombre_corto(t_) for _, t_ in tecnica],
                           "Carpeta comercial": [_nombre_corto(t_) for _, t_ in comercial]}
                rpdf = os.path.join(carpeta, f"{base}-resumen-ejecutivo.pdf")
                cx, cy = t.centroide()
                partes_r = [{**p, "anillo": [(x + cx, y + cy) for x, y in p["anillo"]]} for p in s.partes] or None
                res = generar_resumen(rpdf, t, lot, m if t.ubicado else None, dict(s.ficha, opamss=s.opamss, osm=s.osm), nombre,
                                      partes_r, s.proyecto.get("ubicacion", ""), filas, img_res, entrega)
                avisos = res["avisos"]
                from motor.revisor import avisos_de_revision

                avisos += avisos_de_revision(chequeos)
                archivos.append((rpdf, "Resumen ejecutivo (PDF): áreas, lotes, calle, relieve, normas y siguientes pasos"))
            except Exception:  # noqa: BLE001
                log.exception("resumen ejecutivo")
        archivos += [(r_, f"Carpeta técnica · {t_}") for r_, t_ in tecnica]
        archivos += [(r_, f"Carpeta comercial · {t_}") for r_, t_ in comercial]
        # todo en un solo PDF (para abrirlo sin AutoCAD); aparte solo van las vistas (para reenviarlas por WhatsApp) y
        # los archivos de CAD y Google Earth
        try:
            from motor.carpeta import generar_carpeta

            img_tec = [(r_, t_) for r_, t_ in tecnica if r_.endswith(".jpg")]
            lam = next((r_ for r_, _ in tecnica if r_.endswith(".pdf") and "resumen" not in r_), None)
            cad = [(r_, t_) for r_, t_ in tecnica if r_.endswith((".dxf", ".kml"))]
            usos = {".kml": "Abrir en Google Earth: el terreno y los lotes sobre el mapa.",
                    "-3d.dxf": "Abrir en AutoCAD (vista Realista o Render): terreno, lotes, calles y casas en 3D.",
                    ".dxf": "Abrir en AutoCAD o Civil 3D: plano 2D con coordenadas Lambert SIRGAS por capas."}
            anexos = [(os.path.basename(r_), next(v for k, v in usos.items() if r_.endswith(k))) for r_, _ in cad]
            res_pdf = next((r_ for r_, _ in archivos if r_.endswith("-resumen-ejecutivo.pdf")), None)
            portada = next((p["ruta"] for p in presentacion_ if p["vista"] in ("ambiente", "aerea")), None) or \
                next((p["ruta"] for p in presentacion_ if p["vista"] == "planta"), None)
            cifras = []
            if lot is not None:
                st_ = lot.stats
                ar = [l["area"] for l in lot.lotes]
                cifras = [(f"{st_['lotes']}", "lotes"), (f"{st_['area_util']:,.0f} m²", "vendibles"),
                          (f"{min(ar):,.0f}–{max(ar):,.0f} m²", "por lote"),
                          (f"{st_['area_util'] / st_['area_total'] * 100:.0f} %", "del terreno se vende")]
            cpdf = os.path.join(carpeta, f"{base}-carpeta-prefactibilidad.pdf")
            generar_carpeta(cpdf, nombre, s.proyecto.get("ubicacion", ""), cifras, portada, res_pdf, img_tec, lam,
                            comercial, anexos, avisos)
            archivos = ([(cpdf, "Carpeta de prefactibilidad (PDF con todo: resumen ejecutivo, planos y vistas)")]
                        + [(r_, f"Carpeta comercial · {t_}") for r_, t_ in comercial]
                        + [(r_, f"Para AutoCAD y Google Earth · {t_}") for r_, t_ in cad])
        except Exception:  # noqa: BLE001
            log.exception("carpeta de prefactibilidad")
        for ruta, titulo in archivos:
            salidas.append({"tipo": "archivo", "ruta": ruta, "nombre": os.path.basename(ruta), "titulo": titulo,
                            "imagen": ruta.endswith(".jpg")})
        out = {"ok": True, "archivos_enviados": [os.path.basename(a) for a, _ in archivos], "escala": f"1:{info['escala']:,}",
               "partes": {"carpeta_prefactibilidad_pdf": any(a.endswith("-carpeta-prefactibilidad.pdf") for a, _ in archivos),
                          "carpeta_tecnica": len(tecnica), "carpeta_comercial": len(comercial)},
               "hojas": info["hojas"], "nota": "Los archivos se envían automáticamente después de tu mensaje."}
        if avisos:
            out["avisos"] = avisos
        revision = {}
        if chequeos:
            revision["geometria"] = ([{"chequeo": c["chequeo"], "grave": c["grave"], "detalle": c["detalle"]}
                                      for c in chequeos if not c["ok"]] or "sin fallas")
        laminas = [{"vista": p["vista"], **q, **({"repintada": True} if p.get("repintada") else {})}
                   for p in presentacion_ for q in p.get("revision") or []]
        if getattr(self.llm, "revisar", None) is not None and presentacion_:
            if any(p.get("revision_fallo") for p in presentacion_):
                revision["laminas"] = "no se pudieron revisar: falló el modelo revisor (no quiere decir que estén bien)"
            else:
                revision["laminas"] = laminas or "sin problemas"
        repintadas = [p["vista"] for p in presentacion_ if p.get("repintada")]
        if repintadas:
            revision["vistas_repintadas"] = repintadas
        if revision:
            out["revision"] = revision
        if presentacion_:
            out["presentacion"] = [{"vista": p["vista"], "pintada_con_ia": p["ia"], **({"motivo": p["motivo"]} if p.get("motivo") else {})}
                                   for p in presentacion_]
            out["nota_presentacion"] = ("La imagen de ambiente es ilustrativa y no a escala (la IA la pinta libre). Los "
                                        "planos de presentación y la lámina llevan el trazo y los números del motor.")
        return out


def _nombre_corto(titulo: str) -> str:
    """'Plano altimétrico: pendientes, ...' -> 'plano altimétrico'."""
    t = re.split(r"[:(]", titulo)[0].strip()
    return t[:1].lower() + t[1:]


def _slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9]+", "-", t).strip("-").lower()
    return (t or "anteproyecto")[:50]
