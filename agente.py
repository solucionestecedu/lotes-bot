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
from motor import exportar, lamina, proyeccion, relieve
from motor.lotificacion import ErrorLotificacion, Parametros, lotificar
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
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def get_terreno(self) -> Terreno | None:
        return Terreno.de_dict(self.terreno) if self.terreno else None

    def set_terreno(self, t: Terreno | None):
        self.terreno = t.a_dict() if t else None
        self.malla = None
        self.lotes_param = None
        self.lotes_stats = None

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
    {"nombre": "leer_plano", "descripcion": "Lee la foto o PDF de un plano topográfico (cuadro de rumbos y distancias y/o "
     "coordenadas) y verifica matemáticamente que la poligonal cierre. Úsala cuando el usuario mande un adjunto.",
     "parametros": _obj({"adjunto": {"type": "string", "description": "id del adjunto, por ejemplo 'a1'"}}, ["adjunto"])},
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
    {"nombre": "obtener_relieve", "descripcion": "Descarga alturas satelitales del terreno ubicado y calcula desnivel, "
     "pendientes y curvas de nivel (aproximadas).", "parametros": _obj({})},
    {"nombre": "predisenar", "descripcion": "Prediseña la lotificación automáticamente. Todos los parámetros son opcionales.",
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
     "lotificación), el KML para Google Earth y el DXF para AutoCAD/Civil 3D, y se los envía al usuario.",
     "parametros": _obj({"nombre_proyecto": {"type": "string"}, "ubicacion": {"type": "string",
                         "description": "municipio y departamento"}, "propietario": {"type": "string"}})},
]

ESQUEMA_PLANO = {
    "type": "object",
    "properties": {
        "es_plano": {"type": "boolean", "description": "true si la imagen contiene un cuadro de rumbos/distancias o de coordenadas"},
        "tramos": {"type": "array", "description": "cuadro de rumbos y distancias, en orden, tal como está escrito", "items": {
            "type": "object", "properties": {
                "de": {"type": "string"}, "a": {"type": "string"},
                "rumbo": {"type": "string", "description": "copiado exacto, ej. S 82°40'17.03\" E (O = oeste)"},
                "distancia": {"type": "number"}},
            "required": ["rumbo", "distancia"]}},
        "coordenadas": {"type": "array", "description": "vértices con coordenadas X (Este) y Y (Norte) si el plano las trae", "items": {
            "type": "object", "properties": {"punto": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"]}},
        "area_declarada": {"type": "number", "description": "área total escrita en el plano"},
        "unidad_area": {"type": "string", "enum": ["m2", "v2", "ha"]},
        "sistema": {"type": "string", "enum": ["SIRGAS", "NAD27", "desconocido"],
                    "description": "SIRGAS-ES2007 o NAD27/Ocotepeque si el plano lo dice"},
        "dudas": {"type": "string", "description": "números que no se leen bien, en una frase"},
    },
    "required": ["es_plano", "tramos", "coordenadas"],
}

INSTRUCCION_PLANO = (
    "Eres un dibujante topográfico de El Salvador. Transcribe EXACTAMENTE el cuadro de rumbos y distancias de este plano "
    "(estación, punto visado, rumbo con grados, minutos y segundos con sus decimales, y distancia en metros), en el orden en "
    "que aparece, y la tabla de coordenadas si existe. En los rumbos escribe E para este y O para oeste. No calcules, no "
    "redondees, no inventes: si un dígito no se lee, pon tu mejor lectura y menciónalo en 'dudas'. No transcribas nombres "
    "de personas ni números de documentos de identidad."
)

SISTEMA = """Eres Lotes-Bot, asistente de anteproyectos de lotificación en El Salvador, por WhatsApp.
Hablas en español sencillo y cercano, con mensajes cortos (WhatsApp; usa *negritas* con asteriscos, sin tablas ni #).

QUÉ HACES (en este orden):
1. Recibir el plano del topógrafo (foto o PDF) → leer_plano. Si lo escriben a mano → cargar_rumbos o cargar_coordenadas.
2. Verificar: si la poligonal no cierra o el área no coincide, NO avances. Explica en simple, muestra el o los tramos
   sospechosos que te da la herramienta y pide que confirmen ese dato viendo el plano. Corrige con corregir_tramo.
3. Si el terreno no está ubicado, pide la ubicación (que compartan su ubicación de WhatsApp parados en el terreno, o la
   coordenada del vértice 1) → ubicar_terreno.
4. obtener_relieve.
5. predisenar. Usa los valores por defecto salvo que el usuario diga otros (calle 8 m, lote 10×25 m, mínimo 200 m²,
   área verde 10 % del área útil). Si el terreno está fuera de San Salvador, la ley pide 12.5 m² de área verde por lote:
   pregunta si el terreno está dentro o fuera del área urbana cuando sea relevante.
6. generar_entregables: lámina PDF lista para imprimir + KML + DXF. Pide nombre del proyecto y municipio si no los tienes.

REGLAS:
- Nunca inventes ni calcules números tú: todos salen de las herramientas. Si no hay dato, dilo.
- Siempre que entregues resultados, recuerda en una línea que es un *anteproyecto*: no sirve para trámite; para la DOT,
  OPAMSS o CNR se necesita levantamiento topográfico y firma de un profesional autorizado.
- No pidas ni repitas DUI, NIT ni datos personales que aparezcan en los planos.
- Las curvas de nivel salen de datos satelitales (error de metros): son orientativas.
- Si te preguntan algo fuera de esto, responde breve y vuelve al proyecto.
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

    def h_leer_plano(self, s: Sesion, adjunto: str, salidas=None) -> dict:
        a = s.adjuntos.get(adjunto)
        if a is None:
            return {"ok": False, "error": f"no existe el adjunto {adjunto}"}
        if not self.llm.puede_ver:
            return {"ok": False, "error": "este modo no puede leer imágenes; pide los rumbos escritos"}
        datos = self.llm.leer_imagen(a["datos"], a["mime"], INSTRUCCION_PLANO, ESQUEMA_PLANO)
        return self._desde_lectura(s, datos, origen=a.get("nombre") or adjunto)

    def _desde_lectura(self, s: Sesion, d: dict, origen: str = "") -> dict:
        if not d.get("es_plano") and not d.get("tramos") and not d.get("coordenadas"):
            return {"ok": False, "error": "no encontré un cuadro de rumbos ni de coordenadas en el archivo"}
        sistema = d.get("sistema") if d.get("sistema") in ("SIRGAS", "NAD27") else "SIRGAS"
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
            return {"ok": False, "error": "leí menos de 3 tramos o coordenadas; manda una foto más cercana y derecha del cuadro",
                    "errores_lectura": errores}
        s.conocidas = {k: list(v) for k, v in conocidas.items()}
        extra = {"sistema_detectado": d.get("sistema") or "desconocido"}
        if errores:
            extra["errores_lectura"] = errores
        if fuera:
            extra["coordenadas_fuera_de_el_salvador"] = fuera
        if d.get("dudas"):
            extra["dudas_de_lectura"] = d["dudas"]
        if sistema == "NAD27":
            extra["nota_sistema"] = "Coordenadas NAD27 convertidas a SIRGAS-ES2007."
        return self._informe_terreno(s, t, extra)

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

    def _lotificar(self, s: Sesion):
        t = s.get_terreno()
        return lotificar(t.puntos, Parametros.de_dict(s.lotes_param), s.get_malla())

    def h_predisenar(self, s: Sesion, salidas=None, **params) -> dict:
        t = s.get_terreno()
        if t is None:
            return {"ok": False, "error": "primero hay que cargar el terreno"}
        validos = {k: v for k, v in params.items() if k in Parametros.__dataclass_fields__ and v is not None}
        base = dict(s.lotes_param or {})
        base.update(validos)
        s.lotes_param = base
        r = self._lotificar(s)
        st = r.stats
        s.lotes_stats = {k: _r(v) for k, v in st.items()}
        inclinados = sum(1 for l in r.lotes if l["pendiente"] is not None and l["pendiente"] > 30)
        out = {"ok": True, "lotes": st["lotes"], "area_util_m2": _r(st["area_util"]),
               "area_calles_m2": _r(st["area_circulacion"]), "area_verde_m2": _r(st["area_verde"]),
               "area_verde_requerida_m2": _r(st["area_verde_requerida"]),
               "lotes_pasados_a_area_verde": st["lotes_convertidos_a_verde"],
               "lote_promedio_m2": _r(st["lote_promedio"]), "lote_min_m2": _r(st["lote_min"]), "lote_max_m2": _r(st["lote_max"]),
               "aprovechamiento_pct": _r(st["area_util"] / st["area_total"] * 100, 1),
               "parametros": r.parametros, "orientacion_calles_grados": r.angulo}
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
        lot = self._lotificar(s) if s.lotes_param is not None else None
        carpeta = os.path.join(self.dir, uuid.uuid4().hex)
        os.makedirs(carpeta, exist_ok=True)
        base = _slug(nombre)
        pdf = os.path.join(carpeta, f"{base}.pdf")
        info = lamina.generar(pdf, t, m, lot, datos)
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
        archivos.append((dxf, "Para AutoCAD / Civil 3D (coordenadas Lambert SIRGAS)"))
        for ruta, titulo in archivos:
            salidas.append({"tipo": "archivo", "ruta": ruta, "nombre": os.path.basename(ruta), "titulo": titulo})
        return {"ok": True, "archivos_enviados": [os.path.basename(a) for a, _ in archivos], "escala": f"1:{info['escala']:,}",
                "hojas": info["hojas"], "nota": "Los archivos se envían automáticamente después de tu mensaje."}


def _slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9]+", "-", t).strip("-").lower()
    return (t or "anteproyecto")[:50]
