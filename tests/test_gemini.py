"""El adaptador de Gemini sin red: se revisa que arme bien las peticiones y entienda las respuestas."""
import json

from google.genai import types

from agente import HERRAMIENTAS, ESQUEMA_PLANO
from servicios.llm import GeminiLLM


def _respuesta(*parts):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))])


class ModelsFalso:
    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.peticiones = []

    def generate_content(self, model, contents, config):
        self.peticiones.append({"model": model, "contents": contents, "config": config})
        return self.respuestas.pop(0)


def _llm(respuestas):
    g = GeminiLLM.__new__(GeminiLLM)
    g.modelo_chat, g.modelo_vision = "gemini-2.5-flash", "gemini-2.5-flash"
    g.client = type("C", (), {})()
    g.client.models = ModelsFalso(respuestas)
    return g


def test_llamada_a_herramienta_y_respuesta_final():
    g = _llm([
        _respuesta(types.Part.from_function_call(name="leer_plano", args={"adjunto": "a1"})),
        _respuesta(types.Part(text="pensando…", thought=True), types.Part.from_text(text="El plano cierra bien.")),
    ])
    r1 = g.turno("sistema", [{"rol": "usuario", "texto": "hola\n[adjunto: a1, plano.jpg]"}], HERRAMIENTAS)
    assert r1.llamadas == [{"nombre": "leer_plano", "args": {"adjunto": "a1"}}] and r1.crudo is not None
    pet = g.client.models.peticiones[0]
    nombres = [f.name for f in pet["config"].tools[0].function_declarations]
    assert "predisenar" in nombres and all(n.isascii() for n in nombres)
    assert pet["config"].automatic_function_calling.disable is True

    msgs = [{"rol": "usuario", "texto": "hola"}, {"rol": "modelo", "llamadas": r1.llamadas, "crudo": r1.crudo},
            {"rol": "herramienta", "nombre": "leer_plano", "resultado": {"ok": True, "area": 1.5}}]
    r2 = g.turno("sistema", msgs, HERRAMIENTAS)
    assert r2.texto == "El plano cierra bien." and not r2.llamadas
    contenidos = g.client.models.peticiones[1]["contents"]
    assert [c.role for c in contenidos] == ["user", "model", "user"]
    assert contenidos[2].parts[0].function_response.name == "leer_plano"


def test_varias_respuestas_de_herramienta_van_juntas():
    g = _llm([])
    c = g._contenidos([{"rol": "usuario", "texto": "x"},
                       {"rol": "modelo", "llamadas": [{"nombre": "a", "args": {}}, {"nombre": "b", "args": {}}]},
                       {"rol": "herramienta", "nombre": "a", "resultado": {}},
                       {"rol": "herramienta", "nombre": "b", "resultado": {}}])
    assert len(c) == 3 and len(c[2].parts) == 2


def test_lectura_de_plano_con_esquema_json():
    g = _llm([_respuesta(types.Part.from_text(text=json.dumps({"es_plano": True, "tramos": [], "coordenadas": []})))])
    d = g.leer_imagen(b"\x89PNG", "image/png", "Transcribe", ESQUEMA_PLANO)
    assert d["es_plano"] is True
    pet = g.client.models.peticiones[0]
    assert pet["config"].response_mime_type == "application/json" and pet["config"].response_json_schema == ESQUEMA_PLANO
    assert pet["contents"][0].inline_data.mime_type == "image/png"


def test_gemini_3_usa_temperatura_por_defecto():
    g = _llm([_respuesta(types.Part.from_text(text="hola"))])
    g.modelo_chat = "gemini-3.8-flash"
    g.turno("s", [{"rol": "usuario", "texto": "hola"}], HERRAMIENTAS)
    assert g.client.models.peticiones[0]["config"].temperature is None
