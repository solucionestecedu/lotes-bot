"""Modelos de lenguaje intercambiables.

- GeminiLLM: el real (chat con herramientas + lectura de planos con visión).
- ReglasLLM: sin IA, por palabras clave. Sirve para probar el flujo sin llave de Gemini
  (no puede leer fotos; pide los rumbos escritos).
- FakeLLM (en tests): respuestas guionizadas.

Formato neutro de mensajes que maneja el agente:
  {"rol": "usuario", "texto": str}
  {"rol": "modelo", "texto": str}
  {"rol": "modelo", "llamadas": [{"nombre", "args"}], "crudo": <objeto del proveedor>}
  {"rol": "herramienta", "nombre": str, "resultado": dict}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class Respuesta:
    texto: str = ""
    llamadas: list[dict] = field(default_factory=list)
    crudo: object = None


class LLM:
    puede_ver = False

    def turno(self, sistema: str, mensajes: list[dict], herramientas: list[dict]) -> Respuesta:
        raise NotImplementedError

    def leer_imagen(self, datos: bytes, mime: str, instruccion: str, esquema: dict) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------- Gemini
class GeminiLLM(LLM):
    puede_ver = True

    def __init__(self, api_key: str, modelo_chat: str, modelo_vision: str):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.modelo_chat = modelo_chat
        self.modelo_vision = modelo_vision

    def _contenidos(self, mensajes):
        from google.genai import types

        out = []
        for m in mensajes:
            if m["rol"] == "usuario":
                out.append(types.Content(role="user", parts=[types.Part.from_text(text=m["texto"])]))
            elif m["rol"] == "modelo" and m.get("crudo") is not None:
                out.append(m["crudo"])
            elif m["rol"] == "modelo" and m.get("llamadas"):
                out.append(types.Content(role="model", parts=[types.Part.from_function_call(name=c["nombre"], args=c["args"])
                                                              for c in m["llamadas"]]))
            elif m["rol"] == "modelo":
                out.append(types.Content(role="model", parts=[types.Part.from_text(text=m.get("texto") or "…")]))
            elif m["rol"] == "herramienta":
                parte = types.Part.from_function_response(name=m["nombre"], response={"resultado": m["resultado"]})
                # Las respuestas de varias herramientas del mismo paso van juntas en un solo mensaje.
                if out and out[-1].role == "user" and out[-1].parts and out[-1].parts[0].function_response:
                    out[-1].parts.append(parte)
                else:
                    out.append(types.Content(role="user", parts=[parte]))
        return out

    def turno(self, sistema, mensajes, herramientas):
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=sistema,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=h["nombre"], description=h["descripcion"],
                                          parameters_json_schema=h["parametros"] if h["parametros"].get("properties") else None)
                for h in herramientas])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.3,
        )
        r = self.client.models.generate_content(model=self.modelo_chat, contents=self._contenidos(mensajes), config=config)
        llamadas = [{"nombre": fc.name, "args": dict(fc.args or {})} for fc in (r.function_calls or [])]
        crudo = r.candidates[0].content if r.candidates and r.candidates[0].content else None
        texto = ""
        if not llamadas and crudo is not None:
            texto = "".join(p.text for p in (crudo.parts or []) if getattr(p, "text", None) and not getattr(p, "thought", False))
        return Respuesta(texto=texto.strip(), llamadas=llamadas, crudo=crudo if llamadas else None)

    def leer_imagen(self, datos, mime, instruccion, esquema):
        from google.genai import types

        r = self.client.models.generate_content(
            model=self.modelo_vision,
            contents=[types.Part.from_bytes(data=datos, mime_type=mime), instruccion],
            config=types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=esquema,
                                               temperature=0.0),
        )
        return json.loads(r.text)


# ---------------------------------------------------------------------------- sin IA
_PATRON_RUMBO = re.compile(r"^\s*[NS]\s*\d", re.I | re.M)


class ReglasLLM(LLM):
    """Agente mínimo sin IA: reconoce rumbos pegados, ubicaciones y palabras clave."""

    puede_ver = False

    def turno(self, sistema, mensajes, herramientas):
        ultimo = mensajes[-1]
        if ultimo["rol"] == "herramienta":
            return Respuesta(texto=self._resumen(ultimo["nombre"], ultimo["resultado"]))
        t = ultimo["texto"]
        low = t.lower()
        m = re.search(r"\[ubicación compartida: ([-\d.]+), ([-\d.]+)\]", t)
        if m:
            return Respuesta(llamadas=[{"nombre": "ubicar_terreno", "args": {"lat": float(m.group(1)), "lng": float(m.group(2)),
                                                                              "referencia": "centro"}}])
        if "[adjunto:" in low:
            return Respuesta(texto="Recibí el archivo, pero en modo sin IA no puedo leer fotos ni PDF. Pégame los rumbos "
                                   "y distancias como texto, uno por línea (ejemplo: S 82°40'17\" E 82.87).")
        if len(_PATRON_RUMBO.findall(t)) >= 3:
            xy = re.search(r"(?:inicio|x)\s*[:=]?\s*([\d,]+\.?\d*)\s*[, ]\s*(?:y\s*[:=]?\s*)?([\d,]+\.?\d*)", low)
            args = {"texto": t}
            if xy:
                args.update(x_inicio=float(xy.group(1).replace(",", "")), y_inicio=float(xy.group(2).replace(",", "")))
            return Respuesta(llamadas=[{"nombre": "cargar_rumbos", "args": args}])
        if "relieve" in low or "curvas" in low:
            return Respuesta(llamadas=[{"nombre": "obtener_relieve", "args": {}}])
        if "lote" in low or "prediseñ" in low or "predisen" in low:
            return Respuesta(llamadas=[{"nombre": "predisenar", "args": {}}])
        if "lámina" in low or "lamina" in low or "pdf" in low or "entregable" in low:
            return Respuesta(llamadas=[{"nombre": "generar_entregables", "args": {}}])
        return Respuesta(texto="Hola. Mándame los rumbos y distancias del plano (uno por línea) y, si los tienes, la "
                               "coordenada del primer vértice. Luego te pido la ubicación, saco el relieve, prediseño "
                               "los lotes y te armo la lámina en PDF.")

    @staticmethod
    def _resumen(nombre, r):
        if not r.get("ok", True):
            return "No se pudo: " + r.get("error", "error desconocido")
        if nombre in ("cargar_rumbos", "leer_plano", "corregir_tramo"):
            s = r["resumen"]
            txt = f"Terreno cargado: {s['vertices']} vértices, área {s['area_m2']:,} m² ({s['area_v2']:,} v²), cierre {s['error_cierre_m']} m."
            if r.get("problemas"):
                txt += "\n⚠️ " + "\n⚠️ ".join(r["problemas"])
            if not s["ubicado"]:
                txt += "\nFalta ubicarlo: compárteme la ubicación del terreno o la coordenada del vértice 1."
            return txt
        if nombre == "ubicar_terreno":
            return "Terreno ubicado. Escribe *relieve* para sacar las curvas de nivel."
        if nombre == "obtener_relieve":
            e = r["estadisticas"]
            return f"Relieve listo: desnivel {e['desnivel']} m, pendiente media {e['pendiente_media']} %. Escribe *lotes* para prediseñar."
        if nombre == "predisenar":
            return f"Prediseño: {r['lotes']} lotes. Escribe *lámina* para recibir el PDF, KML y DXF."
        if nombre == "generar_entregables":
            return "Listo, ahí van los archivos."
        return json.dumps(r, ensure_ascii=False)[:500]


def crear(config) -> LLM:
    if config.GEMINI_API_KEY:
        return GeminiLLM(config.GEMINI_API_KEY, config.MODELO_CHAT, config.MODELO_VISION)
    return ReglasLLM()
