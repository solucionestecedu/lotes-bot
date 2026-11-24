"""Prueba real de la imagen de presentación con Nano Banana (corre en Cloud Shell, la llave viene de GEMINI_API_KEY).

    python pruebas/probar_presentacion.py [modelo_imagen ...]

Arma las quintas A+B (calle en medio, quebrada al fondo), baja el relieve real, pinta planta y aérea con cada
modelo y deja todo en /tmp/presentacion/<modelo>/ (calco, lo que devolvió el modelo y la imagen final).
"""
import json
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tests"))

from agente import Agente, Sesion  # noqa: E402
from motor import presentacion  # noqa: E402
from servicios.llm import GeminiLLM  # noqa: E402
from test_calle_central import GuionQuintas  # noqa: E402

PIN = (13.72121, -89.24686)


def main():
    modelos = sys.argv[1:] or ["gemini-nano-banana-2.1"]
    ag = Agente(GuionQuintas(), "/tmp/presentacion/_agente")
    s = Sesion("prueba")
    ag.atender(s, "plano", [{"mime": "image/jpeg", "nombre": "p.jpg", "datos": b"x"}])
    ag.atender(s, "unir")
    print("ubicar:", ag.h_ubicar_terreno(s, lat=PIN[0], lng=PIN[1]).get("ok"), flush=True)
    r = ag.h_obtener_relieve(s)
    print("relieve:", r.get("ok"), r.get("fuente"), r.get("desnivel_m"), flush=True)
    ag.atender(s, "quintas")
    print("prediseño:", json.loads(ag.atender(s, "lotes")[0]["texto"])["lotes"], "lotes", flush=True)
    t, m = s.get_terreno(), s.get_malla()
    lot, _ = ag._lotificar(s)
    cx, cy = t.centroide()
    partes = [{"origen": p["origen"], "anillo": [(x + cx, y + cy) for x, y in p["anillo"]]} for p in s.partes]
    for modelo in modelos:
        carpeta = f"/tmp/presentacion/{modelo}"
        os.makedirs(carpeta, exist_ok=True)
        os.environ["PRESENTACION_CRUDO"] = carpeta
        llm = GeminiLLM(os.environ["GEMINI_API_KEY"], "gemini-3.8-flash", "gemini-3.8-flash", modelo_imagen=modelo)
        t0 = time.time()
        out = presentacion.generar(carpeta, "quintas", t, lot, m, s.ficha, "Quintas A y B", partes, llm.pintar)
        print(f"\n=== {modelo} ({time.time() - t0:.0f} s) ===")
        for o in out:
            print(f"  {o['vista']}: pintada con IA = {o['ia']} · coincidencia {o.get('coincidencia')} "
                  f"(corrida {o.get('coincidencia_corrida')}) {o.get('motivo', '')}")
    print("\nArchivos en /tmp/presentacion/")


if __name__ == "__main__":
    main()
