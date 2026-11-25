"""Entrega completa de ejemplo (Quintas A+B) tal como la manda el bot, con relieve real e imágenes de Nano Banana.

    python pruebas/entrega_ejemplo.py [modelo_imagen]

Corre en Cloud Shell (la llave viene de GEMINI_API_KEY). Deja los archivos ordenados en /tmp/entrega-quintas/:
1 Resumen ejecutivo/, 2 Carpeta tecnica/ y 3 Carpeta comercial/.
"""
import json
import os
import shutil
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tests"))

from agente import Agente, Sesion  # noqa: E402
from motor import proyeccion  # noqa: E402
from servicios.llm import GeminiLLM  # noqa: E402
from test_calle_central import GuionQuintas  # noqa: E402

PIN = (13.72121, -89.24686)
ESQUINA = (473280.676, 289049.929)
SALIDA = "/tmp/entrega-quintas"


def main():
    modelo = (sys.argv[1:] or ["gemini-nano-banana-2.1"])[0]
    ag = Agente(GuionQuintas(), "/tmp/entrega-quintas-_agente")
    ag.llm.pintar = GeminiLLM(os.environ["GEMINI_API_KEY"], "gemini-3.8-flash", "gemini-3.8-flash",
                              modelo_imagen=modelo).pintar
    s = Sesion("ejemplo")
    ag.atender(s, "plano", [{"mime": "image/jpeg", "nombre": "p.jpg", "datos": b"x"}])
    ag.atender(s, "unir")
    s.partes[1]["area_declarada"] = 6220.08  # lo que dice el plano del terreno B (su cuadro de rumbos no cuadra)
    # ubicación con las coordenadas del plano: la esquina de la calle donde termina el lindero común de A y B
    # (E 473,280.676 · N 289,049.929, Lambert SIRGAS); si no hay lindero, el pin aproximado
    t = s.get_terreno()
    if s.lindero:
        cx, cy = t.centroide()
        tope = max(s.lindero, key=lambda q: q[1])
        t.mover(ESQUINA[0] - (tope[0] + cx), ESQUINA[1] - (tope[1] + cy))
        t.ubicado = True
        s.set_terreno(t)
        print("ubicado con la esquina del plano:", [round(v, 6) for v in proyeccion.a_latlng(*t.centroide())], flush=True)
    else:
        print("ubicar:", ag.h_ubicar_terreno(s, lat=PIN[0], lng=PIN[1]).get("ok"), flush=True)
    r = ag.h_obtener_relieve(s)
    print("relieve:", r.get("ok"), json.dumps(r.get("estadisticas"), ensure_ascii=False), flush=True)
    ag.atender(s, "quintas")
    pre = json.loads(ag.atender(s, "lotes")[0]["texto"])
    print("prediseño:", pre["lotes"], "lotes ·", pre.get("barrancos_fuera_de_lotes", "sin caídas fuertes fuera"), flush=True)
    t0 = time.time()
    salidas = []
    out = ag.h_generar_entregables(s, nombre_proyecto="Quintas A y B",
                                   ubicacion="Cantón San Antonio Abad, San Salvador", salidas=salidas)
    print(f"entregables en {time.time() - t0:.0f} s:", json.dumps({k: out.get(k) for k in ("partes", "avisos", "presentacion")},
                                                              ensure_ascii=False), flush=True)
    shutil.rmtree(SALIDA, ignore_errors=True)
    carpetas = {"Resumen": "1 Resumen ejecutivo", "Carpeta técnica": "2 Carpeta tecnica", "Carpeta comercial": "3 Carpeta comercial"}
    for i, o in enumerate(salidas, 1):
        parte = next((v for k, v in carpetas.items() if o["titulo"].startswith(k)), "otros")
        os.makedirs(os.path.join(SALIDA, parte), exist_ok=True)
        shutil.copy(o["ruta"], os.path.join(SALIDA, parte, f"{i:02d}-{o['nombre']}"))
        print(f"  {parte} · {o['nombre']} · {o['titulo']}")


if __name__ == "__main__":
    main()
