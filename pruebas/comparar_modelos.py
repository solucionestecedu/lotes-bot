"""Compara modelos de Gemini en las dos tareas del bot: leer el cuadro de rumbos y llevar la conversación.

Uso (la llave se toma de la variable GEMINI_API_KEY; nunca se imprime):
    python pruebas/comparar_modelos.py --img-a foto_lote_a.jpg --img-b foto_lote_b.jpg \
        --modelos gemini-4-carbon gemini-3.8-flash

Prueba 1 · Lectura: cada modelo lee las dos fotos y se compara tramo por tramo contra la transcripción verificada
(tests/datos/plano_a.txt y plano_b.txt).
Prueba 2 · Conversación: el mismo guion de 6 mensajes para cada modelo, con el agente completo
(fotos → unir → ubicación → ficha → prediseño → planos).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from agente import ESQUEMA_PLANO, INSTRUCCION_PLANO, Agente, Sesion  # noqa: E402
from motor.rumbos import leer_rumbo, leer_tramos  # noqa: E402
from servicios.llm import GeminiLLM  # noqa: E402

# (adjunto, texto, ubicación). Lo de quintas, la calle en medio y la quebrada al fondo es lo que dijo el usuario;
# el lado de la quebrada (sur), su profundidad, el acceso (norte), la zona y los servicios son supuestos de prueba.
# La ubicación es la cuadrícula X 473300 / Y 288950 que trae el plano A.
PIN = (13.72121, -89.24686)
GUION = [
    ("a", "Hola, este es el cuadro de rumbos del lote A.", None),
    ("b", "Y este es el lote B, colinda con el A.", None),
    (None, "Quiero unirlos, idealmente con una calle en medio de los dos terrenos. Haré 4 o 5 lotes por área, como "
           "quintas. Al final el área protegida de la quebrada. Te mando la ubicación.", PIN),
    (None, "La quebrada queda al sur, tiene unos 8 m de profundidad. La calle pública llega por el norte. Es zona urbana, "
           "fuera del AMSS. Va cerrado con portón, se vende como lotificación, hay agua de ANDA y luz.", None),
    (None, "Dale, hazme el prediseño.", None),
    (None, "Mándame los planos.", None),
]


class Contador:
    """Envuelve el cliente de Gemini para contar llamadas, tokens y tiempo."""

    def __init__(self, llm: GeminiLLM):
        self.llamadas = 0
        self.tokens = {"entrada": 0, "salida": 0, "razonamiento": 0}
        self.segundos = 0.0
        self.errores: list[str] = []
        original = llm.client.models.generate_content

        def envuelto(**kw):
            t0 = time.time()
            try:
                r = original(**kw)
            except Exception as e:  # noqa: BLE001
                self.errores.append(f"{type(e).__name__}: {str(e)[:160]}")
                raise
            finally:
                self.segundos += time.time() - t0
            self.llamadas += 1
            u = getattr(r, "usage_metadata", None)
            if u:
                self.tokens["entrada"] += u.prompt_token_count or 0
                self.tokens["salida"] += u.candidates_token_count or 0
                self.tokens["razonamiento"] += getattr(u, "thoughts_token_count", 0) or 0
            return r

        llm.client.models.generate_content = envuelto


def comparar_tramos(esperados, leidos):
    bien = 0
    malos = []
    for i in range(max(len(esperados), len(leidos))):
        e = esperados[i] if i < len(esperados) else None
        l = leidos[i] if i < len(leidos) else None
        ok = (e and l and "error" not in l and abs(e["az"] - l["az"]) < 0.6 / 3600 and abs(e["dist"] - l["dist"]) < 0.0051)
        if ok:
            bien += 1
        else:
            malos.append(f"{i + 1}: esperado {e['rumbo'] + ' ' + str(e['dist']) if e else '—'} / leído "
                         f"{(l.get('texto') or l['rumbo'] + ' ' + str(l['dist'])) if l else '—'}")
    return bien, malos


def prueba_lectura(modelo: str, imagenes: dict, esperados: dict) -> dict:
    llm = GeminiLLM(os.environ["GEMINI_API_KEY"], modelo, modelo)
    c = Contador(llm)
    out = {}
    for k in ("a", "b"):
        with open(imagenes[k], "rb") as f:
            datos = f.read()
        t0 = time.time()
        try:
            d = llm.leer_imagen(datos, "image/jpeg", INSTRUCCION_PLANO, ESQUEMA_PLANO)
        except Exception as e:  # noqa: BLE001
            out[k] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        terrenos = d.get("terrenos") or []
        tramos = terrenos[0].get("tramos", []) if terrenos else []
        leidos = []
        for t in tramos:
            try:
                leidos.append(leer_rumbo(f"{t['rumbo']} {t['distancia']}"))
            except Exception as e:  # noqa: BLE001
                leidos.append({"error": str(e), "texto": f"{t.get('rumbo')} {t.get('distancia')}"})
        bien, malos = comparar_tramos(esperados[k], leidos)
        out[k] = {"terrenos_detectados": len(terrenos), "tramos_leidos": len(tramos), "aciertos": bien,
                  "total": len(esperados[k]), "errores": malos, "segundos": round(time.time() - t0, 1),
                  "area_declarada_leida": terrenos[0].get("area_declarada") if terrenos else None}
    out["uso"] = {"llamadas": c.llamadas, "tokens": c.tokens}
    out["errores_modelo"] = c.errores
    return out


def prueba_conversacion(modelo: str, imagenes: dict) -> dict:
    llm = GeminiLLM(os.environ["GEMINI_API_KEY"], modelo, modelo)
    c = Contador(llm)
    ag = Agente(llm, tempfile.mkdtemp())
    usadas = []
    original = ag.ejecutar

    def registrar(s, nombre, args, salidas):
        r = original(s, nombre, args, salidas)
        usadas.append({"herramienta": nombre, "args": {k: v for k, v in args.items() if k != "texto"},
                       "ok": r.get("ok", True), "error": r.get("error")})
        return r

    ag.ejecutar = registrar
    s = Sesion("comparacion")
    turnos = []
    for adj, texto, pin in GUION:
        adjuntos = []
        if adj:
            with open(imagenes[adj], "rb") as f:
                adjuntos.append({"mime": "image/jpeg", "nombre": f"lote-{adj}.jpg", "datos": f.read()})
        n0 = len(usadas)
        t0 = time.time()
        try:
            salidas = ag.atender(s, texto, adjuntos, pin)
            respuesta = salidas[0]["texto"]
            archivos = [x["nombre"] for x in salidas if x.get("tipo") == "archivo"]
        except Exception as e:  # noqa: BLE001
            respuesta = f"[ERROR {type(e).__name__}: {str(e)[:200]}]"
            archivos = []
        print(f"    ({len(turnos) + 1}/{len(GUION)} listo en {round(time.time() - t0, 1)} s)", flush=True)
        turnos.append({"usuario": texto, "herramientas": usadas[n0:], "respuesta": respuesta, "archivos": archivos,
                       "preguntas": respuesta.count("?"), "segundos": round(time.time() - t0, 1)})
    t = s.get_terreno()
    st = s.lotes_stats or {}
    final = {"terreno_m2": round(t.area, 2) if t else None, "vertices": len(t.puntos) if t else None,
             "lotes": st.get("lotes"), "modo": st.get("modo", "automatica"),
             "por_lado": [(pl.get("terreno"), pl.get("lotes"), pl.get("areas_m2")) for pl in st.get("por_lado", [])],
             "ficha": {k: v for k, v in s.ficha.items() if k != "del_plano"},
             "cumplimiento": [(f["regla"], f["estado"]) for f in s.cumplimiento]}
    return {"turnos": turnos, "final": final, "uso": {"llamadas": c.llamadas, "tokens": c.tokens,
                                                        "segundos_modelo": round(c.segundos, 1)},
            "errores_modelo": c.errores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-a", required=True)
    ap.add_argument("--img-b", required=True)
    ap.add_argument("--modelos", nargs="+", default=["gemini-4-carbon", "gemini-3.8-flash"])
    ap.add_argument("--salida", default="resultado_comparacion.json")
    a = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Falta GEMINI_API_KEY")
    esperados = {}
    for k, archivo in (("a", "plano_a.txt"), ("b", "plano_b.txt")):
        with open(os.path.join(RAIZ, "tests", "datos", archivo), encoding="utf-8") as f:
            esperados[k], _ = leer_tramos(f.read())
    imagenes = {"a": a.img_a, "b": a.img_b}
    resultado = {}
    for m in a.modelos:
        print(f"\n=== {m} ===", flush=True)
        print("Prueba 1: lectura de las dos fotos...", flush=True)
        lec = prueba_lectura(m, imagenes, esperados)
        for k in ("a", "b"):
            r = lec.get(k, {})
            if "error" in r:
                print(f"  lote {k.upper()}: ERROR {r['error']}")
            else:
                print(f"  lote {k.upper()}: {r['aciertos']}/{r['total']} tramos bien, {r['segundos']} s"
                      + (f" · errores: {r['errores'][:3]}" if r["errores"] else ""))
        print(f"Prueba 2: conversación de {len(GUION)} mensajes...", flush=True)
        conv = prueba_conversacion(m, imagenes)
        for i, tu in enumerate(conv["turnos"], 1):
            hs = ", ".join(f"{h['herramienta']}{'' if h['ok'] else ' (falló)'}" for h in tu["herramientas"]) or "ninguna"
            print(f"  {i}. {tu['segundos']} s · herramientas: {hs} · preguntas: {tu['preguntas']}"
                  + (f" · archivos: {len(tu['archivos'])}" if tu["archivos"] else ""))
            print("     " + tu["respuesta"][:350].replace("\n", " ") + ("…" if len(tu["respuesta"]) > 350 else ""))
        print(f"  Final: {json.dumps(conv['final'], ensure_ascii=False)[:600]}")
        print(f"  Uso: lectura {lec['uso']} · conversación {conv['uso']}")
        errs = lec["errores_modelo"] + conv["errores_modelo"]
        if errs:
            print(f"  Errores del modelo ({len(errs)}): {errs[:3]}")
        resultado[m] = {"lectura": lec, "conversacion": conv}
    with open(a.salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=1)
    print(f"\nDetalle completo en {a.salida}")


if __name__ == "__main__":
    main()
