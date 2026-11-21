"""¿Qué tan bien lee Gemini los cuadros de rumbos? Compara su lectura contra una transcripción verificada.

Uso:
    GEMINI_API_KEY=... python evaluar_lectura.py foto_del_cuadro.jpg rumbos_correctos.txt

rumbos_correctos.txt: un rumbo y distancia por línea, en el mismo orden del plano (como tests/datos/*.txt).
Muestra tramo por tramo qué leyó bien y qué no, y si el cierre matemático habría atrapado el error.
"""
import mimetypes
import sys

import config
from agente import ESQUEMA_PLANO, INSTRUCCION_PLANO
from motor.rumbos import leer_rumbo, leer_tramos
from motor.terreno import desde_tramos
from servicios.llm import GeminiLLM


def main(imagen: str, esperado: str) -> int:
    if not config.GEMINI_API_KEY:
        print("Falta GEMINI_API_KEY")
        return 2
    with open(esperado, encoding="utf-8") as f:
        correctos, _ = leer_tramos(f.read())
    with open(imagen, "rb") as f:
        datos = f.read()
    mime = mimetypes.guess_type(imagen)[0] or "image/jpeg"
    llm = GeminiLLM(config.GEMINI_API_KEY, config.MODELO_CHAT, config.MODELO_VISION)
    lectura = llm.leer_imagen(datos, mime, INSTRUCCION_PLANO, ESQUEMA_PLANO)
    leidos = []
    for t in lectura.get("tramos") or []:
        try:
            leidos.append(leer_rumbo(f"{t['rumbo']} {t['distancia']}"))
        except Exception as e:  # noqa: BLE001
            leidos.append({"error": str(e), "texto": f"{t.get('rumbo')} {t.get('distancia')}"})
    print(f"Tramos esperados: {len(correctos)}   leídos: {len(leidos)}\n")
    bien = 0
    for i in range(max(len(correctos), len(leidos))):
        c = correctos[i] if i < len(correctos) else None
        l = leidos[i] if i < len(leidos) else None
        if c and l and "error" not in l and abs(c["az"] - l["az"]) < 1 / 3600 * 0.6 and abs(c["dist"] - l["dist"]) < 0.0051:
            bien += 1
            marca = "OK "
        else:
            marca = "MAL"
        ct = f"{c['rumbo']} {c['dist']:.2f}" if c else "—"
        lt = (l.get("texto") or f"{l['rumbo']} {l['dist']:.2f}") if l else "—"
        print(f"{marca} {i + 1:>2}. esperado {ct:<28} leído {lt}")
    print(f"\nAciertos: {bien}/{len(correctos)}")
    validos = [x for x in leidos if "error" not in x]
    if len(validos) >= 3:
        t = desde_tramos(validos)
        print(f"Cierre con la lectura: {t.cierre:.3f} m → {'cierra' if t.cierra_bien() else 'NO cierra: el bot habría pedido revisar'}")
        for s in t.sospechosos():
            print(f"   sospechoso: tramo {s['tramo']} ({s['distancia_leida']} → {s['distancia_que_cerraria']:.2f})")
    if lectura.get("dudas"):
        print("Dudas que reportó el modelo:", lectura["dudas"])
    return 0 if bien == len(correctos) else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
