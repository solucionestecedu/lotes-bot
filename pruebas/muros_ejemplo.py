"""Terrazas, muros, revisión y alternativas del ejemplo Quintas A+B con el relieve real (sin pintar nada).
En Cloud Shell: bash muros_ejemplo.sh"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrega_ejemplo import preparar  # noqa: E402
from motor import altimetria, revisor  # noqa: E402


def main():
    ag, s = preparar(None)
    t, m = s.get_terreno(), s.get_malla()
    lot, filas = ag._lotificar(s)
    terr = altimetria.terrazas(lot, m)
    print("\nLOTE  CASA   CALLE  TERR  MURO m2  MAX   entre lotes / calle / lindero / entre terrazas")
    for p in terr:
        tipos = {}
        for mu in p["muros"]:
            tipos.setdefault(mu["tipo"], []).append(mu["h"])
        mx = {k: round(max(v), 1) for k, v in tipos.items()}
        print(f"{p['id']:>4} {p['z']:7.1f} {p['z_calle']:7.1f} {len(p['terrazas']):4} {p['muro_m2']:8} {p['muro_max']:5.1f}   "
              f"{mx.get('entre lotes', 0)} / {mx.get('calle', 0)} / {mx.get('lindero', 0)} / {mx.get('entre terrazas', 0)}")
    c = altimetria.comparar_terraceria(lot, m)
    print("\ntaludes:", json.dumps(c["taludes"], ensure_ascii=False))
    print("muros:  ", json.dumps(c["muros"], ensure_ascii=False))
    print("\nrevisión:")
    for ch in revisor.revisar(t, lot, m, s.ficha, terr):
        print(" ", "OK   " if ch["ok"] else ("GRAVE" if ch["grave"] else "aviso"), ch["detalle"])
    print("\nnormas:", [(f["regla"], f["proyecto"], f["estado"]) for f in filas if f["estado"] != "cumple"])
    r = ag.h_proponer_alternativas(s, salidas=[])
    print("\nalternativas:")
    for o in r.get("opciones", []):
        md = o["medidas"]
        print(f"  {o['opcion']}. {o['titulo']}: {md['lotes']} lotes, {md['area_vendible_m2']} m2, muro máx "
              f"{md.get('muro_max_m')} m, graves {len(md.get('fallas_graves') or [])}, no cumple {md.get('no_cumple')}")


if __name__ == "__main__":
    main()
