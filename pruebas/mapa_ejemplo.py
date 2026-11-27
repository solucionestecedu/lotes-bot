"""Lo que dice OpenStreetMap alrededor de los terrenos de prueba, con el paquete real armado con
herramientas/construir_osm.py: OSM_DATOS=/tmp/osm_sv.npz python pruebas/mapa_ejemplo.py"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motor.terreno import desde_texto_rumbos  # noqa: E402
from servicios import osm  # noqa: E402

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "datos")
txt = lambda f: open(os.path.join(D, f), encoding="utf-8").read()  # noqa: E731

terrenos = {
    "Lote A": desde_texto_rumbos(txt("plano_a.txt"), inicio=(473280.676, 289049.929))[0],
    "Chiltiupan": desde_texto_rumbos(txt("chiltiupan_rumbos.txt"),
                                     inicio=tuple(json.loads(txt("chiltiupan_inicio.json"))["mo1"]))[0],
}
for nombre, t in terrenos.items():
    t0 = time.time()
    r = osm.consultar(t.puntos)
    print(f"\n=== {nombre} · {time.time() - t0:.2f} s")
    print(json.dumps(osm.resumen(r), ensure_ascii=False, indent=1) if r else "SIN PAQUETE")
    if r and os.environ.get("CRUDO"):
        print(json.dumps({k: r[k] for k in ("calles", "senderos", "aguas")}, ensure_ascii=False, indent=1))
