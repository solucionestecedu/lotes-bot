"""Lo que dice el geovisor de la OPAMSS de los terrenos de prueba (servidor real): python pruebas/opamss_ejemplo.py"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motor.terreno import desde_texto_rumbos  # noqa: E402
from servicios import opamss  # noqa: E402

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "datos")
txt = lambda f: open(os.path.join(D, f), encoding="utf-8").read()  # noqa: E731

terrenos = {
    "Lote A (M1 E 473280.676 N 289049.929)": desde_texto_rumbos(txt("plano_a.txt"), inicio=(473280.676, 289049.929))[0],
    "Chiltiupan (Mo. 1)": desde_texto_rumbos(txt("chiltiupan_rumbos.txt"),
                                             inicio=tuple(json.loads(txt("chiltiupan_inicio.json"))["mo1"]))[0],
}
for nombre, t in terrenos.items():
    t0 = time.time()
    op = opamss.consultar(t.puntos)
    print(f"\n=== {nombre} · {t.area:,.0f} m2 · {time.time() - t0:.1f} s")
    print(json.dumps(op, ensure_ascii=False, indent=1) if op else "SIN RESPUESTA")
    if os.environ.get("CRUDO"):  # atributos tal como llegan de la capa de uso de suelo
        import requests
        from shapely.geometry import Polygon
        poli = {"rings": [opamss._a_wgs(list(Polygon(t.puntos).exterior.coords)[:-1])], "spatialReference": {"wkid": 4326}}
        d = opamss._consulta(requests.Session(), opamss.USO_SUELO, poli, "esriGeometryPolygon", "*")
        for f in d.get("features", [])[:3]:
            print("CRUDO", {k: v for k, v in f["attributes"].items() if not str(k).startswith("st_")})
