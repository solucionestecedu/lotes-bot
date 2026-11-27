"""Calles, ríos y construcciones de OpenStreetMap (paquete local armado con datos de prueba)."""
import json

import numpy as np
import pytest
from shapely.geometry import Polygon

from motor import proyeccion
from servicios import osm

pytestmark = pytest.mark.mapa

# terreno y vías inventados para la prueba (no son de ningún plano real)
X0, Y0 = 473200.0, 289000.0
ANILLO = [(X0, Y0), (X0 + 100, Y0), (X0 + 100, Y0 + 60), (X0, Y0 + 60)]
VIAS = [
    ("highway=residential", "Calle El Progreso", [(X0 - 50, Y0 + 65), (X0 + 150, Y0 + 65)]),  # 5 m al norte
    ("waterway=stream", None, [(X0 - 50, Y0 - 20), (X0 + 150, Y0 - 22)]),                     # 20 m al sur
    ("highway=footway", None, [(X0 + 200, Y0 - 50), (X0 + 200, Y0 + 100)]),                    # 100 m al oriente
    ("highway=primary", "Carretera X", [(X0 - 300, Y0), (X0 - 300, Y0 + 60)]),                # 300 m poniente
    ("highway=primary", "Lejos", [(X0 - 900, Y0), (X0 - 900, Y0 + 60)]),                      # fuera del alcance
    ("highway=proposed", None, [(X0 + 10, Y0 + 70), (X0 + 90, Y0 + 70)]),                    # no entra al paquete
]
# 87 construcciones a unos 200 m al oriente y 5 muy lejos
EDIFICIOS = [(X0 + 280 + (i % 10), Y0 + 20 + i // 10) for i in range(87)] + [(X0 + 3000, Y0)] * 5
PAQUETE = osm.empacar(VIAS, EDIFICIOS, fecha="2026-11-27")


def test_lados_colindancias_y_construcciones():
    r = osm.consultar(ANILLO, datos=PAQUETE)
    calle = r["calles"][0]
    assert calle["tipo"] == "calle residencial" and calle["lado"] == "norte" and calle["colinda"]
    assert calle["distancia_m"] == pytest.approx(5, abs=0.5) and calle["nombre"] == "Calle El Progreso"
    assert r["calles"][1]["tipo"] == "carretera primaria" and r["calles"][1]["lado"] == "poniente"
    assert not r["calles"][1]["colinda"] and len(r["calles"]) == 2  # la lejana y la propuesta no salen
    agua = r["aguas"][0]
    assert agua["tipo"] == "quebrada" and agua["lado"] == "sur" and agua["colinda"]
    assert r["senderos"][0]["lado"] == "oriente" and not r["senderos"][0]["colinda"]
    assert r["construcciones_500m"] == 87 and "ODbL" in r["fuente"] and r["fecha_mapa"] == "2026-11-27"
    res = osm.resumen(r)
    assert res["calles"][0].startswith("calle residencial «Calle El Progreso»: pegada al lado norte")
    assert "2026-11-27" in res["fuente"]


def test_lo_que_cruza_el_terreno_y_el_lago():
    vias = [("waterway=stream", "Q. Seca", [(X0 + 50, Y0 - 40), (X0 + 50, Y0 + 100)]),
            ("natural=water", None, [(X0 + 80, Y0 + 40), (X0 + 140, Y0 + 40), (X0 + 140, Y0 + 90),
                                     (X0 + 80, Y0 + 90), (X0 + 80, Y0 + 40)])]
    r = osm.consultar(ANILLO, datos=osm.empacar(vias))
    q = next(a for a in r["aguas"] if a["tipo"] == "quebrada")
    assert q["cruza_m"] == pytest.approx(60, abs=0.5) and q["distancia_m"] == 0
    lago = next(a for a in r["aguas"] if a["tipo"] == "cuerpo de agua")
    assert lago["dentro_m2"] == pytest.approx(20 * 20, abs=2) and lago["lado"] in ("noreste", "oriente", "norte")
    assert "cruza el terreno (60 m)" in osm.resumen(r)["rios_y_quebradas"][0] or \
        "cruza el terreno (60 m)" in osm.resumen(r)["rios_y_quebradas"][1]
    assert r["construcciones_500m"] == 0


def test_lo_que_va_por_el_lindero_no_cruza():
    """Una calle o quebrada dibujada 2 m adentro del lindero (error normal del mapa) colinda, no cruza."""
    vias = [("highway=secondary", "Carretera", [(X0 + 98, Y0 - 50), (X0 + 98, Y0 + 110)]),
            ("waterway=stream", None, [(X0 - 20, Y0 + 2), (X0 + 120, Y0 + 2)])]
    r = osm.consultar(ANILLO, datos=osm.empacar(vias))
    c, q = r["calles"][0], r["aguas"][0]
    assert c["lado"] == "oriente" and c["colinda"] and "cruza_m" not in c
    assert q["lado"] == "sur" and q["colinda"] and "cruza_m" not in q
    assert osm.resumen(r)["calles"][0] == "carretera secundaria «Carretera»: pegada al lado oriente"


def test_guardar_y_cargar_y_sin_paquete(tmp_path):
    ruta = str(tmp_path / "osm.npz")
    osm.guardar(PAQUETE, ruta)
    p = osm.cargar(ruta)
    assert p["nombres"] == PAQUETE["nombres"] and p["meta"]["fecha"] == "2026-11-27"
    assert osm.consultar(ANILLO, datos=p) == osm.consultar(ANILLO, datos=PAQUETE)
    assert osm.cargar(str(tmp_path / "no_esta.npz")) is None
    osm._cache.clear()
    try:
        osm.ARCHIVO, viejo = str(tmp_path / "no_esta.npz"), osm.ARCHIVO
        assert osm.consultar(ANILLO) is None  # sin paquete el bot sigue, solo sin datos del mapa
    finally:
        osm.ARCHIVO = viejo
        osm._cache.clear()


def _osm_xml(tmp_path):
    """Archivo .osm chiquito (formato de OpenStreetMap) con vías, una quebrada, un estanque, un lago multipolígono y
    tres construcciones, para probar el armado del paquete."""
    nodos, n = [], [0]

    def nodo(x, y):
        n[0] += 1
        lat, lng = proyeccion.a_latlng(x, y)
        nodos.append(f'<node id="{n[0]}" version="1" lat="{lat:.8f}" lon="{lng:.8f}"/>')
        return n[0]

    def via(i, pts, tags):
        nd = "".join(f'<nd ref="{k}"/>' for k in pts)
        tg = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
        return f'<way id="{i}" version="1">{nd}{tg}</way>'

    calle = [nodo(X0 - 50, Y0 + 65), nodo(X0 + 150, Y0 + 65)]
    queb = [nodo(X0 - 50, Y0 - 20), nodo(X0 + 50, Y0 - 21), nodo(X0 + 150, Y0 - 22)]
    est = [nodo(X0 + 200, Y0), nodo(X0 + 220, Y0), nodo(X0 + 220, Y0 + 20), nodo(X0 + 200, Y0 + 20)]
    lago_a = [nodo(X0 - 100, Y0 + 100), nodo(X0 - 60, Y0 + 100), nodo(X0 - 60, Y0 + 140)]
    lago_b = [lago_a[-1], nodo(X0 - 100, Y0 + 140), lago_a[0]]
    casas = [[nodo(X0 + 300 + 15 * k, Y0), nodo(X0 + 308 + 15 * k, Y0), nodo(X0 + 308 + 15 * k, Y0 + 8)] for k in range(3)]
    ways = [via(1, calle, {"highway": "residential", "name": "Calle Ñandú"}),
            via(2, queb, {"waterway": "stream"}),
            via(3, est + [est[0]], {"natural": "water"}),
            via(4, lago_a, {}), via(5, lago_b, {}),
            via(6, calle, {"highway": "proposed"})]
    ways += [via(10 + k, c + [c[0]], {"building": "yes"}) for k, c in enumerate(casas)]
    rel = ('<relation id="1" version="1"><member type="way" ref="4" role="outer"/>'
           '<member type="way" ref="5" role="outer"/><tag k="type" v="multipolygon"/>'
           '<tag k="natural" v="water"/><tag k="name" v="Laguna de prueba"/></relation>')
    ruta = tmp_path / "prueba.osm"
    ruta.write_text('<?xml version="1.0" encoding="UTF-8"?><osm version="0.6">' + "".join(nodos) + "".join(ways)
                    + rel + "</osm>", encoding="utf-8")
    return str(ruta)


def test_construir_paquete_desde_un_archivo_osm(tmp_path):
    pytest.importorskip("osmium")
    from herramientas import construir_osm

    salida = str(tmp_path / "osm_sv.npz")
    meta = construir_osm.construir(_osm_xml(tmp_path), salida)
    assert meta["vias"] == 4 and meta["construcciones"] == 3  # calle, quebrada, estanque y laguna
    r = osm.consultar(ANILLO, datos=osm.cargar(salida))
    assert r["calles"][0]["nombre"] == "Calle Ñandú" and r["calles"][0]["lado"] == "norte"
    assert r["calles"][0]["distancia_m"] == pytest.approx(5, abs=0.2) and len(r["calles"]) == 1
    tipos = sorted((a["tipo"], a.get("nombre") or "") for a in r["aguas"])
    assert tipos == [("cuerpo de agua", ""), ("cuerpo de agua", "Laguna de prueba"), ("quebrada", "")]
    assert r["construcciones_500m"] == 3


def test_el_bot_saca_acceso_y_quebrada_del_mapa(tmp_path, monkeypatch):
    """Sin colindancias en el plano, el acceso y la quebrada salen del mapa (como sugerencia para confirmar)."""
    import test_agente as TA
    from agente import Agente, Sesion
    from motor import relieve
    from motor.terreno import Terreno

    m = relieve.Malla.de_dict(json.loads(TA._texto("chiltiupan_malla.json")))
    monkeypatch.setattr(relieve, "descargar", lambda anillo, paso=5.0, http=None: m)
    monkeypatch.setattr(osm, "consultar", lambda anillo, datos=None: _consultar_real(anillo, datos=PAQUETE))
    ag = Agente(TA.Guion(TA._lectura_chiltiupan()), str(tmp_path))
    s = Sesion("mapa")
    s.set_terreno(Terreno(nombres=["1", "2", "3", "4"], puntos=ANILLO, tramos=[], cierre=0.0, perimetro=320.0,
                          ubicado=True))
    sug = ag._averiguar(s, s.get_terreno(), m, {"pendiente_media": 20})
    assert sug["ficha"]["acceso_lado"] == "norte" and "según el mapa" in sug["encontrado"]["acceso"]
    assert sug["ficha"]["quebradas"][0]["lado"] == "sur"
    assert "mapa" in sug["encontrado"]["quebradas"][0]["origen"]
    assert sug["encontrado"]["mapa"]["construcciones_500m"] == 87
    # si el plano dice otro lado, se respeta el plano y se avisa la diferencia
    s.ficha["del_plano"] = [{"tipo": "calle", "lado": "oriente"}]
    sug = ag._averiguar(s, s.get_terreno(), m, {"pendiente_media": 20})
    assert sug["ficha"]["acceso_lado"] == "oriente" and "norte" in sug["encontrado"]["acceso_mapa"]
    # la quebrada que dice el usuario no se cuestiona si el mapa la muestra de ese lado
    s.ficha["quebradas"] = [{"tipo": "quebrada", "lado": "sur", "profundidad_m": 5}]
    s.malla = relieve.Malla(X0 - 100, Y0 - 100, 5.0, 61, 53,
                            np.fromfunction(lambda i, j: 0.37 * (i * 5.0), (53, 61)), "x").a_dict()
    assert ag._revisar_quebradas(s) == []
    s.osm = None
    assert ag._revisar_quebradas(s)  # sin mapa, el relieve (ladera sin cauce) sí pide confirmarla


_consultar_real = osm.consultar
