"""Calles, ríos y construcciones de OpenStreetMap (Overpass simulado)."""
import json

import pytest
from shapely.geometry import Polygon

from motor import proyeccion
from servicios import osm

pytestmark = pytest.mark.mapa

X0, Y0 = 473200.0, 289000.0
ANILLO = [(X0, Y0), (X0 + 100, Y0), (X0 + 100, Y0 + 60), (X0, Y0 + 60)]


def _via(coords, **tags):
    return {"type": "way", "tags": tags,
            "geometry": [dict(zip(("lat", "lon"), proyeccion.a_latlng(x, y))) for x, y in coords]}


RESPUESTA = {"elements": [
    _via([(X0 - 50, Y0 + 65), (X0 + 150, Y0 + 65)], highway="residential", name="Calle El Progreso"),  # 5 m al norte
    _via([(X0 - 50, Y0 - 20), (X0 + 150, Y0 - 22)], waterway="stream"),                               # 20 m al sur
    _via([(X0 + 200, Y0 - 50), (X0 + 200, Y0 + 100)], highway="footway"),                              # 100 m al oriente
    _via([(X0 - 300, Y0), (X0 - 300, Y0 + 60)], highway="primary", name="Carretera X"),               # 300 m poniente
    {"type": "count", "tags": {"ways": "87", "total": "87"}},
]}


class Overpass:
    def __init__(self, falla_primero=False):
        self.falla_primero, self.urls = falla_primero, []

    def post(self, url, data=None, timeout=None, headers=None):
        self.urls.append(url)
        assert "way[\"highway\"]" in data["data"] and "out count" in data["data"] and headers["User-Agent"]
        if self.falla_primero and len(self.urls) == 1:
            raise ConnectionError("ocupado")

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return RESPUESTA
        return R()


def test_interpretar_lados_y_colindancias():
    r = osm.interpretar(RESPUESTA, Polygon(ANILLO))
    calle = r["calles"][0]
    assert calle["tipo"] == "calle residencial" and calle["lado"] == "norte" and calle["colinda"]
    assert calle["distancia_m"] == pytest.approx(5, abs=0.5) and calle["nombre"] == "Calle El Progreso"
    assert r["calles"][1]["tipo"] == "carretera primaria" and r["calles"][1]["lado"] == "poniente"
    assert not r["calles"][1]["colinda"]
    agua = r["aguas"][0]
    assert agua["tipo"] == "quebrada" and agua["lado"] == "sur" and agua["colinda"]
    assert r["senderos"][0]["lado"] == "oriente" and not r["senderos"][0]["colinda"]
    assert r["construcciones_500m"] == 87 and "ODbL" in r["fuente"]
    res = osm.resumen(r)
    assert res["calles"][0].startswith("calle residencial «Calle El Progreso»: pegada al lado norte")


def test_si_un_servidor_falla_usa_el_otro_y_si_ninguno_responde_none():
    http = Overpass(falla_primero=True)
    r = osm.consultar(ANILLO, http=http)
    assert r and len(http.urls) == 2 and http.urls[0] != http.urls[1]

    class Nada:
        def post(self, *a, **k):
            raise ConnectionError("sin red")
    assert osm.consultar(ANILLO, http=Nada()) is None


def test_el_bot_saca_acceso_y_quebrada_del_mapa(tmp_path, monkeypatch):
    """Sin colindancias en el plano, el acceso y la quebrada salen del mapa (como sugerencia para confirmar)."""
    import test_agente as TA
    from agente import Agente, Sesion
    from motor import relieve
    from motor.terreno import Terreno

    m = relieve.Malla.de_dict(json.loads(TA._texto("chiltiupan_malla.json")))
    monkeypatch.setattr(relieve, "descargar", lambda anillo, paso=5.0, http=None: m)
    monkeypatch.setattr(osm, "consultar", lambda anillo, http=None: osm.interpretar(RESPUESTA, Polygon(anillo)))
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
                            __import__("numpy").fromfunction(lambda i, j: 0.37 * (i * 5.0), (53, 61)), "x").a_dict()
    assert ag._revisar_quebradas(s) == []
    s.osm = None
    assert ag._revisar_quebradas(s)  # sin mapa, el relieve (ladera sin cauce) sí pide confirmarla
