"""Ubicar a mano: con solo rumbos y distancias se sabe la forma, no dónde está. Basta la X/Y de un mojón."""
import json
import os

from agente import Agente, Sesion
from motor.terreno import desde_texto_rumbos
from servicios.llm import ReglasLLM

DATOS = os.path.join(os.path.dirname(__file__), "datos")


def _txt(n):
    with open(os.path.join(DATOS, n), encoding="utf-8") as f:
        return f.read()


def _referencia():
    """Chiltiupán ubicado con su Mo. 1: de aquí salen las coordenadas "verdaderas" de cada mojón."""
    ini = json.loads(_txt("chiltiupan_inicio.json"))["mo1"]
    t, _ = desde_texto_rumbos(_txt("chiltiupan_rumbos.txt"), inicio=tuple(ini))
    return t


def _solo_rumbos(tmp_path):
    ag = Agente(ReglasLLM(), str(tmp_path))
    s = Sesion("u")
    out = ag.atender(s, _txt("chiltiupan_rumbos.txt"))
    return ag, s, out[-1]["texto"]


def test_nombres_de_mojon():
    t = _referencia()
    assert [t.indice_vertice(r) for r in ("Mo. 1", "mo1", "Mojón 5", "5", "Mo.18", "vertice1")] == [0, 0, 4, 4, 17, 0]
    assert t.indice_vertice("Mo. 40") is None and t.indice_vertice("") is None


def test_solo_rumbos_pide_la_coordenada_de_un_mojon_y_ubica_con_ella(tmp_path):
    ag, s, txt = _solo_rumbos(tmp_path)
    assert not s.get_terreno().ubicado
    assert "X y la Y (Lambert)" in txt and "mojón" in txt
    ref = _referencia()
    x5, y5 = ref.puntos[4]
    out = ag.atender(s, f"Mo. 5 X {x5:,.3f} Y {y5:,.3f}")
    t = s.get_terreno()
    assert t.ubicado and "Ubicado con la coordenada del Mo. 5" in out[-1]["texto"]
    for (xa, ya), (xb, yb) in zip(t.puntos, ref.puntos):
        assert abs(xa - xb) < 0.01 and abs(ya - yb) < 0.01


def test_columnas_al_reves_locales_y_mojon_que_no_existe(tmp_path):
    ag, s, _ = _solo_rumbos(tmp_path)
    ref = _referencia()
    x1, y1 = ref.puntos[0]
    r = ag.h_ubicar_terreno(s, x=y1, y=x1, referencia="1")  # escribieron primero la Y
    assert r["ok"] and abs(s.get_terreno().puntos[0][0] - x1) < 0.01
    r = ag.h_ubicar_terreno(s, x=1000, y=1000, referencia="Mo. 1")
    assert not r["ok"] and "locales" in r["error"]
    r = ag.h_ubicar_terreno(s, x=x1, y=y1, referencia="Mo. 40")
    assert not r["ok"] and r["mojones"][0] == "1"


def test_mas_de_un_mojon_se_comprueba_y_reubicar_borra_el_relieve_viejo(tmp_path):
    ag, s, _ = _solo_rumbos(tmp_path)
    ref = _referencia()
    (x1, y1), (x9, y9) = ref.puntos[0], ref.puntos[8]
    r = ag.h_ubicar_terreno(s, x=x1, y=y1, referencia="Mo. 1", puntos=[{"mojon": "Mo. 9", "x": x9, "y": y9}])
    assert r["ok"] and r["comprobacion"][0]["diferencia_m"] < 0.05 and "aviso" not in r
    s.malla = {"viejo": True}
    r = ag.h_ubicar_terreno(s, x=x1, y=y1, referencia="Mo. 1", puntos=[{"mojon": "Mo. 9", "x": x9 + 10, "y": y9}])
    assert "aviso" in r and "Mo. 9" in r["aviso"]
    assert s.malla is None


def test_coordenadas_siempre_de_un_mojon(tmp_path):
    """Sin mojón no se ubica (ya no hay "centro" aproximado); latitud y longitud escritas de un mojón sí sirven."""
    from motor import proyeccion

    ag, s, _ = _solo_rumbos(tmp_path)
    lat, lng = proyeccion.a_latlng(*_referencia().puntos[0])
    r = ag.h_ubicar_terreno(s, lat=lat, lng=lng)
    assert not r["ok"] and "mojón" in r["error"] and not s.get_terreno().ubicado
    r = ag.h_ubicar_terreno(s, lat=lat, lng=lng, referencia="Mo. 1")
    assert r["ok"] and "Mo. 1" in r["nota"]
    assert abs(s.get_terreno().puntos[0][0] - _referencia().puntos[0][0]) < 0.5
