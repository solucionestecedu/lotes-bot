"""Lo que el bot averigua solo: dónde cae, si está en el AMSS, acceso y quebradas del plano (con su profundidad por el
relieve) y el tipo de proyecto que le queda. Se confirma en un solo mensaje."""
import json
import os

from agente import Agente, Sesion
from motor import proyeccion, relieve, territorio
from motor.terreno import desde_texto_rumbos
from servicios.llm import ReglasLLM

DATOS = os.path.join(os.path.dirname(__file__), "datos")


def _txt(n):
    with open(os.path.join(DATOS, n), encoding="utf-8") as f:
        return f.read()


def _chiltiupan():
    ini = json.loads(_txt("chiltiupan_inicio.json"))["mo1"]
    t, _ = desde_texto_rumbos(_txt("chiltiupan_rumbos.txt"), inicio=tuple(ini))
    return t, relieve.Malla.de_dict(json.loads(_txt("chiltiupan_malla.json")))


def test_donde_cae_y_amss():
    t, _ = _chiltiupan()
    d = territorio.donde(*t.centroide())
    assert (d["distrito"], d["municipio"], d["departamento"], d["amss"]) == ("Chiltiupán", "La Libertad Costa",
                                                                               "La Libertad", True)
    for (lat, lng), (dist, amss) in {(13.6989, -89.1914): ("San Salvador", True), (13.9942, -89.5597): ("Santa Ana", False),
                                     (13.938, -89.028): ("Suchitoto", False)}.items():
        d = territorio.donde(*proyeccion.a_lambert(lat, lng))
        assert d["distrito"] == dist and d["amss"] is amss
    assert territorio.donde(1000, 1000) is None


def test_los_28_distritos_del_amss():
    props, _, _ = territorio._datos()
    amss = {p["distrito"] for p in props if p["amss"]}
    assert len(amss) == 28 and {"Comasagua", "Panchimalco", "Chiltiupán", "Santa Tecla", "Nejapa"} <= amss
    assert len(props) >= 260


def test_profundidad_y_propuesta():
    t, m = _chiltiupan()
    p = territorio.profundidad_al_lado(m, t.puntos, "oriente")
    assert p and 0 < p["tipica_m"] <= p["maxima_m"]
    assert territorio.profundidad_al_lado(m, t.puntos, "arriba") is None
    assert territorio.propuesta_proyecto(t.area, 37, True)["tipo"] == "campestre"
    assert territorio.propuesta_proyecto(30000, 8, True) == {"tipo": "residencial", "lote_objetivo_m2": 250,
                                                             "por_que": "plano y dentro del área metropolitana: lotes residenciales"}


def test_relieve_trae_lo_encontrado_y_se_confirma_de_una_vez(tmp_path, monkeypatch):
    t, m = _chiltiupan()
    monkeypatch.setattr(relieve, "descargar", lambda puntos, **k: m)
    ag = Agente(ReglasLLM(), str(tmp_path))
    s = Sesion("c")
    s.set_terreno(t)
    s.ficha["del_plano"] = [{"lado": "norte", "tipo": "calle", "descripcion": "calle vecinal"},
                            {"lado": "oriente", "tipo": "quebrada"}]
    r = ag.h_obtener_relieve(s)
    enc = r["encontrado"]
    assert enc["lugar"]["distrito"] == "Chiltiupán" and "OPAMSS" in enc["zona"]
    assert "norte" in enc["acceso"] and enc["quebradas"][0]["lado"] == "oriente" and enc["quebradas"][0]["profundidad_m"] > 0
    assert enc["propuesta"]["tipo"] == "campestre"
    # lo que el bot ya sabe no se pregunta: solo lo que decide el dueño
    falta = " ".join(r["preguntar_solo"])
    assert "zona" not in falta and "calle pública" not in falta and "quebradas" not in falta and "tipo" not in falta
    assert "portón" in falta and "agua" in falta
    g = ag.h_guardar_ficha(s, usar_sugerido=True, cerrado=True, lote_objetivo_m2=800)
    f = g["ficha"]
    assert f["zona"] == "amss" and f["acceso_lado"] == "norte" and f["quebradas"][0]["lado"] == "oriente"
    assert f["lote_objetivo_m2"] == 800 and f["tipo"] == "campestre"  # lo que corrige el usuario manda
    assert "sugerido" not in f
