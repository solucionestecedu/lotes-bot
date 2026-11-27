"""Lotes de un solo lado: en una franja angosta la calle va pegada a un lindero largo desde la entrada y los lotes
quedan en una sola fila (caso real: Chiltiupán, entrada al oriente y sin quebrada)."""
import json

import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from agente import Agente, Sesion
from motor import altimetria, relieve
from motor.calle_central import bordes_largos, eje_de_borde, es_angosto, lotificar_un_lado
from motor.lotificacion import Parametros, proteccion_de_quebrada
import test_agente as TA
from test_normas import caso_chiltiupan  # noqa: F401


def _P(**kw):
    return Parametros.de_dict(dict(frente=15, fondo=40, area_minima=540, regla_verde="pct_util", valor_verde=10,
                                   pct_recreativa=50, lado_acceso="oriente", **kw))


def test_franja_y_sus_linderos_largos(caso_chiltiupan):
    t, _, _ = caso_chiltiupan
    poly = Polygon(t.puntos)
    assert es_angosto(poly)
    assert sorted(bordes_largos(poly)) == ["norte", "sur"]
    assert not es_angosto(Polygon([(0, 0), (100, 0), (100, 80), (0, 80)]))  # un terreno ancho no es franja
    for lado in ("norte", "sur"):
        eje = LineString(eje_de_borde(poly, lado, 8.0))
        assert poly.contains(eje.interpolate(0.5, normalized=True))
        assert abs(eje.interpolate(0.5, normalized=True).distance(poly.exterior) - 4.0) < 0.05  # media calle adentro
        assert eje.length > 350  # todo el lindero largo, sin doblar las esquinas


@pytest.mark.parametrize("borde", ["norte", "sur"])
def test_una_fila_desde_la_entrada(caso_chiltiupan, borde):
    t, m, _ = caso_chiltiupan
    poly = Polygon(t.puntos)
    barr = altimetria.barrancos(m, t.puntos, partir=True)
    r = lotificar_un_lado(t.puntos, _P(), m, [], borde=borde, cruzables=barr)
    st = r.stats
    assert st["un_lado"] and st["calle_pegada_a"] == borde
    assert st["lotes"] >= 11 and st["area_util"] / st["area_total"] > 0.55  # antes, en retícula: 8 lotes y 51 %
    assert all(l["area"] >= 540 - 0.5 for l in r.lotes)
    # todos del mismo lado de la calle: ninguno toca el lindero del lado de la calle
    calle = unary_union(r.calles)
    assert calle.within(poly.buffer(0.5))
    # el lote 1 queda junto a la entrada (al oriente) y la calle entra por ahí
    ent = st["entrada_calle"]
    assert ent[0] > poly.centroid.x + 150
    assert r.lotes[0]["poly"].centroid.x > r.lotes[-1]["poly"].centroid.x
    # la boca de la calle es su sección en la entrada (8 m), no todo el costado pegado al lindero
    (ax, ay), (bx, by) = st["boca_calle"]
    assert abs(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 - 8.0) < 0.1
    # ningún lote pisa una caída fuerte; la calle sí la cruza
    zona = unary_union(barr)
    assert all(l["poly"].intersection(zona).area < 1.0 for l in r.lotes)
    # el área verde (recreativa + caídas fuertes que quedan) alcanza el 10 % del área útil
    assert st["area_verde"] + st["area_proteccion"] >= 0.10 * st["area_util"] - 1.0


def test_sin_quebrada_no_hay_zona_de_quebrada():
    a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    b = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])
    assert proteccion_de_quebrada([a, b], [(0, 0), (40, 0), (40, 20), (0, 20)], {"sin_quebradas": True}) is None
    assert proteccion_de_quebrada([a, b], [(0, 0), (40, 0), (40, 20), (0, 20)], None) is not None


def test_agente_entrada_al_oriente_sin_quebrada(tmp_path, monkeypatch):
    """El caso que marcó el usuario: la entrada al oriente, no hay quebrada y los lotes de un solo lado."""
    m = relieve.Malla.de_dict(json.loads(TA._texto("chiltiupan_malla.json")))
    monkeypatch.setattr(relieve, "descargar", lambda anillo, paso=5.0, http=None: m)
    ag = Agente(TA.Guion(TA._lectura_chiltiupan()), str(tmp_path))
    s = Sesion("un_lado")
    ag.atender(s, "este es el plano", [{"mime": "image/jpeg", "nombre": "plano.jpg", "datos": b"foto-del-plano"}])
    assert ag.h_obtener_relieve(s)["ok"]
    ag.h_guardar_ficha(s, zona="amss", tipo="campestre", lote_objetivo_m2=600, acceso_lado="oriente", sin_quebradas=True,
                       cerrado=True, figura="lotificacion", agua="anda", luz="si")
    out = ag.h_predisenar(s)
    assert out["distribucion"] == "un_lado" and out["lotes"] >= 12 and out["aprovechamiento_pct"] > 60
    estados = {f["regla"]: f["estado"] for f in out["cumplimiento"]}
    assert estados.get("Equipamiento social") == "cumple"  # OPAMSS: 8 m² por lote, se deja junto al retorno
    assert estados.get("Área verde total") == "cumple"
    assert not any("quebrada" in f["regla"].lower() for f in out["cumplimiento"] if f["estado"] == "no_cumple")
    # pedir la calle del otro lado
    out2 = ag.h_predisenar(s, calle_pegada_a="sur")
    assert out2["calle_pegada_a"] == "sur" and out2["lotes"] >= 11
    # el diseñador ofrece la calle del otro lado
    from motor import disenador

    hoy = {"frente": 15, "fondo": 40, "area_minima": 540}
    vs = disenador.variantes({}, s.ficha, False, "un_lado", None, medidas=hoy, angosto=True, bordes=["norte", "sur"],
                             borde_actual="norte")
    assert any(v["param"].get("calle_pegada_a") == "sur" for v in vs)
    vs = disenador.variantes({}, s.ficha, False, "automatica", None, angosto=True)
    assert any(v["param"].get("distribucion") == "un_lado" for v in vs)


# ------------------------------------------------------------------ la quebrada que dice el usuario contra el relieve
def _malla(z_de):
    import numpy as np

    xs, ys = np.arange(-60, 261, 5.0), np.arange(-70, 111, 5.0)
    X, Y = np.meshgrid(xs, ys)
    return relieve.Malla(xs[0], ys[0], 5.0, len(xs), len(ys), z_de(X, Y), "prueba")


ANILLO = [(0, 0), (200, 0), (200, 40), (0, 40)]


def test_cauce_ladera_o_sube():
    import numpy as np

    from motor import territorio

    valle = _malla(lambda X, Y: np.where(Y < 0, -6 * np.exp(-((Y + 20) / 8) ** 2), 0.37 * Y))
    c = territorio.cauce_al_lado(valle, ANILLO, "sur")
    assert c["veredicto"] == "cauce" and c["hondo_m"] == pytest.approx(6, abs=1) and c["a_m"] == 20
    assert territorio.cauce_al_lado(_malla(lambda X, Y: 0.37 * Y), ANILLO, "sur")["veredicto"] == "ladera"
    assert territorio.cauce_al_lado(_malla(lambda X, Y: -0.2 * Y), ANILLO, "sur")["veredicto"] == "sube"


def test_el_bot_pregunta_si_el_relieve_no_muestra_la_quebrada(tmp_path):
    from servicios.llm import ReglasLLM
    from motor.terreno import Terreno

    ag = Agente(ReglasLLM(), str(tmp_path))
    s = Sesion("q")
    t = Terreno(nombres=["1", "2", "3", "4"], puntos=ANILLO, tramos=[], cierre=0.0, perimetro=480.0, ubicado=True)
    s.set_terreno(t)
    s.malla = _malla(lambda X, Y: 0.37 * Y).a_dict()  # ladera: baja hacia el sur sin fondo
    r = ag.h_guardar_ficha(s, quebradas=[{"tipo": "quebrada", "lado": "sur", "profundidad_m": 6}])
    assert r["ok"] and "revisar_quebrada" in r and "sur" in r["revisar_quebrada"][0]
    r = ag.h_guardar_ficha(s, quebrada_confirmada=True)  # el usuario confirma: ya no se vuelve a preguntar
    assert "revisar_quebrada" not in r
    s.ficha = {"quebradas": [{"tipo": "quebrada", "lado": "sur", "profundidad_m": 6}],
               "del_plano": [{"lado": "sur", "tipo": "quebrada"}]}
    assert ag._revisar_quebradas(s) == []  # lo que dice el plano no se cuestiona
