"""Diseñador: el motor arma y mide variantes; el modelo (aquí uno de mentira) solo elige y explica."""
from agente import Agente, Sesion
from motor import disenador
from test_calle_central import GuionQuintas


def _quintas(tmp_path):
    ag = Agente(GuionQuintas(), str(tmp_path))
    s = Sesion("d")
    ag.atender(s, "plano", [{"mime": "image/jpeg", "nombre": "p.jpg", "datos": b"x"}])
    ag.atender(s, "unir")
    ag.atender(s, "quintas")
    ag.atender(s, "lotes")
    return ag, s


def test_variantes_cambian_una_palanca_sin_repetir():
    vs = disenador.variantes({"lotes_por_lado": [4, 5]}, {}, True, "calle_central", [4, 5])
    assert vs[0]["cambio"] == "el prediseño actual" and len({v["id"] for v in vs}) == len(vs) >= 4
    assert any(v["param"].get("lotes_por_lado") == [3, 4] for v in vs)
    assert any(v["param"].get("calle_sobre") == "terreno_mas_chico" for v in vs)


def test_proponer_alternativas_sin_modelo_usa_la_regla_y_no_toca_el_prediseno(tmp_path):
    ag, s = _quintas(tmp_path)
    antes, cumpl = dict(s.lotes_param or {}), s.cumplimiento
    salidas = []
    r = ag.h_proponer_alternativas(s, salidas=salidas)
    assert r["ok"] and r["eligio"] == "regla" and 1 <= len(r["opciones"]) <= 3
    for o in r["opciones"]:
        assert {"lotes", "area_vendible_m2", "vendible_pct"} <= set(o["medidas"]) and "parametros" in o
    assert salidas and salidas[0]["nombre"] == "opciones.jpg"
    assert (s.lotes_param or {}) == antes and s.cumplimiento == cumpl


def test_el_modelo_elige_pero_no_inventa_variantes(tmp_path):
    ag, s = _quintas(tmp_path)
    vistos = []

    def pensar(texto, esquema):
        vistos.append(texto)
        return {"recomendadas": [{"id": "Z", "titulo": "inventada", "por_que": "", "contra": ""},
                                 {"id": "B", "titulo": "Lotes más grandes", "por_que": "se venden mejor",
                                  "contra": "un lote menos"}],
                "advertencias": ["la calle pasa de 12 %"]}

    ag.llm.pensar = pensar
    r = ag.h_proponer_alternativas(s, salidas=[])
    assert r["eligio"] == "modelo" and [o["titulo"] for o in r["opciones"]] == ["Lotes más grandes"]
    assert r["advertencias"] == ["la calle pasa de 12 %"]
    assert '"id": "A"' in vistos[0] and "área del terreno" in vistos[0]  # el modelo recibe las cifras del motor
