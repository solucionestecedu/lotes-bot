"""Quintas con una calle en medio de dos terrenos y el área de la quebrada al fondo."""
import json

import pytest
from pypdf import PdfReader
from shapely.geometry import Polygon
from shapely.ops import unary_union

from agente import Agente, Sesion
from motor import colindancia
from motor.calle_central import lindero_comun, lotificar_calle_central
from motor.lotificacion import Parametros, franja_de_proteccion
from servicios.llm import Respuesta
from test_normas import Guion, _txt, caso_chiltiupan  # noqa: F401
from motor.terreno import desde_texto_rumbos


@pytest.fixture(scope="module")
def union():
    a, _ = desde_texto_rumbos(_txt("plano_a.txt"))
    b, _ = desde_texto_rumbos(_txt("plano_b.txt"))
    b2, _ = colindancia.alinear(a, b)
    return a, b2, colindancia.unir([a, b2])


def _quintas(a, b2, u, rango=(4, 5)):
    P = Parametros.de_dict(dict(frente=20, fondo=35, area_minima=630, regla_verde="pct_util", valor_verde=10,
                                lado_acceso="norte"))
    franja = franja_de_proteccion(u.puntos, "sur", 12.0)
    return P, lotificar_calle_central(u.puntos, P, None, [franja], eje=lindero_comun(a.puntos, b2.puntos),
                                      lotes_por_lado=rango,
                                      partes=[{"origen": "A", "anillo": a.puntos}, {"origen": "B", "anillo": b2.puntos}])


def test_lindero_comun_mide_lo_del_plano(union):
    a, b2, _ = union
    lc = lindero_comun(a.puntos, b2.puntos)
    largo = sum(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 for p, q in zip(lc, lc[1:]))
    assert largo == pytest.approx(105.54, abs=0.3)


def test_calle_en_medio_con_lotes_a_cada_lado(union):
    a, b2, u = union
    P, r = _quintas(a, b2, u)
    st = r.stats
    lados = {pl["terreno"]: pl for pl in st["por_lado"]}
    assert set(lados) == {"A", "B"}
    for pl in lados.values():  # 4 o 5 por terreno, o el motivo de por qué no cupo
        assert 4 <= pl["lotes"] <= 5
        assert pl["lotes"] == 5 or "no cabe" in pl["nota"]
    calle = unary_union(r.calles)
    assert len(r.calles) == 1 and calle.geom_type == "Polygon"
    for l in r.lotes:  # todos con frente a la calle y del tamaño mínimo, sin pisar la calle
        assert l["frente"] >= 0.6 * P.frente - 1e-6 and l["area"] >= P.area_minima
        assert l["poly"].boundary.intersection(calle.buffer(0.3)).length >= 0.6 * P.frente - 1e-6
        assert l["poly"].intersection(calle).area < 1.0
    # cada lote queda en su terreno (la calle va sobre el lindero)
    for l in r.lotes:
        parte = Polygon(a.puntos if l["terreno"] == "A" else b2.puntos)
        assert l["poly"].intersection(parte).area >= 0.97 * l["area"]
    # al fondo: zona de protección y área verde con entrada desde el retorno, sin sobrar de más
    req = st["area_verde_requerida"]
    meta = max(req - st["area_proteccion"], req * P.pct_recreativa / 100)
    assert st["area_proteccion"] > 500 and meta - 1 <= st["area_verde"] <= meta + P.verde_minimo + 150
    assert any(v["poly"].boundary.intersection(calle.buffer(0.3)).length >= 1.0 for v in r.verdes)
    # nada se pierde: lotes + calle + verde + protección + remanentes = terreno
    suma = st["area_util"] + st["area_circulacion"] + st["area_verde"] + st["area_proteccion"] + st["area_remanente"]
    assert suma == pytest.approx(u.area, abs=2.0)
    assert st["area_util"] / u.area > 0.72  # con la franja de la quebrada a lo largo de toda la orilla


def test_rango_fijo_y_alternativas(union):
    a, b2, u = union
    _, r = _quintas(a, b2, u, rango=(4, 4))
    assert all(pl["lotes"] == 4 for pl in r.stats["por_lado"])
    _, r = _quintas(a, b2, u, rango=(3, 5))
    b = next(pl for pl in r.stats["por_lado"] if pl["terreno"] == "B")
    assert {x["lotes"] for x in b["alternativas"]} >= {3, 4}


def test_un_solo_terreno_sin_lindero():
    a, _ = desde_texto_rumbos(_txt("plano_a.txt"))
    P = Parametros.de_dict(dict(frente=20, fondo=35, area_minima=500, lado_acceso="norte"))
    r = lotificar_calle_central(a.puntos, P, None, [], lotes_por_lado=(2, 3))
    assert 4 <= r.stats["lotes"] <= 6 and r.stats["accesos"] == 1


class GuionQuintas(Guion):
    def turno(self, sistema, mensajes, herramientas):
        u = mensajes[-1]
        if u["rol"] == "usuario" and u["texto"] == "quintas":
            return Respuesta(llamadas=[{"nombre": "guardar_ficha", "args": {
                "zona": "urbano", "tipo": "campestre", "acceso_lado": "norte", "cerrado": True,
                "quebradas": [{"tipo": "quebrada", "lado": "sur", "profundidad_m": 8}], "figura": "lotificacion",
                "agua": "anda", "luz": "si", "distribucion": "calle_central", "lotes_por_lado": [5, 4]}}])
        return super().turno(sistema, mensajes, herramientas)


def test_flujo_quintas_con_calle_en_medio(tmp_path):
    ag = Agente(GuionQuintas(), str(tmp_path))
    s = Sesion("quintas")
    ag.atender(s, "plano", [{"mime": "image/jpeg", "nombre": "p.jpg", "datos": b"x"}])
    out = json.loads(ag.atender(s, "unir")[0]["texto"])
    assert out["lindero_comun_m"] == pytest.approx(105.54, abs=0.3)
    assert [p["origen"] for p in s.partes] == ["Lote A", "Lote B"] and s.lindero

    out = json.loads(ag.atender(s, "quintas")[0]["texto"])
    assert out["ficha"]["lotes_por_lado"] == [4, 5] and out["falta_preguntar"] == []
    out = json.loads(ag.atender(s, "lotes")[0]["texto"])
    assert out["distribucion"] == "calle_central" and 8 <= out["lotes"] <= 10
    assert out["calle_dentro_de"] == "Lote B"  # el terreno más grande
    assert {pl["terreno"] for pl in out["por_lado"]} == {"Lote A", "Lote B"}
    assert all(d["terreno"] in ("Lote A", "Lote B") for d in out["lotes_detalle"])
    estados = {f["regla"]: f for f in out["cumplimiento"]}
    assert estados["Calles sin salida"]["proyecto"].startswith("redondel de")
    assert estados["Protección de quebrada (sur)"]["estado"] == "cumple"

    salidas = ag.atender(s, "lamina")
    pdf = next(o["ruta"] for o in salidas if o.get("nombre", "").endswith("carpeta-prefactibilidad.pdf"))
    assert "CUADRO DE CUMPLIMIENTO" in " ".join(p.extract_text() for p in PdfReader(pdf).pages)


def test_calle_central_con_relieve_y_3d(tmp_path, caso_chiltiupan):
    from motor import modelo3d

    t, m, _ = caso_chiltiupan
    P = Parametros.de_dict(dict(frente=20, fondo=35, area_minima=600, lado_acceso="norte"))
    r = lotificar_calle_central(t.puntos, P, m, [], lotes_por_lado=(4, 6))
    assert r.stats["lotes"] >= 6 and r.stats["pendiente_calles_pct"] > 0
    assert all(l["pendiente"] is not None for l in r.lotes)
    escena = modelo3d.construir(t, m, r)
    modelo3d.dxf(str(tmp_path / "q.dxf"), escena)
    assert (tmp_path / "q.dxf").stat().st_size > 10000


def test_calle_dentro_del_terreno_mas_grande(union):
    a, b2, u = union
    P = Parametros.de_dict(dict(frente=20, fondo=35, area_minima=630, lado_acceso="norte"))
    franja = franja_de_proteccion(u.puntos, "sur", 12.0)
    r = lotificar_calle_central(u.puntos, P, None, [franja], eje=lindero_comun(a.puntos, b2.puntos), lotes_por_lado=(4, 5),
                                partes=[{"origen": "A", "anillo": a.puntos}, {"origen": "B", "anillo": b2.puntos}],
                                calle_sobre="mayor")
    calle = r.calles[0]
    assert r.stats["calle_dentro_de"] == "B"
    assert calle.intersection(Polygon(a.puntos)).area < 1.0  # ni la calle ni el redondel pisan el terreno A
    lados = {pl["terreno"]: pl for pl in r.stats["por_lado"]}
    assert lados["B"]["lotes"] == 5 and 4 <= lados["A"]["lotes"] <= 5


def test_reparto_parejo_por_lado():
    """La tierra detrás del área verde no se queda toda en el último lote: se reparte entre los de su lado girando
    los cortes, sin cambiar el frente de cada lote a la calle."""
    ag = Agente(GuionQuintas(), "/tmp/lotes_reparto")
    s = Sesion("reparto")
    ag.atender(s, "plano", [{"mime": "image/jpeg", "nombre": "p.jpg", "datos": b"x"}])
    ag.atender(s, "unir")
    ag.atender(s, "quintas")
    ag.atender(s, "lotes")
    lot, _ = ag._lotificar(s)
    fmin = lot.parametros["frente_minimo"]
    for pl in lot.stats["por_lado"]:
        areas = [lot.lotes[i - 1]["area"] for i in pl["ids"]]
        assert max(areas) <= 1.03 * min(areas), (pl["terreno"], areas)
        assert all(lot.lotes[i - 1]["frente"] >= fmin - 0.01 for i in pl["ids"])
    # (con la franja de la quebrada completa ya no sobra tierra detrás del área verde: los lotes salen parejos
    # sin necesidad de girar los cortes, así que "reparto" puede quedar normal)
    # los lotes no se enciman entre sí
    polys = [l["poly"] for l in lot.lotes]
    for i, a in enumerate(polys):
        for b in polys[i + 1:]:
            assert a.intersection(b).area < 0.5


def test_franja_de_quebrada_cubre_toda_la_orilla(tmp_path):
    """Error que encontró el usuario: la franja solo cubría el tramo más grande de la orilla y el lote 9 quedaba pegado
    a la quebrada sin protección. Ningún lote puede quedar a menos de 1.5 × profundidad de la orilla."""
    from shapely.geometry import Polygon

    from motor.lotificacion import orilla_hacia

    ag = Agente(GuionQuintas(), str(tmp_path))
    s = Sesion("f")
    ag.atender(s, "plano", [{"mime": "image/jpeg", "nombre": "p.jpg", "datos": b"x"}])
    ag.atender(s, "unir")
    ag.atender(s, "quintas")
    lot, _ = ag._lotificar(s)
    from motor.lotificacion import franja_de_proteccion

    q = s.ficha["quebradas"][0]
    poly = Polygon(s.get_terreno().puntos).buffer(0)
    orilla = orilla_hacia(poly, q["lado"])
    franja = franja_de_proteccion(s.get_terreno().puntos, q["lado"], 1.5 * q["profundidad_m"])
    assert orilla.within(franja.buffer(0.5))  # la franja va a lo largo de toda la orilla, no solo del tramo mayor
    for l in lot.lotes:
        assert l["poly"].intersection(franja).area < 0.5, f"el lote {l['id']} invade la franja"
