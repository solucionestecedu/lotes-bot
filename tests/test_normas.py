"""Varios terrenos, unión de colindantes, ficha del proyecto y revisión de normas."""
import json
import os

import pytest
from pypdf import PdfReader

from conftest import DATOS
from agente import Agente, Sesion
from motor import colindancia, normas
from motor.lotificacion import Parametros, franja_de_proteccion, lotificar, punto_de_acceso
from motor.terreno import desde_texto_rumbos
from servicios.llm import LLM, ReglasLLM, Respuesta


def _txt(nombre):
    with open(os.path.join(DATOS, nombre), encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def ab():
    a, _ = desde_texto_rumbos(_txt("plano_a.txt"))
    b, _ = desde_texto_rumbos(_txt("plano_b.txt"))
    return a, b


def test_lindero_comun_por_rumbos_inversos(ab):
    a, b = ab
    comunes = colindancia.tramos_comunes(a, b)
    assert comunes == [{"tramo_a": 2, "tramo_b": 10, "distancia": 23.785}]
    b2, info = colindancia.alinear(a, b)
    assert info["traslape_m2"] < 0.01 and info["lindero_comun_m"] > 100


def test_union_suma_areas(ab):
    a, b = ab
    u = colindancia.unir([a, colindancia.alinear(a, b)[0]])
    assert u.area == pytest.approx(a.area + b.area, abs=0.5)
    assert u.nombres[0] == "U1" and u.cierre == 0


def test_sin_lindero_comun_no_se_unen(ab):
    a, _ = ab
    c, _ = desde_texto_rumbos("N 10 E 50\nS 80 E 50\nS 10 O 50\nN 80 O 50")
    assert colindancia.alinear(a, c) is None


def test_acceso_y_franja_por_lado(ab):
    a, _ = ab
    ys = [p[1] for p in a.puntos]
    norte = punto_de_acceso(a.puntos, "norte")
    sur = punto_de_acceso(a.puntos, "sur")
    assert norte[1] > sur[1]
    f = franja_de_proteccion(a.puntos, "sur", 12)
    assert f is not None and 0 < f.area < a.area and f.centroid.y < (max(ys) + min(ys)) / 2


def test_area_verde_por_zona():
    assert normas.area_verde_requerida("urbano", 10, 7000)[0] == pytest.approx(700)
    assert normas.area_verde_requerida("rural", 10, 7000)[0] == pytest.approx(125)
    assert normas.area_verde_requerida("amss", 10, 7000)[0] == pytest.approx(700)
    req, base, claro = normas.equipamiento_requerido("urbano", 80)
    assert req == 640 and claro
    assert not normas.equipamiento_requerido("urbano", 20)[2]
    assert normas.equipamiento_requerido("amss", 20, "interes_social")[0] == pytest.approx(128)


def test_revision_detecta_incumplimientos(ab):
    a, _ = ab
    r = lotificar(a.puntos, Parametros(regla_verde="pct_util", valor_verde=0.0))  # diseño sin área verde
    r.verdes.clear()
    r.stats["area_verde"] = 0.0
    filas = normas.revisar(r, a, None, {"zona": "urbano"})
    verde = next(f for f in filas if f["regla"] == "Área verde total")
    assert verde["estado"] in ("no_cumple", "revisar")
    assert next(f for f in filas if f["regla"] == "Área verde recreativa")["estado"] == "no_cumple"


def test_sin_zona_falta_dato(ab):
    a, _ = ab
    r = lotificar(a.puntos, Parametros())
    filas = normas.revisar(r, a, None, {})
    assert filas[0]["estado"] == "falta_dato"


def test_pendiente_de_calles(caso_chiltiupan):
    t, m, r = caso_chiltiupan
    filas = normas.revisar(r, t, m, {"zona": "rural"})
    calle = next(f for f in filas if f["regla"] == "Pendiente de calles")
    assert calle["estado"] in ("cumple", "no_cumple") and "RLUC" in calle["ref"]
    assert any(f["regla"] == "Lotes con pendiente > 30 %" for f in filas)


@pytest.fixture(scope="module")
def caso_chiltiupan():
    from motor import relieve

    t, _ = desde_texto_rumbos(_txt("chiltiupan_rumbos.txt"), inicio=tuple(json.loads(_txt("chiltiupan_inicio.json"))["mo1"]))
    m = relieve.Malla.de_dict(json.loads(_txt("chiltiupan_malla.json")))
    return t, m, lotificar(t.puntos, Parametros(regla_verde="por_lote", valor_verde=12.5), m)


# ------------------------------------------------------------------ el agente
def _lectura_ab():
    def terr(nombre, archivo, area, cols):
        tr = []
        for linea in _txt(archivo).strip().splitlines():
            *r, d = linea.split()
            tr.append({"rumbo": " ".join(r), "distancia": float(d)})
        return {"nombre": nombre, "tramos": tr, "coordenadas": [], "area_declarada": area, "unidad_area": "m2",
                "colindancias": cols}
    return {"es_plano": True, "sistema": "desconocido", "terrenos": [
        terr("Lote A", "plano_a.txt", 5623.20, [{"lado": "norte", "tipo": "calle", "descripcion": "calle vecinal"}]),
        terr("Lote B", "plano_b.txt", None, [{"lado": "sur", "tipo": "quebrada"}])]}


class Guion(LLM):
    puede_ver = True

    def leer_imagen(self, datos, mime, instruccion, esquema):
        assert "VARIOS terrenos" in instruccion and "terrenos" in esquema["properties"]
        return _lectura_ab()

    def turno(self, sistema, mensajes, herramientas):
        self.sistema = sistema
        u = mensajes[-1]
        if u["rol"] == "herramienta":
            return Respuesta(texto=json.dumps(u["resultado"], ensure_ascii=False, default=str))
        t = u["texto"]
        if "[adjunto:" in t:
            return Respuesta(llamadas=[{"nombre": "leer_plano", "args": {"adjunto": "a1"}}])
        if t == "unir":
            return Respuesta(llamadas=[{"nombre": "elegir_terreno", "args": {"opcion": "unir"}}])
        if t == "ficha":
            return Respuesta(llamadas=[{"nombre": "guardar_ficha", "args": {
                "zona": "urbano", "tipo": "residencial", "lote_objetivo_m2": 700, "acceso_lado": "norte", "cerrado": True,
                "quebradas": [{"tipo": "quebrada", "lado": "sur", "profundidad_m": 8}], "agua": "anda", "luz": "si"}}])
        if t == "lotes":
            return Respuesta(llamadas=[{"nombre": "predisenar", "args": {}}])
        if t == "lamina":
            return Respuesta(llamadas=[{"nombre": "generar_entregables", "args": {"nombre_proyecto": "San Antonio Abad"}}])
        return Respuesta(texto="ok")


def test_flujo_dos_terrenos_unidos(tmp_path):
    llm = Guion()
    ag = Agente(llm, str(tmp_path))
    s = Sesion("ab")
    out = json.loads(ag.atender(s, "plano", [{"mime": "image/jpeg", "nombre": "p.jpg", "datos": b"x"}])[0]["texto"])
    assert len(out["varios_terrenos"]) == 2 and "colindantes" in out["colindantes"][0]
    assert out["varios_terrenos"][1]["problemas"]  # el B no cierra y se avisa
    assert s.terreno is None and len(s.candidatos) == 2
    assert {"lado": "sur", "tipo": "quebrada"} in s.ficha["del_plano"]

    out = json.loads(ag.atender(s, "unir")[0]["texto"])
    assert out["resumen"]["area_m2"] == pytest.approx(13652.22, abs=0.5)
    assert any("no cierra" in p for p in out["problemas_de_los_originales"])

    ag.atender(s, "lotes")  # sin ficha: diseña igual, pero la revisión pide la zona
    assert s.cumplimiento[0]["estado"] == "falta_dato"
    assert "Falta preguntar" in llm.sistema and "zona" in llm.sistema

    out = json.loads(ag.atender(s, "ficha")[0]["texto"])
    assert out["falta_preguntar"] == []
    out = json.loads(ag.atender(s, "lotes")[0]["texto"])
    assert 10 <= out["lotes"] <= 16 and 600 <= out["lote_promedio_m2"] <= 850
    assert out["area_proteccion_m2"] > 500
    estados = {f["regla"]: f["estado"] for f in out["cumplimiento"]}
    assert estados["Área verde total"] == "cumple" and estados["Protección de quebrada (sur)"] == "cumple"

    salidas = ag.atender(s, "lamina")
    pdf = next(o["ruta"] for o in salidas if o.get("nombre", "").endswith(".pdf"))
    texto = PdfReader(pdf).pages[1].extract_text()
    assert "CUADRO DE CUMPLIMIENTO" in texto and "RLUC 50 a)" in texto and "ZONA DE PROTECCIÓN" in texto


def test_elegir_uno(tmp_path):
    ag = Agente(Guion(), str(tmp_path))
    s = Sesion("uno")
    ag._desde_lectura(s, _lectura_ab())
    r = ag.h_elegir_terreno(s, "1")
    assert r["ok"] and r["resumen"]["area_m2"] == pytest.approx(5623.28, abs=0.01)
    assert ag.h_elegir_terreno(s, "5")["ok"] is False


def test_ficha_rechaza_lado_raro(tmp_path):
    ag = Agente(ReglasLLM(), str(tmp_path))
    assert ag.h_guardar_ficha(Sesion("f"), acceso_lado="arriba")["ok"] is False


def test_equipamiento_obligatorio_en_amss(tmp_path, caso_chiltiupan):
    t, m, _ = caso_chiltiupan
    ag = Agente(ReglasLLM(), str(tmp_path))
    s = Sesion("eq")
    s.set_terreno(t)
    s.malla = m.a_dict()
    s.ficha = {"zona": "amss", "tipo": "interes_social"}
    s.lotes_param = {}
    r, filas = ag._lotificar(s)
    eq = next(f for f in filas if f["regla"] == "Equipamiento social")
    assert eq["estado"] == "cumple" and r.stats["area_equipamiento"] >= 6.4 * r.stats["lotes"]


def test_segunda_foto_se_suma(tmp_path):
    """Una foto por lote: la segunda se suma a la primera con agregar=true y se ofrecen unir."""
    lect = _lectura_ab()

    class DosFotos(Guion):
        def leer_imagen(self, datos, mime, instruccion, esquema):
            return {"es_plano": True, "terrenos": [lect["terrenos"][0 if datos == b"A" else 1]]}

    ag = Agente(DosFotos(), str(tmp_path))
    s = Sesion("dos")
    s.adjuntos = {"a1": {"mime": "image/jpeg", "nombre": "lote-a.jpg", "datos": b"A"},
                  "a2": {"mime": "image/jpeg", "nombre": "lote-b.jpg", "datos": b"B"}}
    r1 = ag.h_leer_plano(s, adjuntos=["a1"])
    assert r1["resumen"]["area_m2"] == pytest.approx(5623.28, abs=0.01)
    r2 = ag.h_leer_plano(s, adjuntos=["a2"], agregar=True)
    assert len(r2["varios_terrenos"]) == 2 and r2["varios_terrenos"][0]["ya_estaba"]
    assert "colindantes" in r2["colindantes"][0]
    r3 = ag.h_leer_plano(s, adjuntos=["a1", "a2"])  # las dos fotos de una vez
    assert len(r3["varios_terrenos"]) == 2


def test_todas_las_calles_llegan_a_la_calle_publica(ab):
    a, b = ab
    u = colindancia.unir([a, colindancia.alinear(a, b)[0]])
    P = Parametros(frente=20, fondo=35, area_minima=630, entrada=punto_de_acceso(u.puntos, "norte"), lado_acceso="norte")
    r = lotificar(u.puntos, P, None, excluir=[franja_de_proteccion(u.puntos, "sur", 12)])
    assert r.stats["calles_desconectadas"] == 0 and r.stats["lotes"] >= 12
    filas = normas.revisar(r, u, None, {"zona": "urbano", "cerrado": True})
    if r.stats["accesos"] > 1:
        assert any(f["regla"].startswith("Un solo acceso") and f["estado"] == "revisar" for f in filas)


def test_lotes_cortados_por_area_parecida(ab):
    a, _ = ab
    r = lotificar(a.puntos, Parametros(frente=20, fondo=35, area_minima=630))
    areas = [l["area"] for l in r.lotes]
    assert len(areas) >= 5 and min(areas) >= 0.95 * 630
