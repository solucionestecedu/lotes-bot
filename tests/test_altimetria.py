"""Altimetría: barrancos fuera de los lotes, relieve por lote, plano altimétrico, guía del render y resumen ejecutivo."""
import math

import numpy as np
import pytest
from pypdf import PdfReader
from shapely.geometry import LineString, Polygon

from motor import altimetria
from motor.calle_central import lotificar_calle_central
from motor.lotificacion import Parametros
from motor.relieve import Malla, malla_para

RECT = [(0, 0), (120, 0), (120, 100), (0, 100)]


def _ladera(anillo, caida_borde=True):
    """Baja 10 % hacia el sur y, pegado al lindero sur, un barranco de ~9 m."""
    x0, y0, dx, nx, ny = malla_para(anillo, 5.0, 30.0)
    z = np.empty((ny, nx))
    for j in range(ny):
        for i in range(nx):
            x, y = x0 + i * dx, y0 + j * dx
            z[j, i] = 500 + 0.10 * y - (9.0 / (1 + math.exp((y - 8) / 2.0)) if caida_borde else 0)
    return Malla(x0, y0, dx, nx, ny, z, "archivo")


def test_barranco_del_borde_sale_y_la_ladera_no():
    m = _ladera(RECT)
    b = altimetria.barrancos(m, RECT)
    assert b and all(q.bounds[1] < 20 for q in b)  # solo la franja del sur
    assert sum(q.area for q in b) < 0.2 * 12000
    assert altimetria.barrancos(_ladera(RECT, caida_borde=False), RECT) == []  # 10 % parejo: nada que quitar


def test_no_quita_caidas_que_parten_el_terreno():
    m = _ladera(RECT)
    # una pared empinada de norte a sur por la mitad no es barranco de borde que se pueda quitar sin partir el terreno
    x0, y0, dx, nx, ny = m.x0, m.y0, m.dx, m.nx, m.ny
    z = np.array([[500 + 0.6 * max(0, min(20, x0 + i * dx - 50)) for i in range(nx)] for j in range(ny)])
    pared = Malla(x0, y0, dx, nx, ny, z, "archivo")
    for q in altimetria.barrancos(pared, RECT):
        resto = Polygon(RECT).difference(q)
        assert resto.geom_type == "Polygon"


def test_relieve_por_lote_y_resumen():
    m = _ladera(RECT, caida_borde=False)
    r = altimetria.por_lote(m, Polygon([(0, 50), (30, 50), (30, 90), (0, 90)]))
    assert r["pend_media"] == pytest.approx(10, abs=0.5) and r["cae_hacia"] == "sur"
    assert r["desnivel"] == pytest.approx(4.0, abs=0.6) and r["construible_pct"] == 100
    res = altimetria.resumen(m, RECT)
    assert res["baja_hacia"] == "sur" and res["desnivel"] == pytest.approx(10, abs=1)
    assert sum(c["pct"] for c in res["clases"]) == pytest.approx(100, abs=0.5)
    largo, media, peor = altimetria.pendiente_eje(m, LineString([(60, 95), (60, 15)]))
    assert largo == 80 and media == pytest.approx(10, abs=0.3) and peor == pytest.approx(10, abs=0.5)


def test_guia_del_render_dice_hacia_donde_baja():
    m = _ladera(RECT)
    g = altimetria.guia_render(m, RECT, lambda x, y: (x * 10, (100 - y) * 6), 1200, 600, altimetria.barrancos(m, RECT))
    assert "NOT flat" in g and "top" in g and "bottom" in g and "steep drop" in g
    plano = Malla(m.x0, m.y0, m.dx, m.nx, m.ny, np.full_like(m.z, 500.0), "archivo")
    assert "almost flat" in altimetria.guia_render(plano, RECT, lambda x, y: (x, y), 100, 100)


def test_lamina_altimetrica_y_lotes_con_relieve():
    from motor.plano_altimetria import generar_altimetria
    from motor.terreno import Terreno

    m = _ladera(RECT)
    t = Terreno(nombres=[f"V{i}" for i in range(4)], puntos=RECT)
    exc = altimetria.barrancos(m, RECT)
    lot = lotificar_calle_central(RECT, Parametros.de_dict(dict(frente=20, fondo=40, area_minima=700, lado_acceso="norte")),
                                  m, exc, lotes_por_lado=(2, 3))
    altimetria.enriquecer(lot, m)
    assert all(l["relieve"]["cae_hacia"] == "sur" for l in lot.lotes)
    assert all(not l["poly"].intersects(exc[0].buffer(-0.5)) for l in lot.lotes)  # ningún lote en el barranco
    img = generar_altimetria(t, lot, m, {}, "Prueba", None, "")
    assert img.size == (2048, 1448)


def test_resumen_ejecutivo_cuadra_por_terreno_y_avisa_el_plano_que_no_cierra(tmp_path):
    import json

    from agente import Agente, Sesion
    from motor.resumen import cuadro_areas, generar_resumen
    from test_calle_central import GuionQuintas

    ag = Agente(GuionQuintas(), str(tmp_path))
    s = Sesion("r")
    ag.atender(s, "plano", [{"mime": "image/jpeg", "nombre": "p.jpg", "datos": b"x"}])
    ag.atender(s, "unir")
    assert s.partes[1]["cierre"] == pytest.approx(7.70, abs=0.01) and s.partes[1]["area_cuadro"] == pytest.approx(8028.94)
    s.partes[1]["area_declarada"] = 6220.08  # lo que dice el plano del terreno B
    ag.atender(s, "quintas")
    json.loads(ag.atender(s, "lotes")[0]["texto"])
    t = s.get_terreno()
    lot, filas = ag._lotificar(s)
    cx, cy = t.centroide()
    partes = [{**p, "anillo": [(x + cx, y + cy) for x, y in p["anillo"]]} for p in s.partes]
    ca = cuadro_areas(t, lot, partes)
    assert ca["terrenos"] == ["Lote A", "Lote B"]
    usos = dict(ca["filas"])
    assert usos["Calle y redondel"][0] < 1.0  # la calle va completa en B
    for i, b in enumerate(ca["bruta"]):  # cada columna cuadra con su terreno
        assert sum(v[i] for v in usos.values()) == pytest.approx(b, abs=0.5)
    r = generar_resumen(str(tmp_path / "r.pdf"), t, lot, None, s.ficha, "Quintas A y B", partes, "", filas)
    assert len(r["avisos"]) == 1 and "Lote B" in r["avisos"][0] and "1,808.86" in r["avisos"][0]
    texto = " ".join(p.extract_text() for p in PdfReader(r["ruta"]).pages)
    assert "Cuadro general de áreas" in texto and "Verificación de los planos" in texto and "Revisar" in texto
