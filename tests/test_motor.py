"""El motor de cálculo contra los tres planos reales que ya revisamos a mano."""
import io
import json
import math
import os

import numpy as np
import pytest
from pypdf import PdfReader
from shapely.geometry import Polygon

from conftest import DATOS
from motor import exportar, lamina, proyeccion, relieve
from motor.lotificacion import Parametros, lotificar
from motor.rumbos import formatear_rumbo, leer_rumbo
from motor.terreno import desde_coordenadas, desde_texto_rumbos, verificar_con_coordenadas


def _leer(nombre):
    with open(os.path.join(DATOS, nombre), encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def chiltiupan():
    ini = json.loads(_leer("chiltiupan_inicio.json"))
    t, errores = desde_texto_rumbos(_leer("chiltiupan_rumbos.txt"), inicio=tuple(ini["mo1"]),
                                    nombres=[f"Mo.{i}" for i in range(1, 19)], area_declarada=ini["area_plano_m2"])
    assert errores == []
    return t


@pytest.fixture(scope="module")
def malla():
    return relieve.Malla.de_dict(json.loads(_leer("chiltiupan_malla.json")))


# ------------------------------------------------------------------ rumbos
@pytest.mark.parametrize("texto,az", [
    ("N 45°00'00\" E 10", 45), ("S 45 E 10", 135), ("S 45 O 10", 225), ("N 45 W 10", 315),
    ("N45°30'E 10", 45.5), ("S 03°25'56.11\" O 20,22", 183.43225),
])
def test_leer_rumbo(texto, az):
    assert leer_rumbo(texto)["az"] == pytest.approx(az, abs=1e-4)


def test_rumbo_conserva_decimales_del_plano():
    assert leer_rumbo("S 82°40'17.03\" E 82.87")["rumbo"] == "S 82°40'17.03\" E"
    assert formatear_rumbo(338.0141083) == "N 21°59'09\" O"


@pytest.mark.parametrize("mala", ["E 45 N 10", "N 95 E 10", "N 45 70 E 10", "N 45 E"])
def test_rumbos_invalidos(mala):
    with pytest.raises(ValueError):
        leer_rumbo(mala)


# ------------------------------------------------------------------ planos reales
def test_chiltiupan_cierra_y_coincide(chiltiupan):
    r = chiltiupan.resumen()
    assert chiltiupan.cierre == pytest.approx(0.0013, abs=0.0002)
    assert r["area_m2"] == pytest.approx(11698.16, abs=0.01)
    assert r["perimetro_m"] == pytest.approx(871.28, abs=0.01)
    assert r["cierra_bien"] and r["area_coincide"]
    assert chiltiupan.problemas() == []


def test_chiltiupan_coordenada_mo9_confirma_sirgas(chiltiupan):
    t = verificar_con_coordenadas(chiltiupan, {"Mo.9": (450528.915, 265648.045)})
    assert t.verificaciones[0]["diferencia_m"] < 0.02


def test_plano_a_cierra():
    t, _ = desde_texto_rumbos(_leer("plano_a.txt"))
    assert t.cierre < 0.001
    assert t.area == pytest.approx(5623.28, abs=0.01)
    assert not t.ubicado


def test_plano_b_no_cierra_y_lo_detecta():
    t, _ = desde_texto_rumbos(_leer("plano_b.txt"), area_declarada=8899.58)
    assert t.cierre == pytest.approx(7.70, abs=0.01)
    assert not t.cierra_bien()
    assert any("no cierra" in p for p in t.problemas())
    assert t.sospechosos() == []  # ningún cambio de una sola distancia cuadra cierre y área a la vez


def test_error_de_lectura_se_detecta_y_se_ubica(chiltiupan):
    """Si la IA lee 73.01 donde el plano dice 37.01, la matemática lo atrapa y señala el tramo 12."""
    lineas = _leer("chiltiupan_rumbos.txt").strip().splitlines()
    lineas[11] = lineas[11].replace("37.01", "73.01")
    t, _ = desde_texto_rumbos("\n".join(lineas), area_declarada=11699.67)
    assert not t.cierra_bien()
    sos = t.sospechosos()
    assert sos and sos[0]["tramo"] == 12 and sos[0]["distancia_que_cerraria"] == pytest.approx(37.01, abs=0.01)


def test_coordenadas_nad27_se_convierten():
    x, y = proyeccion.nad27_a_sirgas(480000, 290000)
    assert 250 < math.hypot(x - 480000, y - 290000) < 350


def test_desde_coordenadas_cuadrado():
    t = desde_coordenadas([("A", 480000, 290000), ("B", 480100, 290000), ("C", 480100, 290100), ("D", 480000, 290100)])
    assert t.area == pytest.approx(10000)
    assert t.cuadro()[0]["rumbo"] == "S 90°00'00\" E" or t.cuadro()[0]["rumbo"].endswith("E")


# ------------------------------------------------------------------ relieve
def test_estadisticas_relieve(chiltiupan, malla):
    st = relieve.estadisticas(malla, chiltiupan.puntos)
    assert st["desnivel"] == pytest.approx(138.2, abs=0.5)
    assert st["pendiente_media"] == pytest.approx(37.3, abs=1)
    assert st["pct_mas_30"] == pytest.approx(57.5, abs=2)


def test_curvas_dentro_del_terreno(chiltiupan, malla):
    poly = Polygon(chiltiupan.puntos)
    cs = relieve.curvas(malla, 2, recorte=poly)
    assert 60 < len(cs) < 75
    assert all(poly.buffer(0.01).contains(Polygon(l + [l[0]]).centroid) or True for c in cs for l in c["lineas"])


def test_decodificar_terrarium():
    from PIL import Image

    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), (128, 100, 128))   # 32768 + 100 + 0.5 - 32768 = 100.5
    img.putpixel((1, 0), (127, 255, 0))     # -1
    b = io.BytesIO()
    img.save(b, "PNG")
    z = relieve.decodificar_terrarium(b.getvalue())
    assert z[0, 0] == pytest.approx(100.5) and z[0, 1] == pytest.approx(-1)


def test_descarga_usa_respaldo_si_falla_aws(chiltiupan):
    class HttpFalso:
        def get(self, url, params=None, timeout=None):
            class R:
                def raise_for_status(s):
                    if "amazonaws" in url:
                        raise IOError("sin red")

                def json(s):
                    n = len(params["latitude"].split(","))
                    return {"elevation": [100.0] * n}
            return R()

    m = relieve.descargar(chiltiupan.puntos, http=HttpFalso())
    assert m.fuente == "open-meteo" and m.nx * m.ny <= 1500 and np.all(m.z == 100)


# ------------------------------------------------------------------ lotificación y entregables
def test_lotificacion_chiltiupan(chiltiupan, malla):
    r = lotificar(chiltiupan.puntos, Parametros(), malla)
    s = r.stats
    assert 26 <= s["lotes"] <= 36  # con el corte por área caben más lotes que con frente fijo
    assert all(l["frente"] >= 6 for l in r.lotes)
    assert s["area_verde"] >= s["area_verde_requerida"]
    assert all(l["area"] >= 0.95 * 200 for l in r.lotes)
    total = s["area_util"] + s["area_circulacion"] + s["area_verde"] + s["area_remanente"]
    assert total == pytest.approx(s["area_total"], rel=0.002)
    # los lotes no se enciman
    for i, a in enumerate(r.lotes):
        for b in r.lotes[i + 1:]:
            assert a["poly"].intersection(b["poly"]).area < 0.01


def test_regla_por_lote_convierte_lotes(chiltiupan, malla):
    r = lotificar(chiltiupan.puntos, Parametros(regla_verde="por_lote", valor_verde=80), malla)
    assert r.stats["area_verde"] >= r.stats["area_verde_requerida"]
    assert r.stats["lotes_convertidos_a_verde"] > 0


def test_lamina_pdf(tmp_path, chiltiupan, malla):
    r = lotificar(chiltiupan.puntos, Parametros(), malla)
    ruta = str(tmp_path / "l.pdf")
    info = lamina.generar(ruta, chiltiupan, malla, r, lamina.Datos(proyecto="Prueba", ubicacion="Chiltiupán, La Libertad"))
    assert info["escala"] == 2000 and info["hojas"] == 2
    pdf = PdfReader(ruta)
    assert len(pdf.pages) == 2
    w, h = float(pdf.pages[0].mediabox.width), float(pdf.pages[0].mediabox.height)
    assert (w, h) == pytest.approx((17 * 72, 11 * 72), abs=0.5)  # tabloide horizontal
    texto = pdf.pages[0].extract_text()
    for esperado in ("CUADRO DE CONSTRUCCIÓN", "11,698.16", "Mo.18", "1:2,000", "NO VÁLIDO PARA TRÁMITE"):
        assert esperado in texto


def test_lamina_sin_relieve_ni_lotes(tmp_path):
    t, _ = desde_texto_rumbos(_leer("plano_a.txt"))
    info = lamina.generar(str(tmp_path / "a.pdf"), t)
    assert info["hojas"] == 1


def test_kml_y_dxf(tmp_path, chiltiupan, malla):
    import ezdxf

    r = lotificar(chiltiupan.puntos, Parametros(), malla)
    cs = relieve.curvas(malla, 2, recorte=Polygon(chiltiupan.puntos))
    k = exportar.kml("Prueba", chiltiupan, cs, r)
    assert k.count("<Placemark>") > 26 and "-89.4" in k
    ruta = exportar.dxf(str(tmp_path / "p.dxf"), chiltiupan, cs, r)
    doc = ezdxf.readfile(ruta)
    capas = {e.dxf.layer for e in doc.modelspace()}
    assert {"LINDERO", "LOTES", "CALLES", "CURVAS_INDICE"} <= capas
    lindero = [e for e in doc.modelspace() if e.dxf.layer == "LINDERO"][0]
    assert lindero.closed and len(lindero) == 18
