"""Modelo 3D: maqueta del terreno, DXF 3D para AutoCAD y vista aérea."""
import json
import os

import ezdxf
import numpy as np
import pytest
from PIL import Image

from conftest import DATOS
from motor import modelo3d, relieve
from motor.lotificacion import Parametros, lotificar
from motor.terreno import desde_texto_rumbos


@pytest.fixture(scope="module")
def caso():
    with open(os.path.join(DATOS, "chiltiupan_rumbos.txt"), encoding="utf-8") as f:
        rumbos = f.read()
    with open(os.path.join(DATOS, "chiltiupan_inicio.json"), encoding="utf-8") as f:
        ini = json.load(f)["mo1"]
    with open(os.path.join(DATOS, "chiltiupan_malla.json"), encoding="utf-8") as f:
        m = relieve.Malla.de_dict(json.load(f))
    t, _ = desde_texto_rumbos(rumbos, inicio=tuple(ini))
    lot = lotificar(t.puntos, Parametros(), m)
    return t, m, lot, modelo3d.construir(t, m, lot)


def test_escena(caso):
    t, m, lot, e = caso
    assert {"entorno", "lote", "calle", "verde", "muro", "techo", "base"} <= set(e.triangulos)
    assert 0.8 * len(lot.lotes) <= e.casas <= len(lot.lotes)
    assert len(e.triangulos["techo"]) == 4 * e.casas
    assert len(e.lineas["lotes"]) == len(lot.lotes)
    # el terreno sigue la malla: las alturas de la superficie están dentro del rango del relieve
    zs = np.array([p[2] for c in ("entorno", "lote", "calle", "verde") for tri in e.triangulos[c] for p in tri])
    assert np.nanmin(m.z) - 0.01 <= zs.min() and zs.max() <= np.nanmax(m.z) + 0.01


def test_casas_dentro_de_su_lote(caso):
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    _, _, lot, e = caso
    lotes = unary_union([l["poly"] for l in lot.lotes])
    pts = MultiPoint([(p[0], p[1]) for tri in e.triangulos["muro"] for p in tri])
    assert lotes.buffer(0.01).contains(pts)


def test_dxf_3d(tmp_path, caso):
    _, _, _, e = caso
    ruta = modelo3d.dxf(str(tmp_path / "m.dxf"), e)
    doc = ezdxf.readfile(ruta)
    msp = doc.modelspace()
    capas = {x.dxf.layer for x in msp}
    assert {"3D_LOTES", "3D_CALLES", "3D_CASAS_MUROS", "3D_CASAS_TECHOS", "3D_LINDERO", "3D_BASE"} <= capas
    mallas = [x for x in msp if x.dxftype() == "POLYLINE" and x.is_poly_face_mesh]
    assert mallas and all(len(list(x.vertices)) < 32767 for x in mallas)
    techo = next(x for x in mallas if x.dxf.layer == "3D_CASAS_TECHOS")
    assert max(v.dxf.location.z for v in techo.vertices) > 100  # en cotas reales, no en cero


def test_vista(tmp_path, caso):
    t, m, _, e = caso
    ruta = modelo3d.vista(str(tmp_path / "v.jpg"), e, t, "Prueba", "26 lotes", malla=m)
    im = Image.open(ruta)
    assert im.size == (1600, 1000)
    a = np.asarray(im.convert("RGB"), dtype=float)
    centro = a[250:750, 200:1400]
    assert centro.std() > 20  # hay dibujo, no solo cielo
