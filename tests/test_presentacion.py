"""Imagen de presentación: el motor traza, el modelo de imagen solo pinta, los letreros los pone el motor."""
import io
import json
import math

import numpy as np
import pytest
from PIL import Image, ImageFilter
from shapely.geometry import Point, Polygon
from pypdf import PdfReader

from agente import Agente, Sesion
from motor import presentacion
from test_calle_central import GuionQuintas
from test_normas import caso_chiltiupan  # noqa: F401


@pytest.fixture(scope="module")
def quintas(tmp_path_factory):
    ag = Agente(GuionQuintas(), str(tmp_path_factory.mktemp("q")))
    s = Sesion("q")
    ag.atender(s, "plano", [{"mime": "image/jpeg", "nombre": "p.jpg", "datos": b"x"}])
    ag.atender(s, "unir")
    ag.atender(s, "quintas")
    ag.atender(s, "lotes")
    lot, _ = ag._lotificar(s)
    return ag, s, lot


def _png(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def _repintar(img):
    """Simula un modelo que pinta sin mover nada: cambia colores, mete textura y suaviza."""
    a = np.asarray(img).astype(np.float32)
    lote = np.all(np.abs(a - np.array(presentacion.C["lote"])) < 25, axis=2)
    a[lote] = [120, 170, 90]
    a += np.random.default_rng(1).normal(0, 15, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))


def test_calco_sin_texto_y_con_aspecto_valido(quintas):
    _, s, lot = quintas
    d = presentacion.calco_planta(s.get_terreno(), lot, None, s.ficha)
    assert d.aspecto in presentacion.ASPECTOS and max(d.imagen.size) == presentacion.LADO_LARGO
    w, h = d.imagen.size
    assert abs(w / h - presentacion.ASPECTOS[d.aspecto]) < 0.01
    assert "porton" in d.extra and "boca" in d.extra  # proyecto cerrado: portón dibujado
    assert len(d.extra["casas"]) == len(lot.lotes)    # una casa tipo por lote


def test_verificacion_acepta_repintado_y_rechaza_movido(quintas):
    _, s, lot = quintas
    d = presentacion.calco_planta(s.get_terreno(), lot, None, s.ficha)
    ok, info, _ = presentacion.alinear(d.imagen, _repintar(d.imagen), d.zona)
    assert ok and info["coincidencia"] > 0.4
    movida = d.imagen.transform(d.imagen.size, Image.AFFINE, (1, 0, 40, 0, 1, 25))
    assert not presentacion.alinear(d.imagen, movida, d.zona)[0]
    agrandada = d.imagen.resize((int(d.imagen.width * 1.06), int(d.imagen.height * 1.06))).crop((0, 0, *d.imagen.size))
    assert not presentacion.alinear(d.imagen, agrandada, d.zona)[0]
    assert not presentacion.alinear(d.imagen, Image.new("RGB", d.imagen.size, (90, 140, 80)), d.zona)[0]
    # corrida poquito: se acepta y se acomoda
    ok, info, _ = presentacion.alinear(d.imagen, _repintar(d.imagen).transform(d.imagen.size, Image.AFFINE, (1, 0, 8, 0, 1, -8)), d.zona)
    assert ok and info.get("corrimiento_px")


def test_generar_con_y_sin_pintor(tmp_path, quintas):
    _, s, lot = quintas
    t = s.get_terreno()
    llamadas = []

    def pintor_bueno(png, instruccion, aspecto):
        llamadas.append((aspecto, "text" in instruccion.lower()))
        return _png(_repintar(Image.open(io.BytesIO(png)).convert("RGB")))

    out = presentacion.generar(str(tmp_path), "q", t, lot, None, s.ficha, "Quintas", None, pintor_bueno)
    # sin relieve no hay vista aérea: la planta (pintada) y el plano técnico (sin IA)
    assert [o["vista"] for o in out] == ["planta", "tecnica"] and out[0]["ia"] and llamadas
    assert out[1]["ia"] is False and len(llamadas) == 1
    assert Image.open(out[0]["ruta"]).size == presentacion.calco_planta(t, lot, None, s.ficha).imagen.size

    def pintor_que_inventa(png, instruccion, aspecto):
        return _png(Image.new("RGB", (1024, 1024), (120, 170, 90)))

    out = presentacion.generar(str(tmp_path), "q2", t, lot, None, s.ficha, "Quintas", None, pintor_que_inventa)
    assert not out[0]["ia"] and "calco" in out[0]["motivo"]

    def pintor_que_falla(png, instruccion, aspecto):
        raise TimeoutError("se trabó")

    out = presentacion.generar(str(tmp_path), "q3", t, lot, None, s.ficha, "Quintas", None, pintor_que_falla)
    assert not out[0]["ia"] and "falló" in out[0]["motivo"]
    out = presentacion.generar(str(tmp_path), "q4", t, lot, None, s.ficha, "Quintas", None, None)
    assert not out[0]["ia"]


def test_aerea_con_relieve(tmp_path, caso_chiltiupan):
    from motor.lotificacion import Parametros
    from motor.calle_central import lotificar_calle_central

    t, m, _ = caso_chiltiupan
    lot = lotificar_calle_central(t.puntos, Parametros.de_dict(dict(frente=20, fondo=35, area_minima=600, lado_acceso="norte")),
                                  m, [], lotes_por_lado=(4, 6))
    out = presentacion.generar(str(tmp_path), "ch", t, lot, m, {"acceso_lado": "norte"}, "Chiltiupán", None, None)
    # sin modelo de imagen: la vista 3D exacta del motor en lugar del ambiente, la planta y el técnico
    assert [o["vista"] for o in out] == ["aerea", "planta", "tecnica"]
    assert Image.open(out[0]["ruta"]).size == (presentacion.LADO_LARGO, round(presentacion.LADO_LARGO * 9 / 16))


def test_ambiente_es_libre_y_no_lleva_medidas(tmp_path, caso_chiltiupan):
    from motor.lotificacion import Parametros
    from motor.calle_central import lotificar_calle_central

    t, m, _ = caso_chiltiupan
    lot = lotificar_calle_central(t.puntos, Parametros.de_dict(dict(frente=20, fondo=35, area_minima=600, lado_acceso="norte")),
                                  m, [], lotes_por_lado=(4, 6))
    vistos = []

    def pintor(png, instruccion, aspecto):
        vistos.append(instruccion)
        if instruccion.startswith(presentacion.INSTRUCCION_AMBIENTE):  # la IA cambia todo: igual se acepta
            return _png(Image.new("RGB", (1600, 900), (90, 140, 80)))
        return _png(_repintar(Image.open(io.BytesIO(png)).convert("RGB")))

    out = presentacion.generar(str(tmp_path), "ch", t, lot, m, {"acceso_lado": "norte"}, "Chiltiupán", None, pintor)
    assert [o["vista"] for o in out] == ["ambiente", "planta", "tecnica"]
    assert out[0]["ia"]
    amb = next(v for v in vistos if v.startswith(presentacion.INSTRUCCION_AMBIENTE))
    # le dice dónde va la entrada y cuántas casas van de cada lado (las del motor)
    assert "gated entrance" in amb and f"{len(lot.lotes)} in total" in amb
    # el ambiente solo lleva título y aviso abajo/arriba: el centro de la imagen queda como lo pintó la IA
    img = np.asarray(Image.open(out[0]["ruta"]).convert("RGB")).astype(int)
    h, w, _ = img.shape
    centro = img[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    assert np.abs(centro - np.array([90, 140, 80])).max() < 12


def test_plano_tecnico_es_blanco_y_negro_con_medidas(quintas):
    _, s, lot = quintas
    dib = presentacion.calco_planta(s.get_terreno(), lot, None, s.ficha, estilo="tecnico")
    assert dib.vista == "tecnica"
    final = np.asarray(presentacion.rotular_planta(dib.imagen, dib, lot, s.ficha, "Prueba", estilo="tecnico"))
    # el lote es blanco (casi todo: hay casa, alberca y árboles) y su lindero negro
    l1 = lot.lotes[0]["poly"]
    x0, y0, x1, y1 = l1.bounds
    muestras = [final[round(dib.a_px(x, y)[1]), round(dib.a_px(x, y)[0])].min()
                for x in np.linspace(x0, x1, 25) for y in np.linspace(y0, y1, 25) if l1.buffer(-2).contains(Point(x, y))]
    assert np.median(muestras) > 245
    comun = l1.intersection(lot.lotes[1]["poly"])
    negros = 0
    for fr in (0.2, 0.25, 0.3, 0.75, 0.8):
        q = comun.interpolate(fr, normalized=True)
        x, y = dib.a_px(q.x, q.y)
        negros += int(final[round(y), round(x)].max() < 90)
    assert negros >= 3


def test_entregables_traen_presentacion_y_hoja_en_pdf(tmp_path):
    class G(GuionQuintas):
        pass

    ag = Agente(G(), str(tmp_path))
    s = Sesion("p")
    ag.atender(s, "plano", [{"mime": "image/jpeg", "nombre": "p.jpg", "datos": b"x"}])
    ag.atender(s, "unir")
    ag.atender(s, "quintas")
    ag.atender(s, "lotes")
    out = ag.atender(s, "lamina")
    res = json.loads(out[0]["texto"][len("RESULTADO "):]) if out[0]["texto"].startswith("RESULTADO ") else None
    archivos = [o for o in out if o["tipo"] == "archivo"]
    assert archivos[0]["nombre"].endswith("presentacion-planta.jpg") and archivos[0]["imagen"]
    pdf = next(a["ruta"] for a in archivos if a["nombre"].endswith(".pdf"))
    paginas = PdfReader(pdf).pages
    assert len(paginas) == 3 and "PLANO DE PRESENTACIÓN" in paginas[2].extract_text().upper()
    if res:
        assert res["presentacion"][0]["vista"] == "planta" and res["presentacion"][0]["pintada_con_ia"] is False


def test_rechaza_calle_inventada_sobre_el_area_verde(quintas):
    _, s, lot = quintas
    d = presentacion.calco_planta(s.get_terreno(), lot, None, s.ficha)
    bien = _repintar(d.imagen)
    a = np.asarray(bien).copy()
    zona_verde = np.isin(d.extra["mascara"], [presentacion.modelo3d.ID_CLASE["verde"], presentacion.modelo3d.ID_CLASE["proteccion"]])
    ys, xs = np.nonzero(zona_verde)
    franja = zona_verde & (np.arange(a.shape[0])[:, None] > np.percentile(ys, 40))  # media zona verde hecha "calle"
    a[franja] = [128, 128, 126]
    mala = Image.fromarray(a)
    assert presentacion.clases_ok(presentacion.medir_clases(bien, d.extra["mascara"]))[0]
    ok, motivo = presentacion.clases_ok(presentacion.medir_clases(mala, d.extra["mascara"]))
    assert not ok and "área verde" in motivo
    img, info = presentacion.pintar(d, lambda png, i, a_: _png(mala), presentacion.INSTRUCCION_PLANTA)
    assert not info["ia"] and "calco" in info["motivo"]


def test_porton_en_la_entrada_y_no_en_el_retorno():
    """Caso real: un solo terreno donde el redondel toca el lindero; el portón iba al revés."""
    from motor.calle_central import lotificar_calle_central
    from motor.lotificacion import Parametros, franja_de_proteccion
    from motor.terreno import desde_texto_rumbos
    from shapely.geometry import Point
    from test_normas import _txt

    a, _ = desde_texto_rumbos(_txt("plano_a.txt"))
    P = Parametros.de_dict(dict(frente=20, fondo=35, area_minima=400, lado_acceso="norte"))
    lot = lotificar_calle_central(a.puntos, P, None, [franja_de_proteccion(a.puntos, "sur", 12.0)], lotes_por_lado=(4, 5))
    ficha = {"acceso_lado": "norte", "cerrado": True, "quebradas": [{"lado": "sur", "profundidad_m": 8}]}
    d = presentacion.calco_planta(a, lot, None, ficha)
    ent, ret = Point(lot.stats["entrada_calle"]), Point(lot.stats["retorno_centro"])
    boca, porton = Point(d.extra["boca"]), Point(d.extra["porton"])
    assert boca.distance(ent) < boca.distance(ret)
    assert porton.distance(ent) < porton.distance(ret)


def test_tramos_de_lindero_y_medidas_sin_repetir():
    from shapely.geometry import Point, box as caja

    # rectángulo con puntos de más (alineados y casi pegados): 4 tramos con sus medidas
    r = Polygon([(0, 0), (10, 0), (10.1, 0.0), (20, 0), (20, 30), (20, 30.05), (0, 30)])
    assert sorted(round(t.length, 2) for t in presentacion._tramos(r)) == [20.0, 20.0, 30.0, 30.0]
    # lote junto al redondel: el frente curvo es un solo tramo, no un pedazo por cada segmento del arco
    lote = caja(0, 0, 30, 30).difference(Point(0, 0).buffer(12))
    tramos = presentacion._tramos(lote)
    assert sorted(round(t.length, 1) for t in tramos) == [18.0, 18.0, 18.8, 30.0, 30.0]
    arco = max(tramos, key=lambda t: len(t.coords))
    assert arco.length == pytest.approx(math.pi * 12 / 2, rel=0.01)


def test_planta_lleva_linderos_blancos_y_medidas(quintas):
    _, s, lot = quintas
    dib = presentacion.calco_planta(s.get_terreno(), lot, None, s.ficha)
    verde = Image.new("RGB", dib.imagen.size, (100, 140, 60))  # como si la IA pintara todo de césped
    final = presentacion.rotular_planta(verde, dib, lot, s.ficha, "Prueba", ia=True)
    a = np.asarray(final)
    # un punto a media altura de un lindero entre dos lotes vecinos queda blanco (encima de la pintura)
    l1, l2 = lot.lotes[0]["poly"], lot.lotes[1]["poly"]
    comun = l1.intersection(l2)
    assert comun.length > 5
    blancos = 0
    for fr in (0.2, 0.25, 0.3, 0.75, 0.8):
        q = comun.interpolate(fr, normalized=True)
        x, y = dib.a_px(q.x, q.y)
        blancos += int(a[round(y), round(x)].min() > 200)
    assert blancos >= 3
