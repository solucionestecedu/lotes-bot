"""Revisor automático: la geometría la revisa el motor con reglas exactas y las láminas un modelo que piensa (aquí,
uno de mentira). Una vista con un problema grave se repinta una vez con la corrección."""
import io

from PIL import Image
from shapely.geometry import Polygon

from motor import presentacion, revisor
from motor.calle_central import lotificar_calle_central
from motor.lotificacion import Parametros, franja_de_proteccion
from motor.terreno import Terreno
from test_normas import caso_chiltiupan  # noqa: F401

RECT = [(0, 0), (120, 0), (120, 100), (0, 100)]
FICHA = {"quebradas": [{"tipo": "quebrada", "lado": "sur", "profundidad_m": 6}]}


def _caso():
    franja = franja_de_proteccion(RECT, "sur", 9.0)
    lot = lotificar_calle_central(RECT, Parametros.de_dict(dict(frente=20, fondo=40, area_minima=600, lado_acceso="norte")),
                                  None, [franja], lotes_por_lado=(2, 3))
    t = Terreno(nombres=[f"V{i}" for i in range(4)], puntos=RECT)
    return t, lot


def _por_clave(ch):
    return {c["chequeo"]: c for c in ch}


def test_prediseno_bueno_pasa_la_revision():
    t, lot = _caso()
    ch = revisor.revisar(t, lot, None, FICHA)
    assert not revisor.fallas(ch), revisor.fallas(ch)
    c = _por_clave(ch)
    assert c["franja_cubre_la_orilla"]["ok"] and c["lotes_fuera_de_la_quebrada"]["ok"] and c["proteccion_respetada"]["ok"]


def test_lote_que_se_mete_a_la_quebrada_es_grave():
    t, lot = _caso()
    # el error del lote 9: un lote que llega hasta la orilla, encima de la franja
    l = lot.lotes[0]
    l["poly"] = l["poly"].union(Polygon([(0, 0), (40, 0), (40, 20), (0, 20)])).buffer(0)
    lot.protecciones = [p.difference(l["poly"]).buffer(0) for p in lot.protecciones]
    c = _por_clave(revisor.revisar(t, lot, None, FICHA))
    assert not c["lotes_fuera_de_la_quebrada"]["ok"] and c["lotes_fuera_de_la_quebrada"]["grave"]
    assert not c["franja_cubre_la_orilla"]["ok"]
    assert str(l["id"]) in c["lotes_fuera_de_la_quebrada"]["detalle"]
    assert revisor.avisos_de_revision(revisor.revisar(t, lot, None, FICHA))


def test_hueco_en_lote_y_muros_altos():
    t, lot = _caso()
    l = lot.lotes[-1]
    c0 = l["poly"].centroid
    l["poly"] = l["poly"].difference(c0.buffer(2.0))
    terr = [{"id": x["id"], "muro_max": 7.5 if x is l else 1.0} for x in lot.lotes]
    c = _por_clave(revisor.revisar(t, lot, None, FICHA, terr))
    assert not c["lotes_sin_huecos"]["ok"] and c["lotes_sin_huecos"]["grave"]
    assert not c["muros"]["ok"] and c["muros"]["grave"]


def test_revisor_de_laminas_ordena_y_filtra(tmp_path):
    ruta = str(tmp_path / "a.jpg")
    Image.new("RGB", (400, 300), (200, 200, 200)).save(ruta)
    vistos = {}

    def falso(partes, instruccion, esquema):
        vistos["partes"], vistos["esquema"] = partes, esquema
        return {"problemas": [{"id": "tecnica", "problema": "texto encimado", "gravedad": "media"},
                              {"id": "otra", "problema": "no existe", "gravedad": "alta"},
                              {"id": "porton", "problema": "hay 6 casas y son 9 lotes", "gravedad": "alta"}]}

    probs = revisor.revisar_laminas(falso, [("tecnica", "Plano técnico", ruta), ("porton", "Vista desde el portón", ruta),
                                            ("lamina", "Lámina PDF", "x.pdf")], "- 9 lotes")
    assert [p["id"] for p in probs] == ["porton", "tecnica"]  # lo grave primero; lo que no es lámina, fuera
    assert vistos["partes"][0][0] == "[tecnica] Plano técnico" and len(vistos["partes"]) == 2
    assert vistos["esquema"]["properties"]["problemas"]["items"]["properties"]["id"]["enum"] == ["tecnica", "porton"]


def test_vista_con_problema_grave_se_repinta_una_vez(tmp_path, caso_chiltiupan):
    t, m, _ = caso_chiltiupan
    lot = lotificar_calle_central(t.puntos, Parametros.de_dict(dict(frente=20, fondo=35, area_minima=600, lado_acceso="norte")),
                                  m, [], lotes_por_lado=(4, 6))
    pedidos = []

    def pintor(png, instruccion, aspecto):
        pedidos.append(instruccion)
        b = io.BytesIO()
        Image.new("RGB", (1600, 900), (90, 140, 80)).save(b, "PNG")
        return b.getvalue()

    vueltas = []

    def falso(partes, instruccion, esquema):
        vueltas.append([p[0] for p in partes])
        if len(vueltas) == 1:
            return {"problemas": [{"id": "porton", "problema": "la calle hace curva", "gravedad": "alta"},
                                  {"id": "tecnica", "problema": "letrero encimado", "gravedad": "media"}]}
        return {"problemas": []}

    out = presentacion.generar(str(tmp_path), "ch", t, lot, m, {"acceso_lado": "norte"}, "Chiltiupán", None, pintor,
                               revisor=falso)
    por = {o["vista"]: o for o in out}
    assert len(vueltas) == 2 and vueltas[1] == ["[porton] Vista desde el portón (ilustrativa)"]
    assert por["porton"].get("repintada") and "revision" not in por["porton"]
    assert por["tecnica"]["revision"][0]["problema"] == "letrero encimado" and not por["tecnica"].get("repintada")
    rep = [p for p in pedidos if "CORRECTIONS" in p]
    assert len(rep) == 1 and "la calle hace curva" in rep[0] and rep[0].startswith(presentacion.INSTRUCCION_PORTON)


def test_si_el_revisor_falla_no_se_reporta_sin_problemas(tmp_path):
    ruta = str(tmp_path / "a.jpg")
    Image.new("RGB", (400, 300)).save(ruta)

    def caido(partes, instruccion, esquema):
        raise RuntimeError("402 sin créditos")

    assert revisor.revisar_laminas(caido, [("tecnica", "Plano", ruta)], "") is None
    assert revisor.revisar_laminas(None, [("tecnica", "Plano", ruta)], "") is None
    assert revisor.revisar_laminas(lambda *a: {"problemas": []}, [("tecnica", "Plano", ruta)], "") == []
