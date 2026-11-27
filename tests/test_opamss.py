"""Geovisor de la OPAMSS (las capas del Esquema Director que usa NIX), con el servidor simulado."""
import json
from urllib.parse import parse_qs, urlparse

import pytest
from shapely.geometry import Polygon

from motor import normas, proyeccion
from servicios import opamss

pytestmark = pytest.mark.geovisor

# terreno de 100 × 60 m cerca del lote A (Lambert)
X0, Y0 = 473200.0, 289000.0
ANILLO = [(X0, Y0), (X0 + 100, Y0), (X0 + 100, Y0 + 60), (X0, Y0 + 60)]


def _esri(poly_lambert: Polygon) -> dict:
    """Polígono Lambert → ESRI JSON en WGS84 (exterior en sentido horario)."""
    pts = [list(reversed(proyeccion.a_latlng(x, y))) for x, y in poly_lambert.exterior.coords]
    if opamss._area_firmada([tuple(p) for p in pts[:-1]]) > 0:
        pts = pts[::-1]
    return {"rings": [pts]}


def _linea(coords) -> dict:
    return {"paths": [[list(reversed(proyeccion.a_latlng(x, y))) for x, y in coords]]}


class ServidorFalso:
    """Responde como el MapServer de la OPAMSS según la capa pedida."""

    def __init__(self, respuestas: dict, falla: set | None = None):
        self.respuestas, self.falla, self.pedidos = respuestas, falla or set(), []

    def get(self, url, params=None, timeout=None):
        capa = int(urlparse(url).path.rstrip("/").split("/")[-2])
        self.pedidos.append((capa, params))
        if capa in self.falla:
            raise ConnectionError("sin respuesta")
        d = self.respuestas.get(capa, {"features": []})
        if params.get("returnCountOnly") == "true":
            d = {"count": len(d.get("features", []))}

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return d
        return R()


def test_consulta_completa_y_porcentajes():
    via_proy = _linea([(X0 + 50, Y0 - 100), (X0 + 50, Y0 + 200)])  # cruza el terreno de sur a norte (60 m)
    via_mayor = _linea([(X0 - 100, Y0 + 120), (X0 + 300, Y0 + 120)])  # 60 m al norte
    srv = ServidorFalso({
        opamss.USO_SUELO: {"features": [
            {"attributes": {"cod_clas": "URB", "des_clas": "Suelo urbano", "des_tra": "Consolidacion (CON)",
                            "alt_bas": 3, "alt_amp": 5, "imp_base": 70, "imp_amp": 80, "ien_base": 1.5, "ien_amplia": 2.0},
             "geometry": _esri(Polygon([(X0 - 10, Y0 - 10), (X0 + 75, Y0 - 10), (X0 + 75, Y0 + 70), (X0 - 10, Y0 + 70)]))},
            {"attributes": {"cod_clas": "NUR", "des_clas": "Suelo no urbanizable", "des_tra": "Conservacion ambiental (CAM)",
                            "alt_bas": 1, "imp_base": 10, "ien_base": 0.1},
             "geometry": _esri(Polygon([(X0 + 75, Y0 - 10), (X0 + 110, Y0 - 10), (X0 + 110, Y0 + 70), (X0 + 75, Y0 + 70)]))},
        ]},
        opamss.VIA_PROYECTADA: {"features": [{"attributes": {}, "geometry": via_proy}]},
        opamss.VIA_MAYOR: {"features": [{"attributes": {}, "geometry": via_mayor}]},
        opamss.TECTONICA: {"features": [{"attributes": {}}]},
        opamss.PELIGROSIDAD: {"features": [{"attributes": {"problemati": "Alta"}, "geometry": _esri(Polygon(ANILLO))}]},
    })
    op = opamss.consultar(ANILLO, http=srv)
    assert op and op["fuente"].startswith("OPAMSS")
    zonas = {z["codigo"]: z for z in op["uso_suelo"]}
    assert zonas["URB"]["pct"] == pytest.approx(75, abs=1) and zonas["NUR"]["pct"] == pytest.approx(25, abs=1)
    assert op["uso_suelo"][0]["codigo"] == "URB"  # la de más área primero
    assert op["vias"]["proyectada"]["cruza_m"] == pytest.approx(60, abs=1)
    assert op["vias"]["mayor"]["distancia_m"] == pytest.approx(60, abs=1)
    assert op["vias"]["menor"] is None
    assert op["zona_tectonica"] is True and op["peligro_volcanico"] is False
    assert op["peligrosidad_superficial"][0] == {"valor": "Alta", "pct": pytest.approx(100, abs=1)}
    # se consulta en WGS84 y con el polígono del terreno
    capa, p = next(x for x in srv.pedidos if x[0] == opamss.USO_SUELO)
    assert p["inSR"] == 4326 and json.loads(p["geometry"])["rings"]

    # el cuadro de normas
    from types import SimpleNamespace
    lot = SimpleNamespace(lotes=[{"poly": Polygon([(X0, Y0), (X0 + 25, Y0), (X0 + 25, Y0 + 24), (X0, Y0 + 24)]),
                                  "area": 600.0, "frente": 25.0}],
                          calles=[Polygon([(X0, Y0 - 8), (X0 + 100, Y0 - 8), (X0 + 100, Y0), (X0, Y0)])], angulo=0.0)
    filas = {f["regla"]: f for f in normas.filas_opamss(lot, op)}
    assert filas["Clasificación del suelo"]["estado"] == "no_cumple"  # 25 % no urbanizable
    assert filas["Tratamiento urbanístico"]["estado"] == "revisar"   # conservación ambiental
    assert filas["Altura máxima"]["requerido"].startswith("3 niveles")
    assert filas["Impermeabilización por lote"]["estado"] == "cumple"
    assert filas["Vía proyectada"]["estado"] == "revisar"
    assert "tectónica" in filas["Peligros del lugar"]["proyecto"] and "alta" in filas["Peligros del lugar"]["proyecto"]
    assert all(f["fuente"] == op["fuente"] for f in filas.values())


def test_si_el_servidor_no_responde_no_se_inventa():
    todas = {opamss.USO_SUELO, opamss.VIA_MAYOR, opamss.VIA_MENOR, opamss.VIA_PROYECTADA, opamss.TECTONICA,
             opamss.VOLCANICO, opamss.RECARGA, opamss.PELIGROSIDAD}
    assert opamss.consultar(ANILLO, http=ServidorFalso({}, falla=todas)) is None
    op = opamss.consultar(ANILLO, http=ServidorFalso({}, falla={opamss.USO_SUELO}))
    assert op["no_se_pudo_consultar"] == ["uso_suelo"] and "uso_suelo" not in op
    filas = normas.filas_opamss(type("L", (), {"lotes": [], "calles": [], "angulo": 0.0})(), op)
    assert any(f["estado"] == "falta_dato" and "uso suelo" in f["proyecto"] for f in filas)
    assert not any(f["regla"] == "Clasificación del suelo" for f in filas)


def test_suelo_urbano_sin_peligros_cumple():
    srv = ServidorFalso({opamss.USO_SUELO: {"features": [
        {"attributes": {"cod_clas": "URB", "des_clas": "Suelo urbano", "des_tra": "Consolidacion (CON)",
                        "alt_bas": 4, "imp_base": 70, "ien_base": 2.0},
         "geometry": _esri(Polygon([(X0 - 10, Y0 - 10), (X0 + 110, Y0 - 10), (X0 + 110, Y0 + 70), (X0 - 10, Y0 + 70)]))}]}})
    op = opamss.consultar(ANILLO, http=srv)
    filas = {f["regla"]: f for f in normas.filas_opamss(type("L", (), {"lotes": [], "calles": [], "angulo": 0.0})(), op)}
    assert filas["Clasificación del suelo"]["estado"] == "cumple"
    assert filas["Tratamiento urbanístico"]["estado"] == "cumple"
    assert filas["Peligros del lugar"]["estado"] == "cumple"
    assert "Vía proyectada" not in filas


def test_el_bot_lo_consulta_en_el_amss_y_lo_lleva_al_cuadro(tmp_path, monkeypatch):
    """En el AMSS, al sacar el relieve, el bot consulta la OPAMSS y lo pasa a la ficha y al cuadro de normas."""
    import test_agente as TA
    from agente import Agente, Sesion
    from motor import relieve

    m = relieve.Malla.de_dict(json.loads(TA._texto("chiltiupan_malla.json")))
    monkeypatch.setattr(relieve, "descargar", lambda anillo, paso=5.0, http=None: m)
    op = {"fuente": opamss.FUENTE, "uso_suelo": [{"codigo": "RUR", "clasificacion": "Suelo rural",
                                                  "tratamiento": "Desarrollo agropecuario (DAG)", "pct": 100.0,
                                                  "altura_base": 2, "impermeabilizacion_base": 30,
                                                  "edificabilidad_base": 0.3}],
          "vias": {"mayor": None, "menor": {"distancia_m": 12.0, "cruza_m": 0.0}, "proyectada": None},
          "zona_tectonica": False, "peligro_volcanico": False, "peligrosidad_superficial": []}
    monkeypatch.setattr(opamss, "consultar", lambda anillo, http=None: op)
    ag = Agente(TA.Guion(TA._lectura_chiltiupan()), str(tmp_path))
    s = Sesion("op")
    ag.atender(s, "este es el plano", [{"mime": "image/jpeg", "nombre": "plano.jpg", "datos": b"foto-del-plano"}])
    r = ag.h_obtener_relieve(s)
    enc = r["encontrado"]["opamss"]
    assert enc["clasificacion_del_suelo"][0].startswith("100 % Suelo rural")
    assert "impermeabilizar hasta el 30 %" in " ".join(enc["limites"])
    ag.h_guardar_ficha(s, zona="amss", tipo="campestre", lote_objetivo_m2=600, acceso_lado="oriente", sin_quebradas=True,
                       cerrado=True, figura="lotificacion", agua="anda", luz="si")
    out = ag.h_predisenar(s)
    reglas = {f["regla"]: f["estado"] for f in out["cumplimiento"]}
    assert reglas["Clasificación del suelo"] == "revisar"     # suelo rural: calificación de lugar
    assert reglas["Tratamiento urbanístico"] == "revisar"     # desarrollo agropecuario
    assert "Edificabilidad (IEN)" in reglas
    s.set_terreno(s.get_terreno())                            # reubicar borra lo consultado
    assert s.opamss is None
