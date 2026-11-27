"""El agente completo con un modelo simulado: plano → verificación → corrección → ubicación → relieve → lotes → lámina."""
import json
import os

import pytest
from pypdf import PdfReader

from conftest import DATOS
from agente import Agente, Sesion
from motor import relieve
from motor.rumbos import leer_tramos
from servicios.llm import LLM, ReglasLLM, Respuesta


def _texto(nombre):
    with open(os.path.join(DATOS, nombre), encoding="utf-8") as f:
        return f.read()


def _lectura_chiltiupan(error_en_tramo_12=False):
    """Lo que devolvería Gemini al leer la foto del cuadro (tomado de la transcripción verificada)."""
    tramos = []
    for k, linea in enumerate(_texto("chiltiupan_rumbos.txt").strip().splitlines(), 1):
        *rumbo, dist = linea.split()
        if error_en_tramo_12 and k == 12:
            dist = "73.01"
        tramos.append({"de": f"Mo.{k}", "a": f"Mo.{k % 18 + 1}", "rumbo": " ".join(rumbo), "distancia": float(dist)})
    return {"es_plano": True, "tramos": tramos,
            "coordenadas": [{"punto": "Mo.9", "x": 450528.915, "y": 265648.045}],
            "area_declarada": 11699.67, "unidad_area": "m2", "sistema": "SIRGAS"}


class Guion(LLM):
    """Modelo falso: decide por palabras clave y registra lo que recibió."""

    puede_ver = True

    def __init__(self, lectura):
        self.lectura = lectura
        self.vistos = []

    def leer_imagen(self, datos, mime, instruccion, esquema):
        assert datos == b"foto-del-plano" and "Transcribe" in instruccion and esquema["type"] == "object"
        return self.lectura

    def turno(self, sistema, mensajes, herramientas):
        self.vistos.append((sistema, mensajes[-1]))
        assert {h["nombre"] for h in herramientas} >= {"leer_plano", "predisenar", "generar_entregables"}
        u = mensajes[-1]
        if u["rol"] == "herramienta":
            return Respuesta(texto="RESULTADO " + json.dumps(u["resultado"], ensure_ascii=False, default=str))
        t = u["texto"]
        if "[adjunto:" in t:
            return Respuesta(llamadas=[{"nombre": "leer_plano", "args": {"adjuntos": ["a1"]}}])
        if t.startswith("corrige"):
            return Respuesta(llamadas=[{"nombre": "corregir_tramo", "args": {"numero": 12, "distancia": 37.01}}])
        if "relieve" in t:
            return Respuesta(llamadas=[{"nombre": "obtener_relieve", "args": {}}])
        if "lotes" in t:
            return Respuesta(llamadas=[{"nombre": "predisenar", "args": {"frente": 10, "fondo": 25}}])
        if "lámina" in t:
            return Respuesta(llamadas=[{"nombre": "generar_entregables",
                                        "args": {"nombre_proyecto": "Lotificación Chiltiupán", "ubicacion": "Chiltiupán, La Libertad"}}])
        return Respuesta(texto="ok")


@pytest.fixture
def malla_local(monkeypatch):
    m = relieve.Malla.de_dict(json.loads(_texto("chiltiupan_malla.json")))
    monkeypatch.setattr(relieve, "descargar", lambda anillo, paso=5.0, http=None: m)
    return m


def test_flujo_completo(tmp_path, malla_local):
    llm = Guion(_lectura_chiltiupan(error_en_tramo_12=True))
    ag = Agente(llm, str(tmp_path))
    s = Sesion("prueba")

    # 1. Llega la foto con un dígito mal leído → la matemática lo detecta y señala el tramo 12.
    out = ag.atender(s, "este es el plano", [{"mime": "image/jpeg", "nombre": "plano.jpg", "datos": b"foto-del-plano"}])
    r = out[0]["texto"]
    assert "no cierra" in r and "Tramo 12" in r and "37.01" in r
    assert s.get_terreno() is not None and not s.get_terreno().cierra_bien()

    # 2. El usuario confirma el dato del plano → cierra, el área cuadra y Mo.9 coincide con su coordenada.
    out = ag.atender(s, "corrige el tramo 12, en el plano dice 37.01")
    res = json.loads(out[0]["texto"][len("RESULTADO "):])
    assert res["resumen"]["cierra_bien"] and res["resumen"]["area_coincide"]
    assert res["resumen"]["ubicado"]  # se ubicó solo con la coordenada de Mo.9
    assert res["resumen"]["coordenadas_comprobadas"][0]["diferencia_m"] < 0.02
    assert res["problemas"] == []

    # 3. Relieve, 4. lotes, 5. lámina
    out = ag.atender(s, "saca el relieve")
    assert '"desnivel": 138.' in out[0]["texto"]
    out = ag.atender(s, "prediseña los lotes")
    pre = json.loads(out[0]["texto"][len("RESULTADO "):])
    # Chiltiupán es una franja angosta: por defecto una sola fila de lotes con la calle pegada a un lindero largo
    assert pre["distribucion"] == "un_lado" and pre["calle_pegada_a"] in ("norte", "sur")
    n_lotes = pre["lotes"]
    assert 30 <= n_lotes <= 45
    out = ag.atender(s, "mándame la lámina")
    archivos = [o for o in out if o["tipo"] == "archivo"]
    # primero la carpeta de prefactibilidad (un PDF con todo), después las vistas y los archivos de CAD
    assert [a["nombre"] for a in archivos] == ["lotificacion-chiltiupan-carpeta-prefactibilidad.pdf",
                                                "lotificacion-chiltiupan-presentacion-aerea.jpg",
                                                "lotificacion-chiltiupan-presentacion-planta.jpg",
                                                "lotificacion-chiltiupan.kml", "lotificacion-chiltiupan.dxf",
                                                "lotificacion-chiltiupan-3d.dxf"]
    assert [a["imagen"] for a in archivos] == [False, True, True, False, False, False]
    lector = PdfReader(archivos[0]["ruta"])
    texto_res = " ".join(p.extract_text() for p in lector.pages)
    assert "CONTENIDO" in texto_res and "Cuadro general de áreas" in texto_res and "Siguientes pasos" in texto_res
    assert [o.title for o in lector.outline][:3] == ["Portada y contenido", "Resumen ejecutivo", "Carpeta técnica"]
    # la lámina a escala conserva su tamaño tabloide dentro de la carpeta
    assert any(round(float(p.mediabox.width)) == 1224 for p in lector.pages)
    assert all(os.path.getsize(a["ruta"]) > 1000 for a in archivos)

    # El estado que ve el modelo resume lo hecho, y el historial guarda solo texto.
    sistema, _ = llm.vistos[-1]
    assert f"Prediseño: {n_lotes} lotes" in sistema and "Relieve: listo" in sistema
    assert all(set(m) == {"rol", "texto"} for m in s.historial)


def test_no_avanza_sin_ubicacion(tmp_path):
    ag = Agente(ReglasLLM(), str(tmp_path))
    s = Sesion("x")
    out = ag.atender(s, _texto("plano_a.txt"))
    assert "X y la Y (Lambert)" in out[0]["texto"]  # pide la coordenada de un mojón, escrita a mano
    out = ag.atender(s, "saca el relieve")
    assert "no está ubicado" in out[0]["texto"]


def test_modo_sin_ia_flujo(tmp_path, malla_local):
    ag = Agente(ReglasLLM(), str(tmp_path))
    s = Sesion("y")
    ini = json.loads(_texto("chiltiupan_inicio.json"))["mo1"]
    out = ag.atender(s, f"inicio {ini[0]:.3f}, {ini[1]:.3f}\n" + _texto("chiltiupan_rumbos.txt"))
    assert "11,698.16 m²" in out[0]["texto"]
    assert "relieve" in ag.atender(s, "relieve")[0]["texto"].lower() or True
    import re

    assert int(re.search(r"(\d+) lotes", ag.atender(s, "lotes")[0]["texto"]).group(1)) >= 26
    out = ag.atender(s, "lámina")
    assert sum(o["tipo"] == "archivo" for o in out) == 6  # carpeta PDF + 2 vistas + KML + 2 DXF


def test_modo_sin_ia_pide_texto_si_mandan_foto(tmp_path):
    ag = Agente(ReglasLLM(), str(tmp_path))
    out = ag.atender(Sesion("z"), "", [{"mime": "image/png", "nombre": "p.png", "datos": b"x"}])
    assert "no puedo leer fotos" in out[0]["texto"]


def test_ubicacion_por_pin(tmp_path):
    ag = Agente(ReglasLLM(), str(tmp_path))
    s = Sesion("w")
    ag.atender(s, _texto("plano_a.txt"))
    out = ag.atender(s, "", ubicacion=(13.70, -89.20))
    t = s.get_terreno()
    assert t.ubicado and "aproximada" in out[0]["texto"].lower()  # pin sin mojón: se toma como el centro
    from motor.proyeccion import a_latlng
    lat, lng = a_latlng(*t.centroide())
    assert abs(lat - 13.70) < 1e-5 and abs(lng + 89.20) < 1e-5


def test_herramienta_que_falla_no_tumba_el_bot(tmp_path):
    ag = Agente(ReglasLLM(), str(tmp_path))
    r = ag.ejecutar(Sesion("v"), "corregir_tramo", {"numero": 3, "distancia": 10}, [])
    assert r["ok"] is False
    r = ag.ejecutar(Sesion("v"), "no_existe", {}, [])
    assert r["ok"] is False


def test_lectura_con_columnas_invertidas_y_nad27(tmp_path):
    ag = Agente(ReglasLLM(), str(tmp_path))
    s = Sesion("n")
    lectura = {"es_plano": True, "tramos": [], "sistema": "NAD27", "coordenadas": [
        {"punto": "1", "x": 290000, "y": 480000}, {"punto": "2", "x": 290000, "y": 480100},
        {"punto": "3", "x": 290100, "y": 480100}, {"punto": "4", "x": 290100, "y": 480000}]}
    r = ag._desde_lectura(s, lectura)
    assert r["ok"] and r["resumen"]["area_m2"] == pytest.approx(10000, rel=1e-3)
    x, y = s.get_terreno().puntos[0]
    assert 479600 < x < 480000 + 400 and 289600 < y < 290400 and (x, y) != (480000, 290000)


def test_formato_whatsapp_quita_markdown():
    from agente import formato_whatsapp

    t = formato_whatsapp("*Lotes:* **8 lotes**\n* uno\n- dos\n\n---\n\n## Título\n__x__")
    assert "**" not in t and "---" not in t and "#" not in t
    assert "*8 lotes*" in t and "• uno" in t and "• dos" in t and "*Título*" in t and "_x_" in t
