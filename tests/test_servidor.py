"""Webhook de WhatsApp y chat de prueba."""
import hashlib
import hmac
import json
import os

import pytest
from fastapi.testclient import TestClient

from conftest import DATOS

os.environ["META_APP_SECRET"] = "secreto"
os.environ["META_WEBHOOK_VERIFY_TOKEN"] = "verifica"
os.environ["PRUEBA_TOKEN"] = "tok"

import config  # noqa: E402

config.META_APP_SECRET = "secreto"
config.META_WEBHOOK_VERIFY_TOKEN = "verifica"
config.PRUEBA_TOKEN = "tok"
import main  # noqa: E402
from servicios.whatsapp import firma_valida, mensajes_entrantes  # noqa: E402

cli = TestClient(main.app)


def _firmar(cuerpo: bytes) -> str:
    return "sha256=" + hmac.new(b"secreto", cuerpo, hashlib.sha256).hexdigest()


def test_verificacion_webhook():
    r = cli.get("/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "verifica", "hub.challenge": "123"})
    assert r.status_code == 200 and r.text == "123"
    assert cli.get("/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "mal", "hub.challenge": "1"}).status_code == 403


def test_firma():
    assert firma_valida(b"abc", _firmar(b"abc"), "secreto")
    assert not firma_valida(b"abc", _firmar(b"abd"), "secreto")
    assert not firma_valida(b"abc", None, "secreto")
    assert not firma_valida(b"abc", _firmar(b"abc"), "")


def _payload(*mensajes):
    return {"entry": [{"changes": [{"value": {"messages": list(mensajes), "statuses": [{"id": "x", "status": "read"}]}}]}]}


def test_parseo_mensajes():
    p = _payload({"id": "1", "from": "50370000000", "type": "text", "text": {"body": "hola"}},
                 {"id": "2", "from": "50370000000", "type": "image", "image": {"id": "M1", "mime_type": "image/jpeg", "caption": "plano"}},
                 {"id": "3", "from": "50370000000", "type": "location", "location": {"latitude": 13.7, "longitude": -89.2}},
                 {"id": "4", "from": "50370000000", "type": "document", "document": {"id": "M2", "mime_type": "application/pdf", "filename": "p.pdf"}})
    m = mensajes_entrantes(p)
    assert [x["tipo"] for x in m] == ["text", "image", "location", "document"]
    assert m[1]["media_id"] == "M1" and m[1]["texto"] == "plano" and m[2]["lat"] == 13.7 and m[3]["nombre"] == "p.pdf"


def test_webhook_rechaza_sin_firma():
    cuerpo = json.dumps(_payload()).encode()
    assert cli.post("/webhook", content=cuerpo, headers={"content-type": "application/json"}).status_code == 401


def test_webhook_procesa_una_vez(monkeypatch):
    llamados = []
    monkeypatch.setattr(main, "procesar_whatsapp", lambda m: llamados.append(m["id"]))
    cuerpo = json.dumps(_payload({"id": "wamid.A", "from": "503", "type": "text", "text": {"body": "hola"}})).encode()
    h = {"content-type": "application/json", "x-hub-signature-256": _firmar(cuerpo)}
    assert cli.post("/webhook", content=cuerpo, headers=h).status_code == 200
    assert cli.post("/webhook", content=cuerpo, headers=h).status_code == 200  # reintento de Meta
    assert llamados == ["wamid.A"]


def test_procesar_whatsapp_envia_respuesta(monkeypatch):
    enviados = []

    class WAFalso:
        def enviar_texto(self, destino, texto):
            enviados.append(("texto", destino, texto))
            return True

        def enviar_documento(self, destino, ruta, titulo=""):
            enviados.append(("doc", destino, os.path.basename(ruta)))
            return True

        def descargar_media(self, media_id):
            raise AssertionError("no debería descargar")

    monkeypatch.setattr(main, "WhatsApp", WAFalso)
    with open(os.path.join(DATOS, "plano_a.txt"), encoding="utf-8") as f:
        rumbos = f.read()
    main.procesar_whatsapp({"id": "z", "de": "50370000001", "tipo": "text", "texto": rumbos})
    assert enviados and enviados[0][0] == "texto" and "5,623.28" in enviados[0][2]


def test_numero_no_permitido(monkeypatch):
    enviados = []
    monkeypatch.setattr(main.config, "NUMEROS_PERMITIDOS", {"50399999999"})
    monkeypatch.setattr(main, "WhatsApp", lambda: type("W", (), {"enviar_texto": lambda s, d, t: enviados.append(t)})())
    main.procesar_whatsapp({"id": "q", "de": "50370000002", "tipo": "text", "texto": "hola"})
    assert enviados == []


def test_chat_de_prueba_y_descarga():
    assert cli.get("/prueba").status_code == 404
    assert cli.get("/prueba", params={"t": "tok"}).status_code == 200
    with open(os.path.join(DATOS, "plano_a.txt"), encoding="utf-8") as f:
        rumbos = f.read()
    r = cli.post("/chat", data={"token": "tok", "sesion": "s1", "texto": rumbos}).json()
    assert "5,623.28" in r["salidas"][0]["texto"]
    r = cli.post("/chat", data={"token": "tok", "sesion": "s1", "texto": "lámina"}).json()
    docs = [s for s in r["salidas"] if s["tipo"] == "archivo"]
    assert [d["nombre"].rsplit(".", 1)[1] for d in docs] == ["pdf", "dxf"]  # sin ubicar: sin KML
    pdf = cli.get(docs[0]["url"])
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert cli.post("/chat", data={"token": "mal", "sesion": "s1", "texto": "x"}).status_code == 404


def test_chat_rechaza_tipo_de_archivo():
    r = cli.post("/chat", data={"token": "tok", "sesion": "s2"}, files={"archivo": ("a.exe", b"MZ", "application/octet-stream")})
    assert "Solo fotos" in r.json()["salidas"][0]["texto"]
