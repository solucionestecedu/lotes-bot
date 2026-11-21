"""WhatsApp Cloud API (Meta): firma del webhook, lectura de mensajes, descarga de archivos y envíos."""
from __future__ import annotations

import hashlib
import hmac
import logging
import mimetypes
import os
import re

import requests

import config

log = logging.getLogger("lotes-bot.whatsapp")
LIMITE_TEXTO = 4000


def firma_valida(cuerpo: bytes, firma: str | None, secreto: str) -> bool:
    if not secreto or not re.fullmatch(r"sha256=[0-9a-fA-F]{64}", firma or ""):
        return False
    esperado = hmac.new(secreto.encode(), cuerpo, hashlib.sha256).hexdigest()
    return hmac.compare_digest(esperado, firma[7:].lower())


def mensajes_entrantes(payload: dict) -> list[dict]:
    """Aplana el webhook de Meta en mensajes simples. Ignora estados (entregado, leído)."""
    out = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            for m in value.get("messages") or []:
                tipo = m.get("type")
                msg = {"id": m.get("id"), "de": m.get("from"), "tipo": tipo, "texto": ""}
                if tipo == "text":
                    msg["texto"] = (m.get("text") or {}).get("body", "")
                elif tipo in ("image", "document"):
                    media = m.get(tipo) or {}
                    msg.update(media_id=media.get("id"), mime=media.get("mime_type", ""), nombre=media.get("filename", ""),
                               texto=media.get("caption", ""))
                elif tipo == "location":
                    loc = m.get("location") or {}
                    msg.update(lat=loc.get("latitude"), lng=loc.get("longitude"))
                out.append(msg)
    return out


class WhatsApp:
    def __init__(self, http=None):
        self.http = http or requests.Session()
        self.base = f"https://graph.facebook.com/{config.META_GRAPH_VERSION}"
        self.timeout = (5, 60)

    @property
    def _h(self):
        return {"Authorization": f"Bearer {config.META_WA_TOKEN}"}

    def descargar_media(self, media_id: str) -> tuple[bytes, str]:
        r = self.http.get(f"{self.base}/{media_id}", headers=self._h, timeout=self.timeout)
        r.raise_for_status()
        info = r.json()
        if int(info.get("file_size") or 0) > config.MAX_ARCHIVO_MB * 1024 * 1024:
            raise ValueError("archivo demasiado grande")
        r = self.http.get(info["url"], headers=self._h, timeout=self.timeout)
        r.raise_for_status()
        return r.content, info.get("mime_type", "")

    def _enviar(self, destino: str, cuerpo: dict) -> bool:
        r = self.http.post(f"{self.base}/{config.META_PHONE_NUMBER_ID}/messages", headers=self._h, timeout=self.timeout,
                           json={"messaging_product": "whatsapp", "to": destino, **cuerpo})
        if r.status_code >= 300:
            log.warning("Meta rechazó envío: %s", r.status_code)
        return r.status_code < 300

    def enviar_texto(self, destino: str, texto: str) -> bool:
        ok = True
        for k in range(0, max(len(texto), 1), LIMITE_TEXTO):
            ok &= self._enviar(destino, {"type": "text", "text": {"preview_url": False, "body": texto[k:k + LIMITE_TEXTO]}})
        return ok

    def enviar_documento(self, destino: str, ruta: str, titulo: str = "") -> bool:
        # Meta solo acepta ciertos tipos de documento (PDF, Office, texto). KML y DXF son texto: se suben como
        # text/plain conservando su nombre y extensión, y se abren normal en la computadora.
        mime = mimetypes.guess_type(ruta)[0] or "text/plain"
        if ruta.lower().endswith((".kml", ".dxf")):
            mime = "text/plain"
        with open(ruta, "rb") as f:
            r = self.http.post(f"{self.base}/{config.META_PHONE_NUMBER_ID}/media", headers=self._h, timeout=self.timeout,
                               data={"messaging_product": "whatsapp", "type": mime},
                               files={"file": (os.path.basename(ruta), f, mime)})
        if r.status_code >= 300 or "id" not in r.json():
            log.warning("Meta rechazó la subida: %s", r.status_code)
            return False
        return self._enviar(destino, {"type": "document", "document": {"id": r.json()["id"], "filename": os.path.basename(ruta),
                                                                        "caption": titulo}})
