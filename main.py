"""Servidor: webhook de WhatsApp, chat de prueba en el navegador y descarga de archivos."""
from __future__ import annotations

import hmac
import logging
import os
import threading
import uuid
from collections import OrderedDict

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

import config
from agente import Agente, MemoriaSesiones
from servicios import llm as llm_mod
from servicios.whatsapp import WhatsApp, firma_valida, mensajes_entrantes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("lotes-bot")

app = FastAPI(title="Lotes-Bot", docs_url=None, redoc_url=None)
sesiones = MemoriaSesiones()
agente = Agente(llm_mod.crear(config))
_vistos: OrderedDict[str, bool] = OrderedDict()
_vistos_lock = threading.Lock()
_archivos: dict[str, str] = {}  # id público -> ruta
MIME_OK = ("image/jpeg", "image/png", "image/webp", "application/pdf")


def _nuevo(msg_id: str) -> bool:
    """Meta reintenta webhooks: cada mensaje se procesa una sola vez."""
    with _vistos_lock:
        if msg_id in _vistos:
            return False
        _vistos[msg_id] = True
        while len(_vistos) > 5000:
            _vistos.popitem(last=False)
        return True


@app.get("/health")
def health():
    return {"ok": True, "modo": type(agente.llm).__name__}


# ---------------------------------------------------------------------------- WhatsApp
@app.get("/webhook")
def verificar(request: Request):
    q = request.query_params
    token = q.get("hub.verify_token") or ""
    if q.get("hub.mode") == "subscribe" and config.META_WEBHOOK_VERIFY_TOKEN and \
            hmac.compare_digest(token, config.META_WEBHOOK_VERIFY_TOKEN):
        return PlainTextResponse(q.get("hub.challenge", ""))
    raise HTTPException(403)


@app.post("/webhook")
async def webhook(request: Request, tareas: BackgroundTasks):
    cuerpo = await request.body()
    if not firma_valida(cuerpo, request.headers.get("x-hub-signature-256"), config.META_APP_SECRET):
        raise HTTPException(401)
    for m in mensajes_entrantes(await request.json()):
        if m.get("id") and _nuevo(m["id"]):
            tareas.add_task(procesar_whatsapp, m)
    return {"ok": True}


def procesar_whatsapp(m: dict):
    wa = WhatsApp()
    destino = m["de"]
    if config.NUMEROS_PERMITIDOS and destino not in config.NUMEROS_PERMITIDOS:
        log.info("número no autorizado")
        return
    try:
        adjuntos, ubicacion = [], None
        if m["tipo"] in ("image", "document"):
            datos, mime = wa.descargar_media(m["media_id"])
            mime = mime or m.get("mime", "")
            if mime not in MIME_OK:
                wa.enviar_texto(destino, "Solo puedo leer fotos (JPG/PNG) o PDF del plano.")
                return
            adjuntos.append({"mime": mime, "nombre": m.get("nombre") or ("plano.pdf" if mime.endswith("pdf") else "foto"),
                             "datos": datos})
        elif m["tipo"] == "location":
            ubicacion = (m["lat"], m["lng"])
        elif m["tipo"] != "text":
            wa.enviar_texto(destino, "Por ahora entiendo texto y fotos o PDF del plano.")
            return
        sesion = sesiones.obtener("wa:" + destino)
        if m.get("texto", "").strip().lower() in ("/reiniciar", "reiniciar"):
            sesiones.borrar("wa:" + destino)
            wa.enviar_texto(destino, "Listo, empezamos de cero. Mándame el plano.")
            return
        for s in agente.atender(sesion, m.get("texto", ""), adjuntos, ubicacion):
            if s["tipo"] == "texto":
                wa.enviar_texto(destino, s["texto"])
            elif s.get("imagen"):
                wa.enviar_imagen(destino, s["ruta"], s["titulo"])
            elif not wa.enviar_documento(destino, s["ruta"], s["titulo"]) and config.URL_PUBLICA:
                wa.enviar_texto(destino, f"{s['titulo']}: {config.URL_PUBLICA}{_publicar(s['ruta'])}")
    except Exception:
        log.exception("error procesando mensaje de WhatsApp")
        try:
            wa.enviar_texto(destino, "Uy, algo falló de mi lado. Intenta de nuevo en un momento.")
        except Exception:
            pass


# ---------------------------------------------------------------------------- chat de prueba
def _token_ok(t: str | None) -> bool:
    if config.ENTORNO == "dev" and not config.PRUEBA_TOKEN:
        return True
    return bool(config.PRUEBA_TOKEN) and hmac.compare_digest(t or "", config.PRUEBA_TOKEN)


def _publicar(ruta: str) -> str:
    fid = uuid.uuid4().hex
    _archivos[fid] = ruta
    return f"/archivos/{fid}/{os.path.basename(ruta)}"


@app.get("/prueba", response_class=HTMLResponse)
def prueba(t: str = ""):
    if not _token_ok(t):
        raise HTTPException(404)
    with open(os.path.join(os.path.dirname(__file__), "web", "prueba.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/chat")
async def chat(token: str = Form(""), sesion: str = Form(...), texto: str = Form(""), lat: str = Form(""),
               lng: str = Form(""), archivo: UploadFile | None = File(None)):
    if not _token_ok(token):
        raise HTTPException(404)
    adjuntos = []
    if archivo is not None and archivo.filename:
        datos = await archivo.read()
        if len(datos) > config.MAX_ARCHIVO_MB * 1024 * 1024:
            return JSONResponse({"salidas": [{"tipo": "texto", "texto": "El archivo es muy grande."}]})
        mime = archivo.content_type or ""
        if mime not in MIME_OK:
            return JSONResponse({"salidas": [{"tipo": "texto", "texto": "Solo fotos (JPG/PNG) o PDF."}]})
        adjuntos.append({"mime": mime, "nombre": archivo.filename, "datos": datos})
    ubicacion = (float(lat), float(lng)) if lat and lng else None
    sid = "web:" + sesion[:64]
    if texto.strip().lower() in ("/reiniciar", "reiniciar"):
        sesiones.borrar(sid)
        return {"salidas": [{"tipo": "texto", "texto": "Listo, empezamos de cero."}]}
    from starlette.concurrency import run_in_threadpool

    salidas = await run_in_threadpool(agente.atender, sesiones.obtener(sid), texto, adjuntos, ubicacion)
    out = []
    for s in salidas:
        if s["tipo"] == "archivo":
            out.append({"tipo": "archivo", "nombre": s["nombre"], "titulo": s["titulo"], "url": _publicar(s["ruta"]),
                        "imagen": bool(s.get("imagen"))})
        else:
            out.append(s)
    return {"salidas": out}


@app.get("/archivos/{fid}/{nombre}")
def archivo(fid: str, nombre: str):
    ruta = _archivos.get(fid)
    if not ruta or not os.path.exists(ruta) or os.path.basename(ruta) != nombre:
        raise HTTPException(404)
    return FileResponse(ruta, filename=nombre)
