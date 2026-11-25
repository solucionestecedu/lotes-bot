"""Configuración por variables de entorno. Las llaves nunca van en el código ni en el repositorio."""
import os


def _env(nombre: str, defecto: str = "") -> str:
    return os.getenv(nombre, defecto).strip()


ENTORNO = _env("ENTORNO", "produccion")  # produccion | dev

# Gemini (Google AI Studio)
GEMINI_API_KEY = _env("GEMINI_API_KEY")
MODELO_CHAT = _env("MODELO_CHAT", "gemini-3.8-flash")
MODELO_VISION = _env("MODELO_VISION", "gemini-3.8-flash")
# Revisor de la entrega: un modelo que piensa revisa las láminas y vistas antes de mandarlas (vacío = sin revisor).
# Opciones: gemini-3.8-flash con pensamiento alto (rápido) o gemini-3.1-pro-preview (más fino, más lento).
MODELO_REVISOR = _env("MODELO_REVISOR", "gemini-3.8-flash")
PENSAMIENTO_REVISOR = _env("PENSAMIENTO_REVISOR", "high")  # low | medium | high
# Modelo de imagen (Nano Banana) que pinta la imagen de presentación sobre el trazo exacto. Vacío = sin pintar.
MODELO_IMAGEN = _env("MODELO_IMAGEN", "gemini-nano-banana-2.1")
TAMANO_IMAGEN = _env("TAMANO_IMAGEN", "2K")  # 1K | 2K | 4K

# WhatsApp Cloud API (Meta)
META_APP_SECRET = _env("META_APP_SECRET")
META_WA_TOKEN = _env("META_WA_TOKEN")
META_PHONE_NUMBER_ID = _env("META_PHONE_NUMBER_ID")
META_WEBHOOK_VERIFY_TOKEN = _env("META_WEBHOOK_VERIFY_TOKEN")
META_GRAPH_VERSION = _env("META_GRAPH_VERSION", "v20.0")
# Piloto: solo estos números pueden usar el bot (separados por coma, con código de país: 503XXXXXXXX). Vacío = todos.
NUMEROS_PERMITIDOS = {n.strip() for n in _env("NUMEROS_PERMITIDOS").split(",") if n.strip()}

# Chat de prueba en el navegador (/prueba?t=TOKEN). Vacío = desactivado en producción.
PRUEBA_TOKEN = _env("PRUEBA_TOKEN")

ARCHIVOS_DIR = _env("ARCHIVOS_DIR", "/tmp/lotes-bot")
URL_PUBLICA = _env("URL_PUBLICA").rstrip("/")
MAX_ARCHIVO_MB = int(_env("MAX_ARCHIVO_MB", "15"))
