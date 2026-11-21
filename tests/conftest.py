import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.environ.setdefault("ENTORNO", "dev")
os.environ.setdefault("ARCHIVOS_DIR", "/tmp/lotes-bot-tests")
DATOS = os.path.join(RAIZ, "tests", "datos")
