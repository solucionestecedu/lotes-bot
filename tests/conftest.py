import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.environ.setdefault("ENTORNO", "dev")
os.environ.setdefault("ARCHIVOS_DIR", "/tmp/lotes-bot-tests")
DATOS = os.path.join(RAIZ, "tests", "datos")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _sin_geovisor_opamss(monkeypatch, request):
    """Las pruebas no salen a internet: el geovisor de la OPAMSS no responde salvo en sus propias pruebas."""
    if request.node.get_closest_marker("geovisor"):
        return
    from servicios import opamss

    monkeypatch.setattr(opamss, "consultar", lambda anillo, http=None: None)


def pytest_configure(config):
    config.addinivalue_line("markers", "geovisor: prueba que usa el geovisor de la OPAMSS simulado")
