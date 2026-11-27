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
    """Las pruebas no salen a internet ni usan el paquete del mapa: la OPAMSS y OpenStreetMap solo en sus propias pruebas."""
    from servicios import opamss, osm

    if not request.node.get_closest_marker("geovisor"):
        monkeypatch.setattr(opamss, "consultar", lambda anillo, http=None: None)
    if not request.node.get_closest_marker("mapa"):
        monkeypatch.setattr(osm, "consultar", lambda anillo, datos=None: None)


def pytest_configure(config):
    config.addinivalue_line("markers", "geovisor: prueba que usa el geovisor de la OPAMSS simulado")
    config.addinivalue_line("markers", "mapa: prueba que usa OpenStreetMap simulado")
