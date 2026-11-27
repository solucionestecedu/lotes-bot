#!/usr/bin/env bash
# Arma el paquete de OpenStreetMap de El Salvador (como al publicar) y muestra lo que dice alrededor de los terrenos
# de prueba: bash mapa_ejemplo.sh   (CRUDO=1 para ver todo)
set -euo pipefail
[ -x /tmp/venv-lotes/bin/python ] || python3 -m venv /tmp/venv-lotes
/tmp/venv-lotes/bin/pip install -q -r requirements.txt osmium
[ -f /tmp/osm_sv.npz ] || /tmp/venv-lotes/bin/python -u herramientas/construir_osm.py /tmp/osm_sv.npz
OSM_DATOS=/tmp/osm_sv.npz /tmp/venv-lotes/bin/python -u pruebas/mapa_ejemplo.py 2>&1 | grep -v Warning
