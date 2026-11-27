#!/usr/bin/env bash
# Calles, ríos y construcciones de OpenStreetMap alrededor de los terrenos de prueba: bash mapa_ejemplo.sh
set -euo pipefail
[ -x /tmp/venv-lotes/bin/python ] || python3 -m venv /tmp/venv-lotes
/tmp/venv-lotes/bin/pip install -q -r requirements.txt
/tmp/venv-lotes/bin/python -u pruebas/mapa_ejemplo.py 2>&1 | grep -v Warning
