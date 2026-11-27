#!/usr/bin/env bash
# Consulta real al geovisor de la OPAMSS con los terrenos de prueba: bash opamss_ejemplo.sh
set -euo pipefail
[ -x /tmp/venv-lotes/bin/python ] || python3 -m venv /tmp/venv-lotes
/tmp/venv-lotes/bin/pip install -q -r requirements.txt
/tmp/venv-lotes/bin/python -u pruebas/opamss_ejemplo.py 2>&1 | grep -v Warning
