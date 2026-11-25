#!/usr/bin/env bash
# Terrazas, muros y revisión del ejemplo con relieve real, sin IA: bash muros_ejemplo.sh
set -euo pipefail
[ -x /tmp/venv-lotes/bin/python ] || python3 -m venv /tmp/venv-lotes
/tmp/venv-lotes/bin/pip install -q -r requirements.txt
/tmp/venv-lotes/bin/python -u pruebas/muros_ejemplo.py 2>&1 | grep -v Warning
