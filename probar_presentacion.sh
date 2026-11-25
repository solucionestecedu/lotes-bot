#!/usr/bin/env bash
# Prueba la imagen de presentación con Nano Banana en Cloud Shell: bash probar_presentacion.sh [modelos...]
set -euo pipefail
[ -x /tmp/venv-lotes/bin/python ] || python3 -m venv /tmp/venv-lotes
/tmp/venv-lotes/bin/pip install -q -r requirements.txt
/tmp/venv-lotes/bin/pip install -q -r requirements-dev.txt >/dev/null 2>&1 || true
if [[ " $* " == *" --reusar "* ]]; then export GEMINI_API_KEY=x; else export GEMINI_API_KEY="$(gcloud secrets versions access latest --secret GEMINI_API_KEY)"; fi

/tmp/venv-lotes/bin/python -u pruebas/probar_presentacion.py "$@"
