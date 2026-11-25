#!/usr/bin/env bash
# Entrega completa de ejemplo (Quintas A+B) en Cloud Shell y la baja como zip: bash entrega_ejemplo.sh [modelo]
set -euo pipefail
[ -x /tmp/venv-lotes/bin/python ] || python3 -m venv /tmp/venv-lotes
/tmp/venv-lotes/bin/pip install -q -r requirements.txt
/tmp/venv-lotes/bin/pip install -q -r requirements-dev.txt >/dev/null 2>&1 || true
export GEMINI_API_KEY="$(gcloud secrets versions access latest --secret GEMINI_API_KEY)"
/tmp/venv-lotes/bin/python -u pruebas/entrega_ejemplo.py "$@"
rm -f /tmp/entrega-quintas.zip
(cd /tmp && zip -qr entrega-quintas.zip entrega-quintas)
ls -la /tmp/entrega-quintas.zip
cp /tmp/entrega-quintas.zip ~/ && cloudshell download ~/entrega-quintas.zip  # solo baja lo que está en el home
