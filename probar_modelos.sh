#!/usr/bin/env bash
# Compara modelos de Gemini con las fotos de prueba. Córrelo en Cloud Shell dentro de la carpeta lotes-bot:
#   bash probar_modelos.sh                      (compara gemini-4-carbon y gemini-3.8-flash)
#   bash probar_modelos.sh gemini-3.8-flash     (solo uno)
# Necesita las fotos en pruebas_privadas/lote-a.jpg y pruebas_privadas/lote-b.jpg (no se suben a GitHub).
set -euo pipefail
MODELOS=("$@")
[ ${#MODELOS[@]} -gt 0 ] || MODELOS=(gemini-4-carbon gemini-3.8-flash)
for f in pruebas_privadas/lote-a.jpg pruebas_privadas/lote-b.jpg; do
  [ -f "$f" ] || { echo "Falta $f"; exit 1; }
done
if [ ! -x /tmp/venv-lotes/bin/python ]; then
  echo "Preparando Python (la primera vez tarda 1 o 2 minutos)..."
  python3 -m venv /tmp/venv-lotes
  /tmp/venv-lotes/bin/pip install -q -r requirements.txt
fi
# La llave sale de Secret Manager directo a la variable: no se imprime ni queda en archivos.
GEMINI_API_KEY="$(gcloud secrets versions access latest --secret GEMINI_API_KEY)" \
  /tmp/venv-lotes/bin/python -u pruebas/comparar_modelos.py \
  --img-a pruebas_privadas/lote-a.jpg --img-b pruebas_privadas/lote-b.jpg --modelos "${MODELOS[@]}" \
  --salida /tmp/resultado_comparacion.json
