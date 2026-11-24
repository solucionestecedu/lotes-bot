#!/usr/bin/env bash
# Publica Lotes-Bot en Google Cloud Run.
# Córrelo en Cloud Shell, dentro de la carpeta lotes-bot:   bash desplegar.sh
# Te pregunta el proyecto y la llave de Gemini. Al final te da el enlace del chat de prueba.
set -euo pipefail
REGION="${REGION:-us-central1}"
SERVICIO="${SERVICIO:-lotes-bot}"

PROYECTO="$(gcloud config get-value project 2>/dev/null || true)"
read -rp "ID del proyecto de Google Cloud [${PROYECTO}]: " P
PROYECTO="${P:-$PROYECTO}"
[ -n "$PROYECTO" ] || { echo "Falta el ID del proyecto."; exit 1; }
gcloud config set project "$PROYECTO" >/dev/null

echo "1/5 Activando servicios de Google Cloud (la primera vez tarda 1 o 2 minutos)..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

echo "2/5 Llave de Gemini..."
if gcloud secrets describe GEMINI_API_KEY >/dev/null 2>&1; then
  echo "    Ya estaba guardada; se usa esa."
else
  read -rsp "    Pega tu llave de Gemini (no se ve al pegarla) y da Enter: " LLAVE; echo
  [ -n "$LLAVE" ] || { echo "No se pegó ninguna llave."; exit 1; }
  printf "%s" "$LLAVE" | gcloud secrets create GEMINI_API_KEY --data-file=- >/dev/null
  unset LLAVE
fi

echo "3/5 Permisos..."
NUM="$(gcloud projects describe "$PROYECTO" --format='value(projectNumber)')"
SA="${NUM}-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding GEMINI_API_KEY --member="serviceAccount:${SA}" \
  --role=roles/secretmanager.secretAccessor --quiet >/dev/null
gcloud projects add-iam-policy-binding "$PROYECTO" --member="serviceAccount:${SA}" \
  --role=roles/run.builder --condition=None --quiet >/dev/null || true

# Si el servicio ya existe, se conserva su token del chat de prueba (el enlace no cambia).
if [ -z "${PRUEBA_TOKEN:-}" ]; then
  PRUEBA_TOKEN="$(gcloud run services describe "$SERVICIO" --region "$REGION" --format=json 2>/dev/null \
    | python3 -c 'import json,sys
try:
    env = json.load(sys.stdin)["spec"]["template"]["spec"]["containers"][0].get("env", [])
    print(next((e["value"] for e in env if e.get("name") == "PRUEBA_TOKEN"), ""))
except Exception:
    print("")' || true)"
fi
PRUEBA_TOKEN="${PRUEBA_TOKEN:-$(openssl rand -hex 12)}"
echo "4/5 Construyendo y publicando (de 5 a 10 minutos)..."
gcloud run deploy "$SERVICIO" --source . --region "$REGION" --allow-unauthenticated \
  --memory 2Gi --cpu 1 --no-cpu-throttling --max-instances 1 --timeout 600 \
  --update-env-vars "PRUEBA_TOKEN=${PRUEBA_TOKEN}" \
  --update-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest" --quiet

URL="$(gcloud run services describe "$SERVICIO" --region "$REGION" --format='value(status.url)')"
gcloud run services update "$SERVICIO" --region "$REGION" --update-env-vars "URL_PUBLICA=${URL}" --quiet >/dev/null

echo "5/5 Comprobando..."
curl -s "${URL}/health"; echo
echo
echo "Listo. Chat de prueba (guárdalo, el token es tu contraseña de prueba):"
echo "    ${URL}/prueba?t=${PRUEBA_TOKEN}"
echo "Dirección para el webhook de WhatsApp (más adelante):"
echo "    ${URL}/webhook"
