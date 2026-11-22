# Lotes-Bot

Agente de WhatsApp para **anteproyectos de lotificación en El Salvador**.

Le mandas la foto o el PDF del plano del topógrafo y te devuelve:

- la **verificación** del plano (cierre de la poligonal, área, coordenadas),
- el **relieve** aproximado (curvas de nivel con datos satelitales),
- un **prediseño de lotes** (calles, lotes y área verde),
- la **lámina en PDF** lista para imprimir en tabloide a escala, más **KML** (Google Earth) y **DXF** (AutoCAD / Civil 3D),
- una **vista 3D** del fraccionamiento y un **modelo 3D en DXF** (terreno, lotes, calles y casas tipo) para renderizar en AutoCAD.

> Es un **anteproyecto**: sirve para decidir rápido si un terreno vale la pena y cuántos lotes da.
> No sirve para trámite. La DOT, OPAMSS o el CNR piden levantamiento topográfico y la firma de un profesional autorizado.

## Cómo piensa

**La IA lee; la matemática verifica.** Gemini (3.8 Flash) transcribe el cuadro de rumbos del plano. Después, el código recorre
la poligonal y revisa que cierre, que el área cuadre con la del plano y que las coordenadas escritas coincidan.
Si Gemini leyó mal un número, el cierre falla y el bot señala el tramo sospechoso para que lo confirmes.
Ningún número que ves sale del modelo de lenguaje: todos salen del motor de cálculo (`motor/`).

```
WhatsApp ─► /webhook ─► agente.py ─► Gemini (conversa y decide)
                           │
                           └─► herramientas ─► motor/  (rumbos, coordenadas, relieve, lotes, lámina, KML, DXF)
```

| Archivo | Qué hace |
|---|---|
| `motor/rumbos.py` | Lee rumbos (N 45°30'15" E 32.50), recorre la poligonal, calcula cierre y área |
| `motor/proyeccion.py` | Lambert de El Salvador sobre SIRGAS-ES2007; convierte planos viejos NAD27 |
| `motor/terreno.py` | El terreno: verificaciones, tramos sospechosos, cuadro de construcción |
| `motor/relieve.py` | Alturas satelitales (AWS Terrain Tiles; respaldo Open-Meteo), pendientes, curvas |
| `motor/lotificacion.py` | Prediseño automático de calles, lotes y área verde |
| `motor/lamina.py` | Lámina PDF tabloide a escala: planialtimétrico + lotificación |
| `motor/exportar.py` | KML y DXF |
| `motor/modelo3d.py` | Maqueta 3D: DXF 3D para renderizar en AutoCAD y vista aérea en imagen |
| `agente.py` | El agente: herramientas, reglas y memoria de la conversación |
| `servicios/llm.py` | Gemini, y un "modo sin IA" para probar sin llave |
| `servicios/whatsapp.py` | WhatsApp Cloud API de Meta |
| `main.py` | Servidor: `/webhook`, `/prueba` (chat en el navegador), `/chat`, `/archivos` |
| `evaluar_lectura.py` | Mide qué tan bien lee Gemini un cuadro de rumbos contra la transcripción correcta |

## Correr en tu computadora

```bash
python -m venv .venv && source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                     # y llena GEMINI_API_KEY
set -a && source .env && set +a                          # carga las variables (Mac/Linux)
uvicorn main:app --reload --port 8080
```

Abre <http://localhost:8080/prueba>. Sin `GEMINI_API_KEY` funciona en **modo sin IA**: no lee fotos, pero puedes pegar
los rumbos como texto y probar todo lo demás.

Pruebas automáticas (46, con los tres planos reales que revisamos):

```bash
pytest -q
```

## Publicarlo en Google Cloud Run

**Forma fácil:** abre [Cloud Shell](https://shell.cloud.google.com), sube la carpeta (o el .zip), entra a ella y corre
`bash desplegar.sh`. Te pregunta el proyecto y la llave de Gemini, y al final te da el enlace del chat de prueba.

**Forma manual**, paso por paso:

Necesitas el [SDK de gcloud](https://cloud.google.com/sdk/docs/install) y un proyecto de Google Cloud con facturación.
Cambia `MI_PROYECTO` por el tuyo.

```bash
gcloud config set project MI_PROYECTO
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

# 1. Guarda las llaves como secretos (nunca en el código)
printf "%s" "TU_LLAVE_DE_GEMINI" | gcloud secrets create GEMINI_API_KEY --data-file=-
printf "%s" "SECRETO_DE_LA_APP_META" | gcloud secrets create META_APP_SECRET --data-file=-
printf "%s" "TOKEN_DE_WHATSAPP" | gcloud secrets create META_WA_TOKEN --data-file=-

# 2. Deja que Cloud Run lea esos secretos
NUM=$(gcloud projects describe MI_PROYECTO --format="value(projectNumber)")
gcloud projects add-iam-policy-binding MI_PROYECTO \
  --member="serviceAccount:${NUM}-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"

# 3. Publica (desde la carpeta del repo)
gcloud run deploy lotes-bot --source . --region us-central1 --allow-unauthenticated \
  --memory 1Gi --cpu 1 --no-cpu-throttling --max-instances 1 --timeout 300 \
  --set-env-vars "PRUEBA_TOKEN=inventa-uno-largo,META_PHONE_NUMBER_ID=XXXX,META_WEBHOOK_VERIFY_TOKEN=otra-frase-larga,NUMEROS_PERMITIDOS=503XXXXXXXX" \
  --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest,META_APP_SECRET=META_APP_SECRET:latest,META_WA_TOKEN=META_WA_TOKEN:latest"
```

Al terminar te da una URL tipo `https://lotes-bot-xxxx.a.run.app`. El chat de prueba queda en
`https://lotes-bot-xxxx.a.run.app/prueba?t=inventa-uno-largo`. Ponle esa URL a `URL_PUBLICA` con
`gcloud run services update lotes-bot --region us-central1 --update-env-vars URL_PUBLICA=https://...`.

Por qué esas opciones:

- `--max-instances 1`: en el piloto, la conversación vive en la memoria del servidor. Con una sola instancia no se pierde
  a media plática. Si el servidor se apaga por falta de uso, la siguiente plática empieza de cero (hay que reenviar el plano).
- `--no-cpu-throttling`: el bot le contesta "recibido" a Meta de inmediato y sigue trabajando por detrás (relieve, PDF).
- `NUMEROS_PERMITIDOS`: mientras pruebas, solo esos números pueden gastar tu cuota de Gemini.

## Conectar WhatsApp (Meta)

1. En [Meta for Developers](https://developers.facebook.com/) crea una app tipo **Business** y agrega el producto **WhatsApp**.
2. En **WhatsApp → Configuración de la API** Meta te da un **número de prueba**. Agrega tu celular como destinatario
   (permite hasta 5). Así pruebas desde ya, sin el número nuevo.
3. Copia el **Phone number ID** (va en `META_PHONE_NUMBER_ID`) y un **token**. El token temporal dura 24 h; para dejarlo
   fijo crea un *usuario del sistema* en el Business Manager con permiso `whatsapp_business_messaging`.
4. En **Configuración → Básica** copia la **Clave secreta de la app** (va en `META_APP_SECRET`).
5. En **WhatsApp → Configuración → Webhook**:
   - URL de devolución de llamada: `https://TU-URL/webhook`
   - Token de verificación: el mismo de `META_WEBHOOK_VERIFY_TOKEN`
   - Suscríbete al campo **messages**.
6. Cuando tengas el número nuevo, lo registras en la misma app y cambias `META_PHONE_NUMBER_ID`. El código no cambia.

En WhatsApp escribe `reiniciar` para empezar de cero.

## Límites que hay que tener presentes

- **Relieve**: datos satelitales de ~30 m de origen; el error vertical es de metros, peor en laderas con árboles.
  Las curvas son para ver la forma del terreno, no para diseñar terracería.
- **Ubicación por pin**: si el plano no trae coordenadas, el pin del celular ubica el terreno con error de varios metros.
- **Prediseño**: manzanas rectas de doble fondo. No traza calles curvas ni sigue curvas de nivel todavía.
- **Normativa**: aplica la regla de área verde que le indiques (10 % del área útil o m² por lote). No revisa todas las
  normas de la DOT u OPAMSS.
- **Datos personales**: el bot no transcribe nombres ni DUI/NIT. Los archivos que mandas se procesan en memoria y no se
  guardan en disco; los PDF generados quedan temporalmente en el servidor.
