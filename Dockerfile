FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Calles, ríos y construcciones de El Salvador (OpenStreetMap, © colaboradores, ODbL): se bajan del sitio oficial de
# descargas (Geofabrik) en cada publicación y quedan dentro del bot. Si falla, el bot sale igual, sin datos del mapa.
FROM base AS mapa
RUN pip install --no-cache-dir osmium
COPY motor/ motor/
COPY servicios/ servicios/
COPY herramientas/construir_osm.py herramientas/
RUN mkdir -p /mapa && python herramientas/construir_osm.py /mapa/osm_sv.npz

FROM base
COPY . .
COPY --from=mapa /mapa/ datos/
RUN useradd -m bot && chown -R bot /app
USER bot
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
