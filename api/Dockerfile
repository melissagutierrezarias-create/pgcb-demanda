# Imagen de la API de pronóstico (FastAPI + paquete pgcb_modelo)
FROM python:3.11-slim

# LightGBM necesita la librería de OpenMP del sistema
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios para correr la API
RUN useradd -m -s /bin/bash api-user
WORKDIR /opt/api

# Primero las dependencias (se reaprovechan en la caché si solo cambia el código)
COPY requirements.txt .
COPY model-pkg/ model-pkg/
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY app/ app/
RUN chown -R api-user:api-user /opt/api
USER api-user

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
