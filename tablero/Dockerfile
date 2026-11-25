# Imagen del tablero (Dash). No lleva modelo: consulta la API definida en API_URL.
FROM python:3.11-slim

# Usuario sin privilegios, igual que la imagen de la API
RUN useradd -m -s /bin/bash tablero-user
WORKDIR /opt/tablero

# Primero las dependencias (se reaprovechan en la caché si solo cambia el código)
COPY tablero/requirements.txt tablero/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r tablero/requirements.txt

# Serie histórica completa: selector de fechas, comparación con lo real y paneles históricos
COPY data/processed/pgcb_horario_limpio.csv data/processed/pgcb_horario_limpio.csv
COPY tablero/ tablero/
RUN chown -R tablero-user:tablero-user /opt/tablero
USER tablero-user

ENV PORT=8050 API_URL=http://localhost:8001
EXPOSE 8050
CMD ["python", "-m", "tablero.app"]