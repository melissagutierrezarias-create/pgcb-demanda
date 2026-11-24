# API de pronóstico de demanda y alertas de déficit

API en FastAPI que sirve el paquete `pgcb_modelo`. Recibe la historia horaria reciente de la red y
devuelve, para las 24 horas siguientes, la demanda pronosticada, la generación disponible estimada,
el déficit, la probabilidad de racionamiento y si la hora queda en alerta.

## Rutas

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/` | Página de inicio con enlace a la documentación |
| GET | `/docs` | Documentación interactiva (Swagger), con un ejemplo listo para probar |
| GET | `/api/v1/health` | Nombre, versión de la API y del modelo, fecha de entrenamiento |
| POST | `/api/v1/predict` | Pronóstico de las 24 horas siguientes |

**Entrada de `/api/v1/predict`:** al menos 192 horas (8 días) de historia.

```json
{"historia": [
  {"datetime": "2025-06-10T00:00:00", "demand_mw": 12850.0, "generation_mw": 12400.0,
   "load_shedding": 450.0, "liquid_fuel": 1300.0},
  ...
]}
```

**Salida:**

```json
{"version_modelo": "0.1.0", "umbral_alerta": 0.52,
 "resumen": {"demanda_pico_mw": 15210.4, "hora_pico": "2025-06-18T20:00:00",
             "horas_en_alerta": 6, "deficit_maximo_mw": 980.2},
 "predicciones": [{"datetime": "2025-06-18T00:00:00", "demanda_mw": 13020.1,
                   "generacion_disponible_mw": 12700.0, "deficit_mw": 320.1,
                   "prob_racionamiento": 0.61, "alerta": true}, ...]}
```

Errores: `422` si la historia tiene menos de 192 horas o le faltan campos; `400` si tiene demasiados
vacíos para pronosticar.

## Llamarla desde el tablero (Python)

```python
import os, requests, pandas as pd
API_URL = os.getenv("API_URL", "http://localhost:8001")

def pronosticar_api(historia: pd.DataFrame) -> pd.DataFrame:
    h = historia.reset_index()[["datetime", "demand_mw", "generation_mw", "load_shedding", "liquid_fuel"]]
    h = h.tail(14 * 24).astype({"datetime": str})
    h = h.astype(object).where(h.notna(), None)  # NaN -> null en el JSON
    r = requests.post(f"{API_URL}/api/v1/predict", json={"historia": h.to_dict("records")}, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json()["predicciones"]).assign(
        datetime=lambda d: pd.to_datetime(d["datetime"])).set_index("datetime")
```

## Correr en local

```bash
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt
pytest                                    # 4 pruebas
uvicorn app.main:app --port 8001          # http://localhost:8001/docs
```

## Correr con Docker

```bash
cd api
docker build -t pgcb-api:0.1.0 .
docker run -d --name pgcb-api -p 8001:8001 pgcb-api:0.1.0
curl http://localhost:8001/api/v1/health
```

En EC2 hay que abrir el puerto 8001 en el *security group* (regla de entrada TCP 8001).
