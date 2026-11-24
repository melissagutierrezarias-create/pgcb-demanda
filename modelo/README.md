# pgcb-modelo — modelo empaquetado

Paquete de Python instalable con los dos modelos finales del proyecto:

| Modelo | Configuración (escogida en validación, Entrega 2) | Archivo |
|---|---|---|
| Demanda a 24 h | LightGBM, variables `basico_festivos`, objetivo relativo | `pgcb_modelo/trained/modelo_pgcb_v<versión>.joblib` |
| Alerta de déficit | LightGBM, variables `completo` (nl7, mc400), umbral para 80 % de sensibilidad | (mismo archivo) |

El paquete trae también `pgcb_modelo/datos/historia_ejemplo.csv` (últimos 14 días del dataset), que usan
las pruebas, el ejemplo de la API y el tablero.

## Uso

```python
import pandas as pd
from pgcb_modelo import pronosticar

historia = pd.read_csv("historia.csv", parse_dates=["datetime"])  # al menos 192 horas (8 días)
res = pronosticar(historia)  # 24 filas: demanda_mw, generacion_disponible_mw, deficit_mw, prob_racionamiento, alerta
```

Columnas necesarias en la historia: `datetime`, `demand_mw`, `generation_mw`, `load_shedding`, `liquid_fuel`.

## Entrenar y construir el paquete (desde la raíz del repositorio)

```bash
source .venv/bin/activate && pip install -r requirements.txt build
dvc pull                                          # datos
export MLFLOW_TRACKING_URI=http://<IP-EC2>:5000   # opcional: registra la corrida en MLflow (experimento pgcb-paquete)
python -m src.empaquetar                          # entrena, evalúa en prueba y guarda el modelo en el paquete
cd modelo && python -m build --wheel              # crea modelo/dist/pgcb_modelo-<versión>-py3-none-any.whl
pip install pytest dist/*.whl && pytest           # 6 pruebas
cp dist/*.whl ../api/model-pkg/                   # la API instala el paquete desde aquí
```

Para publicar una versión nueva: cambie `pgcb_modelo/VERSION`, vuelva a correr los pasos y actualice
el nombre del `.whl` en `api/requirements.txt`.
