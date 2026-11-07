# Pronóstico de demanda y alertas de déficit eléctrico — PGCB (Bangladesh)

Producto de datos que pronostica la demanda eléctrica horaria a 24 horas y marca las horas
con déficit esperado. Se sirve con una API y un tablero desplegados en Docker.

## Datos
- Fuente: [PGCB Hourly Generation Dataset (Bangladesh), UCI](https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh)) — CC BY 4.0, DOI 10.24432/C59P6V.
- `data/raw/PGCB_date_power_demand.xlsx`: archivo original (versionado con DVC).
- `data/processed/pgcb_horario_limpio.csv`: serie horaria limpia (versionada con DVC).

## Estructura
```
data/                datos originales y limpios (DVC)
notebooks/           exploración (Entrega 1)
src/limpieza.py      limpieza de datos
src/features.py      variables para pronosticar a 24 h y particiones temporales (entrenamiento / validación / prueba)
src/entrenar.py      entrena un modelo de demanda y lo registra en MLflow
src/experimentos.py  corre los experimentos de demanda, registra el modelo final y lo exporta a models/
src/figuras_demanda.py  figuras de evaluación del modelo de demanda
docs/                guía de MLflow en EC2
```

## Cómo reproducir
```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
dvc pull                                            # descarga los datos

export MLFLOW_TRACKING_URI=http://<IP-EC2>:5000     # ver docs/guia_ec2_mlflow.md
python -m src.experimentos                          # 27 experimentos + modelo final (≈3 min)
python -m src.figuras_demanda                       # figuras del reporte
```

## Modelo de demanda (últimos 12 meses, 17-jun-2024 a 17-jun-2025)
| Modelo | MAPE | MAE (MW) |
|---|---|---|
| Línea base: misma hora del día anterior | 5,99 % | 681 |
| LightGBM (6 variables, objetivo relativo), registrado como `pgcb-demanda-24h` | **4,92 %** | 575 |
