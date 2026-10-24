# Pronóstico de demanda y alertas de déficit eléctrico — PGCB (Bangladesh)

Proyecto del curso: producto de datos que pronostica la demanda eléctrica horaria a 24 horas
y marca las horas con déficit esperado, servido con una API y un tablero en Docker.

## Datos
- Fuente: [PGCB Hourly Generation Dataset (Bangladesh), UCI](https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh)) — CC BY 4.0, DOI 10.24432/C59P6V.
- `data/raw/PGCB_date_power_demand.xlsx`: archivo original (versionado con DVC).
- `data/processed/pgcb_horario_limpio.csv`: serie horaria limpia (versionada con DVC).

## Estructura
```
data/raw/         datos originales (DVC)
data/processed/   datos limpios (DVC)
notebooks/        exploración
src/              código reutilizable (limpieza)
reports/figures/  gráficos del reporte
```

## Cómo reproducir
```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
dvc pull                                            # descarga los datos
jupyter notebook notebooks/01_exploracion.ipynb
```
