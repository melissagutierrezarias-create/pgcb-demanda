"""Variables para pronosticar la demanda horaria con 24 horas de anticipación.

Regla clave: para pronosticar la hora t solo se usa información disponible 24 horas antes
(rezagos >= 24 h). Así el modelo puede producir el pronóstico completo del día siguiente
sin conocer ninguna hora de ese día.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

OBJETIVO = "demand_mw"
HORIZONTE = 24  # horas

# Conjuntos de variables que se comparan en los experimentos (selección de características).
CALENDARIO = ["hora", "dia_semana", "mes", "dia_anio_sin", "dia_anio_cos", "anio", "es_viernes"]
FESTIVOS = ["festivo", "dias_a_eid"]
REZAGOS = ["dem_lag24", "dem_lag48", "dem_lag168", "dem_lag336",
           "dem_media_24h_lag24", "dem_max_24h_lag24", "dem_media_7d_misma_hora"]
GENERACION = ["gen_max_7d_misma_hora", "racion_lag24", "racion_media_7d", "liquid_fuel_share_lag24"]

BASICO = ["hora", "dia_semana", "dem_lag24", "dem_media_7d_misma_hora"]

# Conjuntos que se comparan en MLflow, de menos a más variables.
CONJUNTOS = {
    "calendario": CALENDARIO + FESTIVOS,                      # sin rezagos: ¿basta el calendario?
    "basico": BASICO,                                          # hora, día y dos rezagos (EDA 5.6)
    "basico_festivos": BASICO + FESTIVOS,                      # + festivos y Eid
    "basico_festivos_mes": BASICO + FESTIVOS + ["mes"],        # + estacionalidad anual
    "rezagos_festivos": CALENDARIO + FESTIVOS + REZAGOS,       # todos los rezagos y el calendario
    "completo": CALENDARIO + FESTIVOS + REZAGOS + GENERACION,  # + variables de generación y racionamiento
}


def _festivos(idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    anios = range(idx.year.min() - 1, idx.year.max() + 2)
    cal = holidays.country_holidays("BD", years=anios, language="en_US")
    fechas = pd.to_datetime(list(cal.keys()))
    nombres = list(cal.values())
    eid = pd.DatetimeIndex(sorted({f for f, n in zip(fechas, nombres) if "Eid" in n and "Milad" not in n}))
    dias = idx.normalize()
    festivo = dias.isin(fechas).astype(int)
    # Distancia (en días, con signo) al Eid más cercano, recortada a ±10: captura el puente largo de Eid.
    d = np.array([(dias.values[:, None] - eid.values[None, :]).astype("timedelta64[D]").astype(int)][0])
    cerca = d[np.arange(len(d)), np.abs(d).argmin(axis=1)]
    return festivo, np.clip(cerca, -10, 10)


def construir(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe la serie horaria limpia (índice datetime) y devuelve objetivo + variables."""
    df = df.sort_index()
    X = pd.DataFrame(index=df.index)
    idx = df.index
    X["hora"] = idx.hour
    X["dia_semana"] = idx.dayofweek
    X["mes"] = idx.month
    X["dia_anio_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    X["dia_anio_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    X["anio"] = idx.year + idx.dayofyear / 365.25
    X["es_viernes"] = (idx.dayofweek == 4).astype(int)  # fin de semana en Bangladesh
    X["festivo"], X["dias_a_eid"] = _festivos(idx)

    d = df[OBJETIVO]
    X["dem_lag24"] = d.shift(24)
    X["dem_lag48"] = d.shift(48)
    X["dem_lag168"] = d.shift(168)
    X["dem_lag336"] = d.shift(336)
    X["dem_media_24h_lag24"] = d.shift(24).rolling(24, min_periods=18).mean()
    X["dem_max_24h_lag24"] = d.shift(24).rolling(24, min_periods=18).max()
    X["dem_media_7d_misma_hora"] = pd.concat([d.shift(24 * k) for k in range(1, 8)], axis=1).mean(axis=1)

    g = df["generation_mw"]
    # Capacidad que el sistema logró entregar recientemente a esa hora: proxy de la generación disponible.
    X["gen_max_7d_misma_hora"] = pd.concat([g.shift(24 * k) for k in range(1, 8)], axis=1).max(axis=1)
    X["racion_lag24"] = df["load_shedding"].shift(24)
    X["racion_media_7d"] = df["load_shedding"].shift(24).rolling(168, min_periods=120).mean()
    X["liquid_fuel_share_lag24"] = (df["liquid_fuel"] / g).shift(24)

    X[OBJETIVO] = d
    X["generation_mw"] = g
    X["load_shedding"] = df["load_shedding"]
    return X


def cargar(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta, parse_dates=["datetime"], index_col="datetime")
    return df.asfreq("h")


def particionar(X: pd.DataFrame):
    """Particiones temporales (nunca aleatorias: es una serie de tiempo).

    - Prueba: últimos 12 meses (la misma ventana de la línea base de la Entrega 1).
    - Validación: los 12 meses anteriores; se usa para comparar y escoger modelos.
    - Entrenamiento: todo lo anterior (abril 2015 a junio 2023).
    """
    fin = X.index.max()
    ini_prueba = fin - pd.DateOffset(years=1)
    ini_val = ini_prueba - pd.DateOffset(years=1)
    tr = X.loc[X.index < ini_val]
    va = X.loc[(X.index >= ini_val) & (X.index < ini_prueba)]
    te = X.loc[X.index >= ini_prueba]
    return tr, va, te
