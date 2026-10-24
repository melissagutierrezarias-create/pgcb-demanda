"""Carga y limpieza del dataset PGCB Hourly Generation (UCI id 1175).

Uso:
    from src.limpieza import cargar_crudo, limpiar
    crudo = cargar_crudo("data/raw/PGCB_date_power_demand.xlsx")
    horario, reporte = limpiar(crudo)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FUENTES = [
    "gas", "liquid_fuel", "coal", "hydro", "solar", "wind",
    "india_bheramara_hvdc", "india_tripura", "india_adani", "nepal",
]
# Fuentes que no existían al inicio de la serie: su vacío significa 0 MW, no un dato perdido.
FUENTES_NUEVAS = ["solar", "wind", "india_adani", "nepal"]
NUMERICAS = ["generation_mw", "demand_mw", "load_shedding"] + FUENTES


def cargar_crudo(ruta: str) -> pd.DataFrame:
    """Lee el Excel original y lo ordena cronológicamente."""
    df = pd.read_excel(ruta)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def _atipicos_por_hora(s: pd.Series, dias: int = 15, bajo: float = 0.5, alto: float = 1.5) -> pd.Series:
    """Marca valores que se alejan de la mediana de la MISMA hora en los días vecinos.

    Comparar con la misma hora evita confundir el ciclo diario (noche vs. pico) con errores.
    """
    ref = s.groupby(s.index.hour).transform(
        lambda g: g.rolling(dias, center=True, min_periods=5).median()
    )
    ratio = s / ref
    return (ratio < bajo) | (ratio > alto)


def limpiar(df: pd.DataFrame, max_hueco: int = 3) -> tuple[pd.DataFrame, dict]:
    """Devuelve una serie horaria regular y un reporte de calidad.

    Pasos:
      1. Quita registros que no están en punta de hora (pico de la noche: 17:30 y 18:30).
      2. Agrega marcas de tiempo duplicadas con la mediana.
      3. Rellena con 0 las fuentes que aún no existían.
      4. Marca como faltantes los valores atípicos (errores de digitación).
      5. Reindexa a frecuencia horaria e interpola huecos de hasta `max_hueco` horas.
    """
    rep: dict = {"filas_originales": len(df)}

    en_punto = df["datetime"].dt.minute == 0
    rep["registros_no_horarios"] = int((~en_punto).sum())
    df = df[en_punto].copy()

    rep["marcas_duplicadas"] = int(df["datetime"].duplicated().sum())
    df = df.groupby("datetime")[NUMERICAS].median()

    df[FUENTES_NUEVAS] = df[FUENTES_NUEVAS].fillna(0)

    atip = {}
    for c in ["demand_mw", "generation_mw"]:
        m = _atipicos_por_hora(df[c]) | (df[c] < 1000) | (df[c] > 25000)
        atip[c] = int(m.sum())
        df.loc[m, c] = np.nan
    # Racionamiento: no puede superar la mitad de la demanda ni 8.000 MW (máx. plausible ~4.000 MW).
    m = (df["load_shedding"] > 0.5 * df["demand_mw"]) | (df["load_shedding"] > 8000)
    atip["load_shedding"] = int(m.sum())
    df.loc[m, "load_shedding"] = np.nan
    for c in FUENTES:
        # Una fuente no puede aportar más que el total, ni triplicar su nivel habitual a esa hora.
        ref = df[c].groupby(df.index.hour).transform(
            lambda g: g.rolling(15, center=True, min_periods=5).median()
        )
        m = (df[c] > df["generation_mw"]) | ((ref > 50) & (df[c] > 3 * ref))
        atip[c] = int(m.sum())
        df.loc[m, c] = np.nan
    rep["atipicos_marcados"] = atip

    idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    rep["horas_esperadas"] = len(idx)
    rep["horas_faltantes"] = int(len(idx) - len(df))
    df = df.reindex(idx)
    df.index.name = "datetime"
    df = df.interpolate(limit=max_hueco, limit_area="inside")
    rep["horas_sin_demanda_tras_interpolar"] = int(df["demand_mw"].isna().sum())
    return df, rep
