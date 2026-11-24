"""Pronóstico de las próximas 24 horas a partir de la historia horaria reciente.

    import pandas as pd
    from pgcb_modelo import pronosticar
    historia = pd.read_csv("historia.csv", parse_dates=["datetime"])   # al menos 8 días
    res = pronosticar(historia)

Devuelve una fila por cada una de las 24 horas siguientes a la última hora de la historia:
demanda pronosticada, generación disponible estimada, déficit, probabilidad de racionamiento y alerta.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pgcb_modelo import features as F

RUTA_MODELO = Path(__file__).parent / "trained"
# Columnas que necesita el modelo (el resto del CSV se ignora)
COLUMNAS_ENTRADA = ["datetime", "demand_mw", "generation_mw", "load_shedding", "liquid_fuel"]
# 7 días para la media de la misma hora + 1 día de rezago para el historial de racionamiento
HISTORIA_MIN_HORAS = 192
HORIZONTE = 24


@lru_cache(maxsize=1)
def cargar_modelo() -> dict:
    """Carga el archivo entrenado que viene dentro del paquete (una sola vez)."""
    from pgcb_modelo import __version__
    ruta = RUTA_MODELO / f"modelo_pgcb_v{__version__}.joblib"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Corra primero: python -m src.empaquetar")
    return joblib.load(ruta)


def _validar(historia: pd.DataFrame) -> pd.DataFrame:
    faltan = [c for c in COLUMNAS_ENTRADA if c not in historia.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en la historia: {faltan}")
    h = historia[COLUMNAS_ENTRADA].copy()
    h["datetime"] = pd.to_datetime(h["datetime"])
    if h["datetime"].duplicated().any():
        raise ValueError("Hay horas repetidas en la historia.")
    h = h.set_index("datetime").sort_index().asfreq("h")  # las horas faltantes quedan vacías
    h = h.iloc[-HISTORIA_MIN_HORAS * 3:]  # no se necesita más historia que esta
    if len(h) < HISTORIA_MIN_HORAS:
        raise ValueError(f"Se necesitan al menos {HISTORIA_MIN_HORAS} horas seguidas de historia "
                         f"(8 días); llegaron {len(h)}.")
    return h.astype(float)


def pronosticar(historia: pd.DataFrame, modelo: dict | None = None) -> pd.DataFrame:
    """Pronostica las 24 horas siguientes a la última hora de `historia`."""
    m = modelo or cargar_modelo()
    h = _validar(historia)
    futuro = pd.date_range(h.index[-1] + pd.Timedelta(hours=1), periods=HORIZONTE, freq="h")
    serie = h.reindex(h.index.append(futuro))
    X = F.construir(serie)

    # 1) Demanda: el modelo predice la razón frente a la media de los últimos 7 días a la misma hora
    d = m["demanda"]
    Xf = X.loc[futuro]
    if Xf[d["base_rel"]].isna().any():
        raise ValueError("La historia tiene demasiados vacíos en demand_mw en los últimos 7 días.")
    dem = np.asarray(d["estimador"].predict(Xf[d["columnas"]]), dtype=float)
    if d["objetivo"] == "relativo":
        dem = dem * Xf[d["base_rel"]].to_numpy()
    pred = pd.Series(np.nan, index=X.index)
    pred.loc[futuro] = dem

    # 2) Alerta: clasificador sobre la brecha pronosticada y el historial de racionamiento
    a = m["alerta"]
    A = F.variables_alerta(X, pred).loc[futuro]
    prob = a["estimador"].predict_proba(A[a["columnas"]])[:, 1]

    res = pd.DataFrame({
        "demanda_mw": dem,
        "generacion_disponible_mw": A["gen_disp"].to_numpy(),
        "deficit_mw": A["deficit_pron"].to_numpy(),
        "prob_racionamiento": prob,
        "alerta": prob >= a["umbral"],
    }, index=pd.DatetimeIndex(futuro, name="datetime"))
    return res
