"""Entrena un modelo de pronóstico de demanda a 24 h y registra el experimento en MLflow.

Ejemplos:
    python -m src.entrenar --modelo lightgbm --conjunto rezagos_festivos --objetivo relativo
    python -m src.entrenar --modelo ridge --conjunto completo --params '{"alpha": 10}'

El servidor de MLflow se toma de la variable de entorno MLFLOW_TRACKING_URI
(por ejemplo http://<IP-EC2>:5000). Si no está definida, se guarda en ./mlruns.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import joblib
import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import features as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATOS = os.getenv("PGCB_DATOS", "data/processed/pgcb_horario_limpio.csv")
EXPERIMENTO = "pgcb-pronostico-demanda"
CATEGORICAS = ["hora", "dia_semana", "mes"]
BASE_REL = "dem_media_7d_misma_hora"  # referencia para el objetivo relativo


# ---------------------------------------------------------------- métricas
def metricas(y: pd.Series, yhat: pd.Series) -> dict:
    e = y - yhat
    return {
        "mape": float((e.abs() / y).mean() * 100),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "sesgo": float(e.mean()),
    }


def metricas_alerta(pred: pd.Series, gen_disp: pd.Series, racion: pd.Series, margen: float = 0.0) -> dict:
    """Alerta = demanda pronosticada supera la generación disponible estimada (+ margen en MW).

    Se compara contra las horas que realmente tuvieron racionamiento (load_shedding > 0).
    """
    alerta = pred > (gen_disp + margen)
    real = racion > 0
    vp = int((alerta & real).sum())
    sens = vp / max(int(real.sum()), 1)
    prec = vp / max(int(alerta.sum()), 1)
    f1 = 2 * sens * prec / (sens + prec) if sens + prec else 0.0
    return {"alerta_sensibilidad": sens * 100, "alerta_precision": prec * 100, "alerta_f1": f1 * 100}


# ---------------------------------------------------------------- modelos
class Ingenuo:
    """Línea base: copia un rezago de la demanda (no se entrena)."""

    def __init__(self, columna: str):
        self.columna = columna

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[self.columna].to_numpy()


def construir_modelo(nombre: str, columnas: list[str], params: dict):
    if nombre == "ingenuo":
        return Ingenuo(params.get("columna", "dem_lag24"))
    if nombre == "ridge":
        cat = [c for c in CATEGORICAS if c in columnas]
        num = [c for c in columnas if c not in cat]
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), num),
        ])
        return Pipeline([("prep", prep), ("modelo", Ridge(alpha=params.get("alpha", 1.0)))])
    if nombre == "random_forest":
        p = {"n_estimators": 200, "max_depth": 18, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42} | params
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("modelo", RandomForestRegressor(**p))])
    if nombre == "lightgbm":
        p = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 40,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": 42,
             "verbose": -1} | params
        return lgb.LGBMRegressor(**p)
    raise ValueError(f"Modelo desconocido: {nombre}")


class ModeloDemanda:
    """Envuelve el estimador y el objetivo (nivel o relativo) para predecir siempre en MW."""

    def __init__(self, estimador, columnas: list[str], objetivo: str):
        self.estimador, self.columnas, self.objetivo = estimador, columnas, objetivo

    def _y(self, X, y):
        return y / X[BASE_REL] if self.objetivo == "relativo" else y

    def fit(self, X, y):
        self.estimador.fit(X[self.columnas], self._y(X, y))
        return self

    def predict(self, X) -> np.ndarray:
        p = np.asarray(self.estimador.predict(X[self.columnas]), dtype=float)
        return p * X[BASE_REL].to_numpy() if self.objetivo == "relativo" else p


# ---------------------------------------------------------------- utilidades
def preparar(X: pd.DataFrame) -> pd.DataFrame:
    """Filas válidas: objetivo conocido y los rezagos básicos disponibles (misma base para todos los modelos)."""
    return X.dropna(subset=[F.OBJETIVO, "dem_lag24", BASE_REL, "generation_mw"])


def grafico_semana(idx, y, yhat, ruta):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(idx, y, color="black", lw=1.3, label="Demanda real")
    ax.plot(idx, yhat, color="#2f6db5", lw=1.3, ls="--", label="Pronóstico")
    ax.set_ylabel("MW")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def importancias(modelo: ModeloDemanda) -> pd.Series | None:
    est = modelo.estimador
    if isinstance(est, lgb.LGBMRegressor):
        imp = est.booster_.feature_importance("gain")
    elif isinstance(est, Pipeline) and isinstance(est[-1], RandomForestRegressor):
        imp = est[-1].feature_importances_
    else:
        return None
    s = pd.Series(imp, index=modelo.columnas)
    return (s / s.sum()).sort_values(ascending=False)


# ---------------------------------------------------------------- ejecución
def ejecutar(modelo: str, conjunto: str, objetivo: str = "nivel", params: dict | None = None,
             nombre_run: str | None = None, X: pd.DataFrame | None = None, final: bool = False) -> dict:
    """Entrena, evalúa en validación y prueba, y registra todo en MLflow.

    final=True reentrena con entrenamiento + validación antes de evaluar en prueba
    (es el modelo que se despliega) y lo guarda en el Model Registry.
    """
    params = params or {}
    if X is None:
        X = F.construir(F.cargar(DATOS))
    X = preparar(X)
    tr, va, te = F.particionar(X)
    columnas = F.CONJUNTOS[conjunto]

    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name=nombre_run or f"{modelo}-{conjunto}-{objetivo}"):
        mlflow.set_tags({"modelo": modelo, "conjunto": conjunto, "objetivo": objetivo,
                         "tipo": "final" if final else "experimento"})
        mlflow.log_params({"modelo": modelo, "conjunto_variables": conjunto, "objetivo": objetivo,
                           "n_variables": len(columnas), "horizonte_h": F.HORIZONTE,
                           "n_entrenamiento": len(tr) + (len(va) if final else 0),
                           "inicio_validacion": str(va.index.min()), "inicio_prueba": str(te.index.min()),
                           **{f"hp_{k}": v for k, v in params.items()}})
        mlflow.log_text("\n".join(columnas), "variables.txt")

        datos_fit = pd.concat([tr, va]) if final else tr
        m = ModeloDemanda(construir_modelo(modelo, columnas, params), columnas, objetivo)
        m.fit(datos_fit, datos_fit[F.OBJETIVO])

        res = {}
        for nombre, parte in [("val", va), ("prueba", te)]:
            pred = pd.Series(m.predict(parte), index=parte.index)
            met = metricas(parte[F.OBJETIVO], pred)
            # Generación disponible estimada = generación de la misma hora del día anterior (definición Entrega 1)
            gen_disp = X["generation_mw"].shift(24).reindex(parte.index)
            met |= metricas_alerta(pred, gen_disp, parte["load_shedding"])
            for k, v in met.items():
                res[f"{nombre}_{k}"] = v
            mlflow.log_metrics({f"{nombre}_{k}": round(v, 4) for k, v in met.items()})

        with tempfile.TemporaryDirectory() as tmp:
            semana = te.index[-24 * 7:]
            ruta = os.path.join(tmp, "semana_prueba.png")
            grafico_semana(semana, te.loc[semana, F.OBJETIVO], m.predict(te.loc[semana]), ruta)
            mlflow.log_artifact(ruta, "graficos")
            imp = importancias(m)
            if imp is not None:
                ruta = os.path.join(tmp, "importancia_variables.csv")
                imp.rename("importancia").to_csv(ruta)
                mlflow.log_artifact(ruta, "graficos")
                res["importancias"] = imp
            if modelo != "ingenuo":
                ruta = os.path.join(tmp, "modelo.joblib")
                joblib.dump(m, ruta)
                mlflow.log_artifact(ruta, "modelo")
                ej = te[list(dict.fromkeys(columnas + [BASE_REL]))].head(5)
                firma = infer_signature(ej, m.predict(ej))
                mlflow.pyfunc.log_model(
                    name="modelo_pyfunc", python_model=_Pyfunc(), artifacts={"modelo": ruta},
                    signature=firma, code_paths=["src"],
                    registered_model_name="pgcb-demanda-24h" if final else None,
                )
        res["run_id"] = mlflow.active_run().info.run_id
        res["modelo_obj"] = m
        return res


class _Pyfunc(mlflow.pyfunc.PythonModel):
    """Permite cargar el modelo desde MLflow (lo usará la API en la Entrega 3)."""

    def load_context(self, context):
        self.m = joblib.load(context.artifacts["modelo"])

    def predict(self, context, model_input, params=None):
        return self.m.predict(model_input)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, choices=["ingenuo", "ridge", "random_forest", "lightgbm"])
    ap.add_argument("--conjunto", default="rezagos_festivos", choices=list(F.CONJUNTOS))
    ap.add_argument("--objetivo", default="nivel", choices=["nivel", "relativo"])
    ap.add_argument("--params", default="{}", help="hiperparámetros en JSON")
    ap.add_argument("--final", action="store_true", help="reentrenar con train+val y registrar el modelo")
    a = ap.parse_args()
    r = ejecutar(a.modelo, a.conjunto, a.objetivo, json.loads(a.params), final=a.final)
    print({k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
