"""Modelo de alerta de déficit a 24 h: ¿en qué horas de mañana habrá racionamiento?

La alerta usa el pronóstico de demanda (src.entrenar) y la generación disponible estimada
(la generación de la misma hora del día anterior, definición de la Entrega 1). Se comparan
reglas y clasificadores, y todo se registra en MLflow, en el experimento "pgcb-alerta-deficit".

    export MLFLOW_TRACKING_URI=http://<IP-EC2>:5000     # ver docs/guia_ec2_mlflow.md
    python -m src.alerta                                # experimentos + modelo final + figuras

Variable objetivo: hay racionamiento en la hora (load_shedding > 0).

Umbral de alerta: la pregunta de negocio prioriza no dejar pasar horas con déficit. Por eso,
en cada modelo el umbral se escoge en VALIDACIÓN como el más alto que detecta al menos el
80 % de las horas con racionamiento (la sensibilidad de la regla de la Entrega 1). Con esa
sensibilidad fija, el mejor modelo es el que da menos falsas alarmas (mayor precisión).
La prueba (últimos 12 meses) nunca se usa para escoger.
"""
from __future__ import annotations

import argparse
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import features as F
from src.entrenar import DATOS, ModeloDemanda, construir_modelo, preparar

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPERIMENTO = "pgcb-alerta-deficit"
SENS_OBJETIVO = 0.80  # sensibilidad mínima exigida en validación (regla de la Entrega 1: 79,9 %)
SALIDA_FIG = "reports/figures"

# Modelo de demanda escogido en los experimentos de demanda (FINAL-lightgbm-basico_festivos-relativo)
DEMANDA = {"modelo": "lightgbm", "conjunto": "basico_festivos", "objetivo": "relativo",
           "params": {"num_leaves": 15, "min_child_samples": 200, "learning_rate": 0.03}}

# Conjuntos de variables del clasificador (todas disponibles 24 h antes)
DEFICIT = ["deficit_pron", "deficit_rel", "deficit_vs_gen_max"]
HISTORIAL = ["racion_lag24", "racion_media_7d", "racion_horas_7d", "liquid_fuel_share_lag24"]
CALENDARIO = ["hora", "mes", "dia_semana", "festivo", "dias_a_eid"]
CONJUNTOS_ALERTA = {
    "deficit": DEFICIT,                                   # solo la brecha pronosticada
    "deficit_historial": DEFICIT + HISTORIAL,             # + racionamiento de los días anteriores
    "completo": DEFICIT + HISTORIAL + CALENDARIO,         # + calendario
}
CATEGORICAS = ["hora", "mes", "dia_semana"]


# ---------------------------------------------------------------- pronóstico de demanda sin fuga
def _modelo_demanda():
    d = DEMANDA
    cols = F.CONJUNTOS[d["conjunto"]]
    return ModeloDemanda(construir_modelo(d["modelo"], cols, d["params"]), cols, d["objetivo"])


def pronosticos_demanda(X: pd.DataFrame, tr, va, te) -> pd.Series:
    """Pronóstico de demanda fuera de muestra para cada hora.

    - Entrenamiento: validación temporal por años (cada año se pronostica con un modelo
      entrenado solo con los años anteriores). Así el clasificador aprende con errores de
      pronóstico realistas, no con el ajuste dentro de muestra.
    - Validación: modelo entrenado con el periodo de entrenamiento.
    - Prueba: modelo entrenado con entrenamiento + validación (el modelo desplegado).
    """
    pred = pd.Series(np.nan, index=X.index)
    cortes = pd.date_range(tr.index.min() + pd.DateOffset(years=2), va.index.min(), freq="12MS")
    cortes = list(cortes) + [va.index.min()]
    for ini, fin in zip(cortes[:-1], cortes[1:]):
        fit = tr.loc[tr.index < ini]
        bloque = tr.loc[(tr.index >= ini) & (tr.index < fin)]
        if len(bloque):
            m = _modelo_demanda().fit(fit, fit[F.OBJETIVO])
            pred.loc[bloque.index] = m.predict(bloque)
    m = _modelo_demanda().fit(tr, tr[F.OBJETIVO])
    pred.loc[va.index] = m.predict(va)
    trva = pd.concat([tr, va])
    m = _modelo_demanda().fit(trva, trva[F.OBJETIVO])
    pred.loc[te.index] = m.predict(te)
    return pred


def variables_alerta(X: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """Agrega al conjunto de variables la brecha pronosticada y el historial de racionamiento."""
    A = X.copy()
    A["dem_pron"] = pred
    A["gen_disp"] = X["generation_mw"].shift(24)  # definición de la Entrega 1
    A["deficit_pron"] = A["dem_pron"] - A["gen_disp"]
    A["deficit_rel"] = A["deficit_pron"] / A["gen_disp"]
    A["deficit_vs_gen_max"] = A["dem_pron"] - X["gen_max_7d_misma_hora"]
    hay = (X["load_shedding"] > 0).astype(float).where(X["load_shedding"].notna())
    A["racion_horas_7d"] = hay.shift(24).rolling(168, min_periods=120).mean()
    A["y"] = hay
    return A


# ---------------------------------------------------------------- modelos de alerta
class Regla:
    """Alerta por regla: puntaje = demanda (pronosticada o ingenua) menos generación disponible."""

    def __init__(self, columna_demanda: str):
        self.col = columna_demanda

    def fit(self, A, y):
        return self

    def puntaje(self, A) -> np.ndarray:
        return (A[self.col] - A["gen_disp"]).to_numpy(dtype=float)


class Clasificador:
    def __init__(self, estimador, columnas):
        self.est, self.columnas = estimador, columnas

    def fit(self, A, y):
        self.est.fit(A[self.columnas], y)
        return self

    def puntaje(self, A) -> np.ndarray:
        return self.est.predict_proba(A[self.columnas])[:, 1]


def construir_alerta(tipo: str, conjunto: str | None, params: dict):
    if tipo == "regla_t24":
        return Regla("dem_lag24")
    if tipo == "regla_modelo":
        return Regla("dem_pron")
    cols = CONJUNTOS_ALERTA[conjunto]
    if tipo == "logistica":
        cat = [c for c in CATEGORICAS if c in cols]
        num = [c for c in cols if c not in cat]
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), num),
        ])
        est = Pipeline([("prep", prep), ("modelo", LogisticRegression(
            C=params.get("C", 1.0), class_weight="balanced", max_iter=2000))])
        return Clasificador(est, cols)
    if tipo == "lightgbm":
        p = {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 200,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": 42,
             "verbose": -1} | params
        return Clasificador(lgb.LGBMClassifier(**p), cols)
    raise ValueError(f"Tipo desconocido: {tipo}")


# ---------------------------------------------------------------- métricas
def umbral_para_sensibilidad(y, s, objetivo=SENS_OBJETIVO) -> float:
    """Umbral más alto cuya sensibilidad es >= objetivo (menos falsas alarmas a esa sensibilidad)."""
    prec, rec, umb = precision_recall_curve(y, s)
    ok = np.where(rec[:-1] >= objetivo)[0]
    return float(umb[ok[-1]]) if len(ok) else float(umb[0])


def metricas_alerta(y, s, umbral) -> dict:
    y = np.asarray(y).astype(bool)
    a = np.asarray(s) >= umbral
    vp, fp, fn = int((a & y).sum()), int((a & ~y).sum()), int((~a & y).sum())
    sens = vp / max(vp + fn, 1)
    prec = vp / max(vp + fp, 1)
    f1 = 2 * sens * prec / (sens + prec) if sens + prec else 0.0
    p, r, _ = precision_recall_curve(y, s)
    return {"sensibilidad": sens * 100, "precision": prec * 100, "f1": f1 * 100,
            "tasa_alertas": a.mean() * 100, "auc_roc": roc_auc_score(y, s),
            "auc_pr": average_precision_score(y, s),
            "precision_a_80_sens": float(p[:-1][r[:-1] >= 0.8].max() * 100),
            "falsas_alarmas": fp, "horas_no_detectadas": fn}


# ---------------------------------------------------------------- ejecución
def preparar_datos(X: pd.DataFrame | None = None):
    if X is None:
        X = F.construir(F.cargar(DATOS))
    X = preparar(X)
    tr, va, te = F.particionar(X)
    A = variables_alerta(X, pronosticos_demanda(X, tr, va, te))
    A = A.dropna(subset=["y", "dem_pron", "gen_disp", "dem_lag24"])
    ini_va, ini_te = va.index.min(), te.index.min()  # mismas fechas de corte del modelo de demanda
    return A.loc[A.index < ini_va], A.loc[(A.index >= ini_va) & (A.index < ini_te)], A.loc[A.index >= ini_te]


def ejecutar(tipo, conjunto=None, params=None, nombre=None, datos=None, final=False) -> dict:
    params = params or {}
    tr, va, te = datos
    mlflow.set_experiment(EXPERIMENTO)
    nombre = nombre or "-".join(x for x in [tipo, conjunto] if x)
    with mlflow.start_run(run_name=("FINAL-" + nombre) if final else nombre):
        cols = CONJUNTOS_ALERTA.get(conjunto, []) if conjunto else []
        mlflow.set_tags({"tipo_modelo": tipo, "conjunto": conjunto or "-",
                         "tipo": "final" if final else "experimento"})
        mlflow.log_params({"tipo_modelo": tipo, "conjunto_variables": conjunto or "-",
                           "n_variables": len(cols), "sensibilidad_objetivo": SENS_OBJETIVO,
                           "modelo_demanda": f"{DEMANDA['modelo']}-{DEMANDA['conjunto']}-{DEMANDA['objetivo']}",
                           "n_entrenamiento": len(tr) + (len(va) if final else 0),
                           "inicio_validacion": str(va.index.min()), "inicio_prueba": str(te.index.min()),
                           **{f"hp_{k}": v for k, v in params.items()}})
        if cols:
            mlflow.log_text("\n".join(cols), "variables.txt")

        # 1) Umbral escogido en validación con el modelo entrenado solo con entrenamiento
        m = construir_alerta(tipo, conjunto, params).fit(tr, tr["y"])
        s_va = m.puntaje(va)
        # Las reglas de la Entrega 1 usan umbral 0 (alerta si la demanda supera la generación disponible)
        # (estrictamente mayor: se suma un margen mínimo para que una diferencia de 0 MW no alerte)
        umbral = (float(params["umbral_fijo"]) + 1e-6 if "umbral_fijo" in params
                  else umbral_para_sensibilidad(va["y"], s_va))
        # 2) Final: reentrena con entrenamiento + validación y conserva el umbral
        if final:
            trva = pd.concat([tr, va])
            m = construir_alerta(tipo, conjunto, params).fit(trva, trva["y"])
        mlflow.log_param("umbral", round(umbral, 4))

        res = {"umbral": umbral}
        for parte_nombre, parte, s in [("val", va, s_va), ("prueba", te, m.puntaje(te))]:
            met = metricas_alerta(parte["y"], s, umbral)
            mlflow.log_metrics({f"{parte_nombre}_{k}": round(float(v), 4) for k, v in met.items()})
            res |= {f"{parte_nombre}_{k}": float(v) for k, v in met.items()}
            res[f"puntaje_{parte_nombre}"] = pd.Series(s, index=parte.index)

        with tempfile.TemporaryDirectory() as tmp:
            ruta = os.path.join(tmp, "curva_pr_prueba.png")
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            p, r, _ = precision_recall_curve(te["y"], res["puntaje_prueba"])
            ax.plot(r * 100, p * 100, lw=1.5)
            ax.scatter(res["prueba_sensibilidad"], res["prueba_precision"], color="#eb6834", zorder=3)
            ax.set_xlabel("Sensibilidad (%)")
            ax.set_ylabel("Precisión (%)")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(ruta, dpi=130)
            plt.close(fig)
            mlflow.log_artifact(ruta, "graficos")
            if isinstance(m, Clasificador) and isinstance(m.est, lgb.LGBMClassifier):
                imp = pd.Series(m.est.booster_.feature_importance("gain"), index=m.columnas)
                imp = (imp / imp.sum()).sort_values(ascending=False)
                ruta = os.path.join(tmp, "importancia_variables.csv")
                imp.rename("importancia").to_csv(ruta)
                mlflow.log_artifact(ruta, "graficos")
                res["importancias"] = imp
            ruta = os.path.join(tmp, "modelo_alerta.joblib")
            joblib.dump({"modelo": m, "umbral": umbral}, ruta)
            mlflow.log_artifact(ruta, "modelo")
            if final:
                ej = te[cols or ["dem_pron", "gen_disp", "dem_lag24"]].head(5)
                mlflow.pyfunc.log_model(
                    name="modelo_pyfunc", python_model=_Pyfunc(), artifacts={"modelo": ruta},
                    signature=infer_signature(ej, np.zeros(len(ej), dtype=int)), code_paths=["src"],
                    registered_model_name="pgcb-alerta-deficit")
        res["run_id"] = mlflow.active_run().info.run_id
        res["modelo_obj"] = m
        return res


class _Pyfunc(mlflow.pyfunc.PythonModel):
    """Devuelve 1 si la hora queda en alerta de déficit (puntaje >= umbral)."""

    def load_context(self, context):
        d = joblib.load(context.artifacts["modelo"])
        self.m, self.umbral = d["modelo"], d["umbral"]

    def predict(self, context, model_input, params=None):
        return (self.m.puntaje(model_input) >= self.umbral).astype(int)


EXPERIMENTOS = [
    ("regla_t24", None, {"umbral_fijo": 0}, "regla-E1-t24"),          # regla de la Entrega 1, tal cual
    ("regla_modelo", None, {"umbral_fijo": 0}, "regla-pronostico-lgbm"),  # misma regla con el pronóstico
    ("regla_t24", None, {}, "regla-E1-t24-umbral80"),                 # reglas con umbral ajustado a 80 %
    ("regla_modelo", None, {}, "regla-pronostico-lgbm-umbral80"),
    *[("logistica", c, {"C": 1.0}, None) for c in CONJUNTOS_ALERTA],
    *[("lightgbm", c, {}, None) for c in CONJUNTOS_ALERTA],
    ("lightgbm", "completo", {"num_leaves": 31, "min_child_samples": 50}, "lightgbm-completo-nl31-mc50"),
    ("lightgbm", "completo", {"num_leaves": 7, "min_child_samples": 400}, "lightgbm-completo-nl7-mc400"),
]


# ---------------------------------------------------------------- figuras del reporte
def figuras(datos, resultados: dict, mejor: str):
    _, _, te = datos
    os.makedirs(SALIDA_FIG, exist_ok=True)
    colores = {"regla-E1-t24": "#898781", "regla-pronostico-lgbm": "#2a78d6", mejor: "#eb6834"}
    etiquetas = {"regla-E1-t24": "Regla Entrega 1 (t-24)", "regla-pronostico-lgbm": "Regla con pronóstico",
                 mejor: f"Mejor clasificador ({mejor})"}

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for n in colores:
        r = resultados[n]
        p, rc, _ = precision_recall_curve(te["y"], r["puntaje_prueba"])
        ax.plot(rc * 100, p * 100, color=colores[n], lw=1.6, label=etiquetas[n])
        ax.scatter(r["prueba_sensibilidad"], r["prueba_precision"], color=colores[n], s=35, zorder=3)
    ax.axvline(80, color="#c3c2b7", ls="--", lw=1)
    ax.set_xlabel("Sensibilidad: % de horas con racionamiento detectadas")
    ax.set_ylabel("Precisión: % de alertas correctas")
    ax.set_xlim(40, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA_FIG, "fig_alerta_precision_sensibilidad.png"), dpi=170)
    plt.close(fig)

    meses = te.index.to_period("M")
    filas = []
    for n in ["regla-E1-t24", mejor]:
        r = resultados[n]
        a = r["puntaje_prueba"] >= r["umbral"]
        df = pd.DataFrame({"y": te["y"].astype(bool), "a": a, "mes": meses})
        g = df.groupby("mes").apply(lambda d: pd.Series({
            "sens": (d.a & d.y).sum() / max(d.y.sum(), 1) * 100,
            "prec": (d.a & d.y).sum() / max(d.a.sum(), 1) * 100,
            "horas_racion": d.y.mean() * 100}), include_groups=False)
        g["modelo"] = n
        filas.append(g)
    g = pd.concat(filas).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    x = np.arange(g["mes"].nunique())
    etiquetas_mes = [str(m) for m in g["mes"].unique()]
    for n, desp in [("regla-E1-t24", -0.2), (mejor, 0.2)]:
        d = g[g.modelo == n]
        axs[0].bar(x + desp, d["sens"], width=0.4, color=colores[n], label=etiquetas[n])
        axs[1].bar(x + desp, d["prec"], width=0.4, color=colores[n])
    axs[0].set_title("Sensibilidad por mes (%)", fontsize=10)
    axs[1].set_title("Precisión por mes (%)", fontsize=10)
    for ax in axs:
        ax.set_xticks(x, etiquetas_mes, rotation=60, fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(SALIDA_FIG, "fig_alerta_por_mes.png"), dpi=170)
    plt.close(fig)


def main(final: bool = True):
    datos = preparar_datos()
    tr, va, te = datos
    print(f"Horas: entrenamiento {len(tr)}, validación {len(va)}, prueba {len(te)}; "
          f"con racionamiento en prueba: {te['y'].mean():.1%}")
    filas, resultados = [], {}
    for tipo, conjunto, params, nombre in EXPERIMENTOS:
        n = nombre or "-".join(x for x in [tipo, conjunto] if x)
        r = ejecutar(tipo, conjunto, params, n, datos)
        resultados[n] = r
        filas.append({"run": n, "tipo": tipo, "conjunto": conjunto or "-", **{
            k: v for k, v in r.items() if isinstance(v, float)}})
        print(f"{n:32s} val sens {r['val_sensibilidad']:5.1f} prec {r['val_precision']:5.1f} | "
              f"prueba sens {r['prueba_sensibilidad']:5.1f} prec {r['prueba_precision']:5.1f} "
              f"F1 {r['prueba_f1']:5.1f}")
    tabla = pd.DataFrame(filas)
    os.makedirs("reports", exist_ok=True)
    tabla.round(3).to_csv("reports/resultados_alerta.csv", index=False)

    # Escoge con VALIDACIÓN: mayor precisión con sensibilidad >= 80 % (empate: mayor AUC-PR)
    candidatos = tabla[tabla.tipo.isin(["logistica", "lightgbm"])]
    mejor = candidatos.sort_values(["val_precision", "val_auc_pr"], ascending=False).iloc[0]
    print("\nMejor en validación:", mejor["run"])
    figuras(datos, resultados, mejor["run"])
    if final:
        p = next(e[2] for e in EXPERIMENTOS if (e[3] or "-".join(x for x in e[:2] if x)) == mejor["run"])
        r = ejecutar(mejor.tipo, mejor.conjunto, p, mejor["run"], datos, final=True)
        os.makedirs("models", exist_ok=True)
        joblib.dump({"modelo": r["modelo_obj"], "umbral": r["umbral"]}, "models/modelo_alerta.joblib")
        print({k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
    return tabla


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-final", action="store_true")
    a = ap.parse_args()
    main(final=not a.sin_final)
