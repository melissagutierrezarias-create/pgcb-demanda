"""Corre todos los experimentos de la Entrega 2 y registra cada uno en MLflow.

    export MLFLOW_TRACKING_URI=http://<IP-EC2>:5000   # o déjelo vacío para usar ./mlruns
    python -m src.experimentos            # todos los experimentos + modelo final
    python -m src.experimentos --grupo lightgbm

Al final escoge el modelo con menor MAPE en VALIDACIÓN (la prueba no se usa para escoger),
lo reentrena con entrenamiento + validación, lo evalúa en prueba y lo registra como
"pgcb-demanda-24h" en el Model Registry.
"""
from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd

from src import features as F
from src.entrenar import DATOS, ejecutar

GRUPOS = {
    # 1. Líneas base (las mismas de la Entrega 1)
    "ingenuo": [
        ("ingenuo", "basico", "nivel", {"columna": "dem_lag24"}, "base-t24"),
        ("ingenuo", "rezagos_festivos", "nivel", {"columna": "dem_lag168"}, "base-t168"),
        ("ingenuo", "basico", "nivel", {"columna": "dem_media_7d_misma_hora"}, "base-media7d"),
    ],
    # 2. Regresión lineal regularizada: ¿qué conjunto de variables sirve?
    "ridge": [("ridge", c, "nivel", {"alpha": 1.0}, None) for c in F.CONJUNTOS]
    + [("ridge", "rezagos_festivos", "relativo", {"alpha": 1.0}, None),
       ("ridge", "rezagos_festivos", "nivel", {"alpha": 10.0}, "ridge-rezagos_festivos-nivel-a10")],
    # 3. Random forest
    "random_forest": [
        ("random_forest", "basico_festivos", "relativo", {}, None),
        ("random_forest", "rezagos_festivos", "relativo", {}, None),
        ("random_forest", "basico_festivos", "nivel", {}, None),
    ],
    # 4. LightGBM: conjuntos de variables, objetivo y luego hiperparámetros
    "lightgbm": [("lightgbm", c, "relativo", {"num_leaves": 15, "min_child_samples": 200,
                                               "learning_rate": 0.03}, None)
                 for c in F.CONJUNTOS if c != "calendario"]
    + [("lightgbm", "basico_festivos", "nivel", {"num_leaves": 15, "min_child_samples": 200,
                                                  "learning_rate": 0.03}, None),
       ("lightgbm", "rezagos_festivos", "relativo", {}, "lightgbm-rezagos_festivos-relativo-default")]
    + [("lightgbm", "basico_festivos", "relativo",
        {"num_leaves": nl, "min_child_samples": mc, "learning_rate": 0.03},
        f"lgbm-grid-nl{nl}-mc{mc}")
       for nl in (7, 31, 63) for mc in (50, 400)],
}


def main(grupos: list[str], final: bool = True):
    X = F.construir(F.cargar(DATOS))
    filas = []
    for g in grupos:
        for modelo, conjunto, objetivo, params, nombre in GRUPOS[g]:
            r = ejecutar(modelo, conjunto, objetivo, params, nombre, X=X)
            filas.append({"grupo": g, "modelo": modelo, "conjunto": conjunto, "objetivo": objetivo,
                          "params": params, "run": nombre or f"{modelo}-{conjunto}-{objetivo}",
                          **{k: v for k, v in r.items() if isinstance(v, float)}})
            print(f"{filas[-1]['run']:45s} val MAPE {r['val_mape']:.2f}  prueba MAPE {r['prueba_mape']:.2f}")
    tabla = pd.DataFrame(filas)
    os.makedirs("reports", exist_ok=True)
    tabla.round(3).to_csv("reports/resultados_experimentos.csv", index=False)

    if final:
        mejor = tabla[tabla.modelo != "ingenuo"].sort_values("val_mape").iloc[0]
        print("\nMejor en validación:", mejor["run"])
        r = ejecutar(mejor.modelo, mejor.conjunto, mejor.objetivo, mejor.params,
                     "FINAL-" + mejor["run"], X=X, final=True)
        os.makedirs("models", exist_ok=True)
        joblib.dump(r["modelo_obj"], "models/modelo_demanda.joblib")
        print({k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
    return tabla


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grupo", nargs="*", default=list(GRUPOS))
    ap.add_argument("--sin-final", action="store_true")
    a = ap.parse_args()
    main(a.grupo, final=not a.sin_final)
