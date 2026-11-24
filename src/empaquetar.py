"""Entrena los modelos finales y los guarda DENTRO del paquete modelo/pgcb_modelo (Entrega 3).

    python -m src.empaquetar                     # entrena, evalúa en prueba y guarda el modelo
    cd modelo && python -m build --wheel         # construye el paquete instalable (.whl)

Usa exactamente la configuración que ganó en validación en la Entrega 2 (ver MLflow):
- Demanda: FINAL-lightgbm-basico_festivos-relativo (experimento pgcb-pronostico-demanda).
- Alerta:  lightgbm-completo-nl7-mc400 con umbral para 80 % de sensibilidad (experimento pgcb-alerta-deficit).
Ambos se reentrenan con entrenamiento + validación y se evalúan en prueba (últimos 12 meses).

Si MLFLOW_TRACKING_URI está definida, la corrida queda registrada en el experimento "pgcb-paquete"
con la versión del paquete, sus métricas y el archivo del modelo.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import features as F
from src.alerta import (DEMANDA, construir_alerta, metricas_alerta, preparar_datos,
                        umbral_para_sensibilidad)
from src.entrenar import BASE_REL, DATOS, ModeloDemanda, construir_modelo, metricas, preparar

PAQUETE = Path("modelo/pgcb_modelo")
VERSION = (PAQUETE / "VERSION").read_text().strip()
ALERTA = {"tipo": "lightgbm", "conjunto": "completo", "params": {"num_leaves": 7, "min_child_samples": 400}}


def entrenar_demanda(X: pd.DataFrame):
    tr, va, te = F.particionar(preparar(X))
    trva = pd.concat([tr, va])
    cols = F.CONJUNTOS[DEMANDA["conjunto"]]
    m = ModeloDemanda(construir_modelo(DEMANDA["modelo"], cols, DEMANDA["params"]), cols, DEMANDA["objetivo"])
    m.fit(trva, trva[F.OBJETIVO])
    met = metricas(te[F.OBJETIVO], pd.Series(m.predict(te), index=te.index))
    return m, met


def entrenar_alerta(X: pd.DataFrame):
    tr, va, te = preparar_datos(X)
    # Umbral escogido en validación con el modelo entrenado solo con entrenamiento (igual que src/alerta.py)
    m = construir_alerta(ALERTA["tipo"], ALERTA["conjunto"], ALERTA["params"]).fit(tr, tr["y"])
    umbral = umbral_para_sensibilidad(va["y"], m.puntaje(va))
    trva = pd.concat([tr, va])
    m = construir_alerta(ALERTA["tipo"], ALERTA["conjunto"], ALERTA["params"]).fit(trva, trva["y"])
    met = metricas_alerta(te["y"], m.puntaje(te), umbral)
    return m, umbral, met


def main():
    X = F.construir(F.cargar(DATOS))
    print("Entrenando el modelo de demanda…")
    dem, met_dem = entrenar_demanda(X)
    print(f"  prueba: MAPE {met_dem['mape']:.2f} %, MAE {met_dem['mae']:.0f} MW")
    print("Entrenando el modelo de alerta (incluye pronósticos fuera de muestra, ≈2 min)…")
    ale, umbral, met_ale = entrenar_alerta(X)
    print(f"  prueba: sensibilidad {met_ale['sensibilidad']:.1f} %, precisión {met_ale['precision']:.1f} %, "
          f"umbral {umbral:.3f}")

    # Se guardan los estimadores de scikit-learn / LightGBM y su configuración, sin clases de src/,
    # para que el paquete funcione solo (por ejemplo, dentro del contenedor de la API).
    paquete = {
        "version": VERSION,
        "entrenado": dt.datetime.now().isoformat(timespec="seconds"),
        "datos_hasta": str(X.index.max()),
        "demanda": {"estimador": dem.estimador, "columnas": dem.columnas, "objetivo": dem.objetivo,
                    "base_rel": BASE_REL, "config": DEMANDA},
        "alerta": {"estimador": ale.est, "columnas": ale.columnas, "umbral": float(umbral), "config": ALERTA},
        "metricas_prueba": {**{f"demanda_{k}": v for k, v in met_dem.items()},
                            **{f"alerta_{k}": float(v) for k, v in met_ale.items()}},
    }
    (PAQUETE / "trained").mkdir(exist_ok=True)
    (PAQUETE / "datos").mkdir(exist_ok=True)
    for viejo in (PAQUETE / "trained").glob("modelo_pgcb_v*.joblib"):
        viejo.unlink()  # el paquete lleva solo la versión actual
    ruta = PAQUETE / "trained" / f"modelo_pgcb_v{VERSION}.joblib"
    joblib.dump(paquete, ruta, compress=3)
    print(f"Modelo guardado en {ruta} ({ruta.stat().st_size / 1e6:.1f} MB)")

    # Historia de ejemplo (últimos 14 días): la usan las pruebas, la API (ejemplo de Swagger) y el tablero.
    cols = ["demand_mw", "generation_mw", "load_shedding", "liquid_fuel"]
    ej = F.cargar(DATOS)[cols].tail(14 * 24).round(2).reset_index()
    ej.to_csv(PAQUETE / "datos" / "historia_ejemplo.csv", index=False)

    if os.getenv("MLFLOW_TRACKING_URI"):
        import mlflow
        mlflow.set_experiment("pgcb-paquete")
        with mlflow.start_run(run_name=f"pgcb-modelo-v{VERSION}"):
            mlflow.set_tag("version_paquete", VERSION)
            mlflow.log_params({"demanda": f"{DEMANDA['modelo']}-{DEMANDA['conjunto']}-{DEMANDA['objetivo']}",
                               "alerta": f"{ALERTA['tipo']}-{ALERTA['conjunto']}-nl7-mc400",
                               "umbral_alerta": round(float(umbral), 4)})
            mlflow.log_metrics({k: round(float(v), 4) for k, v in paquete["metricas_prueba"].items()
                                if np.isfinite(v)})
            mlflow.log_artifact(str(ruta), "paquete")
        print("Corrida registrada en MLflow (experimento pgcb-paquete).")


if __name__ == "__main__":
    main()
