"""Figuras de evaluación del modelo de demanda (Figuras 1 y 2 del reporte de la Entrega 2).

    python -m src.figuras_demanda     # requiere haber corrido src.experimentos
"""
from __future__ import annotations

import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

from src import features as F
from src.entrenar import DATOS, preparar

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SALIDA = "reports/figures"
AZUL, NARANJA, VERDE, GRIS, TINTA = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#c3c2b7", "axes.axisbelow": True})


def guardar(fig, nombre):
    os.makedirs(SALIDA, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, nombre), dpi=170)
    plt.close(fig)


def fig_comparacion(mape_final: float):
    t = pd.read_csv("reports/resultados_experimentos.csv")
    mejores = (t.sort_values("val_mape").groupby("modelo").head(1)
               .set_index("modelo").loc[["ingenuo", "ridge", "random_forest", "lightgbm"]])
    etiquetas = ["Línea base t-24", "Ridge", "Random forest", "LightGBM", "LightGBM final\n(train + val)"]
    x = np.arange(len(mejores))
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    ax.bar(len(mejores) + 0.2, mape_final, 0.38, color=NARANJA, edgecolor="white", linewidth=1.5)
    ax.text(len(mejores) + 0.2, mape_final + 0.08, f"{mape_final:.2f}".replace(".", ","), ha="center", fontsize=8.5,
            fontweight="bold")
    ax.bar(x - 0.2, mejores["val_mape"], 0.38, color=AZUL, label="Validación", edgecolor="white", linewidth=1.5)
    ax.bar(x + 0.2, mejores["prueba_mape"], 0.38, color=NARANJA, label="Prueba", edgecolor="white", linewidth=1.5)
    for i, (v, p) in enumerate(zip(mejores["val_mape"], mejores["prueba_mape"])):
        ax.text(i - 0.2, v + 0.08, f"{v:.2f}".replace(".", ","), ha="center", fontsize=8.5)
        ax.text(i + 0.2, p + 0.08, f"{p:.2f}".replace(".", ","), ha="center", fontsize=8.5)
    ax.set_xticks(list(x) + [len(mejores) + 0.1], etiquetas)
    ax.set_ylabel("MAPE (%)")
    ax.set_ylim(0, 7)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    guardar(fig, "fig_e2_1_comparacion_modelos.png")


def fig_error_por_hora(te, pred):
    y = te[F.OBJETIVO]
    mod = ((y - pred).abs() / y * 100).groupby(te.index.hour).mean()
    base = ((y - te["dem_lag24"]).abs() / y * 100).groupby(te.index.hour).mean()
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    ax.plot(base.index, base, color=GRIS, lw=2, ls="--", marker="o", ms=4, label="Línea base t-24")
    ax.plot(mod.index, mod, color=AZUL, lw=2, marker="o", ms=4, label="LightGBM final")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("MAPE (%)")
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(0, 8.5)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    guardar(fig, "fig_e2_2_error_por_hora.png")


if __name__ == "__main__":
    X = F.construir(F.cargar(DATOS))
    Xp = preparar(X)
    _, _, te = F.particionar(Xp)
    modelo = joblib.load("models/modelo_demanda.joblib")
    pred = pd.Series(modelo.predict(te), index=te.index)
    y = te[F.OBJETIVO]
    fig_comparacion(float(((y - pred).abs() / y).mean() * 100))
    fig_error_por_hora(te, pred)
    print("Figuras en", SALIDA)
