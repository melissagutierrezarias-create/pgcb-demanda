"""Modelo empaquetado de PGCB: pronóstico de demanda a 24 h y alerta de déficit."""
from pathlib import Path

__version__ = (Path(__file__).parent / "VERSION").read_text().strip()

from pgcb_modelo.predecir import COLUMNAS_ENTRADA, HISTORIA_MIN_HORAS, cargar_modelo, pronosticar  # noqa: E402

__all__ = ["__version__", "pronosticar", "cargar_modelo", "COLUMNAS_ENTRADA", "HISTORIA_MIN_HORAS"]
