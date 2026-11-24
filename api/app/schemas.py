from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

from pgcb_modelo import HISTORIA_MIN_HORAS


class Salud(BaseModel):
    nombre: str
    version_api: str
    version_modelo: str
    modelo_entrenado: str
    datos_hasta: str


class RegistroHorario(BaseModel):
    """Una hora de la historia de la red (mismas columnas del dataset PGCB)."""
    datetime: dt.datetime = Field(description="Hora del registro, por ejemplo 2025-06-10T18:00:00")
    demand_mw: float | None = Field(description="Demanda en MW")
    generation_mw: float | None = Field(description="Generación entregada en MW")
    load_shedding: float | None = Field(0.0, description="Racionamiento en MW (0 si no hubo)")
    liquid_fuel: float | None = Field(description="Generación con fuel/diésel en MW")


class EntradaPronostico(BaseModel):
    historia: list[RegistroHorario] = Field(
        min_length=HISTORIA_MIN_HORAS,
        description=f"Historia horaria más reciente, al menos {HISTORIA_MIN_HORAS} horas (8 días). "
                    "Se pronostican las 24 horas siguientes a la última hora enviada.")


class PrediccionHora(BaseModel):
    datetime: dt.datetime
    demanda_mw: float = Field(description="Demanda pronosticada")
    generacion_disponible_mw: float | None = Field(description="Generación de la misma hora del día anterior")
    deficit_mw: float | None = Field(description="Demanda pronosticada menos generación disponible")
    prob_racionamiento: float = Field(description="Probabilidad de racionamiento según el modelo de alerta")
    alerta: bool = Field(description="True si la probabilidad supera el umbral del modelo")


class Resumen(BaseModel):
    demanda_pico_mw: float
    hora_pico: dt.datetime
    horas_en_alerta: int
    deficit_maximo_mw: float | None = Field(description="Mayor déficit entre las horas en alerta")


class ResultadoPronostico(BaseModel):
    version_modelo: str
    umbral_alerta: float
    resumen: Resumen
    predicciones: list[PrediccionHora]
