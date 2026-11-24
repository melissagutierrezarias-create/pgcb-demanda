from __future__ import annotations

import math
from importlib.resources import files

import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from app import schemas
from app.config import settings
from pgcb_modelo import __version__ as version_modelo
from pgcb_modelo import cargar_modelo, pronosticar

api_router = APIRouter()


def _ejemplo() -> dict:
    """Los últimos 8 días de la historia que viene con el paquete: ejemplo listo para Swagger."""
    df = pd.read_csv(files("pgcb_modelo") / "datos" / "historia_ejemplo.csv").tail(192)
    return {"historia": df.to_dict(orient="records")}


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


@api_router.get("/health", response_model=schemas.Salud, status_code=200)
def health() -> dict:
    m = cargar_modelo()
    return schemas.Salud(nombre=settings.PROJECT_NAME, version_api=settings.API_VERSION,
                         version_modelo=version_modelo, modelo_entrenado=m["entrenado"],
                         datos_hasta=m["datos_hasta"]).model_dump()


@api_router.post("/predict", response_model=schemas.ResultadoPronostico, status_code=200)
def predict(entrada: schemas.EntradaPronostico = Body(openapi_examples={
        "ultimos_8_dias": {"summary": "Últimos 8 días del dataset", "value": _ejemplo()}})) -> dict:
    """Pronostica la demanda y las alertas de déficit de las 24 horas siguientes a la historia enviada."""
    historia = pd.DataFrame([r.model_dump() for r in entrada.historia])
    historia["datetime"] = pd.to_datetime(historia["datetime"]).dt.tz_localize(None)
    try:
        res = pronosticar(historia)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    alertas = res[res["alerta"]]
    resumen = {
        "demanda_pico_mw": float(res["demanda_mw"].max()),
        "hora_pico": res["demanda_mw"].idxmax(),
        "horas_en_alerta": int(len(alertas)),
        "deficit_maximo_mw": _num(alertas["deficit_mw"].max()) if len(alertas) else None,
    }
    predicciones = [{"datetime": t, "demanda_mw": float(f.demanda_mw),
                     "generacion_disponible_mw": _num(f.generacion_disponible_mw),
                     "deficit_mw": _num(f.deficit_mw), "prob_racionamiento": float(f.prob_racionamiento),
                     "alerta": bool(f.alerta)} for t, f in res.iterrows()]
    return {"version_modelo": version_modelo, "umbral_alerta": cargar_modelo()["alerta"]["umbral"],
            "resumen": resumen, "predicciones": predicciones}
