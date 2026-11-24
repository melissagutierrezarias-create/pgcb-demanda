from pydantic import BaseModel


class Settings(BaseModel):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "API de pronóstico de demanda y alertas de déficit — PGCB"
    API_VERSION: str = "0.1.0"
    # Orígenes que pueden llamar la API desde un navegador (el tablero la llama desde el servidor)
    BACKEND_CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
