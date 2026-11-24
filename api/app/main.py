from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api import api_router
from app.config import settings
from pgcb_modelo import cargar_modelo

app = FastAPI(title=settings.PROJECT_NAME, version=settings.API_VERSION,
              openapi_url=f"{settings.API_V1_STR}/openapi.json")

root_router = APIRouter()


@root_router.get("/")
def index() -> HTMLResponse:
    body = (
        "<html><body style='font-family: sans-serif; padding: 10px;'>"
        "<h1>API de pronóstico de demanda y alertas de déficit (PGCB, Bangladesh)</h1>"
        "<div>Documentación interactiva: <a href='/docs'>/docs</a></div>"
        "</body></html>"
    )
    return HTMLResponse(content=body)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

app.add_middleware(CORSMiddleware, allow_origins=settings.BACKEND_CORS_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

cargar_modelo()  # carga el modelo al arrancar, no en la primera petición

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
