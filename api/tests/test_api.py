from fastapi.testclient import TestClient

from app.api import _ejemplo
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["version_modelo"] == "0.1.0"


def test_predict_devuelve_24_horas():
    r = client.post("/api/v1/predict", json=_ejemplo())
    assert r.status_code == 200
    d = r.json()
    assert len(d["predicciones"]) == 24
    assert d["resumen"]["horas_en_alerta"] == sum(p["alerta"] for p in d["predicciones"])
    assert all(0 <= p["prob_racionamiento"] <= 1 for p in d["predicciones"])


def test_historia_corta_responde_422():
    entrada = _ejemplo()
    entrada["historia"] = entrada["historia"][-50:]
    assert client.post("/api/v1/predict", json=entrada).status_code == 422


def test_horas_con_vacios_responde_400():
    entrada = _ejemplo()
    for r in entrada["historia"][-170:]:
        r["demand_mw"] = None
    assert client.post("/api/v1/predict", json=entrada).status_code == 400
