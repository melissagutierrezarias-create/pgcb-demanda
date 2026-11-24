from pathlib import Path

import pandas as pd
import pytest

from pgcb_modelo import HISTORIA_MIN_HORAS, __version__, cargar_modelo, pronosticar

EJEMPLO = Path(__file__).parents[1] / "pgcb_modelo" / "datos" / "historia_ejemplo.csv"


@pytest.fixture
def historia():
    return pd.read_csv(EJEMPLO, parse_dates=["datetime"])


def test_modelo_cargado_es_de_esta_version():
    m = cargar_modelo()
    assert m["version"] == __version__
    assert 0 < m["alerta"]["umbral"] < 1


def test_pronostica_las_24_horas_siguientes(historia):
    res = pronosticar(historia)
    assert len(res) == 24
    assert res.index[0] == historia["datetime"].max() + pd.Timedelta(hours=1)
    assert res["demanda_mw"].notna().all() and (res["demanda_mw"] > 0).all()
    assert res["prob_racionamiento"].between(0, 1).all()
    assert res["alerta"].dtype == bool


def test_demanda_en_rango_razonable(historia):
    # El pronóstico no debería alejarse más de 50 % de la demanda observada en la última semana
    res = pronosticar(historia)
    ult = historia["demand_mw"].tail(168)
    assert res["demanda_mw"].between(ult.min() * 0.5, ult.max() * 1.5).all()


def test_ignora_columnas_extra_y_orden(historia):
    desordenada = historia.sample(frac=1, random_state=0).assign(gas=1.0)
    pd.testing.assert_frame_equal(pronosticar(desordenada), pronosticar(historia))


def test_historia_corta_da_error(historia):
    with pytest.raises(ValueError, match="al menos"):
        pronosticar(historia.tail(HISTORIA_MIN_HORAS - 1))


def test_columna_faltante_da_error(historia):
    with pytest.raises(ValueError, match="Faltan columnas"):
        pronosticar(historia.drop(columns="liquid_fuel"))
