"""Prueba de punta a punta con los CSV reales del Caso A.

Se omite sola si los datos no están en `data/01_supply_optimization`
(o en la carpeta indicada con la variable de entorno `TOSTAO_DATA_DIR`).
Corre el flujo completo y revisa que las salidas cumplan condiciones de sentido común.

    pytest tests/test_datos_reales.py -v
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tostao_abastecimiento.optimizer import ParametrosPedido
from tostao_abastecimiento.pipeline import ejecutar_pipeline

RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TOSTAO_DATA_DIR", RAIZ / "data" / "01_supply_optimization"))
ARCHIVOS = ["ventas_historicas.csv", "inventario_actual.csv", "catalogo_productos.csv", "maestro_tiendas.csv"]

pytestmark = pytest.mark.skipif(
    not all((DATA_DIR / a).exists() for a in ARCHIVOS),
    reason=f"No están los CSV reales en {DATA_DIR}")


@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    salida = tmp_path_factory.mktemp("outputs")
    return ejecutar_pipeline(DATA_DIR, ParametrosPedido(), salida), salida


def test_se_generan_los_archivos_de_salida(corrida):
    _, salida = corrida
    esperados = {"pronostico_semanal.csv", "pedido_semana.csv", "metricas_backtest.csv", "simulacion_costos.csv"}
    assert esperados <= {p.name for p in salida.iterdir()}


def test_un_pronostico_y_un_pedido_por_sku_tienda(corrida):
    res, _ = corrida
    for df in (res.pronostico, res.pedido):
        assert not df.duplicated(["id_tienda", "id_producto"]).any()
        assert not df.isna().any().any()
    assert len(res.pronostico) == len(res.pedido)


def test_cuantiles_ordenados_y_no_negativos(corrida):
    res, _ = corrida
    f = res.pronostico
    assert (f[["pronostico_media", "p10", "p50", "p90"]] >= 0).all().all()
    assert (f["p10"] <= f["p50"]).all() and (f["p50"] <= f["p90"]).all()


def test_pedido_valido(corrida):
    res, _ = corrida
    ped = res.pedido
    assert (ped["pedido"] >= 0).all()
    assert (ped["tau"].between(0.5, 1, inclusive="neither")).all()
    assert (ped["q_objetivo"] >= ped["demanda_media_horizonte"] * 0.99).all()   # τ > 0.5: objetivo sobre la media
    p = ParametrosPedido()
    pedidos = ped.loc[ped["pedido"] > 0, "pedido"].to_numpy()
    assert np.allclose(pedidos % p.multiplo, 0) and (pedidos >= p.moq).all()


def test_backtest_usa_solo_semanas_de_prueba_y_baja_el_error(corrida):
    res, _ = corrida
    assert set(res.backtest["semana"]) == set(range(6, 14))
    wape = res.metricas["wape_medio"]
    assert wape["gbm_poisson"] < wape["naive"]          # el modelo debe mejorar la referencia ingenua


def test_cobertura_cercana_al_nivel_nominal(corrida):
    res, _ = corrida
    cob = res.cobertura["cobertura_observada"]
    for nivel in (0.8, 0.9, 0.95):
        assert abs(cob.loc[nivel] - nivel) < 0.07


def test_newsvendor_es_la_politica_mas_barata(corrida):
    res, _ = corrida
    costos = res.simulacion_resumen["costo_total"]
    assert costos.idxmin() == "Modelo, τ Newsvendor"
