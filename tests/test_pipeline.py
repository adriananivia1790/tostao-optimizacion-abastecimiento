import pandas as pd

from tostao_abastecimiento.features import construir_panel
from tostao_abastecimiento.models import ModeloPoissonGBM
from tostao_abastecimiento.optimizer import ParametrosPedido
from tostao_abastecimiento.pipeline import MODELOS_BACKTEST
from tostao_abastecimiento.validation import backtest_origen_movil, semanas_de_prueba


def test_backtest_solo_usa_semanas_de_prueba(datos_sinteticos):
    panel = construir_panel(datos_sinteticos)
    assert semanas_de_prueba(panel) == list(range(6, 14))
    bt = backtest_origen_movil(panel, MODELOS_BACKTEST)
    assert set(bt["semana"]) == set(range(6, 14))
    assert (bt[list(MODELOS_BACKTEST)] >= 0).all().all()


def test_modelo_pronostica_la_semana_siguiente(datos_sinteticos):
    panel = construir_panel(datos_sinteticos)
    hist = panel[panel["unidades_vendidas"].notna()]
    fut = panel[panel["unidades_vendidas"].isna()]
    pred = ModeloPoissonGBM().fit(hist).predict(fut)
    assert len(pred) == len(fut) and (pred >= 0).all()
