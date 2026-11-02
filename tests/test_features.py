import numpy as np
import pandas as pd

from tostao_abastecimiento.config import COLUMNAS_MODELO
from tostao_abastecimiento.data import DatosCasoA
from tostao_abastecimiento.features import construir_panel, dividir_historia_futuro


def test_panel_incluye_siete_dias_futuros(datos_sinteticos):
    panel = construir_panel(datos_sinteticos)
    _, futuro = dividir_historia_futuro(panel)
    assert len(futuro) == 7 * 6
    assert futuro["fecha"].min() == pd.Timestamp("2024-04-01")
    assert futuro["fecha"].max() == pd.Timestamp("2024-04-07")


def test_variables_futuras_no_tienen_nulos_criticos(datos_sinteticos):
    _, futuro = dividir_historia_futuro(construir_panel(datos_sinteticos))
    assert not futuro[list(COLUMNAS_MODELO)].isna().any().any()


def test_sin_fuga_de_informacion(datos_sinteticos):
    """Cambiar las ventas de una semana no debe alterar las variables de esa semana."""
    v = datos_sinteticos.ventas.copy()
    panel_a = construir_panel(datos_sinteticos)
    semana = 10
    inicio = v["fecha"].min() + pd.Timedelta(days=7 * (semana - 1))
    en_semana = (v["fecha"] >= inicio) & (v["fecha"] < inicio + pd.Timedelta(days=7))
    v2 = v.copy()
    v2.loc[en_semana, "unidades_vendidas"] += 1000
    panel_b = construir_panel(DatosCasoA(v2, datos_sinteticos.inventario,
                                         datos_sinteticos.catalogo, datos_sinteticos.tiendas))
    cols = list(COLUMNAS_MODELO)
    a = panel_a[panel_a["semana"] == semana][cols].reset_index(drop=True)
    b = panel_b[panel_b["semana"] == semana][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
