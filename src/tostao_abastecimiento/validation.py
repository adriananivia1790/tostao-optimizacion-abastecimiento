"""Backtest con origen móvil (rolling origin) y métricas de error."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import PRIMERA_SEMANA_PRUEBA


def wape(y: np.ndarray, pred: np.ndarray) -> float:
    """Error porcentual absoluto ponderado: suma de |real − pronóstico| ÷ suma de lo vendido."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    total = y.sum()
    if total <= 0:
        raise ValueError("La suma de ventas reales debe ser positiva para calcular el WAPE")
    return float(np.abs(y - pred).sum() / total)


def semanas_de_prueba(panel: pd.DataFrame, primera: int = PRIMERA_SEMANA_PRUEBA) -> List[int]:
    """Semanas con ventas observadas que se pronostican, de `primera` hasta la última."""
    ultima = int(panel.loc[panel["unidades_vendidas"].notna(), "semana"].max())
    return list(range(primera, ultima + 1))


def backtest_origen_movil(
    panel: pd.DataFrame,
    modelos: Dict[str, Callable[[], object]],
    primera_semana: int = PRIMERA_SEMANA_PRUEBA,
) -> pd.DataFrame:
    """Pronostica cada semana de prueba con un modelo entrenado solo con las semanas previas.

    Args:
        panel: salida de `construir_panel`.
        modelos: nombre -> función sin argumentos que devuelve un modelo nuevo con
            métodos `fit(df)` y `predict(df)`.
        primera_semana: primera semana que se pronostica (se entrena con las anteriores).

    Returns:
        Una fila por SKU-tienda-día de prueba, con `unidades_vendidas`, `semana` y
        una columna por modelo con su pronóstico diario (fuera de muestra).
    """
    historia = panel[panel["unidades_vendidas"].notna()]
    salidas: List[pd.DataFrame] = []
    for w in semanas_de_prueba(panel, primera_semana):
        entrena = historia[historia["semana"] < w]
        prueba = historia[historia["semana"] == w].copy()
        for nombre, fabrica in modelos.items():
            modelo = fabrica()
            modelo.fit(entrena)
            prueba[nombre] = np.clip(modelo.predict(prueba), 0, None)
        salidas.append(prueba)
    return pd.concat(salidas, ignore_index=True)


def resumen_wape(bt: pd.DataFrame, columnas: Iterable[str]) -> pd.DataFrame:
    """WAPE diario por semana de prueba y su promedio ± desviación entre semanas."""
    filas = []
    for w, d in bt.groupby("semana"):
        filas.append({"semana": w, **{c: wape(d["unidades_vendidas"], d[c]) for c in columnas}})
    por_semana = pd.DataFrame(filas).set_index("semana")
    resumen = por_semana.agg(["mean", "std"]).T
    resumen.columns = ["wape_medio", "desv_entre_semanas"]
    return resumen


def agregar_semanal(bt: pd.DataFrame, columnas: Iterable[str]) -> pd.DataFrame:
    """Suma a nivel SKU-tienda-semana las ventas reales y los pronósticos diarios."""
    cols = ["unidades_vendidas", *columnas]
    return bt.groupby(["id_tienda", "id_producto", "semana"], as_index=False)[cols].sum()
