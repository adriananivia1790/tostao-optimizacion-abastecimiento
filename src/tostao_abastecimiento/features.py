"""Construcción de variables sin mirar el futuro.

Regla de oro: toda variable de una fila (fecha *t*) usa solo información
disponible hasta el domingo previo al lunes de su semana. Los rezagos son de al
menos 7 días y las medias móviles se evalúan en ese domingo.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import DIAS_SEMANA, REZAGOS
from .data import DatosCasoA


def _futuro(ventas: pd.DataFrame, dias: int) -> pd.DataFrame:
    """Filas vacías (sin ventas) para los `dias` siguientes al último día observado."""
    llaves = ventas[["id_tienda", "id_producto"]].drop_duplicates()
    fechas = pd.DataFrame(
        {"fecha": pd.date_range(ventas["fecha"].max() + pd.Timedelta(days=1), periods=dias)}
    )
    fut = llaves.merge(fechas, how="cross")
    fut["unidades_vendidas"] = np.nan
    return fut


def construir_panel(datos: DatosCasoA, dias_futuros: int = DIAS_SEMANA) -> pd.DataFrame:
    """Arma el panel SKU-tienda-día con variables para entrenar y para pronosticar.

    El panel incluye las filas históricas y, al final, `dias_futuros` filas sin
    ventas (`unidades_vendidas` = NaN) con sus variables ya calculadas: esas son
    las que se pronostican.

    Args:
        datos: tablas del Caso A.
        dias_futuros: días a pronosticar (máximo 7, porque el rezago mínimo es 7).

    Returns:
        DataFrame ordenado por tienda, producto y fecha, con las columnas `dow`,
        `semana`, `tienda_cod`, `producto_cod`, `lag_*`, `dow_mean4`, `rm7`, `rm28`.
    """
    if not 1 <= dias_futuros <= 7:
        raise ValueError("dias_futuros debe estar entre 1 y 7 (el rezago mínimo es 7)")
    v = datos.ventas[["fecha", "id_tienda", "id_producto", "unidades_vendidas"]]
    panel = pd.concat([v, _futuro(v, dias_futuros)], ignore_index=True)
    panel = panel.sort_values(["id_tienda", "id_producto", "fecha"]).reset_index(drop=True)

    panel["dow"] = panel["fecha"].dt.dayofweek
    panel["semana"] = (panel["fecha"] - v["fecha"].min()).dt.days // 7 + 1
    panel["tienda_cod"] = pd.Categorical(
        panel["id_tienda"], categories=sorted(datos.tiendas["id_tienda"])).codes
    panel["producto_cod"] = pd.Categorical(
        panel["id_producto"], categories=sorted(datos.catalogo["id_producto"])).codes

    g = panel.groupby(["id_tienda", "id_producto"])["unidades_vendidas"]
    for lag in REZAGOS:
        panel[f"lag_{lag}"] = g.shift(lag)
    panel["dow_mean4"] = panel[[f"lag_{lag}" for lag in REZAGOS]].mean(axis=1)

    # Nivel "hasta el domingo previo al lunes de la semana de cada fila".
    panel["rm7_dia"] = g.transform(lambda s: s.rolling(7).mean())
    panel["rm28_dia"] = g.transform(lambda s: s.rolling(28, min_periods=14).mean())
    panel["fecha_asof"] = panel["fecha"] - pd.to_timedelta(panel["dow"] + 1, unit="D")
    ref = panel[["id_tienda", "id_producto", "fecha", "rm7_dia", "rm28_dia"]].rename(
        columns={"fecha": "fecha_asof", "rm7_dia": "rm7", "rm28_dia": "rm28"})
    panel = panel.merge(ref, on=["id_tienda", "id_producto", "fecha_asof"], how="left")
    return panel.drop(columns=["rm7_dia", "rm28_dia"])


def dividir_historia_futuro(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa las filas con ventas observadas de las filas a pronosticar."""
    con_ventas = panel["unidades_vendidas"].notna()
    return panel[con_ventas].copy(), panel[~con_ventas].copy()
