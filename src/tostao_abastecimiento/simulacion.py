"""Simulación de costos sobre el backtest: ¿cuánto vale pedir con el modelo y con Newsvendor?

Cada semana de prueba se simula como una semana que empieza sin stock (el historial
de stock no está disponible), y se compara el costo de varias políticas de pedido:
la demanda real de la semana se mide contra la cantidad que cada política habría pedido.
La calibración de cuantiles de cada semana usa solo errores de semanas anteriores.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .calibration import CalibradorCuantiles
from .optimizer import ParametrosPedido, costo_esperado, economia_producto
from .validation import agregar_semanal


def simular_costos(bt: pd.DataFrame, catalogo: pd.DataFrame, p: ParametrosPedido,
                   modelo: str = "gbm_poisson", semanas_minimas_calibracion: int = 2,
                   tau_fijo: float = 0.90) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compara políticas de pedido semana a semana con la demanda real.

    Políticas:
        - Naive: pedir lo vendido la semana pasada.
        - Media 4 semanas: pedir el promedio de las últimas 4 semanas del mismo día.
        - Modelo, mediana: pronóstico del modelo calibrado al cuantil 0.5.
        - Modelo, τ fijo: pronóstico calibrado a un nivel de servicio igual para todos.
        - Modelo, τ Newsvendor: pronóstico calibrado al ratio crítico de cada producto.

    Args:
        bt: salida de `backtest_origen_movil`, con columnas `naive`, `media4` y `modelo`.
        catalogo: catálogo de productos.
        p: parámetros (α, β) que fijan Cu y Co.
        modelo: columna de `bt` con el pronóstico del modelo.
        semanas_minimas_calibracion: semanas previas necesarias para calibrar cuantiles.
        tau_fijo: nivel de servicio de la política de τ fijo.

    Returns:
        (por_semana, resumen): costo, nivel de servicio y sobrante por política.
    """
    weekly = agregar_semanal(bt, ["naive", "media4", modelo])
    econ = economia_producto(catalogo, p)[["id_producto", "cu", "co", "tau"]]
    weekly = weekly.merge(econ, on="id_producto")
    semanas = sorted(weekly["semana"].unique())
    filas: List[dict] = []
    for w in semanas[semanas_minimas_calibracion:]:
        prev, act = weekly[weekly["semana"] < w], weekly[weekly["semana"] == w]
        cal = CalibradorCuantiles().ajustar(prev["unidades_vendidas"], prev[modelo])
        pron = act[modelo].to_numpy()
        cantidades: Dict[str, np.ndarray] = {
            "Naive (semana pasada)": act["naive"].to_numpy(),
            "Media 4 semanas": act["media4"].to_numpy(),
            "Modelo, mediana": cal.cuantil_demanda(np.full(len(act), 0.5), pron),
            f"Modelo, τ fijo {tau_fijo:.2f}": cal.cuantil_demanda(np.full(len(act), tau_fijo), pron),
            "Modelo, τ Newsvendor": cal.cuantil_demanda(act["tau"].to_numpy(), pron),
        }
        d = act["unidades_vendidas"].to_numpy()
        for nombre, q in cantidades.items():
            filas.append({
                "semana": w, "politica": nombre,
                "costo": float(costo_esperado(d, q, act["cu"].to_numpy(), act["co"].to_numpy()).sum()),
                "faltante_u": float(np.maximum(d - q, 0).sum()),
                "sobrante_u": float(np.maximum(q - d, 0).sum()),
                "demanda_u": float(d.sum()),
            })
    por_semana = pd.DataFrame(filas)
    resumen = por_semana.groupby("politica", sort=False).agg(
        costo_total=("costo", "sum"), faltante_u=("faltante_u", "sum"),
        sobrante_u=("sobrante_u", "sum"), demanda_u=("demanda_u", "sum"))
    resumen["nivel_de_servicio"] = 1 - resumen["faltante_u"] / resumen["demanda_u"]
    resumen["sobrante_%_demanda"] = 100 * resumen["sobrante_u"] / resumen["demanda_u"]
    base = resumen.loc["Naive (semana pasada)", "costo_total"]
    resumen["ahorro_vs_naive_%"] = 100 * (1 - resumen["costo_total"] / base)
    return por_semana, resumen.drop(columns=["demanda_u"])
