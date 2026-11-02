"""Sensibilidad del pedido a los parámetros que no vienen en los datos (equivale a H13 del EDA)."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import pandas as pd

from .config import ALFA_POR_PRODUCTO
from .data import DatosCasoA
from .optimizer import Incertidumbre, ParametrosPedido, calcular_pedido


def escenarios(base: ParametrosPedido, capacidad_referencia: float) -> Dict[str, ParametrosPedido]:
    """Escenarios que mueven un parámetro a la vez respecto del caso base.

    Args:
        base: parámetros del caso base.
        capacidad_referencia: unidades (stock + pedido) por tienda sin límite de capacidad,
            para expresar la capacidad como porcentaje de ella.
    """
    todos = lambda v: {k: v for k in ALFA_POR_PRODUCTO}  # noqa: E731
    return {
        "Base": base,
        "Lead time 0": replace(base, lead_time_dias=0),
        "Lead time 2": replace(base, lead_time_dias=2),
        "Lead time 3": replace(base, lead_time_dias=3),
        "Lead time 7": replace(base, lead_time_dias=7),
        "Revisión cada 3 días": replace(base, revision_dias=3),
        "En tránsito: 1 día de venta": replace(base, en_transito_dias=1.0),
        "En tránsito: 2 días de venta": replace(base, en_transito_dias=2.0),
        "MOQ y empaque de 6": replace(base, moq=6, multiplo=6),
        "MOQ y empaque de 12": replace(base, moq=12, multiplo=12),
        "MOQ y empaque de 24": replace(base, moq=24, multiplo=24),
        "Capacidad 90 % del pedido libre": replace(base, capacidad_tienda=0.9 * capacidad_referencia),
        "Capacidad 70 % del pedido libre": replace(base, capacidad_tienda=0.7 * capacidad_referencia),
        "α = 0 (nada se pierde)": replace(base, alfa=todos(0.0)),
        "α = 1 (todo lo que sobra se pierde)": replace(base, alfa=todos(1.0)),
        "β = 0": replace(base, beta=0.0),
        "β = 100 %": replace(base, beta=1.0),
    }


def tabla_sensibilidad(datos: DatosCasoA, base: Optional[ParametrosPedido] = None,
                       pronostico_diario: Optional[pd.DataFrame] = None,
                       incertidumbre: Optional[Incertidumbre] = None) -> pd.DataFrame:
    """Pedido total y ventas perdidas inevitables bajo cada escenario.

    Returns:
        Una fila por escenario con `pedido_total_u`, `venta_perdida_inevitable_u`,
        `sku_sin_pedido` y `cambio_vs_base_%`.
    """
    base = base or ParametrosPedido()
    sin_limite = calcular_pedido(datos, base, pronostico_diario, incertidumbre)
    cap_ref = float((sin_limite["stock_actual"] + sin_limite["pedido"]).groupby(sin_limite["id_tienda"]).sum().mean())
    filas = []
    for nombre, p in escenarios(base, cap_ref).items():
        r = calcular_pedido(datos, p, pronostico_diario, incertidumbre)
        filas.append({"escenario": nombre, "pedido_total_u": r["pedido"].sum(),
                      "venta_perdida_inevitable_u": r["venta_perdida_inevitable"].sum(),
                      "sku_sin_pedido": int((r["pedido"] == 0).sum())})
    t = pd.DataFrame(filas).set_index("escenario")
    t["cambio_vs_base_%"] = 100 * (t["pedido_total_u"] / t.loc["Base", "pedido_total_u"] - 1)
    return t
