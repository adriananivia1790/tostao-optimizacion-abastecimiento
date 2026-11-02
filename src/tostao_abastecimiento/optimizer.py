"""Optimizador de pedido (Newsvendor) por SKU-tienda.

Para cada SKU-tienda se pide la cantidad que minimiza el costo esperado de quedarse
corto (Cu) y de pasarse (Co). La solución es el cuantil τ = Cu / (Cu + Co) de la demanda
durante el horizonte de cobertura (lead time + período de revisión), menos el stock y
lo que ya viene en tránsito:

    pedido = max(0, Q*(τ) − stock − en tránsito)

Lo que no viene en los datos (lead time, MOQ, capacidad, tránsito, merma α, venta
cruzada β) se parametriza en `ParametrosPedido` con un valor base declarado.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol

import numpy as np
import pandas as pd
from scipy.stats import norm

from .calibration import CalibradorCuantiles
from .config import ALFA_POR_PRODUCTO, CV_DIARIO, MARGEN_CRUZADO_GENERAL
from .data import DatosCasoA

LLAVES = ["id_tienda", "id_producto"]


@dataclass
class ParametrosPedido:
    """Parámetros del problema de pedido. Los valores base son supuestos declarados, no datos.

    Attributes:
        lead_time_dias: días entre hacer el pedido y recibirlo.
        revision_dias: días que cubre cada pedido (cada cuánto se pide).
        en_transito_dias: unidades ya pedidas que no han llegado, en días de venta del SKU.
        moq: mínimo de unidades por pedido por SKU (si se pide algo).
        multiplo: múltiplo de empaque en que vende el proveedor.
        beta: fracción de la venta cruzada que se pierde si falta el producto.
        margen_cruzado: margen de la venta cruzada por unidad que falta (pesos).
        alfa: fracción del costo unitario que se pierde si sobra, por id_producto.
        tau_min, tau_max: piso y techo del nivel de servicio (cuantil de pedido).
        capacidad_tienda: máximo de unidades (stock + pedido) por tienda; None = sin límite.
        cv_diario: variabilidad diaria propia del SKU, solo para la incertidumbre paramétrica.
    """

    lead_time_dias: int = 1
    revision_dias: int = 7
    en_transito_dias: float = 0.0
    moq: int = 1
    multiplo: int = 1
    beta: float = 0.25
    margen_cruzado: float = MARGEN_CRUZADO_GENERAL
    alfa: Dict[str, float] = field(default_factory=lambda: dict(ALFA_POR_PRODUCTO))
    tau_min: float = 0.50
    tau_max: float = 0.99
    capacidad_tienda: Optional[float] = None
    cv_diario: float = CV_DIARIO

    @property
    def horizonte_dias(self) -> int:
        """Días que debe cubrir el pedido: lead time + período de revisión."""
        return self.lead_time_dias + self.revision_dias


def razon_critica(cu: np.ndarray, co: np.ndarray) -> np.ndarray:
    """Ratio crítico del Newsvendor: Cu / (Cu + Co)."""
    cu = np.asarray(cu, dtype=float)
    co = np.asarray(co, dtype=float)
    return cu / (cu + co)


def economia_producto(catalogo: pd.DataFrame, p: ParametrosPedido) -> pd.DataFrame:
    """Cu, Co y ratio crítico por producto.

    Cu = margen unitario + β × margen cruzado; Co = almacenamiento semanal + α × costo unitario.

    Raises:
        KeyError: si algún producto del catálogo no tiene α definido.
    """
    faltan = set(catalogo["id_producto"]) - set(p.alfa)
    if faltan:
        raise KeyError(f"Falta α para los productos: {sorted(faltan)}")
    e = catalogo.copy()
    e["margen_unit"] = e["precio_venta"] - e["costo_unitario"]
    e["cu"] = e["margen_unit"] + p.beta * p.margen_cruzado
    e["co"] = e["costo_almacenamiento_semanal"] + e["id_producto"].map(p.alfa) * e["costo_unitario"]
    e["tau"] = razon_critica(e["cu"], e["co"]).clip(p.tau_min, p.tau_max)
    return e[["id_producto", "nombre", "margen_unit", "costo_unitario", "cu", "co", "tau"]]


def perfil_semanal(ventas: pd.DataFrame) -> pd.Series:
    """Índice de venta por día de la semana (0 = lunes), con media 1."""
    m = ventas.groupby(ventas["fecha"].dt.dayofweek)["unidades_vendidas"].mean()
    return m / m.mean()


def nivel_reciente(ventas: pd.DataFrame, dias: int = 28) -> pd.DataFrame:
    """Venta media diaria por SKU-tienda en los últimos `dias` (por defecto 4 semanas completas)."""
    corte = ventas["fecha"].max() - pd.Timedelta(days=dias - 1)
    r = ventas[ventas["fecha"] >= corte]
    return r.groupby(LLAVES)["unidades_vendidas"].mean().rename("nivel_dia").reset_index()


def mu_dias_horizonte(pronostico_diario: Optional[np.ndarray], nivel_dia: np.ndarray,
                      perfil: pd.Series, dow_inicio: int, dias: int) -> np.ndarray:
    """Demanda media esperada por día durante el horizonte, forma (n SKU-tienda, dias).

    Los primeros 7 días salen del pronóstico del modelo (si se entrega); los días
    siguientes y el caso sin modelo usan nivel reciente × perfil semanal.
    """
    n = len(nivel_dia)
    if dias <= 0:
        return np.zeros((n, 0))
    idx = [(dow_inicio + k) % 7 for k in range(dias)]
    mu = nivel_dia[:, None] * perfil.values[idx][None, :]
    if pronostico_diario is not None:
        usar = min(dias, pronostico_diario.shape[1])
        mu[:, :usar] = pronostico_diario[:, :usar]
    return mu


class Incertidumbre(Protocol):
    """Convierte la demanda media por día en el cuantil τ de la demanda del horizonte."""

    def cuantil(self, tau: np.ndarray, mu_dias: np.ndarray) -> np.ndarray: ...


class IncertidumbreNormal:
    """Aproximación normal con ruido diario independiente (la del EDA, H13)."""

    def __init__(self, cv_diario: float = CV_DIARIO) -> None:
        self.cv_diario = cv_diario

    def cuantil(self, tau: np.ndarray, mu_dias: np.ndarray) -> np.ndarray:
        mu = mu_dias.sum(axis=1)
        sd = self.cv_diario * np.sqrt((mu_dias ** 2).sum(axis=1))
        return np.maximum(mu + norm.ppf(tau) * sd, 0)


class IncertidumbreEmpirica:
    """Cuantiles calibrados con el error fuera de muestra del backtest (ver `calibration`).

    La calibración se hizo con demanda semanal; para horizontes distintos de 7 días se
    aplica la misma venta relativa a la demanda media del horizonte (aproximación).
    """

    def __init__(self, calibrador: CalibradorCuantiles) -> None:
        self.calibrador = calibrador

    def cuantil(self, tau: np.ndarray, mu_dias: np.ndarray) -> np.ndarray:
        return self.calibrador.cuantil_demanda(tau, mu_dias.sum(axis=1))


def calcular_pedido(datos: DatosCasoA, p: ParametrosPedido,
                    pronostico_diario: Optional[pd.DataFrame] = None,
                    incertidumbre: Optional[Incertidumbre] = None) -> pd.DataFrame:
    """Pedido por SKU-tienda para la semana siguiente al último día de ventas.

    Args:
        datos: tablas del Caso A (se usan ventas, inventario y catálogo).
        p: parámetros del pedido.
        pronostico_diario: columnas `id_tienda`, `id_producto`, `fecha`, `pronostico`
            (los 7 días a pronosticar). Si es None se usa nivel reciente × perfil semanal.
        incertidumbre: objeto con método `cuantil`. Por defecto, aproximación normal.

    Returns:
        Una fila por SKU-tienda con tau, q_objetivo, pedido y ventas perdidas inevitables.
    """
    if p.horizonte_dias < 1:
        raise ValueError("lead_time_dias + revision_dias debe ser al menos 1")
    inc = incertidumbre or IncertidumbreNormal(p.cv_diario)
    perfil = perfil_semanal(datos.ventas)
    dow0 = int((datos.ventas["fecha"].max() + pd.Timedelta(days=1)).dayofweek)
    econ = economia_producto(datos.catalogo, p)
    d = (nivel_reciente(datos.ventas)
         .merge(datos.inventario, on=LLAVES)
         .merge(econ, on="id_producto")
         .sort_values(LLAVES).reset_index(drop=True))

    pron = None
    if pronostico_diario is not None:
        piv = (pronostico_diario.pivot_table(index=LLAVES, columns="fecha", values="pronostico")
               .reindex(pd.MultiIndex.from_frame(d[LLAVES])))
        pron = piv.sort_index(axis=1).to_numpy(dtype=float)

    nivel = d["nivel_dia"].to_numpy()
    mu_h = mu_dias_horizonte(pron, nivel, perfil, dow0, p.horizonte_dias)
    d["demanda_media_horizonte"] = mu_h.sum(axis=1)
    d["q_objetivo"] = inc.cuantil(d["tau"].to_numpy(), mu_h)
    d["en_transito"] = nivel * p.en_transito_dias
    bruto = np.maximum(d["q_objetivo"] - d["stock_actual"] - d["en_transito"], 0)
    pedido = np.where(bruto > 0, np.maximum(np.ceil(bruto / p.multiplo) * p.multiplo, p.moq), 0.0)
    d["piso_moq_activo"] = (bruto > 0) & (pedido > bruto + 1e-9) & (pedido <= p.moq)

    if p.capacidad_tienda is not None:
        d["pedido"] = pedido
        ocupado = (d["stock_actual"] + d["en_transito"]).groupby(d["id_tienda"]).transform("sum")
        pedir = d["pedido"].groupby(d["id_tienda"]).transform("sum")
        factor = np.clip((p.capacidad_tienda - ocupado) / pedir.replace(0, np.nan), 0, 1).fillna(1)
        pedido = np.floor(pedido * factor / p.multiplo) * p.multiplo
    d["pedido"] = pedido

    mu_lead = mu_dias_horizonte(pron, nivel, perfil, dow0, p.lead_time_dias).sum(axis=1)
    d["venta_perdida_inevitable"] = np.maximum(mu_lead - d["stock_actual"] - d["en_transito"], 0)
    return d


def costo_esperado(demanda: np.ndarray, cantidad: np.ndarray, cu: np.ndarray, co: np.ndarray) -> np.ndarray:
    """Costo de pedir `cantidad` cuando la demanda fue `demanda`: Cu × faltante + Co × sobrante."""
    demanda = np.asarray(demanda, dtype=float)
    cantidad = np.asarray(cantidad, dtype=float)
    return cu * np.maximum(demanda - cantidad, 0) + co * np.maximum(cantidad - demanda, 0)
