"""Incertidumbre del pronóstico semanal calibrada con el error fuera de muestra.

En lugar de suponer una distribución (normal, Poisson), se usa la distribución
empírica de la *venta relativa semanal* (real ÷ pronóstico) observada en el
backtest. El cuantil de demanda para un nivel de servicio τ es el pronóstico
por el cuantil τ de esa venta relativa. Es una calibración conformal sencilla y
no depende de supuestos de forma.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class CalibradorCuantiles:
    """Cuantiles de la venta relativa semanal, por grupos de volumen.

    El error relativo es mayor en los SKU-tienda de poca venta, así que la
    distribución se estima por terciles del pronóstico semanal.

    Args:
        n_grupos: número de grupos de volumen (por cuantiles del pronóstico).
        minimo_por_grupo: si un grupo tiene menos observaciones, se usa la distribución global.
    """

    def __init__(self, n_grupos: int = 3, minimo_por_grupo: int = 30) -> None:
        self.n_grupos = n_grupos
        self.minimo_por_grupo = minimo_por_grupo

    def ajustar(self, real: np.ndarray, pronostico: np.ndarray) -> "CalibradorCuantiles":
        """Estima la distribución de real ÷ pronóstico con pares semanales fuera de muestra."""
        real = np.asarray(real, dtype=float)
        pron = np.asarray(pronostico, dtype=float)
        ok = pron > 0
        real, pron = real[ok], pron[ok]
        if len(real) == 0:
            raise ValueError("No hay pares con pronóstico positivo para calibrar")
        self.cortes_ = np.quantile(pron, np.linspace(0, 1, self.n_grupos + 1)[1:-1])
        grupo = np.searchsorted(self.cortes_, pron)
        relativa = real / pron
        self.global_ = np.sort(relativa)
        self.por_grupo_: Dict[int, np.ndarray] = {}
        for g in range(self.n_grupos):
            r = relativa[grupo == g]
            self.por_grupo_[g] = np.sort(r) if len(r) >= self.minimo_por_grupo else self.global_
        return self

    def multiplicador(self, tau: np.ndarray, pronostico: np.ndarray) -> np.ndarray:
        """Cuantil τ de la venta relativa para cada SKU-tienda (según su volumen)."""
        tau = np.broadcast_to(np.asarray(tau, dtype=float), np.shape(pronostico)).astype(float)
        grupo = np.searchsorted(self.cortes_, np.asarray(pronostico, dtype=float))
        salida = np.empty(len(tau))
        for g in np.unique(grupo):
            en_g = grupo == g
            for t in np.unique(tau[en_g]):
                sel = en_g & (tau == t)
                salida[sel] = np.quantile(self.por_grupo_[int(g)], t)
        return salida

    def cuantil_demanda(self, tau: np.ndarray, pronostico: np.ndarray) -> np.ndarray:
        """Demanda semanal que se cubre con probabilidad τ: pronóstico × cuantil de la venta relativa."""
        return np.asarray(pronostico, dtype=float) * self.multiplicador(tau, pronostico)


def cobertura_empirica(real: np.ndarray, limite: np.ndarray) -> float:
    """Proporción de casos en que la demanda real quedó por debajo del cuantil pronosticado."""
    return float(np.mean(np.asarray(real) <= np.asarray(limite) + 1e-9))


def tabla_cobertura(real: np.ndarray, pronostico: np.ndarray, calibrador: CalibradorCuantiles,
                    taus=(0.5, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    """Cobertura nominal frente a la observada. Con datos fuera de la calibración debería coincidir."""
    filas = []
    for t in taus:
        q = calibrador.cuantil_demanda(np.full(len(pronostico), t), pronostico)
        filas.append({"nivel_nominal": t, "cobertura_observada": cobertura_empirica(real, q)})
    return pd.DataFrame(filas).set_index("nivel_nominal")


def cobertura_secuencial(semanal: pd.DataFrame, columna_pronostico: str, semanas_minimas: int = 2,
                         taus=(0.5, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    """Cobertura fuera de muestra: cada semana se calibra solo con las semanas anteriores.

    Args:
        semanal: salida de `validation.agregar_semanal` (real y pronóstico por SKU-tienda-semana).
        columna_pronostico: columna con el pronóstico semanal del modelo.
        semanas_minimas: semanas previas necesarias para calibrar.
        taus: niveles de servicio a verificar.

    Returns:
        Cobertura observada por nivel nominal, sobre todas las semanas evaluadas.
    """
    semanas = sorted(semanal["semana"].unique())
    aciertos = {t: [] for t in taus}
    for w in semanas[semanas_minimas:]:
        prev, act = semanal[semanal["semana"] < w], semanal[semanal["semana"] == w]
        cal = CalibradorCuantiles().ajustar(prev["unidades_vendidas"], prev[columna_pronostico])
        pron = act[columna_pronostico].to_numpy()
        for t in taus:
            q = cal.cuantil_demanda(np.full(len(act), t), pron)
            aciertos[t].append(act["unidades_vendidas"].to_numpy() <= q + 1e-9)
    filas = [{"nivel_nominal": t, "cobertura_observada": float(np.concatenate(a).mean())}
             for t, a in aciertos.items()]
    return pd.DataFrame(filas).set_index("nivel_nominal")
