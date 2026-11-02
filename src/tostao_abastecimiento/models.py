"""Modelos de pronóstico diario y referencias simples (baselines).

Todos exponen `fit(df)` y `predict(df)` sobre el panel de `features.construir_panel`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import COLUMNAS_CATEGORICAS, COLUMNAS_MODELO


class NaiveSemanaPasada:
    """Pronostica lo vendido el mismo día de la semana pasada (rezago de 7 días)."""

    def fit(self, df: pd.DataFrame) -> "NaiveSemanaPasada":
        self.media_ = float(df["unidades_vendidas"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["lag_7"].fillna(self.media_).to_numpy(dtype=float)


class MediaCuatroSemanasMismoDia:
    """Pronostica el promedio de las últimas 4 semanas del mismo día de la semana."""

    def fit(self, df: pd.DataFrame) -> "MediaCuatroSemanasMismoDia":
        self.media_ = float(df["unidades_vendidas"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["dow_mean4"].fillna(self.media_).to_numpy(dtype=float)


class PerfilPorNivel:
    """Modelo estadístico simple: nivel reciente del SKU-tienda × perfil del día de la semana.

    El nivel es la venta media diaria de las últimas 4 semanas hasta el domingo previo
    (`rm28`); el perfil es el índice semanal (venta del día ÷ venta media), con media 1.
    """

    def fit(self, df: pd.DataFrame) -> "PerfilPorNivel":
        por_dia = df.groupby("dow")["unidades_vendidas"].mean()
        self.perfil_ = (por_dia / por_dia.mean()).reindex(range(7)).fillna(1.0)
        self.media_ = float(df["unidades_vendidas"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        nivel = df["rm28"].fillna(self.media_)
        return (nivel * df["dow"].map(self.perfil_)).to_numpy(dtype=float)


class ModeloPoissonGBM:
    """Gradient boosting con pérdida de Poisson (scikit-learn).

    Se usa Poisson porque la demanda son conteos no negativos cuya variabilidad crece
    con el nivel. Los hiperparámetros son los de la ablación H12 del EDA.
    """

    def __init__(self, columnas: Sequence[str] = COLUMNAS_MODELO, max_iter: int = 200,
                 learning_rate: float = 0.05, max_depth: int = 4, min_samples_leaf: int = 20,
                 random_state: int = 0, loss: str = "poisson") -> None:
        self.columnas = list(columnas)
        self._params = dict(loss=loss, max_iter=max_iter, learning_rate=learning_rate,
                            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                            random_state=random_state)

    def fit(self, df: pd.DataFrame) -> "ModeloPoissonGBM":
        mascara = [c in COLUMNAS_CATEGORICAS for c in self.columnas]
        self.modelo_ = HistGradientBoostingRegressor(categorical_features=mascara, **self._params)
        self.modelo_.fit(df[self.columnas].astype(float), df["unidades_vendidas"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.modelo_.predict(df[self.columnas].astype(float)), 0, None)
