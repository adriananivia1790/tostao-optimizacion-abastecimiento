"""Constantes del problema y supuestos declarados.

Los valores marcados como *supuesto* no vienen en los datos del enunciado: se
documentan en el EDA (secciones 1.5, 1.7, H9, H10 y H13) y se pueden cambiar
sin tocar la lógica.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

RAIZ_REPO: Path = Path(__file__).resolve().parents[2]
DATA_DIR_DEFECTO: Path = RAIZ_REPO / "data" / "01_supply_optimization"

# Semana a pronosticar: lunes 1 a domingo 7 de abril de 2024.
INICIO_PRONOSTICO: pd.Timestamp = pd.Timestamp("2024-04-01")
DIAS_SEMANA: int = 7

# Backtest con origen móvil: se pronostica cada semana desde la 6 con todo lo anterior.
PRIMERA_SEMANA_PRUEBA: int = 6

# Variables de rezago (en días). Todas son >= 7 para no mirar el futuro.
REZAGOS: Tuple[int, ...] = (7, 14, 21, 28)

# Variables extra que se prueban en el notebook 02 (experimentos de mejora).
COLUMNAS_EXTRA: Tuple[str, ...] = ("tamano_m2", "ciudad_cod", "tendencia_rm")

# Variables del modelo final (conjunto "C" de la ablación H12 del EDA).
COLUMNAS_CATEGORICAS: Tuple[str, ...] = ("dow", "tienda_cod", "producto_cod")
COLUMNAS_MODELO: Tuple[str, ...] = (
    "dow", "tienda_cod", "producto_cod",
    "lag_7", "lag_14", "lag_21", "lag_28", "dow_mean4",
    "rm7", "rm28",
)

# Supuesto (H10): fracción del costo unitario que se pierde si un producto sobra.
# Se obtuvo de referencias de vida útil, no de mermas medidas: hay que validarla con el negocio.
ALFA_POR_PRODUCTO: Dict[str, float] = {
    "PROD_001": 0.25,  # Tinto
    "PROD_002": 0.25,  # Café con Leche
    "PROD_003": 0.25,  # Cappuccino
    "PROD_004": 0.75,  # Pan de Bono
    "PROD_005": 1.00,  # Buñuelo
    "PROD_006": 1.00,  # Croissant
    "PROD_007": 0.50,  # Pastel de Pollo
    "PROD_008": 1.00,  # Jugo de Naranja
}

# Supuesto (H9): margen de la venta cruzada por unidad que falta, en pesos.
# Sale de `detalle_tickets` (carpeta 02) y se usa como cifra general para los 8 productos.
MARGEN_CRUZADO_GENERAL: float = 2833.86

# Ruido diario residual medido en el EDA (H6): se usa solo en la incertidumbre paramétrica.
CV_DIARIO: float = 0.31
