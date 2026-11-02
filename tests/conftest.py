"""Datos sintéticos pequeños para probar el código sin los CSV reales."""
import numpy as np
import pandas as pd
import pytest

from tostao_abastecimiento.data import DatosCasoA


@pytest.fixture(scope="session")
def datos_sinteticos() -> DatosCasoA:
    rng = np.random.default_rng(0)
    fechas = pd.date_range("2024-01-01", "2024-03-31")          # 91 días = 13 semanas
    tiendas_ids = ["STORE_01", "STORE_02", "STORE_03"]
    productos = ["PROD_001", "PROD_002"]
    perfil = np.array([0.8, 0.8, 0.85, 0.9, 1.25, 1.3, 1.3])
    filas = []
    for i, t in enumerate(tiendas_ids):
        for j, p in enumerate(productos):
            nivel = 8 + 4 * i + 3 * j
            mu = nivel * perfil[fechas.dayofweek]
            for f, m in zip(fechas, mu):
                filas.append((f, t, p, int(rng.poisson(m))))
    ventas = pd.DataFrame(filas, columns=["fecha", "id_tienda", "id_producto", "unidades_vendidas"])
    inventario = pd.DataFrame([(t, p, int(rng.integers(0, 30))) for t in tiendas_ids for p in productos],
                              columns=["id_tienda", "id_producto", "stock_actual"])
    catalogo = pd.DataFrame({
        "id_producto": productos, "nombre": ["Tinto", "Café con Leche"], "categoria": ["Bebidas"] * 2,
        "costo_unitario": [800, 1200], "precio_venta": [2500, 3500], "costo_almacenamiento_semanal": [10, 15]})
    tiendas = pd.DataFrame({"id_tienda": tiendas_ids, "ciudad": ["Bogotá"] * 3, "tamaño_m2": [20, 30, 40]})
    return DatosCasoA(ventas=ventas, inventario=inventario, catalogo=catalogo, tiendas=tiendas)
