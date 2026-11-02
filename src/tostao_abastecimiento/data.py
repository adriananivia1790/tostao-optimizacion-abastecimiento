"""Carga y validación de las tablas del Caso A (carpeta 01_supply_optimization)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .config import DATA_DIR_DEFECTO

COLUMNAS_ESPERADAS = {
    "ventas": ["fecha", "id_tienda", "id_producto", "unidades_vendidas"],
    "inventario": ["id_tienda", "id_producto", "stock_actual"],
    "catalogo": ["id_producto", "nombre", "categoria", "costo_unitario", "precio_venta",
                 "costo_almacenamiento_semanal"],
    "tiendas": ["id_tienda", "ciudad", "tamaño_m2"],
}


@dataclass(frozen=True)
class DatosCasoA:
    """Las cuatro tablas del Caso A que alimentan el pronóstico y el pedido."""

    ventas: pd.DataFrame
    inventario: pd.DataFrame
    catalogo: pd.DataFrame
    tiendas: pd.DataFrame


def cargar_caso_a(data_dir: Optional[Union[str, Path]] = None) -> DatosCasoA:
    """Lee los CSV del Caso A.

    Args:
        data_dir: carpeta con `ventas_historicas.csv`, `inventario_actual.csv`,
            `catalogo_productos.csv` y `maestro_tiendas.csv`. Por defecto,
            `data/01_supply_optimization` en la raíz del repositorio.

    Returns:
        Un `DatosCasoA` con las tablas ya tipadas (fecha como datetime).

    Raises:
        FileNotFoundError: si falta alguno de los archivos.
    """
    carpeta = Path(data_dir) if data_dir is not None else DATA_DIR_DEFECTO
    archivos = {
        "ventas": "ventas_historicas.csv",
        "inventario": "inventario_actual.csv",
        "catalogo": "catalogo_productos.csv",
        "tiendas": "maestro_tiendas.csv",
    }
    faltan = [a for a in archivos.values() if not (carpeta / a).exists()]
    if faltan:
        raise FileNotFoundError(
            f"Faltan en {carpeta}: {', '.join(faltan)}. Revisa la sección de datos del README."
        )
    ventas = pd.read_csv(carpeta / archivos["ventas"], parse_dates=["fecha"])
    return DatosCasoA(
        ventas=ventas.sort_values(["id_tienda", "id_producto", "fecha"]).reset_index(drop=True),
        inventario=pd.read_csv(carpeta / archivos["inventario"]),
        catalogo=pd.read_csv(carpeta / archivos["catalogo"]),
        tiendas=pd.read_csv(carpeta / archivos["tiendas"]),
    )


def validar_datos(datos: DatosCasoA) -> List[str]:
    """Revisa estructura, grilla completa, nulos, negativos y llaves cruzadas.

    Returns:
        Lista de problemas encontrados; vacía si los datos están en orden.
    """
    problemas: List[str] = []
    tablas = {"ventas": datos.ventas, "inventario": datos.inventario,
              "catalogo": datos.catalogo, "tiendas": datos.tiendas}
    for nombre, df in tablas.items():
        faltan = set(COLUMNAS_ESPERADAS[nombre]) - set(df.columns)
        if faltan:
            problemas.append(f"{nombre}: faltan columnas {sorted(faltan)}")
        elif df[COLUMNAS_ESPERADAS[nombre]].isna().any().any():
            problemas.append(f"{nombre}: hay valores nulos")
    if problemas:
        return problemas

    v = datos.ventas
    if (v["unidades_vendidas"] < 0).any():
        problemas.append("ventas: hay unidades negativas")
    if v.duplicated(["fecha", "id_tienda", "id_producto"]).any():
        problemas.append("ventas: hay filas duplicadas por fecha-tienda-producto")
    n_dias = v["fecha"].nunique()
    esperado = v["id_tienda"].nunique() * v["id_producto"].nunique() * n_dias
    if len(v) != esperado:
        problemas.append(f"ventas: grilla incompleta ({len(v):,} filas de {esperado:,} esperadas)")
    if (v["fecha"].max() - v["fecha"].min()).days + 1 != n_dias:
        problemas.append("ventas: hay fechas faltantes en el rango")
    if not set(v["id_tienda"]) <= set(datos.tiendas["id_tienda"]):
        problemas.append("ventas: hay tiendas que no están en maestro_tiendas")
    if not set(v["id_producto"]) <= set(datos.catalogo["id_producto"]):
        problemas.append("ventas: hay productos que no están en el catálogo")
    llaves_v = set(zip(v["id_tienda"], v["id_producto"]))
    llaves_i = set(zip(datos.inventario["id_tienda"], datos.inventario["id_producto"]))
    if llaves_v != llaves_i:
        problemas.append("inventario: las llaves SKU-tienda no coinciden con las de ventas")
    if (datos.inventario["stock_actual"] < 0).any():
        problemas.append("inventario: hay stock negativo")
    return problemas
