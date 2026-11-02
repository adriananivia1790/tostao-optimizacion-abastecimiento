"""Flujo completo: datos → variables → backtest → calibración → pronóstico → pedido."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import pandas as pd

from .calibration import CalibradorCuantiles, cobertura_secuencial
from .config import PRIMERA_SEMANA_PRUEBA
from .data import DatosCasoA, cargar_caso_a, validar_datos
from .features import construir_panel, dividir_historia_futuro
from .models import MediaCuatroSemanasMismoDia, ModeloPoissonGBM, NaiveSemanaPasada, PerfilPorNivel
from .optimizer import IncertidumbreEmpirica, ParametrosPedido, calcular_pedido
from .simulacion import simular_costos
from .validation import agregar_semanal, resumen_wape

MODELOS_BACKTEST: Dict[str, Callable[[], object]] = {
    "naive": NaiveSemanaPasada,
    "media4": MediaCuatroSemanasMismoDia,
    "perfil_nivel": PerfilPorNivel,
    "gbm_poisson": ModeloPoissonGBM,
}
MODELO_FINAL = "gbm_poisson"


@dataclass
class ResultadosPipeline:
    """Salidas del flujo completo."""

    backtest: pd.DataFrame
    metricas: pd.DataFrame
    cobertura: pd.DataFrame
    simulacion_por_semana: pd.DataFrame
    simulacion_resumen: pd.DataFrame
    pronostico: pd.DataFrame
    pedido: pd.DataFrame


def pronosticar_semana(panel: pd.DataFrame, datos: DatosCasoA,
                       calibrador: CalibradorCuantiles) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrena con todas las semanas observadas y pronostica la semana siguiente.

    Returns:
        (diario, semanal): `diario` tiene el pronóstico por SKU-tienda-día; `semanal`
        trae la media y los cuantiles P10, P50 y P90 calibrados.
    """
    historia, futuro = dividir_historia_futuro(panel)
    modelo = ModeloPoissonGBM().fit(historia)
    futuro = futuro.copy()
    futuro["pronostico"] = modelo.predict(futuro)
    diario = futuro[["id_tienda", "id_producto", "fecha", "pronostico"]].reset_index(drop=True)

    semanal = diario.groupby(["id_tienda", "id_producto"], as_index=False)["pronostico"].sum()
    semanal = semanal.rename(columns={"pronostico": "pronostico_media"})
    for etiqueta, tau in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
        semanal[etiqueta] = calibrador.cuantil_demanda(
            [tau] * len(semanal), semanal["pronostico_media"].to_numpy())
    semanal = semanal.merge(datos.catalogo[["id_producto", "nombre"]], on="id_producto")
    cols = ["id_tienda", "id_producto", "nombre", "pronostico_media", "p10", "p50", "p90"]
    return diario, semanal[cols].round(2)


def ejecutar_pipeline(data_dir: Optional[Union[str, Path]] = None,
                      params: Optional[ParametrosPedido] = None,
                      salida: Optional[Union[str, Path]] = None,
                      primera_semana: int = PRIMERA_SEMANA_PRUEBA) -> ResultadosPipeline:
    """Corre el flujo completo y, si se indica `salida`, guarda los resultados en CSV.

    Args:
        data_dir: carpeta con los CSV del Caso A (por defecto `data/01_supply_optimization`).
        params: parámetros del pedido (por defecto, los valores base declarados).
        salida: carpeta donde guardar los CSV; None no escribe nada.
        primera_semana: primera semana del backtest.

    Raises:
        ValueError: si los datos no pasan la validación estructural.
    """
    from .validation import backtest_origen_movil  # evita ciclo en importaciones tempranas

    p = params or ParametrosPedido()
    datos = cargar_caso_a(data_dir)
    problemas = validar_datos(datos)
    if problemas:
        raise ValueError("Datos inválidos: " + "; ".join(problemas))

    panel = construir_panel(datos)
    bt = backtest_origen_movil(panel, MODELOS_BACKTEST, primera_semana)
    metricas = resumen_wape(bt, MODELOS_BACKTEST.keys())

    semanal = agregar_semanal(bt, [MODELO_FINAL])
    calibrador = CalibradorCuantiles().ajustar(semanal["unidades_vendidas"], semanal[MODELO_FINAL])
    cobertura = cobertura_secuencial(semanal, MODELO_FINAL)
    sim_sem, sim_res = simular_costos(bt, datos.catalogo, p, modelo=MODELO_FINAL)

    diario, pronostico = pronosticar_semana(panel, datos, calibrador)
    pedido = calcular_pedido(datos, p, diario, IncertidumbreEmpirica(calibrador))

    res = ResultadosPipeline(bt, metricas, cobertura, sim_sem, sim_res, pronostico, pedido)
    if salida is not None:
        carpeta = Path(salida)
        carpeta.mkdir(parents=True, exist_ok=True)
        pronostico.to_csv(carpeta / "pronostico_semanal.csv", index=False)
        pedido_cols = ["id_tienda", "id_producto", "nombre", "stock_actual", "demanda_media_horizonte",
                       "tau", "q_objetivo", "pedido", "venta_perdida_inevitable"]
        pedido[pedido_cols].round(2).to_csv(carpeta / "pedido_semana.csv", index=False)
        metricas.round(4).to_csv(carpeta / "metricas_backtest.csv")
        sim_res.round(3).to_csv(carpeta / "simulacion_costos.csv")
    return res
