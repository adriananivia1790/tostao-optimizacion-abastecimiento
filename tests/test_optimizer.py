import numpy as np
import pytest

from tostao_abastecimiento.optimizer import (ParametrosPedido, calcular_pedido, costo_esperado,
                                              economia_producto, razon_critica)


def test_razon_critica_del_tinto():
    # Tinto: margen 1,700; almacenamiento 10; costo unitario 800.
    assert razon_critica(1700, 10) == pytest.approx(0.9942, abs=1e-4)
    assert razon_critica(1700, 810) == pytest.approx(0.6773, abs=1e-4)


def test_economia_usa_alfa_y_beta(datos_sinteticos):
    p = ParametrosPedido(beta=0.0, alfa={"PROD_001": 1.0, "PROD_002": 0.0})
    e = economia_producto(datos_sinteticos.catalogo, p).set_index("id_producto")
    assert e.loc["PROD_001", "co"] == 10 + 800
    assert e.loc["PROD_002", "co"] == 15
    assert e.loc["PROD_001", "cu"] == 1700


def test_pedido_respeta_multiplo_y_no_es_negativo(datos_sinteticos):
    p = ParametrosPedido(multiplo=6, moq=6, alfa={"PROD_001": 0.5, "PROD_002": 0.5})
    d = calcular_pedido(datos_sinteticos, p)
    assert (d["pedido"] >= 0).all()
    assert (d["pedido"] % 6 == 0).all()


def test_mas_lead_time_pide_mas(datos_sinteticos):
    alfa = {"PROD_001": 0.5, "PROD_002": 0.5}
    base = calcular_pedido(datos_sinteticos, ParametrosPedido(lead_time_dias=1, alfa=alfa))["pedido"].sum()
    mas = calcular_pedido(datos_sinteticos, ParametrosPedido(lead_time_dias=3, alfa=alfa))["pedido"].sum()
    assert mas > base


def test_transito_reduce_el_pedido(datos_sinteticos):
    alfa = {"PROD_001": 0.5, "PROD_002": 0.5}
    base = calcular_pedido(datos_sinteticos, ParametrosPedido(alfa=alfa))["pedido"].sum()
    con = calcular_pedido(datos_sinteticos, ParametrosPedido(alfa=alfa, en_transito_dias=1.0))["pedido"].sum()
    assert con < base


def test_costo_esperado_asimetrico():
    d = np.array([10.0, 10.0])
    q = np.array([8.0, 12.0])
    c = costo_esperado(d, q, cu=np.array([100.0, 100.0]), co=np.array([10.0, 10.0]))
    assert c[0] == 200 and c[1] == 20


def test_tabla_sensibilidad_tiene_la_base(datos_sinteticos):
    from tostao_abastecimiento.sensibilidad import tabla_sensibilidad
    t = tabla_sensibilidad(datos_sinteticos, ParametrosPedido(alfa={"PROD_001": 0.5, "PROD_002": 0.5}))
    assert t.loc["Base", "cambio_vs_base_%"] == 0
    assert t.loc["Lead time 3", "pedido_total_u"] > t.loc["Base", "pedido_total_u"]
