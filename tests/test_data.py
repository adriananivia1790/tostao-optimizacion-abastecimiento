from dataclasses import replace

from tostao_abastecimiento.data import validar_datos


def test_datos_validos_no_tienen_problemas(datos_sinteticos):
    assert validar_datos(datos_sinteticos) == []


def test_detecta_grilla_incompleta(datos_sinteticos):
    incompleto = replace(datos_sinteticos, ventas=datos_sinteticos.ventas.iloc[:-1])
    assert any("grilla incompleta" in p for p in validar_datos(incompleto))


def test_detecta_negativos(datos_sinteticos):
    v = datos_sinteticos.ventas.copy()
    v.loc[0, "unidades_vendidas"] = -1
    assert any("negativas" in p for p in validar_datos(replace(datos_sinteticos, ventas=v)))
