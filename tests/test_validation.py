import numpy as np
import pytest

from tostao_abastecimiento.validation import wape


def test_wape_valor_conocido():
    y = np.array([10, 20, 5])
    p = np.array([8, 25, 5])
    assert wape(y, p) == pytest.approx(7 / 35)


def test_wape_perfecto_es_cero():
    assert wape(np.array([3, 4]), np.array([3, 4])) == 0


def test_wape_exige_ventas_positivas():
    with pytest.raises(ValueError):
        wape(np.array([0, 0]), np.array([1, 1]))
