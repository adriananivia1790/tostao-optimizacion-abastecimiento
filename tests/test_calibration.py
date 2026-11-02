import numpy as np

from tostao_abastecimiento.calibration import CalibradorCuantiles, cobertura_empirica


def test_cobertura_cercana_a_la_nominal_fuera_de_muestra():
    rng = np.random.default_rng(1)
    pron = rng.uniform(50, 150, 6000)
    real = pron * rng.lognormal(0, 0.15, 6000)
    cal = CalibradorCuantiles().ajustar(real[:3000], pron[:3000])
    for tau in (0.5, 0.8, 0.9):
        q = cal.cuantil_demanda(np.full(3000, tau), pron[3000:])
        assert abs(cobertura_empirica(real[3000:], q) - tau) < 0.03


def test_cuantil_es_creciente_en_tau():
    rng = np.random.default_rng(2)
    pron = rng.uniform(10, 100, 2000)
    real = pron * rng.lognormal(0, 0.2, 2000)
    cal = CalibradorCuantiles().ajustar(real, pron)
    q50 = cal.cuantil_demanda(np.full(2000, 0.5), pron)
    q90 = cal.cuantil_demanda(np.full(2000, 0.9), pron)
    assert (q90 >= q50).all()


def test_perdida_cuantil_es_asimetrica():
    from tostao_abastecimiento.calibration import perdida_cuantil
    # cuantil 0.9: faltar por 10 duele 9; sobrar por 10 duele 1
    assert abs(perdida_cuantil([110], [100], 0.9) - 9.0) < 1e-9
    assert abs(perdida_cuantil([90], [100], 0.9) - 1.0) < 1e-9


def test_comparar_incertidumbre_devuelve_tres_metodos(datos_sinteticos):
    from tostao_abastecimiento.calibration import comparar_incertidumbre
    from tostao_abastecimiento.features import construir_panel
    from tostao_abastecimiento.models import PerfilPorNivel
    from tostao_abastecimiento.validation import backtest_origen_movil
    bt = backtest_origen_movil(construir_panel(datos_sinteticos), {"perfil": PerfilPorNivel})
    r = comparar_incertidumbre(bt, "perfil")
    assert set(r["metodo"]) == {"normal", "empirica_global", "empirica_por_volumen"}
    assert len(r) == 12 and r["perdida_cuantil"].gt(0).all()
