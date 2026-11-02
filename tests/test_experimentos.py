import numpy as np

from tostao_abastecimiento.features import agregar_variables_extra, construir_panel, dividir_historia_futuro
from tostao_abastecimiento.models import ModeloPoissonGBM, ModeloPorSegmento, PerfilPorNivel, PromedioModelos


def test_variables_extra_no_cambian_filas_ni_ventas(datos_sinteticos):
    panel = construir_panel(datos_sinteticos)
    extra = agregar_variables_extra(panel, datos_sinteticos)
    assert len(extra) == len(panel)
    assert {"tamano_m2", "ciudad_cod", "tendencia_rm"} <= set(extra.columns)
    assert extra["unidades_vendidas"].equals(panel["unidades_vendidas"])


def test_ensamble_es_el_promedio(datos_sinteticos):
    historia, _ = dividir_historia_futuro(construir_panel(datos_sinteticos))
    a, b = PerfilPorNivel().fit(historia), ModeloPoissonGBM(max_iter=30).fit(historia)
    ens = PromedioModelos([PerfilPorNivel, lambda: ModeloPoissonGBM(max_iter=30)]).fit(historia)
    assert np.allclose(ens.predict(historia), (a.predict(historia) + b.predict(historia)) / 2)


def test_modelo_por_segmento_predice_todas_las_filas(datos_sinteticos):
    historia, futuro = dividir_historia_futuro(construir_panel(datos_sinteticos))
    m = ModeloPorSegmento(lambda: ModeloPoissonGBM(max_iter=30), minimo_filas=50).fit(historia)
    p = m.predict(futuro)
    assert len(p) == len(futuro) and np.isfinite(p).all() and (p >= 0).all()
