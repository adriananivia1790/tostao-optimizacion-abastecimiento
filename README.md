# Tostao · Caso A: Optimización de abastecimiento

Prueba técnica para Subgerente de Data. El proyecto pronostica la demanda semanal por SKU-tienda para la semana del **lunes 1 al domingo 7 de abril de 2024**, con incertidumbre, y calcula la cantidad a pedir que minimiza el costo esperado de quedarse corto o pasarse (Newsvendor).

## Contenido

| Ruta | Qué es |
|---|---|
| `notebooks/01_EDA_abastecimiento_tostao.ipynb` | Análisis exploratorio: calidad de datos, estructura de la demanda, economía del producto, hipótesis y decisiones de modelado. |
| `notebooks/02_experimentacion_pronostico.ipynb` | Experimentación: backtest de origen móvil, comparación de modelos, sensibilidad, calibración de cuantiles y pronóstico final. |
| `notebooks/03_optimizacion_costos.ipynb` | Pedido óptimo, simulación de costos contra reglas simples y sensibilidad a los parámetros que no vienen en los datos. |
| `src/tostao_abastecimiento/` | Código modular con tipos y docstrings (ver arquitectura). |
| `scripts/generar_pedido.py` | Ejecuta todo el flujo desde la terminal y escribe los CSV de salida. |
| `tests/` | Pruebas unitarias (incluye una contra fuga de información). |
| `outputs/` | Resultados generados: `pronostico_semanal.csv`, `pedido_semana.csv`, `metricas_backtest.csv`, `simulacion_costos.csv`. |

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/adriananivia1790/<nombre-del-repo>.git
cd <nombre-del-repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # opcional: instala el paquete
```

## Datos

Los datos no se publican. Colócalos como indica `data/README.md` (carpeta `data/01_supply_optimization/` con los cinco CSV del Caso A; el EDA además lee `02_product_bundles` y `03_aov_drivers`).

## Cómo ejecutar

**Flujo completo desde la terminal:**

```bash
python scripts/generar_pedido.py --data-dir data/01_supply_optimization --salida outputs
# Parámetros del pedido (los que no vienen en los datos):
python scripts/generar_pedido.py --lead-time 2 --revision 7 --beta 0.25
```

Imprime las métricas del backtest, la cobertura fuera de muestra de los cuantiles, la simulación de costos y el pedido total.

**Notebooks:** abrir con `jupyter lab` y ejecutar en orden 01 → 02 → 03. Los notebooks 02 y 03 escriben sus resultados en `outputs/`.

**Pruebas:**

```bash
pytest
```

## Arquitectura

```
datos ──► data.py ──► features.py ──► validation.py (backtest origen móvil) ◄── models.py
                          │                    │
                          │                    ▼
                          │             calibration.py (cuantiles empíricos por volumen)
                          ▼                    │
                    pipeline.py ◄──────────────┘
                          │
                          ▼
                    optimizer.py (Newsvendor) ──► simulacion.py (costos) ──► sensibilidad.py
```

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Constantes y supuestos declarados (α por producto, margen cruzado, fechas). |
| `data.py` | Carga y validación de las tablas del Caso A. |
| `features.py` | Panel SKU-tienda-día y variables sin fuga de información (rezagos, promedios y medias móviles al domingo previo). |
| `models.py` | Naive (misma semana pasada), media de 4 semanas del mismo día, perfil × nivel y gradient boosting con pérdida de Poisson. |
| `validation.py` | WAPE, semanas de prueba y backtest de origen móvil (semanas 6 a 13). |
| `calibration.py` | Cuantiles de la demanda a partir del error relativo semanal del backtest, por terciles de volumen; cobertura secuencial fuera de muestra. |
| `optimizer.py` | Cu, Co, ratio crítico τ y pedido = máx(0, Q\*(τ) − stock − en tránsito); incertidumbre normal o empírica. |
| `simulacion.py` | Costo de cada política de pedido sobre las semanas de prueba. |
| `sensibilidad.py` | Escenarios de lead time, revisión, tránsito, α, β, empaque y capacidad. |
| `pipeline.py` | Orquesta todo y escribe los CSV. |

## Decisiones principales

- **Validación fuera de muestra siempre.** Cada semana se pronostica con un modelo entrenado solo con las anteriores. Se reporta el WAPE medio y su desviación entre semanas (8 pliegues).
- **Referencias simples primero.** El modelo de boosting solo se prefiere si supera a la media de 4 semanas por más que la desviación entre semanas y en la mayoría de las semanas (regla impresa en el notebook 02).
- **Pérdida de Poisson** porque la demanda son conteos no negativos con varianza que crece con el nivel. No pondera por importancia del producto; eso se hace con el WAPE (que pesa por volumen).
- **Cuantiles empíricos** en lugar de suponer una distribución. Se comprueba que la cobertura observada fuera de muestra se acerque a la nominal.
- **Newsvendor:** Cu = margen + β × margen cruzado; Co = almacenamiento + α × costo unitario; τ = Cu / (Cu + Co).

## Supuestos y límites

- **Parámetros que no vienen en los datos** y se declaran como supuestos: lead time (1 día), revisión (7 días), tránsito (0), MOQ y empaque (1), capacidad (sin límite), β = 25 % y α por producto (según vida útil de referencia). La sensibilidad en el notebook 03 muestra cuáles mueven más el resultado; α es el más influyente.
- **El inventario es una foto**, sin historia: no permite evaluar la política actual ni reconstruir el stock en el backtest. La simulación de costos asume que cada semana parte sin stock.
- **Pocas semanas de historia:** no hay estacionalidad mensual o anual que aprender.
- **El EDA menciona gradient boosting en general;** la implementación usa `HistGradientBoostingRegressor` de scikit-learn (sin dependencias adicionales).
- **Navegación en GitHub:** los enlaces internos del índice de los notebooks no funcionan en el visor de GitHub; usar el índice como lista de contenido.

## Datos que pediría al negocio

Historial diario de stock, pedidos recibidos y mermas; lead time y calendario de entrega por proveedor; MOQ y empaque; capacidad de almacenamiento por tienda; órdenes en tránsito; merma real por producto.
