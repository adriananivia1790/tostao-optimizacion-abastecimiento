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
| `tests/` | Pruebas unitarias con datos sintéticos (incluye una contra fuga de información) y una prueba de punta a punta con los CSV reales. |
| `outputs/` | Resultados generados: `pronostico_semanal.csv`, `pedido_semana.csv`, `metricas_backtest.csv`, `simulacion_costos.csv`. |

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/adriananivia1790/tostao-optimizacion-abastecimiento.git
cd tostao-optimizacion-abastecimiento
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
pytest                                  # unitarias, con datos sintéticos
pytest tests/test_datos_reales.py -v    # punta a punta con los CSV reales (se omite si no están)
```

## Qué entrega el flujo

Los resultados quedan en `outputs/`:

| Archivo | Contenido | Uso |
|---|---|---|
| `pedido_semana.csv` | Por tienda y producto: stock actual, demanda esperada del horizonte, τ (nivel de cobertura del producto), cantidad objetivo, **pedido recomendado** y venta perdida inevitable | Compras y abastecimiento |
| `pronostico_semanal.csv` | Pronóstico semanal con rango: media, P10, P50 y P90 | Planeación |
| `metricas_backtest.csv` | Error (WAPE) de cada modelo en el backtest | Seguimiento de calidad |
| `simulacion_costos.csv` | Costo, nivel de servicio y sobrante de cada política de pedido | Caso de negocio |

**Dos horizontes distintos.** `pronostico_semanal.csv` cubre los 7 días de la semana a pronosticar (1 al 7 de abril). `pedido_semana.csv` trabaja con el horizonte del pedido, que es el lead time más la revisión (1 + 7 = 8 días, ambos supuestos declarados): `demanda_media_horizonte` y `q_objetivo` son de 8 días. Por eso no son comparables con la media ni con el P90 del pronóstico semanal, y es normal que el objetivo de un producto quede por encima de su P90 semanal.

Para conectarlo a otro sistema basta con alimentar los 4 CSV de entrada (ventas, inventario, catálogo y tiendas) y consumir `pedido_semana.csv`. La integración automática con un ERP o con el inventario no está incluida; es un siguiente paso.

## Resultados principales

Con los datos del Caso A (backtest de 8 semanas, pedido para el 1 al 7 de abril de 2024):

| Indicador | Resultado |
|---|---|
| WAPE diario: naive / media 4 semanas / perfil × nivel / boosting | 0.342 / 0.274 / 0.261 / 0.261 |
| Cobertura fuera de muestra de los cuantiles 0.8 / 0.9 / 0.95 | 0.805 / 0.905 / 0.948 |
| Costo simulado (6 semanas): Newsvendor frente a pedir lo de la semana pasada | −33.6 % |
| Newsvendor frente a la media de 4 semanas / a un nivel de servicio fijo de 0.90 | −25 % / −16 % |
| Pedido recomendado | 15,453 unidades en 158 de 160 SKU-tienda |

Los costos usan supuestos (α, β, márgenes) y las políticas se evalúan con los mismos Cu y Co con los que se optimiza el Newsvendor; el ahorro es la ventaja bajo esos supuestos, no una medición del costo real.

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
| `features.py` | Panel SKU-tienda-día y variables sin fuga de información (rezagos, promedios y medias móviles al domingo previo); variables extra para los experimentos. |
| `models.py` | Naive (misma semana pasada), media de 4 semanas del mismo día, perfil × nivel y gradient boosting con pérdida de Poisson; ensamble y modelo por segmento de volumen (experimentos). |
| `validation.py` | WAPE, semanas de prueba y backtest de origen móvil (semanas 6 a 13). |
| `calibration.py` | Cuantiles de la demanda a partir del error relativo semanal del backtest, por terciles de volumen; cobertura secuencial fuera de muestra; pérdida de cuantil (pinball) y comparación normal vs. empírica. |
| `optimizer.py` | Cu, Co, ratio crítico τ y pedido = máx(0, Q\*(τ) − stock − en tránsito); incertidumbre normal o empírica. |
| `simulacion.py` | Costo de cada política de pedido sobre las semanas de prueba. |
| `sensibilidad.py` | Escenarios de lead time, revisión, tránsito, α, β, empaque y capacidad. |
| `pipeline.py` | Orquesta todo y escribe los CSV. |

## Decisiones principales

- **Validación fuera de muestra siempre.** Cada semana se pronostica con un modelo entrenado solo con las anteriores. Se reporta el WAPE medio y su desviación entre semanas (8 pliegues).
- **Referencias simples primero.** El boosting se compara contra la media de 4 semanas y contra el perfil × nivel (regla impresa en el notebook 02). Con estos datos mejora a la media de 4 semanas, pero empata con el perfil × nivel (WAPE 0.261 en ambos); se mantiene el boosting como modelo de trabajo y se deja documentado que existe una alternativa más simple con el mismo error.
- **Pérdida de Poisson** porque la demanda son conteos no negativos con varianza que crece con el nivel. No pondera por importancia del producto; eso se hace con el WAPE (que pesa por volumen).
- **Cuantiles empíricos** en lugar de suponer una distribución. Se comprueba que la cobertura observada fuera de muestra se acerque a la nominal.
- **Newsvendor:** Cu = margen + β × margen cruzado; Co = almacenamiento + α × costo unitario; τ = Cu / (Cu + Co). Un producto con τ alto (por ejemplo, Tinto, 0.92) se pide de forma agresiva y uno con τ bajo (Jugo de Naranja, 0.64) de forma conservadora, más cerca de la mediana.
- **Mejoras probadas y descartadas:** ensamble de modelos, modelo por segmento de volumen y variables adicionales (tamaño de tienda, ciudad, tendencia). Ninguna superó el umbral definido en el notebook 02.

## Supuestos y límites

- **Parámetros que no vienen en los datos** y se declaran como supuestos: lead time (1 día), revisión (7 días), tránsito (0), MOQ y empaque (1), capacidad (sin límite), β = 25 % y α por producto (según vida útil de referencia). La sensibilidad en el notebook 03 muestra cuáles mueven más el pedido: cada día de lead time mueve el pedido ≈ 12 a 14 % y pasar de α = 0 a α = 1 lo reduce 22.9 %.
- **El inventario es una foto**, sin historia: no permite evaluar la política actual ni reconstruir el stock en el backtest. La simulación de costos asume que cada semana parte sin stock.
- **Pocas semanas de historia:** no hay estacionalidad mensual o anual que aprender.
- **Escala:** el modelo es global (un solo modelo para las 160 series SKU-tienda) y se validó con ese tamaño. Para miles de productos haría falta probar muestreo, un boosting más rápido o segmentación; no está probado.
- **El EDA menciona gradient boosting en general;** la implementación usa `HistGradientBoostingRegressor` de scikit-learn (sin dependencias adicionales).
- **Navegación en GitHub:** los enlaces internos del índice de los notebooks no funcionan en el visor de GitHub; usar el índice como lista de contenido.

## Datos que pediría al negocio

Historial diario de stock, pedidos recibidos y mermas; lead time y calendario de entrega por proveedor; MOQ y empaque; capacidad de almacenamiento por tienda; órdenes en tránsito; merma real por producto.
