"""Genera el pronóstico y el pedido de la semana 1–7 de abril de 2024.

Uso:
    python scripts/generar_pedido.py --data-dir data/01_supply_optimization --salida outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tostao_abastecimiento.optimizer import ParametrosPedido  # noqa: E402
from tostao_abastecimiento.pipeline import ejecutar_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None, help="Carpeta con los CSV del Caso A")
    ap.add_argument("--salida", default="outputs", help="Carpeta de resultados")
    ap.add_argument("--lead-time", type=int, default=1, help="Días entre pedir y recibir (supuesto)")
    ap.add_argument("--revision", type=int, default=7, help="Días que cubre cada pedido")
    ap.add_argument("--beta", type=float, default=0.25, help="Fracción de la venta cruzada que se pierde")
    args = ap.parse_args()

    p = ParametrosPedido(lead_time_dias=args.lead_time, revision_dias=args.revision, beta=args.beta)
    r = ejecutar_pipeline(args.data_dir, p, args.salida)

    print("WAPE diario, media ± desviación entre semanas de prueba:")
    print(r.metricas.round(3).to_string())
    print("\nCobertura de los cuantiles (fuera de muestra: cada semana se calibra con las anteriores):")
    print(r.cobertura.round(3).to_string())
    print("\nSimulación de costos en el backtest:")
    print(r.simulacion_resumen.round(2).to_string())
    print(f"\nPedido total: {r.pedido['pedido'].sum():,.0f} u en {int((r.pedido['pedido'] > 0).sum())} SKU-tienda")
    print(f"Resultados guardados en: {Path(args.salida).resolve()}")


if __name__ == "__main__":
    main()
