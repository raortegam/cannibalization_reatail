# -*- coding: utf-8 -*-
"""
Patch para añadir logging estructurado a competitive_exposure.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

def enhanced_compute_competitive_exposure_with_logging(
    train_path: str,
    items_path: str = None,
    # ... (mismos parámetros que compute_competitive_exposure)
    save_path: str = None,
    save_format: str = None,
    log_output_dir: str = None
) -> Dict[str, Any]:
    """
    Versión mejorada con logging estructurado.
    Retorna diccionario con metadatos además del DataFrame.
    """
    from competitive_exposure import compute_competitive_exposure
    
    logging.info("Iniciando cómputo de exposición competitiva")
    
    # Ejecutar función original
    result_df = compute_competitive_exposure(
        train_path=train_path,
        items_path=items_path,
        # ... pasar todos los parámetros
        save_path=save_path,
        save_format=save_format
    )
    
    # Generar reporte estructurado
    report = {
        "step": "competitive_exposure",
        "input_train": train_path,
        "input_items": items_path,
        "output_path": save_path,
        "output_shape": result_df.shape,
        "columns": list(result_df.columns),
        "h_statistics": {
            "H_prop_mean": float(result_df["H_prop"].mean()),
            "H_prop_std": float(result_df["H_prop"].std()),
            "H_prop_min": float(result_df["H_prop"].min()),
            "H_prop_max": float(result_df["H_prop"].max()),
            "H_bin_rate": float(result_df["H_bin"].mean()),
            "H_disc_mean": float(result_df["H_disc"].mean()),
            "H_disc_max": int(result_df["H_disc"].max())
        },
        "data_quality": {
            "null_rate_H_prop": float(result_df["H_prop"].isna().mean()),
            "null_rate_H_prop_raw": float(result_df["H_prop_raw"].isna().mean()),
            "unique_dates": int(result_df["date"].nunique()),
            "unique_stores": int(result_df["store_nbr"].nunique()),
            "unique_items": int(result_df["item_nbr"].nunique())
        }
    }
    
    # Guardar reporte si se especificó directorio
    if log_output_dir:
        report_path = Path(log_output_dir) / "step2_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logging.info("Reporte de Step 2 guardado: %s", report_path)
    
    # Logging de estadísticas clave
    logging.info("Step 2 completado:")
    for key, value in report["h_statistics"].items():
        logging.info("  - %s: %.4f", key, value)
    
    return {
        "data": result_df,
        "report": report
    }

# Instrucciones para integrar:
# 1. Reemplazar la llamada a compute_competitive_exposure en 00_run_pipeline.py
# 2. Añadir parámetro log_output_dir=str(cfg.paths.processed_dir)
# 3. Actualizar el manifest con el reporte generado
