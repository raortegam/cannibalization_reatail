#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar visualizaciones EDA de algoritmos para A_base.
"""

import logging
import sys
from pathlib import Path

# Añadir EDA al path
sys.path.insert(0, str(Path(__file__).parent / "EDA"))

from EDA_algorithms import EDAConfig, run

def main():
    exp_tag = "A_base"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger = logging.getLogger("eda_runner")
    
    logger.info("=" * 60)
    logger.info("INICIANDO EDA DE ALGORITMOS")
    logger.info("Experimento: %s", exp_tag)
    logger.info("=" * 60)
    
    config = EDAConfig(
        episodes_index=Path(f".data/processed_data/{exp_tag}/episodes_index.parquet"),
        gsc_out_dir=Path(f".data/processed_data/{exp_tag}/gsc"),
        meta_out_root=Path(f".data/processed_data/{exp_tag}"),
        meta_learners=(),
        figures_dir=Path(f"figures/{exp_tag}"),
        orientation="landscape",
        dpi=300,
        font_size=10,
        grid=True,
        export_pdf=True,
    )
    
    logger.info("Configuración:")
    logger.info("  episodes_index: %s", config.episodes_index)
    logger.info("  gsc_out_dir: %s", config.gsc_out_dir)
    logger.info("  meta_out_root: %s", config.meta_out_root)
    logger.info("  figures_dir: %s", config.figures_dir)
    
    # Validar entrada
    if not config.episodes_index.exists():
        logger.error("❌ No existe: %s", config.episodes_index)
        return
    
    logger.info("Generando visualizaciones...")
    try:
        run(config)
        logger.info("=" * 60)
        logger.info("✅ EDA COMPLETADO")
        logger.info("=" * 60)
        logger.info("Archivos generados en: %s", config.figures_dir)
        logger.info("")
        logger.info("Archivos principales:")
        logger.info("  - eda_algorithms_summary.pdf")
        logger.info("  - eda_algorithms_series.pdf")
        logger.info("  - tables/eda_algorithms_coverage.parquet")
        
    except Exception as e:
        logger.error("❌ Error: %s", e)
        logger.exception("Traceback:")
        raise

if __name__ == "__main__":
    main()
