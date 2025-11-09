#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para ejecutar Step 3 (select_pairs_and_donors) para A_base.
Ejecuta como subprocess para evitar problemas de importación.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

def setup_logging(exp_tag: str) -> Path:
    log_dir = Path("diagnostics")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"step3_{exp_tag}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    return log_file

def main():
    exp_tag = "A_base"
    log_file = setup_logging(exp_tag)
    logger = logging.getLogger(f"step3_{exp_tag}")
    
    logger.info("=" * 60)
    logger.info("INICIANDO STEP 3: SELECT PAIRS AND DONORS")
    logger.info("Experimento: %s", exp_tag)
    logger.info("=" * 60)
    
    config = {
        "exposure_csv": ".data/processed_data/A_base/competitive_exposure.csv",
        "train_filtered": ".data/processed_data/A_base/train_filtered.csv",
        "items_csv": ".data/raw_data/items.csv",
        "stores_csv": ".data/raw_data/stores.csv",
        "out_dir": ".data/processed_data",
        "exp_tag": exp_tag,
        "top_k_donors": 12,  # De experiments.yaml A_base
    }
    
    # Validar archivos de entrada
    for key in ["exposure_csv", "train_filtered", "items_csv", "stores_csv"]:
        if not Path(config[key]).exists():
            logger.error("❌ Archivo no existe: %s -> %s", key, config[key])
            raise FileNotFoundError(f"Falta: {config[key]}")
        logger.info("✅ Archivo OK: %s", config[key])
    
    start_time = time.time()
    
    try:
        src_path = Path(__file__).resolve().parent / "src"
        spd_path = src_path / "preprocess_data" / "3. select_pairs_and_donors.py"
        
        cmd = [
            sys.executable,
            str(spd_path),
            "--H", config["exposure_csv"],
            "--train", config["train_filtered"],
            "--items", config["items_csv"],
            "--stores", config["stores_csv"],
            "--out_dir", str(config["out_dir"]),
            "--exp_tag", config["exp_tag"],
            "--top_k_donors", str(config["top_k_donors"]),
            "--log_level", "INFO"
        ]
        
        logger.info("Comando: %s", " ".join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            logger.info("✅ Script ejecutado exitosamente")
            if result.stdout:
                logger.info("STDOUT:\n%s", result.stdout)
        else:
            logger.error("❌ Error (código %d)", result.returncode)
            if result.stderr:
                logger.error("STDERR:\n%s", result.stderr)
            raise RuntimeError(f"Falló con código {result.returncode}")
        
        execution_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("STEP 3 COMPLETADO")
        logger.info("Tiempo: %.2f segundos", execution_time)
        logger.info("=" * 60)
        
        out_dir = Path(config["out_dir"]) / config["exp_tag"]
        outputs = {
            "pairs": str(out_dir / "pairs_windows.csv"),
            "donors": str(out_dir / "donors_per_victim.csv"),
            "episodes": str(out_dir / "episodes_index.parquet")
        }
        
        for key, path in outputs.items():
            if Path(path).exists():
                size = Path(path).stat().st_size / (1024*1024)
                logger.info("  - %s: %.1f MB", key, size)
        
        logger.info("✅ Step 3 A_base finalizado")
        
    except Exception as e:
        logger.error("❌ Error: %s", e)
        logger.exception("Traceback:")
        raise

if __name__ == "__main__":
    main()
