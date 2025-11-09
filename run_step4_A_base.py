#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para ejecutar Step 4 (pre_algorithm) para A_base.
Ejecuta como subprocess para evitar problemas de importación.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

def setup_logging(exp_tag: str) -> Path:
    log_dir = Path("diagnostics")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"step4_{exp_tag}.log"
    
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
    logger = logging.getLogger(f"step4_{exp_tag}")
    
    logger.info("=" * 60)
    logger.info("INICIANDO STEP 4: PRE-ALGORITHM")
    logger.info("Experimento: %s", exp_tag)
    logger.info("=" * 60)
    
    exp_config = {
        "params": {
            "top_k_donors": 12,
            "lags_days": [7, 14, 28, 56],
            "fourier_k": 3,
            "use_stl": True,
        }
    }
    
    config = {
        "episodes_index": ".data/processed_data/A_base/episodes_index.parquet",
        "donors_csv": ".data/processed_data/A_base/donors_per_victim.csv",
        "raw_dir": ".data/raw_data",  # Directorio, no archivo
        "out_gsc": ".data/processed/A_base/gsc",
        "out_meta": ".data/processed_meta/A_base",
        "top_k_donors": exp_config["params"]["top_k_donors"],
        "lags_days": exp_config["params"]["lags_days"],
        "fourier_k": exp_config["params"]["fourier_k"],
        "use_stl": exp_config["params"]["use_stl"],
        "gsc_eval_n": 10,
        "meta_units": "victims_plus_donors"
    }
    
    # Validar entrada
    for key in ["episodes_index", "donors_csv", "raw_dir"]:
        if not Path(config[key]).exists():
            logger.error("❌ Archivo no existe: %s", config[key])
            raise FileNotFoundError(f"Falta: {config[key]}")
        logger.info("✅ Archivo OK: %s", config[key])
    
    start_time = time.time()
    
    try:
        src_path = Path(__file__).resolve().parent / "src"
        prep_path = src_path / "preprocess_data" / "4. pre_algorithm.py"
        
        cmd = [
            sys.executable,
            str(prep_path),
            "--episodes", config["episodes_index"],
            "--donors", config["donors_csv"],
            "--raw", config["raw_dir"],
            "--out", config["out_gsc"],
            "--out_meta", config["out_meta"],
            "--top_k", str(config["top_k_donors"]),
            "--lags", ",".join(map(str, config["lags_days"])),
            "--fourier_k", str(config["fourier_k"]),
            "--gsc_eval_n", str(config["gsc_eval_n"]),
            "--meta_units", config["meta_units"],
            "--log_level", "INFO"
        ]
        
        if config["use_stl"]:
            # El argumento es --skip_stl para desactivar, así que NO lo agregamos si use_stl=True
            pass
        else:
            cmd.append("--skip_stl")
        
        logger.info("Comando: %s", " ".join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        
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
        logger.info("STEP 4 COMPLETADO")
        logger.info("Tiempo: %.2f segundos (%.2f minutos)", execution_time, execution_time/60)
        logger.info("=" * 60)
        
        # Verificar outputs
        gsc_dir = Path(config["out_gsc"])
        meta_dir = Path(config["out_meta"])
        
        if gsc_dir.exists():
            gsc_files = list(gsc_dir.glob("*.parquet"))
            logger.info("GSC: %d archivos generados", len(gsc_files))
        
        if meta_dir.exists():
            meta_file = meta_dir / "windows.parquet"
            if meta_file.exists():
                size = meta_file.stat().st_size / (1024*1024)
                logger.info("Meta: windows.parquet (%.1f MB)", size)
        
        logger.info("✅ Step 4 A_base finalizado")
        
    except Exception as e:
        logger.error("❌ Error: %s", e)
        logger.exception("Traceback:")
        raise

if __name__ == "__main__":
    main()
