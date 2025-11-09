#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para ejecutar Step 5 (GSC) para A_base.
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
    log_file = log_dir / f"step5_gsc_{exp_tag}.log"
    
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
    logger = logging.getLogger(f"step5_gsc_{exp_tag}")
    
    logger.info("=" * 60)
    logger.info("INICIANDO STEP 5: GSC")
    logger.info("Experimento: %s", exp_tag)
    logger.info("=" * 60)
    
    config = {
        "gsc_dir": f".data/processed/{exp_tag}/gsc",
        "donors_csv": f".data/processed_data/{exp_tag}/donors_per_victim.csv",
        "episodes_index": f".data/processed/{exp_tag}/episodes_index.parquet",
        "out_dir": f".data/processed_data/{exp_tag}/gsc",
        # Parámetros GSC de A_base - ESTRATEGIA BALANCEADA
        "rank": 3,           # Aumentado a 3 (más capacidad para capturar patrones)
        "tau": 0.001,        # Reducido para permitir más flexibilidad
        "alpha": 0.001,      # Reducido para permitir más uso de covariables
        "cv_folds": 3,
        "cv_holdout": 21,
        "cv_gap": 7,         # Gap moderado en CV (7 días)
        "train_gap": 7,      # Gap moderado (7 días)
        "hpo_trials": 500,   # Aumentado de 300 a 500 para mejor exploración
        "include_covariates": True,
        # Grillas más amplias para HPO - mejor cobertura del espacio
        "grid_ranks": "1,2,3,4,5,6",                    # Más opciones de rank
        "grid_tau": "0.0001,0.0005,0.001,0.005,0.01",   # Más granularidad en tau
        "grid_alpha": "0.00005,0.0001,0.0005,0.001,0.005,0.01", # Más granularidad en alpha
    }
    
    # Validar entrada
    for key in ["gsc_dir", "donors_csv", "episodes_index"]:
        path = Path(config[key])
        if not path.exists():
            logger.error("❌ No existe: %s", config[key])
            raise FileNotFoundError(f"Falta: {config[key]}")
        
        if path.is_dir():
            n_files = len(list(path.glob("*.parquet")))
            logger.info("✅ Directorio OK: %s (%d episodios)", config[key], n_files)
        else:
            logger.info("✅ Archivo OK: %s", config[key])
    
    start_time = time.time()
    
    try:
        src_path = Path(__file__).resolve().parent / "src"
        gsc_path = src_path / "models" / "synthetic_control.py"
        
        cmd = [
            sys.executable,
            str(gsc_path),
            "--gsc_dir", config["gsc_dir"],
            "--donors_csv", config["donors_csv"],
            "--episodes_index", config["episodes_index"],
            "--out_dir", config["out_dir"],
            "--rank", str(config["rank"]),
            "--tau", str(config["tau"]),
            "--alpha", str(config["alpha"]),
            "--cv_folds", str(config["cv_folds"]),
            "--cv_holdout", str(config["cv_holdout"]),
            "--cv_gap", str(config["cv_gap"]),
            "--train_gap", str(config["train_gap"]),
            "--grid_ranks", config["grid_ranks"],
            "--grid_tau", config["grid_tau"],
            "--grid_alpha", config["grid_alpha"],
            "--hpo_trials", str(config["hpo_trials"]),
            "--log_level", "INFO",
        ]
        
        if config["include_covariates"]:
            cmd.append("--include_covariates")
        
        logger.info("Comando: %s", " ".join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        
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
        logger.info("STEP 5 GSC COMPLETADO")
        logger.info("Tiempo: %.2f segundos (%.2f minutos)", execution_time, execution_time/60)
        logger.info("=" * 60)
        
        # Verificar outputs
        out_dir = Path(config["out_dir"])
        metrics_file = out_dir / "gsc_metrics.parquet"
        cf_dir = out_dir / "cf_series"
        
        if metrics_file.exists():
            logger.info("✅ Métricas: %s", metrics_file)
        if cf_dir.exists():
            n_cf = len(list(cf_dir.glob("*.parquet")))
            logger.info("✅ Series CF: %d archivos", n_cf)
        
        logger.info("✅ Step 5 GSC A_base finalizado")
        
    except Exception as e:
        logger.error("❌ Error: %s", e)
        logger.exception("Traceback:")
        raise

if __name__ == "__main__":
    main()
