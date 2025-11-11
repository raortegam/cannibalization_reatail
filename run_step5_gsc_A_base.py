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
        # Parámetros GSC de A_base - ANTI-OVERFITTING con Rolling Window CV
        "rank": 3,           # Rank moderado (balance capacidad/regularización)
        "tau": 0.01,         # AUMENTADO 10x para mayor regularización nuclear
        "alpha": 0.005,      # AUMENTADO 5x para mayor regularización L2
        "cv_folds": 5,       # AUMENTADO a 5 para rolling window CV
        "cv_holdout": 14,    # REDUCIDO a 14 días (ventanas más pequeñas)
        "cv_gap": 7,         # Gap de 7 días para evitar filtración temporal
        "train_gap": 7,      # Gap de 7 días antes del tratamiento
        "hpo_trials": 500,   # HPO con Optuna (500 trials para buena exploración)
        "include_covariates": True,
        # Grillas para fallback (si Optuna no está disponible)
        # Nota: Con HPO activo, Optuna explorará rangos más amplios automáticamente
        "grid_ranks": "1,2,3,4",                        # Limitado a 4 (evitar overfitting)
        "grid_tau": "0.005,0.01,0.02,0.05",             # Rangos MÁS ALTOS
        "grid_alpha": "0.001,0.005,0.01,0.02",          # Rangos MÁS ALTOS
    }
    
    # Mostrar parámetros clave para validación
    logger.info("="*60)
    logger.info("PARÁMETROS GSC ANTI-OVERFITTING:")
    logger.info("  rank: %s", config["rank"])
    logger.info("  tau: %s", config["tau"])
    logger.info("  alpha: %s", config["alpha"])
    logger.info("  cv_folds: %s", config["cv_folds"])
    logger.info("  cv_holdout: %s días", config["cv_holdout"])
    logger.info("  cv_gap: %s días", config["cv_gap"])
    logger.info("  hpo_trials: %s", config["hpo_trials"])
    logger.info("="*60)
    
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
            "--cv_gap", str(config["cv_gap"]),           # CRÍTICO: gap para rolling window
            "--train_gap", str(config["train_gap"]),
            "--grid_ranks", config["grid_ranks"],
            "--grid_tau", config["grid_tau"],
            "--grid_alpha", config["grid_alpha"],
            "--hpo_trials", str(config["hpo_trials"]),  # CRÍTICO: 500 trials
            "--log_level", "INFO",
        ]
        
        if config["include_covariates"]:
            cmd.append("--include_covariates")
        
        logger.info("Comando: %s", " ".join(cmd))
        logger.info("="*60)
        logger.info("EJECUTANDO GSC (output en tiempo real):")
        logger.info("="*60)
        
        # NO capturar output para ver logs en tiempo real
        result = subprocess.run(cmd, timeout=14400)
        
        if result.returncode == 0:
            logger.info("="*60)
            logger.info("✅ Script ejecutado exitosamente")
        else:
            logger.error("❌ Error (código %d)", result.returncode)
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
