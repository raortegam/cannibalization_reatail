#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para ejecutar Step 6 (Meta-learners) para A_base.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

def setup_logging(exp_tag: str, learner: str) -> Path:
    log_dir = Path("diagnostics")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"step6_meta_{learner}_{exp_tag}.log"
    
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
    learner = "x"  # X-learner (de experiments.yaml)
    log_file = setup_logging(exp_tag, learner)
    logger = logging.getLogger(f"step6_meta_{learner}_{exp_tag}")
    
    logger.info("=" * 60)
    logger.info("INICIANDO STEP 6: META-LEARNERS (%s)", learner.upper())
    logger.info("Experimento: %s", exp_tag)
    logger.info("=" * 60)
    
    config = {
        "learner": learner,
        "meta_parquet": f".data/processed_meta/{exp_tag}/windows.parquet",
        "episodes_index": f".data/processed/{exp_tag}/episodes_index.parquet",
        "donors_csv": f".data/processed_data/{exp_tag}/donors_per_victim.csv",
        "out_dir": f".data/processed_data/{exp_tag}/meta_outputs/{learner}",
        "exp_tag": exp_tag,
        # Parámetros Meta de A_base
        "cv_folds": 3,
        "cv_holdout": 21,
        "hpo_trials": 10,
        "treat_col_s": "H_disc",
    }
    
    # Validar entrada
    for key in ["meta_parquet", "episodes_index", "donors_csv"]:
        if not Path(config[key]).exists():
            logger.error("❌ No existe: %s", config[key])
            raise FileNotFoundError(f"Falta: {config[key]}")
        logger.info("✅ Archivo OK: %s", config[key])
    
    start_time = time.time()
    
    try:
        src_path = Path(__file__).resolve().parent / "src"
        meta_path = src_path / "models" / "meta_learners.py"
        
        cmd = [
            sys.executable,
            str(meta_path),
            "--learner", config["learner"],
            "--meta_parquet", config["meta_parquet"],
            "--episodes_index", config["episodes_index"],
            "--donors_csv", config["donors_csv"],
            "--out_dir", config["out_dir"],
            "--exp_tag", config["exp_tag"],
            "--cv_folds", str(config["cv_folds"]),
            "--cv_holdout", str(config["cv_holdout"]),
            "--hpo_trials", str(config["hpo_trials"]),
            "--treat_col_s", config["treat_col_s"],
            "--log_level", "INFO",
        ]
        
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
        logger.info("STEP 6 META-%s COMPLETADO", learner.upper())
        logger.info("Tiempo: %.2f segundos (%.2f minutos)", execution_time, execution_time/60)
        logger.info("=" * 60)
        
        # Verificar outputs
        out_dir = Path(config["out_dir"])
        metrics_file = out_dir / f"meta_metrics_{learner}.parquet"
        cf_dir = out_dir / "cf_series"
        
        if metrics_file.exists():
            logger.info("✅ Métricas: %s", metrics_file)
        if cf_dir.exists():
            n_cf = len(list(cf_dir.glob("*.parquet")))
            logger.info("✅ Series CF: %d archivos", n_cf)
        
        logger.info("✅ Step 6 Meta-%s A_base finalizado", learner.upper())
        
    except Exception as e:
        logger.error("❌ Error: %s", e)
        logger.exception("Traceback:")
        raise

if __name__ == "__main__":
    main()
