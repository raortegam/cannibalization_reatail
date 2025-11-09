#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para re-ejecutar Step 4 (pre_algorithm) SIN filtros de calidad de donantes.
Esto permite que todos los donantes se incluyan en los archivos de GSC.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

def setup_logging(exp_tag: str) -> Path:
    log_dir = Path("diagnostics")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"step4_{exp_tag}_no_filters.log"
    
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
    logger.info("INICIANDO STEP 4: PRE_ALGORITHM (SIN FILTROS)")
    logger.info("Experimento: %s", exp_tag)
    logger.info("=" * 60)
    
    # Parámetros - FILTROS DESACTIVADOS
    config = {
        "episodes": f".data/processed_data/{exp_tag}/pairs_windows.csv",
        "donors": f".data/processed_data/{exp_tag}/donors_per_victim.csv",
        "raw": ".data/raw_data",
        "out": f".data/processed/{exp_tag}",
        "out_meta": f".data/processed_meta/{exp_tag}",
        # FILTROS DESACTIVADOS - Permitir todos los donantes
        "max_donor_promo_share": "1.0",    # 100% (sin filtro)
        "min_availability_share": "0.0",   # 0% (sin filtro)
        # Otros parámetros de A_base
        "top_k": "20",
        "donor_kind": "same_item",
        "lags_days": "7,14,28,56",
        "fourier_k": "3",
        "gsc_eval_n": "999999",  # Todos los episodios
        "gsc_eval_selection": "head",
    }
    
    # Validar entrada
    for key in ["episodes", "donors"]:
        path = Path(config[key])
        if not path.exists():
            logger.error("❌ No existe: %s", config[key])
            raise FileNotFoundError(f"Falta: {config[key]}")
        logger.info("✅ Archivo OK: %s", config[key])
    
    start_time = time.time()
    
    try:
        # Ejecutar directamente el script
        script_path = Path(__file__).resolve().parent / "src" / "preprocess_data" / "4. pre_algorithm.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--episodes", config["episodes"],
            "--donors", config["donors"],
            "--raw", config["raw"],
            "--out", config["out"],
            "--out_meta", config["out_meta"],
            "--max_donor_promo_share", config["max_donor_promo_share"],
            "--min_availability_share", config["min_availability_share"],
            "--top_k", config["top_k"],
            "--lags", config["lags_days"],
            "--fourier_k", config["fourier_k"],
            "--gsc_eval_n", config["gsc_eval_n"],
            "--gsc_eval_selection", config["gsc_eval_selection"],
            "--log_level", "INFO",
        ]
        
        logger.info("Comando: %s", " ".join(cmd))
        logger.info("\n⚠️  FILTROS DE CALIDAD DESACTIVADOS:")
        logger.info("  max_donor_promo_share: 1.0 (100% - sin filtro)")
        logger.info("  min_availability_share: 0.0 (0% - sin filtro)")
        logger.info("  Esto permitirá que TODOS los donantes se incluyan\n")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        
        if result.returncode == 0:
            logger.info("✅ Script ejecutado exitosamente")
            if result.stdout:
                logger.info("STDOUT:\n%s", result.stdout[-2000:])  # Últimas 2000 chars
        else:
            logger.error("❌ Error (código %d)", result.returncode)
            if result.stderr:
                logger.error("STDERR:\n%s", result.stderr)
            raise RuntimeError(f"Falló con código {result.returncode}")
        
        execution_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("STEP 4 COMPLETADO (SIN FILTROS)")
        logger.info("Tiempo: %.2f segundos (%.2f minutos)", execution_time, execution_time/60)
        logger.info("=" * 60)
        
        # Verificar outputs
        out_dir = Path(config["out"])
        gsc_dir = out_dir / "gsc"
        
        if gsc_dir.exists():
            n_episodes = len(list(gsc_dir.glob("*.parquet")))
            logger.info("✅ Episodios GSC: %d archivos", n_episodes)
        
        logger.info("✅ Step 4 A_base (sin filtros) finalizado")
        logger.info("\n⚠️  IMPORTANTE: Ahora los archivos de GSC incluyen TODOS los donantes")
        logger.info("   Puedes verificar con: python diagnose_dates.py")
        
    except Exception as e:
        logger.error("❌ Error: %s", e)
        logger.exception("Traceback:")
        raise

if __name__ == "__main__":
    main()
