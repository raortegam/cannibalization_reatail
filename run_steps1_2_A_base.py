#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para ejecutar Steps 1 y 2 para A_base:
  - Step 1: Filtrar train y calcular exposure
  - Step 2: Calcular competitive exposure
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
    log_file = log_dir / f"steps1_2_{exp_tag}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    return log_file

def run_step(step_num: int, script_path: Path, cmd: list, logger) -> float:
    """Ejecuta un step y retorna el tiempo de ejecución."""
    logger.info("=" * 60)
    logger.info("EJECUTANDO STEP %d", step_num)
    logger.info("=" * 60)
    logger.info("Comando: %s", " ".join(cmd))
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        logger.info("✅ Step %d completado (%.2f segundos)", step_num, elapsed)
        if result.stdout:
            logger.info("STDOUT:\n%s", result.stdout)
    else:
        logger.error("❌ Step %d falló (código %d)", step_num, result.returncode)
        if result.stderr:
            logger.error("STDERR:\n%s", result.stderr)
        if result.stdout:
            logger.error("STDOUT:\n%s", result.stdout)
        raise RuntimeError(f"Step {step_num} falló con código {result.returncode}")
    
    return elapsed

def main():
    exp_tag = "A_base"
    log_file = setup_logging(exp_tag)
    logger = logging.getLogger(f"steps1_2_{exp_tag}")
    
    logger.info("=" * 60)
    logger.info("INICIANDO STEPS 1 Y 2")
    logger.info("Experimento: %s", exp_tag)
    logger.info("=" * 60)
    
    config = {
        # Archivos de entrada (raw)
        "train_csv": ".data/raw_data/train.csv",
        "items_csv": ".data/raw_data/items.csv",
        "stores_csv": ".data/raw_data/stores.csv",
        "holidays_csv": ".data/raw_data/holidays_events.csv",
        # Salidas
        "out_dir": f".data/processed_data/{exp_tag}",
        "exp_tag": exp_tag,
        # Parámetros de A_base
        "h_bin_threshold": 0.02,  # De experiments.yaml
    }
    
    # Validar archivos de entrada
    for key in ["train_csv", "items_csv", "stores_csv", "holidays_csv"]:
        if not Path(config[key]).exists():
            logger.error("❌ Archivo no existe: %s", config[key])
            raise FileNotFoundError(f"Falta: {config[key]}")
        logger.info("✅ Archivo OK: %s", config[key])
    
    # Crear directorio de salida
    Path(config["out_dir"]).mkdir(parents=True, exist_ok=True)
    
    src_path = Path(__file__).resolve().parent / "src" / "preprocess_data"
    
    total_start = time.time()
    times = {}
    
    try:
        # ============================================================
        # STEP 1: Data quality y filtrado
        # ============================================================
        step1_path = src_path / "1. data_quality.py"
        cmd1 = [
            sys.executable,
            str(step1_path),
            "--train_csv", config["train_csv"],
            "--items_csv", config["items_csv"],
            "--stores_csv", config["stores_csv"],
            "--out_dir", config["out_dir"],
            "--exp_tag", config["exp_tag"],
            "--log_level", "INFO"
        ]
        
        times["step1"] = run_step(1, step1_path, cmd1, logger)
        
        # Verificar outputs de Step 1
        train_filtered = Path(config["out_dir"]) / "train_filtered.csv"
        if not train_filtered.exists():
            raise FileNotFoundError(f"Step 1 no generó: {train_filtered}")
        logger.info("✅ Step 1 output: %s (%.1f MB)", 
                   train_filtered, train_filtered.stat().st_size / (1024*1024))
        
        # ============================================================
        # STEP 2: Calcular competitive exposure
        # ============================================================
        step2_path = src_path / "2. competitive_exposure.py"
        cmd2 = [
            sys.executable,
            str(step2_path),
            "--train_filtered", str(train_filtered),
            "--items_csv", config["items_csv"],
            "--stores_csv", config["stores_csv"],
            "--holidays_csv", config["holidays_csv"],
            "--out_dir", config["out_dir"],
            "--exp_tag", config["exp_tag"],
            "--h_bin_threshold", str(config["h_bin_threshold"]),
            "--log_level", "INFO"
        ]
        
        times["step2"] = run_step(2, step2_path, cmd2, logger)
        
        # Verificar outputs de Step 2
        comp_exposure = Path(config["out_dir"]) / "competitive_exposure.csv"
        if not comp_exposure.exists():
            raise FileNotFoundError(f"Step 2 no generó: {comp_exposure}")
        logger.info("✅ Step 2 output: %s (%.1f MB)", 
                   comp_exposure, comp_exposure.stat().st_size / (1024*1024))
        
        # ============================================================
        # RESUMEN FINAL
        # ============================================================
        total_time = time.time() - total_start
        
        logger.info("=" * 60)
        logger.info("✅ STEPS 1 Y 2 COMPLETADOS")
        logger.info("=" * 60)
        logger.info("Tiempos de ejecución:")
        logger.info("  - Step 1: %.2f segundos", times["step1"])
        logger.info("  - Step 2: %.2f segundos", times["step2"])
        logger.info("  - Total: %.2f segundos (%.2f minutos)", total_time, total_time/60)
        logger.info("")
        logger.info("Archivos generados en: %s", config["out_dir"])
        logger.info("  - train_filtered.csv")
        logger.info("  - competitive_exposure.csv")
        logger.info("")
        logger.info("Siguiente paso: python run_step3_A_base.py")
        
        # Guardar reporte
        report = {
            "experiment": exp_tag,
            "steps": [1, 2],
            "execution_times": times,
            "total_time_seconds": total_time,
            "outputs": {
                "train_filtered": str(train_filtered),
                "competitive_exposure": str(comp_exposure)
            }
        }
        
        report_path = Path(config["out_dir"]) / "steps1_2_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Reporte guardado: %s", report_path)
        
    except Exception as e:
        logger.error("❌ Error en Steps 1-2: %s", e)
        logger.exception("Traceback completo:")
        raise

if __name__ == "__main__":
    main()
