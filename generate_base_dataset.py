#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_base_dataset.py

Script para generar el dataset base (Steps 1-3) con configuración óptima.
Este dataset se reutilizará en todos los experimentos para ahorrar tiempo.

Uso:
    python generate_base_dataset.py

Genera:
    - .data/processed_data/train_filtered.csv (Step 1)
    - .data/processed_data/transactions_filtered.csv (Step 1)
    - .data/processed_data/competitive_exposure.csv (Step 2)
    - .data/processed_data/_shared_base/pairs_windows.csv (Step 3) ← CENTRAL
    - .data/processed_data/_shared_base/donors_per_victim.csv (Step 3) ← CENTRAL
    - .data/processed_data/_shared_base/episodes_index.parquet (Step 3) ← CENTRAL
"""

import yaml
from pathlib import Path
import shutil
import sys

def main():
    print("=" * 80)
    print("GENERACIÓN DE DATASET BASE (Steps 1-3)")
    print("=" * 80)
    print()
    
    # Cargar configuración base
    config_path = Path("pipeline_config.yaml")
    if not config_path.exists():
        print(f"❌ No existe: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Crear configuración temporal para generar dataset base
    temp_config = config.copy()
    
    # Forzar ejecución de Steps 1-3 (sin cache)
    temp_config["toggles"]["use_cached_step1"] = False
    temp_config["toggles"]["use_cached_step2"] = False
    temp_config["toggles"]["use_cached_step3"] = False
    
    # Deshabilitar Steps 4-6 y EDAs (solo queremos 1-3)
    temp_config["toggles"]["step4"] = False
    temp_config["toggles"]["step5_gsc"] = False
    temp_config["toggles"]["step6_meta"] = False
    temp_config["toggles"]["eda1"] = False
    temp_config["toggles"]["eda2"] = False
    temp_config["toggles"]["eda3"] = False
    temp_config["toggles"]["eda4"] = False
    temp_config["toggles"]["eda_algorithms"] = False
    
    # Configurar exp_tag
    temp_config["exp_tag"] = "A_base"

    # Asegurar 12 donantes por víctima en Step 3 (cuando existan candidatos suficientes)
    try:
        if "params" not in temp_config:
            temp_config["params"] = {}
        temp_config["params"]["top_k_donors"] = 12
    except Exception:
        pass
    
    # NO limpiar outputs (queremos conservarlos)
    temp_config["clean_outputs"] = False
    
    # Guardar configuración temporal
    temp_config_path = Path(".data/_sweeps/generate_base_config.yaml")
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(temp_config, f, sort_keys=False, allow_unicode=True)
    
    print("📋 Configuración:")
    print(f"   • Steps a ejecutar: 1, 2, 3")
    print(f"   • Cache: DESHABILITADO (regenera todo)")
    print(f"   • Exp tag: A_base")
    print()
    
    # Verificar configuración actual de Step 3
    print("⚙️  Configuración de Step 3 (select_pairs_and_donors.py):")
    print(f"   • N_CANNIBALS_META: 100 (hardcoded)")
    print(f"   • N_VICTIMS_PER_I_META: 50 (hardcoded)")
    print(f"   • MAX_EPISODES_FOR_DONORS: 150 (hardcoded)")
    print(f"   • EPISODE_SELECTION_STRATEGY: top_delta_abs")
    print()
    
    # Confirmar con usuario
    print("⚠️  ADVERTENCIA:")
    print("   Este proceso:")
    print("   1. Regenerará train_filtered.csv (~5-10 min)")
    print("   2. Regenerará competitive_exposure.csv (~10-15 min)")
    print("   3. Regenerará pairs/donors/episodes (~30-45 min)")
    print("   4. TOTAL: ~45-70 minutos")
    print()
    
    response = input("¿Continuar? (s/N): ").strip().lower()
    if response not in ["s", "si", "sí", "yes", "y"]:
        print("❌ Cancelado por el usuario")
        sys.exit(0)
    
    print()
    print("🚀 Ejecutando pipeline...")
    print("=" * 80)
    print()
    
    # Ejecutar pipeline
    import importlib.util
    
    pipeline_path = Path("00_run_pipeline.py")
    spec = importlib.util.spec_from_file_location("pipeline", str(pipeline_path))
    if spec is None or spec.loader is None:
        print(f"❌ No se pudo cargar {pipeline_path}")
        sys.exit(1)
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline"] = module
    spec.loader.exec_module(module)
    
    # Ejecutar
    try:
        manifest = module.run_pipeline(temp_config_path)
        
        print()
        print("=" * 80)
        print("✅ DATASET BASE GENERADO EXITOSAMENTE")
        print("=" * 80)
        print()
        
        # Verificar outputs
        exp_tag = temp_config.get("exp_tag", "")
        exposure_check_path = f".data/processed_data/{exp_tag}/competitive_exposure.csv" if exp_tag else ".data/processed_data/competitive_exposure.csv"
        outputs_to_check = [
            (".data/processed_data/train_filtered.csv", "Step 1: train_filtered.csv"),
            (exposure_check_path, "Step 2: competitive_exposure.csv"),
            (".data/processed_data/_shared_base/pairs_windows.csv", "Step 3: pairs_windows.csv (CENTRAL)"),
            (".data/processed_data/_shared_base/donors_per_victim.csv", "Step 3: donors_per_victim.csv (CENTRAL)"),
            (".data/processed_data/_shared_base/episodes_index.parquet", "Step 3: episodes_index.parquet (CENTRAL)"),
        ]
        
        print("📁 Outputs generados:")
        all_exist = True
        for path_str, label in outputs_to_check:
            path = Path(path_str)
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   ✓ {label}: {size_mb:.2f} MB")
            else:
                print(f"   ✗ {label}: NO EXISTE")
                all_exist = False
        
        print()
        
        if all_exist:
            print("🎉 Todos los outputs fueron generados correctamente")
            print()
            print("📝 Próximos pasos:")
            print("   1. Verificar número de episodios:")
            print("      python -c \"import pandas as pd; df = pd.read_parquet('.data/processed_data/_shared_base/episodes_index.parquet'); print(f'Episodios: {len(df)}')\"")
            print()
            print("   2. Ejecutar experimentos (usarán este dataset base CENTRAL):")
            print("      python 01_run_sweep.py --experiments experiments.yaml")
            print()
            print("   3. Los experimentos reutilizarán Steps 1-3 (ahorro ~45-70 min por experimento)")
            print("   4. Todos los experimentos leerán de .data/processed_data/_shared_base/")
        else:
            print("⚠️  Algunos outputs no fueron generados. Revisa los logs.")
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR AL GENERAR DATASET BASE")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
