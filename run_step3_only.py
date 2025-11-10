#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_step3_only.py

Script para ejecutar SOLO el Step 3 (select_pairs_and_donors).
Útil cuando Steps 1 y 2 ya están completos pero Step 3 falló.

Uso:
    python run_step3_only.py
"""

import yaml
from pathlib import Path
import sys

def main():
    print("=" * 80)
    print("EJECUTAR SOLO STEP 3 (Select Pairs & Donors)")
    print("=" * 80)
    print()
    
    # Cargar configuración base
    config_path = Path("pipeline_config.yaml")
    if not config_path.exists():
        print(f"❌ No existe: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Crear configuración temporal
    temp_config = config.copy()
    
    # DESHABILITAR Steps 1 y 2 (ya están completos)
    temp_config["toggles"]["step1"] = False
    temp_config["toggles"]["step2"] = False
    
    # HABILITAR Step 3 SIN cache (forzar regeneración)
    temp_config["toggles"]["step3"] = True
    temp_config["toggles"]["use_cached_step3"] = False
    
    # Deshabilitar Steps 4-6 y EDAs
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
    
    # NO limpiar outputs
    temp_config["clean_outputs"] = False
    
    # Guardar configuración temporal
    temp_config_path = Path(".data/_sweeps/run_step3_only_config.yaml")
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(temp_config, f, sort_keys=False, allow_unicode=True)
    
    print("📋 Configuración:")
    print(f"   • Step 1: DESHABILITADO (ya existe)")
    print(f"   • Step 2: DESHABILITADO (ya existe)")
    print(f"   • Step 3: HABILITADO (regenerar)")
    print(f"   • Cache Step 3: DESHABILITADO")
    print()
    
    print("📁 Outputs esperados:")
    print(f"   • .data/processed_data/_shared_base/pairs_windows.csv")
    print(f"   • .data/processed_data/_shared_base/donors_per_victim.csv")
    print(f"   • .data/processed_data/_shared_base/episodes_index.parquet")
    print()
    
    print("⏱️  Tiempo estimado: ~30-45 minutos")
    print()
    
    response = input("¿Continuar? (s/N): ").strip().lower()
    if response not in ["s", "si", "sí", "yes", "y"]:
        print("❌ Cancelado por el usuario")
        sys.exit(0)
    
    print()
    print("🚀 Ejecutando Step 3...")
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
        print("✅ STEP 3 COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print()
        
        # Verificar outputs
        outputs_to_check = [
            (".data/processed_data/_shared_base/pairs_windows.csv", "pairs_windows.csv"),
            (".data/processed_data/_shared_base/donors_per_victim.csv", "donors_per_victim.csv"),
            (".data/processed_data/_shared_base/episodes_index.parquet", "episodes_index.parquet"),
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
            print("🎉 Dataset base completado!")
            print()
            print("📝 Próximos pasos:")
            print("   1. Verificar número de episodios:")
            print("      python -c \"import pandas as pd; df = pd.read_parquet('.data/processed_data/_shared_base/episodes_index.parquet'); print(f'Episodios: {len(df)}')\"")
            print()
            print("   2. Ejecutar experimentos:")
            print("      python 01_run_sweep.py --experiments experiments.yaml")
        else:
            print("⚠️  Algunos outputs no fueron generados. Revisa los logs.")
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR AL EJECUTAR STEP 3")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
