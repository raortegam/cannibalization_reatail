#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para re-ejecutar solo Step 4 (preprocesamiento) para A_base
"""

import yaml
from pathlib import Path
import sys

def main():
    print("=" * 80)
    print("RE-EJECUTAR STEP 4 (Preprocesamiento) - A_base")
    print("=" * 80)
    print()
    
    # Cargar configuración de A_base
    config_path = Path(".data/_sweeps/cfg_A_base.yaml")
    if not config_path.exists():
        print(f"❌ No existe: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Crear configuración temporal solo para Step 4
    temp_config = config.copy()
    
    # DESHABILITAR todos los pasos excepto Step 4
    temp_config["toggles"]["step1"] = False
    temp_config["toggles"]["step2"] = False
    temp_config["toggles"]["step3"] = False
    temp_config["toggles"]["step4"] = True  # ← SOLO ESTE
    temp_config["toggles"]["step5_gsc"] = False
    temp_config["toggles"]["step6_meta"] = False
    temp_config["toggles"]["eda1"] = False
    temp_config["toggles"]["eda2"] = False
    temp_config["toggles"]["eda3"] = False
    temp_config["toggles"]["eda4"] = False
    temp_config["toggles"]["eda_algorithms"] = False
    
    # Asegurar que gsc_eval_n está en null (procesar todos)
    temp_config["params"]["gsc_eval_n"] = None
    
    # NO limpiar outputs
    temp_config["clean_outputs"] = False
    
    # Guardar configuración temporal
    temp_config_path = Path(".data/_sweeps/rerun_step4_A_base.yaml")
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(temp_config, f, sort_keys=False, allow_unicode=True)
    
    print("📋 Configuración:")
    print(f"   • Step 4: HABILITADO")
    print(f"   • Todos los demás: DESHABILITADOS")
    print(f"   • gsc_eval_n: {temp_config['params']['gsc_eval_n']} (procesar TODOS)")
    print()
    
    print("📁 Outputs esperados:")
    print(f"   • .data/processed_data/A_base/gsc/*.parquet (150 episodios)")
    print(f"   • .data/processed_data/A_base/meta/all_units.parquet")
    print(f"   • .data/processed_data/A_base/episodes_index.parquet")
    print()
    
    print("⏱️  Tiempo estimado: ~30-60 minutos")
    print()
    
    response = input("¿Continuar? (s/N): ").strip().lower()
    if response not in ["s", "si", "sí", "yes", "y"]:
        print("❌ Cancelado por el usuario")
        sys.exit(0)
    
    print()
    print("🚀 Ejecutando Step 4...")
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
        print("✅ STEP 4 COMPLETADO")
        print("=" * 80)
        print()
        
        # Verificar outputs
        gsc_dir = Path('.data/processed_data/A_base/gsc')
        if gsc_dir.exists():
            episode_files = [f for f in gsc_dir.glob('*.parquet') 
                           if f.name not in ['donor_quality.parquet', 'gsc_metrics.parquet']]
            print(f"📁 Archivos de episodios generados: {len(episode_files)}")
            
            if len(episode_files) >= 150:
                print("   ✅ ¡Todos los episodios generados!")
            else:
                print(f"   ⚠️  Solo {len(episode_files)} de 150 esperados")
        else:
            print("   ❌ No se encontró el directorio gsc/")
        
        print()
        print("📝 Próximo paso:")
        print("   python rerun_step5_gsc_A_base.py")
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR AL EJECUTAR STEP 4")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
