#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para re-ejecutar solo Step 5 (GSC) para A_base
"""

import yaml
from pathlib import Path
import sys

def main():
    print("=" * 80)
    print("RE-EJECUTAR STEP 5 (GSC) - A_base")
    print("=" * 80)
    print()
    
    # Cargar configuración de A_base
    config_path = Path(".data/_sweeps/cfg_A_base.yaml")
    if not config_path.exists():
        print(f"❌ No existe: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Crear configuración temporal solo para Step 5
    temp_config = config.copy()
    
    # DESHABILITAR todos los pasos excepto Step 5
    temp_config["toggles"]["step1"] = False
    temp_config["toggles"]["step2"] = False
    temp_config["toggles"]["step3"] = False
    temp_config["toggles"]["step4"] = False
    temp_config["toggles"]["step5_gsc"] = True  # ← SOLO ESTE
    temp_config["toggles"]["step6_meta"] = False
    temp_config["toggles"]["eda1"] = False
    temp_config["toggles"]["eda2"] = False
    temp_config["toggles"]["eda3"] = False
    temp_config["toggles"]["eda4"] = False
    temp_config["toggles"]["eda_algorithms"] = False
    
    # NO limpiar outputs
    temp_config["clean_outputs"] = False
    
    # Guardar configuración temporal
    temp_config_path = Path(".data/_sweeps/rerun_step5_gsc_A_base.yaml")
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(temp_config, f, sort_keys=False, allow_unicode=True)
    
    # Verificar que Step 4 se ejecutó
    gsc_dir = Path('.data/processed_data/A_base/gsc')
    if not gsc_dir.exists():
        print("❌ Error: No existe .data/processed_data/A_base/gsc/")
        print("   Primero ejecuta: python rerun_step4_A_base.py")
        sys.exit(1)
    
    episode_files = [f for f in gsc_dir.glob('*.parquet') 
                    if f.name not in ['donor_quality.parquet', 'gsc_metrics.parquet']]
    
    print("📋 Configuración:")
    print(f"   • Step 5 (GSC): HABILITADO")
    print(f"   • Todos los demás: DESHABILITADOS")
    print(f"   • Episodios encontrados: {len(episode_files)}")
    print()
    
    if len(episode_files) < 150:
        print(f"⚠️  Solo hay {len(episode_files)} episodios (esperados: 150)")
        print("   ¿Continuar de todos modos?")
    
    print("📁 Outputs esperados:")
    print(f"   • .data/processed_data/A_base/gsc/gsc_metrics.parquet")
    print(f"   • .data/processed_data/A_base/gsc/cf_series/*.parquet ({len(episode_files)} archivos)")
    print(f"   • .data/processed_data/A_base/gsc/causal_metrics/*.parquet")
    print()
    
    print("⏱️  Tiempo estimado: ~2-4 horas (para 150 episodios)")
    print()
    
    response = input("¿Continuar? (s/N): ").strip().lower()
    if response not in ["s", "si", "sí", "yes", "y"]:
        print("❌ Cancelado por el usuario")
        sys.exit(0)
    
    print()
    print("🚀 Ejecutando Step 5 (GSC)...")
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
        print("✅ STEP 5 (GSC) COMPLETADO")
        print("=" * 80)
        print()
        
        # Verificar outputs
        import pandas as pd
        
        gsc_metrics = gsc_dir / "gsc_metrics.parquet"
        if gsc_metrics.exists():
            df = pd.read_parquet(gsc_metrics)
            print(f"📊 Episodios procesados por GSC: {len(df)}")
            print(f"   ATT_sum promedio: {df['att_sum'].mean():.2f}")
            print(f"   RMSPE_pre promedio: {df['rmspe_pre'].mean():.4f}")
        else:
            print("   ❌ No se generó gsc_metrics.parquet")
        
        cf_series_dir = gsc_dir / "cf_series"
        if cf_series_dir.exists():
            cf_files = list(cf_series_dir.glob("*.parquet"))
            print(f"📁 Series CF generadas: {len(cf_files)}")
        
        print()
        print("📝 Próximo paso (opcional):")
        print("   python rerun_step6_meta_A_base.py")
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR AL EJECUTAR STEP 5 (GSC)")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
