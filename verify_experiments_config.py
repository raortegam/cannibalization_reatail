#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_experiments_config.py

Script para verificar que todos los parámetros de experiments.yaml
se aplicarán correctamente al ejecutar 01_run_sweep.py

Uso:
    python verify_experiments_config.py
"""

import yaml
from pathlib import Path
from pprint import pprint

def deep_update(d: dict, u: dict) -> dict:
    """Actualización recursiva de diccionarios (deep-merge) - igual que en 01_run_sweep.py"""
    import copy
    out = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, dict):
            out[k] = deep_update(out.get(k, {}), v)
        else:
            out[k] = v
    return out

def main():
    # Cargar archivos
    experiments_path = Path("experiments.yaml")
    base_config_path = Path("pipeline_config.yaml")
    
    if not experiments_path.exists():
        print(f"❌ No existe: {experiments_path}")
        return
    
    if not base_config_path.exists():
        print(f"❌ No existe: {base_config_path}")
        return
    
    with open(experiments_path, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f)
    
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    
    print("=" * 80)
    print("VERIFICACIÓN DE CONFIGURACIÓN DE EXPERIMENTOS")
    print("=" * 80)
    print()
    
    # Parámetros críticos que deben estar en todos los experimentos
    critical_params = [
        "meta_learners",
        "meta_hpo_trials",
        "meta_max_iter",
        "top_k_donors",
        "gsc_rank",
        "gsc_tau",
    ]
    
    experiments = suite["experiments"]
    
    for exp in experiments:
        exp_id = exp["id"]
        desc = exp.get("desc", "")
        overrides = exp.get("overrides", {})
        
        # Simular el merge que hace 01_run_sweep.py
        exp_cfg = deep_update(base_cfg, overrides)
        
        print(f"📋 Experimento: {exp_id}")
        print(f"   Descripción: {desc}")
        print(f"   Parámetros críticos:")
        
        params = exp_cfg.get("params", {})
        
        for param in critical_params:
            value = params.get(param, "❌ NO DEFINIDO")
            
            # Verificar si viene del override o del base
            is_override = param in overrides.get("params", {})
            source = "override" if is_override else "base_config"
            
            if value == "❌ NO DEFINIDO":
                print(f"      ⚠️  {param}: {value}")
            else:
                print(f"      ✅ {param}: {value} (desde {source})")
        
        # Verificar meta_learners específicamente
        meta_learners = params.get("meta_learners", [])
        if isinstance(meta_learners, list):
            n_learners = len(meta_learners)
            if n_learners == 3:
                print(f"      ✅ Correrá {n_learners} meta-learners: {meta_learners}")
            elif n_learners > 0:
                print(f"      ⚠️  Solo correrá {n_learners} meta-learner(s): {meta_learners}")
            else:
                print(f"      ❌ NO correrá meta-learners")
        
        print()
    
    print("=" * 80)
    print("RESUMEN")
    print("=" * 80)
    
    # Contar experimentos con configuración completa
    complete = 0
    incomplete = []
    
    for exp in experiments:
        exp_id = exp["id"]
        overrides = exp.get("overrides", {})
        exp_cfg = deep_update(base_cfg, overrides)
        params = exp_cfg.get("params", {})
        
        has_all = all(params.get(p) is not None for p in critical_params)
        has_3_learners = len(params.get("meta_learners", [])) == 3
        
        if has_all and has_3_learners:
            complete += 1
        else:
            incomplete.append(exp_id)
    
    print(f"✅ Experimentos con configuración completa: {complete}/{len(experiments)}")
    
    if incomplete:
        print(f"⚠️  Experimentos incompletos: {', '.join(incomplete)}")
    else:
        print("🎉 ¡Todos los experimentos tienen configuración completa!")
    
    print()
    print("=" * 80)
    print("VERIFICACIÓN DE PARÁMETROS ESPECÍFICOS")
    print("=" * 80)
    print()
    
    # Verificar que todos tengan HPO mejorado
    print("🔍 Verificando meta_hpo_trials:")
    for exp in experiments:
        exp_id = exp["id"]
        overrides = exp.get("overrides", {})
        exp_cfg = deep_update(base_cfg, overrides)
        params = exp_cfg.get("params", {})
        hpo_trials = params.get("meta_hpo_trials", "NO DEFINIDO")
        
        if hpo_trials == 100:
            print(f"   ✅ {exp_id}: {hpo_trials} trials")
        elif hpo_trials == "NO DEFINIDO":
            print(f"   ❌ {exp_id}: NO DEFINIDO (usará default de ParamsConfig)")
        else:
            print(f"   ⚠️  {exp_id}: {hpo_trials} trials (no es 100)")
    
    print()
    print("🔍 Verificando meta_learners:")
    for exp in experiments:
        exp_id = exp["id"]
        overrides = exp.get("overrides", {})
        exp_cfg = deep_update(base_cfg, overrides)
        params = exp_cfg.get("params", {})
        learners = params.get("meta_learners", [])
        
        if len(learners) == 3 and set(learners) == {"x", "s", "t"}:
            print(f"   ✅ {exp_id}: {learners}")
        elif len(learners) > 0:
            print(f"   ⚠️  {exp_id}: {learners} (no son los 3)")
        else:
            print(f"   ❌ {exp_id}: [] (no correrá meta-learners)")
    
    print()
    print("=" * 80)
    print("ESTIMACIÓN DE EPISODIOS")
    print("=" * 80)
    print()
    
    total_episodes = 0
    for exp in experiments:
        exp_id = exp["id"]
        overrides = exp.get("overrides", {})
        exp_cfg = deep_update(base_cfg, overrides)
        params = exp_cfg.get("params", {})
        
        n_learners = len(params.get("meta_learners", []))
        # GSC + n_learners
        algorithms = 1 + n_learners
        
        # Estimación conservadora: 100 episodios por algoritmo
        episodes_per_exp = algorithms * 100
        total_episodes += episodes_per_exp
        
        print(f"   {exp_id}: ~{episodes_per_exp} episodios (GSC + {n_learners} learners)")
    
    print()
    print(f"📊 Total estimado: ~{total_episodes} episodios en {len(experiments)} experimentos")
    print(f"📊 Promedio por experimento: ~{total_episodes // len(experiments)} episodios")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
