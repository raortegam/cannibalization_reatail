#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_final_paper_params.py
==============================
Valida que todos los parámetros del experimento FINAL_PAPER están correctamente
definidos y serán utilizados por el pipeline.
"""

import sys
from pathlib import Path
import yaml
from typing import Dict, Any, Set

# Cargar configuración
ROOT = Path(__file__).resolve().parent
EXPERIMENTS_YAML = ROOT / "experiments.yaml"
PIPELINE_CONFIG_YAML = ROOT / "pipeline_config.yaml"

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_final_paper_config() -> Dict[str, Any]:
    """Extrae la configuración del experimento FINAL_PAPER."""
    suite = load_yaml(EXPERIMENTS_YAML)
    base_cfg = load_yaml(ROOT / suite["base_config"])
    
    # Buscar FINAL_PAPER
    final_paper = None
    for exp in suite["experiments"]:
        if exp["id"] == "FINAL_PAPER":
            final_paper = exp
            break
    
    if not final_paper:
        raise ValueError("No se encontró experimento FINAL_PAPER en experiments.yaml")
    
    # Merge base + overrides
    overrides = final_paper.get("overrides", {})
    
    # Deep merge de params
    merged_params = base_cfg.get("params", {}).copy()
    if "params" in overrides:
        merged_params.update(overrides["params"])
    
    return {
        "toggles": overrides.get("toggles", base_cfg.get("toggles", {})),
        "params": merged_params,
        "paths": base_cfg.get("paths", {})
    }

def get_pipeline_params_definition() -> Set[str]:
    """Extrae todos los parámetros definidos en ParamsConfig del pipeline."""
    # Estos son los parámetros definidos en la clase ParamsConfig
    return {
        # Step 1
        "min_date", "date_col", "store_col", "item_col", "promo_col", 
        "sales_col", "tx_col", "save_report_json", "h_bin_threshold",
        
        # Step 2
        "neighborhood_col", "save_format",
        
        # Step 3
        "outdir_pairs_donors",
        
        # Step 4
        "top_k_donors", "donor_kind", "lags_days", "fourier_k",
        "max_donor_promo_share", "min_availability_share", "save_intermediate",
        "use_stl", "drop_city", "dry_run", "max_episodes", "prep_log_level",
        "fail_fast",
        
        # EDA 2
        "eda2_orientation", "eda2_dpi", "eda2_bins", "eda2_top_stores",
        "eda2_min_store_obs", "eda2_heatmap_stores", "eda2_chunksize", "eda2_log_level",
        
        # EDA 3
        "eda3_n", "eda3_strategy", "eda3_seed", "eda3_orientation", "eda3_dpi",
        
        # EDA 4
        "eda4_promo_thresh", "eda4_avail_thresh", "eda4_limit_episodes",
        "eda4_example_episode_id", "eda4_orientation", "eda4_dpi", "eda4_log_level",
        
        # GSC
        "gsc_log_level", "gsc_max_episodes", "gsc_do_placebo_space", "gsc_do_placebo_time",
        "gsc_do_loo", "gsc_max_loo", "gsc_sens_samples", "gsc_cv_folds", "gsc_cv_holdout",
        "gsc_cv_gap", "gsc_train_gap", "gsc_eval_n", "gsc_eval_selection",
        "gsc_rank", "gsc_tau", "gsc_alpha", "gsc_hpo_trials", "gsc_include_covariates",
        "gsc_link", "gsc_link_eps", "gsc_pred_clip_min", "gsc_calibrate_victim",
        "gsc_post_smooth_window",
        
        # Meta-learners
        "meta_learners", "meta_model", "meta_prop_model", "meta_random_state",
        "meta_cv_folds", "meta_cv_holdout", "meta_min_train_samples", "meta_max_episodes",
        "meta_do_placebo_space", "meta_do_placebo_time", "meta_do_loo", "meta_max_loo",
        "meta_sens_samples", "meta_max_depth", "meta_learning_rate", "meta_max_iter",
        "meta_min_samples_leaf", "meta_l2", "meta_hpo_trials",
        
        # Tratamiento
        "treat_col_s", "s_ref", "treat_col_b", "bin_threshold",
        
        # EDA algoritmos
        "eda_alg_orientation", "eda_alg_dpi", "eda_alg_learners", "eda_alg_export_pdf",
        "eda_alg_max_episodes_gsc", "eda_alg_max_episodes_meta",
        
        # Filtrado de episodios
        "n_cannibals", "n_victims_per_cannibal", "pairs_cannibal_col", "pairs_victim_col"
    }

def validate_params():
    """Valida que todos los parámetros de FINAL_PAPER son reconocidos."""
    print("=" * 80)
    print("VALIDACIÓN DE PARÁMETROS - EXPERIMENTO FINAL_PAPER")
    print("=" * 80)
    print()
    
    # Cargar configuración
    try:
        final_cfg = get_final_paper_config()
    except Exception as e:
        print(f"❌ ERROR: No se pudo cargar configuración FINAL_PAPER: {e}")
        return False
    
    pipeline_params = get_pipeline_params_definition()
    final_params = final_cfg["params"]
    
    # Validar cada parámetro
    print("📋 PARÁMETROS DEFINIDOS EN FINAL_PAPER:")
    print("-" * 80)
    
    unknown_params = []
    valid_params = []
    
    for param_name, param_value in sorted(final_params.items()):
        if param_name in pipeline_params:
            status = "✅"
            valid_params.append(param_name)
        else:
            status = "⚠️  DESCONOCIDO"
            unknown_params.append(param_name)
        
        # Formatear valor
        if isinstance(param_value, list):
            value_str = f"[{', '.join(map(str, param_value))}]"
        elif isinstance(param_value, str):
            value_str = f'"{param_value}"'
        else:
            value_str = str(param_value)
        
        print(f"{status} {param_name:35s} = {value_str}")
    
    print()
    print("=" * 80)
    print("RESUMEN DE VALIDACIÓN")
    print("=" * 80)
    print(f"✅ Parámetros válidos: {len(valid_params)}")
    print(f"⚠️  Parámetros desconocidos: {len(unknown_params)}")
    
    if unknown_params:
        print()
        print("⚠️  ADVERTENCIA: Los siguientes parámetros NO están definidos en ParamsConfig:")
        for param in unknown_params:
            print(f"   - {param}")
        print()
        print("   Estos parámetros serán IGNORADOS por el pipeline.")
    
    print()
    print("=" * 80)
    print("PARÁMETROS CRÍTICOS PARA FINAL_PAPER")
    print("=" * 80)
    
    critical_params = {
        "n_cannibals": "Número de caníbales (debe ser 10)",
        "n_victims_per_cannibal": "Víctimas por caníbal (debe ser 5 para 50 total)",
        "top_k_donors": "Donantes por víctima (debe ser ~18)",
        "gsc_hpo_trials": "Trials de Optuna para GSC (debe ser 500)",
        "meta_hpo_trials": "Trials de Optuna para Meta (debe ser 200)",
        "meta_learners": "Meta-learners a usar (debe ser ['x', 's', 't'])",
        "gsc_max_episodes": "Episodios GSC (debe ser null para todos)",
        "meta_max_episodes": "Episodios Meta (debe ser null para todos)",
        "eda_alg_max_episodes_gsc": "Gráficas GSC (debe ser 50)",
        "eda_alg_max_episodes_meta": "Gráficas Meta (debe ser 50)",
    }
    
    all_critical_ok = True
    for param, description in critical_params.items():
        if param in final_params:
            value = final_params.get(param)
            # Formatear valor apropiadamente
            if isinstance(value, list):
                value_str = str(value)
            elif value is None:
                value_str = "null"
            else:
                value_str = str(value)
            print(f"✅ {param:30s} = {value_str:20s} # {description}")
        else:
            print(f"❌ {param:30s} = NO DEFINIDO       # {description}")
            all_critical_ok = False
    
    print()
    print("=" * 80)
    print("TOGGLES (PASOS HABILITADOS)")
    print("=" * 80)
    
    toggles = final_cfg["toggles"]
    for toggle_name, toggle_value in sorted(toggles.items()):
        status = "✅ ON " if toggle_value else "⚪ OFF"
        print(f"{status} {toggle_name}")
    
    print()
    print("=" * 80)
    print("VERIFICACIÓN DE VALORES ESPERADOS")
    print("=" * 80)
    
    checks = []
    
    # Verificar valores específicos
    if final_params.get("n_cannibals") == 10:
        checks.append(("✅", "n_cannibals = 10 (correcto)"))
    else:
        checks.append(("❌", f"n_cannibals = {final_params.get('n_cannibals')} (esperado: 10)"))
    
    if final_params.get("n_victims_per_cannibal") == 5:
        checks.append(("✅", "n_victims_per_cannibal = 5 (correcto, 50 total)"))
    else:
        checks.append(("❌", f"n_victims_per_cannibal = {final_params.get('n_victims_per_cannibal')} (esperado: 5)"))
    
    if 15 <= final_params.get("top_k_donors", 0) <= 20:
        checks.append(("✅", f"top_k_donors = {final_params.get('top_k_donors')} (en rango 15-20)"))
    else:
        checks.append(("⚠️ ", f"top_k_donors = {final_params.get('top_k_donors')} (recomendado: 15-20)"))
    
    if final_params.get("gsc_hpo_trials") == 500:
        checks.append(("✅", "gsc_hpo_trials = 500 (correcto)"))
    else:
        checks.append(("❌", f"gsc_hpo_trials = {final_params.get('gsc_hpo_trials')} (esperado: 500)"))
    
    if final_params.get("meta_hpo_trials") == 200:
        checks.append(("✅", "meta_hpo_trials = 200 (correcto)"))
    else:
        checks.append(("❌", f"meta_hpo_trials = {final_params.get('meta_hpo_trials')} (esperado: 200)"))
    
    meta_learners = final_params.get("meta_learners", [])
    if set(meta_learners) == {"x", "s", "t"}:
        checks.append(("✅", f"meta_learners = {meta_learners} (correcto: X, S, T)"))
    else:
        checks.append(("❌", f"meta_learners = {meta_learners} (esperado: ['x', 's', 't'])"))
    
    if final_params.get("gsc_max_episodes") is None:
        checks.append(("✅", "gsc_max_episodes = null (procesará todos los episodios)"))
    else:
        checks.append(("⚠️ ", f"gsc_max_episodes = {final_params.get('gsc_max_episodes')} (limitará episodios)"))
    
    if final_params.get("meta_max_episodes") is None:
        checks.append(("✅", "meta_max_episodes = null (procesará todos los episodios)"))
    else:
        checks.append(("⚠️ ", f"meta_max_episodes = {final_params.get('meta_max_episodes')} (limitará episodios)"))
    
    if final_params.get("eda_alg_max_episodes_gsc") == 50:
        checks.append(("✅", "eda_alg_max_episodes_gsc = 50 (graficará 50 víctimas)"))
    else:
        checks.append(("⚠️ ", f"eda_alg_max_episodes_gsc = {final_params.get('eda_alg_max_episodes_gsc')}"))
    
    if final_params.get("eda_alg_max_episodes_meta") == 50:
        checks.append(("✅", "eda_alg_max_episodes_meta = 50 (graficará 50 víctimas)"))
    else:
        checks.append(("⚠️ ", f"eda_alg_max_episodes_meta = {final_params.get('eda_alg_max_episodes_meta')}"))
    
    for status, message in checks:
        print(f"{status} {message}")
    
    print()
    print("=" * 80)
    
    # Resultado final
    errors = sum(1 for s, _ in checks if s == "❌")
    warnings = sum(1 for s, _ in checks if s == "⚠️ ")
    
    if errors == 0 and warnings == 0 and not unknown_params:
        print("✅ VALIDACIÓN EXITOSA - Todos los parámetros son correctos")
        print()
        print("Puedes ejecutar el experimento con:")
        print("  python 01_run_sweep.py --experiments experiments.yaml --only FINAL_PAPER")
        return True
    elif errors == 0:
        print(f"⚠️  VALIDACIÓN CON ADVERTENCIAS ({warnings} advertencias, {len(unknown_params)} params desconocidos)")
        print()
        print("El experimento puede ejecutarse, pero revisa las advertencias.")
        return True
    else:
        print(f"❌ VALIDACIÓN FALLIDA ({errors} errores, {warnings} advertencias)")
        print()
        print("Corrige los errores antes de ejecutar el experimento.")
        return False

if __name__ == "__main__":
    try:
        success = validate_params()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
