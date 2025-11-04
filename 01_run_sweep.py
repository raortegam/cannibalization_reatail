#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_run_sweep.py
Ejecuta múltiples iteraciones del pipeline a partir de 'experiments.yaml'.

Uso (Windows):
    py -3 01_run_sweep.py --experiments .\experiments.yaml --only A_base
o
    python 01_run_sweep.py --experiments .\experiments.yaml --only A_base
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import yaml
import importlib.util

ROOT = Path(__file__).resolve().parent
TMP_DIR = ROOT / ".data" / "_sweeps"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def _load_run_pipeline_from_file() -> callable:
    """
    Carga '00_run_pipeline.py' por ruta, aunque el nombre del archivo empiece por dígitos.
    REGISTRA el módulo en sys.modules ANTES de ejecutarlo, para que dataclasses
    y otras introspecciones encuentren su namespace.
    """
    pipeline_path = ROOT / "00_run_pipeline.py"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"No encuentro {pipeline_path}. ¿Está en la raíz del repo?")

    # Nombre único para evitar colisiones de caché si se reimporta en el mismo proceso
    mod_name = f"pipeline_runner_{int(time.time() * 1000)}"

    spec = importlib.util.spec_from_file_location(mod_name, str(pipeline_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"No pude crear spec para {pipeline_path}")

    module = importlib.util.module_from_spec(spec)

    # **CLAVE**: registrar el módulo en sys.modules ANTES de exec_module
    sys.modules[mod_name] = module

    # Ejecutar el código del archivo dentro del objeto módulo
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    if not hasattr(module, "run_pipeline"):
        raise AttributeError("El módulo 00_run_pipeline.py no expone 'run_pipeline'.")
    return module.run_pipeline


# Import dinámico y seguro (evita 'from 00_run_pipeline import ...')
run_pipeline = _load_run_pipeline_from_file()


def deep_update(d: dict, u: dict) -> dict:
    """Actualización recursiva de diccionarios (deep-merge)."""
    out = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, dict):
            out[k] = deep_update(out.get(k, {}), v)
        else:
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", type=str, required=True,
                   help="Ruta al YAML con la suite de experimentos (e.g., experiments.yaml)")
    p.add_argument("--only", type=str, default=None,
                   help="Lista separada por coma de exp_id a correr (e.g., A_base,D_seasonal_rich)")
    args = p.parse_args()

    experiments_path = Path(args.experiments).resolve()
    if not experiments_path.exists():
        raise FileNotFoundError(f"No existe: {experiments_path}")

    with open(experiments_path, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f)

    base_cfg_path = Path(suite["base_config"]).resolve()
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"No existe base_config: {base_cfg_path}")

    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    if args.only:
        allow = set([s.strip() for s in args.only.split(",") if s.strip()])
        experiments = [e for e in suite["experiments"] if e["id"] in allow]
        if not experiments:
            raise ValueError(f"--only no coincide con ningún id en {experiments_path}")
    else:
        experiments = suite["experiments"]

    manifests = {}
    for exp in experiments:
        exp_id = exp["id"]
        overrides = exp.get("overrides", {})
        exp_cfg = deep_update(base_cfg, overrides)

        # Etiqueta de iteración (para rutas y sellado de EDA)
        exp_cfg["exp_tag"] = exp_id

        # YAML temporal por experimento
        tmp_yaml = TMP_DIR / f"cfg_{exp_id}.yaml"
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(exp_cfg, f, sort_keys=False, allow_unicode=True)

        print(f"▶️ Ejecutando {exp_id} con {tmp_yaml}")
        manifest = run_pipeline(tmp_yaml)
        manifests[exp_id] = manifest

    # Manifest agregador del sweep
    out_manifest = TMP_DIR / "sweep_manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifests, f, ensure_ascii=False, indent=2)
    print(f"📄 Sweep manifest: {out_manifest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Mensaje claro en caso de errores comunes (import, rutas, YAML).
        print(f"❌ Error en 01_run_sweep.py: {e}", file=sys.stderr)
        sys.exit(1)