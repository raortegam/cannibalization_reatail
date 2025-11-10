#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_run_pipeline.py
Orquestador del pipeline de preprocesamiento, modelado y EDA para canibalización promocional en retail.

Novedades (parches):
- Soporte de barridos/iteraciones vía 'exp_tag' (inyectado desde YAML), con rutas derivadas por iteración.
- Sellado de imágenes PNG/JPG de TODOS los EDA con 'exp_tag' y un resumen corto de configuración.
- Filtrado post Step 3 a 10 caníbales x 10 víctimas (parametrizable en ParamsConfig).
- Registro de README por iteración con la configuración efectiva.

Uso:
    python 00_run_pipeline.py --config pipeline_config.yaml
"""
# 00_run_pipeline.py
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from importlib import util as importlib_util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd  # <- usado en validaciones rápidas
try:
    import yaml  # PyYAML
except Exception as e:
    raise RuntimeError("Se requiere PyYAML. Instálalo con `pip install pyyaml`.") from e

# Opcional: para sellar imágenes de EDA con exp_tag
try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


# -------------------- Constantes / Paths base --------------------

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
EDA_DIR = REPO_ROOT / "EDA"
DIAG_DIR = REPO_ROOT / "diagnostics"
RMSPE_EPS = 1e-6  # umbral para detectar "calce perfecto" sospechoso


# -------------------- Utilidades generales --------------------

def _parquet_has_rows(p: Path) -> bool:
    try:
        if not p.exists():
            return False
        df = pd.read_parquet(p)
        return (df is not None) and (not df.empty)
    except Exception:
        return False


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(level: str = "INFO", log_to_file: bool = True, exp_tag: Optional[str] = None) -> Path:
    """
    Configura logging. Si 'exp_tag' viene definido, nombra el archivo con dicha etiqueta.
    """
    ensure_dir(DIAG_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"pipeline_{exp_tag}_{ts}.log" if exp_tag else f"pipeline_{ts}.log"
    log_path = DIAG_DIR / log_name

    # limpiar handlers previos
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    logging.info("Logging configurado. Nivel: %s", level.upper())
    if log_to_file:
        logging.info("Archivo de log: %s", str(log_path))
    return log_path


def seed_everything(seed: int = 2025) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def _maybe_copy(src: Path, dst: Path, label: str) -> None:
    """Si src existe y dst no, copia y deja traza."""
    try:
        if dst.exists():
            return
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            logging.info("[mirror] %s: %s -> %s", label, str(src), str(dst))
    except Exception as e:
        logging.warning("[mirror] No se pudo copiar %s -> %s: %s", str(src), str(dst), e)

def _log_toggles(toggles: 'StepToggles') -> None:
    try:
        d = toggles.__dict__
        on = [k for k, v in d.items() if v]
        off = [k for k, v in d.items() if not v]
        logging.info("Toggles ON: %s", ", ".join(sorted(on)))
        logging.info("Toggles OFF: %s", ", ".join(sorted(off)))
    except Exception:
        pass


def import_module_spawn_safe(module_name: str, real_path: Path):
    """
    Garantiza que el módulo sea importable por nombre en procesos hijos (Windows spawn).
    Copia el módulo Y su directorio padre completo para resolver dependencias locales.
    """
    tmp_dir = REPO_ROOT / ".data" / "_temp_modules"
    ensure_dir(tmp_dir)

    # Copiar el archivo principal
    dest = tmp_dir / f"{module_name}.py"
    try:
        if (not dest.exists()) or (real_path.stat().st_mtime > dest.stat().st_mtime):
            shutil.copy2(str(real_path), str(dest))
    except Exception as e:
        raise ImportError(f"No pude preparar alias spawn-safe para {module_name}: {e}")

    # Copiar todo el directorio padre (para dependencias locales)
    parent_dir = real_path.parent
    tmp_parent = tmp_dir / parent_dir.name
    if parent_dir.exists() and parent_dir.is_dir():
        try:
            ensure_dir(tmp_parent)
            for py_file in parent_dir.glob("*.py"):
                dest_py = tmp_parent / py_file.name
                if (not dest_py.exists()) or (py_file.stat().st_mtime > dest_py.stat().st_mtime):
                    shutil.copy2(str(py_file), str(dest_py))
        except Exception as e:
            logging.warning(f"No pude copiar dependencias del directorio {parent_dir}: {e}")

    # Añadir ambas rutas al sys.path
    if str(tmp_dir) not in sys.path:
        sys.path.insert(0, str(tmp_dir))
    if str(tmp_parent) not in sys.path:
        sys.path.insert(0, str(tmp_parent))

    return importlib.import_module(module_name)


def _dynamic_import(module_name: str, py_path: Path):
    """
    Carga dinámica de un módulo desde ruta absoluta (soporta nombres con espacios o puntos).
    """
    if not py_path.exists():
        raise FileNotFoundError(f"No existe el archivo de módulo: {py_path}")
    spec = importlib_util.spec_from_loader(module_name, SourceFileLoader(module_name, str(py_path)))
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear el spec para {module_name} desde {py_path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _timeit(logger: logging.Logger, label: str):
    """
    Context manager simple para medir tiempos por etapa.
    """
    class _T:
        def __enter__(self_):
            self_.t0 = time.time()
            logger.info("➡️  Iniciando: %s", label)
            return self_
        def __exit__(self_, exc_type, exc, tb):
            dt = time.time() - self_.t0
            if exc is None:
                logger.info("✅ Finalizado: %s (%.2f s)", label, dt)
            else:
                logger.error("❌ Error en: %s (%.2f s) | %s", label, dt, exc)
            return False
    return _T()


def _safe_call(fn, logger: logging.Logger, fail_fast: bool, step_name: str, **kwargs):
    """
    Ejecuta una función con manejo de excepciones y logging.
    """
    try:
        with _timeit(logger, step_name):
            return fn(**kwargs)
    except Exception as e:
        logger.exception("Fallo en '%s'.", step_name)
        if fail_fast:
            raise
        logger.warning("Continuando a la siguiente etapa (fail_fast=False).")
        return None


# -------------------- NUEVO: helpers de aislamiento por experimento --------------------

def _touch_fingerprint(dirpath: Path, exp_tag: Optional[str]) -> None:
    """Crea un 'fingerprint' en cada carpeta crítica para evidenciar el experimento activo."""
    if not exp_tag:
        return
    ensure_dir(dirpath)
    stamp = dirpath / f".exp_{exp_tag}.stamp"
    try:
        stamp.write_text(
            json.dumps({"exp_tag": exp_tag, "ts": datetime.now().isoformat(), "dir": str(dirpath)}, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception:
        pass


def _assert_scoped(path: Path, exp_tag: Optional[str], label: str, hard: bool = False) -> None:
    """Advierte o lanza si 'path' NO está bajo una carpeta cuyo nombre termina en exp_tag."""
    if not exp_tag:
        return
    p = Path(path).resolve()
    ok = any(part == exp_tag for part in p.parts)
    if not ok:
        msg = f"[SCOPE] {label} NO está bajo exp_tag='{exp_tag}' -> {str(p)}"
        if hard:
            raise RuntimeError(msg)
        logging.warning(msg)


def _assert_exists(path: Path, label: str) -> None:
    if not Path(path).exists():
        logging.warning("[MISSING] %s no existe -> %s", label, str(path))


def _purge_dir_if_needed(p: Path) -> None:
    """Borra y recrea un directorio (corrida limpia)."""
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


# -------------------- Configuración --------------------

@dataclass
class StepToggles:
    step1: bool = True
    step2: bool = True
    step3: bool = True
    step4: bool = True
    eda1: bool = True
    eda2: bool = True
    eda3: bool = True
    eda4: bool = True
    # Nuevos pasos
    step5_gsc: bool = True
    step6_meta: bool = True
    eda_algorithms: bool = True
    # Cache control
    use_cached_step1: bool = True  # Reutilizar outputs de Step 1 si existen
    use_cached_step2: bool = True  # Reutilizar outputs de Step 2 si existen
    use_cached_step3: bool = True  # Reutilizar outputs de Step 3 si existen


@dataclass
class PathsConfig:
    # Entradas
    raw_dir: Path
    train_csv: Path
    transactions_csv: Optional[Path] = None
    items_csv: Optional[Path] = None
    stores_csv: Optional[Path] = None

    # Salidas/Intermedios
    processed_dir: Path = Path(".data/processed_data")
    preprocessed_dir: Path = Path(".data/processed_data")
    figures_dir: Path = Path("figures/favorita_analysis")
    exposure_csv: Path = Path(".data/processed_data/competitive_exposure.csv")

    # Artefactos de Step 1 (si se desea forzar rutas)
    step1_out_dir: Optional[Path] = None

    # NUEVO: rutas de salida de algoritmos
    gsc_out_dir: Path = Path(".data/processed_data")
    meta_out_root: Path = Path(".data/processed_data")


@dataclass
class ParamsConfig:
    # Step 1: Data Quality
    min_date: str = "2016-01-01"
    date_col: str = "date"
    store_col: str = "store_nbr"
    item_col: str = "item_nbr"
    promo_col: str = "onpromotion"
    sales_col: Optional[str] = "unit_sales"
    tx_col: str = "transactions"
    save_report_json: Optional[Path] = None
    h_bin_threshold: float = 0.02

    # Step 2: Competitive Exposure
    neighborhood_col: Optional[str] = None
    save_format: Optional[str] = None

    # Step 3: select_pairs_and_donors
    outdir_pairs_donors: Path = Path(".data/processed_data")

    # Step 4: pre_algorithm -> PrepConfig
    top_k_donors: int = 10
    donor_kind: str = "same_item"
    lags_days: Tuple[int, ...] = (7, 14, 28, 56)
    fourier_k: int = 3
    max_donor_promo_share: float = 0.02
    min_availability_share: float = 0.90
    save_intermediate: bool = True
    use_stl: bool = True
    drop_city: bool = True
    dry_run: bool = False
    max_episodes: Optional[int] = None
    prep_log_level: str = "INFO"
    fail_fast: bool = False

    # EDA 2
    eda2_orientation: str = "portrait"
    eda2_dpi: int = 300
    eda2_bins: int = 50
    eda2_top_stores: int = 30
    eda2_min_store_obs: int = 5
    eda2_heatmap_stores: Optional[int] = None
    eda2_chunksize: int = 1_000_000
    eda2_log_level: str = "INFO"

    # EDA 3
    eda3_n: int = 5
    eda3_strategy: str = "stratified"
    eda3_seed: int = 2025
    eda3_orientation: str = "portrait"
    eda3_dpi: int = 300

    # EDA 4
    eda4_promo_thresh: float = 0.02
    eda4_avail_thresh: float = 0.90
    eda4_limit_episodes: int = 1
    eda4_example_episode_id: Optional[str] = None
    eda4_orientation: str = "portrait"
    eda4_dpi: int = 300
    eda4_log_level: str = "INFO"

    # -------- NUEVO: Algoritmos ----------
    # GSC
    gsc_log_level: str = "INFO"
    gsc_max_episodes: Optional[int] = None
    gsc_do_placebo_space: bool = True
    gsc_do_placebo_time: bool = True
    gsc_do_loo: bool = False
    gsc_max_loo: int = 5
    gsc_sens_samples: int = 0
    gsc_cv_folds: int = 3
    gsc_cv_holdout: int = 21
    gsc_eval_n: int = 10
    gsc_eval_selection: str = "head"
    # Hiperparámetros ejemplo
    gsc_rank: int = 5
    gsc_tau: float = 0.0001
    gsc_alpha: float = 0.0

    # Meta-learners
    meta_learners: Tuple[str, ...] = ("x",)  # subset de {"t","s","x"}
    meta_model: str = "hgbt"
    meta_prop_model: str = "logit"
    meta_random_state: int = 42
    meta_cv_folds: int = 3
    meta_cv_holdout: int = 21
    meta_min_train_samples: int = 50
    meta_max_episodes: Optional[int] = None
    meta_do_placebo_space: bool = True
    meta_do_placebo_time: bool = True
    meta_do_loo: bool = False
    meta_max_loo: int = 5
    meta_sens_samples: int = 0
    meta_max_depth: int = 6
    meta_learning_rate: float = 0.05
    meta_max_iter: int = 500
    meta_min_samples_leaf: int = 20
    meta_l2: float = 0.0
    meta_hpo_trials: int = 100  # Número de trials de Optuna para HPO
    # Tratamientos
    treat_col_s: str = "H_disc"
    s_ref: float = 0.0
    treat_col_b: str = "H_prop"
    bin_threshold: float = 0.0

    # EDA final de algoritmos
    eda_alg_orientation: str = "landscape"
    eda_alg_dpi: int = 300
    eda_alg_learners: Tuple[str, ...] = ("t", "s", "x")
    eda_alg_export_pdf: bool = True
    eda_alg_max_episodes_gsc: Optional[int] = None
    eda_alg_max_episodes_meta: Optional[int] = None

    # --- NUEVO: objetivo 10x10 episodios y nombres de columnas en pairs_windows
    n_cannibals: int = 10
    n_victims_per_cannibal: int = 10
    pairs_cannibal_col: Optional[str] = None
    pairs_victim_col: Optional[str] = None


@dataclass
class OrchestratorConfig:
    project_name: str
    seed: int = 2025
    log_level: str = "INFO"
    log_to_file: bool = True
    fail_fast: bool = False
    use_filtered_from_step1: bool = True
    exp_tag: Optional[str] = None
    clean_outputs: bool = True
    hard_scope: bool = True
    toggles: StepToggles = field(default_factory=StepToggles)
    paths: PathsConfig = field(default=None)   # type: ignore[assignment]
    params: ParamsConfig = field(default_factory=ParamsConfig)


def load_yaml_config(path: Path) -> OrchestratorConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    toggles = StepToggles(**cfg.get("toggles", {}))

    paths_raw = cfg["paths"]
    paths = PathsConfig(
        raw_dir=Path(paths_raw["raw_dir"]),
        train_csv=Path(paths_raw["train_csv"]),
        transactions_csv=Path(paths_raw["transactions_csv"]) if paths_raw.get("transactions_csv") else None,
        items_csv=Path(paths_raw["items_csv"]) if paths_raw.get("items_csv") else None,
        stores_csv=Path(paths_raw["stores_csv"]) if paths_raw.get("stores_csv") else None,
        processed_dir=Path(paths_raw.get("processed_dir", ".data/processed_data")),
        preprocessed_dir=Path(paths_raw.get("preprocessed_dir", ".data/processed_data")),
        figures_dir=Path(paths_raw.get("figures_dir", "figures/favorita_analysis")),
        exposure_csv=Path(paths_raw.get("exposure_csv", ".data/processed_data/competitive_exposure.csv")),
        step1_out_dir=Path(paths_raw["step1_out_dir"]) if paths_raw.get("step1_out_dir") else None,
        gsc_out_dir=Path(paths_raw.get("gsc_out_dir", ".data/processed_data")),
        meta_out_root=Path(paths_raw.get("meta_out_root", ".data/processed_data/meta_outputs")),
    )

    params = ParamsConfig(**cfg.get("params", {}))

    orch = OrchestratorConfig(
        project_name=cfg.get("project_name", "cannibalization_retail"),
        seed=cfg.get("seed", 2025),
        log_level=cfg.get("log_level", "INFO"),
        log_to_file=cfg.get("log_to_file", True),
        fail_fast=cfg.get("fail_fast", False),
        use_filtered_from_step1=cfg.get("use_filtered_from_step1", True),
        exp_tag=cfg.get("exp_tag"),
        clean_outputs=cfg.get("clean_outputs", True),
        hard_scope=cfg.get("hard_scope", True),
        toggles=toggles,
        paths=paths,
        params=params,
    )
    return orch


# -------------------- Helpers para iteraciones/sello/README --------------------

def derive_iter_paths(paths: 'PathsConfig', exp_tag: Optional[str]) -> 'PathsConfig':
    """
    Si se provee exp_tag, deriva subcarpetas por iteración para:
      processed_dir, preprocessed_dir, figures_dir, gsc_out_dir, meta_out_root y exposure_csv.
    """
    if not exp_tag:
        return paths
    suffix = Path(exp_tag)
    processed_dir = (paths.processed_dir / suffix)
    preprocessed_dir = (paths.preprocessed_dir / suffix)
    figures_dir = (paths.figures_dir / suffix)
    gsc_out_dir = (paths.gsc_out_dir / suffix)
    meta_out_root = (paths.meta_out_root / suffix)
    exposure_csv = processed_dir / "competitive_exposure.csv"
    return PathsConfig(
        raw_dir=paths.raw_dir,
        train_csv=paths.train_csv,
        transactions_csv=paths.transactions_csv,
        items_csv=paths.items_csv,
        stores_csv=paths.stores_csv,
        processed_dir=processed_dir,
        preprocessed_dir=preprocessed_dir,
        figures_dir=figures_dir,
        exposure_csv=exposure_csv,
        step1_out_dir=paths.step1_out_dir,
        gsc_out_dir=gsc_out_dir,
        meta_out_root=meta_out_root,
    )


def _short_cfg_summary(params: 'ParamsConfig') -> str:
    return (
        f"donors={params.top_k_donors}({params.donor_kind}) | "
        f"lags={list(params.lags_days)} | Fourier={params.fourier_k} | "
        f"STL={'on' if params.use_stl else 'off'} | "
        f"GSC(rank={params.gsc_rank},tau={params.gsc_tau},alpha={params.gsc_alpha}) | "
        f"Meta:{','.join(params.meta_learners)}->{params.meta_model} | "
        f"Hbin={getattr(params,'h_bin_threshold',None)}"
    )


def stamp_pngs(fig_dir: Path, exp_tag: str, summary: str) -> None:
    """
    Agrega una franja con exp_tag y un resumen de configuración a PNG/JPG en fig_dir (recursivo).
    Si Pillow no está disponible o falla, se omite silenciosamente.
    """
    if not _HAS_PIL:
        return
    if not fig_dir.exists():
        return
    png_exts = {".png", ".jpg", ".jpeg"}
    for p in fig_dir.rglob("*"):
        if p.suffix.lower() not in png_exts or not p.is_file():
            continue
        try:
            img = Image.open(p).convert("RGBA")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            text = f"{exp_tag}  |  {summary}"
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            w, h = img.size
            pad = max(8, int(w * 0.01))
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            rect_h = th + 2 * pad
            draw.rectangle([(0, h - rect_h), (w, h)], fill=(0, 0, 0, 140))
            draw.text((pad, h - rect_h + pad), text, fill=(255, 255, 255, 230), font=font)
            out = Image.alpha_composite(img, overlay).convert("RGB")
            out.save(p)
        except Exception:
            continue  # No bloquear el pipeline si falla el sellado


def _params_to_plain_dict(params: 'ParamsConfig') -> Dict[str, Any]:
    """
    Convierte ParamsConfig a dict serializable (paths->str, tuples->list).
    """
    from pathlib import Path as _Path
    plain: Dict[str, Any] = {}
    for k, v in params.__dict__.items():
        if isinstance(v, tuple):
            plain[k] = list(v)
        elif isinstance(v, _Path):
            plain[k] = str(v)
        else:
            plain[k] = v
    return plain


def write_iter_readme(fig_dir: Path, exp_tag: str, orch_cfg: 'OrchestratorConfig') -> None:
    """Guarda un README.md con la configuración efectiva de la iteración."""
    ensure_dir(fig_dir)
    readme = fig_dir / "00__experiment_meta.md"
    lines = [
        f"# Experimento {exp_tag}",
        "",
        f"- Proyecto: {orch_cfg.project_name}",
        f"- Procesados: {orch_cfg.paths.processed_dir}",
        f"- GSC out: {orch_cfg.paths.gsc_out_dir}",
        f"- Meta out: {orch_cfg.paths.meta_out_root}",
        "",
        "## Parámetros",
        "```yaml",
        yaml.safe_dump({"params": _params_to_plain_dict(orch_cfg.params)}, sort_keys=False, allow_unicode=True).strip(),
        "```",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")


def _filter_pairs_and_donors(
    pairs_csv: Path,
    donors_csv: Path,
    outdir: Path,
    n_cann: int,
    n_vict: int,
    cann_col: Optional[str],
    vict_col: Optional[str],
    seed: int = 2025
) -> tuple[Path, Path]:
    """
    Filtra a n_cann caníbales y n_vict víctimas por caníbal. Requiere columnas identificables.
    Si no se especifican columnas, intenta detectarlas por heurística.
    """
    df = pd.read_csv(pairs_csv)

    # Heurística de columnas si no se proveen
    if cann_col is None or vict_col is None:
        cann_candidates = [c for c in df.columns if any(k in c.lower() for k in
                           ["cannibal", "canibal", "treated", "i_", "iitem", "treat_item", "treated_item", "i_id"])]
        vict_candidates = [c for c in df.columns if any(k in c.lower() for k in
                           ["victim", "victima", "j_", "jitem", "target_item", "victim_item", "j_id"])]
        if cann_col is None and len(cann_candidates) > 0:
            cann_col = cann_candidates[0]
        if vict_col is None and len(vict_candidates) > 0:
            vict_col = vict_candidates[0]
    if cann_col is None or vict_col is None:
        raise ValueError("No pude identificar columnas de caníbal/víctima. Define params.pairs_cannibal_col / pairs_victim_col.")

    # Muestreo determinista: caníbales más frecuentes y top víctimas por caníbal
    cann_counts = df[cann_col].value_counts()
    chosen_cann = list(cann_counts.index[:n_cann]) if len(cann_counts) >= n_cann else list(cann_counts.index)
    df = df[df[cann_col].isin(chosen_cann)].copy()

    keep_idx: List[int] = []
    for _, g in df.groupby(cann_col):
        vict_counts = g[vict_col].value_counts().index.tolist()
        chosen_v = vict_counts[:n_vict]
        keep_idx.extend(g[g[vict_col].isin(chosen_v)].index.tolist())
    df_f = df.loc[sorted(set(keep_idx))].copy()

    outdir.mkdir(parents=True, exist_ok=True)
    pairs_out = outdir / (pairs_csv.stem + f"__subset_{n_cann}x{n_vict}.csv")
    df_f.to_csv(pairs_out, index=False)

    # Filtra donantes solo a víctimas retenidas
    dv = set(df_f[vict_col].unique())
    ddf = pd.read_csv(donors_csv)
    vict_cols_don = [c for c in ddf.columns if any(k in c.lower() for k in
                    ["victim", "victima", "target_item", "victim_item", "j_id", "victim_id"])]
    if len(vict_cols_don) == 0:
        raise ValueError("No pude identificar la columna de víctima en donors_per_victim.csv")
    vict_don_col = vict_cols_don[0]
    ddf_f = ddf[ddf[vict_don_col].isin(dv)].copy()
    donors_out = outdir / (donors_csv.stem + f"__subset_{n_cann}x{n_vict}.csv")
    ddf_f.to_csv(donors_out, index=False)
    return pairs_out, donors_out


# -------------------- Orquestador --------------------

def run_pipeline(config_path: Path) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)

    # exp_tag opcional desde YAML (inyectado por 01_run_sweep.py)
    exp_tag: Optional[str] = None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _raw = yaml.safe_load(f)
            exp_tag = _raw.get("exp_tag")
    except Exception:
        pass

    # Deriva rutas por iteración (si aplica)
    cfg.paths = derive_iter_paths(cfg.paths, exp_tag)
    if exp_tag:
        cfg.project_name = f"{cfg.project_name}__{exp_tag}"

    # # --- Validación de scoping por experimento (antes de crear nada) ---
    for name in ["processed_dir", "preprocessed_dir", "figures_dir", "gsc_out_dir", "meta_out_root"]:
        _assert_scoped(getattr(cfg.paths, name), exp_tag, name, hard=cfg.hard_scope)
        _touch_fingerprint(getattr(cfg.paths, name), exp_tag)
    _assert_scoped(cfg.paths.exposure_csv.parent, exp_tag, "exposure_csv.parent", hard=cfg.hard_scope)

    # # Logging y seeds
    log_path = setup_logging(cfg.log_level, cfg.log_to_file, exp_tag=exp_tag)
    logger = logging.getLogger("pipeline")
    seed_everything(cfg.seed)
    _log_toggles(cfg.toggles)

    logger.info("Proyecto: %s", cfg.project_name)
    logger.info("Configuración cargada desde: %s", config_path)

    # Asegurar/limpiar rutas base
    ensure_dir(cfg.paths.raw_dir)
    if cfg.clean_outputs:
        for p in [cfg.paths.processed_dir, cfg.paths.preprocessed_dir, cfg.paths.figures_dir,
                  cfg.paths.gsc_out_dir, cfg.paths.meta_out_root]:
            _purge_dir_if_needed(p)
    else:
        for p in [cfg.paths.processed_dir, cfg.paths.preprocessed_dir, cfg.paths.figures_dir,
                  cfg.paths.gsc_out_dir, cfg.paths.meta_out_root]:
            ensure_dir(p)

    # Ajustar sys.path para 'src' y 'EDA'
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(EDA_DIR))

    manifest: Dict[str, Any] = {
        "project_name": cfg.project_name,
        "start_time": datetime.now().isoformat(),
        "log_file": str(log_path),
        "steps": {},
        "exp_tag": exp_tag,
        "paths_effective": {
            "processed_dir": str(cfg.paths.processed_dir),
            "preprocessed_dir": str(cfg.paths.preprocessed_dir),
            "figures_dir": str(cfg.paths.figures_dir),
            "gsc_out_dir": str(cfg.paths.gsc_out_dir),
            "meta_out_root": str(cfg.paths.meta_out_root),
            "exposure_csv": str(cfg.paths.exposure_csv),
        },
    }

    # # # -------------------- Paso 1: Data Quality --------------------
    step1_outputs = {}
    if cfg.toggles.step1:
        out_dir = cfg.paths.step1_out_dir or cfg.paths.processed_dir
        train_filtered = out_dir / "train_filtered.csv"
        transactions_filtered = out_dir / "transactions_filtered.csv"
        
        # Check if cached outputs exist
        if cfg.toggles.use_cached_step1 and train_filtered.exists():
            logger.info("⚡ Paso 1: Usando outputs cacheados (train_filtered.csv existe)")
            step1_outputs = {
                "paths": {
                    "train_filtered": str(train_filtered),
                    "transactions_filtered": str(transactions_filtered) if transactions_filtered.exists() else None,
                }
            }
            logger.info("   ✓ Cache hit: %s", train_filtered)
        else:
            logger.info("Paso 1: Control de calidad y filtrado de datos 'train' y 'transactions'.")

            # assert de scoping de salida
            _assert_scoped(cfg.paths.processed_dir, exp_tag, "step1.out_dir(processed_dir)", hard=cfg.hard_scope)

            try:
                dq_path = SRC_DIR / "preprocess_data" / "1. data_quality.py"
                if dq_path.exists():
                    dq_mod = _dynamic_import("data_quality", dq_path)
                    run_data_quality = getattr(dq_mod, "run_data_quality")
                else:
                    from preprocess_data.data_quality import run_data_quality  # type: ignore
            except Exception:
                logger.exception("No se pudo localizar 'run_data_quality'.")
                if cfg.fail_fast:
                    raise
                run_data_quality = None  # type: ignore

            if run_data_quality is not None:
                ensure_dir(out_dir)
                step1_outputs = _safe_call(
                    run_data_quality,
                    logger,
                    cfg.fail_fast,
                    "1. Data Quality",
                    train_path=str(cfg.paths.train_csv),
                    transactions_path=str(cfg.paths.transactions_csv) if cfg.paths.transactions_csv else None,
                    out_dir=str(out_dir),
                    min_date=cfg.params.min_date,
                    date_col=cfg.params.date_col,
                    store_col=cfg.params.store_col,
                    item_col=cfg.params.item_col,
                    promo_col=cfg.params.promo_col,
                    sales_col=cfg.params.sales_col,
                    tx_col=cfg.params.tx_col,
                    save_report_json=str(cfg.params.save_report_json) if cfg.params.save_report_json else None,
                ) or {}
            else:
                logger.warning("Paso 1 omitido (función no disponible).")

        manifest["steps"]["step1"] = {
            "status": "ok" if step1_outputs else "skipped_or_failed",
            "outputs": step1_outputs,
        }
    else:
        logger.info("Paso 1 deshabilitado por configuración.")
        manifest["steps"]["step1"] = {"status": "disabled"}

    # # Resolver rutas de train/transactions a usar a partir del Step 1 si corresponde
    train_for_next = cfg.paths.train_csv
    transactions_for_next = cfg.paths.transactions_csv
    if cfg.toggles.step1 and cfg.use_filtered_from_step1 and step1_outputs:
        try:
            paths_dict = step1_outputs.get("paths", {})
            if paths_dict.get("train_filtered"):
                train_for_next = Path(paths_dict["train_filtered"])
            if paths_dict.get("transactions_filtered"):
                transactions_for_next = Path(paths_dict["transactions_filtered"])
            logger.info("Usando archivos filtrados del Paso 1.")
        except Exception:
            logger.warning("No se pudo extraer rutas filtradas del Paso 1; se usarán las originales.")

    # # -------------------- Paso 2: Competitive Exposure + EDA 1 y 2 --------------------
    exposure_path = cfg.paths.exposure_csv
    if cfg.toggles.step2:
        # Check if cached output exists
        if cfg.toggles.use_cached_step2 and exposure_path.exists():
            logger.info("⚡ Paso 2: Usando output cacheado (competitive_exposure.csv existe)")
            logger.info("   ✓ Cache hit: %s", exposure_path)
        else:
            logger.info("Paso 2: Cálculo de exposición competitiva (H).")
            _assert_scoped(cfg.paths.exposure_csv.parent, exp_tag, "step2.exposure_dir", hard=cfg.hard_scope)
            _touch_fingerprint(cfg.paths.exposure_csv.parent, exp_tag)

            try:
                ce_path = SRC_DIR / "preprocess_data" / "2. competitive_exposure.py"
                if ce_path.exists():
                    ce_mod = _dynamic_import("competitive_exposure", ce_path)
                    compute_competitive_exposure = getattr(ce_mod, "compute_competitive_exposure")
                else:
                    from preprocess_data.competitive_exposure import compute_competitive_exposure  # type: ignore
            except Exception:
                logger.exception("No se pudo localizar 'compute_competitive_exposure'.")
                if cfg.fail_fast:
                    raise
                compute_competitive_exposure = None  # type: ignore

            if compute_competitive_exposure is not None:
                ensure_dir(cfg.paths.processed_dir)
                _ = _safe_call(
                    compute_competitive_exposure,
                    logger,
                    cfg.fail_fast,
                    "2. Competitive Exposure",
                    train_path=str(train_for_next),
                    items_path=str(cfg.paths.items_csv) if cfg.paths.items_csv else None,
                    date_col=cfg.params.date_col,
                    store_col=cfg.params.store_col,
                    item_col=cfg.params.item_col,
                    promo_col=cfg.params.promo_col,
                    neighborhood_col=cfg.params.neighborhood_col if cfg.params.neighborhood_col else "class",
                    bin_threshold=cfg.params.h_bin_threshold,
                    save_path=str(exposure_path),
                    save_format=cfg.params.save_format,
                )
            else:
                logger.warning("Paso 2 omitido (función no disponible).")

        manifest["steps"]["step2"] = {
            "status": "ok" if exposure_path.exists() else "skipped_or_failed",
            "outputs": {"exposure_csv": str(exposure_path)},
        }
    else:
        logger.info("Paso 2 deshabilitado por configuración.")
    manifest["steps"]["step2"] = manifest["steps"].get("step2", {"status": "disabled"})

    # Validación ligera del archivo de exposure
    logger.info("Competitive Exposure guardado en: %s", str(exposure_path))
    try:
        if str(exposure_path).lower().endswith((".csv", ".txt")):
            head = pd.read_csv(exposure_path, nrows=5)
        else:
            head = pd.read_parquet(exposure_path)
        must = {"H_prop", "H_prop_raw", "competitive_exposure"}
        miss = must - set(head.columns)
        if miss:
            logger.warning("Archivo de exposure SIN columnas %s. Revisa que el código actualizado esté en uso (%s).", miss, exposure_path)
    except Exception as e:
        logger.warning("No pude validar columnas de exposure (%s): %s", exposure_path, e)

    # # # -------------------- Paso 3: select_pairs_and_donors + EDA 3 --------------------
    pairs_path = cfg.paths.preprocessed_dir / "pairs_windows.csv"
    donors_path = cfg.paths.preprocessed_dir / "donors_per_victim.csv"
    
    # Definir ubicación CENTRAL compartida y exp_tag_local (fuera del if para scope)
    outdir_central = cfg.params.outdir_pairs_donors / "_shared_base"
    exp_tag_local = cfg.exp_tag if hasattr(cfg, 'exp_tag') and cfg.exp_tag else (exp_tag or "default")

    if cfg.toggles.step3:
        
        # Check if cached outputs exist en ubicación central
        cached_pairs = outdir_central / "pairs_windows.csv"
        cached_donors = outdir_central / "donors_per_victim.csv"
        cached_episodes = outdir_central / "episodes_index.parquet"
        
        if cfg.toggles.use_cached_step3 and cached_pairs.exists() and cached_donors.exists():
            logger.info("⚡ Paso 3: Usando outputs cacheados CENTRALES (compartidos por todos los experimentos)")
            logger.info("   ✓ Cache hit: %s", outdir_central)
            pairs_path = cached_pairs
            donors_path = cached_donors
            if cached_episodes.exists():
                logger.info("   ✓ episodes_index.parquet también disponible")
        else:
            ensure_dir(outdir_central)
            # No usar exp_tag para scoping ya que es compartido
            _touch_fingerprint(outdir_central, "_shared_base")

            logger.info(f"📁 Guardando pairs/donors en ubicación CENTRAL: {outdir_central}")
            logger.info("   (Compartido por todos los experimentos)")
            logger.info("Paso 3: Selección de episodios (pares i-j) y donantes.")

            try:
                sp_path = SRC_DIR / "preprocess_data" / "3. select_pairs_and_donors.py"
                if sp_path.exists():
                    sp_mod = import_module_spawn_safe("select_pairs_and_donors", sp_path)
                    select_pairs_and_donors = getattr(sp_mod, "select_pairs_and_donors")
                else:
                    from preprocess_data.select_pairs_and_donors import select_pairs_and_donors  # type: ignore
            except Exception:
                logger.exception("No se pudo localizar 'select_pairs_and_donors'.")
                if cfg.fail_fast:
                    raise
                select_pairs_and_donors = None  # type: ignore

            if select_pairs_and_donors is not None:
                ensure_dir(cfg.params.outdir_pairs_donors)
                out = _safe_call(
                    select_pairs_and_donors,
                    logger,
                    cfg.fail_fast,
                    "3. Select Pairs & Donors",
                    H_csv=str(exposure_path),
                    train_csv=str(train_for_next),
                    items_csv=str(cfg.paths.items_csv),
                    stores_csv=str(cfg.paths.stores_csv),
                    outdir=str(outdir_central),  # ← ubicación CENTRAL compartida
                )
                # Si el módulo devolvió paths concretos, respétalos
                if isinstance(out, (tuple, list)) and len(out) >= 2 and out[0] and out[1]:
                    pairs_path = Path(out[0])
                    donors_path = Path(out[1])
            else:
                logger.warning("Paso 3 omitido (función no disponible).")

        # Filtrado a 10x10 episodios (si aplica)
        if cfg.params.n_cannibals and cfg.params.n_victims_per_cannibal:
            try:
                subset_dir = cfg.paths.preprocessed_dir / "subset"
                pairs_path, donors_path = _filter_pairs_and_donors(
                    pairs_path, donors_path, subset_dir,
                    n_cann=cfg.params.n_cannibals,
                    n_vict=cfg.params.n_victims_per_cannibal,
                    cann_col=getattr(cfg.params, "pairs_cannibal_col", None),
                    vict_col=getattr(cfg.params, "pairs_victim_col", None),
                    seed=cfg.seed
                )
                logger.info("Pairs/Donors filtrados a %dx%d (outdir=%s).",
                            cfg.params.n_cannibals, cfg.params.n_victims_per_cannibal, str(subset_dir))
            except Exception as e:
                logger.warning("No se pudo filtrar a %dx%d: %s (se continúa con el total).",
                               cfg.params.n_cannibals, cfg.params.n_victims_per_cannibal, e)

        # Normalizar nombres canónicos (por si el selector usó sufijos)
        canonical_pairs = outdir_central / "pairs_windows.csv"
        canonical_donors = outdir_central / "donors_per_victim.csv"
        try:
            if pairs_path.resolve() != canonical_pairs.resolve() and pairs_path.exists():
                shutil.copy2(pairs_path, canonical_pairs)
                pairs_path = canonical_pairs
            if donors_path.resolve() != canonical_donors.resolve() and donors_path.exists():
                shutil.copy2(donors_path, canonical_donors)
                donors_path = canonical_donors
        except Exception as e:
            logger.warning("No pude normalizar nombres de pairs/donors: %s", e)

        _assert_exists(pairs_path, "step3.pairs_path")
        _assert_exists(donors_path, "step3.donors_path")
        # NO validar scope para _shared_base ya que es compartido por todos los experimentos
        if "_shared_base" not in str(pairs_path.parent):
            _assert_scoped(pairs_path.parent, exp_tag_local, "step3.pairs_dir", hard=cfg.hard_scope)
            _assert_scoped(donors_path.parent, exp_tag_local, "step3.donors_dir", hard=cfg.hard_scope)

        manifest["steps"]["step3"] = {
            "status": "ok" if pairs_path.exists() and donors_path.exists() else "skipped_or_failed",
            "outputs": {"pairs_path": str(pairs_path), "donors_path": str(donors_path)},
        }
    else:
        logger.info("Paso 3 deshabilitado por configuración.")
        manifest["steps"]["step3"] = {"status": "disabled"}


        # --- Mirror opcional desde outdir_central a processed_dir/<exp_tag> ---
    try:
        # processed_out_dir ya existe más abajo; aquí usamos la misma convención
        processed_out_dir = cfg.paths.processed_dir
        exp_tag_local = cfg.exp_tag if getattr(cfg, "exp_tag", None) else (exp_tag or "default")
        # episodios (GSC) y meta
        _maybe_copy(outdir_central / "episodes_index.parquet",
                    processed_out_dir / "episodes_index.parquet",
                    "episodes_index (step3->processed_dir)")
        _maybe_copy(outdir_central / "episodes_index_meta.parquet",
                    processed_out_dir / "episodes_index_meta.parquet",
                    "episodes_index_meta (step3->processed_dir)")
        _maybe_copy(pairs_path, processed_out_dir / "pairs_windows.csv",
                    "pairs_windows (step3->processed_dir)")
        _maybe_copy(donors_path, processed_out_dir / "donors_per_victim.csv",
                    "donors (step3->processed_dir)")
    except Exception:
        pass
  

    # # Actualizar config para el paso 4
    cfg.params.episodes_path = pairs_path
    cfg.params.donors_path = donors_path
    cfg.params.outdir_pairs_donors = pairs_path.parent

    # # # -------------------- EDA 3 --------------------
    if cfg.toggles.eda3:
        logger.info("EDA 3: Muestreo de episodios y diagnóstico de calidad.")
        try:
            from eda_3 import run as eda3_run  # type: ignore
        except Exception:
            try:
                from EDA.eda_3 import run as eda3_run  # type: ignore
            except Exception:
                eda3_run = None  # type: ignore

        if eda3_run is None:
            logger.warning("EDA 3 no disponible (run no encontrado).")
            manifest["steps"]["eda3"] = {"status": "skipped_or_failed"}
        else:
            ensure_dir(cfg.paths.figures_dir)
            _safe_call(
                eda3_run,
                logger,
                cfg.fail_fast,
                "EDA 3",
                pairs_path=str(pairs_path),
                donors_path=str(donors_path),
                out_dir=str(cfg.paths.figures_dir),
                n=cfg.params.eda3_n,
                strategy=cfg.params.eda3_strategy,
                seed=cfg.params.eda3_seed,
                orientation=cfg.params.eda3_orientation,
                dpi=cfg.params.eda3_dpi,
                quality_cfg=None,
            )
            manifest["steps"]["eda3"] = {"status": "ok", "figures_dir": str(cfg.paths.figures_dir)}
            if exp_tag:
                stamp_pngs(cfg.paths.figures_dir, exp_tag, _short_cfg_summary(cfg.params))
    else:
        logger.info("EDA 3 deshabilitado por configuración.")
        manifest["steps"]["eda3"] = {"status": "disabled"}

    # # # -------------------- Paso 4: pre_algorithm + EDA 4 --------------------
    processed_out_dir = cfg.paths.processed_dir
    episodes_path_for_step4 = cfg.params.episodes_path
    donors_path_for_step4 = cfg.params.donors_path

    if cfg.toggles.step4:
        logger.info("Paso 4: Preparación de datasets para GSC y Meta-learners.")
        logger.info(f"Usando episodios desde: {episodes_path_for_step4}")
        logger.info(f"Usando donantes desde: {donors_path_for_step4}")

        _assert_scoped(processed_out_dir, exp_tag, "step4.processed_out_dir", hard=cfg.hard_scope)
        _assert_exists(episodes_path_for_step4, "step4.episodes_path")
        _assert_exists(donors_path_for_step4, "step4.donors_path")

        try:
            pa_path = SRC_DIR / "preprocess_data" / "4. pre_algorithm.py"
            if pa_path.exists():
                pa_mod = import_module_spawn_safe("pre_algorithm", pa_path)
                PrepConfig = getattr(pa_mod, "PrepConfig")
                prepare_datasets = getattr(pa_mod, "prepare_datasets")
            else:
                from preprocess_data.pre_algorithm import PrepConfig, prepare_datasets  # type: ignore
        except Exception:
            logger.exception("No se pudo localizar 'prepare_datasets' / 'PrepConfig'.")
            if cfg.fail_fast:
                raise
            prepare_datasets = None  # type: ignore
            PrepConfig = None  # type: ignore

        if prepare_datasets is not None and PrepConfig is not None:
            pa_cfg = PrepConfig(
                episodes_path=str(episodes_path_for_step4),
                donors_path=str(donors_path_for_step4),
                raw_dir=str(cfg.paths.raw_dir),
                out_dir=str(processed_out_dir),
                top_k_donors=cfg.params.top_k_donors,
                donor_kind=cfg.params.donor_kind,
                lags_days=tuple(cfg.params.lags_days),
                fourier_k=cfg.params.fourier_k,
                max_donor_promo_share=cfg.params.max_donor_promo_share,
                min_availability_share=cfg.params.min_availability_share,
                gsc_eval_n=cfg.params.gsc_eval_n,
                gsc_eval_selection=cfg.params.gsc_eval_selection,
                save_intermediate=cfg.params.save_intermediate,
                use_stl=cfg.params.use_stl,
                drop_city=cfg.params.drop_city,
                dry_run=cfg.params.dry_run,
                max_episodes=cfg.params.max_episodes,
                log_level=cfg.params.prep_log_level,
                fail_fast=cfg.params.fail_fast,
            )

            _safe_call(
                prepare_datasets,
                logger,
                cfg.fail_fast,
                "4. Pre-Algorithm Prep",
                cfg=pa_cfg,
            )

            # Esperables del prep
            for must_dir in [processed_out_dir / "gsc", processed_out_dir / "intermediate", processed_out_dir / "meta"]:
                _assert_scoped(must_dir, exp_tag, f"step4.out_dir::{must_dir.name}", hard=cfg.hard_scope)
                _touch_fingerprint(must_dir, exp_tag)

            manifest["steps"]["step4"] = {
                "status": "ok",
                "outputs": {
                    "episodes_index": str(processed_out_dir / "episodes_index.parquet"),
                    "gsc_dir": str(processed_out_dir / "gsc"),
                    "meta_all": str(processed_out_dir / "meta" / "all_units.parquet"),
                    "panel_features": str(processed_out_dir / "intermediate" / "panel_features.parquet"),
                    "donor_quality": str(processed_out_dir / "gsc" / "donor_quality.parquet"),
                },
            }
            # --- Sanidad básica: donantes no pueden incluir víctima/caníbal ---
            try:
                epi_idx = pd.read_parquet(processed_out_dir / "episodes_index.parquet")
                donors_df = pd.read_csv(donors_path_for_step4)
                vict_cols = [c for c in donors_df.columns if any(k in c.lower() for k in ["victim","target_item","j_id","victim_id"])]
                donor_unit_cols = [c for c in donors_df.columns if any(k in c.lower() for k in ["donor","donor_item","k_id","unit_id"])]
                if vict_cols and donor_unit_cols:
                    bad = donors_df[donors_df[vict_cols[0]] == donors_df[donor_unit_cols[0]]]
                    if len(bad) > 0:
                        raise RuntimeError(f"Donantes incluyen a la víctima/caníbal ({len(bad)} filas). Corrige selección de donantes antes de Step 5.")
            except Exception as e:
                logger.warning("Chequeo de donantes/víctima no concluyente: %s", e)
        else:
            logger.warning("Paso 4 omitido (funciones no disponibles).")
            manifest["steps"]["step4"] = {"status": "skipped_or_failed"}
    else:
        logger.info("Paso 4 deshabilitado por configuración.")
        manifest["steps"]["step4"] = {"status": "disabled"}

    # # -------------------- EDA 4 --------------------
    if cfg.toggles.eda4:
        logger.info("EDA 4: Reporte de datasets procesados.")
        try:
            from eda_4 import EDAConfig as EDA4Config, run as eda4_run  # type: ignore
        except Exception:
            try:
                from EDA.eda_4 import EDAConfig as EDA4Config, run as eda4_run  # type: ignore
            except Exception:
                eda4_run = None  # type: ignore
                EDA4Config = None  # type: ignore

        if eda4_run is None or EDA4Config is None:
            logger.warning("EDA 4 no disponible (run/EDAConfig no encontrados).")
            manifest["steps"]["eda4"] = {"status": "skipped_or_failed"}
        else:
            episodes_index = processed_out_dir / "episodes_index.parquet"
            donor_quality = processed_out_dir / "gsc" / "donor_quality.parquet"
            meta_all = processed_out_dir / "meta" / "all_units.parquet"
            panel_features = processed_out_dir / "intermediate" / "panel_features.parquet"
            gsc_dir = processed_out_dir / "gsc"

            eda4_cfg = EDA4Config(
                processed_dir=str(processed_out_dir),
                episodes_index=str(episodes_index) if episodes_index.exists() else None,
                donor_quality=str(donor_quality) if donor_quality.exists() else None,
                meta_all=str(meta_all) if meta_all.exists() else None,
                panel_features=str(panel_features) if panel_features.exists() else None,
                gsc_dir=str(gsc_dir) if gsc_dir.exists() else None,
                out_dir=str(cfg.paths.figures_dir),
                dpi=cfg.params.eda4_dpi,
                orientation=cfg.params.eda4_orientation,
                promo_thresh=cfg.params.eda4_promo_thresh,
                avail_thresh=cfg.params.eda4_avail_thresh,
                limit_episodes=cfg.params.eda4_limit_episodes,
                example_episode_id=cfg.params.eda4_example_episode_id,
                log_level=cfg.params.eda4_log_level,
            )
            _safe_call(eda4_run, logger, cfg.fail_fast, "EDA 4", cfg=eda4_cfg)
            manifest["steps"]["eda4"] = {"status": "ok", "figures_dir": str(cfg.paths.figures_dir)}
            if exp_tag:
                stamp_pngs(cfg.paths.figures_dir, exp_tag, _short_cfg_summary(cfg.params))
    else:
        logger.info("EDA 4 deshabilitado por configuración.")
        manifest["steps"]["eda4"] = {"status": "disabled"}

    # -------------------- Paso 5: Algoritmo GSC --------------------
    if cfg.toggles.step5_gsc:
        logger.info("Paso 5: Ejecución de GSC (synthetic control generalizado).")
        episodes_index = processed_out_dir / "episodes_index.parquet"
        gsc_in_dir = processed_out_dir / "gsc"                     # insumos
        gsc_out_dir = cfg.paths.gsc_out_dir / "gsc"               # <-- salida estandarizada (bajo <exp_tag>/gsc)
        ensure_dir(gsc_out_dir)

        if not episodes_index.exists() or not gsc_in_dir.exists():
            logger.warning("Insumos de GSC no disponibles (episodes_index/gsc_dir). Paso 5 omitido.")
            manifest["steps"]["step5_gsc"] = {"status": "skipped_or_failed"}
        else:
            try:
                from models.synthetic_control import RunConfig as GSCRunConfig, run_batch as gsc_run_batch  # type: ignore
            except Exception:
                try:
                    from src.models.synthetic_control import RunConfig as GSCRunConfig, run_batch as gsc_run_batch  # type: ignore
                except Exception:
                    logger.exception("No se pudieron importar RunConfig/run_batch de synthetic_control.")
                    GSCRunConfig, gsc_run_batch = None, None  # type: ignore

            if GSCRunConfig is None or gsc_run_batch is None:
                manifest["steps"]["step5_gsc"] = {"status": "skipped_or_failed"}
            else:
                _assert_scoped(gsc_in_dir, exp_tag, "step5.gsc_in_dir", hard=cfg.hard_scope)
                _assert_scoped(gsc_out_dir, exp_tag, "step5.gsc_out_dir", hard=cfg.hard_scope)
                _touch_fingerprint(gsc_out_dir, exp_tag)

                gsc_cfg = GSCRunConfig(
                    gsc_dir=Path(gsc_in_dir),
                    donors_csv=Path(donors_path),
                    episodes_index=Path(episodes_index),
                    out_dir=Path(gsc_out_dir),                 # <-- salida unificada aquí
                    log_level=cfg.params.gsc_log_level,
                    max_episodes=cfg.params.gsc_max_episodes,
                    do_placebo_space=cfg.params.gsc_do_placebo_space,
                    do_placebo_time=cfg.params.gsc_do_placebo_time,
                    do_loo=cfg.params.gsc_do_loo,
                    max_loo=cfg.params.gsc_max_loo,
                    sens_samples=cfg.params.gsc_sens_samples,
                    cv_folds=cfg.params.gsc_cv_folds,
                    cv_holdout=cfg.params.gsc_cv_holdout,
                    rank=cfg.params.gsc_rank,
                    tau=cfg.params.gsc_tau,
                    alpha=cfg.params.gsc_alpha,
                )
                _safe_call(gsc_run_batch, logger, cfg.fail_fast, "5. GSC run_batch", cfg=gsc_cfg)

                # Buscamos métricas en la nueva ubicación canónica y con fallbacks
                candidates = [
                    gsc_out_dir / "gsc_metrics.parquet",
                    cfg.paths.gsc_out_dir / "gsc_metrics.parquet",
                    cfg.paths.gsc_out_dir / "metrics" / "gsc_metrics.parquet",
                ]
                out_metrics = None
                for c in candidates:
                    if c.exists():
                        out_metrics = c
                        break
                if out_metrics is None:
                    out_metrics = gsc_out_dir / "gsc_metrics.parquet"  # ruta canónica (para el assert)
                _assert_exists(out_metrics, "step5.gsc_metrics")

                # Episodios procesados por GSC
                episodes_done_gsc: List[Any] = []
                try:
                    if out_metrics.exists():
                        gm = pd.read_parquet(out_metrics)
                        if "episode_id" in gm.columns:
                            episodes_done_gsc = sorted(gm["episode_id"].dropna().unique().tolist())
                except Exception:
                    pass

                manifest["steps"]["step5_gsc"] = {
                    "status": "ok" if (out_metrics and out_metrics.exists()) else "skipped_or_failed",
                    "outputs": {
                        "gsc_metrics": str(out_metrics),
                        "gsc_cf_series_dir": str(gsc_out_dir / "cf_series"),
                        "episodes_done": episodes_done_gsc,
                    },
                }
    else:
        logger.info("Paso 5 (GSC) deshabilitado por configuración.")
        manifest["steps"]["step5_gsc"] = {"status": "disabled"}

    # # -------------------- Paso 6: Algoritmos Meta-learners --------------------
    if cfg.toggles.step6_meta:
        logger.info("Paso 6: Ejecución de Meta‑learners (T/S/X).")
        meta_parquet = processed_out_dir / "meta" / "all_units.parquet"
        episodes_index = processed_out_dir / "episodes_index.parquet"
        if not meta_parquet.exists() or not episodes_index.exists():
            logger.warning("Insumos Meta no disponibles (meta/all_units o episodes_index). Paso 6 omitido.")
            manifest["steps"]["step6_meta"] = {"status": "skipped_or_failed"}
        else:
            try:
                from models.meta_learners import RunCfg as MetaRunCfg, run_batch as meta_run_batch  # type: ignore
            except Exception:
                try:
                    from src.models.meta_learners import RunCfg as MetaRunCfg, run_batch as meta_run_batch  # type: ignore
                except Exception:
                    logger.exception("No se pudieron importar RunCfg/run_batch de meta_learners.")
                    MetaRunCfg, meta_run_batch = None, None  # type: ignore

            if MetaRunCfg is None or meta_run_batch is None:
                manifest["steps"]["step6_meta"] = {"status": "skipped_or_failed"}
            else:
                _assert_scoped(cfg.paths.meta_out_root, exp_tag, "step6.meta_out_root", hard=cfg.hard_scope)
                _touch_fingerprint(cfg.paths.meta_out_root, exp_tag)

                learners_to_run: List[str] = list(cfg.params.meta_learners)
                outputs = {}
                for lr in learners_to_run:
                    out_dir = cfg.paths.meta_out_root / lr
                    ensure_dir(out_dir)
                    mcfg = MetaRunCfg(
                        learner=lr,
                        meta_parquet=Path(meta_parquet),
                        episodes_index=Path(episodes_index),
                        donors_csv=Path(donors_path),
                        out_dir=Path(out_dir),
                        log_level=cfg.log_level,
                        max_episodes=cfg.params.meta_max_episodes,
                        cv_folds=cfg.params.meta_cv_folds,
                        cv_holdout_days=cfg.params.meta_cv_holdout,
                        min_train_samples=cfg.params.meta_min_train_samples,
                        model=cfg.params.meta_model,
                        prop_model=cfg.params.meta_prop_model,
                        random_state=cfg.params.meta_random_state,
                        max_depth=cfg.params.meta_max_depth,
                        learning_rate=cfg.params.meta_learning_rate,
                        max_iter=cfg.params.meta_max_iter,
                        min_samples_leaf=cfg.params.meta_min_samples_leaf,
                        l2=cfg.params.meta_l2,
                        hpo_trials=cfg.params.meta_hpo_trials,
                        do_placebo_space=cfg.params.meta_do_placebo_space,
                        do_placebo_time=cfg.params.meta_do_placebo_time,
                        do_loo=cfg.params.meta_do_loo,
                        max_loo=cfg.params.meta_max_loo,
                        sens_samples=cfg.params.meta_sens_samples,
                        treat_col_s=cfg.params.treat_col_s,
                        s_ref=cfg.params.s_ref,
                        treat_col_b=cfg.params.treat_col_b,
                        bin_threshold=cfg.params.bin_threshold,
                    )
                    _safe_call(meta_run_batch, logger, cfg.fail_fast, f"6.{lr.upper()} Meta run_batch", cfg=mcfg)
                    outputs[lr] = {
                        "meta_metrics": str(out_dir / f"meta_metrics_{lr}.parquet"),
                        "cf_series_dir": str(out_dir / "cf_series"),
                    }
                    _assert_exists(out_dir / f"meta_metrics_{lr}.parquet", f"step6.meta_metrics[{lr}]")

                    # --- Sanidad Meta: no-RMSPE_pre≈0 ---
                    try:
                        mp = out_dir / f"meta_metrics_{lr}.parquet"
                        if mp.exists():
                            mm = pd.read_parquet(mp)
                            if "rmspe_pre" in mm.columns:
                                bad = mm[mm["rmspe_pre"] <= RMSPE_EPS]
                                if len(bad) > 0:
                                    raise RuntimeError(f"Meta-{lr.upper()}: {len(bad)} episodios con rmspe_pre≈0. Revisa unión de observado.")
                    except Exception as e:
                        logger.warning("Chequeo Meta (%s) no concluyente: %s", lr, e)

                    # Episodios procesados por learner
                    episodes_done_meta: List[Any] = []
                    try:
                        mp = out_dir / f"meta_metrics_{lr}.parquet"
                        if mp.exists():
                            mm = pd.read_parquet(mp)
                            if "episode_id" in mm.columns:
                                episodes_done_meta = sorted(mm["episode_id"].dropna().unique().tolist())
                    except Exception:
                        pass
                    outputs[lr]["episodes_done"] = episodes_done_meta

                manifest["steps"]["step6_meta"] = {"status": "ok", "outputs": outputs}
    else:
        logger.info("Paso 6 (Meta) deshabilitado por configuración.")
        manifest["steps"]["step6_meta"] = {"status": "disabled"}

   # -------------------- EDA FINAL: Algoritmos (series y resúmenes) --------------------
    if cfg.toggles.eda_algorithms:
        logger.info("EDA final de algoritmos: render de láminas por episodio y comparativas.")
        try:
            from EDA_algorithms import EDAConfig as EDAAlgConfig, run as eda_alg_run  # type: ignore
        except Exception:
            try:
                from EDA.EDA_algorithms import EDAConfig as EDAAlgConfig, run as eda_alg_run  # type: ignore
            except Exception:
                eda_alg_run = None  # type: ignore
                EDAAlgConfig = None  # type: ignore

        episodes_index_path = processed_out_dir / "episodes_index.parquet"

        # Fallback: usa el índice generado en Step 3 si falta el del Step 4
        if not episodes_index_path.exists():
            alt_from_step3 = cfg.params.outdir_pairs_donors / (cfg.exp_tag or exp_tag or "default") / "episodes_index.parquet"
            if alt_from_step3.exists():
                logging.warning("EDA_algorithms: se usará episodes_index desde Step 3: %s", str(alt_from_step3))
                episodes_index_path = alt_from_step3

        if eda_alg_run is None or EDAAlgConfig is None:
            logger.warning("EDA_algorithms no disponible (run/EDAConfig no encontrados).")
            manifest["steps"]["eda_algorithms"] = {"status": "skipped_or_failed"}
        elif not episodes_index_path.exists():
            logger.warning("EDA_algorithms omitido: episodes_index no existe en %s. Revisa Step 4.", str(episodes_index_path))
            manifest["steps"]["eda_algorithms"] = {
                "status": "skipped_or_failed",
                "reason": f"missing episodes_index at {episodes_index_path}"
            }
        else:
            # Cobertura de episodios para diagnóstico
            coverage: Dict[str, Any] = {}
            try:
                expected = set(pd.read_parquet(episodes_index_path)["episode_id"].unique().tolist())
            except Exception:
                expected = set()

            # GSC: buscar métricas con rutas alternativas
            gsc_mets_candidates = [
                cfg.paths.gsc_out_dir / "gsc" / "gsc_metrics.parquet",   # <-- ubicación canónica nueva
                cfg.paths.gsc_out_dir / "gsc_metrics.parquet",            # backward-compat
                cfg.paths.gsc_out_dir / "metrics" / "gsc_metrics.parquet" # por si el modelo las deja en /metrics
            ]
            gsc_mets_path = next((p for p in gsc_mets_candidates if p.exists()), None)

            eps_gsc: set = set()
            if gsc_mets_path and gsc_mets_path.exists():
                try:
                    gm = pd.read_parquet(gsc_mets_path)
                    if "episode_id" in gm.columns:
                        eps_gsc = set(gm["episode_id"].dropna().unique().tolist())
                except Exception:
                    pass

            # Meta: unión de episodios procesados por los learners ejecutados
            eps_meta_union: set = set()
            if "step6_meta" in manifest["steps"] and manifest["steps"]["step6_meta"].get("outputs"):
                for lr, out in manifest["steps"]["step6_meta"]["outputs"].items():
                    mp = Path(out["meta_metrics"])
                    if mp.exists():
                        try:
                            mm = pd.read_parquet(mp)
                            if "episode_id" in mm.columns:
                                eps_meta_union |= set(mm["episode_id"].unique().tolist())
                        except Exception:
                            pass

            available = eps_gsc | eps_meta_union
            missing = expected - available if expected else set()
            coverage = {
                "expected": len(expected),
                "gsc": len(eps_gsc),
                "meta_union": len(eps_meta_union),
                "available": len(available),
                "missing": len(missing),
            }
            logger.info("Cobertura de episodios para EDA_algorithms: %s", coverage)

            learners_for_eda = tuple(cfg.params.eda_alg_learners) if cfg.params.eda_alg_learners else tuple(cfg.params.meta_learners)

            # IMPORTANTE: pasar el directorio donde REALMENTE quedaron los outputs de GSC
            eda_cfg = EDAAlgConfig(
                episodes_index=Path(episodes_index_path),
                gsc_out_dir=Path(cfg.paths.gsc_out_dir) / "gsc",   # <-- aquí va el subdirectorio 'gsc'
                meta_out_root=Path(cfg.paths.meta_out_root),
                meta_learners=learners_for_eda,
                figures_dir=Path(cfg.paths.figures_dir),
                orientation=cfg.params.eda_alg_orientation,
                dpi=cfg.params.eda_alg_dpi,
                style="academic",
                font_size=10,
                grid=True,
                max_episodes_gsc=cfg.params.eda_alg_max_episodes_gsc,
                max_episodes_meta=cfg.params.eda_alg_max_episodes_meta,
                export_pdf=cfg.params.eda_alg_export_pdf,
            )

            _safe_call(eda_alg_run, logger, cfg.fail_fast, "EDA Algorithms", cfg=eda_cfg)
            _assert_scoped(cfg.paths.figures_dir, exp_tag, "EDA.figures_dir", hard=cfg.hard_scope)
            _touch_fingerprint(cfg.paths.figures_dir, exp_tag)
            manifest["steps"]["eda_algorithms"] = {
                "status": "ok",
                "figures_dir": str(cfg.paths.figures_dir),
                "coverage": coverage,
            }
            if exp_tag:
                stamp_pngs(cfg.paths.figures_dir, exp_tag, _short_cfg_summary(cfg.params))
    else:
        logger.info("EDA Algorithms deshabilitado por configuración.")
        manifest["steps"]["eda_algorithms"] = {"status": "disabled"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Orquestador del pipeline (00_run_pipeline.py).")
    p.add_argument("--config", type=str, default="pipeline_config.yaml", help="Ruta al YAML de configuración.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {config_path}")
    run_pipeline(config_path)