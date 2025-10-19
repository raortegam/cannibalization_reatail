#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_run_pipeline.py
Orquestador del pipeline de preprocesamiento y EDA para canibalización promocional en retail.

Características principales:
- Ejecuta los 4 pasos de preprocesamiento (1 a 4) y los 4 EDAs (1 a 4), de forma configurable.
- Carga configuración desde un archivo YAML.
- Logs dicientes a consola y archivo (diagnostics/).
- Maneja módulos con nombres de archivo no estándar (p.ej., "1. data_quality.py") vía import dinámico.
- Guarda artefactos en las rutas definidas por la configuración.
- Continúa o se detiene ante errores según "fail_fast".

Uso:
    python 00_run_pipeline.py --config pipeline_config.yaml

Requisitos:
    - PyYAML (pip install pyyaml)
    - Estructura de repo según el contexto provisto.
"""
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
from typing import Any, Dict, Optional, Tuple

try:
    import yaml  # PyYAML
except Exception as e:
    raise RuntimeError("Se requiere PyYAML. Instálalo con `pip install pyyaml`.") from e


# -------------------- Utilidades generales --------------------

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
EDA_DIR = REPO_ROOT / "EDA"
DIAG_DIR = REPO_ROOT / "diagnostics"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def setup_logging(level: str = "INFO", log_to_file: bool = True) -> Path:
    ensure_dir(DIAG_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = DIAG_DIR / f"pipeline_{ts}.log"

    # Limpiar handlers previos si se reejecuta en el mismo intérprete
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


def _dynamic_import(module_name: str, py_path: Path):
    """
    Carga dinámica de un módulo desde ruta absoluta (soporta nombres con espacios o puntos).
    Devuelve el módulo cargado.
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
    preprocessed_dir: Path = Path(".data/preprocessed_data")
    figures_dir: Path = Path("figures/favorita_analysis")
    exposure_csv: Path = Path(".data/processed_data/competitive_exposure.csv")

    # Artefactos de Step 1 (si se desea forzar rutas)
    step1_out_dir: Optional[Path] = None


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

    # Step 2: Competitive Exposure
    neighborhood_col: Optional[str] = None  # si aplica
    save_format: Optional[str] = None  # infiere por extensión si None

    # Step 3: select_pairs_and_donors
    outdir_pairs_donors: Path = Path(".data/preprocessed_data")

    # Step 4: pre_algorithm -> PrepConfig
    # Ver defaults en el módulo original; aquí exponemos más usados.
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
    fail_fast: bool = False  # del pre_algorithm

    # EDA 2: parámetros de visualización
    eda2_orientation: str = "portrait"
    eda2_dpi: int = 300
    eda2_bins: int = 50
    eda2_top_stores: int = 30
    eda2_min_store_obs: int = 5
    eda2_heatmap_stores: Optional[int] = None
    eda2_chunksize: int = 1_000_000
    eda2_log_level: str = "INFO"

    # EDA 3: sampleo y calidad
    eda3_n: int = 5
    eda3_strategy: str = "stratified"
    eda3_seed: int = 2025
    eda3_orientation: str = "portrait"
    eda3_dpi: int = 300
    # Para quality_cfg usar los defaults del módulo si existe

    # EDA 4: reporte final
    eda4_promo_thresh: float = 0.02
    eda4_avail_thresh: float = 0.90
    eda4_limit_episodes: int = 1
    eda4_example_episode_id: Optional[str] = None
    eda4_orientation: str = "portrait"
    eda4_dpi: int = 300
    eda4_log_level: str = "INFO"


@dataclass
class OrchestratorConfig:
    project_name: str
    seed: int = 2025
    log_level: str = "INFO"
    log_to_file: bool = True
    fail_fast: bool = False  # a nivel orquestador
    use_filtered_from_step1: bool = True  # usar outputs filtrados del paso 1

    toggles: StepToggles = field(default_factory=StepToggles)
    paths: PathsConfig = field(default=None)   # type: ignore[assignment]
    params: ParamsConfig = field(default_factory=ParamsConfig)


def load_yaml_config(path: Path) -> OrchestratorConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ensamblar las dataclasses con casting a Path donde aplique
    toggles = StepToggles(**cfg.get("toggles", {}))

    paths_raw = cfg["paths"]
    paths = PathsConfig(
        raw_dir=Path(paths_raw["raw_dir"]),
        train_csv=Path(paths_raw["train_csv"]),
        transactions_csv=Path(paths_raw["transactions_csv"]) if paths_raw.get("transactions_csv") else None,
        items_csv=Path(paths_raw["items_csv"]) if paths_raw.get("items_csv") else None,
        stores_csv=Path(paths_raw["stores_csv"]) if paths_raw.get("stores_csv") else None,
        processed_dir=Path(paths_raw.get("processed_dir", ".data/processed_data")),
        preprocessed_dir=Path(paths_raw.get("preprocessed_dir", ".data/preprocessed_data")),
        figures_dir=Path(paths_raw.get("figures_dir", "figures/favorita_analysis")),
        exposure_csv=Path(paths_raw.get("exposure_csv", ".data/processed_data/competitive_exposure.csv")),
        step1_out_dir=Path(paths_raw["step1_out_dir"]) if paths_raw.get("step1_out_dir") else None,
    )

    params = ParamsConfig(**cfg.get("params", {}))

    orch = OrchestratorConfig(
        project_name=cfg.get("project_name", "cannibalization_retail"),
        seed=cfg.get("seed", 2025),
        log_level=cfg.get("log_level", "INFO"),
        log_to_file=cfg.get("log_to_file", True),
        fail_fast=cfg.get("fail_fast", False),
        use_filtered_from_step1=cfg.get("use_filtered_from_step1", True),
        toggles=toggles,
        paths=paths,
        params=params,
    )
    return orch


# -------------------- Orquestador --------------------

def run_pipeline(config_path: Path) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)

    # Logging y seeds
    log_path = setup_logging(cfg.log_level, cfg.log_to_file)
    logger = logging.getLogger("pipeline")
    seed_everything(cfg.seed)

    logger.info("Proyecto: %s", cfg.project_name)
    logger.info("Configuración cargada desde: %s", config_path)

    # Asegurar rutas base
    ensure_dir(cfg.paths.raw_dir)
    ensure_dir(cfg.paths.processed_dir)
    ensure_dir(cfg.paths.preprocessed_dir)
    ensure_dir(cfg.paths.figures_dir)

    # Ajustar sys.path para 'src' y 'EDA'
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(EDA_DIR))

    manifest: Dict[str, Any] = {
        "project_name": cfg.project_name,
        "start_time": datetime.now().isoformat(),
        "log_file": str(log_path),
        "steps": {},
    }

    # -------------------- Paso 1: Data Quality --------------------
    step1_outputs = {}
    if cfg.toggles.step1:
        logger.info("Paso 1: Control de calidad y filtrado de datos 'train' y 'transactions'.")
        try:
            # Intentar importar dinámicamente el archivo con nombre atípico
            dq_path = SRC_DIR / "preprocess_data" / "1. data_quality.py"
            if dq_path.exists():
                dq_mod = _dynamic_import("data_quality", dq_path)
                run_data_quality = getattr(dq_mod, "run_data_quality")
            else:
                # Fallback por si existe un módulo regular sin prefijo
                from preprocess_data.data_quality import run_data_quality  # type: ignore
        except Exception:
            logger.exception("No se pudo localizar 'run_data_quality'.")
            if cfg.fail_fast:
                raise
            run_data_quality = None  # type: ignore

        if run_data_quality is not None:
            out_dir = cfg.paths.step1_out_dir or cfg.paths.processed_dir
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

    # Resolver rutas de train/transactions a usar a partir del Step 1 si corresponde
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

    # -------------------- Paso 2: Competitive Exposure + EDA 1 y 2 --------------------
    exposure_path = cfg.paths.exposure_csv
    if cfg.toggles.step2:
        logger.info("Paso 2: Cálculo de exposición competitiva (H).")
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
                neighborhood_col=cfg.params.neighborhood_col if cfg.params.neighborhood_col else "neighborhood",
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
        manifest["steps"]["step2"] = {"status": "disabled"}

    # EDA 1 (descriptivo general)
    if cfg.toggles.eda1:
        logger.info("EDA 1: Resumen general (ventas, promos, transacciones).")
        try:
            from eda_1 import run_eda  # type: ignore
        except Exception:
            try:
                from EDA.eda_1 import run_eda  # type: ignore
            except Exception:
                run_eda = None  # type: ignore

        if run_eda is None:
            logger.warning("EDA 1 no disponible (run_eda no encontrado).")
            manifest["steps"]["eda1"] = {"status": "skipped_or_failed"}
        else:
            ensure_dir(cfg.paths.figures_dir)
            _safe_call(
                run_eda,
                logger,
                cfg.fail_fast,
                "EDA 1",
                train_path=str(train_for_next),
                out_dir=str(cfg.paths.figures_dir),
                transactions_path=str(transactions_for_next) if transactions_for_next else None,
                items_path=str(cfg.paths.items_csv) if cfg.paths.items_csv else None,
                date_col=cfg.params.date_col,
                store_col=cfg.params.store_col,
                item_col=cfg.params.item_col,
                promo_col=cfg.params.promo_col,
                sales_col=cfg.params.sales_col,
                tx_col=cfg.params.tx_col,
                neighborhood_col=cfg.params.neighborhood_col,
            )
            manifest["steps"]["eda1"] = {"status": "ok", "figures_dir": str(cfg.paths.figures_dir)}
    else:
        logger.info("EDA 1 deshabilitado por configuración.")
        manifest["steps"]["eda1"] = {"status": "disabled"}

    # EDA 2 (exposición competitiva)
    if cfg.toggles.eda2:
        logger.info("EDA 2: Exploración de exposición competitiva.")
        try:
            from eda_2 import run_eda_competitive_exposure  # type: ignore
        except Exception:
            try:
                from EDA.eda_2 import run_eda_competitive_exposure  # type: ignore
            except Exception:
                run_eda_competitive_exposure = None  # type: ignore

        if run_eda_competitive_exposure is None:
            logger.warning("EDA 2 no disponible (run_eda_competitive_exposure no encontrado).")
            manifest["steps"]["eda2"] = {"status": "skipped_or_failed"}
        else:
            ensure_dir(cfg.paths.figures_dir)
            _safe_call(
                run_eda_competitive_exposure,
                logger,
                cfg.fail_fast,
                "EDA 2",
                input_path=str(exposure_path),
                output_dir=str(cfg.paths.figures_dir),
                date_col=cfg.params.date_col,
                store_col=cfg.params.store_col,
                item_col=cfg.params.item_col,
                exposure_col="competitive_exposure",
                orientation=cfg.params.eda2_orientation,
                dpi=cfg.params.eda2_dpi,
                bins=cfg.params.eda2_bins,
                top_stores=cfg.params.eda2_top_stores,
                min_store_obs=cfg.params.eda2_min_store_obs,
                heatmap_stores=cfg.params.eda2_heatmap_stores,
                chunksize=cfg.params.eda2_chunksize,
                log_level=cfg.params.eda2_log_level,
            )
            manifest["steps"]["eda2"] = {"status": "ok", "figures_dir": str(cfg.paths.figures_dir)}
    else:
        logger.info("EDA 2 deshabilitado por configuración.")
        manifest["steps"]["eda2"] = {"status": "disabled"}

    # -------------------- Paso 3: select_pairs_and_donors + EDA 3 --------------------
    pairs_path = cfg.paths.preprocessed_dir / "pairs_windows.csv"
    donors_path = cfg.paths.preprocessed_dir / "donors_per_victim.csv"
    if cfg.toggles.step3:
        logger.info("Paso 3: Selección de episodios (pares i-j) y donantes.")
        try:
            sp_path = SRC_DIR / "preprocess_data" / "3. select_pairs_and_donors.py"
            if sp_path.exists():
                sp_mod = _dynamic_import("select_pairs_and_donors", sp_path)
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
                items_csv=str(cfg.paths.items_csv) if cfg.paths.items_csv else None,
                stores_csv=str(cfg.paths.stores_csv) if cfg.paths.stores_csv else None,
                outdir=str(cfg.params.outdir_pairs_donors),
            )
            # Si la función devuelve rutas, usarlas
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                pairs_path = Path(out[0])
                donors_path = Path(out[1])
        else:
            logger.warning("Paso 3 omitido (función no disponible).")

        manifest["steps"]["step3"] = {
            "status": "ok" if pairs_path.exists() and donors_path.exists() else "skipped_or_failed",
            "outputs": {"pairs_path": str(pairs_path), "donors_path": str(donors_path)},
        }
    else:
        logger.info("Paso 3 deshabilitado por configuración.")
        manifest["steps"]["step3"] = {"status": "disabled"}

    # EDA 3 (muestras de episodios y calidad)
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
                quality_cfg=None,  # usar defaults del módulo
            )
            manifest["steps"]["eda3"] = {"status": "ok", "figures_dir": str(cfg.paths.figures_dir)}
    else:
        logger.info("EDA 3 deshabilitado por configuración.")
        manifest["steps"]["eda3"] = {"status": "disabled"}

    # -------------------- Paso 4: pre_algorithm + EDA 4 --------------------
    processed_out_dir = cfg.paths.processed_dir
    episodes_path_for_step4 = pairs_path
    donors_path_for_step4 = donors_path

    if cfg.toggles.step4:
        logger.info("Paso 4: Preparación de datasets para GSC y Meta-learners.")
        try:
            pa_path = SRC_DIR / "preprocess_data" / "4. pre_algorithm.py"
            if pa_path.exists():
                pa_mod = _dynamic_import("pre_algorithm", pa_path)
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
            # Construir la configuración del pre_algorithm a partir de nuestro YAML
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
        else:
            logger.warning("Paso 4 omitido (funciones no disponibles).")
            manifest["steps"]["step4"] = {"status": "skipped_or_failed"}
    else:
        logger.info("Paso 4 deshabilitado por configuración.")
        manifest["steps"]["step4"] = {"status": "disabled"}

    # EDA 4 (resumen de artefactos procesados)
    if cfg.toggles.eda4:
        logger.info("EDA 4: Reporte de datasets procesados.")
        try:
            from eda_4 import EDAConfig, run as eda4_run  # type: ignore
        except Exception:
            try:
                from EDA.eda_4 import EDAConfig, run as eda4_run  # type: ignore
            except Exception:
                eda4_run = None  # type: ignore
                EDAConfig = None  # type: ignore

        if eda4_run is None or EDAConfig is None:
            logger.warning("EDA 4 no disponible (run/EDAConfig no encontrados).")
            manifest["steps"]["eda4"] = {"status": "skipped_or_failed"}
        else:
            # Resolver rutas esperadas por el EDA 4
            episodes_index = processed_out_dir / "episodes_index.parquet"
            donor_quality = processed_out_dir / "gsc" / "donor_quality.parquet"
            meta_all = processed_out_dir / "meta" / "all_units.parquet"
            panel_features = processed_out_dir / "intermediate" / "panel_features.parquet"
            gsc_dir = processed_out_dir / "gsc"

            eda4_cfg = EDAConfig(
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
    else:
        logger.info("EDA 4 deshabilitado por configuración.")
        manifest["steps"]["eda4"] = {"status": "disabled"}

    # Guardar manifest de ejecución
    manifest["end_time"] = datetime.now().isoformat()
    manifest_path = DIAG_DIR / "pipeline_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("📄 Manifest de ejecución guardado en: %s", str(manifest_path))

    return manifest


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