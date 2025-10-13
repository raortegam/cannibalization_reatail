# run_preprocessors.py
# -*- coding: utf-8 -*-
"""
Runner para preparar datasets de GSC y Meta-learners a partir de:
  - panel_labeled
  - episodes
  - windows_long

Uso CLI:
--------
python -m src.preprocessing.run_preprocessors \
  --panel-labeled data/interim/panel_labeled.parquet \
  --episodes     data/interim/episodes.parquet \
  --windows-long data/interim/windows_long.parquet \
  --outdir       data/processed/prepared \
  --outcome-col  y_isw \
  --treat-col    treatment_used \
  --encoding     ordinal \
  --save-episodes-data

También puedes importarlo y usar `prepare_all(...)` desde código.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocess_data.pre_gsc import prepare_gsc_datasets  # type: ignore
from src.preprocess_data.pre_meta_learners import prepare_ml_datasets  # type: ignore


# =========================
# Utilidades de IO
# =========================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _to_jsonable(obj: Any) -> Any:
    """Convierte objetos numpy/pandas a nativos JSON (list, float, int, str, dict)."""
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if isinstance(obj, (np.ndarray, )):
        return obj.tolist()
    if isinstance(obj, (pd.Index, )):
        return obj.tolist()
    if isinstance(obj, (pd.Series, )):
        return obj.to_dict()
    if isinstance(obj, (pd.Timestamp, )):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

def _load_df(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".feather", ".ft"):
        return pd.read_feather(path)
    if suf in (".csv",):
        return pd.read_csv(path, parse_dates=["week_start"], infer_datetime_format=True)
    if suf in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"Formato no soportado para {path}")

def _save_df(df: pd.DataFrame, path: Path) -> None:
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            _ensure_dir(path.parent)
            df.to_parquet(path, index=True)
            return
        if path.suffix.lower() in (".feather", ".ft"):
            _ensure_dir(path.parent)
            df.reset_index(drop=False).to_feather(path)
            return
        if path.suffix.lower() in (".pkl", ".pickle"):
            _ensure_dir(path.parent)
            df.to_pickle(path)
            return
    except Exception as e:
        print(f"[WARN] No se pudo escribir {path.name} como {path.suffix}: {e}. Guardo CSV.", flush=True)
    # Fallback CSV
    _ensure_dir(path.parent)
    df.to_csv(path.with_suffix(".csv"))

def _save_json(obj: Any, path: Path) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)

def _save_pickle(obj: Any, path: Path) -> None:
    import pickle
    _ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# =========================
# API de alto nivel
# =========================
def run_gsc(
    panel_labeled: pd.DataFrame,
    episodes: pd.DataFrame,
    windows_long: pd.DataFrame,
    outdir: Path,
    outcome_col: str = "y_isw",
    controls_cols: Optional[List[str]] = None,
    min_pre_weeks: Optional[int] = None,
    save_episodes_data: bool = False,
) -> Dict[str, Any]:
    """
    Ejecuta prepare_gsc_datasets y persiste artefactos en 'outdir/gsc'.
    Devuelve un diccionario con rutas a los artefactos guardados.
    """
    gsc_dir = outdir / "gsc"
    _ensure_dir(gsc_dir)

    # Preparación
    gsc_global, episodes_data = prepare_gsc_datasets(
        panel_labeled=panel_labeled,
        episodes=episodes,
        windows_long=windows_long,
        outcome_col=outcome_col,
        controls_cols=controls_cols,
        min_pre_weeks=min_pre_weeks or 8,
    )

    # ---- Guardar artefactos globales
    Y: pd.DataFrame = gsc_global["Y"]  # unidades x w_idx
    T: pd.DataFrame = gsc_global["T"]
    unit_meta: pd.DataFrame = gsc_global["unit_meta"]
    week_map: pd.Series = gsc_global["week_start_map"]

    _save_df(Y, gsc_dir / "Y.parquet")
    _save_df(T, gsc_dir / "T.parquet")
    _save_df(unit_meta, gsc_dir / "unit_meta.parquet")

    week_map_df = week_map.reset_index().rename(columns={"index": "w_idx"})
    _save_df(week_map_df, gsc_dir / "week_map.parquet")

    # Controles
    controls_cube = gsc_global.get("controls_cube", None)
    controls_written = []
    if isinstance(controls_cube, dict):
        for k, v in controls_cube.items():
            if isinstance(v, pd.DataFrame):
                _save_df(v, gsc_dir / f"control_{k}.parquet")
                controls_written.append(k)

    # Config usada
    _save_json(gsc_global.get("config_used", {}), gsc_dir / "config_used.json")

    # ---- Guardar episodios (opcional: pesado)
    episodes_paths: List[str] = []
    if save_episodes_data:
        epi_root = gsc_dir / "episodes"
        for i, ed in enumerate(episodes_data, start=1):
            meta = ed.get("episode_meta", {})
            keys = ed.get("treated_unit", ("_", "_"))
            ep_id = meta.get("episode_id", i)
            ep_dir = epi_root / f"ep_{keys[0]}-{keys[1]}_{ep_id}"
            _ensure_dir(ep_dir)

            # Submatrices
            for part in ["Y_pre", "Y_treat", "Y_post"]:
                dfp = ed.get(part, None)
                if isinstance(dfp, pd.DataFrame):
                    _save_df(dfp, ep_dir / f"{part}.parquet")

            # Controles por ventana
            for part in ["X_pre", "X_treat", "X_post"]:
                cube = ed.get(part, None)
                if isinstance(cube, dict):
                    subdir = ep_dir / part
                    for k, v in cube.items():
                        if isinstance(v, pd.DataFrame):
                            _save_df(v, subdir / f"{k}.parquet")

            # Metadatos y donantes
            meta_out = {k: _to_jsonable(v) for k, v in meta.items()}
            meta_out["donor_units"] = [tuple(u) for u in ed.get("donor_units", [])]
            meta_out["pre_idx"] = ed.get("pre_idx", [])
            meta_out["treat_idx"] = ed.get("treat_idx", [])
            meta_out["post_idx"] = ed.get("post_idx", [])
            _save_json(meta_out, ep_dir / "meta.json")
            episodes_paths.append(str(ep_dir))

    # Resumen
    summary = {
        "Y": str(gsc_dir / "Y.parquet"),
        "T": str(gsc_dir / "T.parquet"),
        "unit_meta": str(gsc_dir / "unit_meta.parquet"),
        "week_map": str(gsc_dir / "week_map.parquet"),
        "controls_written": controls_written,
        "episodes_saved_dirs": episodes_paths,
        "config_used_path": str(gsc_dir / "config_used.json"),
    }
    _save_json(summary, gsc_dir / "summary.json")
    return summary


def run_ml(
    panel_labeled: pd.DataFrame,
    episodes: pd.DataFrame,
    windows_long: pd.DataFrame,
    outdir: Path,
    outcome_col: str = "y_isw",
    treat_col: str = "treatment_used",
    encoding_method: str = "ordinal",
    exclude_roles_for_train: Optional[Iterable[str]] = ("guard_pre", "guard_post"),
    include_roles_for_train: Optional[Iterable[str]] = None,
    drop_na: bool = True,
) -> Dict[str, Any]:
    """
    Ejecuta prepare_ml_datasets y persiste artefactos en 'outdir/ml'.
    Devuelve un diccionario con rutas a los artefactos guardados.
    """
    ml_dir = outdir / "ml"
    _ensure_dir(ml_dir)

    ml_global, t_views, x_proto = prepare_ml_datasets(
        panel_labeled=panel_labeled,
        episodes=episodes,
        windows_long=windows_long,
        outcome_col=outcome_col,
        treat_col=treat_col,
        encoding_method=encoding_method,
        exclude_roles_for_train=tuple(exclude_roles_for_train) if exclude_roles_for_train else None,
        include_roles_for_train=tuple(include_roles_for_train) if include_roles_for_train else None,
        drop_na=drop_na,
    )

    # ml_df base + features
    ml_df: pd.DataFrame = ml_global["ml_df"]
    _save_df(ml_df, ml_dir / "ml_df.parquet")
    _save_json(ml_global.get("features", []), ml_dir / "features.json")
    _save_json(ml_global.get("encoders", {}), ml_dir / "encoders.json")
    _save_json(ml_global.get("folds", []), ml_dir / "folds.json")
    _save_json(ml_global.get("config_used", {}), ml_dir / "config_used.json")
    _save_json({"target_col": outcome_col, "treat_col": treat_col}, ml_dir / "targets.json")

    # T-Learner views
    treated_df: pd.DataFrame = t_views["treated"]
    control_df: pd.DataFrame = t_views["control"]
    _save_df(treated_df, ml_dir / "t_learner_treated.parquet")
    _save_df(control_df, ml_dir / "t_learner_control.parquet")

    # X-Learner proto
    _save_json(x_proto, ml_dir / "x_learner_proto.json")

    summary = {
        "ml_df": str(ml_dir / "ml_df.parquet"),
        "features": str(ml_dir / "features.json"),
        "encoders": str(ml_dir / "encoders.json"),
        "folds": str(ml_dir / "folds.json"),
        "t_learner_treated": str(ml_dir / "t_learner_treated.parquet"),
        "t_learner_control": str(ml_dir / "t_learner_control.parquet"),
        "x_learner_proto": str(ml_dir / "x_learner_proto.json"),
    }
    _save_json(summary, ml_dir / "summary.json")
    return summary


def prepare_all(
    panel_labeled: pd.DataFrame,
    episodes: pd.DataFrame,
    windows_long: pd.DataFrame,
    outdir: Path,
    outcome_col_gsc: str = "y_isw",
    controls_cols_gsc: Optional[List[str]] = None,
    min_pre_weeks_gsc: Optional[int] = None,
    outcome_col_ml: str = "y_isw",
    treat_col_ml: str = "treatment_used",
    encoding_method_ml: str = "ordinal",
    exclude_roles_for_train_ml: Optional[Iterable[str]] = ("guard_pre", "guard_post"),
    include_roles_for_train_ml: Optional[Iterable[str]] = None,
    drop_na_ml: bool = True,
    save_episodes_data: bool = False,
) -> Dict[str, Any]:
    """
    API de conveniencia para correr GSC + ML y guardar artefactos bajo outdir.
    """
    outdir = Path(outdir)
    _ensure_dir(outdir)

    gsc_summary = run_gsc(
        panel_labeled, episodes, windows_long, outdir,
        outcome_col=outcome_col_gsc,
        controls_cols=controls_cols_gsc,
        min_pre_weeks=min_pre_weeks_gsc,
        save_episodes_data=save_episodes_data,
    )
    ml_summary = run_ml(
        panel_labeled, episodes, windows_long, outdir,
        outcome_col=outcome_col_ml,
        treat_col=treat_col_ml,
        encoding_method=encoding_method_ml,
        exclude_roles_for_train=exclude_roles_for_train_ml,
        include_roles_for_train=include_roles_for_train_ml,
        drop_na=drop_na_ml,
    )
    return {"gsc": gsc_summary, "ml": ml_summary}


# =========================
# CLI
# =========================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Runner de preprocesamiento para GSC y Meta-learners.")
    p.add_argument("--panel-labeled", required=True, type=str, help="Ruta a panel_labeled (parquet/csv/feather/pkl)")
    p.add_argument("--episodes", required=True, type=str, help="Ruta a episodes (parquet/csv/feather/pkl)")
    p.add_argument("--windows-long", required=True, type=str, help="Ruta a windows_long (parquet/csv/feather/pkl)")
    p.add_argument("--outdir", required=True, type=str, help="Directorio de salida para artefactos")

    # GSC
    p.add_argument("--outcome-col", default="y_isw", type=str, help="Outcome para GSC y ML (por defecto y_isw)")
    p.add_argument("--controls-cols", default=None, type=str,
                   help="Lista separada por comas para controles GSC; usa 'auto' o vacía para inferir")
    p.add_argument("--min-pre-weeks", default=None, type=int, help="Mínimo de semanas pre para GSC")
    p.add_argument("--save-episodes-data", action="store_true", help="Guardar submatrices por episodio (pesado)")

    # ML
    p.add_argument("--treat-col", default="treatment_used", type=str, help="Columna de tratamiento para ML")
    p.add_argument("--encoding", default="ordinal", choices=["ordinal", "onehot"], help="Codificación categóricas para ML")
    p.add_argument("--exclude-roles", default="guard_pre,guard_post", type=str,
                   help="Roles a excluir en entrenamiento ML (coma-separado); deja vacío para no excluir")
    p.add_argument("--include-roles", default="", type=str,
                   help="Roles a incluir (coma-separado) si quieres restringir explícitamente")

    p.add_argument("--no-drop-na", action="store_true", help="No dropear NAs en features/objetivo para ML")

    return p.parse_args()

def main() -> None:
    args = _parse_args()
    panel_path = Path(args.panel_labeled)
    episodes_path = Path(args.episodes)
    windows_path = Path(args.windows_long)
    outdir = Path(args.outdir)

    print("[INFO] Cargando DataFrames…")
    panel_labeled = _load_df(panel_path)
    episodes = _load_df(episodes_path)
    windows_long = _load_df(windows_path)

    # Parse controls
    controls_cols = None
    if args.controls_cols is not None and args.controls_cols.strip():
        if args.controls_cols.strip().lower() not in ("auto", "none"):
            controls_cols = [c.strip() for c in args.controls_cols.split(",") if c.strip()]

    # Parse roles include/exclude
    exclude_roles = tuple([r.strip() for r in args.exclude_roles.split(",") if r.strip()]) \
        if args.exclude_roles is not None and args.exclude_roles.strip() else None
    include_roles = tuple([r.strip() for r in args.include_roles.split(",") if r.strip()]) \
        if args.include_roles is not None and args.include_roles.strip() else None

    print("[INFO] Ejecutando preparación GSC…")
    gsc_summary = run_gsc(
        panel_labeled=panel_labeled,
        episodes=episodes,
        windows_long=windows_long,
        outdir=outdir,
        outcome_col=args.outcome_col,
        controls_cols=controls_cols,
        min_pre_weeks=args.min_pre_weeks,
        save_episodes_data=args.save_episodes_data,
    )
    print("[OK] GSC artefactos guardados:", gsc_summary)

    print("[INFO] Ejecutando preparación Meta-learners…")
    ml_summary = run_ml(
        panel_labeled=panel_labeled,
        episodes=episodes,
        windows_long=windows_long,
        outdir=outdir,
        outcome_col=args.outcome_col,
        treat_col=args.treat_col,
        encoding_method=args.encoding,
        exclude_roles_for_train=exclude_roles,
        include_roles_for_train=include_roles,
        drop_na=not args.no_drop_na,
    )
    print("[OK] ML artefactos guardados:", ml_summary)

if __name__ == "__main__":
    main()