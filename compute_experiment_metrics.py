import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        try:
            if p.exists():
                return p
        except Exception:
            pass
    return None


def _read_parquet(p: Path) -> Optional[pd.DataFrame]:
    try:
        if p and p.exists():
            return pd.read_parquet(p)
    except Exception:
        return None
    return None


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _episodes_index_candidates(exp_tag: str) -> List[Path]:
    return [
        Path(".data/processed_data") / exp_tag / "episodes_index.parquet",
        Path(".data/processed") / exp_tag / "episodes_index.parquet",
        Path("data/processed_data") / exp_tag / "episodes_index.parquet",
        Path("data/processed") / exp_tag / "episodes_index.parquet",
    ]


def _load_causal_dir(dir_path: Path, model_type: str) -> Optional[pd.DataFrame]:
    try:
        cdir = dir_path / "causal_metrics"
        if not cdir.exists():
            return None
        rows: List[pd.DataFrame] = []
        for p in sorted(cdir.glob("*.parquet")):
            df = _read_parquet(p)
            if df is None or df.empty:
                continue
            if "model_type" not in df.columns:
                df["model_type"] = model_type
            rows.append(df)
        if not rows:
            return None
        out = pd.concat(rows, axis=0, ignore_index=True)
        if "episode_id" in out.columns:
            out["episode_id"] = out["episode_id"].astype(str)
        return out
    except Exception:
        return None


def load_gsc(exp_tag: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    base = Path(".data/processed_data") / exp_tag
    alt = Path(".data/processed") / exp_tag
    gsc_dir = _first_existing([base / "gsc", alt / "gsc"]) or base / "gsc"
    metr_path = _first_existing([
        gsc_dir / "gsc_metrics.parquet",
        base / "gsc_metrics.parquet",
        base / "metrics" / "gsc_metrics.parquet",
    ])
    gdf = _read_parquet(metr_path) if metr_path else None
    if gdf is not None and not gdf.empty:
        gdf = gdf.copy()
        gdf["episode_id"] = gdf["episode_id"].astype(str)
        gdf["model_type"] = "gsc"
        if "rel_att_vs_pre_mean" in gdf.columns and "rel_att" not in gdf.columns:
            gdf["rel_att"] = gdf["rel_att_vs_pre_mean"]
    cdf = _load_causal_dir(gsc_dir, model_type="gsc")
    return gdf, cdf


def load_meta(exp_tag: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    meta_root = Path(".data/processed_data") / "meta_outputs" / exp_tag
    alt_root = Path(".data/processed") / "meta_outputs" / exp_tag
    root = meta_root if meta_root.exists() else alt_root
    out_metrics: Dict[str, pd.DataFrame] = {}
    out_causal: Dict[str, pd.DataFrame] = {}
    if not root.exists():
        return out_metrics, out_causal
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        learner = sub.name.lower()
        metr_path = sub / f"meta_metrics_{learner}.parquet"
        df = _read_parquet(metr_path)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["episode_id"] = df["episode_id"].astype(str)
        df["model_type"] = f"meta-{learner}"
        if "rel_att_vs_pre_mean" in df.columns and "rel_att" not in df.columns:
            df["rel_att"] = df["rel_att_vs_pre_mean"]
        out_metrics[learner] = df
        cdf = _load_causal_dir(sub, model_type=f"meta-{learner}")
        if cdf is not None and not cdf.empty:
            out_causal[learner] = cdf
    return out_metrics, out_causal


def _merge_with_causal(mdf: pd.DataFrame, cdf: Optional[pd.DataFrame]) -> pd.DataFrame:
    if mdf is None or mdf.empty:
        return mdf
    if cdf is None or cdf.empty:
        return mdf
    left = mdf.copy()
    right = cdf.copy()
    keep_cols = [
        "episode_id","model_type","n_pre_periods","n_post_periods","n_control_units",
        "pred_rmspe_pre","pred_mae_pre","pred_corr_pre","pred_r2_pre",
        "het_tau_std","het_tau_cv","het_tau_median","het_pct_positive",
        "plac_p_value_space","plac_p_value_time","plac_effect_ratio","plac_n_space",
        "bal_mean_abs_std_diff","bal_max_abs_std_diff","bal_n_imbalanced","bal_rate",
    ]
    right = right[[c for c in keep_cols if c in right.columns]].drop_duplicates("episode_id", keep="last")
    out = left.merge(right, on="episode_id", how="left", suffixes=("", "_causal"))
    return out


def _coverage(exp_tag: str, gdf: Optional[pd.DataFrame], meta_map: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    epi_path = _first_existing(_episodes_index_candidates(exp_tag))
    expected = set()
    try:
        if epi_path and epi_path.exists():
            edf = pd.read_parquet(epi_path)
            if "episode_id" in edf.columns:
                expected = set(edf["episode_id"].astype(str).unique().tolist())
    except Exception:
        expected = set()
    cov: Dict[str, int] = {}
    cov["expected"] = len(expected)
    cov["gsc"] = 0
    if gdf is not None and not gdf.empty and "episode_id" in gdf.columns:
        cov["gsc"] = int(pd.Series(gdf["episode_id"].astype(str)).nunique())
    for k, df in meta_map.items():
        cov[f"meta_{k}"] = int(pd.Series(df["episode_id"].astype(str)).nunique()) if (df is not None and not df.empty) else 0
    cov["available"] = cov.get("gsc", 0) + sum(v for kk, v in cov.items() if kk.startswith("meta_"))
    if expected:
        cov["missing"] = max(0, cov["expected"] - cov["available"])
    else:
        cov["missing"] = 0
    return cov


def _save_tables(fig_tables: Path, tables: Dict[str, pd.DataFrame]) -> None:
    _ensure_dir(fig_tables)
    for name, df in tables.items():
        if df is None or df.empty:
            continue
        p_parq = fig_tables / f"{name}.parquet"
        p_csv = fig_tables / f"{name}.csv"
        try:
            df.to_parquet(p_parq, index=False)
        except Exception:
            pass
        try:
            df.to_csv(p_csv, index=False)
        except Exception:
            pass


def _summary_by_model(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    cols = [
        "rmspe_pre","mae_pre","att_mean","att_sum","rel_att",
        "pred_rmspe_pre","pred_mae_pre","pred_corr_pre","pred_r2_pre",
        "p_value_placebo_space","plac_p_value_space",
        "het_tau_std","het_tau_cv","bal_mean_abs_std_diff","bal_rate",
    ]
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return None
    g = df.groupby("model_type")[avail]
    agg = g.agg(["median","mean","std","count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    return agg


def run(exp_tag: str) -> Dict[str, str]:
    gdf, gcdf = load_gsc(exp_tag)
    meta_map, meta_causal_map = load_meta(exp_tag)

    merged: List[pd.DataFrame] = []
    if gdf is not None and not gdf.empty:
        g_all = _merge_with_causal(gdf, gcdf)
        merged.append(g_all)
    for k, mdf in meta_map.items():
        cdf = meta_causal_map.get(k)
        merged.append(_merge_with_causal(mdf, cdf))

    combined = pd.concat([d for d in merged if d is not None and not d.empty], axis=0, ignore_index=True) if merged else pd.DataFrame()
    if not combined.empty and "episode_id" in combined.columns:
        combined["episode_id"] = combined["episode_id"].astype(str)

    cov = _coverage(exp_tag, gdf, meta_map)

    fig_tables = Path("figures") / exp_tag / "tables"
    tables: Dict[str, pd.DataFrame] = {}
    if gdf is not None and not gdf.empty:
        tables["gsc_metrics_by_episode"] = gdf
    for k, mdf in meta_map.items():
        if mdf is not None and not mdf.empty:
            tables[f"meta_metrics_{k}_by_episode"] = mdf
    if gcdf is not None and not gcdf.empty:
        tables["causal_metrics_gsc_by_episode"] = gcdf
    for k, cdf in meta_causal_map.items():
        if cdf is not None and not cdf.empty:
            tables[f"causal_metrics_meta_{k}_by_episode"] = cdf
    if combined is not None and not combined.empty:
        tables["metrics_combined_by_episode"] = combined
    summary = _summary_by_model(combined)
    if summary is not None and not summary.empty:
        tables["metrics_summary_by_model"] = summary

    _save_tables(fig_tables, tables)

    cov_path = _ensure_dir(fig_tables) / "coverage_summary.json"
    try:
        cov_path.write_text(json.dumps(cov, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    out = {"figures_tables_dir": str(fig_tables)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_tag", required=True)
    args = ap.parse_args()
    out = run(args.exp_tag)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
