import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------- Helpers ----------------------------

def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        try:
            if p.exists():
                return p
        except Exception:
            pass
    return None


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_parquet(p: Optional[Path]) -> Optional[pd.DataFrame]:
    try:
        if p and p.exists():
            return pd.read_parquet(p)
    except Exception:
        return None
    return None


def _episodes_index_candidates(exp_tag: str) -> List[Path]:
    return [
        Path(".data/processed_data") / exp_tag / "episodes_index.parquet",
        Path(".data/processed") / exp_tag / "episodes_index.parquet",
        Path("data/processed_data") / exp_tag / "episodes_index.parquet",
        Path("data/processed") / exp_tag / "episodes_index.parquet",
    ]


def _gsc_root(exp_tag: str) -> Path:
    base = Path(".data/processed_data") / exp_tag / "gsc"
    alt = Path(".data/processed") / exp_tag / "gsc"
    return base if base.exists() else alt


def _meta_root(exp_tag: str) -> Path:
    base = Path(".data/processed_data") / "meta_outputs" / exp_tag
    alt = Path(".data/processed") / "meta_outputs" / exp_tag
    return base if base.exists() else alt


def _read_base_metrics(path: Path, id_col: str = "episode_id", att_col: str = "att_sum") -> Dict[str, float]:
    df = _read_parquet(path)
    if df is None or df.empty or id_col not in df.columns:
        return {}
    s = df[[id_col, att_col]] if att_col in df.columns else df[[id_col]]
    s = s.copy()
    s[id_col] = s[id_col].astype(str)
    if att_col not in s.columns:
        s[att_col] = np.nan
    return {str(r[id_col]): float(r[att_col]) for _, r in s.iterrows()}


def _extract_series(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            try:
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                return s
            except Exception:
                continue
    # Fallback: primera columna numérica
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        return pd.to_numeric(df[num_cols[0]], errors="coerce").dropna()
    return None


def _sens_episode_stats(att_real: Optional[float], ser: Optional[pd.Series]) -> Dict[str, float]:
    out = {
        "sens_n": 0,
        "sens_mean": np.nan,
        "sens_sd": np.nan,
        "sens_q05": np.nan,
        "sens_q50": np.nan,
        "sens_q95": np.nan,
        "sens_min": np.nan,
        "sens_max": np.nan,
        "sens_range": np.nan,
        "sens_relative_sd": np.nan,
        "sens_sign_flip_rate": np.nan,
        "sens_abs_mean": np.nan,
        "sens_abs_q95": np.nan,
        "sens_robust_ratio_abs": np.nan,  # |ATT_real| / mean(|ATT_sens|)
    }
    if ser is None or ser.empty:
        return out
    x = ser.astype(float).to_numpy()
    out["sens_n"] = int(x.size)
    out["sens_mean"] = float(np.nanmean(x))
    out["sens_sd"] = float(np.nanstd(x, ddof=1)) if x.size > 1 else 0.0
    out["sens_q05"] = float(np.nanquantile(x, 0.05))
    out["sens_q50"] = float(np.nanquantile(x, 0.50))
    out["sens_q95"] = float(np.nanquantile(x, 0.95))
    out["sens_min"] = float(np.nanmin(x))
    out["sens_max"] = float(np.nanmax(x))
    out["sens_range"] = float(out["sens_max"] - out["sens_min"]) if np.isfinite(out["sens_max"]) and np.isfinite(out["sens_min"]) else np.nan
    ax = np.abs(x)
    out["sens_abs_mean"] = float(np.nanmean(ax))
    out["sens_abs_q95"] = float(np.nanquantile(ax, 0.95))
    if att_real is not None and np.isfinite(att_real):
        denom = max(1e-8, abs(att_real))
        out["sens_relative_sd"] = float(out["sens_sd"] / denom) if np.isfinite(out["sens_sd"]) else np.nan
        flips = np.sum(np.sign(x) != np.sign(att_real))
        out["sens_sign_flip_rate"] = float(flips / x.size)
        if out["sens_abs_mean"] > 1e-8:
            out["sens_robust_ratio_abs"] = float(abs(att_real) / out["sens_abs_mean"])
    return out


# ---------------------------- Loaders ----------------------------

def load_gsc_sensitivity(exp_tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = _gsc_root(exp_tag)
    metr_path = _first_existing([
        root / "gsc_metrics.parquet",
        Path(".data/processed_data") / exp_tag / "gsc_metrics.parquet",
        Path(".data/processed_data") / exp_tag / "metrics" / "gsc_metrics.parquet",
    ])
    att_map = _read_base_metrics(metr_path)
    sens_dir = root / "sensitivity"

    rows: List[Dict] = []
    draws_rows: List[Dict] = []
    for ep, att_real in att_map.items():
        sens_df = _read_parquet(sens_dir / f"{ep}_sens.parquet")
        ser = _extract_series(sens_df, ["att_sum", "att"])
        stats = _sens_episode_stats(att_real, ser)
        row = {
            "episode_id": ep,
            "model_type": "gsc",
            "att_sum": float(att_real) if att_real is not None and np.isfinite(att_real) else np.nan,
            **stats,
        }
        rows.append(row)
        if sens_df is not None and not sens_df.empty:
            # Guardar draws largos
            s_long = ser if ser is not None else pd.Series(dtype=float)
            for v in s_long.to_numpy(dtype=float):
                draws_rows.append({"episode_id": ep, "model_type": "gsc", "att_draw": float(v)})

    summary_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["episode_id", "model_type"]) 
    draws_df = pd.DataFrame(draws_rows) if draws_rows else pd.DataFrame(columns=["episode_id", "model_type", "att_draw"]) 
    return summary_df, draws_df


def load_meta_sensitivity(exp_tag: str) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    root = _meta_root(exp_tag)
    out: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    if not root.exists():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        learner = sub.name.lower()
        metr_path = sub / f"meta_metrics_{learner}.parquet"
        att_map = _read_base_metrics(metr_path)
        sens_dir = sub / "sensitivity"

        rows: List[Dict] = []
        draws_rows: List[Dict] = []
        for ep, att_real in att_map.items():
            sens_df = _read_parquet(sens_dir / f"{ep}_sens.parquet")
            ser = _extract_series(sens_df, ["att_sum", "att"])  # soporta variantes
            stats = _sens_episode_stats(att_real, ser)
            row = {
                "episode_id": ep,
                "model_type": f"meta-{learner}",
                "att_sum": float(att_real) if att_real is not None and np.isfinite(att_real) else np.nan,
                **stats,
            }
            rows.append(row)
            if sens_df is not None and not sens_df.empty:
                s_long = ser if ser is not None else pd.Series(dtype=float)
                for v in s_long.to_numpy(dtype=float):
                    draws_rows.append({"episode_id": ep, "model_type": f"meta-{learner}", "att_draw": float(v)})

        summary_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["episode_id", "model_type"]) 
        draws_df = pd.DataFrame(draws_rows) if draws_rows else pd.DataFrame(columns=["episode_id", "model_type", "att_draw"]) 
        out[learner] = (summary_df, draws_df)
    return out


# ---------------------------- Main runner ----------------------------

def run(exp_tag: str) -> Dict[str, str]:
    gsc_summary, gsc_draws = load_gsc_sensitivity(exp_tag)
    meta_map = load_meta_sensitivity(exp_tag)

    # Combinar summaries y draws
    summaries = []
    draws = []
    if gsc_summary is not None and not gsc_summary.empty:
        summaries.append(gsc_summary)
    if gsc_draws is not None and not gsc_draws.empty:
        draws.append(gsc_draws)
    for k, (sdf, ddf) in meta_map.items():
        if sdf is not None and not sdf.empty:
            summaries.append(sdf)
        if ddf is not None and not ddf.empty:
            draws.append(ddf)

    sum_df = pd.concat(summaries, axis=0, ignore_index=True) if summaries else pd.DataFrame()
    drw_df = pd.concat(draws, axis=0, ignore_index=True) if draws else pd.DataFrame()

    if not sum_df.empty and "episode_id" in sum_df.columns:
        sum_df["episode_id"] = sum_df["episode_id"].astype(str)
    if not drw_df.empty and "episode_id" in drw_df.columns:
        drw_df["episode_id"] = drw_df["episode_id"].astype(str)

    # Resumen global por modelo
    def _agg_summary(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        cols = [
            "sens_n","sens_sd","sens_range","sens_relative_sd","sens_sign_flip_rate",
            "sens_abs_mean","sens_abs_q95","sens_robust_ratio_abs",
        ]
        avail = [c for c in cols if c in df.columns]
        if not avail:
            return None
        g = df.groupby("model_type")[avail]
        agg = g.agg(["median","mean","std","count"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg = agg.reset_index()
        return agg

    summary_by_model = _agg_summary(sum_df)

    # Cobertura (sensibilidad disponible vs episodios esperados)
    epi_path = _first_existing(_episodes_index_candidates(exp_tag))
    expected = set()
    try:
        if epi_path and epi_path.exists():
            edf = pd.read_parquet(epi_path)
            if "episode_id" in edf.columns:
                expected = set(edf["episode_id"].astype(str).unique().tolist())
    except Exception:
        expected = set()

    cov = {
        "expected": len(expected),
        "gsc": int(gsc_summary["episode_id"].nunique()) if gsc_summary is not None and not gsc_summary.empty and "episode_id" in gsc_summary.columns else 0,
    }
    for k, (sdf, _) in meta_map.items():
        cov[f"meta_{k}"] = int(sdf["episode_id"].nunique()) if sdf is not None and not sdf.empty and "episode_id" in sdf.columns else 0
    cov["available"] = cov.get("gsc", 0) + sum(v for kk, v in cov.items() if kk.startswith("meta_"))
    cov["missing"] = max(0, cov["expected"] - cov["available"]) if expected else 0

    # Guardado
    fig_tables = Path("figures") / exp_tag / "tables"
    _ensure_dir(fig_tables)

    if gsc_summary is not None and not gsc_summary.empty:
        try:
            gsc_summary.to_parquet(fig_tables / "sensitivity_gsc_by_episode.parquet", index=False)
        except Exception:
            pass
        try:
            gsc_summary.to_csv(fig_tables / "sensitivity_gsc_by_episode.csv", index=False)
        except Exception:
            pass

    for k, (sdf, _) in meta_map.items():
        if sdf is None or sdf.empty:
            continue
        try:
            sdf.to_parquet(fig_tables / f"sensitivity_meta_{k}_by_episode.parquet", index=False)
        except Exception:
            pass
        try:
            sdf.to_csv(fig_tables / f"sensitivity_meta_{k}_by_episode.csv", index=False)
        except Exception:
            pass

    if not sum_df.empty:
        try:
            sum_df.to_parquet(fig_tables / "sensitivity_combined_by_episode.parquet", index=False)
        except Exception:
            pass
        try:
            sum_df.to_csv(fig_tables / "sensitivity_combined_by_episode.csv", index=False)
        except Exception:
            pass

    if not drw_df.empty:
        # Para tamaños grandes, preferir parquet
        try:
            drw_df.to_parquet(fig_tables / "sensitivity_draws_combined.parquet", index=False)
        except Exception:
            pass

    if summary_by_model is not None and not summary_by_model.empty:
        try:
            summary_by_model.to_parquet(fig_tables / "sensitivity_summary_by_model.parquet", index=False)
        except Exception:
            pass
        try:
            summary_by_model.to_csv(fig_tables / "sensitivity_summary_by_model.csv", index=False)
        except Exception:
            pass

    cov_path = fig_tables / "sensitivity_coverage_summary.json"
    try:
        cov_path.write_text(json.dumps(cov, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {"figures_tables_dir": str(fig_tables)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_tag", required=True, help="Nombre del experimento (exp_tag)")
    args = ap.parse_args()
    out = run(args.exp_tag)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
