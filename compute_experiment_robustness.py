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


def _placebo_stats(att_real: Optional[float], ser: Optional[pd.Series]) -> Dict[str, float]:
    out = {
        "placebo_n": 0,
        "placebo_mean": np.nan,
        "placebo_std": np.nan,
        "placebo_q05": np.nan,
        "placebo_q50": np.nan,
        "placebo_q95": np.nan,
        "placebo_mean_abs": np.nan,
        "placebo_q95_abs": np.nan,
        "placebo_p_value_two_sided": np.nan,
        "effect_to_placebo_ratio_abs": np.nan,
    }
    if ser is None or ser.empty:
        return out
    x = ser.astype(float).to_numpy()
    out["placebo_n"] = int(x.size)
    out["placebo_mean"] = float(np.nanmean(x))
    out["placebo_std"] = float(np.nanstd(x, ddof=1)) if x.size > 1 else 0.0
    out["placebo_q05"] = float(np.nanquantile(x, 0.05))
    out["placebo_q50"] = float(np.nanquantile(x, 0.50))
    out["placebo_q95"] = float(np.nanquantile(x, 0.95))
    ax = np.abs(x)
    out["placebo_mean_abs"] = float(np.nanmean(ax))
    out["placebo_q95_abs"] = float(np.nanquantile(ax, 0.95))
    if att_real is not None and np.isfinite(att_real) and x.size > 0:
        n_ext = int(np.sum(np.abs(x) >= abs(att_real)))
        out["placebo_p_value_two_sided"] = float((1 + n_ext) / (1 + x.size))
        if out["placebo_mean_abs"] > 1e-8:
            out["effect_to_placebo_ratio_abs"] = float(abs(att_real) / out["placebo_mean_abs"])
    return out


def _loo_stats(att_real: Optional[float], ser: Optional[pd.Series]) -> Dict[str, float]:
    out = {
        "loo_n": 0,
        "loo_sd": np.nan,
        "loo_min": np.nan,
        "loo_max": np.nan,
        "loo_range": np.nan,
        "loo_sign_flip_rate": np.nan,
    }
    if ser is None or ser.empty:
        return out
    x = ser.astype(float).to_numpy()
    out["loo_n"] = int(x.size)
    out["loo_sd"] = float(np.nanstd(x, ddof=1)) if x.size > 1 else 0.0
    out["loo_min"] = float(np.nanmin(x))
    out["loo_max"] = float(np.nanmax(x))
    out["loo_range"] = float(out["loo_max"] - out["loo_min"]) if np.isfinite(out["loo_max"]) and np.isfinite(out["loo_min"]) else np.nan
    if att_real is not None and np.isfinite(att_real) and x.size > 0:
        flips = np.sum(np.sign(x) != np.sign(att_real))
        out["loo_sign_flip_rate"] = float(flips / x.size)
    return out


def _sens_stats(att_real: Optional[float], ser: Optional[pd.Series]) -> Dict[str, float]:
    out = {
        "sens_n": 0,
        "sens_sd": np.nan,
        "sens_min": np.nan,
        "sens_max": np.nan,
        "sens_range": np.nan,
        "sens_relative_std": np.nan,
    }
    if ser is None or ser.empty:
        return out
    x = ser.astype(float).to_numpy()
    out["sens_n"] = int(x.size)
    out["sens_sd"] = float(np.nanstd(x, ddof=1)) if x.size > 1 else 0.0
    out["sens_min"] = float(np.nanmin(x))
    out["sens_max"] = float(np.nanmax(x))
    out["sens_range"] = float(out["sens_max"] - out["sens_min"]) if np.isfinite(out["sens_max"]) and np.isfinite(out["sens_min"]) else np.nan
    if att_real is not None and np.isfinite(att_real) and out["sens_sd"] is not np.nan:
        denom = max(1e-8, abs(att_real))
        out["sens_relative_std"] = float(out["sens_sd"] / denom)
    return out


# ---------------------------- Loaders ----------------------------

def load_gsc_robustness(exp_tag: str) -> pd.DataFrame:
    root = _gsc_root(exp_tag)
    metr_path = _first_existing([
        root / "gsc_metrics.parquet",
        Path(".data/processed_data") / exp_tag / "gsc_metrics.parquet",
        Path(".data/processed_data") / exp_tag / "metrics" / "gsc_metrics.parquet",
    ])
    att_map = _read_base_metrics(metr_path)
    placebos_dir = root / "placebos"
    loo_dir = root / "loo"
    sens_dir = root / "sensitivity"

    rows: List[Dict] = []
    episodes = list(att_map.keys())
    for ep in episodes:
        att_real = att_map.get(ep, np.nan)
        # Placebo (espacio)
        ps = _read_parquet(placebos_dir / f"{ep}_space.parquet")
        ps_ser = _extract_series(ps, ["att_placebo_sum", "att_sum", "att", "effect_sum"])
        ps_stats = _placebo_stats(att_real, ps_ser)
        # Placebo (tiempo)
        pt = _read_parquet(placebos_dir / f"{ep}_time.parquet")
        pt_ser = _extract_series(pt, ["att_placebo_mean", "att_mean", "att_sum", "att"])  # no p-valor robusto
        time_mean = float(pt_ser.mean()) if pt_ser is not None and not pt_ser.empty else np.nan
        # LOO
        loo = _read_parquet(loo_dir / f"{ep}_loo.parquet")
        loo_ser = _extract_series(loo, ["att_sum", "att"])
        loo_stats = _loo_stats(att_real, loo_ser)
        # Sensibilidad
        sens = _read_parquet(sens_dir / f"{ep}_sens.parquet")
        sens_ser = _extract_series(sens, ["att_sum", "att"])
        sens_stats = _sens_stats(att_real, sens_ser)

        row = {
            "episode_id": ep,
            "model_type": "gsc",
            "att_sum": float(att_real) if att_real is not None and np.isfinite(att_real) else np.nan,
            "time_placebo_mean": time_mean,
            **ps_stats,
            **loo_stats,
            **sens_stats,
        }
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["episode_id", "model_type"]) 


def load_meta_robustness(exp_tag: str) -> Dict[str, pd.DataFrame]:
    root = _meta_root(exp_tag)
    out: Dict[str, pd.DataFrame] = {}
    if not root.exists():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        learner = sub.name.lower()
        metr_path = sub / f"meta_metrics_{learner}.parquet"
        att_map = _read_base_metrics(metr_path)
        placebos_dir = sub / "placebos"
        loo_dir = sub / "loo"
        sens_dir = sub / "sensitivity"

        rows: List[Dict] = []
        for ep, att_real in att_map.items():
            # Placebo (espacio)
            ps = _read_parquet(placebos_dir / f"{ep}_space.parquet")
            ps_ser = _extract_series(ps, ["att_placebo_sum", "att_sum", "att", "effect_sum"])
            ps_stats = _placebo_stats(att_real, ps_ser)
            # Placebo (tiempo)
            pt = _read_parquet(placebos_dir / f"{ep}_time.parquet")
            pt_ser = _extract_series(pt, ["att_placebo_mean", "att_mean", "att_sum", "att"])  # no p-valor robusto
            time_mean = float(pt_ser.mean()) if pt_ser is not None and not pt_ser.empty else np.nan
            # LOO (raro en meta, pero soportado si existe)
            loo = _read_parquet(loo_dir / f"{ep}_loo.parquet")
            loo_ser = _extract_series(loo, ["att_sum", "att"]) 
            loo_stats = _loo_stats(att_real, loo_ser)
            # Sensibilidad (raro en meta, pero soportado si existe)
            sens = _read_parquet(sens_dir / f"{ep}_sens.parquet")
            sens_ser = _extract_series(sens, ["att_sum", "att"]) 
            sens_stats = _sens_stats(att_real, sens_ser)

            row = {
                "episode_id": ep,
                "model_type": f"meta-{learner}",
                "att_sum": float(att_real) if att_real is not None and np.isfinite(att_real) else np.nan,
                "time_placebo_mean": time_mean,
                **ps_stats,
                **loo_stats,
                **sens_stats,
            }
            rows.append(row)
        out[learner] = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["episode_id", "model_type"]) 
    return out


# ---------------------------- Main runner ----------------------------

def run(exp_tag: str) -> Dict[str, str]:
    gsc_df = load_gsc_robustness(exp_tag)
    meta_map = load_meta_robustness(exp_tag)

    combined = []
    if gsc_df is not None and not gsc_df.empty:
        combined.append(gsc_df)
    for k, df in meta_map.items():
        if df is not None and not df.empty:
            combined.append(df)

    combined_df = pd.concat(combined, axis=0, ignore_index=True) if combined else pd.DataFrame()
    if not combined_df.empty and "episode_id" in combined_df.columns:
        combined_df["episode_id"] = combined_df["episode_id"].astype(str)

    # Cobertura
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
        "gsc": int(gsc_df["episode_id"].nunique()) if gsc_df is not None and not gsc_df.empty and "episode_id" in gsc_df.columns else 0,
    }
    for k, df in meta_map.items():
        cov[f"meta_{k}"] = int(df["episode_id"].nunique()) if df is not None and not df.empty and "episode_id" in df.columns else 0
    cov["available"] = cov.get("gsc", 0) + sum(v for kk, v in cov.items() if kk.startswith("meta_"))
    cov["missing"] = max(0, cov["expected"] - cov["available"]) if expected else 0

    # Guardado
    fig_tables = Path("figures") / exp_tag / "tables"
    _ensure_dir(fig_tables)

    # Archivos por algoritmo
    if gsc_df is not None and not gsc_df.empty:
        try:
            gsc_df.to_parquet(fig_tables / "robustness_gsc_by_episode.parquet", index=False)
        except Exception:
            pass
        try:
            gsc_df.to_csv(fig_tables / "robustness_gsc_by_episode.csv", index=False)
        except Exception:
            pass

    for k, df in meta_map.items():
        if df is None or df.empty:
            continue
        try:
            df.to_parquet(fig_tables / f"robustness_meta_{k}_by_episode.parquet", index=False)
        except Exception:
            pass
        try:
            df.to_csv(fig_tables / f"robustness_meta_{k}_by_episode.csv", index=False)
        except Exception:
            pass

    if combined_df is not None and not combined_df.empty:
        try:
            combined_df.to_parquet(fig_tables / "robustness_combined_by_episode.parquet", index=False)
        except Exception:
            pass
        try:
            combined_df.to_csv(fig_tables / "robustness_combined_by_episode.csv", index=False)
        except Exception:
            pass

    cov_path = fig_tables / "robustness_coverage_summary.json"
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
