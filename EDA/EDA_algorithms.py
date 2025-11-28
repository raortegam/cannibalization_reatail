# -*- coding: utf-8 -*-
from __future__ import annotations

"""
EDA_algorithms.py
=================

Exploración y render de resultados de:
  - src/models/synthetic_control.py (GSC)
  - src/models/meta_learners.py (T/S/X learners)

Novedades:
  - Orden canónico único de episodios entre métodos (misma lista y orden).
  - Suavizado opcional del observado para gráficos (EMA causal o MA).
  - Lectura robusta de artefactos y columnas alternativas.

Salidas (por defecto en ./figures):
  - Reportes PDF:
        figures/gsc_report.pdf
        figures/meta_<learner>_report.pdf
  - PNG por episodio:
        figures/gsc_episode_<episode_id>_timeseries.png
        figures/meta_<learner>_episode_<episode_id>_timeseries.png
  - Resúmenes:
        figures/gsc_overview_summary_*.png
        figures/meta_<learner>_overview_summary_*.png
  - Comparación:
        figures/compare_att_sum_gsc_vs_meta_<learner>.png

EDA de algoritmos (GSC + Meta-learners).

- Lee métricas consolidadas (si existen) para histogramas y dispersión.
- Lee series contrafactuales por episodio desde:
    * GSC:   <gsc_out_dir>/cf_series/*.parquet
    * Meta:  <meta_out_root>/<learner>/cf_series/*.parquet
- Renderiza:
    * PDF de resumen: figures/<exp_tag>/algorithms/eda_algorithms_summary.pdf
    * PDF con series por episodio: figures/<exp_tag>/algorithms/series_by_episode.pdf
    * PNG individuales: figures/<exp_tag>/algorithms/series_<SRC>_<episode_id>.png
- Compatibilidad: acepta alias 'gsc_dir' si 'gsc_out_dir' no viene.

Campos esperados en las series (se manejan sinónimos):
- fecha: 'date'
- observado: 'sales' | 'y_obs' | 'observed'
- contrafactual: 'mu0_hat' | 'y0_hat' | 'y_hat' | 'y_cf' | 'counterfactual'
- efecto (opcional): 'effect'
- efecto acumulado (opcional): 'cum_effect'

Si 'effect' no existe, se calcula como (observado - contrafactual).
Si no hay métricas en disco, se calculan por episodio con la ventana de episodes_index.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

# Matplotlib (sin seaborn)
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------

@dataclass
class EDAConfig:
    # Entrada obligatoria
    episodes_index: Path

    # Salidas de algoritmos
    gsc_out_dir: Optional[Path] = None          # preferido
    meta_out_root: Optional[Path] = None

    # Compatibilidad hacia atrás
    gsc_dir: Optional[Path] = None              # alias (se mapea a gsc_out_dir si viene)

    # Selección/estética
    meta_learners: Tuple[str, ...] = ("t", "s", "x")
    figures_dir: Path = Path("figures")
    orientation: str = "landscape"              # "landscape" | "portrait"
    dpi: int = 300
    style: str = "academic"
    font_size: int = 10
    grid: bool = True

    # Límites de episodios (para render)
    max_episodes_gsc: Optional[int] = None
    max_episodes_meta: Optional[int] = None

    # Exportación
    export_pdf: bool = True

    def __post_init__(self):
        # Normalizaciones
        self.episodes_index = Path(self.episodes_index)
        if self.meta_out_root is not None:
            self.meta_out_root = Path(self.meta_out_root)
        if self.figures_dir is not None:
            self.figures_dir = Path(self.figures_dir)

        # Alias backward‑compat
        if self.gsc_out_dir is None and self.gsc_dir is not None:
            self.gsc_out_dir = Path(self.gsc_dir)
        if self.gsc_out_dir is not None:
            self.gsc_out_dir = Path(self.gsc_out_dir)

        # Learners
        ml: List[str] = []
        for m in (self.meta_learners or ()):
            m = (m or "").strip().lower()
            if m in {"t", "s", "x"}:
                ml.append(m)
        self.meta_learners = tuple(ml) if ml else tuple()

        # Estilo básico
        plt.rcParams.update({
            "font.size": self.font_size,
            "axes.grid": self.grid,
        })


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _figsize_from_orientation(orientation: str) -> Tuple[float, float]:
    # Aprox A4
    return (11.69, 8.27) if str(orientation).lower().startswith("land") else (8.27, 11.69)


def _safe_read_parquet(p: Path, cols: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        if not p or not Path(p).exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        if cols:
            keep = [c for c in cols if c in df.columns]
            df = df[keep] if keep else pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _sorted_unique(a: List) -> List:
    return sorted(list(dict.fromkeys(a)))


# ---------------------------------------------------------------------
# Carga de métricas (GSC + Meta)
# ---------------------------------------------------------------------

def _load_gsc_metrics(gsc_out_dir: Optional[Path]) -> pd.DataFrame:
    if not gsc_out_dir:
        return pd.DataFrame()
    path = Path(gsc_out_dir) / "gsc_metrics.parquet"
    df = _safe_read_parquet(path)
    if df.empty:
        return df
    # Normalizar nombres
    ren = {}
    if "effect_sum" in df.columns and "att_sum" not in df.columns:
        ren["effect_sum"] = "att_sum"
    if "effect_mean" in df.columns and "att_mean" not in df.columns:
        ren["effect_mean"] = "att_mean"
    if ren:
        df = df.rename(columns=ren)
    keep = [c for c in ["episode_id", "rmspe_pre", "att_sum", "att_mean"] if c in df.columns]
    df = df[keep].copy()
    df["source"] = "gsc"
    return df


def _load_meta_metrics(meta_out_root: Optional[Path],
                       meta_learners: Tuple[str, ...]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not meta_out_root:
        return out
    base = Path(meta_out_root)
    for lr in meta_learners:
        path = base / lr / f"meta_metrics_{lr}.parquet"
        df = _safe_read_parquet(path)
        if df.empty:
            continue
        ren = {}
        if "effect_sum" in df.columns and "att_sum" not in df.columns:
            ren["effect_sum"] = "att_sum"
        if "effect_mean" in df.columns and "att_mean" not in df.columns:
            ren["effect_mean"] = "att_mean"
        if ren:
            df = df.rename(columns=ren)
        keep = [c for c in ["episode_id", "rmspe_pre", "att_sum", "att_mean", "p_value_placebo_space"] if c in df.columns]
        if not keep:
            continue
        df = df[keep].copy()
        df["source"] = f"meta-{lr.lower()}"
        out[lr] = df
    return out


# ---------------------------------------------------------------------
# Carga de series por episodio
# ---------------------------------------------------------------------

# Sinónimos de columnas
_OBS_COLS = ["sales", "y_obs", "observed", "Y", "y"]
_CF_COLS  = ["mu0_hat", "y0_hat", "y_hat", "y_cf", "counterfactual", "Y0", "cf"]
_EFF_COLS = ["effect", "tau_hat", "att", "delta"]
_CUM_COLS = ["cum_effect", "cum_att", "cum_delta"]

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _read_cf_file(p: Path) -> pd.DataFrame:
    """
    Lee un parquet de serie CF y estandariza columnas a:
    ['episode_id','unit_id','date','obs','cf','effect','cum_effect']
    (las que existan).
    """
    df = _safe_read_parquet(p)
    if df.empty:
        return df

    # Normalizar tipos básicos
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    ep_col = "episode_id" if "episode_id" in df.columns else None
    if ep_col is None:
        # intentar infiriendo desde el nombre del archivo
        ep_guess = p.stem.replace("_cf", "")
        df["episode_id"] = ep_guess
        ep_col = "episode_id"

    # Selección de columnas
    obs_c = _pick_col(df, _OBS_COLS)
    cf_c  = _pick_col(df, _CF_COLS)
    eff_c = _pick_col(df, _EFF_COLS)
    cum_c = _pick_col(df, _CUM_COLS)

    # Si no hay efecto, lo creamos si obs y cf existen
    if eff_c is None and obs_c and cf_c:
        df["__effect__"] = pd.to_numeric(df[obs_c], errors="coerce") - pd.to_numeric(df[cf_c], errors="coerce")
        eff_c = "__effect__"

    # Ensamblar salida
    keep = [c for c in [ep_col, "unit_id", "date", obs_c, cf_c, eff_c, cum_c] if c is not None and c in df.columns]
    out = df[keep].copy()
    ren = {}
    if obs_c and obs_c in out.columns: ren[obs_c] = "obs"
    if cf_c  and cf_c  in out.columns: ren[cf_c]  = "cf"
    if eff_c and eff_c in out.columns: ren[eff_c] = "effect"
    if cum_c and cum_c in out.columns: ren[cum_c] = "cum_effect"
    if ren:
        out = out.rename(columns=ren)

    # Tipos
    for c in ["obs", "cf", "effect", "cum_effect"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Ordenar y deduplicar por fecha
    if "date" in out.columns:
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    return out


def _load_cf_series_from_dir(cf_dir: Path,
                             episodes_allowed: Optional[set] = None,
                             cap: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Devuelve dict episode_id -> DataFrame estandarizado.
    """
    series: Dict[str, pd.DataFrame] = {}
    if not cf_dir.exists():
        return series

    # Buscar *.parquet (nombres libres)
    files = sorted(cf_dir.glob("*.parquet"))
    for p in files:
        df = _read_cf_file(p)
        if df.empty or "episode_id" not in df.columns:
            continue
        ep = str(df["episode_id"].iloc[0])
        if episodes_allowed and ep not in episodes_allowed:
            continue
        series[ep] = df
        if cap is not None and len(series) >= int(cap):
            break
    return series


# ---------------------------------------------------------------------
# Cálculo de métricas y gráficos
# ---------------------------------------------------------------------

def _rmspe(y: np.ndarray, yhat: np.ndarray) -> float:
    # Guardas numéricas
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() == 0:
        return np.nan
    e = y[m] - yhat[m]
    denom = np.sqrt(np.mean(np.square(y[m]))) if np.any(y[m] != 0) else 1.0
    denom = float(max(denom, 1e-8))
    return float(np.sqrt(np.mean(np.square(e))) / denom)


def _compute_metrics_for_window(df: pd.DataFrame,
                                pre_start: pd.Timestamp,
                                treat_start: pd.Timestamp,
                                post_end: pd.Timestamp) -> Dict[str, float]:
    """
    Calcula RMSPE(pre), ATT_sum y ATT_mean en la ventana del episodio.
    """
    d = df.copy()
    d = d[(d["date"] >= pre_start) & (d["date"] <= post_end)].copy()
    pre = d[d["date"] < treat_start]
    post = d[d["date"] >= treat_start]

    # Estimar cf si falta
    if "cf" not in d.columns and "effect" in d.columns and "obs" in d.columns:
        d["cf"] = d["obs"] - d["effect"]
        pre["cf"] = pre["obs"] - pre["effect"]  # type: ignore
        post["cf"] = post["obs"] - post["effect"]  # type: ignore

    rmspe_pre = _rmspe(pre["obs"].to_numpy(float), pre["cf"].to_numpy(float)) if {"obs", "cf"}.issubset(pre.columns) else np.nan

    if "effect" not in d.columns and {"obs", "cf"}.issubset(d.columns):
        d["effect"] = d["obs"] - d["cf"]
        post["effect"] = post["obs"] - post["cf"]  # type: ignore

    att_sum = float(np.nansum(post["effect"])) if "effect" in post.columns else np.nan
    att_mean = float(np.nanmean(post["effect"])) if "effect" in post.columns and len(post) > 0 else np.nan

    return {"rmspe_pre": rmspe_pre, "att_sum": att_sum, "att_mean": att_mean}


def _compute_sensitivity_windows(df: pd.DataFrame,
                                 treat_start: pd.Timestamp,
                                 post_end: pd.Timestamp) -> Dict[str, float]:
    d = df.copy()
    d = d[(d["date"] >= treat_start) & (d["date"] <= post_end)].sort_values("date")
    if "effect" not in d.columns and {"obs", "cf"}.issubset(d.columns):
        d["effect"] = pd.to_numeric(d["obs"], errors="coerce") - pd.to_numeric(d["cf"], errors="coerce")
    if "effect" not in d.columns:
        return {"att_30": np.nan, "att_60": np.nan, "att_90": np.nan, "sens_std": np.nan, "sens_rel_std": np.nan, "sens_range": np.nan, "sens_rel_range": np.nan}
    ts = pd.to_datetime(treat_start)
    def _sum_k(k: int) -> float:
        end_k = ts + pd.Timedelta(days=int(k) - 1)
        sub = d[(d["date"] >= ts) & (d["date"] <= end_k)]
        return float(np.nansum(pd.to_numeric(sub["effect"], errors="coerce"))) if not sub.empty else np.nan
    a30 = _sum_k(30)
    a60 = _sum_k(60)
    a90 = _sum_k(90)
    vec = np.array([a30, a60, a90], dtype=float)
    m = np.isfinite(vec)
    if not np.any(m):
        return {"att_30": a30, "att_60": a60, "att_90": a90, "sens_std": np.nan, "sens_rel_std": np.nan, "sens_range": np.nan, "sens_rel_range": np.nan}
    v = vec[m]
    std = float(np.nanstd(v))
    ref = float(np.nanmean(np.abs(v))) if np.isfinite(np.nanmean(np.abs(v))) else 0.0
    rel_std = float(std / (ref + 1e-8))
    rng = float(np.nanmax(v) - np.nanmin(v))
    rel_rng = float(rng / (np.nanmax(np.abs(v)) + 1e-8))
    return {"att_30": a30, "att_60": a60, "att_90": a90, "sens_std": std, "sens_rel_std": rel_std, "sens_range": rng, "sens_rel_range": rel_rng}


def _compute_effect_heterogeneity(df: pd.DataFrame,
                                  treat_start: pd.Timestamp,
                                  post_end: pd.Timestamp) -> Dict[str, float]:
    d = df.copy()
    d = d[(d["date"] >= treat_start) & (d["date"] <= post_end)].sort_values("date")
    if "effect" not in d.columns and {"obs", "cf"}.issubset(d.columns):
        d["effect"] = pd.to_numeric(d["obs"], errors="coerce") - pd.to_numeric(d["cf"], errors="coerce")
    if "effect" not in d.columns or d.empty:
        return {"het_tau_std": np.nan, "het_tau_cv": np.nan}
    e = pd.to_numeric(d["effect"], errors="coerce").to_numpy(dtype=float)
    std = float(np.nanstd(e))
    mu = float(np.nanmean(e))
    cv = float(std / (abs(mu) + 1e-8))
    return {"het_tau_std": std, "het_tau_cv": cv}


def _compute_balance_for_episode(ep_row: pd.Series,
                                 donors_df: pd.DataFrame,
                                 meta_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if meta_df is None or meta_df.empty:
        return {"bal_mean_abs_std_diff": np.nan, "bal_rate": np.nan}
    try:
        pre_start = pd.to_datetime(ep_row["pre_start"]) if "pre_start" in ep_row else None
        treat_start = pd.to_datetime(ep_row["treat_start"]) if "treat_start" in ep_row else None
        if pre_start is None or treat_start is None:
            return {"bal_mean_abs_std_diff": np.nan, "bal_rate": np.nan}
        if "unit_id" not in meta_df.columns and {"store_nbr", "item_nbr"}.issubset(meta_df.columns):
            meta_df = meta_df.copy()
            meta_df["unit_id"] = meta_df["store_nbr"].astype(str) + ":" + meta_df["item_nbr"].astype(str)
        if "date" in meta_df.columns:
            meta_df = meta_df.copy()
            meta_df["date"] = pd.to_datetime(meta_df["date"], errors="coerce")
        js = int(ep_row["j_store"]) if "j_store" in ep_row else None
        ji = int(ep_row["j_item"]) if "j_item" in ep_row else None
        victim_uid = f"{js}:{ji}" if js is not None and ji is not None else None
        if victim_uid is None or "unit_id" not in meta_df.columns or "date" not in meta_df.columns:
            return {"bal_mean_abs_std_diff": np.nan, "bal_rate": np.nan}
        donors = pd.DataFrame()
        if donors_df is not None and not donors_df.empty:
            ep_id = str(ep_row.get("episode_id", ""))
            try:
                donors = donors_df[donors_df.get("episode_id", "").astype(str) == ep_id]
            except Exception:
                donors = pd.DataFrame()
            if donors.empty and {"j_store", "j_item"}.issubset(donors_df.columns) and js is not None and ji is not None:
                donors = donors_df[(donors_df["j_store"] == js) & (donors_df["j_item"] == ji)]
        if donors.empty or not {"donor_store", "donor_item"}.issubset(donors.columns):
            return {"bal_mean_abs_std_diff": np.nan, "bal_rate": np.nan}
        donor_uids = (donors["donor_store"].astype(str) + ":" + donors["donor_item"].astype(str)).unique().tolist()
        vpre = meta_df[(meta_df["unit_id"] == victim_uid) & (meta_df["date"] < treat_start) & (meta_df["date"] >= pre_start)].copy()
        dpre = meta_df[(meta_df["unit_id"].isin(donor_uids)) & (meta_df["date"] < treat_start) & (meta_df["date"] >= pre_start)].copy()
        if vpre.empty or dpre.empty:
            return {"bal_mean_abs_std_diff": np.nan, "bal_rate": np.nan}
        num_cols = dpre.select_dtypes(include=[np.number]).columns.tolist()
        keep_cols = []
        for c in num_cols:
            if c in {"sales", "D", "treated_unit", "treated_time", "is_pre", "train_mask", "episode_id"}:
                continue
            if c in {"ADI", "CV2", "available_A", "zero_streak", "sc_hat"}:
                continue
            if c.startswith("promo_share") or c.startswith("class_index_excl"):
                continue
            keep_cols.append(c)
        if not keep_cols:
            return {"bal_mean_abs_std_diff": np.nan, "bal_rate": np.nan}
        vm = vpre[keep_cols].mean(numeric_only=True)
        dm = dpre[keep_cols].mean(numeric_only=True)
        vv = vpre[keep_cols].var(numeric_only=True, ddof=1).fillna(0.0)
        dv = dpre[keep_cols].var(numeric_only=True, ddof=1).fillna(0.0)
        pooled = np.sqrt(((vv + dv) / 2.0).replace(0.0, np.nan))
        smd = (vm - dm) / pooled
        smd = smd.replace([np.inf, -np.inf], np.nan).abs()
        mean_abs = float(np.nanmean(smd.values)) if smd.size > 0 else np.nan
        rate = float(np.nanmean((smd < 0.25).to_numpy(dtype=float))) if smd.size > 0 else np.nan
        return {"bal_mean_abs_std_diff": mean_abs, "bal_rate": rate}
    except Exception:
        return {"bal_mean_abs_std_diff": np.nan, "bal_rate": np.nan}


def _plot_episode_series(ax_top: plt.Axes,
                         ax_bot: plt.Axes,
                         df: pd.DataFrame,
                         pre_start: pd.Timestamp,
                         treat_start: pd.Timestamp,
                         post_end: pd.Timestamp,
                         title: str) -> None:
    """
    Dibuja Observado vs. Contrafactual (arriba) y Efecto / Acumulado (abajo).
    """
    d = df[(df["date"] >= pre_start) & (df["date"] <= post_end)].copy()
    d = d.sort_values("date")
    # Estimar columnas faltantes
    if "cf" not in d.columns and {"obs", "effect"}.issubset(d.columns):
        d["cf"] = d["obs"] - d["effect"]
    if "effect" not in d.columns and {"obs", "cf"}.issubset(d.columns):
        d["effect"] = d["obs"] - d["cf"]
    if "cum_effect" not in d.columns and "effect" in d.columns:
        d["cum_effect"] = (d["effect"].where(d["date"] >= treat_start, 0.0)).cumsum()

    # ---- Panel superior: Observado vs Contrafactual
    if {"obs", "cf"}.issubset(d.columns):
        ax_top.plot(d["date"], d["obs"], label="Observado", linewidth=1.5)
        ax_top.plot(d["date"], d["cf"], linestyle="--", label="Contrafactual", linewidth=1.2)
    elif "obs" in d.columns:
        ax_top.plot(d["date"], d["obs"], label="Observado", linewidth=1.5)
    else:
        ax_top.text(0.5, 0.5, "Sin columnas 'obs'/'cf'", ha="center", va="center", transform=ax_top.transAxes)

    ax_top.set_title(title)
    ax_top.set_ylabel("Ventas (unid.)")
    ax_top.legend(loc="upper left")

    # Línea vertical en inicio de tratamiento
    ax_top.axvline(pd.to_datetime(treat_start), linewidth=1.0)

    # ---- Panel inferior: Solo efecto acumulado
    if "effect" in d.columns:
        # Calcular acumulado: 0 antes del tratamiento, acumular solo después
        d_plot = d.copy()
        d_plot["cum_effect_plot"] = d_plot["effect"].where(d_plot["date"] >= treat_start, 0.0).cumsum()
        
        # Graficar solo el efecto acumulado con línea punteada
        ax_bot.plot(d_plot["date"], d_plot["cum_effect_plot"], linestyle="--", label="Acumulado", linewidth=1.5, color='C0')
        ax_bot.set_ylabel("Acumulado")
        ax_bot.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
        ax_bot.axvline(pd.to_datetime(treat_start), linewidth=1.0)
        ax_bot.legend(loc="upper left")
    else:
        ax_bot.text(0.5, 0.5, "Sin columna 'effect' y sin 'obs'/'cf' para derivarlo.",
                    ha="center", va="center", transform=ax_bot.transAxes)


# ---------------------------------------------------------------------
# Gráficos simples (resumen sin seaborn)
# ---------------------------------------------------------------------

def _hist_att(ax: plt.Axes, data: pd.Series, title: str) -> None:
    arr = pd.to_numeric(data, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.hist(arr, bins=30)
    ax.set_title(title)
    ax.set_xlabel("ATT_sum")
    ax.set_ylabel("Frecuencia")


def _scatter_rmspe_vs_att(ax: plt.Axes, rmspe: pd.Series, att: pd.Series, title: str) -> None:
    x = pd.to_numeric(rmspe, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(att, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.scatter(x[m], y[m], s=10)
    ax.set_title(title)
    ax.set_xlabel("RMSPE (pre)")
    ax.set_ylabel("ATT_sum")
    ax.axvline(0.0, linewidth=0.5)
    ax.axhline(0.0, linewidth=0.5)


# ---------------------------------------------------------------------
# Gráficos académicos para métricas causales comparativas
# ---------------------------------------------------------------------

def _plot_causal_metrics_comparison(causal_df: pd.DataFrame, fig_dir: Path, dpi: int = 300) -> None:
    """
    Genera gráficos académicos comparando métricas causales entre modelos.
    Estilo apropiado para publicación en papers.
    """
    if causal_df.empty or "model_type" not in causal_df.columns:
        return
    
    # Configuración de estilo académico
    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
    
    # Colores consistentes por modelo
    model_colors = {
        'gsc': '#2E86AB',      # Azul profesional
        'meta-t': '#A23B72',   # Magenta
        'meta-s': '#F18F01',   # Naranja
        'meta-x': '#C73E1D'    # Rojo
    }
    
    models = sorted(causal_df['model_type'].unique())
    
    # ========== FIGURA 1: Panel de métricas de predicción ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Prediction Quality Metrics Comparison', fontsize=14, fontweight='bold', y=0.995)
    
    # 1.1 RMSPE Pre-tratamiento (boxplot)
    ax = axes[0, 0]
    data_rmspe = []
    labels_rmspe = []
    for model in models:
        vals = causal_df[causal_df['model_type'] == model]['pred_rmspe_pre'].dropna()
        if len(vals) > 0:
            data_rmspe.append(vals)
            labels_rmspe.append(model.upper())
    
    if data_rmspe:
        bp = ax.boxplot(data_rmspe, labels=labels_rmspe, patch_artist=True, widths=0.6)
        for patch, model in zip(bp['boxes'], models[:len(data_rmspe)]):
            patch.set_facecolor(model_colors.get(model, '#888888'))
            patch.set_alpha(0.7)
        ax.set_ylabel('RMSPE (Pre-treatment)', fontsize=11)
        ax.set_title('(A) Prediction Error', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.25, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    
    # 1.2 Correlación Pre-tratamiento
    ax = axes[0, 1]
    data_corr = []
    labels_corr = []
    for model in models:
        vals = causal_df[causal_df['model_type'] == model]['pred_corr_pre'].dropna()
        if len(vals) > 0:
            data_corr.append(vals)
            labels_corr.append(model.upper())
    
    if data_corr:
        bp = ax.boxplot(data_corr, labels=labels_corr, patch_artist=True, widths=0.6)
        for patch, model in zip(bp['boxes'], models[:len(data_corr)]):
            patch.set_facecolor(model_colors.get(model, '#888888'))
            patch.set_alpha(0.7)
        ax.set_ylabel('Correlation (Obs vs Pred)', fontsize=11)
        ax.set_title('(B) Pre-treatment Correlation', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.9, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    # 1.3 R² Pre-tratamiento
    ax = axes[1, 0]
    data_r2 = []
    labels_r2 = []
    for model in models:
        vals = causal_df[causal_df['model_type'] == model]['pred_r2_pre'].dropna()
        if len(vals) > 0:
            data_r2.append(vals)
            labels_r2.append(model.upper())
    
    if data_r2:
        bp = ax.boxplot(data_r2, labels=labels_r2, patch_artist=True, widths=0.6)
        for patch, model in zip(bp['boxes'], models[:len(data_r2)]):
            patch.set_facecolor(model_colors.get(model, '#888888'))
            patch.set_alpha(0.7)
        ax.set_ylabel('R² (Pre-treatment)', fontsize=11)
        ax.set_title('(C) Goodness of Fit', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    # 1.4 Tabla resumen
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_data = []
    for model in models:
        subset = causal_df[causal_df['model_type'] == model]
        if len(subset) > 0:
            summary_data.append([
                model.upper(),
                f"{subset['pred_rmspe_pre'].median():.3f}",
                f"{subset['pred_corr_pre'].median():.3f}",
                f"{subset['pred_r2_pre'].median():.3f}",
                f"{len(subset)}"
            ])
    
    if summary_data:
        table = ax.table(cellText=summary_data,
                        colLabels=['Model', 'RMSPE↓', 'Corr↑', 'R²↑', 'N'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0.1, 0.2, 0.8, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Colorear header
        for i in range(5):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
        
        ax.set_title('(D) Summary Statistics (Median)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(fig_dir / 'causal_metrics_prediction_quality.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # ========== FIGURA 2: Heterogeneidad y Sensibilidad ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Effect Heterogeneity and Sensitivity Analysis', fontsize=14, fontweight='bold', y=0.995)
    
    # 2.1 Heterogeneidad del efecto (CV de tau)
    ax = axes[0, 0]
    data_het = []
    labels_het = []
    for model in models:
        vals = causal_df[causal_df['model_type'] == model]['het_tau_cv'].dropna()
        vals = vals[np.isfinite(vals) & (vals >= 0) & (vals < 10)]  # Filtrar outliers
        if len(vals) > 0:
            data_het.append(vals)
            labels_het.append(model.upper())
    
    if data_het:
        bp = ax.boxplot(data_het, labels=labels_het, patch_artist=True, widths=0.6)
        for patch, model in zip(bp['boxes'], models[:len(data_het)]):
            patch.set_facecolor(model_colors.get(model, '#888888'))
            patch.set_alpha(0.7)
        ax.set_ylabel('CV(τ) - Coefficient of Variation', fontsize=11)
        ax.set_title('(A) Effect Heterogeneity', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 2.2 Sensibilidad relativa
    ax = axes[0, 1]
    data_sens = []
    labels_sens = []
    for model in models:
        vals = causal_df[causal_df['model_type'] == model]['sens_relative_std'].dropna()
        vals = vals[np.isfinite(vals) & (vals >= 0) & (vals < 5)]  # Filtrar outliers
        if len(vals) > 0:
            data_sens.append(vals)
            labels_sens.append(model.upper())
    
    if data_sens:
        bp = ax.boxplot(data_sens, labels=labels_sens, patch_artist=True, widths=0.6)
        for patch, model in zip(bp['boxes'], models[:len(data_sens)]):
            patch.set_facecolor(model_colors.get(model, '#888888'))
            patch.set_alpha(0.7)
        ax.set_ylabel('Relative Std (σ/|ATT|)', fontsize=11)
        ax.set_title('(B) Sensitivity to Specifications', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.3, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='High sensitivity')
    
    # 2.3 Distribución de efectos positivos/negativos
    ax = axes[1, 0]
    pct_pos_data = []
    for model in models:
        vals = causal_df[causal_df['model_type'] == model]['het_pct_positive'].dropna()
        if len(vals) > 0:
            pct_pos_data.append(vals.median())
        else:
            pct_pos_data.append(0)
    
    if any(pct_pos_data):
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, pct_pos_data, width=0.6, alpha=0.7)
        for bar, model in zip(bars, models):
            bar.set_color(model_colors.get(model, '#888888'))
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_ylabel('% Positive Effects (Median)', fontsize=11)
        ax.set_title('(C) Direction of Effects', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylim(0, 100)
    
    # 2.4 Tabla resumen heterogeneidad
    ax = axes[1, 1]
    ax.axis('off')
    
    het_summary = []
    for model in models:
        subset = causal_df[causal_df['model_type'] == model]
        if len(subset) > 0:
            het_summary.append([
                model.upper(),
                f"{subset['het_tau_cv'].median():.3f}",
                f"{subset['het_tau_std'].median():.2f}",
                f"{subset['sens_relative_std'].median():.3f}",
                f"{subset['het_pct_positive'].median():.1f}%"
            ])
    
    if het_summary:
        table = ax.table(cellText=het_summary,
                        colLabels=['Model', 'CV(τ)', 'σ(τ)', 'Sens', '% Pos'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0.05, 0.2, 0.9, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
        
        ax.set_title('(D) Summary Statistics (Median)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(fig_dir / 'causal_metrics_heterogeneity_sensitivity.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # ========== FIGURA 3: Balance y Placebos ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Covariate Balance and Placebo Tests', fontsize=14, fontweight='bold', y=0.995)
    
    # 3.1 Balance de covariables (diferencia estandarizada media)
    ax = axes[0, 0]
    data_bal = []
    labels_bal = []
    for model in models:
        vals = causal_df[causal_df['model_type'] == model]['bal_mean_abs_std_diff'].dropna()
        if len(vals) > 0:
            data_bal.append(vals)
            labels_bal.append(model.upper())
    
    if data_bal:
        bp = ax.boxplot(data_bal, labels=labels_bal, patch_artist=True, widths=0.6)
        for patch, model in zip(bp['boxes'], models[:len(data_bal)]):
            patch.set_facecolor(model_colors.get(model, '#888888'))
            patch.set_alpha(0.7)
        ax.set_ylabel('Mean |Std. Diff|', fontsize=11)
        ax.set_title('(A) Covariate Balance', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.25, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Imbalance threshold')
        ax.axhline(y=0.1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good balance')
    
    # 3.2 Tasa de balance (proporción de covariables balanceadas)
    ax = axes[0, 1]
    data_bal_rate = []
    labels_bal_rate = []
    for model in models:
        vals = causal_df[causal_df['model_type'] == model]['bal_rate'].dropna()
        if len(vals) > 0:
            data_bal_rate.append(vals * 100)  # Convertir a porcentaje
            labels_bal_rate.append(model.upper())
    
    if data_bal_rate:
        bp = ax.boxplot(data_bal_rate, labels=labels_bal_rate, patch_artist=True, widths=0.6)
        for patch, model in zip(bp['boxes'], models[:len(data_bal_rate)]):
            patch.set_facecolor(model_colors.get(model, '#888888'))
            patch.set_alpha(0.7)
        ax.set_ylabel('Balance Rate (%)', fontsize=11)
        ax.set_title('(B) Proportion of Balanced Covariates', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 105)
    
    # 3.3 P-values de tests placebo espaciales
    ax = axes[1, 0]
    for model in models:
        vals = causal_df[causal_df['model_type'] == model]['plac_p_value_space'].dropna()
        vals = vals[(vals >= 0) & (vals <= 1)]
        if len(vals) > 0:
            ax.hist(vals, bins=20, alpha=0.5, label=model.upper(), 
                   color=model_colors.get(model, '#888888'), edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('P-value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('(C) Placebo Test P-values (Spatial)', fontsize=11, fontweight='bold')
    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=1.5, label='α=0.05')
    ax.axvline(x=0.10, color='orange', linestyle='--', linewidth=1, label='α=0.10')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3.4 Tabla resumen validación
    ax = axes[1, 1]
    ax.axis('off')
    
    val_summary = []
    for model in models:
        subset = causal_df[causal_df['model_type'] == model]
        if len(subset) > 0:
            p_vals = subset['plac_p_value_space'].dropna()
            p_vals = p_vals[(p_vals >= 0) & (p_vals <= 1)]
            pct_sig = (p_vals < 0.05).mean() * 100 if len(p_vals) > 0 else np.nan
            
            val_summary.append([
                model.upper(),
                f"{subset['bal_mean_abs_std_diff'].median():.3f}",
                f"{subset['bal_rate'].median()*100:.1f}%",
                f"{p_vals.median():.3f}" if len(p_vals) > 0 else "N/A",
                f"{pct_sig:.1f}%" if not np.isnan(pct_sig) else "N/A"
            ])
    
    if val_summary:
        table = ax.table(cellText=val_summary,
                        colLabels=['Model', 'Bal↓', 'Rate↑', 'p-val', '% Sig'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0.05, 0.2, 0.9, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
        
        ax.set_title('(D) Validation Summary (Median)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(fig_dir / 'causal_metrics_balance_placebo.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # ========== FIGURA 4: Radar chart comparativo (resumen ejecutivo) ==========
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Métricas normalizadas (0-1, donde 1 es mejor)
    metrics_radar = []
    labels_radar = ['Prediction\nQuality', 'Low\nSensitivity', 'Effect\nHeterogeneity',
                    'Covariate\nBalance', 'Placebo\nTests']
    
    for model in models:
        subset = causal_df[causal_df['model_type'] == model]
        if len(subset) > 0:
            # Normalizar cada métrica (1 = mejor)
            pred_quality = 1 - subset['pred_rmspe_pre'].median()  # Menor RMSPE es mejor
            pred_quality = np.clip(pred_quality, 0, 1)
            
            low_sens = 1 / (1 + subset['sens_relative_std'].median())  # Menor sensibilidad es mejor
            
            heterog = subset['het_tau_cv'].median()
            heterog = np.clip(heterog / 2, 0, 1)  # Normalizar CV
            
            balance = 1 - subset['bal_mean_abs_std_diff'].median() / 0.5  # Menor diferencia es mejor
            balance = np.clip(balance, 0, 1)
            
            p_vals = subset['plac_p_value_space'].dropna()
            p_vals = p_vals[(p_vals >= 0) & (p_vals <= 1)]
            placebo = p_vals.median() if len(p_vals) > 0 else 0.5  # P-value alto es mejor
            
            metrics_radar.append([pred_quality, low_sens, heterog, balance, placebo])
    
    # Ángulos para el radar
    angles = np.linspace(0, 2 * np.pi, len(labels_radar), endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo
    
    for i, model in enumerate(models):
        if i < len(metrics_radar):
            values = metrics_radar[i] + metrics_radar[i][:1]  # Cerrar el círculo
            ax.plot(angles, values, 'o-', linewidth=2, label=model.upper(),
                   color=model_colors.get(model, '#888888'))
            ax.fill(angles, values, alpha=0.15, color=model_colors.get(model, '#888888'))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title('Overall Model Performance Comparison\n(Normalized Metrics)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig.savefig(fig_dir / 'causal_metrics_radar_summary.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # Restaurar estilo por defecto
    plt.style.use('default')


# ---------------------------------------------------------------------
# EDA principal
# ---------------------------------------------------------------------

def _infer_exp_tag(fig_dir: Path) -> Optional[str]:
    try:
        parts = list(Path(fig_dir).parts)
        for part in reversed(parts):
            if part and part.lower() != "figures" and part.lower() != "algorithms":
                return part
    except Exception:
        return None
    return None


def _normalize_fig_dir(cfg: EDAConfig) -> Path:
    try:
        p = Path(cfg.figures_dir) if cfg.figures_dir is not None else Path("figures")
        tag = _infer_exp_tag(p)
        if not tag and cfg.episodes_index is not None:
            try:
                tag = Path(cfg.episodes_index).parent.name
            except Exception:
                tag = None
        if not tag and cfg.gsc_out_dir is not None:
            try:
                tag = Path(cfg.gsc_out_dir).parent.name
            except Exception:
                tag = None
        if tag:
            return Path("figures") / tag
        return p
    except Exception:
        return Path("figures")


def _load_donors_map_for_exp(exp_tag: str) -> pd.DataFrame:
    try:
        bases = [Path("./.data"), Path(".data")]  
        for base in bases:
            p = base / "processed_data" / "_shared_base" / "donors_per_victim.csv"
            try:
                if p.exists():
                    return pd.read_csv(p)
            except Exception:
                continue
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _load_meta_panel_windows(exp_tag: Optional[str] = None) -> Optional[pd.DataFrame]:
    bases = [Path("./.data"), Path(".data")]  
    candidates = []
    for base in bases:
        if exp_tag:
            candidates.append(base / "processed_meta" / exp_tag / "windows.parquet")
        candidates.append(base / "processed_meta" / "windows.parquet")
        if exp_tag:
            candidates.append(base / "processed" / exp_tag / "meta" / "all_units.parquet")
            candidates.append(base / "processed" / exp_tag / "gsc" / "meta" / "all_units.parquet")
            candidates.append(base / "processed_data" / exp_tag / "meta" / "all_units.parquet")
        candidates.append(base / "processed" / "meta" / "all_units.parquet")
        candidates.append(base / "processed_data" / "meta" / "all_units.parquet")
    for c in candidates:
        try:
            if c.exists():
                df = pd.read_parquet(c)
                return df
        except Exception:
            continue
    return None


def _read_donors_series(meta_df: Optional[pd.DataFrame],
                        donors_uids: Set[str],
                        date_min: pd.Timestamp,
                        date_max: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if meta_df is None or meta_df.empty or not donors_uids:
        return out
    try:
        df = meta_df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[(df["date"] >= pd.to_datetime(date_min)) & (df["date"] <= pd.to_datetime(date_max))]
        if "unit_id" not in df.columns:
            if {"store_nbr", "item_nbr"}.issubset(df.columns):
                df["unit_id"] = df["store_nbr"].astype(str) + ":" + df["item_nbr"].astype(str)
            else:
                return out
        df = df[df["unit_id"].isin(list(donors_uids))]
        if df.empty or "sales" not in df.columns:
            return out
        for uid, sub in df.groupby("unit_id", sort=False):
            out[str(uid)] = sub.sort_values("date")["date"].to_frame().assign(sales=pd.to_numeric(sub["sales"], errors="coerce").values)
        return out
    except Exception:
        return out


def _run_gsc_donors_overlay_for_base(cfg: EDAConfig,
                                     episodes: pd.DataFrame,
                                     gsc_series: Dict[str, pd.DataFrame]) -> None:
    try:
        fig_dir = _ensure_dir(_normalize_fig_dir(cfg))
        exp_tag = _infer_exp_tag(fig_dir)
        if not gsc_series:
            return

        donors_df = _load_donors_map_for_exp(exp_tag)
        if donors_df is None or donors_df.empty:
            try:
                logging.getLogger("EDA_algorithms").info(
                    "Overlay GSC con donantes omitido: donors_per_victim.csv no encontrado en ./.data o .data bajo processed_data/_shared_base",
                )
            except Exception:
                pass
            return

        epi = episodes.copy()
        if "episode_id" not in epi.columns:
            return
        for c in ["pre_start", "treat_start", "post_end"]:
            if c in epi.columns:
                epi[c] = pd.to_datetime(epi[c], errors="coerce")
        epi["episode_id"] = epi["episode_id"].astype(str)
        epi = epi.set_index("episode_id", drop=False)

        meta_df = _load_meta_panel_windows(exp_tag)

        pdf_path = fig_dir / "series_gsc_with_donors.pdf"
        with PdfPages(pdf_path) as pdf:
            for ep, vdf in gsc_series.items():
                ep_id = str(ep)
                if ep_id not in epi.index:
                    continue
                row = epi.loc[ep_id]
                pre_start = pd.to_datetime(row["pre_start"]) if "pre_start" in row else None
                treat_start = pd.to_datetime(row["treat_start"]) if "treat_start" in row else None
                post_end = pd.to_datetime(row["post_end"]) if "post_end" in row else None
                if pre_start is None or treat_start is None or post_end is None:
                    continue

                # Donantes del episodio
                sub = pd.DataFrame()
                if "episode_id" in donors_df.columns:
                    try:
                        sub = donors_df[donors_df["episode_id"].astype(str) == ep_id]
                    except Exception:
                        sub = pd.DataFrame()
                if sub.empty and {"j_store", "j_item"}.issubset(donors_df.columns) and {"j_store", "j_item"}.issubset(epi.columns):
                    try:
                        js, ji = int(row["j_store"]), int(row["j_item"])
                        sub = donors_df[(donors_df["j_store"] == js) & (donors_df["j_item"] == ji)]
                    except Exception:
                        sub = pd.DataFrame()
                if sub.empty or not {"donor_store", "donor_item"}.issubset(sub.columns):
                    continue
                donors_uids: Set[str] = set((sub["donor_store"].astype(str) + ":" + sub["donor_item"].astype(str)).tolist())
                if not donors_uids:
                    continue

                donors_map = _read_donors_series(meta_df, donors_uids, pre_start, post_end)
                d = vdf.copy()
                
                if len(donors_map) == 0:
                    w_, h_ = _figsize_from_orientation(cfg.orientation)
                    fig = plt.figure(figsize=(w_, h_))
                    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.4])
                    ax_top = fig.add_subplot(gs[0, 0])
                    ax_bot = fig.add_subplot(gs[1, 0])
                    _plot_episode_series(ax_top, ax_bot, d, pre_start, treat_start, post_end,
                                         title=f"GSC + Donors — Episodio {ep_id}")
                    ax_top.text(0.5, 0.5, "No hay datos de donantes disponibles", 
                               transform=ax_top.transAxes, ha='center', va='center',
                               fontsize=12, color='red', weight='bold',
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
                    fig.tight_layout()
                    png = fig_dir / f"series_gsc_with_donors_{ep_id.replace('/', '_')}.png"
                    fig.savefig(png, dpi=cfg.dpi)
                    pdf.savefig(fig, dpi=cfg.dpi)
                    plt.close(fig)
                    continue

                # Estándar para víctima (obs, cf)
                if "date" not in d.columns:
                    continue
                if "obs" not in d.columns:
                    oc = _pick_col(d, _OBS_COLS)
                    if oc:
                        d = d.rename(columns={oc: "obs"})
                if "cf" not in d.columns:
                    cc = _pick_col(d, _CF_COLS)
                    if cc:
                        d = d.rename(columns={cc: "cf"})
                if "effect" not in d.columns and {"obs", "cf"}.issubset(d.columns):
                    d["effect"] = pd.to_numeric(d["obs"], errors="coerce") - pd.to_numeric(d["cf"], errors="coerce")
                if "cum_effect" not in d.columns and "effect" in d.columns:
                    d["cum_effect"] = (d["effect"].where(d["date"] >= treat_start, 0.0)).cumsum()

                w_, h_ = _figsize_from_orientation(cfg.orientation)
                fig = plt.figure(figsize=(w_, h_))
                gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.4])
                ax_top = fig.add_subplot(gs[0, 0])
                ax_bot = fig.add_subplot(gs[1, 0])
                _plot_episode_series(ax_top, ax_bot, d, pre_start, treat_start, post_end,
                                     title=f"GSC + Donors — Episodio {ep_id}")

                # Overlay de donantes (líneas sutiles)
                for uid, sdf in donors_map.items():
                    sd = sdf[(sdf["date"] >= pre_start) & (sdf["date"] <= post_end)].sort_values("date")
                    if sd.empty:
                        continue
                    ax_top.plot(sd["date"], sd["sales"], color="gray", alpha=0.18, linewidth=0.8)

                ax_top.text(0.98, 0.98, f"Donors: {len(donors_map)}", transform=ax_top.transAxes,
                            va="top", ha="right", fontsize=8,
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="0.5"))

                fig.tight_layout()
                png = fig_dir / f"series_gsc_with_donors_{ep_id.replace('/', '_')}.png"
                fig.savefig(png, dpi=cfg.dpi)
                pdf.savefig(fig, dpi=cfg.dpi)
                plt.close(fig)
        try:
            logging.getLogger("EDA_algorithms").info(
                "PDF GSC con donantes guardado en: %s", str(pdf_path)
            )
        except Exception:
            pass
    except Exception:
        return

def _load_causal_metrics(base_dir: Path, model_type: str) -> pd.DataFrame:
    """Carga métricas causales desde carpeta causal_metrics."""
    if not base_dir:
        return pd.DataFrame()
    
    causal_dir = Path(base_dir) / "causal_metrics"
    if not causal_dir.exists():
        return pd.DataFrame()
    
    files = list(causal_dir.glob("*_causal.parquet"))
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    result = pd.concat(dfs, ignore_index=True)
    result["source"] = model_type
    return result


def run(cfg: EDAConfig) -> None:
    logger = logging.getLogger("EDA_algorithms")
    logger.setLevel(logging.INFO)

    # Carpetas de salida
    fig_dir = _ensure_dir(_normalize_fig_dir(cfg))
    tbl_dir = _ensure_dir(fig_dir / "tables")
    logger.info("EDA_algorithms: episodes_index=%s", str(cfg.episodes_index))
    logger.info("EDA_algorithms: gsc_out_dir=%s | meta_out_root=%s", str(cfg.gsc_out_dir), str(cfg.meta_out_root))
    logger.info("EDA_algorithms: figures_dir=%s", str(fig_dir))

    # 1) Cargar episodios (referencia)
    episodes = _safe_read_parquet(Path(cfg.episodes_index))
    if "episode_id" not in episodes.columns:
        raise ValueError("episodes_index.parquet no tiene columna 'episode_id'.")
    # Normalizar fechas usadas en plots/métricas
    for c in ["pre_start", "treat_start", "post_end"]:
        if c in episodes.columns:
            episodes[c] = pd.to_datetime(episodes[c], errors="coerce")

    all_eps_set = set(episodes["episode_id"].astype(str).unique().tolist())

    # 2) Cargar métricas de GSC + Meta (para figuras resumen)
    gsc_df = _load_gsc_metrics(cfg.gsc_out_dir)
    if cfg.gsc_out_dir:
        gsc_mets_path = Path(cfg.gsc_out_dir) / "gsc_metrics.parquet"
        if gsc_df.empty:
            logger.info("Métricas GSC no disponibles en: %s", str(gsc_mets_path))
        else:
            logger.info("Métricas GSC cargadas desde: %s (%d filas)", str(gsc_mets_path), len(gsc_df))
    meta_dict = _load_meta_metrics(cfg.meta_out_root, cfg.meta_learners)

    # 3) Cobertura por episodio
    cover = episodes[["episode_id"]].drop_duplicates().copy()
    cover["episode_id"] = cover["episode_id"].astype(str)
    cover["has_gsc"] = cover["episode_id"].isin(gsc_df["episode_id"]) if not gsc_df.empty else False
    for lr, mdf in meta_dict.items():
        cover[f"has_meta_{lr}"] = cover["episode_id"].isin(mdf["episode_id"]) if not mdf.empty else False
    meta_flags = [c for c in cover.columns if c.startswith("has_meta_")]
    any_cols = ["has_gsc"] + meta_flags
    cover["available"] = cover[any_cols].any(axis=1) if any_cols else False

    cover_path = tbl_dir / "eda_algorithms_coverage.parquet"
    cover.to_parquet(cover_path, index=False)

    # 4) Tablas consolidadas para gráficos simples
    stacks: List[pd.DataFrame] = []
    if not gsc_df.empty:
        stacks.append(gsc_df)
    for _, mdf in meta_dict.items():
        if not mdf.empty:
            stacks.append(mdf)
    all_mets = pd.concat(stacks, ignore_index=True) if stacks else pd.DataFrame()

    # 5) Gráficos resumen (histogramas y dispersión)
    w, h = _figsize_from_orientation(cfg.orientation)
    pdf_sum_path = fig_dir / "eda_algorithms_summary.pdf" if cfg.export_pdf else None
    pdf_sum = PdfPages(pdf_sum_path) if pdf_sum_path else None

    if not all_mets.empty:
        for src in sorted(all_mets["source"].unique()):
            sub = all_mets[all_mets["source"] == src]
            fig = plt.figure(figsize=(w, h))
            ax = fig.add_subplot(111)
            _hist_att(ax, sub.get("att_sum", pd.Series(dtype=float)), f"Distribución ATT_sum — {src.upper()}")
            fig.tight_layout()
            png = fig_dir / f"dist_att_sum__{src}.png"
            fig.savefig(png, dpi=cfg.dpi)
            if pdf_sum:
                pdf_sum.savefig(fig, dpi=cfg.dpi)
            plt.close(fig)

        if "rmspe_pre" in all_mets.columns and "att_sum" in all_mets.columns:
            for src in sorted(all_mets["source"].unique()):
                sub = all_mets[all_mets["source"] == src]
                fig = plt.figure(figsize=(w, h))
                ax = fig.add_subplot(111)
                _scatter_rmspe_vs_att(ax, sub["rmspe_pre"], sub["att_sum"], f"RMSPE(pre) vs ATT_sum — {src.upper()}")
                fig.tight_layout()
                png = fig_dir / f"scatter_rmspe_att__{src}.png"
                fig.savefig(png, dpi=cfg.dpi)
                if pdf_sum:
                    pdf_sum.savefig(fig, dpi=cfg.dpi)
                plt.close(fig)

    if pdf_sum:
        pdf_sum.close()

    # 6) Carga de series contrafactuales para render por episodio
    # --- GSC
    gsc_series: Dict[str, pd.DataFrame] = {}
    if cfg.gsc_out_dir is not None:
        gsc_cf_dir = Path(cfg.gsc_out_dir) / "cf_series"
        allowed = all_eps_set
        cap = cfg.max_episodes_gsc
        gsc_series = _load_cf_series_from_dir(gsc_cf_dir, episodes_allowed=allowed, cap=cap)
        if not gsc_series:
            logger.info("No se encontraron series CF de GSC en %s", str(gsc_cf_dir))

    # --- Meta (por learner)
    meta_series: Dict[str, Dict[str, pd.DataFrame]] = {}
    if cfg.meta_out_root is not None:
        for lr in cfg.meta_learners:
            cf_dir = Path(cfg.meta_out_root) / lr / "cf_series"
            cap = cfg.max_episodes_meta
            meta_series[lr] = _load_cf_series_from_dir(cf_dir, episodes_allowed=all_eps_set, cap=cap)
            if not meta_series[lr]:
                logger.info("No se encontraron series CF de Meta-%s en %s", lr.upper(), str(cf_dir))

    # 7) PDFs separados por fuente (GSC y Meta-learners)
    # Helper para obtener ventana de un episodio
    epi_map = (
        episodes[["episode_id", "pre_start", "treat_start", "post_end"]]
        .dropna(subset=["episode_id"])
        .copy()
    )
    epi_map["episode_id"] = epi_map["episode_id"].astype(str)
    epi_map = epi_map.set_index("episode_id")

    rows: List[Dict[str, float]] = []
    exp_tag = _infer_exp_tag(fig_dir)
    donors_df = _load_donors_map_for_exp(exp_tag) if exp_tag else pd.DataFrame()
    meta_panel = _load_meta_panel_windows(exp_tag)
    if meta_panel is not None and not meta_panel.empty and "date" in meta_panel.columns:
        meta_panel = meta_panel.copy()
        meta_panel["date"] = pd.to_datetime(meta_panel["date"], errors="coerce")
    if "unit_id" not in (meta_panel.columns if meta_panel is not None else []):
        if meta_panel is not None and {"store_nbr", "item_nbr"}.issubset(meta_panel.columns):
            meta_panel = meta_panel.copy()
            meta_panel["unit_id"] = meta_panel["store_nbr"].astype(str) + ":" + meta_panel["item_nbr"].astype(str)

    if gsc_series:
        for ep, d in gsc_series.items():
            ep_id = str(ep)
            if ep_id not in epi_map.index:
                continue
            row = epi_map.loc[ep_id]
            mets = _compute_metrics_for_window(d, row["pre_start"], row["treat_start"], row["post_end"])
            sens = _compute_sensitivity_windows(d, row["treat_start"], row["post_end"])
            het = _compute_effect_heterogeneity(d, row["treat_start"], row["post_end"])
            bal = _compute_balance_for_episode(episodes[episodes["episode_id"].astype(str) == ep_id].iloc[0], donors_df, meta_panel)
            out = {"source": "gsc", "episode_id": ep_id}
            out.update(mets); out.update(sens); out.update(het); out.update(bal)
            rows.append(out)

    for lr, dmap in meta_series.items():
        mdf = meta_dict.get(lr, pd.DataFrame())
        pmap = {}
        if not mdf.empty and "episode_id" in mdf.columns and "p_value_placebo_space" in mdf.columns:
            pmap = dict(zip(mdf["episode_id"].astype(str), pd.to_numeric(mdf["p_value_placebo_space"], errors="coerce")))
        for ep, d in dmap.items():
            ep_id = str(ep)
            if ep_id not in epi_map.index:
                continue
            row = epi_map.loc[ep_id]
            mets = _compute_metrics_for_window(d, row["pre_start"], row["treat_start"], row["post_end"])
            sens = _compute_sensitivity_windows(d, row["treat_start"], row["post_end"])
            het = _compute_effect_heterogeneity(d, row["treat_start"], row["post_end"])
            pval = float(pmap.get(ep_id, np.nan)) if pmap else np.nan
            out = {"source": f"meta-{lr}", "episode_id": ep_id, "plac_p_value_space": pval}
            out.update(mets); out.update(sens); out.update(het)
            rows.append(out)

    if rows:
        diag = pd.DataFrame(rows)
        diag_path = tbl_dir / "episode_causal_diagnostics.parquet"
        diag.to_parquet(diag_path, index=False)
        diag_csv = tbl_dir / "episode_causal_diagnostics.csv"
        diag.to_csv(diag_csv, index=False)

    def _render_one(src: str, ep: str, d: pd.DataFrame, pdf_writer=None) -> None:
        if ep not in epi_map.index:
            return
        row = epi_map.loc[ep]
        pre_start = pd.to_datetime(row["pre_start"])
        treat_start = pd.to_datetime(row["treat_start"])
        post_end = pd.to_datetime(row["post_end"])

        # Asegurar columnas clave
        if "date" not in d.columns:
            return
        if "obs" not in d.columns:
            oc = _pick_col(d, _OBS_COLS)
            if oc:
                d = d.rename(columns={oc: "obs"})
        if "cf" not in d.columns:
            cc = _pick_col(d, _CF_COLS)
            if cc:
                d = d.rename(columns={cc: "cf"})
        if "effect" not in d.columns and {"obs", "cf"}.issubset(d.columns):
            d["effect"] = pd.to_numeric(d["obs"], errors="coerce") - pd.to_numeric(d["cf"], errors="coerce")
        if "cum_effect" not in d.columns and "effect" in d.columns:
            d["cum_effect"] = (d["effect"].where(d["date"] >= treat_start, 0.0)).cumsum()

        # Calcular métricas in‑place (por si no hay métricas en parquet)
        mets = _compute_metrics_for_window(d, pre_start, treat_start, post_end)

        # Figura
        w_, h_ = _figsize_from_orientation(cfg.orientation)
        fig = plt.figure(figsize=(w_, h_))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.4])
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0])

        _plot_episode_series(ax_top, ax_bot, d, pre_start, treat_start, post_end,
                             title=f"{src} — Episodio {ep}")

        # Caja de métricas
        txt = (
            f"RMSPE_pre: {mets['rmspe_pre']:.2f}\n"
            f"ATT_sum: {mets['att_sum']:.2f}\n"
            f"ATT_mean: {mets['att_mean']:.2f}"
        )
        ax_top.text(0.98, 0.98, txt, transform=ax_top.transAxes, va="top", ha="right",
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.5"))

        fig.tight_layout()

        # Guardar PNG y PDF
        safe_ep = ep.replace("/", "_")
        png = fig_dir / f"series_{src.replace(' ', '_').replace('/', '_')}_{safe_ep}.png"
        fig.savefig(png, dpi=cfg.dpi)
        if pdf_writer:
            pdf_writer.savefig(fig, dpi=cfg.dpi)
        plt.close(fig)

    # --- Render GSC (PDF separado)
    if gsc_series and cfg.export_pdf:
        pdf_gsc_path = fig_dir / "series_gsc.pdf"
        with PdfPages(pdf_gsc_path) as pdf_gsc:
            for ep, d in gsc_series.items():
                _render_one("GSC", ep, d, pdf_writer=pdf_gsc)
        logger.info("PDF GSC guardado en: %s", str(pdf_gsc_path))
    elif gsc_series:
        # Solo PNG, sin PDF
        for ep, d in gsc_series.items():
            _render_one("GSC", ep, d, pdf_writer=None)

    # --- Render Meta por learner (PDF separado por learner)
    for lr, dmap in meta_series.items():
        if dmap and cfg.export_pdf:
            pdf_meta_path = fig_dir / f"series_meta_{lr}.pdf"
            with PdfPages(pdf_meta_path) as pdf_meta:
                for ep, d in dmap.items():
                    _render_one(f"Meta-{lr.upper()}", ep, d, pdf_writer=pdf_meta)
            logger.info("PDF Meta-%s guardado en: %s", lr.upper(), str(pdf_meta_path))
        elif dmap:
            # Solo PNG, sin PDF
            for ep, d in dmap.items():
                _render_one(f"Meta-{lr.upper()}", ep, d, pdf_writer=None)

    try:
        _run_gsc_donors_overlay_for_base(cfg, episodes, gsc_series)
    except Exception:
        pass

    # 8) Top episodios por |ATT_sum| (si hubo métricas)
    if not all_mets.empty and "att_sum" in all_mets.columns:
        rows: List[pd.DataFrame] = []
        for src in sorted(all_mets["source"].unique()):
            sub = all_mets[all_mets["source"] == src].copy()
            sub["abs_att_sum"] = np.abs(pd.to_numeric(sub["att_sum"], errors="coerce"))
            sub = sub.sort_values("abs_att_sum", ascending=False)
            k = 25 if sub.shape[0] >= 25 else sub.shape[0]
            rows.append(sub.head(k).assign(rank=np.arange(1, k + 1), source=src))
        top_tbl = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not top_tbl.empty:
            top_tbl = top_tbl[["source", "rank", "episode_id", "att_sum", "rmspe_pre"]]
            top_tbl_path = tbl_dir / "top_episodes_by_abs_att.parquet"
            top_tbl.to_parquet(top_tbl_path, index=False)

    # 9) Cargar y consolidar métricas causales comparativas
    causal_metrics_list = []
    
    # GSC
    if cfg.gsc_out_dir:
        gsc_causal = _load_causal_metrics(cfg.gsc_out_dir, "gsc")
        if not gsc_causal.empty:
            causal_metrics_list.append(gsc_causal)
            logger.info("Métricas causales GSC cargadas: %d episodios", len(gsc_causal))
    
    # Meta-learners
    if cfg.meta_out_root:
        for lr in cfg.meta_learners:
            meta_dir = Path(cfg.meta_out_root) / lr
            meta_causal = _load_causal_metrics(meta_dir, f"meta-{lr}")
            if not meta_causal.empty:
                causal_metrics_list.append(meta_causal)
                logger.info("Métricas causales Meta-%s cargadas: %d episodios", lr.upper(), len(meta_causal))
    
    # Consolidar y guardar tabla comparativa
    if causal_metrics_list:
        causal_all = pd.concat(causal_metrics_list, ignore_index=True)
        causal_path = tbl_dir / "causal_metrics_comparison.parquet"
        causal_all.to_parquet(causal_path, index=False)
        logger.info("Tabla comparativa de métricas causales guardada: %s", str(causal_path))
        
        # Exportar también a CSV para fácil inspección
        causal_csv_path = tbl_dir / "causal_metrics_comparison.csv"
        causal_all.to_csv(causal_csv_path, index=False)
        
        # Resumen por modelo
        summary_cols = [
            "pred_rmspe_pre", "pred_corr_pre", "pred_r2_pre",
            "het_tau_std", "het_tau_cv", "sens_att_cv", "sens_relative_std",
            "bal_mean_abs_std_diff", "bal_rate", "plac_p_value_space"
        ]
        available_cols = [c for c in summary_cols if c in causal_all.columns]
        
        if available_cols:
            summary = causal_all.groupby("model_type")[available_cols].agg(['mean', 'median', 'std']).round(4)
            summary_path = tbl_dir / "causal_metrics_summary_by_model.csv"
            summary.to_csv(summary_path)
            logger.info("Resumen por modelo guardado: %s", str(summary_path))
        
        # ===== GENERAR GRÁFICOS ACADÉMICOS DE MÉTRICAS CAUSALES =====
        logger.info("Generando gráficos académicos de métricas causales comparativas...")
        try:
            _plot_causal_metrics_comparison(causal_all, fig_dir, dpi=cfg.dpi)
            logger.info("Gráficos de métricas causales generados exitosamente:")
            logger.info("  - causal_metrics_prediction_quality.png")
            logger.info("  - causal_metrics_heterogeneity_sensitivity.png")
            logger.info("  - causal_metrics_balance_placebo.png")
            logger.info("  - causal_metrics_radar_summary.png")
        except Exception as e:
            logger.error(f"Error generando gráficos de métricas causales: {e}", exc_info=True)

    # Mensajes finales
    logger.info("EDA_algorithms completado.")
    logger.info("Cobertura guardada en: %s", str(cover_path))