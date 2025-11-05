# -*- coding: utf-8 -*-
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
"""
# eda_algorithms.py
# eda_algorithms.py
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates


# ---------------------------------------------------------------------
# Configuración y utilidades
# ---------------------------------------------------------------------

@dataclass
class EDAConfig:
    # Entradas (alineadas a los módulos de preparación/ejecución)
    episodes_index: Path = field(default_factory=_data_root / "meta" / "episodes_index.parquet")
    gsc_out_dir: Path = field(default_factory=_data_root / "gsc")
    meta_out_root: Path = field(default_factory=_data_root / "meta")
    meta_learners: Tuple[str, ...] = ("t", "s", "x")  # explorará las carpetas si existen

    # Salidas
    figures_dir: Path = field(default_factory=_fig_root)

    # Render
    orientation: str = "landscape"  # 'landscape' (11x8.5) | 'portrait' (8.5x11)
    dpi: int = 300
    style: str = "academic"
    font_size: int = 10
    grid: bool = True

    # Selección de episodios (None = sin límite)
    max_episodes_gsc: Optional[int] = None
    max_episodes_meta: Optional[int] = None

    # Exportar PDFs multipágina
    export_pdf: bool = True

    # Suavizado *para plot* del observado
    plot_sales_smooth: bool = True
    plot_smooth_kind: str = "ema"      # "ema" | "ma"
    plot_ema_span: float = 7.0
    plot_ma_window: int = 7
    exp_tag: str = "A_base"

    # Seguridad de integridad temporal
    strict_unique_dates: bool = False  # si True, lanza si quedan duplicados tras canonizar


# ---------------- helpers de config/paths ---------------- #
from dataclasses import field  # si el archivo define dataclasses con default_factory
def _data_root() -> Path:
    """
    Prefiere ./.data; si no existe, intenta ./data; y por último crea ./.data.
    """
    for cand in (Path("./.data"), Path("./data")):
        try:
            if cand.exists():
                return cand
        except Exception:
            pass
    return Path("./.data")

def _fig_root() -> Path:
    return Path("./figures")

def _to_path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _coerce_all_paths(cfg: EDAConfig) -> EDAConfig:
    cfg.episodes_index = _to_path(cfg.episodes_index)
    cfg.gsc_out_dir = _to_path(cfg.gsc_out_dir)
    cfg.meta_out_root = _to_path(cfg.meta_out_root)
    cfg.figures_dir = _to_path(cfg.figures_dir)
    return cfg

def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _academic_style(font_size: int = 10):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "legend.frameon": False,
        "legend.fontsize": font_size - 1,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

def _fig_letter_size(orientation: str) -> Tuple[float, float]:
    # Carta: 8.5 x 11 pulgadas
    if str(orientation).lower().startswith("land"):
        return (11.0, 8.5)
    return (8.5, 11.0)

def _save_letter(fig: mpl.figure.Figure, path: Path, orientation: str, dpi: int = 300):
    fig.set_size_inches(*_fig_letter_size(orientation), forward=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------- helpers de lectura ---------------- #

def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        if not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
        else:
            df[col] = df[col].dt.tz_localize(None)
    return df

def _format_time_axis(ax: mpl.axes.Axes):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")

def _try_read_parquet(path: Path, name: str) -> Optional[pd.DataFrame]:
    if not _exists(path):
        logging.warning(f"No encontrado: {name} en {path}")
        return None
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        logging.exception(f"Error leyendo {name} en {path}: {e}")
        return None

def _read_episodes_index(cfg: EDAConfig) -> pd.DataFrame:
    df = _try_read_parquet(cfg.episodes_index, "episodes_index")
    if df is None or df.empty:
        logging.warning("episodes_index.parquet no encontrado o vacío; inferiré desde cf_series si es posible.")
        return pd.DataFrame()
    for c in ["pre_start", "treat_start", "post_start", "post_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.tz_localize(None)
    if "episode_id" in df.columns:
        df["episode_id"] = df["episode_id"].astype(str)
    return df

def _read_gsc_metrics(cfg: EDAConfig) -> Optional[pd.DataFrame]:
    # 1) figures/<exp_tag>/gsc/gsc_metrics.parquet
    p1 = cfg.gsc_out_dir / "gsc_metrics.parquet"
    df = _try_read_parquet(p1, "gsc_metrics")
    if (df is None or df.empty):
        # 2) fallback a .data/processed_data/gsc_outputs
        p2 = _data_root() / "processed_data" / "gsc_outputs" / "gsc_metrics.parquet"
        df = _try_read_parquet(p2, "gsc_metrics (.data fallback)")
    if df is None or df.empty:
        return None
    df = df.copy()
    df["episode_id"] = df["episode_id"].astype(str)
    df["abs_att_sum"] = np.abs(df.get("att_sum", np.nan)).astype(float)
    return df

def _read_meta_metrics(cfg: EDAConfig) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for lr in cfg.meta_learners:
        # 1) figures/<exp_tag>/meta/<lr>/meta_metrics_<lr>.parquet
        p1 = cfg.meta_out_root / lr / f"meta_metrics_{lr}.parquet"
        df = _try_read_parquet(p1, f"meta_metrics_{lr}")
        if df is None or df.empty:
            # 2) fallback a .data/processed/meta_outputs/<lr>/
            p2 = _data_root() / "processed" / "meta_outputs" / lr / f"meta_metrics_{lr}.parquet"
            df = _try_read_parquet(p2, f"meta_metrics_{lr} (.data fallback)")
        if df is None or df.empty:
            continue
        df = df.copy()
        df["episode_id"] = df["episode_id"].astype(str)
        df["abs_att_sum"] = np.abs(df.get("att_sum", np.nan)).astype(float)
        out[lr] = df
    return out

def _gsc_cf_path(cfg: EDAConfig, episode_id: str) -> Path:
    return cfg.gsc_out_dir / "cf_series" / f"{episode_id}_cf.parquet"

def _meta_cf_path(cfg: EDAConfig, learner: str, episode_id: str) -> Path:
    return cfg.meta_out_root / learner / "cf_series" / f"{episode_id}_cf.parquet"


# ---------------------------------------------------------------------
# Smoothing, canonización y series para plot
# ---------------------------------------------------------------------

def _smooth_series(y: pd.Series, kind: str = "ema",
                   ema_span: float = 7.0, ma_window: int = 7) -> pd.Series:
    y = y.sort_index()
    if kind.lower() == "ma":
        return y.rolling(int(max(1, ma_window)), min_periods=1).mean()
    # EMA causal (adjust=False)
    alpha = 2.0 / (float(ema_span) + 1.0)
    return y.ewm(alpha=alpha, adjust=False).mean()

def _victim_uid(ep_row: Optional[pd.Series]) -> Optional[str]:
    if isinstance(ep_row, pd.Series):
        try:
            return f"{int(ep_row.get('j_store'))}:{int(ep_row.get('j_item'))}"
        except Exception:
            return None
    return None

def _canonize_cf(cf: pd.DataFrame,
                 ep_row: Optional[pd.Series],
                 cfg: EDAConfig,
                 algo_tag: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Garantiza 1 fila por fecha para la víctima.
    Estrategia:
      1) Normaliza 'date' y orden.
      2) Si existe treated_unit -> quedarnos con treated_unit==1.
      3) Si no, y existe unit_id + episodio -> filtrar por víctima j_store:j_item.
      4) Si persisten duplicados de fecha -> colapsar por mediana (num) / primero (no num).
    Devuelve (cf_canon, info_dedup).
    """
    cf = cf.copy()
    cf = _ensure_datetime(cf, "date")
    cf = cf.sort_values(["date", "unit_id"] if "unit_id" in cf.columns else ["date"]).reset_index(drop=True)

    before_rows = int(cf.shape[0])

    # Paso 1: preferir la víctima (treated_unit == 1)
    if "treated_unit" in cf.columns:
        tu = pd.to_numeric(cf["treated_unit"], errors="coerce").fillna(0).astype(int)
        if (tu == 1).any():
            cf = cf.loc[tu == 1].copy()

    # Paso 2: si hay unit_id y ep_row, filtrar a víctima
    if "unit_id" in cf.columns and isinstance(ep_row, pd.Series):
        vuid = _victim_uid(ep_row)
        if vuid is not None and (cf["unit_id"].astype(str) == vuid).any():
            cf = cf.loc[cf["unit_id"].astype(str) == vuid].copy()

    # Paso 3: nombres flexibles para columnas objetivo/contrafactual
    if "y_hat" not in cf.columns:
        if "mu0_hat" in cf.columns:
            cf["y_hat"] = cf["mu0_hat"]
        elif "y0_hat" in cf.columns:
            cf["y_hat"] = cf["y0_hat"]
    if "y_obs" not in cf.columns and "sales" in cf.columns:
        cf["y_obs"] = cf["sales"]

    # Paso 4: colapsar duplicados por fecha si existen
    dup_dates = int(cf["date"].duplicated(keep=False).sum()) if "date" in cf.columns else 0
    if dup_dates > 0:
        logging.warning(f"[{algo_tag}] Fechas duplicadas detectadas (n={dup_dates}). Se colapsará por fecha (mediana/primero).")
        num_cols = cf.select_dtypes(include=[np.number]).columns.tolist()
        # conservar explícitamente señales de plot si existen
        special_cols = [c for c in ["y_hat", "y_obs", "effect_plot", "y_obs_plot", "cum_effect_obs", "obs_mask"] if c in cf.columns]
        for c in special_cols:
            if c not in num_cols and c in cf.columns:
                # intentar convertir a float si procede
                try:
                    cf[c] = pd.to_numeric(cf[c], errors="ignore")
                except Exception:
                    pass
        num_cols = cf.select_dtypes(include=[np.number]).columns.tolist()
        non_num_cols = [c for c in cf.columns if c not in num_cols + ["date"]]

        agg_spec = {c: "median" for c in num_cols}
        agg_first = {c: "first" for c in non_num_cols}
        cf = (cf.groupby("date", as_index=False)
                .agg({**agg_spec, **agg_first})
                .sort_values("date")
                .reset_index(drop=True))

    # Chequeo final de unicidad temporal
    still_dups = int(cf["date"].duplicated(keep=False).sum())
    if still_dups > 0 and cfg.strict_unique_dates:
        raise AssertionError(f"[{algo_tag}] Persisten {still_dups} fechas duplicadas tras canonizar.")

    info = {
        "n_rows_before": before_rows,
        "n_rows_after": int(cf.shape[0]),
        "dup_dates_fixed": int(max(0, dup_dates - still_dups)),
        "dup_dates_remaining": int(still_dups),
    }
    return cf, info

def _series_for_plot(cf: pd.DataFrame,
                     cfg: EDAConfig,
                     ep_row: Optional[pd.Series],
                     algo_tag: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, Dict[str,int]]:
    """
    Devuelve:
      dates, y_obs_plot, y_hat, effect_plot, effect (analítico), cum_effect_for_plot, dedup_info
    """
    cf, dedup_info = _canonize_cf(cf, ep_row, cfg, algo_tag)
    cf = cf.copy()
    cf = _ensure_datetime(cf, "date").sort_values("date").reset_index(drop=True)

    # y_obs
    if "y_obs" in cf.columns:
        y_obs = cf["y_obs"].to_numpy(float)
    elif "sales" in cf.columns:
        y_obs = cf["sales"].to_numpy(float)
    else:
        raise ValueError("El cf_series no contiene 'y_obs' ni 'sales'.")

    # y_hat
    if "y_hat" in cf.columns:
        y_hat = cf["y_hat"].to_numpy(float)
    elif "mu0_hat" in cf.columns:
        y_hat = cf["mu0_hat"].to_numpy(float)
    else:
        raise ValueError("El cf_series no contiene 'y_hat' ni 'mu0_hat'.")

    # dates
    dates = cf["date"].to_numpy()

    # obs_mask
    if "obs_mask" in cf.columns:
        obs_mask = cf["obs_mask"].astype(int).to_numpy() == 1
    else:
        obs_mask = np.isfinite(y_obs)

    # y_obs_plot
    if "y_obs_plot" in cf.columns:
        y_obs_plot = cf["y_obs_plot"].to_numpy(float)
    else:
        s = pd.Series(y_obs, index=pd.to_datetime(cf["date"]))
        # interp lineal para huecos y, opcionalmente, suavizar sólo para plot:
        y_interp = s.interpolate(limit_direction="both")
        if cfg.plot_sales_smooth:
            y_sm = _smooth_series(y_interp, kind=cfg.plot_smooth_kind,
                                  ema_span=cfg.plot_ema_span, ma_window=cfg.plot_ma_window)
            y_obs_plot = y_sm.to_numpy()
        else:
            y_obs_plot = y_interp.to_numpy()

        if not np.any(np.isfinite(y_obs_plot)):
            y_obs_plot = np.zeros_like(y_obs, dtype=float)

    # effect
    effect = y_obs - y_hat
    if "effect_plot" in cf.columns:
        effect_plot = cf["effect_plot"].to_numpy(float)
    else:
        effect_plot = y_obs_plot - y_hat

    # cum_effect plot‑ready
    if "cum_effect_obs" in cf.columns:
        cum_effect_for_plot = cf["cum_effect_obs"].to_numpy(float)
    else:
        eff_no_nan = np.where(np.isfinite(effect), effect, 0.0)
        cum_effect_for_plot = np.cumsum(eff_no_nan)

    # guardar obs_mask en dedup_info para sombreado
    dedup_info["has_obs_mask"] = int(obs_mask.any())
    cf["_obs_mask_tmp_"] = obs_mask.astype(int)
    return dates, y_obs_plot, y_hat, effect_plot, effect, cum_effect_for_plot, dedup_info


# ---------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------

def _false_spans(dates: np.ndarray, mask_bool: np.ndarray) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Devuelve intervalos [ini, fin] donde mask_bool es False (contiguos)."""
    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if mask_bool.size == 0:
        return spans
    in_span = False
    start = None
    for i, ok in enumerate(mask_bool):
        if (not ok) and (not in_span):
            in_span = True
            start = dates[i]
        if in_span and (ok or i == len(mask_bool) - 1):
            end = dates[i - 1] if ok else dates[i]
            spans.append((start, end))
            in_span = False
    return spans

def _draw_episode_timeseries(ax: mpl.axes.Axes,
                             dates: np.ndarray,
                             y_obs_plot: np.ndarray,
                             y_hat: np.ndarray,
                             treat_start: Optional[pd.Timestamp],
                             post_end: Optional[pd.Timestamp],
                             title: str,
                             subtitle: str,
                             grid: bool = True,
                             obs_mask: Optional[np.ndarray] = None):
    """Curvas principal: Observado (plot‑ready) vs Contrafactual y sombreado de periodo post."""
    ax.plot(dates, y_obs_plot, lw=1.3, label="Observado", alpha=0.95)
    ax.plot(dates, y_hat, lw=1.3, linestyle="--", label="Contrafactual", alpha=0.95)

    if treat_start is not None and post_end is not None:
        ax.axvspan(treat_start, post_end, color="0.90", zorder=0, label="Post")
        ax.axvline(treat_start, color="0.25", lw=0.9, linestyle=":")

    # sombrear tramos sin observado (si existe obs_mask)
    if obs_mask is not None and obs_mask.size == dates.size and np.any(~obs_mask):
        added = False
        for a, b in _false_spans(dates, obs_mask.astype(bool)):
            ax.axvspan(a, b, color="0.97", zorder=0, alpha=0.8,
                       label="(sin observado)" if not added else None)
            added = True

    _format_time_axis(ax)
    ax.set_ylabel("Ventas (unid.)")
    ax.set_title(title, pad=8)
    ax.legend(ncol=2, loc="upper left")
    if grid:
        ax.grid(True, axis="both")

def _draw_episode_effect(ax: mpl.axes.Axes,
                         dates: np.ndarray,
                         effect_plot: np.ndarray,
                         cum_effect_for_plot: Optional[np.ndarray] = None,
                         grid: bool = True):
    """Panel inferior: efecto y (opcional) efecto acumulado, ambos 'plot‑ready'."""
    ax.plot(dates, effect_plot, lw=1.0, label="Efecto (Y - Ŷ0)", alpha=0.95)
    if cum_effect_for_plot is not None:
        ax2 = ax.twinx()
        ax2.plot(dates, cum_effect_for_plot, lw=1.0, linestyle="--", alpha=0.9, label="Efecto acumulado")
        ax2.set_ylabel("Acumulado")
        # merge leyendas
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
    else:
        ax.legend(loc="upper left")
    _format_time_axis(ax)
    ax.set_ylabel("Efecto")
    if grid:
        ax.grid(True, axis="both")

def _annot_metrics_box(ax: mpl.axes.Axes, metrics: Dict[str, float]):
    """Cuadro con métricas (estilo académico)."""
    txt = []
    for k, v in metrics.items():
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            disp = "—"
        else:
            if "p_value" in k:
                disp = f"{v:0.3f}"
            elif "rmspe" in k or "mae" in k:
                disp = f"{v:0.4f}"
            else:
                disp = f"{v:0.2f}"
        txt.append(f"{k}: {disp}")
    ax.text(0.5, 0.98, "\n".join(txt),
            transform=ax.transAxes, va="top", ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.9, edgecolor="0.6"))


def fig_episode_page(cf: pd.DataFrame,
                     ep_row: Optional[pd.Series],
                     algorithm_label: str,
                     out_path: Path,
                     cfg: EDAConfig,
                     metrics_hint: Optional[Dict[str, float]] = None):
    """
    Render de lámina por episodio:
      - Panel 1: Observado (suavizado plot‑ready) vs Contrafactual
      - Panel 2: Efecto y efecto acumulado
    """
    dates, y_obs_plot, y_hat, effect_plot, effect_base, cum_eff, dedup = _series_for_plot(cf, cfg, ep_row, algorithm_label)

    # Ventana del episodio (si existe ep_row)
    treat_start = None
    post_end = None
    subtitle = ""
    title = f"{algorithm_label} — Episodio"

    if isinstance(ep_row, pd.Series):
        treat_start = ep_row.get("treat_start", None)
        post_end = ep_row.get("post_end", None)
        victim = None
        try:
            victim = f"{int(ep_row.get('j_store'))}:{int(ep_row.get('j_item'))}"
        except Exception:
            pass
        title = f"{algorithm_label} — Episodio {ep_row.get('episode_id','?')}"
        if victim:
            title += f" | Víctima {victim}"
        if (ep_row.get("pre_start") is not None) and (post_end is not None):
            subtitle = f"Ventana: {pd.to_datetime(ep_row.get('pre_start')).date()} → {pd.to_datetime(post_end).date()}"

    if cfg.style == "academic":
        _academic_style(cfg.font_size)

    fig = plt.figure()
    fig.set_size_inches(*_fig_letter_size(cfg.orientation), forward=True)

    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.2], hspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # obs_mask (si existe en cf)
    if "obs_mask" in cf.columns:
        obs_mask = cf["obs_mask"].astype(int).to_numpy() == 1
    else:
        # si no está, aproximar con finitud del observado
        obs_col = "y_obs" if "y_obs" in cf.columns else "sales"
        obs_mask = np.isfinite(pd.to_numeric(cf[obs_col], errors="coerce").to_numpy(float))

    _draw_episode_timeseries(ax1, dates, y_obs_plot, y_hat, treat_start, post_end,
                             title, subtitle, grid=cfg.grid, obs_mask=obs_mask)
    _draw_episode_effect(ax2, dates, effect_plot, cum_eff, grid=cfg.grid)

    if metrics_hint is None:
        metrics_hint = {}
    # añadir info de deduplicación a la caja
    metrics_hint = dict(metrics_hint)
    metrics_hint.setdefault("dup_dates_fixed", float(dedup.get("dup_dates_fixed", 0)))
    metrics_hint.setdefault("dup_dates_remaining", float(dedup.get("dup_dates_remaining", 0)))
    _annot_metrics_box(ax1, metrics_hint)

    if subtitle:
        fig.text(0.01, 0.98, subtitle, ha="left", va="top", fontsize=9)

    _save_letter(fig, out_path, cfg.orientation, cfg.dpi)


# ---------------------------------------------------------------------
# Canonicalización de episodios (mismo subconjunto/orden entre métodos)
# ---------------------------------------------------------------------

def _canonical_episode_list(cfg: EDAConfig,
                            gsc_metrics: Optional[pd.DataFrame],
                            meta_metrics: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Devuelve una lista ORDENADA de episode_id a usar por TODOS los métodos, con la misma longitud y orden.
    - Si hay GSC: ranking = |ATT_sum| de GSC desc.
    - Si no hay GSC: usa el primer meta disponible como ranking.
    - Dominio: intersección de episodios presentes en TODOS los métodos encontrados (para comparabilidad).
      Si la intersección es vacía, cae a la unión y mantiene el ranking del método de referencia.
    - Aplica límites max_episodes_* (toma el mínimo de ambos si ambos existen).
    """
    sources: List[Tuple[str, Iterable[str]]] = []
    if gsc_metrics is not None and not gsc_metrics.empty:
        sources.append(("gsc", gsc_metrics["episode_id"].astype(str).unique().tolist()))
    for lr, df in meta_metrics.items():
        if df is not None and not df.empty:
            sources.append((f"meta_{lr}", df["episode_id"].astype(str).unique().tolist()))

    if not sources:
        logging.warning("No hay fuentes de métricas (GSC ni Meta) para construir lista canónica de episodios.")
        return []

    # Intersección (preferida) y, si vacía, unión
    sets = [set(s[1]) for s in sources]
    common = set.intersection(*sets) if len(sets) >= 2 else sets[0]
    use_set = common if len(common) > 0 else set.union(*sets)

    # Ranking de referencia
    ref_df: Optional[pd.DataFrame] = None
    if gsc_metrics is not None and not gsc_metrics.empty:
        ref_df = gsc_metrics[["episode_id", "abs_att_sum"]].copy()
    else:
        for _, df in meta_metrics.items():
            if df is not None and not df.empty:
                ref_df = df[["episode_id", "abs_att_sum"]].copy()
                break

    if ref_df is None or ref_df.empty:
        ordered = sorted(list(use_set))
    else:
        ref_df = ref_df.copy()
        ref_df["episode_id"] = ref_df["episode_id"].astype(str)
        ref_df["abs_att_sum"] = ref_df["abs_att_sum"].astype(float)
        ref_df = ref_df[ref_df["episode_id"].isin(use_set)]
        ordered = ref_df.sort_values("abs_att_sum", ascending=False)["episode_id"].tolist()

    limits = []
    if cfg.max_episodes_gsc is not None:
        limits.append(int(cfg.max_episodes_gsc))
    if cfg.max_episodes_meta is not None:
        limits.append(int(cfg.max_episodes_meta))
    max_common = min(limits) if limits else None

    if max_common is not None:
        ordered = ordered[:max_common]

    return ordered


# ---------------------------------------------------------------------
# Orquestación por algoritmo
# ---------------------------------------------------------------------

def _render_gsc(cfg: EDAConfig,
                episodes: pd.DataFrame,
                canonical_ids: Optional[List[str]]):
    """Genera láminas por episodio + resúmenes para GSC (en orden canónico si se provee)."""
    gsc_metrics = _read_gsc_metrics(cfg)
    if gsc_metrics is None or gsc_metrics.empty:
        logging.warning("No hay métricas GSC para render.")
        return

    # Orden por canónico (si aplica) o por |ATT_sum| desc
    if canonical_ids:
        gsc_metrics = gsc_metrics[gsc_metrics["episode_id"].isin(canonical_ids)].copy()
        gsc_metrics["__ord__"] = gsc_metrics["episode_id"].map({e: i for i, e in enumerate(canonical_ids)})
        gsc_metrics = gsc_metrics.sort_values("__ord__").drop(columns="__ord__")
    else:
        gsc_metrics = gsc_metrics.sort_values("abs_att_sum", ascending=False)
        if cfg.max_episodes_gsc is not None:
            gsc_metrics = gsc_metrics.head(int(cfg.max_episodes_gsc))

    # Asegurar directorios y PDF
    _ensure_outdir(cfg.figures_dir)
    gsc_pdf_path = cfg.figures_dir / "gsc_report.pdf"
    pdf_writer = PdfPages(gsc_pdf_path) if cfg.export_pdf else None

    count = 0
    for _, m in gsc_metrics.iterrows():
        ep_id = str(m["episode_id"])
        cf_path = _gsc_cf_path(cfg, ep_id)
        cf = _try_read_parquet(cf_path, f"gsc cf_series ({ep_id})")
        if cf is None or cf.empty:
            continue

        # episodes_index -> fila del episodio
        ep_row = None
        if not episodes.empty and "episode_id" in episodes.columns:
            tmp = episodes.loc[episodes["episode_id"].astype(str) == ep_id]
            if not tmp.empty:
                ep_row = tmp.iloc[0]

        metrics_hint = {
            "RMSPE_pre": float(m.get("rmspe_pre", np.nan)),
            "ATT_sum": float(m.get("att_sum", np.nan)),
            "ATT_mean": float(m.get("att_mean", np.nan)),
            "p_value_placebo_space": float(m.get("p_value_placebo_space", np.nan)),
            "n_post": float(m.get("n_post", np.nan))
        }

        out_png = cfg.figures_dir / f"gsc_episode_{ep_id}_timeseries.png"

        # Generar figura con canonización/validación temporal
        if cfg.style == "academic":
            _academic_style(cfg.font_size)
        fig = plt.figure()
        fig.set_size_inches(*_fig_letter_size(cfg.orientation), forward=True)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.2], hspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0])

        dates, y_obs_plot, y_hat, effect_plot, effect_base, cum_eff, dedup = _series_for_plot(cf, cfg, ep_row, "GSC")
        obs_mask = cf["obs_mask"].astype(int).to_numpy() == 1 if "obs_mask" in cf.columns else None

        title = f"GSC — Episodio {ep_id}"
        if isinstance(ep_row, pd.Series):
            try:
                vic = f"{int(ep_row.get('j_store'))}:{int(ep_row.get('j_item'))}"
                title += f" | Víctima {vic}"
            except Exception:
                pass
        subtitle = ""
        if isinstance(ep_row, pd.Series) and (ep_row.get("pre_start") is not None) and (ep_row.get("post_end") is not None):
            subtitle = f"Ventana: {pd.to_datetime(ep_row.get('pre_start')).date()} → {pd.to_datetime(ep_row.get('post_end')).date()}"

        _draw_episode_timeseries(ax1, dates, y_obs_plot, y_hat,
                                 ep_row.get("treat_start", None) if isinstance(ep_row, pd.Series) else None,
                                 ep_row.get("post_end", None) if isinstance(ep_row, pd.Series) else None,
                                 title, subtitle, grid=cfg.grid, obs_mask=obs_mask)
        _draw_episode_effect(ax2, dates, effect_plot, cum_eff, grid=cfg.grid)

        # métricas + info de duplicados arreglados
        metrics_hint2 = dict(metrics_hint)
        metrics_hint2["dup_dates_fixed"] = float(dedup.get("dup_dates_fixed", 0))
        metrics_hint2["dup_dates_remaining"] = float(dedup.get("dup_dates_remaining", 0))
        _annot_metrics_box(ax1, metrics_hint2)

        if subtitle:
            fig.text(0.01, 0.98, subtitle, ha="left", va="top", fontsize=9)

        # Guardados
        _save_letter(fig, out_png, cfg.orientation, cfg.dpi)
        if pdf_writer is not None:
            fig_pdf, ax_pdf = plt.subplots()
            fig_pdf.set_size_inches(*_fig_letter_size(cfg.orientation), forward=True)
            ax_pdf.axis("off")
            try:
                img = plt.imread(out_png)
                ax_pdf.imshow(img)
            except Exception:
                pass
            pdf_writer.savefig(fig_pdf, dpi=cfg.dpi)
            plt.close(fig_pdf)

        count += 1

    if pdf_writer is not None:
        pdf_writer.close()
        logging.info(f"Reporte GSC exportado: {gsc_pdf_path} ({count} páginas)")

    # Resúmenes
    fig_prefix = cfg.figures_dir / "gsc_overview_summary"
    _overview_figs(gsc_metrics, "GSC", fig_prefix, cfg)


def _render_meta_for(cfg: EDAConfig,
                     learner: str,
                     episodes: pd.DataFrame,
                     canonical_ids: Optional[List[str]]):
    """Genera láminas por episodio + resúmenes para un Meta‑learner (orden canónico si se provee)."""
    mpath = cfg.meta_out_root / learner / f"meta_metrics_{learner}.parquet"
    mdf = _try_read_parquet(mpath, f"meta_metrics_{learner}")
    if mdf is None or mdf.empty:
        logging.warning(f"No hay métricas para Meta‑learner '{learner}'.")
        return

    mdf = mdf.copy()
    mdf["episode_id"] = mdf["episode_id"].astype(str)
    if "att_sum" in mdf.columns:
        mdf["abs_att_sum"] = np.abs(mdf["att_sum"].astype(float))
    else:
        mdf["abs_att_sum"] = np.nan

    # Orden canónico
    if canonical_ids:
        mdf = mdf[mdf["episode_id"].isin(canonical_ids)].copy()
        mdf["__ord__"] = mdf["episode_id"].map({e: i for i, e in enumerate(canonical_ids)})
        mdf = mdf.sort_values("__ord__").drop(columns="__ord__")
    else:
        mdf = mdf.sort_values("abs_att_sum", ascending=False)
        if cfg.max_episodes_meta is not None:
            mdf = mdf.head(int(cfg.max_episodes_meta))

    _ensure_outdir(cfg.figures_dir)
    pdf_path = cfg.figures_dir / f"meta_{learner}_report.pdf"
    pdf_writer = PdfPages(pdf_path) if cfg.export_pdf else None

    count = 0
    for _, m in mdf.iterrows():
        ep_id = str(m["episode_id"])
        cf_path = _meta_cf_path(cfg, learner, ep_id)
        cf = _try_read_parquet(cf_path, f"meta_{learner} cf_series ({ep_id})")
        if cf is None or cf.empty:
            continue

        ep_row = None
        if not episodes.empty and "episode_id" in episodes.columns:
            tmp = episodes.loc[episodes["episode_id"].astype(str) == ep_id]
            if not tmp.empty:
                ep_row = tmp.iloc[0]

        metrics_hint = {
            "RMSPE_pre": float(m.get("rmspe_pre", np.nan)),
            "ATT_sum": float(m.get("att_sum", np.nan)),
            "ATT_mean": float(m.get("att_mean", np.nan)),
            "p_value_placebo_space": float(m.get("p_value_placebo_space", np.nan)),
            "n_post": float(m.get("n_post_days", np.nan))
        }

        out_png = cfg.figures_dir / f"meta_{learner}_episode_{ep_id}_timeseries.png"

        if cfg.style == "academic":
            _academic_style(cfg.font_size)
        fig = plt.figure()
        fig.set_size_inches(*_fig_letter_size(cfg.orientation), forward=True)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.2], hspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0])

        dates, y_obs_plot, y_hat, effect_plot, effect_base, cum_eff, dedup = _series_for_plot(cf, cfg, ep_row, f"Meta-{learner.upper()}")
        obs_mask = cf["obs_mask"].astype(int).to_numpy() == 1 if "obs_mask" in cf.columns else None

        title = f"Meta‑{learner.upper()} — Episodio {ep_id}"
        if isinstance(ep_row, pd.Series):
            try:
                vic = f"{int(ep_row.get('j_store'))}:{int(ep_row.get('j_item'))}"
                title += f" | Víctima {vic}"
            except Exception:
                pass
        subtitle = ""
        if isinstance(ep_row, pd.Series) and (ep_row.get("pre_start") is not None) and (ep_row.get("post_end") is not None):
            subtitle = f"Ventana: {pd.to_datetime(ep_row.get('pre_start')).date()} → {pd.to_datetime(ep_row.get('post_end')).date()}"

        _draw_episode_timeseries(ax1, dates, y_obs_plot, y_hat,
                                 ep_row.get("treat_start", None) if isinstance(ep_row, pd.Series) else None,
                                 ep_row.get("post_end", None) if isinstance(ep_row, pd.Series) else None,
                                 title, subtitle, grid=cfg.grid, obs_mask=obs_mask)
        _draw_episode_effect(ax2, dates, effect_plot, cum_eff, grid=cfg.grid)

        metrics_hint2 = dict(metrics_hint)
        metrics_hint2["dup_dates_fixed"] = float(dedup.get("dup_dates_fixed", 0))
        metrics_hint2["dup_dates_remaining"] = float(dedup.get("dup_dates_remaining", 0))
        _annot_metrics_box(ax1, metrics_hint2)

        if subtitle:
            fig.text(0.01, 0.98, subtitle, ha="left", va="top", fontsize=9)

        _save_letter(fig, out_png, cfg.orientation, cfg.dpi)
        if pdf_writer is not None:
            fig_pdf, ax_pdf = plt.subplots()
            fig_pdf.set_size_inches(*_fig_letter_size(cfg.orientation), forward=True)
            ax_pdf.axis("off")
            try:
                img = plt.imread(out_png)
                ax_pdf.imshow(img)
            except Exception:
                pass
            pdf_writer.savefig(fig_pdf, dpi=cfg.dpi)
            plt.close(fig_pdf)

        count += 1

    if pdf_writer is not None:
        pdf_writer.close()
        logging.info(f"Reporte Meta‑{learner.upper()} exportado: {pdf_path} ({count} páginas)")

    # Resúmenes
    fig_prefix = cfg.figures_dir / f"meta_{learner}_overview_summary"
    _overview_figs(mdf, f"Meta‑{learner.upper()}", fig_prefix, cfg)


def _overview_figs(metrics: pd.DataFrame,
                   algorithm_label: str,
                   out_prefix: Path,
                   cfg: EDAConfig):
    """
    Láminas de resumen (tamaño carta):
      1) Distribución de RMSPE_pre
      2) Distribución de ATT_sum
      3) Dispersión |ATT_sum| vs RMSPE_pre
      4) (si existe) hist de p-values placebo espacio
    """
    if metrics is None or metrics.empty:
        return

    if cfg.style == "academic":
        _academic_style(cfg.font_size)

    # --- 1) RMSPE_pre
    fig1, ax = plt.subplots()
    x1 = metrics["rmspe_pre"].dropna().to_numpy(float) if "rmspe_pre" in metrics.columns else np.array([])
    if x1.size > 0:
        ax.hist(x1, bins=25, alpha=0.9)
    ax.set_title(f"{algorithm_label}: Distribución de RMSPE (pre)")
    ax.set_xlabel("RMSPE_pre")
    ax.set_ylabel("Frecuencia")
    _save_letter(fig1, out_prefix.with_name(out_prefix.name + "_rmspe.png"), cfg.orientation, cfg.dpi)

    # --- 2) ATT_sum
    fig2, ax = plt.subplots()
    x2 = metrics["att_sum"].dropna().to_numpy(float) if "att_sum" in metrics.columns else np.array([])
    if x2.size > 0:
        ax.hist(x2, bins=25, alpha=0.9)
    ax.set_title(f"{algorithm_label}: Distribución de ATT_sum (post)")
    ax.set_xlabel("ATT_sum")
    ax.set_ylabel("Frecuencia")
    _save_letter(fig2, out_prefix.with_name(out_prefix.name + "_attsum.png"), cfg.orientation, cfg.dpi)

    # --- 3) Dispersión |ATT_sum| vs RMSPE_pre
    fig3, ax = plt.subplots()
    x = metrics["rmspe_pre"].to_numpy(float) if "rmspe_pre" in metrics.columns else np.array([])
    y = np.abs(metrics["att_sum"].to_numpy(float)) if "att_sum" in metrics.columns else np.array([])
    if x.size > 0 and y.size > 0:
        ax.scatter(x, y, s=18, alpha=0.85)
        if np.isfinite(np.nanmedian(x)):
            ax.axvline(np.nanmedian(x), color="0.5", ls=":", lw=0.9)
    ax.set_title(f"{algorithm_label}: |ATT_sum| vs RMSPE_pre")
    ax.set_xlabel("RMSPE_pre")
    ax.set_ylabel("|ATT_sum|")
    _save_letter(fig3, out_prefix.with_name(out_prefix.name + "_scatter_rmspe_attsum.png"), cfg.orientation, cfg.dpi)

    # --- 4) p-values (si la columna existe)
    if "p_value_placebo_space" in metrics.columns:
        fig4, ax = plt.subplots()
        pv = metrics["p_value_placebo_space"].dropna().to_numpy(float)
        if pv.size > 0:
            ax.hist(pv, bins=np.linspace(0, 1, 21), alpha=0.9)
        ax.set_title(f"{algorithm_label}: Distribución de p‑values (placebo espacio)")
        ax.set_xlabel("p‑value")
        ax.set_ylabel("Frecuencia")
        _save_letter(fig4, out_prefix.with_name(out_prefix.name + "_pvalues.png"), cfg.orientation, cfg.dpi)


def _render_cross_method_comparison(cfg: EDAConfig,
                                    canonical_ids: Optional[List[str]]):
    """
    Gráficos de comparación entre métodos:
      - Dispersión ATT_sum(GSC) vs ATT_sum(Meta‑<learner>)
      - Línea y=x para referencia
    Usa sólo episodios de 'canonical_ids' si se definieron.
    """
    gsc = _read_gsc_metrics(cfg)
    meta_all = _read_meta_metrics(cfg)
    if gsc is None or gsc.empty or not meta_all:
        return

    if canonical_ids:
        gsc = gsc[gsc["episode_id"].isin(canonical_ids)]

    for lr, mdf in meta_all.items():
        df = mdf.copy()
        if canonical_ids:
            df = df[df["episode_id"].isin(canonical_ids)]
        try:
            m = gsc.merge(df[["episode_id", "att_sum"]], on="episode_id",
                          suffixes=("_gsc", f"_meta_{lr}"))
        except KeyError:
            continue
        if m.empty:
            continue

        if cfg.style == "academic":
            _academic_style(cfg.font_size)
        fig, ax = plt.subplots()
        x = m["att_sum_gsc"].to_numpy(float)
        y = m[f"att_sum_meta_{lr}"].to_numpy(float)
        ax.scatter(x, y, s=18, alpha=0.85)
        mx = np.nanmax(np.abs(np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]]))) if (x.size>0 and y.size>0) else 1.0
        r = (-mx, mx)
        ax.plot(r, r, color="0.3", lw=0.8, ls=":", label="y = x (referencia)")
        ax.set_title(f"Comparación ATT_sum: GSC vs Meta‑{lr.upper()}")
        ax.set_xlabel("ATT_sum (GSC)")
        ax.set_ylabel(f"ATT_sum (Meta‑{lr.upper()})")
        ax.legend(loc="upper left")
        _save_letter(fig, cfg.figures_dir / f"compare_att_sum_gsc_vs_meta_{lr}.png", cfg.orientation, cfg.dpi)


# ---------------------------------------------------------------------
# API de alto nivel
# ---------------------------------------------------------------------

def run(cfg: EDAConfig):
    cfg = _coerce_all_paths(cfg)

    _setup_logging("INFO")
    cfg.figures_dir = (cfg.figures_dir / cfg.exp_tag).resolve()
    _ensure_outdir(cfg.figures_dir)
    cfg.gsc_out_dir = (cfg.figures_dir / "gsc").resolve()
    cfg.meta_out_root = (cfg.figures_dir / "meta").resolve()
    if cfg.style == "academic":
        _academic_style(cfg.font_size)

    # Episodios index
    episodes = _read_episodes_index(cfg)

    # Métricas para construir orden canónico
    gsc_metrics = _read_gsc_metrics(cfg)
    meta_metrics = _read_meta_metrics(cfg)
    canonical_ids = _canonical_episode_list(cfg, gsc_metrics, meta_metrics)
    if canonical_ids:
        logging.info("Orden canónico de episodios (n=%d) establecido para todos los métodos.", len(canonical_ids))
    else:
        logging.info("No se estableció orden canónico (usará orden interno de cada método).")

    # Reporte GSC
    logging.info("Renderizando informe GSC…")
    _render_gsc(cfg, episodes, canonical_ids)

    # Reportes Meta‑learners
    any_meta = False
    for lr in cfg.meta_learners:
        if _exists(cfg.meta_out_root / lr / f"meta_metrics_{lr}.parquet"):
            logging.info(f"Renderizando informe Meta‑{lr.upper()}…")
            _render_meta_for(cfg, lr, episodes, canonical_ids)
            any_meta = True
    if not any_meta:
        logging.info("No se encontraron métricas para Meta‑learners (t/s/x).")

    # Comparación entre métodos (si ambos existen)
    _render_cross_method_comparison(cfg, canonical_ids)
    logging.info("EDA_algorithms finalizado.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> EDAConfig:
    p = argparse.ArgumentParser(description="EDA de resultados (GSC y Meta‑learners) con láminas tamaño carta y canonización por fecha.")
    p.add_argument("--episodes_index", type=str, default=str(_data_root() / "processed" / "episodes_index.parquet"))
    p.add_argument("--gsc_out_dir", type=str, default=None)
    p.add_argument("--meta_out_root", type=str, default=None)
    p.add_argument("--learners", type=str, default="t,s,x", help="Lista separada por coma de learners meta a renderizar.")
    p.add_argument("--figures_dir", type=str, default=str(fig_root()))
    p.add_argument("--orientation", type=str, default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--font_size", type=int, default=10)
    p.add_argument("--no_grid", action="store_true", help="Desactivar grilla en las figuras.")
    p.add_argument("--max_episodes_gsc", type=int, default=None)
    p.add_argument("--max_episodes_meta", type=int, default=None)
    p.add_argument("--no_pdf", action="store_true", help="No exportar reportes PDF multipágina.")
    p.add_argument("--plot_no_smooth", action="store_true", help="No suavizar observado para plot.")
    p.add_argument("--plot_smooth_kind", type=str, default="ema", choices=["ema", "ma"])
    p.add_argument("--plot_ema_span", type=float, default=7.0)
    p.add_argument("--plot_ma_window", type=int, default=7)
    p.add_argument("--strict_unique_dates", action="store_true", help="Falla si persisten duplicados de fecha tras canonizar.")
    p.add_argument("--exp_tag", type=str, default="A_base")

    a = p.parse_args()
    learners = tuple([s.strip() for s in a.learners.split(",") if s.strip()])
    cfg = EDAConfig(
        episodes_index=Path(a.episodes_index),
        gsc_out_dir=Path(a.gsc_out_dir) if a.gsc_out_dir else (_fig_root() / a.exp_tag / "gsc"),
        meta_out_root=Path(a.meta_out_root) if a.meta_out_root else (_fig_root() / a.exp_tag / "meta"),
        meta_learners=learners,
        figures_dir=Path(a.figures_dir),
        orientation=a.orientation,
        dpi=int(a.dpi),
        style="academic",
        font_size=int(a.font_size),
        grid=(not a.no_grid),
        max_episodes_gsc=a.max_episodes_gsc,
        max_episodes_meta=a.max_episodes_meta,
        export_pdf=(not a.no_pdf),
        plot_sales_smooth=(not a.plot_no_smooth),
        plot_smooth_kind=a.plot_smooth_kind,
        plot_ema_span=float(a.plot_ema_span),
        plot_ma_window=int(a.plot_ma_window),
        strict_unique_dates=a.strict_unique_dates,
        exp_tag=a.exp_tag,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)