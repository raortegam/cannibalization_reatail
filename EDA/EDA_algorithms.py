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
# EDA/EDA_algorithms.py
from __future__ import annotations

"""
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

# EDA/EDA_algorithms.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    # Entradas obligatorias
    episodes_index: Path

    # Salidas de algoritmos
    gsc_out_dir: Optional[Path] = None         # salida de GSC (debe contener cf_series/)
    meta_out_root: Optional[Path] = None       # raíz de meta (subcarpetas t/, s/, x/)

    # Compatibilidad hacia atrás (algunos scripts usan gsc_dir)
    gsc_dir: Optional[Path] = None

    # Selección/estética
    meta_learners: Tuple[str, ...] = ("t", "s", "x")
    figures_dir: Path = Path("figures")
    orientation: str = "landscape"             # "landscape" | "portrait"
    dpi: int = 300
    style: str = "academic"
    font_size: int = 10
    grid: bool = True
    max_episodes_gsc: Optional[int] = None
    max_episodes_meta: Optional[int] = None
    export_pdf: bool = True

    def __post_init__(self):
        # Normalización de rutas/strings
        self.episodes_index = Path(self.episodes_index)
        if self.meta_out_root is not None:
            self.meta_out_root = Path(self.meta_out_root)
        if self.figures_dir is not None:
            self.figures_dir = Path(self.figures_dir)

        # Alias: si llega gsc_dir y no llega gsc_out_dir, usarlo
        if self.gsc_out_dir is None and self.gsc_dir is not None:
            self.gsc_out_dir = Path(self.gsc_dir)
        if self.gsc_out_dir is not None:
            self.gsc_out_dir = Path(self.gsc_out_dir)

        # Saneo de meta_learners
        ml = []
        for m in (self.meta_learners or ()):
            m = (m or "").strip().lower()
            if m in {"t", "s", "x"}:
                ml.append(m)
        self.meta_learners = tuple(ml) if ml else tuple()

        # Tamaños y estilo
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


def _as_dt(s: pd.Series) -> pd.Series:
    if s.dtype.kind == "M":
        return s.dt.tz_localize(None)
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_true - y_pred
    denom = max(1.0, float(np.sqrt(np.mean(np.square(y_true)))))
    return float(np.sqrt(np.mean(np.square(e))) / denom)


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
    rename_map = {"effect_sum": "att_sum", "effect_mean": "att_mean"}
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    keep = [c for c in ["episode_id", "rmspe_pre", "att_sum", "att_mean"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
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
        # Normalizar
        rename_map = {"effect_sum": "att_sum", "effect_mean": "att_mean"}
        for k, v in rename_map.items():
            if k in df.columns and v not in df.columns:
                df = df.rename(columns={k: v})
        keep = [c for c in ["episode_id", "rmspe_pre", "att_sum", "att_mean"] if c in df.columns]
        if not keep:
            continue
        df = df[keep].copy()
        df["source"] = f"meta-{lr.lower()}"
        out[lr] = df
    return out


# ---------------------------------------------------------------------
# Lectura de series contrafactuales (Meta y GSC)
# ---------------------------------------------------------------------

def _find_first_match(dirpath: Path, episode_id: str) -> Optional[Path]:
    """
    Busca un parquet de serie en dirpath/cf_series cuyo nombre contenga episode_id.
    Intenta nombres comunes primero: {ep}.parquet, {ep}_cf.parquet.
    """
    if not dirpath:
        return None
    base = Path(dirpath) / "cf_series"
    if not base.exists():
        return None
    # intentos directos
    direct = [base / f"{episode_id}.parquet", base / f"{episode_id}_cf.parquet"]
    for p in direct:
        if p.exists():
            return p
    # búsqueda laxa
    try:
        for p in base.glob(f"*{episode_id}*.parquet"):
            return p
    except Exception:
        pass
    return None


def _resolve_y_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Devuelve nombres de columnas (date, sales, y0_hat).
    Intenta mapear variantes comunes.
    """
    # fecha
    date_col = "date" if "date" in df.columns else ("ds" if "ds" in df.columns else df.columns[0])
    # observado
    for c in ["sales", "y", "unit_sales", "obs", "observed"]:
        if c in df.columns:
            sales_col = c
            break
    else:
        # fallback: la primera numérica distinta a la fecha
        candid = [c for c in df.columns if c != date_col]
        sales_col = candid[0]
    # cf / mu0
    for c in ["mu0_hat", "y0_hat", "y_cf", "y_synth", "counterfactual", "cf", "y_hat0", "y_hat"]:
        if c in df.columns:
            y0_col = c
            break
    else:
        # si no existe, crear columna nula para evitar crash (se avisará en plot)
        y0_col = "__missing_cf__"
        df[y0_col] = np.nan
    return date_col, sales_col, y0_col


def _read_meta_cf(meta_root: Path, learner: str, episode_id: str) -> pd.DataFrame:
    p = _find_first_match(meta_root / learner, episode_id)
    return pd.read_parquet(p) if p and p.exists() else pd.DataFrame()


def _read_gsc_cf(gsc_root: Path, episode_id: str) -> pd.DataFrame:
    p = _find_first_match(gsc_root, episode_id)
    return pd.read_parquet(p) if p and p.exists() else pd.DataFrame()


def _compute_metrics_from_series(df: pd.DataFrame,
                                 date_col: str,
                                 sales_col: str,
                                 y0_col: str,
                                 treat_start: pd.Timestamp) -> Dict[str, float]:
    """Calcula RMSPE en pre y ATT (sum/mean) en post."""
    dt = _as_dt(df[date_col])
    y = pd.to_numeric(df[sales_col], errors="coerce").to_numpy(float)
    y0 = pd.to_numeric(df[y0_col], errors="coerce").to_numpy(float)

    pre = dt < treat_start
    post = dt >= treat_start

    rmspe_pre = _rmspe(y[pre], y0[pre]) if np.any(pre) else np.nan
    effect_post = (y - y0)[post] if np.any(post) else np.array([], float)

    return {
        "rmspe_pre": float(rmspe_pre) if np.isfinite(rmspe_pre) else np.nan,
        "att_sum": float(np.nansum(effect_post)) if effect_post.size else np.nan,
        "att_mean": float(np.nanmean(effect_post)) if effect_post.size else np.nan,
        "n_pre": int(np.count_nonzero(pre)),
        "n_post": int(np.count_nonzero(post)),
    }


# ---------------------------------------------------------------------
# Gráficos simples (sin seaborn)
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
    ax.axvline(0.0, linewidth=0.6)
    ax.axhline(0.0, linewidth=0.6)


# ---------------------------------------------------------------------
# Páginas de series por episodio (Observado vs CF + efectos)
# ---------------------------------------------------------------------

def _plot_episode_page(
    fig: plt.Figure,
    df: pd.DataFrame,
    title: str,
    treat_start: pd.Timestamp,
    legend_loc: str = "upper left",
    metrics_box_loc: str = "upper right",
    dpi: int = 300
) -> None:
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.6], hspace=0.12)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])

    if df.empty:
        ax_top.text(0.5, 0.5, "Serie no disponible", ha="center", va="center", transform=ax_top.transAxes)
        ax_top.set_title(title)
        return

    # Columnas
    date_col, sales_col, y0_col = _resolve_y_cols(df)
    df = df.copy()
    df[date_col] = _as_dt(df[date_col])
    df = df.sort_values(date_col)

    # Series principales
    x = df[date_col]
    y = pd.to_numeric(df[sales_col], errors="coerce").astype(float)
    y0 = pd.to_numeric(df[y0_col], errors="coerce").astype(float)

    # Plot superior: Observado vs Contrafactual
    ax_top.plot(x, y, label="Observado", linewidth=1.3)
    if y0.notna().any():
        ax_top.plot(x, y0, label="Contrafactual", linestyle="--", linewidth=1.2)
    ax_top.axvline(pd.to_datetime(treat_start), linestyle=":", linewidth=1.0)

    ax_top.set_title(title)
    ax_top.legend(loc=legend_loc, frameon=True)

    # Métricas a la DERECHA (caja flotante)
    try:
        mets = _compute_metrics_from_series(df, date_col, sales_col, y0_col, pd.to_datetime(treat_start))
        text = (f"RMSPE_pre: {mets['rmspe_pre']:.2f}\n"
                f"ATT_sum: {mets['att_sum']:.2f}\n"
                f"ATT_mean: {mets['att_mean']:.2f}\n"
                f"n_pre: {mets['n_pre']} | n_post: {mets['n_post']}")
    except Exception:
        text = "Métricas no disponibles"
    bbox = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, linewidth=0.8)
    # coords en ejes para ubicar a la derecha
    ax_top.text(0.98, 0.02, text, ha="right", va="bottom", transform=ax_top.transAxes, bbox=bbox)

    # Plot inferior: efecto y acumulado (post)
    effect = (y - y0) if y0.notna().any() else pd.Series(np.nan, index=df.index)
    is_post = df[date_col] >= pd.to_datetime(treat_start)
    effect_cum = effect.where(is_post, 0.0).cumsum()

    ax_bot.plot(x, effect, label="Efecto (Y - Ŷ₀)", linewidth=1.2)
    ax_bot.axvline(pd.to_datetime(treat_start), linestyle=":", linewidth=1.0)

    ax2 = ax_bot.twinx()
    ax2.plot(x, effect_cum, linestyle="--", linewidth=1.1, label="Efecto acumulado")

    ax_bot.set_ylabel("Efecto")
    ax2.set_ylabel("Acumulado")

    # Leyendas separadas para evitar solapes
    #   - Mantén la del panel superior (observado/cf) a la izquierda.
    #   - En el panel inferior, sólo la curva acumulada a la derecha.
    h2, l2 = ax2.get_legend_handles_labels()
    if h2:
        ax2.legend(h2, l2, loc="upper right", frameon=True)

    fig.tight_layout()


# ---------------------------------------------------------------------
# EDA principal
# ---------------------------------------------------------------------

def run(cfg: EDAConfig) -> None:
    logger = logging.getLogger("EDA_algorithms")
    logger.setLevel(logging.INFO)

    # Carpetas de salida
    fig_root = _ensure_dir(Path(cfg.figures_dir))
    fig_dir = _ensure_dir(fig_root / "algorithms")
    tbl_dir = _ensure_dir(fig_root / "tables")

    # Cargar episodios (referencia/orden)
    episodes = _safe_read_parquet(Path(cfg.episodes_index))
    if "episode_id" not in episodes.columns:
        raise ValueError("episodes_index.parquet no tiene columna 'episode_id'.")
    if "treat_start" not in episodes.columns:
        raise ValueError("episodes_index.parquet no tiene columna 'treat_start'.")

    episodes["treat_start"] = _as_dt(episodes["treat_start"])
    ordered_ep_list = episodes["episode_id"].astype(str).tolist()

    # --- Métricas de GSC + Meta (para resúmenes) ---
    gsc_df = _load_gsc_metrics(cfg.gsc_out_dir)
    meta_dict = _load_meta_metrics(cfg.meta_out_root, cfg.meta_learners)

    # ---------------- Cobertura por episodio ----------------
    cover = episodes[["episode_id"]].drop_duplicates().copy()
    cover["has_gsc"] = cover["episode_id"].isin(gsc_df["episode_id"]) if not gsc_df.empty else False
    for lr, mdf in meta_dict.items():
        cover[f"has_meta_{lr}"] = cover["episode_id"].isin(mdf["episode_id"]) if not mdf.empty else False
    meta_flags = [c for c in cover.columns if c.startswith("has_meta_")]
    any_cols = ["has_gsc"] + meta_flags
    cover["available"] = cover[any_cols].any(axis=1) if any_cols else False

    cover_path = tbl_dir / "eda_algorithms_coverage.parquet"
    cover.to_parquet(cover_path, index=False)

    # ---------------- Tablas consolidadas (para resúmenes) ----------------
    stacks: List[pd.DataFrame] = []
    if not gsc_df.empty:
        stacks.append(gsc_df)
    for lr, mdf in meta_dict.items():
        if not mdf.empty:
            stacks.append(mdf)
    all_mets = pd.concat(stacks, ignore_index=True) if stacks else pd.DataFrame()

    # ---------------- Gráficos de resumen ----------------
    w, h = _figsize_from_orientation(cfg.orientation)
    pdf_summary_path = fig_dir / "eda_algorithms_summary.pdf" if cfg.export_pdf else None
    pdf_sum = PdfPages(pdf_summary_path) if pdf_summary_path else None

    if not all_mets.empty:
        # 1) Distribuciones de ATT por fuente
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

        # 2) Scatter RMSPE(pre) vs ATT_sum
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

    # ---------------- Páginas de series por episodio ----------------
    # Definimos LA LISTA DE EPISODIOS A PLOTear segun meta y MANTENEMOS ESE ORDEN para GSC.
    # Prioridad del learner de referencia: X, luego T, luego S.
    learners_pref = [lr for lr in ["x", "t", "s"] if lr in set(cfg.meta_learners)]
    ref_lr = None
    for lr in learners_pref or []:
        if (cfg.meta_out_root and (cfg.meta_out_root / lr / "cf_series").exists()):
            ref_lr = lr
            break
    # Si no hay carpeta de meta, usamos todo el episodes_index como referencia
    if ref_lr is None:
        ep_ref_list = ordered_ep_list
    else:
        cf_dir = cfg.meta_out_root / ref_lr / "cf_series"
        # Episodios que tenemos en meta (por archivos)
        meta_files_eps = set([p.stem.replace("_cf", "") for p in cf_dir.glob("*.parquet")])
        # mantener ORDEN según episodes_index
        ep_ref_list = [ep for ep in ordered_ep_list if ep in meta_files_eps]

    # aplicar límites si se solicitaron
    if cfg.max_episodes_meta is not None:
        ep_ref_list = ep_ref_list[: int(cfg.max_episodes_meta)]
    if cfg.max_episodes_gsc is not None:
        # queremos que GSC siga el MISMO orden/episodios; no recortamos distinto
        ep_ref_list = ep_ref_list[: int(cfg.max_episodes_gsc)]

    if not ep_ref_list:
        logger.warning("No se encontraron episodios de referencia para renderizar series (¿meta/cf_series vacío?).")

    pdf_series_path = fig_dir / "eda_algorithms_series.pdf" if cfg.export_pdf else None
    pdf_ser = PdfPages(pdf_series_path) if pdf_series_path else None

    # Render por episodio: primero cada meta-learner, luego GSC, EN EL MISMO ORDEN DE ep_ref_list
    for ep_id in ep_ref_list:
        # Tratamiento start por episodio
        row = episodes.loc[episodes["episode_id"] == ep_id]
        tstart = _as_dt(row["treat_start"]).iloc[0] if not row.empty else None

        # META: por cada learner solicitado
        for lr in cfg.meta_learners:
            title = f"Meta-{lr.upper()} — Episodio {ep_id}"
            df_cf = _read_meta_cf(cfg.meta_out_root, lr, ep_id) if cfg.meta_out_root else pd.DataFrame()
            fig = plt.figure(figsize=(w, h))
            _plot_episode_page(fig, df_cf, title, tstart, legend_loc="upper left", metrics_box_loc="upper right", dpi=cfg.dpi)
            if pdf_ser:
                pdf_ser.savefig(fig, dpi=cfg.dpi)
            # opcional PNG por página:
            png = fig_dir / f"series__meta-{lr}_{ep_id}.png"
            fig.savefig(png, dpi=cfg.dpi)
            plt.close(fig)

        # GSC: MISMO episodio en el MISMO orden
        title = f"GSC — Episodio {ep_id}"
        df_gsc = _read_gsc_cf(cfg.gsc_out_dir, ep_id) if cfg.gsc_out_dir else pd.DataFrame()
        fig = plt.figure(figsize=(w, h))
        _plot_episode_page(fig, df_gsc, title, tstart, legend_loc="upper left", metrics_box_loc="upper right", dpi=cfg.dpi)
        if pdf_ser:
            pdf_ser.savefig(fig, dpi=cfg.dpi)
        png = fig_dir / f"series__gsc_{ep_id}.png"
        fig.savefig(png, dpi=cfg.dpi)
        plt.close(fig)

    if pdf_ser:
        pdf_ser.close()

    # ---------------- Top episodios (por |ATT_sum|) ----------------
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

    # Mensajes finales
    logger.info("EDA_algorithms completado.")
    logger.info("Cobertura guardada en: %s", str(cover_path))
    if pdf_summary_path:
        logger.info("Resumen PDF: %s", str(pdf_summary_path))
    if pdf_series_path:
        logger.info("Series PDF: %s", str(pdf_series_path))