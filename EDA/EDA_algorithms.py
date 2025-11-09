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
from typing import Dict, List, Optional, Tuple

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
        keep = [c for c in ["episode_id", "rmspe_pre", "att_sum", "att_mean"] if c in df.columns]
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

    # ---- Panel inferior: Efecto y acumulado
    if "effect" in d.columns:
        ax_bot.plot(d["date"], d["effect"], label="Efecto (Y - Y0)", linewidth=1.2)
        ax_bot.set_ylabel("Efecto")
        # Acumulado en eje secundario - solo post-tratamiento
        if "cum_effect" in d.columns:
            # Recalcular acumulado: 0 antes del tratamiento, acumular solo después
            d_plot = d.copy()
            d_plot["cum_effect_plot"] = d_plot["effect"].where(d_plot["date"] >= treat_start, 0.0).cumsum()
            ax2 = ax_bot.twinx()
            ax2.plot(d_plot["date"], d_plot["cum_effect_plot"], linestyle="--", label="Efecto acumulado", linewidth=1.0)
            ax2.set_ylabel("Acumulado")
            
            # Alinear el cero de ambos ejes
            y1_min, y1_max = ax_bot.get_ylim()
            y2_min, y2_max = ax2.get_ylim()
            
            # Para alinear el cero, necesitamos que la proporción sea la misma:
            # |y1_min| / y1_max = |y2_min| / y2_max
            # Ajustamos el eje 2 para que coincida con el eje 1
            
            if y1_min < 0 and y1_max > 0:
                # El eje 1 cruza el cero - usamos su proporción
                ratio = abs(y1_min) / y1_max
                
                if y2_max > 0:
                    # Ajustar y2_min para que tenga la misma proporción
                    new_y2_min = -y2_max * ratio
                    ax2.set_ylim(new_y2_min, y2_max)
                elif y2_min < 0:
                    # Ajustar y2_max para que tenga la misma proporción
                    new_y2_max = abs(y2_min) / ratio
                    ax2.set_ylim(y2_min, new_y2_max)
        
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
# EDA principal
# ---------------------------------------------------------------------

def run(cfg: EDAConfig) -> None:
    logger = logging.getLogger("EDA_algorithms")
    logger.setLevel(logging.INFO)

    # Carpetas de salida
    fig_root = _ensure_dir(Path(cfg.figures_dir))
    fig_dir = _ensure_dir(fig_root / "algorithms")
    tbl_dir = _ensure_dir(fig_root / "tables")

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

    # Mensajes finales
    logger.info("EDA_algorithms completado.")
    logger.info("Cobertura guardada en: %s", str(cover_path))
    if pdf_sum_path:
        logger.info("Resumen PDF: %s", str(pdf_sum_path))
    logger.info("PDFs de series generados:")
    if gsc_series and cfg.export_pdf:
        logger.info("  - series_gsc.pdf")
    for lr in meta_series.keys():
        if meta_series[lr] and cfg.export_pdf:
            logger.info("  - series_meta_%s.pdf", lr)