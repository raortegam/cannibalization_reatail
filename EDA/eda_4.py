# -*- coding: utf-8 -*-
"""
eda_outputs.py
==============

Exploratory Data Analysis (EDA) para los artefactos emitidos por pre_algorithm.py.

Produce láminas en formato carta (8.5 x 11 in) con:
  1) Resumen de episodios y donantes (episodes_index + donor_quality)
  2) Calidad de donantes: distribución de métricas y umbrales (donor_quality)
  3) Panorama de panel/meta: actividad, promociones y disponibilidad (meta/all_units o panel_features)
  4) Lámina por episodio de ejemplo: trayectoria de la víctima vs. promedio de donantes (gsc/<episode_id>.parquet)

Uso (ejemplos):
---------------
python -m src.analysis.eda_outputs \
  --processed ./data/processed \
  --out ./reports/eda \
  --dpi 300 --orientation portrait \
  --promo-thresh 0.02 --avail-thresh 0.90 \
  --limit-episodes 1

python -m src.analysis.eda_outputs \
  --episodes-index ./data/processed/episodes_index.parquet \
  --donor-quality ./data/processed/gsc/donor_quality.parquet \
  --meta-all ./data/processed/meta/all_units.parquet \
  --gsc-dir ./data/processed/gsc \
  --out ./reports/eda

Requisitos:
-----------
- pandas, numpy, matplotlib
- Soporte parquet (pyarrow o fastparquet)

Autor: generado por GPT-5 Pro.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter


# ---------------------------------------------------------------------
# Configuración y utilidades
# ---------------------------------------------------------------------

@dataclass
class EDAConfig:
    # Entradas (pueden dejarse en None si se usa --processed)
    processed_dir: Optional[Path] = None
    episodes_index: Optional[Path] = None
    donor_quality: Optional[Path] = None
    meta_all: Optional[Path] = None
    panel_features: Optional[Path] = None
    gsc_dir: Optional[Path] = None

    # Salida
    out_dir: Path = Path("./reports/eda")
    dpi: int = 300
    orientation: str = "portrait"  # 'portrait' o 'landscape'

    # Parámetros analíticos
    promo_thresh: float = 0.02
    avail_thresh: float = 0.90
    limit_episodes: int = 1  # nº de láminas de episodios a renderizar
    example_episode_id: Optional[str] = None  # para forzar un episodio específico

    # Logging
    log_level: str = "INFO"

# --- Helpers de rutas robustos a str | Path ---

PathLike = Union[str, Path]

def _to_path(p: Optional[PathLike]) -> Optional[Path]:
    if p is None:
        return None
    return p if isinstance(p, Path) else Path(p)

def _coerce_all_paths(cfg: "EDAConfig") -> "EDAConfig":
    # Entradas
    cfg.processed_dir  = _to_path(cfg.processed_dir)
    cfg.episodes_index = _to_path(cfg.episodes_index)
    cfg.donor_quality  = _to_path(cfg.donor_quality)
    cfg.meta_all       = _to_path(cfg.meta_all)
    cfg.panel_features = _to_path(cfg.panel_features)
    cfg.gsc_dir        = _to_path(cfg.gsc_dir)
    # Salida
    cfg.out_dir        = _to_path(cfg.out_dir) or Path("./reports/eda")
    return cfg

def _setup_logging(level: str = "INFO"):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def _resolve_paths(cfg: EDAConfig) -> EDAConfig:
    """
    Completa rutas a partir de processed_dir si no se proveen explícitamente.
    Acepta str | Path en los atributos de cfg.
    """
    # Asegura que todo en cfg sean Path (o None)
    cfg = _coerce_all_paths(cfg)

    if cfg.processed_dir is not None:
        p = cfg.processed_dir  # Path garantizado
        cfg.episodes_index  = cfg.episodes_index  or (p / "episodes_index.parquet")
        cfg.donor_quality   = cfg.donor_quality   or (p / "gsc" / "donor_quality.parquet")
        cfg.meta_all        = cfg.meta_all        or (p / "meta" / "all_units.parquet")
        cfg.panel_features  = cfg.panel_features  or (p / "intermediate" / "panel_features.parquet")
        cfg.gsc_dir         = cfg.gsc_dir         or (p / "gsc")

    # Asegurar directorio de salida
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Log informativo: rutas resueltas
    logging.info("Rutas resueltas EDA4:")
    logging.info("  processed_dir  : %s", cfg.processed_dir)
    logging.info("  episodes_index : %s", cfg.episodes_index)
    logging.info("  donor_quality  : %s", cfg.donor_quality)
    logging.info("  meta_all       : %s", cfg.meta_all)
    logging.info("  panel_features : %s", cfg.panel_features)
    logging.info("  gsc_dir        : %s", cfg.gsc_dir)
    logging.info("  out_dir        : %s", cfg.out_dir)

    return cfg


def _academic_style():
    """
    Ajustes visuales sobrios para publicación académica.
    """
    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 9.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.autolayout": False,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.25,
    })


def _fig_letter_size(orientation: str = "portrait") -> Tuple[float, float]:
    """
    Dimensiones de hoja carta en pulgadas.
    """
    if orientation.lower() == "landscape":
        return (11.0, 8.5)
    return (8.5, 11.0)


def _save_letter(fig: mpl.figure.Figure, path: Path, orientation: str, dpi: int):
    w, h = _fig_letter_size(orientation)
    fig.set_size_inches(w, h)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Lámina guardada: {path}")


def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _exists(p: Optional[PathLike]) -> bool:
    return (p is not None) and Path(p).exists()


# ---------------------------------------------------------------------
# Carga de datos con tolerancia a ausencias
# ---------------------------------------------------------------------

def _try_read_parquet(path: Optional[Path], name: str) -> Optional[pd.DataFrame]:
    if not _exists(path):
        logging.warning(f"No se encontró {name}: {path}")
        return None
    try:
        df = pd.read_parquet(path)
        logging.info(f"{name} cargado: {path} ({df.shape[0]:,} filas, {df.shape[1]} cols)")
        return df
    except Exception as e:
        logging.error(f"Error leyendo {name} en {path}: {e}")
        return None


def _read_episodes_index(cfg: EDAConfig) -> Optional[pd.DataFrame]:
    df = _try_read_parquet(cfg.episodes_index, "episodes_index")
    if df is not None:
        for c in ["pre_start", "treat_start", "post_start", "post_end"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _read_donor_quality(cfg: EDAConfig) -> Optional[pd.DataFrame]:
    return _try_read_parquet(cfg.donor_quality, "donor_quality")


def _read_meta_or_panel(cfg: EDAConfig) -> Optional[pd.DataFrame]:
    df = None
    if _exists(cfg.meta_all):
        df = _try_read_parquet(cfg.meta_all, "meta_all")
    if df is None and _exists(cfg.panel_features):
        df = _try_read_parquet(cfg.panel_features, "panel_features")
    if df is not None:
        df = _ensure_datetime(df, "date")
    return df


def _read_gsc_episode(cfg: EDAConfig, episode_id: str) -> Optional[pd.DataFrame]:
    if cfg.gsc_dir is None:
        return None
    path = cfg.gsc_dir / f"{episode_id}.parquet"
    df = _try_read_parquet(path, f"gsc/{episode_id}")
    if df is not None:
        df = _ensure_datetime(df, "date")
    return df


# ---------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------

def _format_time_axis(ax: mpl.axes.Axes):
    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def fig_episodes_summary(episodes: pd.DataFrame, out_path: Path, cfg: EDAConfig):
    """
    Lámina 1: resumen de episodios y donantes.
    """
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.2, 1.0, 1.2], hspace=0.6, wspace=0.35)

    # --- Panel A: Scatter donors_input vs donors_kept ---
    axA = fig.add_subplot(gs[0, :])
    x = episodes["n_donors_input"].astype(float)
    y = episodes["n_donors_kept"].astype(float)
    axA.scatter(x, y, alpha=0.7)
    axA.set_title("Capacidad de retención de donantes por episodio")
    axA.set_xlabel("Donantes candidatos (input)")
    axA.set_ylabel("Donantes retenidos (tras filtros)")
    if len(x) > 1:
        # Línea identidad
        lim = max(x.max(), y.max()) + 0.5
        axA.plot([0, lim], [0, lim], linestyle="--", alpha=0.6)
    # Resumen estadístico
    ratio = (episodes["n_donors_kept"] / episodes["n_donors_input"].replace(0, np.nan)).dropna()
    subtitle = (f"Episodios: {episodes.shape[0]} | Mediana retención: {np.nanmedian(ratio):.2f} "
                f"| IQR: {np.nanpercentile(ratio, 75) - np.nanpercentile(ratio, 25):.2f}")
    axA.text(0.01, 1.02, subtitle, transform=axA.transAxes, va="bottom", ha="left")

    # --- Panel B: Histograma de actores por episodio (filas en GSC/meta) ---
    axB = fig.add_subplot(gs[1, 0])
    if "n_meta_rows" in episodes.columns:
        axB.hist(episodes["n_meta_rows"], bins=20, alpha=0.9)
        axB.set_title("Tamaño de panel por episodio (Meta)")
        axB.set_xlabel("Filas en panel meta por episodio")
        axB.set_ylabel("Frecuencia")
    else:
        axB.axis("off")
        axB.text(0.5, 0.5, "Sin columna n_meta_rows", ha="center", va="center")

    # --- Panel C: Duración de ventanas ---
    axC = fig.add_subplot(gs[1, 1])
    try:
        pre_days = (episodes["treat_start"] - episodes["pre_start"]).dt.days
        post_days = (episodes["post_end"] - episodes["post_start"]).dt.days
        axC.hist(pre_days.dropna(), bins=20, alpha=0.9, label="Pre-tratamiento")
        axC.hist(post_days.dropna(), bins=20, alpha=0.6, label="Post-tratamiento")
        axC.set_title("Distribución de duraciones de ventana")
        axC.set_xlabel("Días")
        axC.set_ylabel("Frecuencia")
        axC.legend()
    except Exception:
        axC.axis("off")
        axC.text(0.5, 0.5, "No fue posible calcular duraciones", ha="center", va="center")

    # --- Panel D: Tabla de resumen ---
    axD = fig.add_subplot(gs[2, :])
    axD.axis("off")
    summary = {
        "Episodios": f"{episodes.shape[0]}",
        "Mediana donantes (input)": f"{episodes['n_donors_input'].median():.1f}",
        "Mediana donantes (retenidos)": f"{episodes['n_donors_kept'].median():.1f}",
        "Pre (mediana días)": f"{np.nanmedian(pre_days) if 'pre_days' in locals() else 'NA'}",
        "Post (mediana días)": f"{np.nanmedian(post_days) if 'post_days' in locals() else 'NA'}"
    }
    table_data = [[k, v] for k, v in summary.items()]
    tbl = axD.table(cellText=table_data, colLabels=["Métrica", "Valor"], loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.2)

    fig.suptitle("Resumen de episodios y estructura muestral", x=0.02, ha="left", y=0.98, fontsize=13)
    fig.text(0.02, 0.01, "Nota: Retención tras filtros de promoción y disponibilidad; figura en formato carta.", ha="left")
    _save_letter(fig, out_path, cfg.orientation, cfg.dpi)


def fig_donor_quality(quality: pd.DataFrame, out_path: Path, cfg: EDAConfig):
    """
    Lámina 2: calidad de donantes — distribuciones y umbrales.
    """
    df = quality.copy()
    if "is_victim" in df.columns:
        df = df.loc[df["is_victim"] == False].copy()

    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.6, wspace=0.35)

    # --- Promo share ---
    ax1 = fig.add_subplot(gs[0, 0])
    if "promo_share" in df.columns:
        ax1.hist(df["promo_share"].dropna(), bins=30, alpha=0.9)
        ax1.axvline(cfg.promo_thresh, linestyle="--", alpha=0.8)
        ax1.set_title("Distribución de proporción de días en promoción (donantes)")
        ax1.set_xlabel("Proporción de días con promoción")
        ax1.set_ylabel("Frecuencia")
        # % sobre umbral
        above = (df["promo_share"] > cfg.promo_thresh).mean()
        ax1.text(0.02, 0.92, f"Sobre umbral: {100*above:.1f}%", transform=ax1.transAxes)
    else:
        ax1.axis("off")
        ax1.text(0.5, 0.5, "Sin columna promo_share", ha="center", va="center")

    # --- Availability share ---
    ax2 = fig.add_subplot(gs[0, 1])
    if "avail_share" in df.columns:
        ax2.hist(df["avail_share"].dropna(), bins=30, alpha=0.9)
        ax2.axvline(cfg.avail_thresh, linestyle="--", alpha=0.8)
        ax2.set_title("Distribución de disponibilidad (donantes)")
        ax2.set_xlabel("Fracción de días disponibles")
        ax2.set_ylabel("Frecuencia")
        below = (df["avail_share"] < cfg.avail_thresh).mean()
        ax2.text(0.02, 0.92, f"Bajo umbral: {100*below:.1f}%", transform=ax2.transAxes)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Sin columna avail_share", ha="center", va="center")

    # --- Motivos de descarte ---
    ax3 = fig.add_subplot(gs[1, :])
    if "keep" in df.columns and "reason" in df.columns:
        reasons = df.loc[df["keep"] == False, "reason"].fillna("sin_razón")
        # normalizar etiquetas compuestas
        exploded = reasons.str.split(";").explode().str.strip()
        counts = exploded.value_counts().head(10)
        ax3.bar(counts.index.astype(str), counts.values)
        ax3.set_title("Principales motivos de descarte de donantes (Top 10)")
        ax3.set_ylabel("Frecuencia")
        ax3.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "Sin columnas keep/reason", ha="center", va="center")

    fig.suptitle("Calidad de donantes: distribución de métricas y filtros", x=0.02, ha="left", y=0.98, fontsize=13)
    fig.text(0.02, 0.01, f"Umbrales: promoción ≤ {cfg.promo_thresh:.2f}, disponibilidad ≥ {cfg.avail_thresh:.2f}.", ha="left")
    _save_letter(fig, out_path, cfg.orientation, cfg.dpi)


def fig_meta_overview(df: pd.DataFrame, out_path: Path, cfg: EDAConfig):
    """
    Lámina 3: panorama temporal agregado (ventas, promociones, disponibilidad, proxies).
    Usa meta/all_units si está disponible; de lo contrario panel_features.
    """
    d = df.copy()
    d = _ensure_datetime(d, "date")

    # Inferencias robustas
    has_sales = "sales" in d.columns
    has_promo = "onpromotion" in d.columns
    has_avail = "available_A" in d.columns
    has_ow = "Ow" in d.columns
    has_fsw = "Fsw_log1p" in d.columns

    # Agregaciones por fecha
    daily = pd.DataFrame({"date": pd.to_datetime(d["date"]).dropna().unique()})
    daily = daily.sort_values("date")
    daily = daily.set_index("date")

    if has_sales:
        tmp = d.groupby("date", as_index=True)["sales"].sum().rename("sales_sum")
        daily = daily.join(tmp, how="left")
    if has_promo:
        # proporción de filas con promo>0
        tmp = d.assign(promo_flag=(d["onpromotion"] > 0).astype(float)).groupby("date")["promo_flag"].mean().rename("promo_share")
        daily = daily.join(tmp, how="left")
    if has_avail:
        tmp = d.groupby("date")["available_A"].mean().rename("avail_share")
        daily = daily.join(tmp, how="left")
    if has_ow:
        tmp = d.groupby("date")["Ow"].mean().rename("Ow_mean")
        daily = daily.join(tmp, how="left")
    if has_fsw:
        tmp = d.groupby("date")["Fsw_log1p"].mean().rename("Fsw_mean")
        daily = daily.join(tmp, how="left")

    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=4, ncols=1, hspace=0.65)

    # Panel A: Ventas agregadas
    axA = fig.add_subplot(gs[0, 0])
    if "sales_sum" in daily.columns:
        axA.plot(daily.index, daily["sales_sum"])
        axA.set_title("Ventas agregadas diarias (suma)")
        axA.set_ylabel("Unidades")
        _format_time_axis(axA)
    else:
        axA.axis("off")
        axA.text(0.5, 0.5, "Sin columna 'sales'", ha="center", va="center")

    # Panel B: Proporción en promoción
    axB = fig.add_subplot(gs[1, 0])
    if "promo_share" in daily.columns:
        axB.plot(daily.index, daily["promo_share"])
        axB.set_title("Proporción de ítems en promoción (promedio diario)")
        axB.set_ylabel("Proporción")
        _format_time_axis(axB)
    else:
        axB.axis("off")
        axB.text(0.5, 0.5, "Sin columna 'onpromotion'", ha="center", va="center")

    # Panel C: Disponibilidad
    axC = fig.add_subplot(gs[2, 0])
    if "avail_share" in daily.columns:
        axC.plot(daily.index, daily["avail_share"])
        axC.set_title("Disponibilidad promedio (A_it)")
        axC.set_ylabel("Fracción")
        _format_time_axis(axC)
    else:
        axC.axis("off")
        axC.text(0.5, 0.5, "Sin columna 'available_A'", ha="center", va="center")

    # Panel D: Proxies (Ow y Fsw)
    axD = fig.add_subplot(gs[3, 0])
    lines = 0
    if "Ow_mean" in daily.columns:
        axD.plot(daily.index, daily["Ow_mean"], label="Precio del petróleo (Ow)")
        lines += 1
    if "Fsw_mean" in daily.columns:
        axD.plot(daily.index, daily["Fsw_mean"], label="Tráfico tienda – log1p (Fsw)")
        lines += 1
    if lines > 0:
        axD.set_title("Proxies macro y de tráfico")
        _format_time_axis(axD)
        axD.legend()
    else:
        axD.axis("off")
        axD.text(0.5, 0.5, "Sin columnas 'Ow'/'Fsw_log1p'", ha="center", va="center")

    fig.suptitle("Panorama temporal agregado del panel", x=0.02, ha="left", y=0.98, fontsize=13)
    fig.text(0.02, 0.01, "Nota: estadísticas por fecha a nivel agregado; valores faltantes se dejan en blanco.", ha="left")
    _save_letter(fig, out_path, cfg.orientation, cfg.dpi)


def fig_episode_page(gsc: pd.DataFrame,
                     ep_row: Dict,
                     out_path: Path,
                     cfg: EDAConfig):
    """
    Lámina 4: episodio de ejemplo — trayectoria de víctima vs. promedio de donantes.
    """
    df = gsc.copy()
    df = _ensure_datetime(df, "date")

    # Identificación de víctima y periodo
    treated_mask = (df.get("treated_unit", 0) == 1)
    victim = df.loc[treated_mask].sort_values("date")
    donors = df.loc[~treated_mask].sort_values("date")

    # Promedio de donantes por fecha (ventas)
    donors_avg = donors.groupby("date", as_index=False)["sales"].mean().rename(columns={"sales": "sales_donors_mean"})
    merged = victim[["date", "sales"]].merge(donors_avg, on="date", how="left")

    # Fechas de ventanas
    pre_start = ep_row.get("pre_start")
    treat_start = ep_row.get("treat_start")
    post_start = ep_row.get("post_start")
    post_end = ep_row.get("post_end")

    # Métricas diagnósticas simples
    pre_mask = (merged["date"] >= pre_start) & (merged["date"] < treat_start) if (pre_start is not None and treat_start is not None) else None
    post_mask = (merged["date"] >= post_start) & (merged["date"] <= post_end) if (post_start is not None and post_end is not None) else None

    def _safe_mean(x): return float(np.nanmean(x)) if len(x) else np.nan

    diff_pre = _safe_mean(merged.loc[pre_mask, "sales"] - merged.loc[pre_mask, "sales_donors_mean"]) if pre_mask is not None else np.nan
    diff_post = _safe_mean(merged.loc[post_mask, "sales"] - merged.loc[post_mask, "sales_donors_mean"]) if post_mask is not None else np.nan
    did_naive = diff_post - diff_pre if (np.isfinite(diff_pre) and np.isfinite(diff_post)) else np.nan

    # RMSPE pre
    if pre_mask is not None and pre_mask.any():
        base = merged.loc[pre_mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = (base["sales"] - base["sales_donors_mean"]) / base["sales"].replace({0: np.nan})
        rmspe_pre = float(np.sqrt(np.nanmean((pct * 100) ** 2)))
    else:
        rmspe_pre = np.nan

    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.4, 0.8, 0.8], hspace=0.6)

    # Panel A: Serie temporal
    axA = fig.add_subplot(gs[0, 0])
    axA.plot(merged["date"], merged["sales"], label="Víctima (ventas)")
    axA.plot(merged["date"], merged["sales_donors_mean"], label="Donantes (promedio)")
    # Líneas de ventana
    for t, label in [(pre_start, "pre_start"), (treat_start, "treat_start"),
                     (post_start, "post_start"), (post_end, "post_end")]:
        if pd.notnull(t):
            axA.axvline(pd.to_datetime(t), linestyle="--", alpha=0.7)
            axA.text(pd.to_datetime(t), axA.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=8)
    axA.set_title("Trayectorias de ventas: víctima vs. promedio de donantes")
    axA.set_ylabel("Unidades")
    _format_time_axis(axA)
    axA.legend()

    # Panel B: Diferencia (víctima - donantes)
    axB = fig.add_subplot(gs[1, 0])
    axB.plot(merged["date"], merged["sales"] - merged["sales_donors_mean"])
    axB.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    axB.set_title("Diferencia diaria (víctima – prom. donantes)")
    _format_time_axis(axB)

    # Panel C: Resumen numérico
    axC = fig.add_subplot(gs[2, 0])
    axC.axis("off")
    meta_lines = [
        f"Episodio: {ep_row.get('episode_id', 'NA')}",
        f"Víctima: store {ep_row.get('j_store','?')}, item {ep_row.get('j_item','?')}",
        f"Caníbal: store {ep_row.get('i_store','?')}, item {ep_row.get('i_item','?')}",
        f"Donantes retenidos: {ep_row.get('n_donors_kept','?')} / {ep_row.get('n_donors_input','?')}",
        f"Naive DiD (post-pre): {did_naive:.2f}",
        f"RMSPE pre (donantes vs víctima): {rmspe_pre:.2f}%",
    ]
    axC.text(0.02, 0.95, "\n".join(meta_lines), va="top", ha="left", fontsize=10)

    fig.suptitle("Episodio de ejemplo — diagnóstico descriptivo", x=0.02, ha="left", y=0.98, fontsize=13)
    fig.text(0.02, 0.01, "Nota: Medidas descriptivas; no sustituyen la inferencia GSC/Meta.", ha="left")
    _save_letter(fig, out_path, cfg.orientation, cfg.dpi)


# ---------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------

def _select_example_episodes(ep_idx: pd.DataFrame, cfg: EDAConfig) -> pd.DataFrame:
    if cfg.example_episode_id:
        sel = ep_idx.loc[ep_idx["episode_id"] == cfg.example_episode_id]
        if not sel.empty:
            return sel.head(1)
        logging.warning(f"episode_id no encontrado: {cfg.example_episode_id}. Se seleccionará automáticamente.")
    # Heurística: priorizar episodios con donantes retenidos > 0
    ep_idx = ep_idx.sort_values(["n_donors_kept", "n_meta_rows"], ascending=[False, False])
    return ep_idx.head(cfg.limit_episodes)


def run(cfg: EDAConfig) -> None:
    _setup_logging(cfg.log_level)
    _academic_style()
    cfg = _resolve_paths(cfg)

    # 1) Episodes summary
    ep_idx = _read_episodes_index(cfg)
    dq = _read_donor_quality(cfg)
    meta_or_panel = _read_meta_or_panel(cfg)

    # Lámina 1
    if ep_idx is not None and not ep_idx.empty:
        fig_episodes_summary(ep_idx, cfg.out_dir / "01_resumen_episodios.png", cfg)
    else:
        logging.warning("No se generó '01_resumen_episodios.png' por ausencia de episodes_index.")

    # Lámina 2
    if dq is not None and not dq.empty:
        fig_donor_quality(dq, cfg.out_dir / "02_calidad_donantes.png", cfg)
    else:
        logging.warning("No se generó '02_calidad_donantes.png' por ausencia de donor_quality.")

    # Lámina 3
    if meta_or_panel is not None and not meta_or_panel.empty:
        fig_meta_overview(meta_or_panel, cfg.out_dir / "03_panorama_panel.png", cfg)
    else:
        logging.warning("No se generó '03_panorama_panel.png' por ausencia de meta/all_units o panel_features.")

    # Lámina 4: episodio(s) de ejemplo
    if ep_idx is not None and not ep_idx.empty and cfg.gsc_dir is not None and cfg.gsc_dir.exists():
        sel = _select_example_episodes(ep_idx, cfg)
        for _, row in sel.iterrows():
            ep_id = row["episode_id"]
            gsc = _read_gsc_episode(cfg, ep_id)
            if gsc is not None and not gsc.empty:
                out = cfg.out_dir / f"04_episodio_{ep_id}.png"
                fig_episode_page(gsc, row.to_dict(), out, cfg)
            else:
                logging.warning(f"Omitida lámina de episodio {ep_id}: no se encontró el panel GSC.")
    else:
        logging.warning("No se generaron láminas de episodios por falta de índice o carpeta GSC.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> EDAConfig:
    p = argparse.ArgumentParser(
        description="Generación de EDA (láminas formato carta) sobre los outputs de pre_algorithm.py."
    )

    # Rutas
    p.add_argument("--processed", type=str, default=None, help="Directorio base de salidas (data/processed).")
    p.add_argument("--episodes-index", type=str, default=None, help="Ruta a episodes_index.parquet.")
    p.add_argument("--donor-quality", type=str, default=None, help="Ruta a gsc/donor_quality.parquet.")
    p.add_argument("--meta-all", type=str, default=None, help="Ruta a meta/all_units.parquet.")
    p.add_argument("--panel-features", type=str, default=None, help="Ruta a intermediate/panel_features.parquet.")
    p.add_argument("--gsc-dir", type=str, default=None, help="Directorio con paneles gsc/<episode_id>.parquet.")
    p.add_argument("--out", type=str, default="./reports/eda", help="Directorio de salida para las láminas.")

    # Parámetros de render y análisis
    p.add_argument("--dpi", type=int, default=300, help="Resolución de guardado (DPI).")
    p.add_argument("--orientation", type=str, default="portrait", choices=["portrait", "landscape"], help="Orientación de la lámina.")
    p.add_argument("--promo-thresh", type=float, default=0.02, help="Umbral de proporción de días en promoción para donantes.")
    p.add_argument("--avail-thresh", type=float, default=0.90, help="Umbral de disponibilidad mínima para donantes.")
    p.add_argument("--limit-episodes", type=int, default=1, help="Número de episodios a graficar como ejemplo.")
    p.add_argument("--example-episode-id", type=str, default=None, help="episode_id específico para la lámina de ejemplo.")
    p.add_argument("--log-level", type=str, default="INFO", help="Nivel de logging (DEBUG, INFO, WARNING, ERROR).")

    args = p.parse_args()
    cfg = EDAConfig(
        processed_dir=Path(args.processed) if args.processed else None,
        episodes_index=Path(args.episodes_index) if args.episodes_index else None,
        donor_quality=Path(args.donor_quality) if args.donor_quality else None,
        meta_all=Path(args.meta_all) if args.meta_all else None,
        panel_features=Path(args.panel_features) if args.panel_features else None,
        gsc_dir=Path(args.gsc_dir) if args.gsc_dir else None,
        out_dir=Path(args.out) if args.out else Path("./reports/eda"),
        dpi=args.dpi,
        orientation=args.orientation,
        promo_thresh=args.promo_thresh,
        avail_thresh=args.avail_thresh,
        limit_episodes=args.limit_episodes,
        example_episode_id=args.example_episode_id,
        log_level=args.log_level
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)