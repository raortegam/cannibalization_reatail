#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA_pairs.py
------------------
Módulo de EDA (Análisis Exploratorio) para visualizar y auditar la calidad
de episodios de canibalización promocional en series temporales de retail.

Este módulo genera una ficha académica por episodio (una página por episodio)
en formato carta (8.5 x 11 pulgadas), que documenta:
  • Definición del episodio (víctima vs. caníbal, ventanas temporales).
  • Línea de tiempo (pre, tratamiento, post) con duraciones.
  • Métricas de calidad (coherencia temporal, tamaño de ventanas, cobertura de donantes).
  • Tabla de donantes top-k (misma referencia, distancia, tipo de tienda, ubicación).
  • Etiqueta de “calidad global” por umbrales configurables.

Uso (CLI)
---------
python EDA/EDA_pairs.py \
  --pairs-path data/processed_data/pairs_windows.csv \
  --donors-path data/processed_data/donors_per_victim.csv \
  --out-dir EDA/outputs/episodios_muestra \
  --n 5 --strategy stratified --seed 2025

Notas
-----
- El archivo de pares/episodios se espera con las columnas:
  [i_store,i_item,class,j_store,j_item,delta_H_j,n_obs_i_on,n_obs_i_off,
   pre_start,treat_start,post_start,post_end]
- El archivo de donantes se espera con las columnas:
  [j_store,j_item,donor_store,donor_item,donor_kind,distance,store_type,
   city,state,rank]
- Las figuras se producen estrictamente en tamaño “carta”.
- Dependencias: pandas, numpy, matplotlib (sin seaborn).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------------------------
# Constantes de formato
# -----------------------------------------------------------------------------

LETTER_PORTRAIT = (8.5, 11)  # pulgadas (ancho, alto)
LETTER_LANDSCAPE = (11, 8.5)
DPI_DEFAULT = 300

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------

def _parse_dates(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    return df


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _format_days(n: Optional[int]) -> str:
    return f"{n} días" if n is not None else "NA"


def _wrap(text: str, width: int = 100) -> str:
    import textwrap as _tw
    return _tw.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def _ensure_outdir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -----------------------------------------------------------------------------
# Selección de muestra de episodios
# -----------------------------------------------------------------------------

def select_sample(
    df_pairs: pd.DataFrame,
    n: int = 5,
    strategy: str = "stratified",
    seed: int = 2025
) -> pd.DataFrame:
    """
    Selecciona una muestra de episodios.

    Estrategias admitidas:
      - "top_delta":   mayores valores de |delta_H_j|
      - "recent":      episodios con treat_start más recientes
      - "random":      muestreo aleatorio reproducible
      - "stratified":  balance por clase y quintiles de |delta_H_j| (por defecto)

    Devuelve un DataFrame con exactamente n filas (si hay suficiente soporte).
    """
    rng = np.random.default_rng(seed)
    df = df_pairs.copy()
    if df.empty:
        raise ValueError("El DataFrame de episodios está vacío.")

    # Normalizamos y preparamos auxiliares
    df["abs_delta"] = df["delta_H_j"].abs()
    if "class" in df.columns:
        df["class_str"] = df["class"].astype(str)
    else:
        df["class_str"] = "NA"

    if strategy == "top_delta":
        out = df.sort_values("abs_delta", ascending=False).head(n)

    elif strategy == "recent":
        out = df.sort_values("treat_start", ascending=False).head(n)

    elif strategy == "random":
        out = df.sample(n=min(n, len(df)), random_state=seed)

    elif strategy == "stratified":
        # Balance por clase y quintil de |delta|
        df["q_abs_delta"] = pd.qcut(
            df["abs_delta"].rank(method="first"),
            q=min(5, len(df)),
            labels=False
        )
        buckets = df.groupby(["class_str", "q_abs_delta"], dropna=False)
        picks: List[pd.DataFrame] = []
        # Cuántos tomar por bucket
        base = n // max(1, buckets.ngroups)
        rem = n - base * max(1, buckets.ngroups)
        for _, g in buckets:
            k = min(base, len(g))
            if k > 0:
                picks.append(g.sample(n=k, random_state=seed))
        # Si faltan por redondeo, completar desde los buckets más grandes
        if rem > 0:
            leftover = (
                df.drop(pd.concat(picks).index, errors="ignore")
                if len(picks) else df
            )
            if len(leftover) > 0:
                picks.append(leftover.sample(n=min(rem, len(leftover)), random_state=seed))
        out = pd.concat(picks).head(n)

    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")

    # Garantizar cardinalidad exacta y orden estable
    out = out.head(n).copy()
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Métricas de calidad por episodio
# -----------------------------------------------------------------------------

@dataclass
class QualityConfig:
    min_pre_days: int = 60
    min_post_days: int = 30
    min_donors: int = 5
    min_same_item_share: float = 0.6
    # Distancia “baja” definida por cuantiles globales por j_item (si hay soporte):
    distance_quantile_threshold: float = 0.50  # mediana


def episode_quality_metrics(
    row: pd.Series,
    donors_all: pd.DataFrame,
    donors_ep: pd.DataFrame,
    cfg: QualityConfig
) -> Dict[str, object]:
    """Calcula métricas y banderas de calidad para un episodio."""
    pre_days = None
    treat_days = None
    post_days = None
    try:
        pre_days = int((row["treat_start"] - row["pre_start"]).days)
        treat_days = int((row["post_start"] - row["treat_start"]).days)
        # asumimos post_end inclusivo para informar duración efectiva:
        post_days = int((row["post_end"] - row["post_start"]).days) + 1
    except Exception:
        pass

    # Coherencia de orden temporal
    coherent = bool(
        (pd.notna(row["pre_start"]) and pd.notna(row["treat_start"]) and
         pd.notna(row["post_start"]) and pd.notna(row["post_end"]) and
         (row["pre_start"] <= row["treat_start"] < row["post_start"] <= row["post_end"]))
    )

    # Donantes
    donors_count = int(donors_ep.shape[0]) if donors_ep is not None else 0
    same_item_share = float(
        (donors_ep["donor_kind"].eq("same_item")).mean()
    ) if donors_count > 0 else np.nan

    # Distancia mediana del episodio vs. cuantil global para ese j_item
    med_distance = float(donors_ep["distance"].median()) if donors_count > 0 else np.nan
    q_thr = np.nan
    try:
        mask_item = donors_all["j_item"].eq(row["j_item"])
        pool = donors_all.loc[mask_item, "distance"]
        if len(pool) == 0:
            pool = donors_all["distance"]
        q_thr = float(pool.quantile(cfg.distance_quantile_threshold))
    except Exception:
        pass
    low_distance = (med_distance <= q_thr) if (np.isfinite(med_distance) and np.isfinite(q_thr)) else False

    flags = {
        "coherente": coherent,
        "pre_suficiente": (pre_days is not None and pre_days >= cfg.min_pre_days),
        "post_suficiente": (post_days is not None and post_days >= cfg.min_post_days),
        "donantes_suficientes": (donors_count >= cfg.min_donors),
        "mucha_misma_referencia": (np.isfinite(same_item_share) and same_item_share >= cfg.min_same_item_share),
        "distancia_baja": bool(low_distance),
    }
    score = int(sum(bool(v) for v in flags.values()))
    if score >= 5:
        label = "Alta"
    elif score >= 3:
        label = "Media"
    else:
        label = "Baja"

    return {
        "pre_days": pre_days,
        "treat_days": treat_days,
        "post_days": post_days,
        "donors_count": donors_count,
        "same_item_share": same_item_share,
        "med_distance": med_distance,
        "distance_threshold": q_thr,
        "flags": flags,
        "score": score,
        "label": label,
    }


# -----------------------------------------------------------------------------
# Render: ficha por episodio (una página carta)
# -----------------------------------------------------------------------------

def _draw_timeline(ax, row: pd.Series) -> None:
    """Dibuja línea de tiempo (pre, tratamiento, post) con marcas de fechas."""
    # Construir eje temporal compacto
    dates = [row["pre_start"], row["treat_start"], row["post_start"], row["post_end"]]
    dates_ok = [d for d in dates if pd.notna(d)]
    if not dates_ok:
        ax.text(0.5, 0.5, "Fechas no disponibles", ha="center", va="center", fontsize=11)
        ax.axis("off")
        return

    start = min(dates_ok)
    end = max(dates_ok)
    margin = pd.Timedelta(days= max(3, int((end - start).days * 0.05)))
    xmin = start - margin
    xmax = end + margin

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, 1])
    ax.get_yaxis().set_visible(False)

    # Segmentos por ventanas
    # Pre
    if pd.notna(row["pre_start"]) and pd.notna(row["treat_start"]):
        ax.plot([row["pre_start"], row["treat_start"]], [0.5, 0.5], lw=6, alpha=0.8)
    # Tratamiento
    if pd.notna(row["treat_start"]) and pd.notna(row["post_start"]):
        ax.plot([row["treat_start"], row["post_start"]], [0.5, 0.5], lw=6, alpha=0.8)
    # Post
    if pd.notna(row["post_start"]) and pd.notna(row["post_end"]):
        ax.plot([row["post_start"], row["post_end"]], [0.5, 0.5], lw=6, alpha=0.8)

    # Marcadores y etiquetas
    def _mark(x, label):
        if pd.notna(x):
            ax.plot([x, x], [0.35, 0.65], lw=2)
            ax.text(x, 0.7, label + "\n" + (x.strftime("%Y-%m-%d") if pd.notna(x) else "NA"),
                    ha="center", va="bottom", fontsize=9, rotation=0)

    _mark(row["pre_start"], "PRE\n(inicio)")
    _mark(row["treat_start"], "TRATAMIENTO\n(inicio)")
    _mark(row["post_start"], "POST\n(inicio)")
    _mark(row["post_end"], "POST\n(fin)")

    ax.set_title("Línea de tiempo del episodio", loc="left", fontsize=12, pad=6)


def _draw_quality(ax, q: Dict[str, object]) -> None:
    """Bloque de métricas y banderas de calidad."""
    ax.axis("off")
    flags = q["flags"]
    items = [
        ("Coherencia temporal", flags.get("coherente", False)),
        (f"Ventana pre ≥ {q.get('pre_days', 'NA')} días (mín. requerido en config)", flags.get("pre_suficiente", False)),
        (f"Ventana post ≥ {q.get('post_days', 'NA')} días (mín. requerido en config)", flags.get("post_suficiente", False)),
        (f"Donantes suficientes (k = {q.get('donors_count', 0)})", flags.get("donantes_suficientes", False)),
        (f"Alta proporción de misma referencia (p = {q.get('same_item_share', np.nan):.2f})", flags.get("mucha_misma_referencia", False)),
        (f"Distancia mediana ≤ cuantil {int(100*0.50)}% global ({q.get('med_distance', np.nan):.4f} ≤ {q.get('distance_threshold', np.nan):.4f})", flags.get("distancia_baja", False)),
    ]

    y = 1.0
    ax.text(0.0, y, "Auditoría de calidad", fontsize=12, fontweight="bold", va="top")
    y -= 0.08
    for label, ok in items:
        mark = "✓" if ok else "✗"
        ax.text(0.02, y, f"{mark} {label}", fontsize=10, va="top")
        y -= 0.065

    # Etiqueta global
    ax.text(0.0, y-0.02, f"Calidad global: {q['label']} (score={q['score']}/6)",
            fontsize=11, fontweight="bold", va="top")


def _draw_donors_table(ax, donors_ep: pd.DataFrame, top_k: int = 7) -> None:
    """Tabla compacta de donantes top-k por 'rank'."""
    ax.axis("off")
    ax.set_title("Donantes (top-k por rank)", loc="left", fontsize=12, pad=6)

    if donors_ep is None or donors_ep.empty:
        ax.text(0.5, 0.5, "Sin donantes disponibles para este (j_store, j_item).",
                ha="center", va="center", fontsize=10)
        return

    cols = ["rank", "donor_store", "donor_item", "donor_kind", "distance", "store_type", "city", "state"]
    use = donors_ep.sort_values("rank").head(top_k).copy()
    use["distance"] = use["distance"].round(5)
    data = use[cols].astype(str).values.tolist()

    tbl = ax.table(
        cellText=data,
        colLabels=[c.upper() for c in cols],
        loc="center",
        cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)


def _draw_definition(ax, row: pd.Series) -> None:
    """Bloque de texto: definición y lectura del episodio."""
    ax.axis("off")
    txt = (
        "Definición académica del episodio.\n\n"
        "Se denomina ‘episodio’ al periodo en el que un producto caníbal (tratado) "
        f"—j_store={row['j_store']}, j_item={row['j_item']}— activa una promoción "
        "y potencialmente desvía demanda de una ‘víctima’ —"
        f"i_store={row['i_store']}, i_item={row['i_item']}— dentro de una misma tienda/clase. "
        "La línea de tiempo se divide en: (i) PRE (línea base), (ii) TRATAMIENTO "
        "(vigencia de la promoción del caníbal) y (iii) POST (ventana de evaluación). "
        "El parámetro δ_H_j (delta_H_j) resume la magnitud relativa de la intervención del caníbal "
        "y sirve como criterio de intensidad. "
        "La calidad metodológica exige ventanas suficientes, coherencia temporal y un conjunto robusto "
        "de donantes (tiendas análogas con la misma referencia) para construir contrafactuales plausibles."
    )
    ax.text(0.0, 1.0, _wrap(txt, 120), ha="left", va="top", fontsize=10)


def plot_episode_card(
    row: pd.Series,
    donors_all: pd.DataFrame,
    donors_ep: pd.DataFrame,
    q: Dict[str, object],
    figsize: Tuple[float, float] = LETTER_PORTRAIT,
    dpi: int = DPI_DEFAULT
) -> plt.Figure:
    """Construye una ‘ficha’ estilo académico (tamaño carta) para un episodio."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(100, 100, figure=fig)

    # Encabezado
    ax_header = fig.add_subplot(gs[0:8, :])
    ax_header.axis("off")
    title = (f"Episodio: Víctima (i_store={row['i_store']}, i_item={row['i_item']})  "
             f"⇢  Caníbal (j_store={row['j_store']}, j_item={row['j_item']})  "
             f"| clase={row.get('class', 'NA')} | δ_H_j={row.get('delta_H_j', np.nan):.4f}")
    ax_header.text(0.0, 0.75, title, fontsize=13, fontweight="bold", va="center")
    ax_header.text(0.0, 0.20,
                   (f"Obs. i_on={row.get('n_obs_i_on', 'NA')} | "
                    f"Obs. i_off={row.get('n_obs_i_off', 'NA')} | "
                    f"pre_start={row.get('pre_start')} | "
                    f"treat_start={row.get('treat_start')} | "
                    f"post_start={row.get('post_start')} | "
                    f"post_end={row.get('post_end')}"),
                   fontsize=9, va="center")

    # Panel A: timeline
    ax_time = fig.add_subplot(gs[10:30, 5:95])
    _draw_timeline(ax_time, row)

    # Panel B: calidad
    ax_q = fig.add_subplot(gs[35:70, 5:55])
    _draw_quality(ax_q, q)

    # Panel C: donantes
    ax_d = fig.add_subplot(gs[35:90, 57:95])
    _draw_donors_table(ax_d, donors_ep, top_k=7)

    # Panel D: definición
    ax_def = fig.add_subplot(gs[75:98, 5:95])
    _draw_definition(ax_def, row)

    # Pie de página
    ax_footer = fig.add_subplot(gs[98:, :])
    ax_footer.axis("off")
    ax_footer.text(0.0, 0.5,
                   "Ficha de episodio | EDA causal (control sintético generalizado)",
                   fontsize=8, va="center")
    ax_footer.text(1.0, 0.5,
                   "Tamaño: Carta (8.5×11 pulgadas) | Generado por EDA_pairs.py",
                   fontsize=8, va="center", ha="right")
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------

def run(
    pairs_path: str,
    donors_path: str,
    out_dir: str,
    n: int = 5,
    strategy: str = "stratified",
    seed: int = 2025,
    orientation: str = "portrait",
    dpi: int = DPI_DEFAULT,
    quality_cfg: Optional[QualityConfig] = None,
) -> Dict[str, object]:
    """Ejecuta el pipeline completo de EDA y exporta la visualización.

    Parámetros
    ----------
    pairs_path : str
        Ruta al CSV de episodios (pairs_windows.csv).
    donors_path : str
        Ruta al CSV de donantes (donors_per_victim.csv).
    out_dir : str
        Directorio de salida para las imágenes PNG.
    n : int
        Número de episodios a visualizar (por defecto 5).
    strategy : {"stratified","top_delta","recent","random"}
        Estrategia de muestreo de episodios.
    seed : int
        Semilla de aleatoriedad para reproducibilidad.
    orientation : {"portrait","landscape"}
        Orientación del lienzo tamaño carta.
    dpi : int
        Resolución de salida.
    quality_cfg : Optional[QualityConfig]
        Configuración de umbrales de calidad.

    Devuelve
    --------
    dict con metadatos de ejecución (muestra, rutas, etc.).
    """
    if quality_cfg is None:
        quality_cfg = QualityConfig()

    # 1) Carga
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"No se encontró pairs_path: {pairs_path}")
    if not os.path.exists(donors_path):
        raise FileNotFoundError(f"No se encontró donors_path: {donors_path}")

    df_pairs = pd.read_csv(pairs_path)
    df_pairs = _parse_dates(df_pairs, ["pre_start", "treat_start", "post_start", "post_end"])

    df_donors = pd.read_csv(donors_path)

    # 2) Muestra
    sample = select_sample(df_pairs, n=n, strategy=strategy, seed=seed)

    # 3) Preparar salida
    # Asegurar directorio de salida
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 4) Tamaño de figura
    figsize = LETTER_PORTRAIT if orientation.lower().startswith("p") else LETTER_LANDSCAPE

    # 5) Render y export
    pages: List[plt.Figure] = []
    metas: List[Dict[str, object]] = []
    for idx, row in sample.iterrows():
        # Donantes para este episodio (por definición, atados al caníbal j_*)
        donors_ep = df_donors[(df_donors["j_store"] == row["j_store"]) & (df_donors["j_item"] == row["j_item"])].copy()

        q = episode_quality_metrics(row, donors_all=df_donors, donors_ep=donors_ep, cfg=quality_cfg)
        fig = plot_episode_card(row, donors_all=df_donors, donors_ep=donors_ep, q=q, figsize=figsize, dpi=dpi)
        pages.append(fig)

        metas.append({
            "index": int(idx),
            "i_store": _safe_int(row.get("i_store")),
            "i_item": _safe_int(row.get("i_item")),
            "j_store": _safe_int(row.get("j_store")),
            "j_item": _safe_int(row.get("j_item")),
            "class": _safe_int(row.get("class")),
            "delta_H_j": float(row.get("delta_H_j", np.nan)),
            "quality_label": q["label"],
            "quality_score": int(q["score"]),
            "pre_days": q["pre_days"],
            "treat_days": q["treat_days"],
            "post_days": q["post_days"],
            "donors_count": q["donors_count"],
            "same_item_share": q["same_item_share"],
            "med_distance": q["med_distance"],
        })

    # Guardar únicamente PNGs numerados en el directorio de salida
    for k, fig in enumerate(pages, start=1):
        p = os.path.join(out_dir, f"eda_3_episodio_{k}.png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    saved_to = os.path.abspath(out_dir)

    # Guardar metadatos resumen
    summary_path = os.path.join(out_dir, "episodios_muestra_resumen.csv")
    pd.DataFrame(metas).to_csv(summary_path, index=False)

    return {
        "n_pages": len(pages),
        "out_dir": saved_to,
        "summary_csv": os.path.abspath(summary_path),
        "strategy": strategy,
        "seed": seed,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EDA académica de episodios de canibalización (tamaño carta).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pairs-path", type=str, default="data/processed_data/pairs_windows.csv",
                        help="Ruta al CSV de episodios (pairs_windows.csv).")
    parser.add_argument("--donors-path", type=str, default="data/processed_data/donors_per_victim.csv",
                        help="Ruta al CSV de donantes (donors_per_victim.csv).")
    parser.add_argument("--out-dir", type=str, default="EDA/outputs/episodios_muestra",
                        help="Directorio de salida para las imágenes PNG.")
    parser.add_argument("--n", type=int, default=5, help="Número de episodios a visualizar.")
    parser.add_argument("--strategy", type=str, choices=["stratified", "top_delta", "recent", "random"],
                        default="stratified", help="Estrategia de muestreo de episodios.")
    parser.add_argument("--seed", type=int, default=2025, help="Semilla aleatoria.")
    parser.add_argument("--orientation", type=str, choices=["portrait", "landscape"], default="portrait",
                        help="Orientación del lienzo tamaño carta.")
    parser.add_argument("--dpi", type=int, default=DPI_DEFAULT, help="Resolución de salida (dpi).")
    # Configuración de calidad
    parser.add_argument("--min-pre-days", type=int, default=QualityConfig.min_pre_days,
                        help="Mínimo de días en ventana PRE.")
    parser.add_argument("--min-post-days", type=int, default=QualityConfig.min_post_days,
                        help="Mínimo de días en ventana POST.")
    parser.add_argument("--min-donors", type=int, default=QualityConfig.min_donors,
                        help="Mínimo de donantes por episodio.")
    parser.add_argument("--min-same-item-share", type=float, default=QualityConfig.min_same_item_share,
                        help="Proporción mínima de donantes con la misma referencia.")
    parser.add_argument("--distance-quantile-threshold", type=float, default=QualityConfig.distance_quantile_threshold,
                        help="Cuantil global de distancia (por j_item) para considerar 'baja distancia'.")

    args = parser.parse_args()

    qc = QualityConfig(
        min_pre_days=args.min_pre_days,
        min_post_days=args.min_post_days,
        min_donors=args.min_donors,
        min_same_item_share=args.min_same_item_share,
        distance_quantile_threshold=args.distance_quantile_threshold
    )

    meta = run(
        pairs_path=args.pairs_path,
        donors_path=args.donors_path,
        out_dir=args.out_dir,
        n=args.n,
        strategy=args.strategy,
        seed=args.seed,
        orientation=args.orientation,
        dpi=args.dpi,
        quality_cfg=qc,
    )

    # Mensaje de éxito minimalista
    print("✅ EDA completada.")
    for k, v in meta.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
