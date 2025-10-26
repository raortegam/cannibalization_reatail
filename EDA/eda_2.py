# eda_2.py
# -*- coding: utf-8 -*-
"""
EDA específico de la variable de exposición competitiva.

Propósito académico:
    Caracterizar la distribución, dinámica temporal y heterogeneidad entre tiendas
    de la métrica 'competitive_exposure' (proporción en [0, 1]) posterior a su
    construcción, preservando criterios de trazabilidad y estabilidad estadística.

Entradas:
    --input: ruta a un archivo con la exposición competitiva (CSV o Parquet)
             Se espera al menos:
                - date (fecha)
                - store_nbr (tienda)
                - item_nbr (ítem)
                - competitive_exposure (exposición competitiva, en [0, 1])
    --output-dir: carpeta donde se guardarán las figuras (formato PNG)

Parámetros de columnas (configurables):
    --date-col, --store-col, --item-col, --exposure-col

Salidas (imágenes tamaño carta):
    - eda2_01_exposure_over_time.png            (media diaria)
    - eda2_02_exposure_histogram.png            (histograma global)
    - eda2_03_exposure_cdf.png                  (CDF global)
    - eda2_04_store_mean_exposure_topN.png      (media por tienda, top N por volumen)
    - eda2_05_heatmap_store_date_topN.png       (heatmap de media diaria para top N tiendas)

Notas:
    - Si el archivo es CSV, se procesa por chunks (memoria eficiente).
    - Si el archivo es Parquet, se carga completo (para parquet no hay chunking nativo en pandas).
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------- Utilidades de figura (hoja carta) ---------------------------- #

def _letter_figsize(orientation: str = "portrait") -> Tuple[float, float]:
    """
    Devuelve (width, height) en pulgadas para hoja carta.
    """
    orientation = (orientation or "portrait").lower()
    if orientation not in {"portrait", "landscape"}:
        orientation = "portrait"
    return (8.5, 11.0) if orientation == "portrait" else (11.0, 8.5)


def _save_letter_fig(fig: plt.Figure, path: Path, orientation: str = "portrait", dpi: int = 300):
    """
    Guarda figura con tamaño de hoja carta.
    """
    w, h = _letter_figsize(orientation)
    fig.set_size_inches(w, h)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------- Acumuladores / Agregadores ---------------------------- #

class GlobalExposureAgg:
    """
    Acumulador global de exposición: histograma, conteos y momentos.
    Permite construir también la CDF a partir del histograma.

    bins: número de bins en [0, 1].
    """
    def __init__(self, bins: int = 50):
        self.bins = int(bins)
        self.edges = np.linspace(0.0, 1.0, self.bins + 1)
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.n_total = 0
        self.n_valid = 0
        self.n_missing = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.n_out_of_range = 0

    def update(self, x: pd.Series):
        # x: exposición competitiva (numérica)
        self.n_total += x.shape[0]
        valid = pd.to_numeric(x, errors="coerce")
        n_miss = valid.isna().sum()
        self.n_missing += int(n_miss)
        valid = valid.dropna()

        # Conteo de out-of-range (antes de clip)
        oor = ((valid < 0) | (valid > 1)).sum()
        self.n_out_of_range += int(oor)

        # Clip conservador a [0,1] (para robustez visual del EDA)
        valid = valid.clip(0.0, 1.0)

        vals = valid.values.astype(float)
        self.n_valid += vals.size
        self.sum_x += float(vals.sum())
        self.sum_x2 += float((vals ** 2).sum())

        # Histograma incremental
        c, _ = np.histogram(vals, bins=self.edges)
        self.counts += c

    def summary(self) -> Dict[str, float]:
        mean = self.sum_x / self.n_valid if self.n_valid > 0 else np.nan
        var = (self.sum_x2 / self.n_valid - mean**2) if self.n_valid > 0 else np.nan
        std = np.sqrt(var) if np.isfinite(var) else np.nan
        miss_frac = self.n_missing / self.n_total if self.n_total > 0 else np.nan
        return {
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_missing": self.n_missing,
            "missing_frac": miss_frac,
            "mean": mean,
            "std": std,
            "out_of_range_count": self.n_out_of_range,
        }

    def histogram_df(self) -> pd.DataFrame:
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        frac = (self.counts / self.n_valid) if self.n_valid > 0 else np.zeros_like(self.counts, dtype=float)
        return pd.DataFrame({"bin_center": centers, "count": self.counts, "fraction": frac})

    def cdf_df(self) -> pd.DataFrame:
        frac = (self.counts / self.n_valid) if self.n_valid > 0 else np.zeros_like(self.counts, dtype=float)
        cdf = np.cumsum(frac)
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        return pd.DataFrame({"bin_center": centers, "cdf": cdf})


class DailyExposureAgg:
    """
    Acumulador diario: suma, conteo y conteo de ceros para exposición.
    """
    def __init__(self):
        self.sum_by_date: Dict[pd.Timestamp, float] = defaultdict(float)
        self.cnt_by_date: Dict[pd.Timestamp, int] = defaultdict(int)
        self.zero_by_date: Dict[pd.Timestamp, int] = defaultdict(int)

    def update(self, dates: pd.Series, exposure: pd.Series):
        d = dates.dt.normalize()
        v = pd.to_numeric(exposure, errors="coerce").clip(0.0, 1.0)
        mask = v.notna()
        if not mask.any():
            return
        d = d[mask]
        v = v[mask]
        g_sum = v.groupby(d).sum()
        g_cnt = v.groupby(d).size()
        g_zero = (v == 0.0).groupby(d).sum()

        for k, s in g_sum.items():
            self.sum_by_date[pd.Timestamp(k)] += float(s)
        for k, c in g_cnt.items():
            self.cnt_by_date[pd.Timestamp(k)] += int(c)
        for k, z in g_zero.items():
            self.zero_by_date[pd.Timestamp(k)] += int(z)

    def to_dataframe(self) -> pd.DataFrame:
        idx = sorted(set(self.sum_by_date.keys()) | set(self.cnt_by_date.keys()))
        rows = []
        for dt in idx:
            s = self.sum_by_date.get(dt, 0.0)
            n = self.cnt_by_date.get(dt, 0)
            z = self.zero_by_date.get(dt, 0)
            mean = s / n if n > 0 else np.nan
            zero_share = z / n if n > 0 else np.nan
            rows.append((pd.Timestamp(dt), n, mean, zero_share))
        return pd.DataFrame(rows, columns=["date", "n_obs", "mean_exposure", "zero_share"]).sort_values("date")


class StoreExposureAgg:
    """
    Acumulador por tienda: suma y conteo para media de exposición.
    """
    def __init__(self):
        self.sum_by_store: Dict[int, float] = defaultdict(float)
        self.cnt_by_store: Dict[int, int] = defaultdict(int)

    def update(self, stores: pd.Series, exposure: pd.Series):
        s = stores
        v = pd.to_numeric(exposure, errors="coerce").clip(0.0, 1.0)
        mask = v.notna() & s.notna()
        if not mask.any():
            return
        s = s[mask].astype("Int64")
        v = v[mask]
        g_sum = v.groupby(s).sum()
        g_cnt = v.groupby(s).size()
        for k, x in g_sum.items():
            self.sum_by_store[int(k)] += float(x)
        for k, c in g_cnt.items():
            self.cnt_by_store[int(k)] += int(c)

    def to_dataframe(self) -> pd.DataFrame:
        stores = sorted(set(self.sum_by_store.keys()) | set(self.cnt_by_store.keys()))
        rows = []
        for st in stores:
            s = self.sum_by_store.get(st, 0.0)
            n = self.cnt_by_store.get(st, 0)
            mean = s / n if n > 0 else np.nan
            rows.append((int(st), int(n), float(mean)))
        return pd.DataFrame(rows, columns=["store_nbr", "n_obs", "mean_exposure"]).sort_values("n_obs", ascending=False)


class StoreDateExposureAgg:
    """
    Acumulador (tienda, fecha) para construir un heatmap de media diaria de exposición
    en un subconjunto de tiendas (top-N por volumen).
    """
    def __init__(self, target_stores: Iterable[int]):
        self.target = set(int(s) for s in target_stores)
        self.sum_by_pair: Dict[Tuple[pd.Timestamp, int], float] = defaultdict(float)
        self.cnt_by_pair: Dict[Tuple[pd.Timestamp, int], int] = defaultdict(int)

    def update(self, dates: pd.Series, stores: pd.Series, exposure: pd.Series):
        mask = stores.isin(self.target)
        if not mask.any():
            return
        d = dates[mask].dt.normalize()
        s = stores[mask].astype("Int64")
        v = pd.to_numeric(exposure[mask], errors="coerce").clip(0.0, 1.0)
        mask2 = v.notna()
        if not mask2.any():
            return
        d = d[mask2]
        s = s[mask2].astype("Int64")
        v = v[mask2]
        g = v.groupby([d, s]).agg(["sum", "size"])
        # g index: MultiIndex(date, store)
        for (dt, st), row in g.iterrows():
            key = (pd.Timestamp(dt), int(st))
            self.sum_by_pair[key] += float(row["sum"])
            self.cnt_by_pair[key] += int(row["size"])

    def to_pivot(self) -> pd.DataFrame:
        # Construye pivot date x store con medias
        keys = sorted(set(self.sum_by_pair.keys()))
        rows = []
        for (dt, st) in keys:
            s = self.sum_by_pair[(dt, st)]
            n = self.cnt_by_pair[(dt, st)]
            mean = s / n if n > 0 else np.nan
            rows.append((pd.Timestamp(dt), int(st), float(mean)))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["date", "store_nbr", "mean_exposure"])
        pv = df.pivot(index="date", columns="store_nbr", values="mean_exposure").sort_index()
        return pv


# ---------------------------- Gráficos ---------------------------- #

def plot_exposure_over_time(df: pd.DataFrame, out_path: Path, orientation: str):
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["mean_exposure"])
    ax.set_title("Exposición competitiva — media diaria")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Media de exposición")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    _save_letter_fig(fig, out_path, orientation)


def plot_histogram(hist_df: pd.DataFrame, out_path: Path, orientation: str):
    # Fracción por bin (porcentaje)
    x = hist_df["bin_center"].values
    y = (hist_df["fraction"].values * 100.0)
    fig, ax = plt.subplots()
    ax.bar(x, y, width=(x[1] - x[0]) if len(x) > 1 else 0.02, align="center")
    ax.set_title("Distribución de exposición competitiva (histograma)")
    ax.set_xlabel("Exposición (proporción)")
    ax.set_ylabel("Frecuencia (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    _save_letter_fig(fig, out_path, orientation)


def plot_cdf(cdf_df: pd.DataFrame, out_path: Path, orientation: str):
    fig, ax = plt.subplots()
    ax.plot(cdf_df["bin_center"], cdf_df["cdf"] * 100.0)
    ax.set_title("Función de distribución acumulada (CDF) de la exposición")
    ax.set_xlabel("Exposición (proporción)")
    ax.set_ylabel("Acumulado (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    _save_letter_fig(fig, out_path, orientation)


def plot_store_mean_bar(store_df: pd.DataFrame, top_stores: int, min_obs: int,
                        out_path: Path, orientation: str):
    df = store_df.copy()
    df = df[df["n_obs"] >= int(min_obs)]
    if df.empty:
        # Crear figura vacía pero informativa
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin suficientes observaciones por tienda", ha="center", va="center")
        ax.axis("off")
        _save_letter_fig(fig, out_path, orientation)
        return

    df = df.sort_values(["n_obs", "mean_exposure"], ascending=[False, False]).head(int(top_stores))
    fig, ax = plt.subplots()
    ax.bar(df["store_nbr"].astype(str).values, df["mean_exposure"].values * 100.0)
    ax.set_title(f"Exposición media por tienda (Top {top_stores} por volumen, min {min_obs} obs)")
    ax.set_xlabel("Tienda")
    ax.set_ylabel("Exposición media (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    _save_letter_fig(fig, out_path, orientation)


def plot_heatmap(pivot_df: pd.DataFrame, out_path: Path, orientation: str):
    if pivot_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Heatmap no disponible (sin datos para top tiendas)", ha="center", va="center")
        ax.axis("off")
        _save_letter_fig(fig, out_path, orientation)
        return

    # Ordenar columnas por exposición media descendente para mejor lectura
    col_order = pivot_df.mean(axis=0).sort_values(ascending=False).index.tolist()
    mat = pivot_df[col_order].values  # filas: fechas; columnas: tiendas

    fig, ax = plt.subplots()
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_title("Heatmap de exposición media diaria — Top tiendas")
    ax.set_xlabel("Tiendas (ordenadas por exposición media)")
    ax.set_ylabel("Índice temporal")
    # ticks parciales para no saturar (opcionales)
    if pivot_df.shape[0] > 1:
        xticks = np.linspace(0, pivot_df.shape[1]-1, min(20, pivot_df.shape[1])).astype(int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(c) for c in np.array(col_order)[xticks]], rotation=90)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Exposición media (proporción)")
    _save_letter_fig(fig, out_path, orientation)


# ---------------------------- Pipeline principal ---------------------------- #

def _read_header_columns_csv(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [c.strip() for c in f.readline().strip().split(",")]


def run_eda_competitive_exposure(
    input_path: str,
    output_dir: str,
    # columnas
    date_col: str = "date",
    store_col: str = "store_nbr",
    item_col: str = "item_nbr",
    exposure_col: str = "competitive_exposure",
    # visual
    orientation: str = "portrait",
    dpi: int = 300,
    bins: int = 50,
    top_stores: int = 30,
    min_store_obs: int = 5,
    heatmap_stores: Optional[int] = None,  # si None, usa top_stores
    # rendimiento
    chunksize: int = 1_000_000,
    log_level: str = "INFO",
):
    """
    Ejecuta el EDA de exposición competitiva y guarda figuras tamaño carta.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    # Inicializar agregadores del primer barrido
    global_agg = GlobalExposureAgg(bins=bins)
    daily_agg = DailyExposureAgg()
    store_agg = StoreExposureAgg()

    # Detectar extensión
    ext = input_path.suffix.lower()
    usecols = [date_col, store_col, item_col, exposure_col]

    if ext in {".csv"}:
        # Primer barrido: métricas globales, por fecha y por tienda
        logging.info("Leyendo CSV por chunks (primer barrido)...")
        hdr = set(_read_header_columns_csv(input_path))
        missing_cols = [c for c in usecols if c not in hdr]
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas en {input_path.name}: {missing_cols}")

        for chunk in pd.read_csv(
            input_path,
            usecols=usecols,
            parse_dates=[date_col],
            dtype={store_col: "Int64", item_col: "Int64"},
            chunksize=chunksize,
            low_memory=True,
        ):
            # Actualizar acumuladores
            global_agg.update(chunk[exposure_col])
            daily_agg.update(chunk[date_col], chunk[exposure_col])
            store_agg.update(chunk[store_col], chunk[exposure_col])

        # Selección de top tiendas por volumen
        store_df = store_agg.to_dataframe()
        n_top = int(heatmap_stores) if heatmap_stores is not None else int(top_stores)
        top_ids = store_df.sort_values("n_obs", ascending=False).head(n_top)["store_nbr"].tolist()

        # Segundo barrido (heatmap): restringido a top tiendas
        heat_agg = StoreDateExposureAgg(target_stores=top_ids)
        logging.info("Leyendo CSV por chunks (segundo barrido para heatmap de top tiendas)...")
        for chunk in pd.read_csv(
            input_path,
            usecols=[date_col, store_col, exposure_col],
            parse_dates=[date_col],
            dtype={store_col: "Int64"},
            chunksize=chunksize,
            low_memory=True,
        ):
            heat_agg.update(chunk[date_col], chunk[store_col], chunk[exposure_col])

        pivot_heat = heat_agg.to_pivot()

    elif ext in {".parquet", ".pq"}:
        logging.warning("Leyendo Parquet completo (sin chunking).")
        df = pd.read_parquet(input_path, columns=usecols)
        # Primer barrido (sobre df completo)
        global_agg.update(df[exposure_col])
        daily_agg.update(pd.to_datetime(df[date_col]), df[exposure_col])
        store_agg.update(df[store_col], df[exposure_col])

        # Selección top tiendas y heatmap
        store_df = store_agg.to_dataframe()
        n_top = int(heatmap_stores) if heatmap_stores is not None else int(top_stores)
        top_ids = store_df.sort_values("n_obs", ascending=False).head(n_top)["store_nbr"].tolist()

        heat_agg = StoreDateExposureAgg(target_stores=top_ids)
        heat_agg.update(pd.to_datetime(df[date_col]), df[store_col], df[exposure_col])
        pivot_heat = heat_agg.to_pivot()
    else:
        raise ValueError(f"Formato no soportado: {ext}. Use CSV o Parquet.")

    # Construir DataFrames finales de EDA
    daily_df = daily_agg.to_dataframe()
    hist_df = global_agg.histogram_df()
    cdf_df = global_agg.cdf_df()
    store_df = store_agg.to_dataframe()
    summary = global_agg.summary()
    logging.info(f"Resumen global: {summary}")

    # ----------------- Guardar figuras (tamaño carta) ----------------- #

    # 01: media diaria
    plot_exposure_over_time(
        daily_df,
        out_path=out_dir / "eda_2_01_exposure_over_time.png",
        orientation=orientation,
    )

    # 02: histograma global
    plot_histogram(
        hist_df,
        out_path=out_dir / "eda_2_02_exposure_histogram.png",
        orientation=orientation,
    )

    # 03: CDF global
    plot_cdf(
        cdf_df,
        out_path=out_dir / "eda_2_03_exposure_cdf.png",
        orientation=orientation,
    )

    # 04: barras por tienda (top N por volumen)
    plot_store_mean_bar(
        store_df, top_stores=top_stores, min_obs=min_store_obs,
        out_path=out_dir / f"eda_2_04_store_mean_exposure_top{top_stores}.png",
        orientation=orientation,
    )

    # 05: heatmap por fecha x top tiendas
    plot_heatmap(
        pivot_heat,
        out_path=out_dir / f"eda_2_05_heatmap_store_date_top{len(pivot_heat.columns) if hasattr(pivot_heat, 'columns') else 0}.png",
        orientation=orientation,
    )

    logging.info("EDA de exposición competitiva finalizado.")


# ---------------------------- CLI ---------------------------- #

def _parse_args():
    ap = argparse.ArgumentParser(
        description="EDA de la variable 'competitive_exposure' (salidas tamaño carta)."
    )
    ap.add_argument("--input", required=True, help="Ruta al archivo con exposición (CSV o Parquet)")
    ap.add_argument("--output-dir", required=True, help="Carpeta de salida para las figuras PNG")

    # Columnas
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--store-col", default="store_nbr")
    ap.add_argument("--item-col", default="item_nbr")
    ap.add_argument("--exposure-col", default="competitive_exposure")

    # Visual
    ap.add_argument("--orientation", choices=["portrait", "landscape"], default="portrait", help="Tamaño carta")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--bins", type=int, default=50, help="Número de bins para el histograma/CDF")

    # Selección de tiendas / estabilidad
    ap.add_argument("--top-stores", type=int, default=30, help="Top N tiendas por volumen para barras")
    ap.add_argument("--min-store-obs", type=int, default=5, help="Mínimo de observaciones por tienda para ser graficada")
    ap.add_argument("--heatmap-stores", type=int, default=None, help="N de tiendas para el heatmap (si None, usa top-stores)")

    # Rendimiento
    ap.add_argument("--chunksize", type=int, default=1_000_000, help="Tamaño de chunk para CSV")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_eda_competitive_exposure(
        input_path=args.input,
        output_dir=args.output_dir,
        date_col=args.date_col,
        store_col=args.store_col,
        item_col=args.item_col,
        exposure_col=args.exposure_col,
        orientation=args.orientation,
        dpi=args.dpi,
        bins=args.bins,
        top_stores=args.top_stores,
        min_store_obs=args.min_store_obs,
        heatmap_stores=args.heatmap_stores,
        chunksize=args.chunksize,
        log_level=args.log_level,
    )