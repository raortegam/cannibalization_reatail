# eda.py
# -*- coding: utf-8 -*-
"""
EDA posterior al paso de 'data_quality.py'.

Este módulo:
1) Lee datasets 'train_filtered.csv' (requerido) y 'transactions_filtered.csv' (opcional),
   además de 'items.csv' (opcional) para enriquecer con vecindario/categoría.
2) Selecciona sólo variables de interés para el estudio causal.
3) Genera un conjunto de figuras (PNG) en tamaño carta (8.5 x 11 pulgadas).
4) Incluye un apartado profundo para analizar 'onpromotion'.

Uso por CLI (ejemplos):

# EDA básico usando sólo train_filtered.csv
python eda.py \
  --train data/clean/train_filtered.csv \
  --out-dir reports/eda

# EDA con transacciones (agregadas por tienda y día)
python eda.py \
  --train data/clean/train_filtered.csv \
  --transactions data/clean/transactions_filtered.csv \
  --out-dir reports/eda

# EDA con items.csv para análisis por vecindario (family/clase)
python eda.py \
  --train data/clean/train_filtered.csv \
  --transactions data/clean/transactions_filtered.csv \
  --items data/items.csv \
  --neighborhood-col family \
  --out-dir reports/eda

Salidas esperadas (todas en tamaño carta):
- 01_data_coverage.png                  (cobertura temporal de filas)
- 02_transactions_trend.png             (si hay transacciones)
- 03_unit_sales_trend.png               (si hay unit_sales)
- 04_missingness_bar.png                (faltantes por columna de interés)
- 10_promo_rate_over_time.png           (tasa de onpromotion en el tiempo)
- 11_promo_rate_by_dow.png              (tasa onpromotion por día de semana)
- 12_promo_rate_store_hist.png          (distribución de tasa onpromotion a nivel tienda)
- 13_promo_run_lengths_hist.png         (longitudes de rachas en promoción por ítem)
- 14_promo_transition_matrix.png        (matriz de transición de onpromotion: 0/1 -> 0/1)
- 15_promo_crowding_hist.png            (# ítems simultáneamente en promo por tienda/día)
- 16_promo_rate_by_neighborhood.png     (si hay neighborhood: tasa por vecindario top-N)

Todas las figuras se guardan en `--out-dir`. Formato por defecto: PNG a 150 DPI.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Para entornos headless
import matplotlib.pyplot as plt


# --------------------------- Config & helpers --------------------------- #

LETTER_SIZE = (8.5, 11)  # pulgadas
DEFAULT_DPI = 150

def _ensure_out_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save_letter_fig(fig: plt.Figure, out_path: Path, dpi: int = DEFAULT_DPI):
    fig.set_size_inches(*LETTER_SIZE)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _title_with_range(ax: plt.Axes, title: str, date_min: pd.Timestamp, date_max: pd.Timestamp):
    suffix = f"  |  Rango temporal: {date_min.date()} – {date_max.date()}"
    ax.set_title(title + suffix)

def _select_vars_of_interest(df: pd.DataFrame,
                             base_cols: List[str],
                             optional_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in base_cols if c in df.columns] + [c for c in optional_cols if c in df.columns]
    return df.loc[:, cols].copy()

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _day_of_week_labels(idx: List[int]) -> List[str]:
    # 0=Lunes ... 6=Domingo para presentación en ES
    mapping = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
    return [mapping.get(i, str(i)) for i in idx]


# --------------------------- IO & merge --------------------------- #

def load_data_for_eda(
    train_path: str,
    transactions_path: Optional[str] = None,
    items_path: Optional[str] = None,
    date_col: str = "date",
    store_col: str = "store_nbr",
    item_col: str = "item_nbr",
    promo_col: str = "onpromotion",
    sales_col: Optional[str] = "unit_sales",
    tx_col: str = "transactions",
    neighborhood_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Carga datasets ya filtrados/limpios desde data_quality.py y (opcional) items.csv.
    """
    # train_filtered: columnas de interés mínimas
    usecols = [date_col, store_col, item_col, promo_col]
    if sales_col:
        usecols.append(sales_col)

    train = pd.read_csv(train_path, usecols=[c for c in usecols if c is not None and c != ""],
                        parse_dates=[date_col])

    # transactions_filtered opcional
    transactions = None
    if transactions_path:
        transactions = pd.read_csv(
            transactions_path,
            usecols=[date_col, store_col, tx_col],
            parse_dates=[date_col]
        )

    # items opcional para neighborhood
    items = None
    if items_path and neighborhood_col:
        items = pd.read_csv(items_path, usecols=[item_col, neighborhood_col])

    return train, transactions, items


# --------------------------- Plots genéricos --------------------------- #

def plot_daily_count(train: pd.DataFrame, out_dir: Path,
                     date_col: str, dpi: int):
    # Conteo de filas por día
    daily = train.groupby(date_col, observed=True).size().rename("rows").reset_index()

    fig, ax = plt.subplots()
    ax.plot(daily[date_col], daily["rows"])
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Filas por día")
    _title_with_range(ax, "Cobertura temporal (filas por día)", daily[date_col].min(), daily[date_col].max())
    _save_letter_fig(fig, out_dir / "eda_1_01_data_coverage.png", dpi=dpi)


def plot_transactions_trend(transactions: pd.DataFrame, out_dir: Path,
                            date_col: str, tx_col: str, dpi: int):
    daily_tx = transactions.groupby(date_col, observed=True)[tx_col].sum().reset_index()
    daily_tx["tx_roll7"] = daily_tx[tx_col].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots()
    ax.plot(daily_tx[date_col], daily_tx[tx_col], label="Total diario")
    ax.plot(daily_tx[date_col], daily_tx["tx_roll7"], label="Media móvil 7d")
    ax.legend()
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Transacciones (total)")
    _title_with_range(ax, "Tendencia de transacciones", daily_tx[date_col].min(), daily_tx[date_col].max())
    _save_letter_fig(fig, out_dir / "eda_1_02_transactions_trend.png", dpi=dpi)


def plot_unit_sales_trend(train: pd.DataFrame, out_dir: Path,
                          date_col: str, sales_col: str, dpi: int):
    tmp = train.copy()
    tmp[sales_col] = _safe_numeric(tmp[sales_col])
    daily_sales = tmp.groupby(date_col, observed=True)[sales_col].sum().reset_index()
    daily_sales["roll7"] = daily_sales[sales_col].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots()
    ax.plot(daily_sales[date_col], daily_sales[sales_col], label="Total diario")
    ax.plot(daily_sales[date_col], daily_sales["roll7"], label="Media móvil 7d")
    ax.legend()
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Ventas (unit_sales)")
    _title_with_range(ax, "Tendencia de ventas", daily_sales[date_col].min(), daily_sales[date_col].max())
    _save_letter_fig(fig, out_dir / "eda_1_03_unit_sales_trend.png", dpi=dpi)


def plot_missingness_bar(df: pd.DataFrame, out_dir: Path, dpi: int):
    miss = df.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.bar(range(len(miss.index)), miss.values)
    ax.set_xticks(range(len(miss.index)))
    ax.set_xticklabels(miss.index, rotation=45, ha="right")
    ax.set_ylabel("Proporción de faltantes")
    ax.set_title("Faltantes por columna (0–1)")
    _save_letter_fig(fig, out_dir / "eda_1_04_missingness_bar.png", dpi=dpi)


# --------------------------- Análisis profundo: onpromotion --------------------------- #

def plot_promo_rate_over_time(train: pd.DataFrame, out_dir: Path,
                              date_col: str, promo_col: str, dpi: int):
    daily_rate = (train.groupby(date_col, observed=True)[promo_col]
                       .mean().reset_index(name="promo_rate"))
    daily_rate["roll7"] = daily_rate["promo_rate"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots()
    ax.plot(daily_rate[date_col], daily_rate["promo_rate"], label="Tasa diaria")
    ax.plot(daily_rate[date_col], daily_rate["roll7"], label="Media móvil 7d")
    ax.legend()
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Tasa onpromotion (0–1)")
    _title_with_range(ax, "Tasa de onpromotion en el tiempo", daily_rate[date_col].min(), daily_rate[date_col].max())
    _save_letter_fig(fig, out_dir / "eda_1_10_promo_rate_over_time.png", dpi=dpi)


def plot_promo_rate_by_dow(train: pd.DataFrame, out_dir: Path,
                           date_col: str, promo_col: str, dpi: int):
    tmp = train.copy()
    tmp["dow"] = tmp[date_col].dt.dayofweek
    by_dow = tmp.groupby("dow", observed=True)[promo_col].mean()
    idx = list(range(7))
    vals = [by_dow.get(i, np.nan) for i in idx]

    fig, ax = plt.subplots()
    ax.bar(idx, vals)
    ax.set_xticks(idx)
    ax.set_xticklabels(_day_of_week_labels(idx))
    ax.set_ylabel("Tasa onpromotion (0–1)")
    ax.set_title("Tasa de onpromotion por día de semana")
    _save_letter_fig(fig, out_dir / "eda_1_11_promo_rate_by_dow.png", dpi=dpi)


def plot_promo_rate_store_hist(train: pd.DataFrame, out_dir: Path,
                               store_col: str, promo_col: str, dpi: int):
    by_store = train.groupby(store_col, observed=True)[promo_col].mean()

    fig, ax = plt.subplots()
    ax.hist(by_store.values, bins=30)
    ax.set_xlabel("Tasa onpromotion por tienda")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de tasa onpromotion a nivel tienda")
    _save_letter_fig(fig, out_dir / "eda_1_12_promo_rate_store_hist.png", dpi=dpi)


def _compute_run_lengths(train: pd.DataFrame,
                         group_cols: List[str],
                         date_col: str,
                         promo_col: str) -> pd.Series:
    """
    Calcula longitudes de rachas (consecutivos de True en onpromotion)
    por grupo (p.ej., tienda-ítem). Devuelve una Serie con las longitudes
    de rachas (sólo para onpromotion=True).
    """
    df = train.sort_values(group_cols + [date_col]).copy()

    # ID incremental de bloque cada vez que cambia el flag, preservando el índice del df
    df["run_id"] = df.groupby(group_cols, observed=True)[promo_col] \
                     .transform(lambda s: s.ne(s.shift()).cumsum())

    # Longitud de cada bloque dentro de cada grupo
    df["run_len"] = df.groupby(group_cols + ["run_id"], observed=True)[promo_col] \
                      .transform("size")

    # Mantener una fila por bloque donde promo=True
    on_blocks = df[df[promo_col]].drop_duplicates(subset=group_cols + ["run_id"])

    return on_blocks["run_len"]


def plot_promo_run_lengths_hist(train: pd.DataFrame, out_dir: Path,
                                date_col: str, store_col: str, item_col: str,
                                promo_col: str, dpi: int):
    runs = _compute_run_lengths(train, [store_col, item_col], date_col, promo_col)

    fig, ax = plt.subplots()
    ax.hist(runs.values, bins=30)
    ax.set_xlabel("Longitud de racha en promoción (días consecutivos)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de longitudes de rachas de onpromotion")
    _save_letter_fig(fig, out_dir / "eda_1_13_promo_run_lengths_hist.png", dpi=dpi)


def plot_promo_transition_matrix(train: pd.DataFrame, out_dir: Path,
                                 date_col: str, store_col: str, item_col: str,
                                 promo_col: str, dpi: int):
    """
    Matriz de transición de onpromotion:
    filas = estado en t-1, columnas = estado en t; valores = proporciones.
    """
    df = train.sort_values([store_col, item_col, date_col]).copy()
    prev = df.groupby([store_col, item_col], observed=True)[promo_col].shift(1)
    mask = prev.notna()
    prev = prev[mask].astype(int)
    curr = df.loc[mask, promo_col].astype(int)

    counts = pd.crosstab(prev, curr)
    # Asegurar 2x2
    for v in [0, 1]:
        if v not in counts.index:
            counts.loc[v] = 0
        if v not in counts.columns:
            counts[v] = 0
    counts = counts.sort_index().sort_index(axis=1)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)

    fig, ax = plt.subplots()
    im = ax.imshow(probs.values, vmin=0, vmax=1)
    # Etiquetas
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["t: 0", "t: 1"]); ax.set_yticklabels(["t-1: 0", "t-1: 1"])
    ax.set_title("Matriz de transición de onpromotion")
    # Anotaciones
    for i in range(2):
        for j in range(2):
            val = probs.values[i, j]
            txt = f"{val:.2f}" if np.isfinite(val) else "nan"
            ax.text(j, i, txt, ha="center", va="center")
    _save_letter_fig(fig, out_dir / "eda_1_14_promo_transition_matrix.png", dpi=dpi)


def plot_promo_crowding_hist(train: pd.DataFrame, out_dir: Path,
                             date_col: str, store_col: str, promo_col: str, dpi: int):
    """
    'Crowding' promocional: # de ítems en promo por tienda y día.
    """
    crowd = (train.groupby([date_col, store_col], observed=True)[promo_col]
                  .sum().reset_index(name="n_items_promo"))

    fig, ax = plt.subplots()
    ax.hist(crowd["n_items_promo"].values, bins=40)
    ax.set_xlabel("# ítems simultáneamente en promoción (tienda/día)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de 'crowding' promocional (tienda/día)")
    _save_letter_fig(fig, out_dir / "eda_1_15_promo_crowding_hist.png", dpi=dpi)


def plot_promo_rate_by_neighborhood(train: pd.DataFrame, items: pd.DataFrame,
                                    out_dir: Path,
                                    item_col: str, promo_col: str,
                                    neighborhood_col: str, dpi: int,
                                    top_n: int = 20):
    if items is None or neighborhood_col not in items.columns:
        return
    df = train.merge(items[[item_col, neighborhood_col]], on=item_col, how="left", validate="many_to_one")
    by_nei = df.groupby(neighborhood_col, observed=True)[promo_col].mean().sort_values(ascending=False)
    top = by_nei.head(top_n)

    fig, ax = plt.subplots()
    ax.bar(range(len(top.index)), top.values)
    ax.set_xticks(range(len(top.index)))
    ax.set_xticklabels(top.index, rotation=45, ha="right")
    ax.set_ylabel("Tasa onpromotion (0–1)")
    ax.set_title(f"Tasa de onpromotion por {neighborhood_col} (top {top_n})")
    _save_letter_fig(fig, out_dir / "eda_1_16_promo_rate_by_neighborhood.png", dpi=dpi)


# --------------------------- Runner principal --------------------------- #

def run_eda(
    train_path: str,
    out_dir: str,
    transactions_path: Optional[str] = None,
    items_path: Optional[str] = None,
    # nombres de columnas
    date_col: str = "date",
    store_col: str = "store_nbr",
    item_col: str = "item_nbr",
    promo_col: str = "onpromotion",
    sales_col: Optional[str] = "unit_sales",
    tx_col: str = "transactions",
    neighborhood_col: Optional[str] = None,
    # rendering
    dpi: int = DEFAULT_DPI,
) -> Dict[str, str]:
    """
    Ejecuta el EDA completo y devuelve un dict {nombre_figura: ruta}.
    """
    out = {}
    out_dir = _ensure_out_dir(out_dir)

    # 1) Carga
    train, transactions, items = load_data_for_eda(
        train_path=train_path,
        transactions_path=transactions_path,
        items_path=items_path,
        date_col=date_col,
        store_col=store_col,
        item_col=item_col,
        promo_col=promo_col,
        sales_col=sales_col,
        tx_col=tx_col,
        neighborhood_col=neighborhood_col,
    )

    # 2) Selección de variables de interés (para el estudio)
    #    Mínimas: date, store, item, onpromotion; opcionales: unit_sales, transactions (+ merge por tienda/día)
    base_cols = [date_col, store_col, item_col, promo_col]
    opt_cols = [c for c in [sales_col] if c]

    train = _select_vars_of_interest(train, base_cols, opt_cols)

    # 2b) Merge transacciones a nivel tienda/día (si se provee)
    if transactions is not None:
        transactions[tx_col] = _safe_numeric(transactions[tx_col])
        train = train.merge(
            transactions[[date_col, store_col, tx_col]],
            on=[date_col, store_col],
            how="left",
            validate="many_to_one"
        )
        opt_cols.append(tx_col)

    # 3) Gráficos generales (tamaño carta)
    plot_daily_count(train, out_dir, date_col, dpi); out["eda_1_01_data_coverage.png"] = str(out_dir / "eda_1_01_data_coverage.png")

    if transactions is not None:
        plot_transactions_trend(transactions, out_dir, date_col, tx_col, dpi)
        out["eda_1_02_transactions_trend.png"] = str(out_dir / "eda_1_02_transactions_trend.png")

    if sales_col and sales_col in train.columns:
        plot_unit_sales_trend(train, out_dir, date_col, sales_col, dpi)
        out["eda_1_03_unit_sales_trend.png"] = str(out_dir / "eda_1_03_unit_sales_trend.png")

    # Missingness sólo sobre columnas de interés
    plot_missingness_bar(train[[c for c in base_cols + opt_cols if c in train.columns]], out_dir, dpi)
    out["eda_1_04_missingness_bar.png"] = str(out_dir / "eda_1_04_missingness_bar.png")

    # 4) Apartado profundo: onpromotion
    # 4.1) Tasa temporal
    plot_promo_rate_over_time(train, out_dir, date_col, promo_col, dpi)
    out["eda_1_10_promo_rate_over_time.png"] = str(out_dir / "eda_1_10_promo_rate_over_time.png")

    # 4.2) Estacionalidad semanal
    plot_promo_rate_by_dow(train, out_dir, date_col, promo_col, dpi)
    out["eda_1_11_promo_rate_by_dow.png"] = str(out_dir / "eda_1_11_promo_rate_by_dow.png")

    # 4.3) Heterogeneidad entre tiendas
    plot_promo_rate_store_hist(train, out_dir, store_col, promo_col, dpi)
    out["eda_1_12_promo_rate_store_hist.png"] = str(out_dir / "eda_1_12_promo_rate_store_hist.png")

    # 4.4) Dinámica de rachas
    plot_promo_run_lengths_hist(train, out_dir, date_col, store_col, item_col, promo_col, dpi)
    out["eda_1_13_promo_run_lengths_hist.png"] = str(out_dir / "eda_1_13_promo_run_lengths_hist.png")

    # 4.5) Persistencia/alternancia (matriz de transición)
    plot_promo_transition_matrix(train, out_dir, date_col, store_col, item_col, promo_col, dpi)
    out["eda_1_14_promo_transition_matrix.png"] = str(out_dir / "eda_1_14_promo_transition_matrix.png")

    # 4.6) Crowding promocional por tienda/día
    plot_promo_crowding_hist(train, out_dir, date_col, store_col, promo_col, dpi)
    out["eda_1_15_promo_crowding_hist.png"] = str(out_dir / "eda_1_15_promo_crowding_hist.png")

    # 4.7) Heterogeneidad por vecindario (si hay items + neighborhood_col)
    if items is not None and neighborhood_col:
        plot_promo_rate_by_neighborhood(train, items, out_dir, item_col, promo_col, neighborhood_col, dpi, top_n=20)
        out["eda_1_16_promo_rate_by_neighborhood.png"] = str(out_dir / "eda_1_16_promo_rate_by_neighborhood.png")

    return out


# --------------------------- CLI --------------------------- #

def _parse_args():
    ap = argparse.ArgumentParser(description="EDA posterior a data_quality (salidas tamaño carta).")
    ap.add_argument("--train", required=True, help="Ruta a train_filtered.csv")
    ap.add_argument("--transactions", default=None, help="Ruta a transactions_filtered.csv (opcional)")
    ap.add_argument("--items", default=None, help="Ruta a items.csv (opcional)")
    ap.add_argument("--neighborhood-col", default=None, help="Nombre de columna de vecindario (p.ej. 'family')")
    ap.add_argument("--out-dir", required=True, help="Directorio de salida para imágenes PNG")
    # columnas
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--store-col", default="store_nbr")
    ap.add_argument("--item-col", default="item_nbr")
    ap.add_argument("--promo-col", default="onpromotion")
    ap.add_argument("--sales-col", default="unit_sales")
    ap.add_argument("--tx-col", default="transactions")
    # rendering
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Resolución de las figuras (default 150)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    paths = run_eda(
        train_path=args.train,
        out_dir=args.out_dir,
        transactions_path=args.transactions,
        items_path=args.items,
        date_col=args.date_col,
        store_col=args.store_col,
        item_col=args.item_col,
        promo_col=args.promo_col,
        sales_col=args.sales_col,
        tx_col=args.tx_col,
        neighborhood_col=args.neighborhood_col,
        dpi=args.dpi,
    )
    # Pequeño log en consola
    print("Figuras generadas (tamaño carta):")
    for k, v in sorted(paths.items()):
        print(f" - {k}: {v}")