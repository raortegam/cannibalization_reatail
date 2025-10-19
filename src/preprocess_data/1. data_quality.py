# data_quality.py
# -*- coding: utf-8 -*-
"""
Módulo de calidad de datos para retail time-series.

Responsabilidades:
- Ingesta segura de train.csv y transactions.csv
- Normalización de tipos (date, bool)
- Desduplicación por llaves naturales
- Agregación coherente en duplicados
- Filtro por fecha >= min_date (default: 2016-01-01)
- Exporta train_filtered.csv y transactions_filtered.csv

Uso por CLI (ejemplos):
    python data_quality.py \
        --train /ruta/data/train.csv \
        --transactions /ruta/data/transactions.csv \
        --min-date 2016-01-01

Parámetros de columna son configurables.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------- utils ---------------------------- #

TRUE_LIKE = {"1", "true", "t", "yes", "y", "si", "sí"}
FALSE_LIKE = {"0", "false", "f", "no", "n", "nan", "none", ""}

def normalize_onpromotion(s: pd.Series) -> pd.Series:
    """Normaliza una serie a booleano robustamente."""
    if s.dtype == bool:
        return s.fillna(False)
    # Nota: astype(str) convierte NaN a 'nan'. Lo mapeamos a False.
    mapped = (
        s.astype(str)
         .str.strip()
         .str.lower()
         .replace({v: True for v in TRUE_LIKE})
         .replace({v: False for v in FALSE_LIKE})
    )
    return mapped.fillna(False).astype(bool)


def ensure_required_columns(df: pd.DataFrame, required: List[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] Faltan columnas requeridas: {missing}. "
            f"Columnas presentes: {list(df.columns)}"
        )


def to_datetime_col(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    if df[date_col].isna().any():
        bad = df[df[date_col].isna()]
        raise ValueError(
            f"Fechas no parseables encontradas en {date_col}. Ejemplos:\n"
            f"{bad.head(5)}"
        )
    # Estándar: mantener timezone-naive diario
    df[date_col] = df[date_col].dt.tz_localize(None)
    return df


def aggregate_train_duplicates(
    df: pd.DataFrame,
    keys: List[str],
    promo_col: str,
    sales_col: Optional[str] = None,
) -> pd.DataFrame:
    """Si hay duplicados por (date, store, item), agrega con reglas coherentes."""
    df = df.copy()
    dup_mask = df.duplicated(subset=keys, keep=False)
    if not dup_mask.any():
        return df

    agg_map: Dict[str, str] = {promo_col: "max"}
    if sales_col and sales_col in df.columns:
        agg_map[sales_col] = "sum"

    # Para el resto de columnas no-clave, usa 'last' determinista
    for c in df.columns:
        if c in keys or c in agg_map:
            continue
        agg_map[c] = "last"

    logging.warning(
        f"[train] Se encontraron {dup_mask.sum()} filas duplicadas por {keys}. "
        "Se agregan con reglas: onpromotion=max, unit_sales=sum, resto=last."
    )
    out = df.groupby(keys, as_index=False, sort=False).agg(agg_map)
    return out


def aggregate_transactions_duplicates(
    df: pd.DataFrame,
    keys: List[str],
    tx_col: str,
) -> pd.DataFrame:
    """Agrega duplicados en transacciones por (date, store) con sum."""
    df = df.copy()
    dup_mask = df.duplicated(subset=keys, keep=False)
    if not dup_mask.any():
        return df

    logging.warning(
        f"[transactions] Se encontraron {dup_mask.sum()} filas duplicadas por {keys}. "
        "Se agregan con transactions=sum, resto=last."
    )
    agg_map: Dict[str, str] = {tx_col: "sum"}
    for c in df.columns:
        if c in keys or c in agg_map:
            continue
        agg_map[c] = "last"

    out = df.groupby(keys, as_index=False, sort=False).agg(agg_map)
    return out


def filter_by_min_date(df: pd.DataFrame, date_col: str, min_date: str) -> pd.DataFrame:
    df = df.copy()
    return df[df[date_col] >= pd.Timestamp(min_date)].reset_index(drop=True)


def report_summary_train(
    df: pd.DataFrame,
    date_col: str,
    store_col: str,
    item_col: str,
    promo_col: str,
    sales_col: Optional[str],
) -> Dict[str, object]:
    by_store = df.groupby(store_col, observed=True)
    out = {
        "rows": int(df.shape[0]),
        "date_range": [str(df[date_col].min().date()), str(df[date_col].max().date())],
        "n_stores": int(df[store_col].nunique()),
        "n_items": int(df[item_col].nunique()),
        "promo_rate": float(df[promo_col].mean()),
    }
    if sales_col and sales_col in df.columns:
        out["sales_nonnull_rate"] = float(df[sales_col].notna().mean())
        out["sales_negative_rate"] = float((df[sales_col] < 0).mean())
    out["stores_rows_min_max"] = [int(by_store.size().min()), int(by_store.size().max())]
    return out


def report_summary_transactions(
    df: pd.DataFrame, date_col: str, store_col: str, tx_col: str
) -> Dict[str, object]:
    out = {
        "rows": int(df.shape[0]),
        "date_range": [str(df[date_col].min().date()), str(df[date_col].max().date())],
        "n_stores": int(df[store_col].nunique()),
        "transactions_nonneg_rate": float((df[tx_col] >= 0).mean()),
    }
    return out


# ---------------------------- pipeline ---------------------------- #

def run_data_quality(
    train_path: str,
    transactions_path: str,
    out_dir: Optional[str] = None,
    min_date: str = "2016-01-01",
    # nombres de columnas (configurables)
    date_col: str = "date",
    store_col: str = "store_nbr",
    item_col: str = "item_nbr",
    promo_col: str = "onpromotion",
    sales_col: Optional[str] = "unit_sales",
    tx_col: str = "transactions",
    # reporte opcional
    save_report_json: Optional[str] = None,
) -> Dict[str, object]:
    """Ejecuta la limpieza, deduplicación y filtra por fecha; guarda *_filtered.csv."""
    logging.info("Iniciando pipeline de calidad de datos...")
    train_path = str(Path(train_path))
    transactions_path = str(Path(transactions_path))
    out_dir_path = Path(out_dir) if out_dir else Path(train_path).parent
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # --- Carga ---
    logging.info(f"Cargando train: {train_path}")
    train = pd.read_csv(train_path)
    logging.info(f"Cargando transactions: {transactions_path}")
    transactions = pd.read_csv(transactions_path)

    # --- Requisitos mínimos ---
    ensure_required_columns(train, [date_col, store_col, item_col, promo_col], "train")
    ensure_required_columns(transactions, [date_col, store_col, tx_col], "transactions")

    # --- Tipos y normalizaciones ---
    train = to_datetime_col(train, date_col)
    transactions = to_datetime_col(transactions, date_col)

    # onpromotion → bool
    train[promo_col] = normalize_onpromotion(train[promo_col])

    # conversión segura de tipos enteros
    if store_col in train.columns:
        train[store_col] = pd.to_numeric(train[store_col], errors="coerce").astype("Int32")
    if item_col in train.columns:
        train[item_col] = pd.to_numeric(train[item_col], errors="coerce").astype("Int32")
    if store_col in transactions.columns:
        transactions[store_col] = pd.to_numeric(transactions[store_col], errors="coerce").astype("Int32")
    transactions[tx_col] = pd.to_numeric(transactions[tx_col], errors="coerce")

    # --- Desduplicación coherente ---
    keys_train = [date_col, store_col, item_col]
    train = aggregate_train_duplicates(train, keys_train, promo_col, sales_col)

    keys_tx = [date_col, store_col]
    transactions = aggregate_transactions_duplicates(transactions, keys_tx, tx_col)

    # --- Filtro por fecha ---
    train = filter_by_min_date(train, date_col, min_date)
    transactions = filter_by_min_date(transactions, date_col, min_date)

    # --- QoL: NaN y restricciones suaves ---
    if sales_col and sales_col in train.columns:
        train[sales_col] = pd.to_numeric(train[sales_col], errors="coerce")
        # no alteramos negativos (devoluciones), sólo informamos en reporte

    transactions[tx_col] = transactions[tx_col].fillna(0)
    # Clip transacciones negativas a 0 (si existieran por errores de captura)
    neg_tx = (transactions[tx_col] < 0).sum()
    if neg_tx > 0:
        logging.warning(f"[transactions] {neg_tx} valores negativos en {tx_col} -> clip a 0")
        transactions.loc[transactions[tx_col] < 0, tx_col] = 0

    # --- Guardado ---
    train_out = out_dir_path / "train_filtered.csv"
    tx_out = out_dir_path / "transactions_filtered.csv"
    train.to_csv(train_out, index=False)
    transactions.to_csv(tx_out, index=False)
    logging.info(f"Guardado: {train_out}")
    logging.info(f"Guardado: {tx_out}")

    # --- Reporte ---
    rep = {
        "train_summary": report_summary_train(train, date_col, store_col, item_col, promo_col, sales_col),
        "transactions_summary": report_summary_transactions(transactions, date_col, store_col, tx_col),
        "paths": {"train_filtered": str(train_out), "transactions_filtered": str(tx_out)},
        "min_date": min_date,
        "keys_train": keys_train,
        "keys_transactions": keys_tx,
    }
    if save_report_json:
        save_path = Path(save_report_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        logging.info(f"Reporte JSON guardado en {save_path}")

    return rep


# ---------------------------- CLI ---------------------------- #

def _parse_args():
    ap = argparse.ArgumentParser(description="Pipeline de calidad de datos para retail.")
    ap.add_argument("--train", required=True, help="Ruta a train.csv")
    ap.add_argument("--transactions", required=True, help="Ruta a transactions.csv")
    ap.add_argument("--out-dir", default=None, help="Directorio de salida (default: igual al de train)")
    ap.add_argument("--min-date", default="2016-01-01", help="Fecha mínima inclusiva (YYYY-MM-DD)")
    # columnas
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--store-col", default="store_nbr")
    ap.add_argument("--item-col", default="item_nbr")
    ap.add_argument("--promo-col", default="onpromotion")
    ap.add_argument("--sales-col", default="unit_sales")
    ap.add_argument("--tx-col", default="transactions")
    # reporte
    ap.add_argument("--save-report-json", default=None, help="Ruta para guardar reporte JSON")
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_data_quality(
        train_path=args.train,
        transactions_path=args.transactions,
        out_dir=args.out_dir,
        min_date=args.min_date,
        date_col=args.date_col,
        store_col=args.store_col,
        item_col=args.item_col,
        promo_col=args.promo_col,
        sales_col=args.sales_col,
        tx_col=args.tx_col,
        save_report_json=args.save_report_json,
    )