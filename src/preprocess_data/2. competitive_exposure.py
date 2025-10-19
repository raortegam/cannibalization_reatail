# competitive_exposure.py
# -*- coding: utf-8 -*-
"""
Cálculo de exposición competitiva SIN validaciones de calidad de datos.

Asume que:
- train_filtered.csv viene limpio (sin duplicados por llave) desde data_quality.py
- onpromotion es booleano
- existe columna 'neighborhood' en train o se provee items.csv + neighborhood_col

Uso por CLI (ejemplos):

A) Si 'neighborhood' ya está en train_filtered.csv:
    python competitive_exposure.py \
        --train /ruta/data/train_filtered.csv \
        --neighborhood-col neighborhood \
        --save /ruta/out/train_with_exposure.parquet

B) Si 'neighborhood' viene desde items.csv:
    python competitive_exposure.py \
        --train /ruta/data/train_filtered.csv \
        --items /ruta/data/items.csv \
        --neighborhood-col family \
        --save /ruta/out/train_with_exposure.parquet
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


def compute_competitive_exposure(
    train_path: str,
    items_path: Optional[str] = None,
    # nombres de columnas
    date_col: str = "date",
    store_col: str = "store_nbr",
    item_col: str = "item_nbr",
    promo_col: str = "onpromotion",
    neighborhood_col: str = "neighborhood",
    # salida
    save_path: Optional[str] = None,
    save_format: Optional[str] = None,  # "csv" | "parquet"; si None, infiere por extensión de save_path
) -> pd.DataFrame:
    """Calcula competitive exposure excluyendo el ítem propio."""
    train = pd.read_csv(train_path, parse_dates=[date_col])

    # neighborhood: merge opcional si no existe en train
    if neighborhood_col not in train.columns:
        if items_path is None:
            raise ValueError(
                f"No se encontró '{neighborhood_col}' en train y no se proporcionó items_path."
            )
        items = pd.read_csv(items_path, usecols=[item_col, neighborhood_col])
        train = train.merge(items, on=item_col, how="left", validate="many_to_one")

    # Definición de grupo (tienda-fecha-vecindario)
    grp_cols: List[str] = [date_col, store_col, neighborhood_col]

    # Conteos por grupo (vectorizado)
    # n_total: número de filas (cada fila debe ser única por item)
    n_total = train.groupby(grp_cols, observed=True)[item_col].transform("size")
    # n_promos: número de items en promo
    # onpromotion se asume booleano; sum cuenta True
    n_promos = train.groupby(grp_cols, observed=True)[promo_col].transform("sum")

    # Excluir el propio ítem (self-exclusion)
    self_in_promo = train[promo_col].astype("int8")
    numer = n_promos - self_in_promo
    denom = n_total - 1

    exposure = np.where(denom > 0, numer / denom, np.nan)
    # Boundaries por estabilidad numérica
    exposure = np.clip(exposure, 0.0, 1.0)

    out_cols = [date_col, store_col, item_col, "competitive_exposure"]
    out = train[[date_col, store_col, item_col]].copy()
    out["competitive_exposure"] = exposure

    # Guardado opcional
    if save_path:
        save_path = str(Path(save_path))
        ext = save_format or Path(save_path).suffix.lower().replace(".", "")
        if ext in ("parquet", "pq"):
            out.to_parquet(save_path, index=False)
        else:
            out.to_csv(save_path, index=False)

    return out


# ---------------------------- CLI ---------------------------- #

def _parse_args():
    ap = argparse.ArgumentParser(description="Competitive exposure (sin data quality).")
    ap.add_argument("--train", required=True, help="Ruta a train_filtered.csv")
    ap.add_argument("--items", default=None, help="Ruta a items.csv (opcional)")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--store-col", default="store_nbr")
    ap.add_argument("--item-col", default="item_nbr")
    ap.add_argument("--promo-col", default="onpromotion")
    ap.add_argument("--neighborhood-col", default="neighborhood", help="Familia/categoría/clase")
    ap.add_argument("--save", default=None, help="Ruta de salida (csv o parquet)")
    ap.add_argument("--save-format", default=None, help='"csv" o "parquet"; si None, infiere de la extensión')
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compute_competitive_exposure(
        train_path=args.train,
        items_path=args.items,
        date_col=args.date_col,
        store_col=args.store_col,
        item_col=args.item_col,
        promo_col=args.promo_col,
        neighborhood_col=args.neighborhood_col,
        save_path=args.save,
        save_format=args.save_format,
    )
