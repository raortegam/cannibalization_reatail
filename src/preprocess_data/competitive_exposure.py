# -*- coding: utf-8 -*-
"""
h_exposure.py — versión robusta (continuidad garantizada desde la primera aparición)
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Literal, Optional

import numpy as np
import pandas as pd

# ----------------------------
# Utilidades
# ----------------------------

def _to_int_nullable(s: pd.Series, bits: int = 32) -> pd.Series:
    """Convierte a entero anulable (Int8/16/32) vía to_numeric (coerce)."""
    s_num = pd.to_numeric(s, errors="coerce")
    if bits == 8:
        return s_num.astype("Int8")
    if bits == 16:
        return s_num.astype("Int16")
    return s_num.astype("Int32")

def _ensure_binary_onpromotion(s: pd.Series) -> pd.Series:
    """
    Convierte {True/False, "True"/"False", "yes"/"no", 1/0, "1"/"0", NaN, espacios} → {0,1} (int8).
    """
    if s.dtype == bool:
        out = s.fillna(False).astype("int8")
    elif pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0, upper=1).astype("int8")
    else:
        tmp = (
            s.astype(str)
             .str.strip()
             .str.lower()
             .replace({"": "false", "nan": "false", "none": "false"})
        )
        out = tmp.isin({"true","1","t","yes","y"}).astype("int8")
    out.name = "onpromotion"
    return out

def _pick_class_column(items: pd.DataFrame) -> str:
    if "class" in items.columns:
        return "class"
    if "family" in items.columns:
        return "family"
    raise KeyError("items.csv debe contener columna 'class' o 'family'.")

def _append_csv(df: pd.DataFrame, path: str, compression: Optional[str]) -> None:
    """
    Escribe en modo append garantizando encabezado solo si el archivo no existe o está vacío.
    Resistente a reintentos/corridas múltiples.
    """
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    df.to_csv(path, mode="a", index=False, header=need_header, compression=compression)

# ----------------------------
# Núcleo
# ----------------------------

def _emit_h_for_block(
    df_block: pd.DataFrame,
    cls_col: str,
    bin_mode: Literal["any", "threshold"] = "any",
    bin_tau: float = 0.30,
) -> pd.DataFrame:
    """
    Calcula exposición H_{ist} para un bloque DENSIFICADO (continuo) donde cada (store,item)
    tiene una fila por día desde su primera aparición.
    """
    g = (
        df_block
        .groupby(["date", "store_nbr", cls_col], observed=True, sort=False)
        .agg(n_items=("item_nbr", "size"),
             n_promo=("onpromotion", "sum"))
        .reset_index()
    )

    df = df_block.merge(g, on=["date", "store_nbr", cls_col], how="left", copy=False)

    neighbors = (df["n_promo"].astype("int32") - df["onpromotion"].astype("int32")).clip(lower=0)
    denom     = (df["n_items"].astype("int32") - 1).clip(lower=1)

    h_count = neighbors.astype("int32")
    h_prop  = (neighbors / denom).astype("float32")

    if bin_mode == "any":
        h_bin = (h_count >= 1).astype("int8")
    elif bin_mode == "threshold":
        h_bin = (h_prop >= float(bin_tau)).astype("int8")
    else:
        raise ValueError("bin_mode debe ser 'any' o 'threshold'.")

    out = df[["date", "store_nbr", "item_nbr"]].copy()
    out["H_prop"]  = h_prop.values.astype("float32")
    out["H_count"] = h_count.values.astype("int32")
    out["H_bin"]   = h_bin.values.astype("int8")

    return out

def _update_first_seen(
    first_seen: pd.DataFrame,
    df_new: pd.DataFrame,
    cls_col: str
) -> pd.DataFrame:
    """
    Actualiza (en streaming) la fecha de primera aparición por (store, clase, item).
    df_new debe tener columnas: ['store_nbr','item_nbr', cls_col, 'date']
    """
    if df_new.empty:
        return first_seen

    min_new = (
        df_new[["store_nbr", "item_nbr", cls_col, "date"]]
        .groupby(["store_nbr", "item_nbr", cls_col], observed=True, sort=False)["date"]
        .min()
        .reset_index()
        .rename(columns={"date": "first_seen_date"})
    )

    if first_seen.empty:
        return min_new

    fs_all = pd.concat([first_seen, min_new], ignore_index=True, copy=False)
    fs_all = (
        fs_all
        .groupby(["store_nbr", "item_nbr", cls_col], observed=True, sort=False)["first_seen_date"]
        .min()
        .reset_index()
    )
    return fs_all

def _densify_block_with_roster(
    emit_block: pd.DataFrame,
    first_seen: pd.DataFrame,
    cls_col: str
) -> pd.DataFrame:
    """
    Densifica el bloque: para cada (date, store, clase) crea una fila por cada item cuyo
    first_seen_date <= date. Si falta onpromotion ese día, se imputa 0.
    """
    if emit_block.empty:
        return emit_block

    # Claves únicas (día-store-clase) presentes en el bloque que se va a emitir
    keys = emit_block[["date", "store_nbr", cls_col]].drop_duplicates()

    if first_seen.empty:
        # Primer pasada: aún no hay roster consolidado. Normalizar y devolver tal cual.
        emit_block = emit_block.copy()
        emit_block["onpromotion"] = _ensure_binary_onpromotion(emit_block["onpromotion"])
        return emit_block[["date", "store_nbr", "item_nbr", "onpromotion", cls_col]]

    # Restringir roster a las combinaciones store-clase del bloque
    roster = first_seen.merge(
        keys[["store_nbr", cls_col]].drop_duplicates(),
        on=["store_nbr", cls_col], how="inner", copy=False
    )

    if roster.empty:
        emit_block = emit_block.copy()
        emit_block["onpromotion"] = _ensure_binary_onpromotion(emit_block["onpromotion"])
        return emit_block[["date", "store_nbr", "item_nbr", "onpromotion", cls_col]]

    # Producto cartesiano por (store, clase): roster x días, y filtrar por first_seen_date <= date
    grid = roster.merge(keys, on=["store_nbr", cls_col], how="inner", copy=False)
    grid = grid.loc[grid["first_seen_date"] <= grid["date"], ["date", "store_nbr", cls_col, "item_nbr"]]

    # Traer onpromotion observada; donde falte, imputar 0 para asegurar continuidad
    present = (
        emit_block[["date", "store_nbr", cls_col, "item_nbr", "onpromotion"]]
        .drop_duplicates()
    )
    grid = grid.merge(present, on=["date", "store_nbr", cls_col, "item_nbr"], how="left", copy=False)
    grid["onpromotion"] = _ensure_binary_onpromotion(grid["onpromotion"])

    # Orden y columnas esperadas
    return grid[["date", "store_nbr", "item_nbr", "onpromotion", cls_col]]

def compute_h_exposure(
    train_csv: str,
    items_csv: str,
    out_csv: str,
    chunk_rows: int = 10_000_000,
    bin_mode: Literal["any", "threshold"] = "any",
    bin_tau: float = 0.30,
    keep_class_in_output: bool = False,
    compression: Optional[str] = None,
    overwrite: bool = True,
) -> None:
    """
    Calcula H_{ist} en streaming por fecha y escribe a CSV (append por chunks).
    Asegura continuidad diaria para cada (store,item) desde su primera aparición.
    """

    # 0) Preparar salida
    if overwrite and os.path.exists(out_csv):
        open(out_csv, "w").close()  # truncar

    # 1) Cargar items una sola vez (evitar doble lectura)
    items_head = pd.read_csv(items_csv, nrows=0)
    usecols_items = [c for c in ["item_nbr", "class", "family"] if c in items_head.columns]
    items = pd.read_csv(items_csv, usecols=usecols_items)

    # Deduplicar por item_nbr para asegurar merge 1:1
    if items.duplicated(subset=["item_nbr"]).any():
        # Heurística: nos quedamos con la primera aparición no-nula de clase/familia
        items = (items
                 .sort_values(usecols_items)
                 .drop_duplicates(subset=["item_nbr"], keep="first"))

    cls_col = _pick_class_column(items)

    # Tipos
    items["item_nbr"] = _to_int_nullable(items["item_nbr"], bits=32)
    if cls_col == "class":
        items["class"] = _to_int_nullable(items["class"], bits=16)
    else:
        items["family"] = items["family"].astype("category")

    # 1.5) Estructura para first_seen (roster incremental)
    first_seen = pd.DataFrame({
        "store_nbr": pd.Series([], dtype="int16"),
        "item_nbr":  pd.Series([], dtype="int32"),
        cls_col:     pd.Series([], dtype=items[cls_col].dtype),
        "first_seen_date": pd.Series([], dtype="datetime64[ns]"),
    })

    # 2) Lectura por chunks del train
    usecols_train = ["date", "store_nbr", "item_nbr", "onpromotion"]
    dtypes_train  = {"store_nbr": "int16", "item_nbr": "int32"}  # compactos

    reader = pd.read_csv(
        train_csv,
        usecols=usecols_train,
        dtype=dtypes_train,
        parse_dates=["date"],
        chunksize=chunk_rows,
        low_memory=False,
    )

    buffer = pd.DataFrame(columns=["date", "store_nbr", "item_nbr", "onpromotion"])
    buffer["date"] = pd.to_datetime(buffer["date"])

    processed_rows = 0
    last_seen_date = None

    for chunk_idx, ch in enumerate(reader, start=1):
        # Normalizar onpromotion
        ch["onpromotion"] = _ensure_binary_onpromotion(ch["onpromotion"])

        # Merge con items (tras deduplicar) → debería ser 1:1
        ch = ch.merge(items, on="item_nbr", how="left", validate="many_to_one")

        # Filas sin clase/familia
        missing_cls = ch[cls_col].isna().sum()
        if missing_cls:
            print(f"[WARN] Filas sin {cls_col} en chunk {chunk_idx}: {missing_cls:,}", file=sys.stderr)
            ch = ch[~ch[cls_col].isna()]

        # Concatenar con buffer del día frontera
        df_all = pd.concat([buffer, ch], ignore_index=True, copy=False)

        # Emitir todo lo que sea < fecha máxima del bloque
        dmax = df_all["date"].max()
        to_emit = df_all["date"] < dmax
        emit_block_raw = df_all.loc[to_emit].copy()
        buffer = df_all.loc[~to_emit].copy()

        if not emit_block_raw.empty:
            # Actualizar roster (first_seen) con lo que efectivamente vamos a emitir ahora
            first_seen = _update_first_seen(first_seen,
                                            emit_block_raw[["store_nbr","item_nbr",cls_col,"date"]],
                                            cls_col=cls_col)

            # Densificar para garantizar continuidad diaria desde la primera aparición
            emit_block_dense = _densify_block_with_roster(
                emit_block_raw[["date","store_nbr","item_nbr","onpromotion",cls_col]],
                first_seen=first_seen,
                cls_col=cls_col
            )

            # Calcular H usando el bloque ya densificado
            block_out = _emit_h_for_block(
                emit_block_dense, cls_col=cls_col, bin_mode=bin_mode, bin_tau=bin_tau
            )

            if keep_class_in_output:
                block_out = block_out.merge(
                    emit_block_dense[["date", "store_nbr", "item_nbr", cls_col]],
                    on=["date", "store_nbr", "item_nbr"],
                    how="left"
                )

            # Garantizar tipos antes de escribir (consistencia de schema)
            block_out = block_out.astype({
                "store_nbr": "int16",
                "item_nbr":  "int32",
                "H_count":   "int32",
                "H_bin":     "int8",
                "H_prop":    "float32",
            })

            _append_csv(block_out, out_csv, compression)
            processed_rows += len(block_out)

        # Chequeo de monotonicidad
        curr_min, curr_max = ch["date"].min(), ch["date"].max()
        if last_seen_date is not None and (curr_min < last_seen_date):
            print(
                f"[WARN] Retroceso de fecha: chunk {chunk_idx} min {curr_min.date()} < última {last_seen_date.date()}."
                " El algoritmo asume orden no-decreciente por 'date'.",
                file=sys.stderr
            )
        last_seen_date = curr_max

    # Último día (frontera)
    if not buffer.empty:
        # Actualizar roster con lo que queda y densificar
        first_seen = _update_first_seen(first_seen,
                                        buffer[["store_nbr","item_nbr",cls_col,"date"]],
                                        cls_col=cls_col)

        buffer_dense = _densify_block_with_roster(
            buffer[["date","store_nbr","item_nbr","onpromotion",cls_col]],
            first_seen=first_seen,
            cls_col=cls_col
        )

        last_out = _emit_h_for_block(buffer_dense, cls_col=cls_col, bin_mode=bin_mode, bin_tau=bin_tau)
        if keep_class_in_output:
            last_out = last_out.merge(
                buffer_dense[["date", "store_nbr", "item_nbr", cls_col]],
                on=["date", "store_nbr", "item_nbr"], how="left"
            )
        last_out = last_out.astype({
            "store_nbr": "int16",
            "item_nbr":  "int32",
            "H_count":   "int32",
            "H_bin":     "int8",
            "H_prop":    "float32",
        })
        _append_csv(last_out, out_csv, compression)
        processed_rows += len(last_out)

    print(f"[OK] H_exposure escrito en: {out_csv}  | Filas: {processed_rows:,}")

# ----------------------------
# CLI
# ----------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cálculo de exposición competitiva H_{ist} en streaming.")
    p.add_argument("--train", required=True, help="Ruta a train.csv")
    p.add_argument("--items", required=True, help="Ruta a items.csv")
    p.add_argument("--out", required=True, help="Ruta de salida CSV (se creará o se truncará)")
    p.add_argument("--chunk-rows", type=int, default=1_500_000, help="Tamaño de chunk (filas)")
    p.add_argument("--bin-mode", choices=["any", "threshold"], default="any")
    p.add_argument("--bin-tau", type=float, default=0.30)
    p.add_argument("--keep-class", action="store_true")
    p.add_argument("--compress", choices=["gzip", "bz2", "xz", "zip"], default=None)
    p.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                   help="No truncar out si ya existe (apéndice a prueba de encabezados).")
    p.set_defaults(overwrite=True)
    return p

def main():
    args = _build_arg_parser().parse_args()
    compute_h_exposure(
        train_csv=args.train,
        items_csv=args.items,
        out_csv=args.out,
        chunk_rows=args.chunk_rows,
        bin_mode=args.bin_mode,
        bin_tau=args.bin_tau,
        keep_class_in_output=args.keep_class,
        compression=args.compress,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()