# -*- coding: utf-8 -*-
"""
Cálculo de exposición competitiva con opción de *preprocesamiento causal* (suavizado + winsor).

- Mantiene compatibilidad con el pipeline:
    * H_prop        : por defecto, suavizada + winsorizada (causal).
    * H_prop_raw    : cruda (sin preprocesar).
    * H_bin         : binaria derivada de la serie cruda (no se suaviza).
    * H_disc        : conteo crudo (no se suaviza).
    * competitive_exposure = H_prop (alias).

- Sin fugas temporales: el suavizado usa sólo pasado y presente (EMA con adjust=False
  o media móvil con rolling(min_periods=1)).

Uso por CLI (ejemplos):

A) Si 'neighborhood' ya está en train_filtered.csv:
    python competitive_exposure.py \
        --train /ruta/data/train_filtered.csv \
        --neighborhood-col class \
        --save /ruta/out/train_with_exposure.parquet

B) Si 'neighborhood' viene desde items.csv:
    python competitive_exposure.py \
        --train /ruta/data/train_filtered.csv \
        --items /ruta/data/items.csv \
        --neighborhood-col family \
        --save /ruta/out/train_with_exposure.parquet

Parámetros de preprocesamiento (principales):
    --preprocess / --no-preprocess
    --smooth-kind {ema,ma}
    --ma-window 7
    --ema-span 7   (o --ema-halflife, o --ema-alpha)
    --winsor-low 0.005 --winsor-high 0.995
    --winsor-by {global,store_item,store_neighborhood}
    --winsor-min-obs 30
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


# ----------------------- utilidades internas ----------------------- #

def _coerce_bool01(x: pd.Series) -> pd.Series:
    """Normaliza onpromotion a {0,1}."""
    if x.dtype == bool:
        return x.astype("int8")
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce").fillna(0).clip(0, 1).astype("int8")
    return x.fillna("False").astype(str).str.lower().isin(["true", "t", "1", "yes"]).astype("int8")


def _ema_alpha(span: Optional[float] = None,
               halflife: Optional[float] = None,
               alpha: Optional[float] = None) -> float:
    """Prioridad: alpha > halflife > span. Devuelve alpha en (0,1]."""
    if alpha is not None:
        return float(np.clip(alpha, 1e-6, 1.0))
    if halflife is not None:
        # alpha = 1 - 0.5**(1/halflife)
        return float(1.0 - np.power(0.5, 1.0 / max(1e-6, float(halflife))))
    if span is None:
        span = 7.0
    # alpha ≈ 2/(span+1)
    return float(2.0 / (float(span) + 1.0))


def _smooth_grouped(
    series: pd.Series,
    group_keys: List[str],
    dates: pd.Series,
    *,
    group_df: pd.DataFrame,
    kind: str = "ema",
    ma_window: int = 7,
    ema_alpha_val: float = 0.25
) -> pd.Series:
    """
    Suaviza por grupos (p.ej., (store,item)) de forma *causal* (sin ver futuro).
    - kind='ema' -> ewm(adjust=False)
    - kind='ma'  -> rolling(window, min_periods=1)
    Requiere group_df con columnas en group_keys (alineadas 1:1 con series).
    """
    # Validaciones mínimas
    missing = [k for k in group_keys if k not in group_df.columns]
    if missing:
        raise KeyError(f"_smooth_grouped: faltan columnas en group_df: {missing}")
    if len(series) != len(dates) or len(series) != len(group_df):
        raise ValueError("_smooth_grouped: series, dates y group_df deben tener la misma longitud.")

    # Construcción del DF con posición original para reordenar luego
    df = pd.DataFrame({"val": series.to_numpy(), "date": pd.to_datetime(dates).to_numpy()})
    for k in group_keys:
        df[k] = group_df[k].to_numpy()
    df = df.reset_index().rename(columns={"index": "_orig_idx"})
    df = df.sort_values(group_keys + ["date"], kind="mergesort")  # estable

    g = df.groupby(group_keys, sort=False, group_keys=False)

    if kind.lower() == "ma":
        sm = g.apply(
            lambda d: d.set_index("date")["val"]
                      .rolling(window=int(max(1, ma_window)), min_periods=1)
                      .mean()
                      .reset_index(drop=True)
        ).reset_index(drop=True).to_numpy()
    else:
        a = float(np.clip(ema_alpha_val, 1e-6, 1.0))
        sm = g["val"].apply(lambda s: s.ewm(alpha=a, adjust=False).mean()).reset_index(drop=True).to_numpy()

    # Reconstruir serie en el orden original
    out = pd.Series(index=df["_orig_idx"].to_numpy(), data=sm, dtype="float32")
    return out.sort_index()


def _winsorize_by(df: pd.DataFrame,
                  value_col: str,
                  by: str = "store_neighborhood",
                  low: float = 0.005,
                  high: float = 0.995,
                  min_obs: int = 30,
                  store_col: str = "store_nbr",
                  item_col: str = "item_nbr",
                  neighborhood_col: str = "class") -> pd.Series:
    """
    Winsoriza columna 'value_col' según agrupamiento:
      - by='global'              -> una sola pareja (q_low, q_high).
      - by='store_item'          -> por (store, item).
      - by='store_neighborhood'  -> por (store, neighborhood).
    Si el grupo tiene menos de 'min_obs' o cuantiles NaN, cae a quantiles globales.
    """
    x = df[value_col].astype("float64").to_numpy(copy=False)
    x_clip = x.copy()

    # Cuantiles globales robustos
    if np.all(np.isnan(x)):
        ql_g, qh_g = 0.0, 1.0
    else:
        ql_g, qh_g = np.nanquantile(x, [low, high])

    if by == "global":
        np.clip(x_clip, ql_g, qh_g, out=x_clip)
        return pd.Series(x_clip, index=df.index, dtype="float32")

    if by == "store_item":
        keys = [store_col, item_col]
    else:  # default
        keys = [store_col, neighborhood_col]

    grp = df.groupby(keys, observed=True, sort=False)
    counts = grp[value_col].size().rename("n").reset_index()

    q = grp[value_col].quantile([low, high]).unstack(level=-1)
    if q is None or q.empty:
        # sin cuantiles por grupo -> global
        np.clip(x_clip, ql_g, qh_g, out=x_clip)
        return pd.Series(x_clip, index=df.index, dtype="float32")

    q.columns = ["q_low", "q_high"]
    # merge por índice derecho (MultiIndex=keys)
    stats = counts.merge(q, left_on=keys, right_index=True, how="left")

    # fallback: si grupo pequeño o cuantiles NaN -> usar globales
    stats["q_low"] = np.where((stats["n"] >= int(min_obs)) & (~stats["q_low"].isna()), stats["q_low"], ql_g)
    stats["q_high"] = np.where((stats["n"] >= int(min_obs)) & (~stats["q_high"].isna()), stats["q_high"], qh_g)

    df_stats = df[keys].merge(stats[keys + ["q_low", "q_high"]], on=keys, how="left")
    np.clip(x_clip, df_stats["q_low"].to_numpy(), df_stats["q_high"].to_numpy(), out=x_clip)
    return pd.Series(x_clip, index=df.index, dtype="float32")


# -------------------- API principal -------------------- #

def compute_competitive_exposure(
    train_path: str,
    items_path: Optional[str] = None,
    # nombres de columnas
    date_col: str = "date",
    store_col: str = "store_nbr",
    item_col: str = "item_nbr",
    promo_col: str = "onpromotion",
    neighborhood_col: str = "class",
    # parámetros de tratamiento
    bin_threshold: float = 0.02,      # UMBRAL para H_bin (desde la serie CRUDA)
    include_self: bool = False,       # si True, no excluye al propio ítem (no recomendado)
    # --- PREPROCESAMIENTO (nuevo) ---
    preprocess: bool = True,          # suavizado causal + winsor activado por defecto
    smooth_kind: str = "ema",         # 'ema' | 'ma'
    ma_window: int = 7,               # si kind='ma'
    ema_span: float = 7.0,            # si kind='ema' (prioridad: alpha > halflife > span)
    ema_halflife: Optional[float] = None,
    ema_alpha: Optional[float] = None,
    winsor_low: float = 0.005,        # cuantiles de winsor
    winsor_high: float = 0.995,
    winsor_by: str = "store_neighborhood",  # 'global'|'store_item'|'store_neighborhood'
    winsor_min_obs: int = 30,
    # salida
    save_path: Optional[str] = None,
    save_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calcula exposición competitiva excluyendo el ítem propio y, opcionalmente,
    aplica *suavizado causal + winsor*.

    Devuelve columnas:
      - date, store_nbr, item_nbr
      - H_prop       (suavizada+winsorizada si preprocess=True; cruda si False)
      - H_prop_raw   (cruda, siempre)
      - H_bin        (derivada de H_prop_raw)
      - H_disc       (conteo de vecinos en promo, crudo)
      - competitive_exposure (alias de H_prop)
      - neighborhood_col si no estaba y se tomó de items.csv
    """
    # Lectura
    train = pd.read_csv(
        train_path,
        parse_dates=[date_col],
        dtype={store_col: "int16", item_col: "int32"},
        low_memory=False
    )

    # Normalizar onpromotion a {0,1}
    train[promo_col] = _coerce_bool01(train[promo_col])

    # neighborhood: merge opcional si no existe en train
    merged_neighborhood = False
    if neighborhood_col not in train.columns:
        if items_path is None:
            raise ValueError(
                f"No se encontró '{neighborhood_col}' en train y no se proporcionó items_path."
            )
        items = pd.read_csv(items_path, usecols=[item_col, neighborhood_col], dtype={item_col: "int32"})
        train = train.merge(items, on=item_col, how="left", validate="many_to_one")
        merged_neighborhood = True

    # Claves de vecindario para el cociente
    grp_cols = [date_col, store_col, neighborhood_col]

    # Categorizar neighborhood para acelerar/memoria
    if not pd.api.types.is_categorical_dtype(train[neighborhood_col]):
        train[neighborhood_col] = train[neighborhood_col].astype("category")

    # Tamaños y conteos por grupo (observed=True evita niveles no usados)
    n_total = train.groupby(grp_cols, observed=True)[item_col].transform("size")
    n_promos = train.groupby(grp_cols, observed=True)[promo_col].transform("sum")

    self_promo = train[promo_col].astype("int16")
    if include_self:
        numer = n_promos
        denom = n_total
    else:
        numer = n_promos - self_promo  # descuenta al propio ítem
        denom = n_total - 1

    # Proporción (otros en promo / otros existentes) y conteo
    H_prop_raw = np.where(denom > 0, numer / denom, np.nan)
    H_prop_raw = np.clip(H_prop_raw, 0.0, 1.0).astype("float32")
    H_disc = np.clip(numer, 0, None).astype("int32")
    H_bin = (H_prop_raw >= float(bin_threshold)).astype("int8")

    out = train[[date_col, store_col, item_col]].copy()
    out["H_prop_raw"] = H_prop_raw
    out["H_disc"] = H_disc
    out["H_bin"] = H_bin

    # --------- PREPROCESAMIENTO: suavizado causal + winsor --------- #
    if preprocess:
        # alpha para EMA (si aplica)
        alpha_val = _ema_alpha(span=ema_span, halflife=ema_halflife, alpha=ema_alpha)

        # suavizado *causal* por (store,item) sobre la serie CRUDA
        out["H_prop_smooth"] = _smooth_grouped(
            series=out["H_prop_raw"],
            group_keys=[store_col, item_col],
            dates=out[date_col],              # usar out para asegurar alineación
            group_df=out[[store_col, item_col]],
            kind=smooth_kind,
            ma_window=ma_window,
            ema_alpha_val=alpha_val
        ).clip(0.0, 1.0)

        # winsor por grupo (por defecto store+neighborhood) sobre la serie suavizada
        tmp = out[[store_col, item_col, "H_prop_smooth"]].copy()
        # adjuntar neighborhood (viene de train, alineado)
        tmp[neighborhood_col] = train[neighborhood_col].values

        out["H_prop"] = _winsorize_by(
            df=tmp,
            value_col="H_prop_smooth",
            by=winsor_by,
            low=float(winsor_low),
            high=float(winsor_high),
            min_obs=int(winsor_min_obs),
            store_col=store_col,
            item_col=item_col,
            neighborhood_col=neighborhood_col
        ).clip(0.0, 1.0).astype("float32")

        out.drop(columns=["H_prop_smooth"], inplace=True)
    else:
        # sin preprocesar, H_prop = cruda
        out["H_prop"] = out["H_prop_raw"].astype("float32")

    # Alias para compatibilidad con otros módulos/plots
    out["competitive_exposure"] = out["H_prop"].astype("float32")

    # Adjuntar vecindario si no estaba en train y lo trajimos desde items
    if merged_neighborhood:
        out[neighborhood_col] = train[neighborhood_col].values

    # Orden final y dtypes compactos
    cols = [date_col, store_col, item_col, "H_prop", "H_bin", "H_disc", "H_prop_raw", "competitive_exposure"]
    if merged_neighborhood:
        cols.append(neighborhood_col)
    out = out[cols]
    out[store_col] = out[store_col].astype("int16")
    out[item_col] = out[item_col].astype("int32")

    # Guardar
    if save_path:
        print(f"Guardando competitive exposure a {save_path}")
        save_path = str(Path(save_path))
        ext = (save_format or Path(save_path).suffix.lower().replace(".", "")).lower()
        if ext in ("parquet", "pq"):
            out.to_parquet(save_path, index=False)
        else:
            out.to_csv(save_path, index=False)

    return out


# ---------------------------- CLI ---------------------------- #

def _parse_args():
    ap = argparse.ArgumentParser(description="Competitive exposure con suavizado causal + winsor.")
    ap.add_argument("--train", required=True, help="Ruta a train_filtered.csv")
    ap.add_argument("--items", default=None, help="Ruta a items.csv (opcional)")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--store-col", default="store_nbr")
    ap.add_argument("--item-col", default="item_nbr")
    ap.add_argument("--promo-col", default="onpromotion")
    ap.add_argument("--neighborhood-col", default="class", help="Familia/categoría/clase")

    ap.add_argument("--bin-threshold", type=float, default=0.02)
    ap.add_argument("--include-self", action="store_true")

    # preprocesamiento
    ap.add_argument("--preprocess", dest="preprocess", action="store_true")
    ap.add_argument("--no-preprocess", dest="preprocess", action="store_false")
    ap.set_defaults(preprocess=True)

    ap.add_argument("--smooth-kind", choices=["ema", "ma"], default="ema")
    ap.add_argument("--ma-window", type=int, default=7)
    ap.add_argument("--ema-span", type=float, default=7.0)
    ap.add_argument("--ema-halflife", type=float, default=None)
    ap.add_argument("--ema-alpha", type=float, default=None)

    ap.add_argument("--winsor-low", type=float, default=0.005)
    ap.add_argument("--winsor-high", type=float, default=0.995)
    ap.add_argument("--winsor-by", choices=["global", "store_item", "store_neighborhood"], default="store_neighborhood")
    ap.add_argument("--winsor-min-obs", type=int, default=30)

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
        bin_threshold=args.bin_threshold,
        include_self=args.include_self,
        preprocess=args.preprocess,
        smooth_kind=args.smooth_kind,
        ma_window=args.ma_window,
        ema_span=args.ema_span,
        ema_halflife=args.ema_halflife,
        ema_alpha=args.ema_alpha,
        winsor_low=args.winsor_low,
        winsor_high=args.winsor_high,
        winsor_by=args.winsor_by,
        winsor_min_obs=args.winsor_min_obs,
        save_path=args.save,
        save_format=args.save_format,
    )