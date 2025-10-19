
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_episodes.py — Reporte diagnóstico de calidad de episodios (i,j)

Uso:
  python diagnose_episodes.py \
    --pairs pairs_windows.csv \
    --train train.csv(.gz) \
    --H features_h_exposure.csv(.gz) \
    --outdir diagnostics \
    [--items items.csv] \
    [--pre_gap 7] \
    [--min_treat_on_i 0.7] \
    [--max_pre_on_i 0.2] \
    [--max_treat_on_j 0.2] \
    [--min_cov 0.8] \
    [--min_d_hj 0.3] \
    [--min_d_ex 0.2] \
    [--date_pad_days 5]

Qué hace:
- Para cada fila de pairs_windows.csv (i_store,i_item,class,j_store,j_item,pre_start,treat_start,post_start,post_end)
  calcula métricas de calidad usando train.csv (onpromotion) y H (H_prop, exposición competitiva).
- Valida que:
  * i esté encendido (onpromotion) en trat. (alta tasa) y apagado en pre (baja tasa)
  * j esté apagado en trat. (baja tasa) — opcional
  * H_j suba en trat. vs pre (Δ y tamaño de efecto d de Cohen)
  * La media de clase excluyendo i (ex_mean) suba en trat. vs pre (Δ y d)
  * Tendencias pre (slopes) pequeñas (sin tendencia previa marcada)
  * Coberturas suficientes en H y train
- Emite:
  * diagnostics/episode_diagnostics.csv — métricas por episodio
  * diagnostics/episode_quality_summary.csv — cuantiles y tasas de pase
  * diagnostics/_debug_shapes.json — tamaños intermedios

Requisitos:
- Python 3.8+
- pandas, numpy
"""

import os
import json
import math
import argparse
import gzip
import warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

def _normalize_onpromotion(series: pd.Series) -> pd.Series:
    if series.dtype == bool: return series.astype("int8")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).clip(0,1).astype("int8")
    return series.fillna("False").astype(str).str.lower().isin(["true","1","t","yes"]).astype("int8")

def _parse_dates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_datetime(df[c]).dt.normalize()
    return df

def _date_range_days(a: pd.Timestamp, b: pd.Timestamp) -> List[pd.Timestamp]:
    if a > b: return []
    return list(pd.date_range(a, b, freq="D"))

def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    # pooled std
    sp2 = ((nx - 1) * vx + (ny - 1) * vy) / max(1, (nx + ny - 2))
    if sp2 <= 0 or not np.isfinite(sp2): return np.nan
    return (mx - my) / math.sqrt(sp2)

def _slope_r2(y: np.ndarray) -> Tuple[float, float]:
    """Devuelve (slope_por_día, r2) para una serie y con pasos diarios uniformes."""
    y = y.astype(float)
    mask = np.isfinite(y)
    if mask.sum() < 3: return (np.nan, np.nan)
    y = y[mask]
    x = np.arange(len(y), dtype=float)
    x_mean = x.mean(); y_mean = y.mean()
    cov_xy = ((x - x_mean) * (y - y_mean)).sum()
    var_x  = ((x - x_mean) ** 2).sum()
    if var_x == 0: return (np.nan, np.nan)
    slope = cov_xy / var_x
    # r2 via correlación lineal al cuadrado
    var_y  = ((y - y_mean) ** 2).sum()
    r2 = 0.0 if var_y == 0 else (cov_xy ** 2) / (var_x * var_y)
    return (float(slope), float(r2))

def _read_pairs(path: str) -> pd.DataFrame:
    use = ["i_store","i_item","class","j_store","j_item","delta_H_j","n_obs_i_on","n_obs_i_off",
           "pre_start","treat_start","post_start","post_end"]
    df = pd.read_csv(path, usecols=use)
    df = _parse_dates(df, ["pre_start","treat_start","post_start","post_end"])
    # Inferir treat_end = post_start - 1; pre_end = treat_start - pre_gap - 1
    return df

def _maybe_add_class_to_H(H_csv: str, items_csv: Optional[str], chunksize: int = 2_000_000) -> str:
    """Si H no trae 'class', la añade via items.csv y escribe temporal al lado."""
    cols = pd.read_csv(H_csv, nrows=0).columns.tolist()
    if "class" in cols:
        return H_csv
    if not items_csv:
        raise ValueError("H no contiene columna 'class'. Provee --items items.csv para poder añadirla.")
    tmp = os.path.splitext(H_csv)[0] + "_with_class.tmp.csv.gz"
    if os.path.exists(tmp): os.remove(tmp)
    items = pd.read_csv(items_csv, usecols=["item_nbr","class"]).dropna()
    items["item_nbr"] = pd.to_numeric(items["item_nbr"], errors="coerce").astype("int64")
    items["class"] = pd.to_numeric(items["class"], errors="coerce").astype("int64")
    header = True
    usecols = [c for c in ["date","store_nbr","item_nbr","H_prop","H_bin"] if c in cols]
    for ch in pd.read_csv(H_csv, usecols=usecols, parse_dates=["date"], chunksize=chunksize):
        ch["item_nbr"] = pd.to_numeric(ch["item_nbr"], errors="coerce").astype("int64")
        ch = ch.merge(items, on="item_nbr", how="left")
        import gzip
        with gzip.open(tmp, "ab") as gz:
            ch.to_csv(gz, index=False, header=header)
        header = False
    return tmp

def _read_train_subset(train_csv: str,
                       stores: List[int],
                       items_needed: List[int],
                       date_min: pd.Timestamp,
                       date_max: pd.Timestamp,
                       chunksize: int = 2_000_000) -> pd.DataFrame:
    dtypes={"store_nbr":"int64","item_nbr":"int64"}
    use = ["date","store_nbr","item_nbr","onpromotion"]
    parts=[]
    store_set = set(int(x) for x in stores)
    item_set = set(int(x) for x in items_needed)
    for ch in pd.read_csv(train_csv, usecols=use, parse_dates=["date"], dtype=dtypes, chunksize=chunksize, low_memory=False):
        ch = ch[(ch["store_nbr"].isin(store_set)) & (ch["item_nbr"].isin(item_set))]
        if ch.empty: continue
        ch = ch[(ch["date"]>=date_min) & (ch["date"]<=date_max)]
        ch["onpromotion"] = _normalize_onpromotion(ch["onpromotion"])
        parts.append(ch[["date","store_nbr","item_nbr","onpromotion"]])
    if not parts:
        return pd.DataFrame(columns=["date","store_nbr","item_nbr","onpromotion"])
    df = pd.concat(parts, ignore_index=True)
    # Deduplicar por fecha si hay repetidos
    df = df.sort_values(["store_nbr","item_nbr","date"]).drop_duplicates(["store_nbr","item_nbr","date"], keep="last")
    return df

def _read_H_subset(H_csv: str,
                   stores_classes: List[Tuple[int,int]],
                   items_needed: List[int],
                   date_min: pd.Timestamp,
                   date_max: pd.Timestamp,
                   chunksize: int = 2_000_000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - H_sc: agregados por (store_nbr,class,date): S_sum (suma H_prop), C_cnt (recuento no-NA)
      - H_items: series por (store_nbr,item_nbr,date) solo para items_needed (i y j)
    """
    dtypes = {"store_nbr":"int64","item_nbr":"int64","class":"int64","H_bin":"int8"}
    use = ["date","store_nbr","item_nbr","class","H_prop","H_bin"]
    sc_set = set((int(s), int(c)) for s,c in stores_classes)
    item_set = set(int(x) for x in items_needed)
    parts_sc = []
    parts_it = []
    for ch in pd.read_csv(H_csv, usecols=use, parse_dates=["date"], dtype=dtypes, chunksize=chunksize, low_memory=False):
        ch = ch[(ch["date"]>=date_min) & (ch["date"]<=date_max)]
        if ch.empty: continue
        ch["key_sc"] = list(zip(ch["store_nbr"].astype(int), ch["class"].astype(int)))
        keep_sc = ch["key_sc"].isin(sc_set)
        if keep_sc.any():
            sc = ch.loc[keep_sc, ["date","store_nbr","class","H_prop"]].copy()
            parts_sc.append(sc)
        keep_it = ch["item_nbr"].isin(item_set)
        if keep_it.any():
            it = ch.loc[keep_it, ["date","store_nbr","item_nbr","class","H_prop"]].copy()
            parts_it.append(it)
    if parts_sc:
        H_sc = pd.concat(parts_sc, ignore_index=True)
        agg = (H_sc.groupby(["store_nbr","class","date"], as_index=False)
                 .agg(S_sum=("H_prop","sum"), C_cnt=("H_prop", lambda s: np.isfinite(s).sum())))
    else:
        agg = pd.DataFrame(columns=["store_nbr","class","date","S_sum","C_cnt"])
    if parts_it:
        H_items = (pd.concat(parts_it, ignore_index=True)
                     .sort_values(["store_nbr","item_nbr","date"])
                     .drop_duplicates(["store_nbr","item_nbr","date"], keep="last"))
    else:
        H_items = pd.DataFrame(columns=["date","store_nbr","item_nbr","class","H_prop"])
    return agg, H_items

def _window_masks(dates: pd.DatetimeIndex,
                  pre_start: pd.Timestamp,
                  treat_start: pd.Timestamp,
                  post_start: pd.Timestamp,
                  post_end: pd.Timestamp,
                  pre_gap: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve máscaras booleanas para pre, treat, post sobre un índice de fechas (diario).
    pre_end = treat_start - pre_gap - 1 día
    treat_end = post_start - 1 día
    """
    treat_end = (post_start - pd.Timedelta(days=1)).normalize()
    pre_end   = (treat_start - pd.Timedelta(days=pre_gap) - pd.Timedelta(days=1)).normalize()
    pre_mask   = (dates>=pre_start) & (dates<=pre_end)
    treat_mask = (dates>=treat_start) & (dates<=treat_end)
    post_mask  = (dates>=post_start) & (dates<=post_end)
    # Devuelve siempre ndarrays compatibles con numpy/pandas
    return np.asarray(pre_mask), np.asarray(treat_mask), np.asarray(post_mask)

def _series_from_df(df: pd.DataFrame, key_cols: List[str], idx_col: str, val_col: str) -> Dict[Tuple, pd.Series]:
    res = {}
    if df.empty: return res
    for keys, g in df.groupby(key_cols):
        s = g.set_index(idx_col)[val_col].sort_index()
        res[tuple(keys)] = s
    return res

def _align_series_into(dates: pd.DatetimeIndex, s: Optional[pd.Series]) -> np.ndarray:
    if s is None or s.empty:
        return np.full(len(dates), np.nan, dtype=float)
    # reindex sin rellenar NaN (H puede no existir ciertos días)
    out = s.reindex(dates)
    return out.to_numpy(dtype=float)

def _rate(x: np.ndarray, mask: np.ndarray) -> float:
    sel = x[mask]
    sel = sel[np.isfinite(sel)]
    if sel.size == 0: return np.nan
    return float(np.nanmean(sel))

def _prop(x: np.ndarray, mask: np.ndarray, cond=lambda v: v>0.5) -> float:
    sel = x[mask]
    sel = sel[np.isfinite(sel)]
    if sel.size == 0: return np.nan
    return float(np.mean(cond(sel)))

def _coverage(arr: np.ndarray, mask: np.ndarray) -> float:
    sel = arr[mask]
    return float(np.isfinite(sel).mean()) if sel.size>0 else np.nan

def _mean(arr: np.ndarray, mask: np.ndarray) -> float:
    sel = arr[mask]
    sel = sel[np.isfinite(sel)]
    return float(np.nanmean(sel)) if sel.size>0 else np.nan

def _diff_means(a: np.ndarray, mask_a: np.ndarray, b: np.ndarray, mask_b: np.ndarray) -> float:
    A = a[mask_a]; B = b[mask_b]
    A = A[np.isfinite(A)]; B = B[np.isfinite(B)]
    if A.size==0 or B.size==0: return np.nan
    return float(np.nanmean(A) - np.nanmean(B))

def _cohen_d_masks(a: np.ndarray, mask_a: np.ndarray, b: np.ndarray, mask_b: np.ndarray) -> float:
    A = a[mask_a]; B = b[mask_b]
    A = A[np.isfinite(A)]; B = B[np.isfinite(B)]
    if A.size<2 or B.size<2: return np.nan
    return _cohen_d(A, B)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--H", required=True)
    ap.add_argument("--outdir", default="diagnostics")
    ap.add_argument("--items", default=None, help="Opcional: items.csv (necesario si H no trae 'class')")

    ap.add_argument("--pre_gap", type=int, default=int(os.environ.get("DIAG_PRE_GAP", 7)))

    # Umbrales de calidad (modificables)
    ap.add_argument("--min_treat_on_i", type=float, default=float(os.environ.get("DIAG_MIN_TREAT_ON_I", 0.7)))
    ap.add_argument("--max_pre_on_i",   type=float, default=float(os.environ.get("DIAG_MAX_PRE_ON_I", 0.2)))
    ap.add_argument("--max_treat_on_j", type=float, default=float(os.environ.get("DIAG_MAX_TREAT_ON_J", 0.2)))
    ap.add_argument("--min_cov",        type=float, default=float(os.environ.get("DIAG_MIN_COV", 0.8)))
    ap.add_argument("--min_d_hj",       type=float, default=float(os.environ.get("DIAG_MIN_D_HJ", 0.3)))
    ap.add_argument("--min_d_ex",       type=float, default=float(os.environ.get("DIAG_MIN_D_EX", 0.2)))
    ap.add_argument("--date_pad_days",  type=int, default=int(os.environ.get("DIAG_DATE_PAD_DAYS", 5)))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pairs = _read_pairs(args.pairs)
    if pairs.empty:
        raise SystemExit("pairs_windows.csv está vacío. No hay episodios que diagnosticar.")

    # Calcular fechas min/max globales para limitar lecturas
    pre_gap = int(args.pre_gap)
    pairs = pairs.copy()
    pairs["treat_end"] = (pairs["post_start"] - pd.Timedelta(days=1)).dt.normalize()
    pairs["pre_end"]   = (pairs["treat_start"] - pd.Timedelta(days=pre_gap) - pd.Timedelta(days=1)).dt.normalize()

    # Padding de días para lecturas (bordes)
    pad = pd.Timedelta(days=int(args.date_pad_days))
    global_min = (pairs[["pre_start","treat_start","post_start"]].min().min() - pad).normalize()
    global_max = (pairs[["treat_end","post_end"]].max().max() + pad).normalize()

    # Conjuntos necesarios
    stores = sorted(set(pairs["i_store"].astype(int).tolist() + pairs["j_store"].astype(int).tolist()))
    classes = sorted(pairs["class"].astype(int).unique().tolist())
    items_i = sorted(pairs["i_item"].astype(int).unique().tolist())
    items_j = sorted(pairs["j_item"].astype(int).unique().tolist())
    items_needed = sorted(set(items_i) | set(items_j))
    sc_keys = sorted(set(zip(pairs["i_store"].astype(int), pairs["class"].astype(int))))  # (store,class)

    # Asegurar que H tenga 'class'
    H_use = _maybe_add_class_to_H(args.H, args.items)

    # Leer subsets
    print("[INFO] Leyendo train subset ...")
    train_sub = _read_train_subset(args.train, stores, items_needed, global_min, global_max)
    print(f"[INFO] train subset: {len(train_sub):,} filas")

    print("[INFO] Leyendo H subset ...")
    H_sc, H_items = _read_H_subset(H_use, sc_keys, items_needed, global_min, global_max)
    print(f"[INFO] H_sc (sumas por (store,class,date)): {len(H_sc):,} filas")
    print(f"[INFO] H_items (series de i y j): {len(H_items):,} filas")

    # Indexar series
    on_map  = _series_from_df(train_sub, ["store_nbr","item_nbr"], "date", "onpromotion")
    Hj_map  = _series_from_df(H_items, ["store_nbr","item_nbr"], "date", "H_prop")
    # Agregados de clase por (store,class)
    ex_map = {}  # (store,class) -> DataFrame date,S_sum,C_cnt
    if not H_sc.empty:
        ex_map = { (int(s),int(c)): g.sort_values("date").reset_index(drop=True)
                   for (s,c), g in H_sc.groupby(["store_nbr","class"], sort=False) }

    # Calendario base por (store,class) para reindexar cómodo
    cal_map = {}
    for (s,c), g in ex_map.items():
        d_min = g["date"].min(); d_max = g["date"].max()
        idx = pd.date_range(d_min, d_max, freq="D")
        cal_map[(int(s),int(c))] = idx

    # Diagnóstico por episodio
    rows = []
    for ridx, r in pairs.iterrows():
        s_i = int(r["i_store"]); i_item = int(r["i_item"]); cls = int(r["class"])
        s_j = int(r["j_store"]); j_item = int(r["j_item"])
        pre_start = pd.to_datetime(r["pre_start"]).normalize()
        treat_start = pd.to_datetime(r["treat_start"]).normalize()
        post_start = pd.to_datetime(r["post_start"]).normalize()
        post_end   = pd.to_datetime(r["post_end"]).normalize()
        treat_end  = pd.to_datetime(r["treat_end"]).normalize()
        pre_end    = pd.to_datetime(r["pre_end"]).normalize()

        # Calendario (usar el de la clase si existe; si no, construir con [pre_start, post_end])
        idx = cal_map.get((s_i, cls), pd.date_range(pre_start, post_end, freq="D"))
        pre_mask, treat_mask, post_mask = _window_masks(idx, pre_start, treat_start, post_start, post_end, pre_gap)

        # Series de onpromotion
        on_i = on_map.get((s_i, i_item), pd.Series(dtype=float))
        on_j = on_map.get((s_j, j_item), pd.Series(dtype=float))
        on_i_arr = _align_series_into(idx, on_i)
        on_j_arr = _align_series_into(idx, on_j)

        # Series de H
        Hj = Hj_map.get((s_j, j_item), pd.Series(dtype=float))
        Hj_arr = _align_series_into(idx, Hj)

        # ex_mean = (S_sum - H_i)/(C_cnt - 1)
        ex_df = ex_map.get((s_i, cls), pd.DataFrame(columns=["date","S_sum","C_cnt"]))
        if not ex_df.empty:
            ex_df = ex_df.set_index("date").reindex(idx)
            S_sum = ex_df["S_sum"].to_numpy(dtype=float)
            C_cnt = ex_df["C_cnt"].to_numpy(dtype=float)
        else:
            S_sum = np.full(len(idx), np.nan, dtype=float)
            C_cnt = np.full(len(idx), np.nan, dtype=float)

        Hi = Hj_map.get((s_i, i_item), pd.Series(dtype=float))
        Hi_arr = _align_series_into(idx, Hi)

        with np.errstate(invalid="ignore", divide="ignore"):
            denom = (C_cnt - 1).astype(float)
            ex_mean_arr = np.where(denom>0, (S_sum - Hi_arr) / denom, np.nan)

        # Métricas de cobertura
        cov_Hj_pre    = _coverage(Hj_arr, pre_mask)
        cov_Hj_treat  = _coverage(Hj_arr, treat_mask)
        cov_ex_pre    = _coverage(ex_mean_arr, pre_mask)
        cov_ex_treat  = _coverage(ex_mean_arr, treat_mask)
        cov_on_i_pre  = _coverage(on_i_arr, pre_mask)
        cov_on_i_treat= _coverage(on_i_arr, treat_mask)
        cov_on_j_pre  = _coverage(on_j_arr, pre_mask)
        cov_on_j_treat= _coverage(on_j_arr, treat_mask)

        # Tasas ON/OFF
        rate_i_treat = _mean(on_i_arr, treat_mask)
        rate_i_pre   = _mean(on_i_arr, pre_mask)
        rate_j_treat = _mean(on_j_arr, treat_mask)
        rate_j_pre   = _mean(on_j_arr, pre_mask)

        # Días "tratamiento limpio": i ON & j OFF & datos Hj presentes
        clean_mask = treat_mask & (on_i_arr>=0.5) & ((on_j_arr<0.5) | ~np.isfinite(on_j_arr)) & np.isfinite(Hj_arr)
        n_treat_days     = int(treat_mask.sum())
        n_clean_treat    = int(clean_mask.sum())
        clean_share      = (n_clean_treat / n_treat_days) if n_treat_days>0 else np.nan

        # H_j: diferencias y tamaños de efecto
        delta_Hj_treat_pre = _diff_means(Hj_arr, treat_mask, Hj_arr, pre_mask)
        d_Hj_treat_pre     = _cohen_d_masks(Hj_arr, treat_mask, Hj_arr, pre_mask)

        # ex_mean: diferencias y tamaños de efecto
        delta_ex_treat_pre = _diff_means(ex_mean_arr, treat_mask, ex_mean_arr, pre_mask)
        d_ex_treat_pre     = _cohen_d_masks(ex_mean_arr, treat_mask, ex_mean_arr, pre_mask)

        # Slopes y r2 en PRE (estabilidad)
        Hj_pre_vals = Hj_arr[pre_mask]
        ex_pre_vals = ex_mean_arr[pre_mask]
        slope_Hj_pre, r2_Hj_pre = _slope_r2(Hj_pre_vals)
        slope_ex_pre, r2_ex_pre = _slope_r2(ex_pre_vals)

        # Promedios
        Hj_pre_mean   = _mean(Hj_arr, pre_mask)
        Hj_treat_mean = _mean(Hj_arr, treat_mask)
        ex_pre_mean   = _mean(ex_mean_arr, pre_mask)
        ex_treat_mean = _mean(ex_mean_arr, treat_mask)

        # Reglas de pase
        reasons = []
        if not (cov_Hj_pre>=args.min_cov and cov_Hj_treat>=args.min_cov):
            reasons.append("cov_Hj_baja")
        if not (cov_ex_pre>=args.min_cov and cov_ex_treat>=args.min_cov):
            reasons.append("cov_ex_baja")
        if not (cov_on_i_treat>=args.min_cov and cov_on_i_pre>=0.5):  # pre puede tener huecos, menos exigente
            reasons.append("cov_on_i_baja")
        if math.isnan(rate_i_treat) or rate_i_treat < args.min_treat_on_i:
            reasons.append("i_poco_ON_en_treat")
        if not math.isnan(rate_i_pre) and rate_i_pre > args.max_pre_on_i:
            reasons.append("i_ON_en_pre")
        if not math.isnan(rate_j_treat) and rate_j_treat > args.max_treat_on_j:
            reasons.append("j_ON_en_treat")
        if math.isnan(d_Hj_treat_pre) or d_Hj_treat_pre < args.min_d_hj:
            reasons.append("efecto_Hj_pequeño")
        if math.isnan(d_ex_treat_pre) or d_ex_treat_pre < args.min_d_ex:
            reasons.append("efecto_ex_pequeño")
        if not (r2_Hj_pre <= 0.3 or np.isnan(r2_Hj_pre)):  # tendencia pre fuerte
            reasons.append("tendencia_pre_Hj")
        if not (r2_ex_pre <= 0.3 or np.isnan(r2_ex_pre)):
            reasons.append("tendencia_pre_ex")

        passed = (len(reasons) == 0)

        rows.append({
            "i_store": s_i, "i_item": i_item, "class": cls,
            "j_store": s_j, "j_item": j_item,
            "pre_start": pre_start.date(), "pre_end": pre_end.date(),
            "treat_start": treat_start.date(), "treat_end": treat_end.date(),
            "post_start": post_start.date(), "post_end": post_end.date(),
            "planned_pre_days": int((pre_end - pre_start).days + 1),
            "planned_treat_days": int((treat_end - treat_start).days + 1),
            "planned_post_days": int((post_end - post_start).days + 1),
            "cov_Hj_pre": cov_Hj_pre, "cov_Hj_treat": cov_Hj_treat,
            "cov_ex_pre": cov_ex_pre, "cov_ex_treat": cov_ex_treat,
            "cov_on_i_pre": cov_on_i_pre, "cov_on_i_treat": cov_on_i_treat,
            "cov_on_j_pre": cov_on_j_pre, "cov_on_j_treat": cov_on_j_treat,
            "rate_i_pre": rate_i_pre, "rate_i_treat": rate_i_treat,
            "rate_j_pre": rate_j_pre, "rate_j_treat": rate_j_treat,
            "n_treat_days": n_treat_days, "n_clean_treat": n_clean_treat, "clean_share": clean_share,
            "Hj_pre_mean": Hj_pre_mean, "Hj_treat_mean": Hj_treat_mean,
            "delta_Hj_treat_pre": delta_Hj_treat_pre, "d_Hj_treat_pre": d_Hj_treat_pre,
            "ex_pre_mean": ex_pre_mean, "ex_treat_mean": ex_treat_mean,
            "delta_ex_treat_pre": delta_ex_treat_pre, "d_ex_treat_pre": d_ex_treat_pre,
            "slope_Hj_pre_per_day": slope_Hj_pre, "r2_Hj_pre": r2_Hj_pre,
            "slope_ex_pre_per_day": slope_ex_pre, "r2_ex_pre": r2_ex_pre,
            "passed": passed, "fail_reasons": ";".join(reasons)
        })

    diag = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "episode_diagnostics.csv")
    diag.to_csv(out_csv, index=False)

    # Resumen de cuantiles y tasas de pase
    summary_rows = []
    numeric_cols = [
        "cov_Hj_pre","cov_Hj_treat","cov_ex_pre","cov_ex_treat","cov_on_i_pre","cov_on_i_treat","cov_on_j_pre","cov_on_j_treat",
        "rate_i_pre","rate_i_treat","rate_j_pre","rate_j_treat",
        "clean_share","delta_Hj_treat_pre","d_Hj_treat_pre","delta_ex_treat_pre","d_ex_treat_pre",
        "slope_Hj_pre_per_day","r2_Hj_pre","slope_ex_pre_per_day","r2_ex_pre"
    ]
    quantiles = [0.01,0.05,0.10,0.25,0.5,0.75,0.90,0.95,0.99]
    for col in numeric_cols:
        if col in diag.columns and pd.api.types.is_numeric_dtype(diag[col]):
            q = diag[col].quantile(quantiles).to_dict()
            row = {"metric": col}; row.update({f"q{int(k*100):02d}": float(v) for k,v in q.items()})
            summary_rows.append(row)
    summary_rows.append({"metric": "pass_rate", "q50": float(diag["passed"].mean() if len(diag)>0 else 0.0)})
    summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.outdir, "episode_quality_summary.csv")
    summary.to_csv(summary_csv, index=False)

    # Debug shapes
    debug = {
        "pairs_rows": int(len(pairs)),
        "train_rows": int(len(train_sub)),
        "H_sc_rows": int(len(H_sc)),
        "H_items_rows": int(len(H_items)),
        "episodes_passed": int(diag["passed"].sum()) if not diag.empty else 0
    }
    with open(os.path.join(args.outdir, "_debug_shapes.json"), "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)

    print("[OK] Reporte diagnóstico generado:")
    print(" -", out_csv)
    print(" -", summary_csv)
    print(" -", os.path.join(args.outdir, "_debug_shapes.json"))

if __name__ == "__main__":
    main()
