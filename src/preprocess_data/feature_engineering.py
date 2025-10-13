# -*- coding: utf-8 -*-
"""
feature_engineering.py (optimizado RAM)

Propósito (sin cambios):
- Construir un panel diario (date, store_nbr, item_nbr) con outcome transformado,
  confounders, exposición competitiva H_{jst}, episodios para GSC y particionado A/B/C.

Entradas esperadas en `raw` (sin cambios).
Salidas (sin cambios):
- build_full_daily_panel(raw): DataFrame diario listo para learners y GSC.
- split_into_chunks(df): {'A','B','C','merged_core_only'}.

Notas de optimización:
- Particionado incremental sin replicar todo el DF.
- Tipos compactos y categóricos.
- GroupBy con observed=True cuando aplica.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import sys

# Rutas utilidades (sin cambios)
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.utils import add_week_start, safe_bool_to_int, ensure_columns, cast_if_exists
from src.conf import config

# ====================================================
# Parámetros (idénticos; con defaults si no existen)
# ====================================================

COVERAGE_TARGET = getattr(config, "COVERAGE_TARGET", 0.85)
UNIVERSE_BY = getattr(config, "UNIVERSE_BY", "class")
MIN_ITEMS_PER_CLASS_STORE = getattr(config, "MIN_ITEMS_PER_CLASS_STORE", 5)
MIN_ITEM_COVERAGE_PRE = getattr(config, "MIN_ITEM_COVERAGE_PRE", 0.70)
STORE_DAILY_COVERAGE_MIN = getattr(config, "STORE_DAILY_COVERAGE_MIN", 0.95)
STORE_ACTIVITY_PCTL = getattr(config, "STORE_ACTIVITY_PCTL", 0.50)

OVERLAP_TRIM_Q_LOW = getattr(config, "OVERLAP_TRIM_Q_LOW", 0.025)
OVERLAP_TRIM_Q_HIGH = getattr(config, "OVERLAP_TRIM_Q_HIGH", 0.975)
OVERLAP_TRIM_LOCAL = getattr(config, "OVERLAP_TRIM_LOCAL", False)

HOLIDAY_WINDOW_K = getattr(config, "HOLIDAY_WINDOW_K", 2)
EQ_DATE = pd.to_datetime(getattr(config, "EQ_DATE", "2016-04-16"))
EQ_WINDOW_DAYS = getattr(config, "EQ_WINDOW_DAYS", 14)
EQ_MISSING_SHARE_MAX = getattr(config, "EQ_MISSING_SHARE_MAX", 0.50)

USE_CORR_WEIGHTS = getattr(config, "USE_CORR_WEIGHTS", False)
ROLLING_H_WINDOW = getattr(config, "ROLLING_H_WINDOW", 3)

# Episodios GSC
H_EPISODE_THRESHOLD = getattr(config, "H_EPISODE_THRESHOLD", 0.30)
H_EPISODE_MIN_DAYS = getattr(config, "H_EPISODE_MIN_DAYS", 3)
GSC_PRE_DAYS = getattr(config, "GSC_PRE_DAYS", 90)
GSC_PRE_GAP_DAYS = getattr(config, "GSC_PRE_GAP_DAYS", 7)
GSC_TREATMENT_LENGTH = getattr(config, "GSC_TREATMENT_LENGTH", 14)
GSC_POST_LENGTH = getattr(config, "GSC_POST_LENGTH", 30)
ALLOW_POST_IN_HALO = getattr(config, "ALLOW_POST_IN_HALO", False)

# Transformaciones outcome
WINSOR_P = getattr(config, "WINSOR_P", 0.995)
USE_HURDLE_FEATURES = getattr(config, "USE_HURDLE_FEATURES", False)

FOURIER_K = getattr(config, "FOURIER_K", 0)

# ====================================================
# Utilitarios
# ====================================================

def _ensure_types_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # cache=True reduce CPU y memoria temporal
    df["date"] = pd.to_datetime(df["date"], cache=True)
    return df

def _as_category(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Castea columnas string/códigos a category si existen."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def _winsorize_by_group(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    upper_p: float = 0.995
) -> pd.Series:
    """
    Winsorización por grupo: clip superior en cuantil 'upper_p'.
    Implementación estable y vectorizada.
    """
    # Cuantil por grupo (serie pequeña)
    q = (
        df.groupby(group_cols, sort=False, observed=True)[value_col]
          .quantile(upper_p)
          .rename("_q")
          .reset_index()
    )
    # Mapear al DF (sin replicar copias profundas)
    capped = df[group_cols].merge(q, on=group_cols, how="left")["_q"].to_numpy(copy=False)
    s = df[value_col].to_numpy(copy=False)
    # clip inplace-like
    s2 = np.minimum(s, np.nan_to_num(capped, nan=np.nanmax(s))).astype("float32", copy=False)
    return pd.Series(s2, index=df.index, name=f"{value_col}_w")

def _add_calendar_features_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["date"].dt.weekday.astype("int8")
    df["dom"] = df["date"].dt.day.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["is_month_start"] = df["date"].dt.is_month_start.astype("int8")
    df["is_month_end"] = df["date"].dt.is_month_end.astype("int8")

    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", dtype="int8")
    if "dow_0" in dow_dummies.columns:
        dow_dummies = dow_dummies.drop(columns=["dow_0"])
    month_dummies = pd.get_dummies(df["month"], prefix="m", dtype="int8")
    if "m_1" in month_dummies.columns:
        month_dummies = month_dummies.drop(columns=["m_1"])
    df = pd.concat([df, dow_dummies, month_dummies], axis=1)

    day_idx = (df["date"] - df["date"].min()).dt.days.astype("int32")
    two_pi = 2 * np.pi
    df["sin_anual"] = np.sin(two_pi * day_idx / 365.25).astype("float32")
    df["cos_anual"] = np.cos(two_pi * day_idx / 365.25).astype("float32")
    df["sin_semanal"] = np.sin(two_pi * day_idx / 7).astype("float32")
    df["cos_semanal"] = np.cos(two_pi * day_idx / 7).astype("float32")
    return df

def _build_holiday_daily(hol: pd.DataFrame, stores: pd.DataFrame, k: int = HOLIDAY_WINDOW_K) -> pd.DataFrame:
    hol = _ensure_types_dates(hol)
    stores = stores[["store_nbr","city","state"]].copy()

    if "transferred" in hol.columns:
        hol = hol[hol["transferred"] == False].copy()
    if "type" in hol.columns:
        hol = hol[hol["type"] != "Work Day"].copy()

    # National → todas las tiendas
    hol_nat = hol[hol["locale"] == "National"].assign(key=1)
    all_stores = stores[["store_nbr"]].assign(key=1)
    hol_nat = hol_nat.merge(all_stores, on="key", how="left").drop(columns="key")
    h_nat = hol_nat[["date", "store_nbr"]].assign(H_nat_d=1)

    # Regional por estado
    if {"locale_name", "state"}.issubset(hol.columns.union(stores.columns)):
        hol_reg = hol[hol["locale"] == "Regional"].merge(
            stores[["state", "store_nbr"]],
            left_on="locale_name", right_on="state", how="left"
        )
        h_reg = hol_reg[["date", "store_nbr"]].assign(H_reg_d=1)
    else:
        h_reg = pd.DataFrame(columns=["date","store_nbr","H_reg_d"])

    # Local por ciudad
    if {"locale_name", "city"}.issubset(hol.columns.union(stores.columns)):
        hol_loc = hol[hol["locale"] == "Local"].merge(
            stores[["city", "store_nbr"]],
            left_on="locale_name", right_on="city", how="left"
        )
        h_loc = hol_loc[["date", "store_nbr"]].assign(H_loc_d=1)
    else:
        h_loc = pd.DataFrame(columns=["date","store_nbr","H_loc_d"])

    H = (
        h_nat.merge(h_reg, on=["date","store_nbr"], how="outer")
             .merge(h_loc, on=["date","store_nbr"], how="outer")
             .fillna(0)
    )
    for c in ["H_nat_d", "H_reg_d", "H_loc_d"]:
        if c not in H.columns:
            H[c] = 0
        H[c] = H[c].astype("int8")

    # Lags/leads ±k por tienda (no expandimos a calendario completo para ahorrar RAM)
    H.sort_values(["store_nbr","date"], inplace=True)
    for base in ["H_nat_d", "H_reg_d", "H_loc_d"]:
        for lag in range(1, k + 1):
            H[f"{base}_lag{lag}"] = H.groupby("store_nbr", sort=False)[base].shift(lag).fillna(0).astype("int8")
            H[f"{base}_lead{lag}"] = H.groupby("store_nbr", sort=False)[base].shift(-lag).fillna(0).astype("int8")

    return H

def _build_transactions_daily(trans: pd.DataFrame) -> pd.DataFrame:
    trans = _ensure_types_dates(trans)
    trans = trans.sort_values(["store_nbr", "date"]).copy()
    trans["transactions"] = pd.to_numeric(trans["transactions"], errors="coerce").fillna(0).astype("float32")

    trans["F_st_lag1"] = trans.groupby("store_nbr", sort=False)["transactions"].shift(1)
    # rolling mean nativa (más eficiente que apply)
    trans["F_st_ma7"] = (
        trans.groupby("store_nbr", sort=False)["transactions"]
             .rolling(window=7, min_periods=1).mean()
             .reset_index(level=0, drop=True)
    )
    trans["F_st_ma28"] = (
        trans.groupby("store_nbr", sort=False)["transactions"]
             .rolling(window=28, min_periods=1).mean()
             .reset_index(level=0, drop=True)
    )
    # Desfase 1 día (solo pasado)
    for c in ["F_st_ma7", "F_st_ma28"]:
        trans[c] = trans.groupby("store_nbr", sort=False)[c].shift(1)

    for c in ["F_st_lag1", "F_st_ma7", "F_st_ma28"]:
        trans[c] = trans[c].astype("float32")

    return trans[["date", "store_nbr", "transactions", "F_st_lag1", "F_st_ma7", "F_st_ma28"]]

def _build_oil_daily(oil: pd.DataFrame) -> pd.DataFrame:
    oil = _ensure_types_dates(oil).sort_values("date").copy()
    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce").ffill()
    oil["log_oil"] = np.log1p(oil["dcoilwtico"]).astype("float32")
    oil["oil_lag7"] = oil["dcoilwtico"].shift(7)
    oil["oil_lag28"] = oil["dcoilwtico"].shift(28)
    oil["d_oil"] = oil["dcoilwtico"].diff(1)
    oil["d_oil_lag7"] = oil["dcoilwtico"].diff(7)
    oil["d_oil_lag28"] = oil["dcoilwtico"].diff(28)
    for c in ["dcoilwtico", "log_oil", "oil_lag7", "oil_lag28", "d_oil", "d_oil_lag7", "d_oil_lag28"]:
        oil[c] = oil[c].astype("float32")
    return oil.rename(columns={"dcoilwtico": "oil"})

def _build_earthquake_features(df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_nbr", "date"]).copy()
    df["eq_day"] = (df["date"] == EQ_DATE).astype("int8")
    for k in range(1, EQ_WINDOW_DAYS + 1):
        df[f"eq_lag{k}"] = (df["date"] == (EQ_DATE + pd.Timedelta(days=k))).astype("int8")
    return df

# ====================================================
# Base diaria
# ====================================================

def build_daily_base(ventas: pd.DataFrame, items: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    ventas = _ensure_types_dates(ventas)
    items = items.copy()
    stores = stores.copy()

    # Casts numéricos
    ventas["store_nbr"] = ventas["store_nbr"].astype("int32")
    ventas["item_nbr"] = ventas["item_nbr"].astype("int32")
    ventas["unit_sales"] = pd.to_numeric(ventas["unit_sales"], errors="coerce").fillna(0).clip(lower=0).astype("float32")
    ventas["onpromotion"] = safe_bool_to_int(ventas["onpromotion"]).astype("int8")

    items["item_nbr"] = items["item_nbr"].astype("int32")
    if "perishable" in items.columns:
        items["perishable"] = items["perishable"].astype("int8")
    stores["store_nbr"] = stores["store_nbr"].astype("int32")

    # Categóricos para ahorrar RAM
    items = _as_category(items, ["family","class"])
    stores = _as_category(stores, ["city","state","type","cluster"])

    # Winsorización por ítem-tienda
    ventas = ventas.sort_values(["store_nbr", "item_nbr", "date"]).copy()
    ventas["unit_sales_raw"] = ventas["unit_sales"].astype("float32")
    ventas["unit_sales_w"] = _winsorize_by_group(
        ventas, ["store_nbr", "item_nbr"], "unit_sales", upper_p=WINSOR_P
    ).astype("float32")
    y_tilde = np.log1p(np.maximum(0, ventas["unit_sales_w"])).astype("float32")
    ventas["y_tilde"] = y_tilde
    if USE_HURDLE_FEATURES:
        ventas["y_pos"] = (ventas["unit_sales_w"] > 0).astype("int8")
        ventas["y_tilde_pos"] = np.where(ventas["unit_sales_w"] > 0, y_tilde, 0).astype("float32")

    base = (
        ventas.merge(items, on="item_nbr", how="left")
              .merge(stores, on="store_nbr", how="left")
    )

    base = _add_calendar_features_daily(base)

    # Tipos finales de esta etapa
    for c in ["unit_sales_raw", "unit_sales_w", "y_tilde"]:
        if c in base.columns:
            base[c] = base[c].astype("float32")

    if USE_HURDLE_FEATURES and "y_tilde_pos" in base.columns:
        base["y_tilde_pos"] = base["y_tilde_pos"].astype("float32")

    gc.collect()
    return base

# ====================================================
# Exposición competitiva H_{jst}
# ====================================================

def _class_store_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["date", "store_nbr", "class"], as_index=False, sort=False, observed=True).agg(
        n_items=("item_nbr", "nunique"),
        n_promo=("onpromotion", "sum")
    )
    g["n_items"] = g["n_items"].astype("int32")
    g["n_promo"] = g["n_promo"].astype("int32")
    return g

def _compute_equal_weight_H(df: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(counts, on=["date", "store_nbr", "class"], how="left")
    denom = (df["n_items"] - 1).clip(lower=1)
    num = (df["n_promo"] - df["onpromotion"]).clip(lower=0)
    df["H_share_w"] = (num / denom).astype("float32")
    df["H_share_simple"] = df["H_share_w"].astype("float32")
    df["H_count_neighbors"] = num.astype("int16")
    return df

def _compute_corr_weights(df_pre: pd.DataFrame) -> pd.DataFrame:
    tmp = df_pre[["date","store_nbr","class","item_nbr","y_tilde","dow"]].copy()
    fe = tmp.groupby(["store_nbr","dow"], sort=False, observed=True)["y_tilde"].transform("mean")
    tmp = tmp.assign(y_res=tmp["y_tilde"] - fe)
    weights = []
    for (s, c), g in tmp.groupby(["store_nbr","class"], sort=False, observed=True):
        pivot = g.pivot_table(index="date", columns="item_nbr", values="y_res")
        corr = pivot.corr(min_periods=10)
        if corr is None:
            continue
        corr = corr.fillna(0).clip(lower=0)
        for j in corr.columns:
            row = corr[j].drop(index=j, errors="ignore")
            if row.sum() <= 0:
                continue
            w = (row / row.sum()).astype("float32")
            weights.append(
                pd.DataFrame({
                    "store_nbr": s,
                    "class": c,
                    "item_nbr": j,
                    "neighbor_item_nbr": w.index.values,
                    "w": w.values
                })
            )
        del pivot, corr
        gc.collect()
    if len(weights) == 0:
        return pd.DataFrame(columns=["store_nbr","class","item_nbr","neighbor_item_nbr","w"])
    W = pd.concat(weights, ignore_index=True)
    return W

def _apply_corr_weighted_H(df: pd.DataFrame, W: pd.DataFrame) -> pd.DataFrame:
    if W.empty:
        df["H_share_w"] = df["H_share_simple"].astype("float32")
        return df
    promo_neighbors = (
        df[["date","store_nbr","class","item_nbr","onpromotion"]]
          .rename(columns={"item_nbr":"neighbor_item_nbr","onpromotion":"onpromotion_neighbor"})
    )
    tmp = (
        W.merge(promo_neighbors, on=["store_nbr","class","neighbor_item_nbr"], how="left")
         .merge(df[["date","store_nbr","class","item_nbr"]].drop_duplicates(),
                on=["date","store_nbr","class","item_nbr"], how="right")
    )
    tmp["w"] = tmp["w"].fillna(0).astype("float32")
    tmp["onpromotion_neighbor"] = tmp["onpromotion_neighbor"].fillna(0).astype("float32")
    contrib = tmp.groupby(["date","store_nbr","class","item_nbr"], as_index=False, sort=False).agg(
        H_share_w=("onpromotion_neighbor", lambda x: float(np.dot(x, tmp.loc[x.index, "w"])))
    )
    df = df.merge(contrib, on=["date","store_nbr","class","item_nbr"], how="left")
    df["H_share_w"] = df["H_share_w"].fillna(df["H_share_simple"]).astype("float32")
    del promo_neighbors, tmp, contrib
    gc.collect()
    return df

def _compute_H_variants(df: pd.DataFrame, pre_start: pd.Timestamp, pre_end: pd.Timestamp) -> pd.DataFrame:
    df = df.sort_values(["store_nbr","class","item_nbr","date"]).copy()

    # Ordinal por cuantiles diarios (clase-tienda en el día)
    def _ord_grp(g: pd.DataFrame) -> pd.Series:
        if g.shape[0] <= 2:
            med = g["H_share_w"].median()
            return (g["H_share_w"] > med).astype("int8")
        try:
            r = g["H_share_w"].rank(method="first")
            q = pd.qcut(r, q=3, labels=[0,1,2])
            return q.astype("int8")
        except ValueError:
            med = g["H_share_w"].median()
            return (g["H_share_w"] > med).astype("int8")

    df["H_ord_q"] = (
        df.groupby(["date","store_nbr","class"], sort=False, observed=True)["H_share_w"]
          .transform(lambda s: _ord_grp(s.to_frame()))
    ).astype("int8", errors="ignore")

    # Umbral binario basado en mediana pre (clase-tienda)
    mask_pre = (df["date"] >= pre_start) & (df["date"] <= pre_end)
    tau = (
        df.loc[mask_pre].groupby(["store_nbr","class"], sort=False, observed=True)["H_share_w"]
          .median()
          .rename("tau_cls")
          .reset_index()
    )
    df = df.merge(tau, on=["store_nbr","class"], how="left")
    df["H_bin"] = (df["H_share_w"] >= df["tau_cls"]).astype("int8")
    df.drop(columns=["tau_cls"], inplace=True)

    # MA(3) por ítem-tienda
    df["H_ma3"] = (
        df.groupby(["store_nbr","item_nbr"], sort=False)["H_share_w"]
          .rolling(window=ROLLING_H_WINDOW, min_periods=1).mean()
          .reset_index(level=[0,1], drop=True)
          .astype("float32")
    )
    return df

# ====================================================
# Universo y filtros
# ====================================================

def _select_top_classes(df: pd.DataFrame, by: str = UNIVERSE_BY, target: float = COVERAGE_TARGET) -> List:
    agg = df.groupby(by, as_index=False, observed=True)[["unit_sales_raw"]].sum().sort_values("unit_sales_raw", ascending=False)
    agg["cum_share"] = agg["unit_sales_raw"].cumsum() / agg["unit_sales_raw"].sum()
    keep_vals = agg.loc[agg["cum_share"] <= target, by].tolist()
    if len(keep_vals) < min(5, len(agg)):
        keep_vals = agg.head(min(5, len(agg)))[by].tolist()
    return keep_vals

def _filter_class_store_density(df: pd.DataFrame, pre_start: pd.Timestamp, pre_end: pd.Timestamp) -> pd.DataFrame:
    mask_pre = (df["date"] >= pre_start) & (df["date"] <= pre_end)
    days_pre = (pre_end - pre_start).days + 1
    cover = (
        df.loc[mask_pre].groupby(["store_nbr","class","item_nbr"], sort=False, observed=True)["date"]
          .nunique()
          .rename("ndays")
          .reset_index()
    )
    cover["coverage"] = cover["ndays"] / float(days_pre)
    active = cover.loc[cover["coverage"] >= MIN_ITEM_COVERAGE_PRE]
    dense = (
        active.groupby(["store_nbr","class"], sort=False, observed=True)["item_nbr"]
              .nunique()
              .rename("n_active_items")
              .reset_index()
    )
    keep_cs = dense.loc[dense["n_active_items"] >= MIN_ITEMS_PER_CLASS_STORE, ["store_nbr","class"]]
    df2 = df.merge(keep_cs.assign(keep_cs=1), on=["store_nbr","class"], how="left")
    out = df2[df2["keep_cs"] == 1].drop(columns=["keep_cs"])
    del cover, active, dense, keep_cs, df2
    gc.collect()
    return out

def _filter_stores_coverage_activity(df: pd.DataFrame, trans: pd.DataFrame, pre_start: pd.Timestamp, pre_end: pd.Timestamp) -> pd.DataFrame:
    mask_pre = (df["date"] >= pre_start) & (df["date"] <= pre_end)
    days_pre = (pre_end - pre_start).days + 1

    store_days = df.loc[mask_pre].groupby(["store_nbr","date"], sort=False)["item_nbr"].nunique().reset_index()
    cover = store_days.groupby("store_nbr", sort=False)["date"].nunique().rename("ndays").reset_index()
    cover["coverage_store"] = cover["ndays"] / float(days_pre)

    trans = _ensure_types_dates(trans)
    trans_pre = trans[(trans["date"] >= pre_start) & (trans["date"] <= pre_end)]
    trans_agg = trans_pre.groupby("store_nbr", sort=False)["transactions"].sum().rename("trans_pre_sum").reset_index()
    thr_trans = trans_agg["trans_pre_sum"].quantile(STORE_ACTIVITY_PCTL)

    promo_days = (
        df.loc[mask_pre, ["store_nbr","date","onpromotion"]]
          .assign(any_promo=lambda x: x["onpromotion"] > 0)
          .groupby(["store_nbr","date"], sort=False)["any_promo"].max().reset_index()
    )
    promo_agg = promo_days.groupby("store_nbr", sort=False)["any_promo"].sum().rename("days_with_promo").reset_index()
    thr_promo = promo_agg["days_with_promo"].quantile(STORE_ACTIVITY_PCTL)

    stores_ok = (
        cover.merge(trans_agg, on="store_nbr", how="left")
             .merge(promo_agg, on="store_nbr", how="left")
    )
    stores_ok["trans_pre_sum"] = stores_ok["trans_pre_sum"].fillna(0)
    stores_ok["days_with_promo"] = stores_ok["days_with_promo"].fillna(0)
    stores_ok["ok_cov"] = stores_ok["coverage_store"] >= STORE_DAILY_COVERAGE_MIN
    stores_ok["ok_act"] = (stores_ok["trans_pre_sum"] >= thr_trans) | (stores_ok["days_with_promo"] >= thr_promo)

    eq_start = EQ_DATE - pd.Timedelta(days=EQ_WINDOW_DAYS)
    eq_end = EQ_DATE + pd.Timedelta(days=EQ_WINDOW_DAYS)
    store_eq_days = df[(df["date"] >= eq_start) & (df["date"] <= eq_end)].groupby(["store_nbr","date"], sort=False)["item_nbr"].nunique().reset_index()
    eq_cover = store_eq_days.groupby("store_nbr", sort=False)["date"].nunique().rename("eq_ndays").reset_index()
    total_eq_days = (eq_end - eq_start).days + 1
    eq_cover["eq_coverage"] = eq_cover["eq_ndays"] / float(total_eq_days)
    stores_ok = stores_ok.merge(eq_cover, on="store_nbr", how="left")
    stores_ok["eq_coverage"] = stores_ok["eq_coverage"].fillna(0)
    stores_ok["ok_eq"] = stores_ok["eq_coverage"] >= (1 - EQ_MISSING_SHARE_MAX)

    keep_stores = stores_ok.loc[stores_ok["ok_cov"] & stores_ok["ok_act"] & stores_ok["ok_eq"], "store_nbr"]
    out = df[df["store_nbr"].isin(keep_stores)].copy()
    del store_days, cover, trans_pre, trans_agg, promo_days, promo_agg, stores_ok, eq_cover, keep_stores
    gc.collect()
    return out

def _overlap_trimming(df: pd.DataFrame, pre_start: pd.Timestamp, pre_end: pd.Timestamp, local: bool = OVERLAP_TRIM_LOCAL) -> pd.DataFrame:
    mask_pre = (df["date"] >= pre_start) & (df["date"] <= pre_end)
    meanH = (
        df.loc[mask_pre].groupby(["store_nbr","item_nbr","class"], sort=False, observed=True)["H_share_w"]
          .mean()
          .rename("H_pre_mean")
          .reset_index()
    )
    if local:
        q = (
            meanH.groupby(["store_nbr","class"], sort=False, observed=True)["H_pre_mean"]
                 .quantile([OVERLAP_TRIM_Q_LOW, OVERLAP_TRIM_Q_HIGH])
                 .unstack(level=-1)
                 .rename(columns={OVERLAP_TRIM_Q_LOW:"q_low", OVERLAP_TRIM_Q_HIGH:"q_high"})
                 .reset_index()
        )
        m = meanH.merge(q, on=["store_nbr","class"], how="left")
        keep_units = m[(m["H_pre_mean"] >= m["q_low"]) & (m["H_pre_mean"] <= m["q_high"])][["store_nbr","item_nbr"]]
    else:
        q_low = meanH["H_pre_mean"].quantile(OVERLAP_TRIM_Q_LOW)
        q_high = meanH["H_pre_mean"].quantile(OVERLAP_TRIM_Q_HIGH)
        keep_units = meanH[(meanH["H_pre_mean"] >= q_low) & (meanH["H_pre_mean"] <= q_high)][["store_nbr","item_nbr"]]
    df2 = df.merge(keep_units.assign(keep=1), on=["store_nbr","item_nbr"], how="left")
    out = df2[df2["keep"] == 1].drop(columns=["keep"])
    del meanH, df2, keep_units
    gc.collect()
    return out

# ====================================================
# Episodios GSC
# ====================================================

def _detect_gsc_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta episodios a nivel (store_nbr, class).
    Reutiliza n_items/n_promo si ya están en df para ahorrar RAM.
    """
    have_counts = ("n_items" in df.columns) and ("n_promo" in df.columns)
    if have_counts:
        cls_daily = (
            df.groupby(["date","store_nbr","class"], as_index=False, sort=False, observed=True)
              .agg(n_items=("n_items","max"), n_promo=("n_promo","max"))
        )
    else:
        cls_daily = (
            df.groupby(["date","store_nbr","class"], as_index=False, sort=False, observed=True)
              .agg(n_items=("item_nbr","nunique"), n_promo=("onpromotion","sum"))
        )

    cls_daily["share_cls_promo"] = (cls_daily["n_promo"] / cls_daily["n_items"].clip(lower=1)).astype("float32")
    cls_daily.sort_values(["store_nbr","class","date"], inplace=True)

    cls_daily["above"] = (cls_daily["share_cls_promo"] >= H_EPISODE_THRESHOLD).astype("int8")

    def _label_runs(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["run_id"] = (g["above"].ne(g["above"].shift())).cumsum()
        runs = g[g["above"] == 1].groupby("run_id")["date"].agg(["min","max","count"]).reset_index()
        runs = runs[runs["count"] >= H_EPISODE_MIN_DAYS]
        g["episode_start"] = pd.NaT
        for _, r in runs.iterrows():
            g.loc[g["run_id"] == r["run_id"], "episode_start"] = r["min"]
        return g.drop(columns=["run_id"])

    cls_daily = cls_daily.groupby(["store_nbr","class"], group_keys=False, sort=False, observed=True).apply(_label_runs)
    cls_daily["has_epi"] = cls_daily["episode_start"].notna().astype("int8")

    def _windows(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["t_rel"] = (g["date"] - g["episode_start"]).dt.days
        g["episode_id"] = np.where(
            g["has_epi"] == 1,
            "E_" + g["store_nbr"].astype(str) + "_" + g["class"].astype(str) + "_" + g["episode_start"].dt.strftime("%Y%m%d"),
            pd.NA
        )
        g["is_treat"] = ((g["t_rel"] >= 0) & (g["t_rel"] <= GSC_TREATMENT_LENGTH) & g["has_epi"].eq(1)).astype("int8")
        g["is_pre"] = ((g["t_rel"] >= -GSC_PRE_DAYS) & (g["t_rel"] <= -GSC_PRE_GAP_DAYS) & g["has_epi"].eq(1)).astype("int8")
        g["is_post"] = ((g["t_rel"] >= (GSC_TREATMENT_LENGTH + 1)) &
                        (g["t_rel"] <= (GSC_TREATMENT_LENGTH + GSC_POST_LENGTH)) &
                        g["has_epi"].eq(1)).astype("int8")
        return g

    cls_daily = cls_daily.groupby(["store_nbr","class"], group_keys=False, sort=False, observed=True).apply(_windows)
    epi_cols = ["date","store_nbr","class","share_cls_promo","episode_start","episode_id","t_rel","is_pre","is_treat","is_post","has_epi"]
    out = cls_daily[epi_cols]
    del cls_daily
    gc.collect()
    return out

# ====================================================
# Partición en chunks con halo y recombinación eficiente
# ====================================================

CHUNK_SPEC_DEFAULT = {
    "A": {
        "core":  (pd.Timestamp("2013-05-01"), pd.Timestamp("2014-12-31")),
        "halo":  (pd.Timestamp("2013-01-01"), pd.Timestamp("2015-01-30")),
    },
    "B": {
        "core":  (pd.Timestamp("2015-01-01"), pd.Timestamp("2016-06-30")),
        "halo":  (pd.Timestamp("2014-09-03"), pd.Timestamp("2016-07-30")),
    },
    "C": {
        "core":  (pd.Timestamp("2016-07-01"), pd.Timestamp("2017-07-15")),
        "halo":  (pd.Timestamp("2016-03-03"), pd.Timestamp("2017-08-15")),
    }
}

def _make_single_chunk(df: pd.DataFrame, ch: str, win: Dict) -> pd.DataFrame:
    """Construye chunk individual (halo + flag core) sin replicar todo el DF."""
    halo_lo, halo_hi = win["halo"]
    core_lo, core_hi = win["core"]
    m_halo = (df["date"] >= halo_lo) & (df["date"] <= halo_hi)
    m_core = (df["date"] >= core_lo) & (df["date"] <= core_hi)
    out = df.loc[m_halo].copy()
    out["chunk"] = ch
    out["is_core"] = m_core.loc[m_halo].astype("int8")
    out["chunk"] = out["chunk"].astype("category")
    return out

def split_into_chunks(df: pd.DataFrame, spec: Dict = CHUNK_SPEC_DEFAULT) -> Dict[str, pd.DataFrame]:
    """
    Retorna {'A':dfA, 'B':dfB, 'C':dfC, 'merged_core_only':df_merged}
    Implementación RAM-friendly: construye A/B/C por separado y recombina con prioridad a is_core.
    """
    # Construcción chunk a chunk (no se concatena todo al mismo tiempo)
    dfA = _make_single_chunk(df, "A", spec["A"])
    dfB = _make_single_chunk(df, "B", spec["B"])
    dfC = _make_single_chunk(df, "C", spec["C"])

    # Recombinación incremental con prioridad is_core==1
    key_cols = ["date","store_nbr","item_nbr"]
    def _to_idx(d: pd.DataFrame) -> pd.DataFrame:
        return d.set_index(key_cols, drop=False)

    merged = None
    for part in (dfA, dfB, dfC):
        part_idx = _to_idx(part)
        # 1) Sobrescribir/insertar cores
        cores = part_idx[part_idx["is_core"] == 1]
        if merged is None:
            merged = cores.copy()
        else:
            # Update valores existentes por core
            merged.update(cores)
            # Agregar cores no existentes
            to_add = cores.index.difference(merged.index)
            if len(to_add) > 0:
                merged = pd.concat([merged, cores.loc[to_add]], axis=0)
        # 2) Agregar halos que no existan aún
        halos = part_idx[part_idx["is_core"] == 0]
        to_add_halo = halos.index.difference(merged.index)
        if len(to_add_halo) > 0:
            merged = pd.concat([merged, halos.loc[to_add_halo]], axis=0)

        del part_idx, cores, halos, to_add_halo
        gc.collect()

    merged = merged.sort_values(key_cols).reset_index(drop=True)

    return {"A": dfA, "B": dfB, "C": dfC, "merged_core_only": merged}

def _flag_episode_eligibility_by_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df = df_chunk.copy()
    if "episode_start" not in df.columns:
        return df.assign(gsc_episode_eligible=np.int8(0))
    core_lo = df.loc[df["is_core"] == 1, "date"].min() if "is_core" in df.columns else df["date"].min()
    core_hi = df.loc[df["is_core"] == 1, "date"].max() if "is_core" in df.columns else df["date"].max()

    def _is_window_inside(start: pd.Timestamp) -> bool:
        pre_lo = start - pd.Timedelta(days=GSC_PRE_DAYS)
        pre_hi = start - pd.Timedelta(days=GSC_PRE_GAP_DAYS)
        treat_hi = start + pd.Timedelta(days=GSC_TREATMENT_LENGTH)
        post_hi = treat_hi + pd.Timedelta(days=GSC_POST_LENGTH)
        ok_pre = (pre_lo >= core_lo) & (pre_hi <= core_hi)
        ok_treat = (start >= core_lo) & (treat_hi <= core_hi)
        ok_post = (post_hi <= core_hi) if not ALLOW_POST_IN_HALO else (post_hi <= df["date"].max())
        return bool(ok_pre and ok_treat and ok_post)

    epi_starts = (
        df.dropna(subset=["episode_start"])
          .groupby(["store_nbr","class","episode_id"], sort=False)["episode_start"]
          .first()
          .reset_index()
    )
    epi_starts["eligible"] = epi_starts["episode_start"].apply(_is_window_inside).astype("int8")
    df = df.merge(epi_starts[["store_nbr","class","episode_id","eligible"]],
                  on=["store_nbr","class","episode_id"], how="left")
    df["gsc_episode_eligible"] = df["eligible"].fillna(0).astype("int8")
    df.drop(columns=["eligible"], inplace=True)
    return df

# ====================================================
# Orquestador diario punta a punta
# ====================================================

def build_full_daily_panel(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    ventas = raw["ventas"]
    items = raw["items"]
    stores = raw["stores"]
    hol = raw["hol"]
    trans = raw["trans"]
    oil = raw["oil"]

    # 1) Base
    base = build_daily_base(ventas, items, stores)

    # 2) Confounders (joins compactos)
    Hday = _build_holiday_daily(hol, stores, k=HOLIDAY_WINDOW_K)
    base = base.merge(Hday, on=["date","store_nbr"], how="left")
    del Hday; gc.collect()

    Tday = _build_transactions_daily(trans)
    base = base.merge(Tday, on=["date","store_nbr"], how="left")
    del Tday; gc.collect()

    Oday = _build_oil_daily(oil)
    base = base.merge(Oday, on="date", how="left")
    del Oday; gc.collect()

    base = _build_earthquake_features(base, stores)

    # 3) Exposición H_{jst}
    counts = _class_store_daily_counts(base)
    base = _compute_equal_weight_H(base, counts)
    del counts; gc.collect()

    # (Opcional) Pesos por correlación en pre
    pre_global_start = pd.Timestamp("2013-01-01")
    pre_global_end = pd.Timestamp("2015-12-31")
    if USE_CORR_WEIGHTS:
        df_pre = base[(base["date"] >= pre_global_start) & (base["date"] <= pre_global_end)]
        W = _compute_corr_weights(df_pre)
        base = _apply_corr_weighted_H(base, W)
        del W, df_pre; gc.collect()

    base = _compute_H_variants(base, pre_start=pre_global_start, pre_end=pre_global_end)

    # 4) Universo y filtros
    keep_vals = _select_top_classes(base, by=UNIVERSE_BY, target=COVERAGE_TARGET)
    base = base[base[UNIVERSE_BY].isin(keep_vals)].copy()

    base = _filter_class_store_density(base, pre_start=pre_global_start, pre_end=pre_global_end)

    base = _filter_stores_coverage_activity(base, trans, pre_start=pre_global_start, pre_end=pre_global_end)

    base = _overlap_trimming(base, pre_start=pre_global_start, pre_end=pre_global_end, local=OVERLAP_TRIM_LOCAL)

    # 5) Episodios GSC
    epi = _detect_gsc_episodes(base)
    base = base.merge(epi, on=["date","store_nbr","class"], how="left")
    del epi; gc.collect()

    # 6) Fourier semanal opcional
    if FOURIER_K > 0:
        tmp = add_week_start(base, "date", "week_start")
        weeks_sorted = np.sort(tmp["week_start"].unique())
        week_to_index = {w: i for i, w in enumerate(weeks_sorted)}
        base["w_idx"] = tmp["week_start"].map(week_to_index).astype("int32")
        for kk in range(1, FOURIER_K + 1):
            base[f"fourier_s{kk}"] = np.sin(2 * np.pi * kk * base["w_idx"] / 52).astype("float32")
            base[f"fourier_c{kk}"] = np.cos(2 * np.pi * kk * base["w_idx"] / 52).astype("float32")

    base = cast_if_exists(base, {"store_nbr": "int32", "item_nbr": "int32", "perishable": "int8"})
    # Categóricos de alta cardinalidad
    base = _as_category(base, ["family","class","city","state","type","cluster","episode_id"])

    gc.collect()
    return base

# ====================================================
# Helpers: columnas finales
# ====================================================

def finalize_daily_columns(panel: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "date","store_nbr","item_nbr","family","class","perishable","city","state","type","cluster",
        "unit_sales_raw","unit_sales_w","y_tilde","onpromotion",
        # Exposición
        "H_share_w","H_share_simple","H_count_neighbors","H_ord_q","H_bin","H_ma3",
        # Episodios
        "share_cls_promo","episode_start","episode_id","t_rel","is_pre","is_treat","is_post","has_epi",
        # Tráfico y petróleo
        "transactions","F_st_lag1","F_st_ma7","F_st_ma28",
        "oil","log_oil","oil_lag7","oil_lag28","d_oil","d_oil_lag7","d_oil_lag28",
        # Festivos
        "H_nat_d","H_reg_d","H_loc_d"
    ]
    for base in ["H_nat_d","H_reg_d","H_loc_d"]:
        for lag in range(1, HOLIDAY_WINDOW_K + 1):
            base_cols += [f"{base}_lag{lag}", f"{base}_lead{lag}"]
    cal_cols = ["dow","dom","month","is_month_start","is_month_end","sin_anual","cos_anual","sin_semanal","cos_semanal"]
    cal_cols += [c for c in panel.columns if c.startswith("dow_") or c.startswith("m_")]
    base_cols += cal_cols
    eq_cols = ["eq_day"] + [f"eq_lag{k}" for k in range(1, EQ_WINDOW_DAYS + 1)]
    base_cols += eq_cols
    if FOURIER_K > 0:
        base_cols += ["w_idx"] + [f"fourier_s{k}" for k in range(1, FOURIER_K + 1)] + [f"fourier_c{k}" for k in range(1, FOURIER_K + 1)]

    panel = ensure_columns(panel, base_cols)
    panel = panel[base_cols].reset_index(drop=True)

    float_cols = ["unit_sales_raw","unit_sales_w","y_tilde","H_share_w","H_share_simple","H_ma3",
                  "share_cls_promo","F_st_lag1","F_st_ma7","F_st_ma28",
                  "oil","log_oil","oil_lag7","oil_lag28","d_oil","d_oil_lag7","d_oil_lag28",
                  "sin_anual","cos_anual","sin_semanal","cos_semanal"]
    for c in float_cols:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")

    int_cols = ["store_nbr","item_nbr","perishable","onpromotion","H_count_neighbors","H_ord_q","H_bin",
                "is_pre","is_treat","is_post","has_epi","dow","dom","month","is_month_start","is_month_end","eq_day"]
    for c in int_cols:
        if c in panel.columns:
            panel[c] = panel[c].astype("int32" if c in ["store_nbr","item_nbr"] else "int8")

    gc.collect()
    return panel

# ====================================================
# API de alto nivel
# ====================================================

def build_and_chunk(raw: Dict[str, pd.DataFrame], spec: Dict = CHUNK_SPEC_DEFAULT) -> Dict[str, pd.DataFrame]:
    df = build_full_daily_panel(raw)
    df = finalize_daily_columns(df)

    chunks = split_into_chunks(df, spec=spec)
    for key in ["A","B","C"]:
        chunks[key] = _flag_episode_eligibility_by_chunk(chunks[key])

    # Recalcular merge final tras añadir elegibilidad (conservar core)
    # Usar recombinación incremental para mantener uso de RAM bajo
    tmp = split_into_chunks(pd.concat([chunks["A"], chunks["B"], chunks["C"]], ignore_index=True), spec=spec)
    chunks["merged_core_only"] = tmp["merged_core_only"]
    del tmp
    gc.collect()
    return chunks

# ====================================================
# Agregación semanal (compatibilidad)
# ====================================================

def aggregate_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    tmp = add_week_start(df_daily, "date", "week_start")
    agg = (
        tmp.groupby(["store_nbr","item_nbr","week_start"], as_index=False, sort=False)
           .agg(
               Y_isw=("unit_sales_w","sum"),
               y_isw=("y_tilde","sum"),
               dias_semana=("date","nunique"),
               promo_dias=("onpromotion","sum"),
               delta_isw=("onpromotion","mean")
           )
    )
    # Umbral de tratamiento semanal (si tu config lo usa)
    theta = getattr(config, "THETA_DIAS_PROMO_SEMANA", 0.4)
    agg["Ptilde_isw"] = (agg["delta_isw"] >= float(theta)).astype("int8")
    return agg