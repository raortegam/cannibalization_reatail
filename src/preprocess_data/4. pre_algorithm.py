
# -*- coding: utf-8 -*-
"""
pre_algorithm.py
=================

Módulo de preprocesamiento para construir datasets listos para:
  (1) Control Sintético Generalizado (GSC)
  (2) Meta-learners adaptados a series temporales

Alineado al objetivo del estudio (canibalización en retail) e incluye controles por confusión:
- Festivos jerárquicos (Nacional/Regional/Local) y tipos (Bridge/Additional/Work Day) con lógica de transferencias
- Choques macro y tráfico: petróleo semanal (Ow) y transacciones semanales por tienda (Fsw_log1p)
- Dinámica temporal: rezagos {7,14,28,56} y términos de Fourier anuales (K armónicos)
- Heterogeneidad entre tiendas: dummies compactas {type, cluster, state} + mes
- Disponibilidad y tendencia específicas por SKU vía STL (A_it y T_it), y tendencia agregada por tienda (Q_store_t)

Entradas por defecto:
- ./data/preprocessed_data/pairs_windows.csv         (episodios con ventanas)
- ./data/preprocessed_data/donors_per_victim.csv     (donantes por víctima con rank)
- ./data/raw_data/train.csv                          (panel diario: date, store_nbr, item_nbr, unit_sales, onpromotion)
- ./data/raw_data/items.csv                          (item_nbr, family, class, perishable)
- ./data/raw_data/stores.csv                         (store_nbr, city, state, type, cluster)
- ./data/raw_data/transactions.csv                   (date, store_nbr, transactions)
- ./data/raw_data/oil.csv                            (date, dcoilwtico)
- ./data/raw_data/holidays_events.csv                (date, type, locale, locale_name, description, transferred)

Salidas:
- data/processed/episodes_index.parquet
- data/processed/gsc/<episode_id>.parquet            (panel por episodio para GSC)
- data/processed/gsc/donor_quality.parquet           (log de calidad/filtrado de donantes)
- data/processed/meta/all_units.parquet              (stack para meta-learners)
- data/processed/intermediate/panel_features.parquet (opcional para iteración rápida)

CLI (ejemplo):
    python -m src.preprocess_data.pre_algorithm \
        --episodes ./data/preprocessed_data/pairs_windows.csv \
        --donors   ./data/preprocessed_data/donors_per_victim.csv \
        --raw      ./data/raw_data \
        --out      ./data/processed \
        --top_k 10 --lags "7,14,28,56" --fourier_k 3 \
        --max_donor_promo_share 0.02 --min_availability_share 0.90 \
        --log_level INFO --max_episodes 50 --dry_run --skip_stl

Mejoras prácticas incluidas para iterar rápido:
- --dry_run: ejecuta pipeline pero omite escrituras pesadas (GSC/meta), útil para validar un setup.
- --max_episodes N: procesa sólo los primeros N episodios.
- --skip_stl: fuerza el fallback (media móvil) si quieres acelerar pruebas cuando no haya statsmodels.
- Logging configurable (--log_level).

Autor: generado por GPT-5 Pro.
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd

try:
    # statsmodels es necesario para STL
    from statsmodels.tsa.seasonal import STL
    _HAS_STL = True
except Exception:
    _HAS_STL = False
    warnings.warn("statsmodels no disponible: la disponibilidad (STL) usará un fallback simple.", RuntimeWarning)


# -------------------------------
# Configuración
# -------------------------------

from typing import Union

PathLike = Union[str, Path]

def _to_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _coerce_cfg(cfg: "PrepConfig") -> "PrepConfig":
    # Coaccionar rutas a Path
    cfg.episodes_path = _to_path(cfg.episodes_path)
    cfg.donors_path   = _to_path(cfg.donors_path)
    cfg.raw_dir       = _to_path(cfg.raw_dir)
    cfg.out_dir       = _to_path(cfg.out_dir)
    # Asegurar tipo de lags
    if isinstance(cfg.lags_days, str):
        cfg.lags_days = tuple(int(x.strip()) for x in cfg.lags_days.split(",") if x.strip())
    elif isinstance(cfg.lags_days, (list, tuple)):
        cfg.lags_days = tuple(int(x) for x in cfg.lags_days)
    else:
        cfg.lags_days = tuple([int(cfg.lags_days)]) if cfg.lags_days is not None else (7, 14, 28, 56)
    return cfg

@dataclass
class PrepConfig:
    episodes_path: Path = Path("./data/preprocessed_data/pairs_windows.csv")
    donors_path: Path   = Path("./data/preprocessed_data/donors_per_victim.csv")
    raw_dir: Path       = Path("./data/raw_data")
    out_dir: Path       = Path("./data/processed")
    top_k_donors: int   = 10
    donor_kind: str     = "same_item"   # filtra donors_per_victim por esta clase
    lags_days: Tuple[int, ...] = (7, 14, 28, 56)  # ~ {1,2,4,8} semanas
    fourier_k: int      = 3             # armónicos anuales
    max_donor_promo_share: float = 0.02 # tolerancia de % de días con promo en ventana
    min_availability_share: float = 0.90# fracción mínima de días disponibles en ventana
    save_intermediate: bool = True
    use_stl: bool = True                # si False, fuerza fallback (media móvil)
    drop_city: bool = True              # quitar 'city' antes de guardar para reducir tamaño
    dry_run: bool = False               # no escribir gsc/meta para acelerar pruebas
    max_episodes: Optional[int] = None  # limitar número de episodios en pruebas
    log_level: str = "INFO"             # nivel de logging
    fail_fast: bool = False             # abortar al primer error por episodio


# -------------------------------
# Logging
# -------------------------------

def _setup_logging(level: str = "INFO"):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# -------------------------------
# Utilidades
# -------------------------------

def _ensure_cols(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas {missing} en {name}.")

def _parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def _first_existing(p: Path, candidates: Sequence[str]) -> Path:
    for c in candidates:
        f = p / c
        if f.exists():
            return f
    raise FileNotFoundError(f"No se encontró ninguno de: {candidates} en {p}")

def _episode_id(row: pd.Series) -> str:
    # i_store,i_item,j_store,j_item,treat_start
    ts = pd.to_datetime(row["treat_start"]).strftime("%Y%m%d")
    return f"{int(row['i_store'])}-{int(row['i_item'])}_{int(row['j_store'])}-{int(row['j_item'])}_{ts}"

def _as_categorical(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def _downcast_numeric(df: pd.DataFrame, float_cols: Sequence[str], int_cols: Sequence[str]) -> pd.DataFrame:
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")
    return df


# -------------------------------
# Lectura de datos
# -------------------------------

def load_panel(raw_dir: Path) -> pd.DataFrame:
    """
    Lee el panel item–tienda–día desde train.csv o sales.csv.

    Retorna columnas normalizadas:
        ['date', 'store_nbr', 'item_nbr', 'sales', 'onpromotion']

    Requiere además items.csv para mapear 'family', 'class' y 'perishable'.
    """
    train_path = _first_existing(raw_dir, ["train.csv", "sales.csv"])
    items_path = _first_existing(raw_dir, ["items.csv"])

    logging.info(f"Cargando panel desde {train_path}")
    df = pd.read_csv(train_path)
    # Normalizar columnas
    colmap = {}
    if "date" not in df.columns:
        raise ValueError("El panel requiere columna 'date'.")
    if "store_nbr" not in df.columns:
        raise ValueError("El panel requiere columna 'store_nbr'.")
    if "item_nbr" not in df.columns and "item_id" in df.columns:
        colmap["item_id"] = "item_nbr"
    if "unit_sales" in df.columns and "sales" not in df.columns:
        colmap["unit_sales"] = "sales"
    if "units" in df.columns and "sales" not in df.columns:
        colmap["units"] = "sales"
    if "onpromotion" not in df.columns and "promo" in df.columns:
        colmap["promo"] = "onpromotion"

    if colmap:
        df = df.rename(columns=colmap)

    _ensure_cols(df, ["date", "store_nbr", "item_nbr", "sales"], "train")
    if "onpromotion" not in df.columns:
        df["onpromotion"] = 0.0

    df["date"] = _parse_date(df["date"])
    df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0.0)

    # Items meta
    items = pd.read_csv(items_path)
    _ensure_cols(items, ["item_nbr"], "items")
    items = items.rename(columns={"family": "family_name"})
    items = _downcast_numeric(items, float_cols=[], int_cols=["item_nbr", "class", "perishable"])
    items = _as_categorical(items, ["family_name"])

    df = df.merge(items, on="item_nbr", how="left")
    df = _downcast_numeric(df, float_cols=["sales", "onpromotion"], int_cols=["store_nbr", "item_nbr", "class", "perishable"])
    logging.info(f"Panel cargado: {df.shape[0]:,} filas")
    return df


def load_stores(raw_dir: Path) -> pd.DataFrame:
    stores_path = _first_existing(raw_dir, ["stores.csv"])
    stores = pd.read_csv(stores_path)
    _ensure_cols(stores, ["store_nbr", "city", "state", "type", "cluster"], "stores")
    stores = _as_categorical(stores, ["city", "state", "type", "cluster"])
    logging.info(f"Tiendas cargadas: {stores.shape[0]}")
    return stores


def load_transactions(raw_dir: Path) -> pd.DataFrame:
    tr_path = _first_existing(raw_dir, ["transactions.csv"])
    tr = pd.read_csv(tr_path)
    _ensure_cols(tr, ["date", "store_nbr", "transactions"], "transactions")
    tr["date"] = _parse_date(tr["date"])
    tr["transactions"] = pd.to_numeric(tr["transactions"], errors="coerce").fillna(0.0)
    return tr


def load_oil(raw_dir: Path) -> pd.DataFrame:
    oil_path = _first_existing(raw_dir, ["oil.csv"])
    oil = pd.read_csv(oil_path)
    _ensure_cols(oil, ["date"], "oil")
    oil = oil.rename(columns={"dcoilwtico": "oil_price"})
    oil["date"] = _parse_date(oil["date"])
    oil["oil_price"] = pd.to_numeric(oil["oil_price"], errors="coerce")
    oil["oil_price"] = oil["oil_price"].ffill().bfill()
    return oil


def load_holidays(raw_dir: Path) -> pd.DataFrame:
    hol_path = _first_existing(raw_dir, ["holidays_events.csv"])
    hol = pd.read_csv(hol_path)
    _ensure_cols(hol, ["date", "type", "locale", "locale_name", "description"], "holidays_events")
    if "transferred" not in hol.columns:
        hol["transferred"] = False
    hol["date"] = _parse_date(hol["date"])
    hol["transferred"] = hol["transferred"].astype(bool)
    hol["type"] = hol["type"].astype("category")
    hol["locale"] = hol["locale"].astype("category")
    hol["locale_name"] = hol["locale_name"].astype("category")
    return hol


# -------------------------------
# Controles por confusión
# -------------------------------

def _effective_holidays(hol: pd.DataFrame) -> pd.DataFrame:
    """
    Construye feriados efectivos (post-transfer) por la lógica del dataset de Favorita.
    """
    hol = hol.copy()

    # Marcar 'originales' movidos (type == 'Transfer')
    transfer_orig = hol.loc[hol["type"].astype(str).str.lower().eq("transfer")]
    originals = set(transfer_orig["description"].astype(str).unique().tolist())

    # Filas transferidas 'nuevas'
    transferred_new = hol.loc[hol["transferred"] == True]

    # Conservar: filas transferidas nuevas, y filas sin transferencia ni tipo 'Transfer'
    keep_mask = (hol["type"].astype(str).str.lower() != "transfer")
    if not transferred_new.empty:
        trans_desc = set(transferred_new["description"].astype(str).unique().tolist())
        keep_mask &= (~hol["description"].astype(str).isin(originals)) | hol["transferred"]

    eff = hol.loc[keep_mask].copy()
    eff = eff.drop_duplicates(subset=["date", "type", "locale", "locale_name", "description", "transferred"])
    return eff


def build_holiday_controls(hol: pd.DataFrame, stores: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Genera indicadores diarios HNat, HReg, HLoc por tienda y flags Bridge/Additional/WorkDay.
    """
    hol = _effective_holidays(hol)

    # Expandir por tienda según locale
    nat = hol.loc[hol["locale"].astype(str).str.lower().eq("national")].copy()
    reg = hol.loc[hol["locale"].astype(str).str.lower().eq("regional")].copy()
    loc = hol.loc[hol["locale"].astype(str).str.lower().eq("local")].copy()

    # Nacional: cruza por todas las tiendas
    nat_flag = nat[["date", "type"]].copy()
    nat_flag["HNat"] = 1.0
    nat_flag = nat_flag.groupby("date", as_index=False)["HNat"].max()

    # Regional: por state
    reg = reg.rename(columns={"locale_name": "state"})
    reg_flag = reg[["date", "state", "type"]].copy()
    reg_flag["HReg"] = 1.0
    reg_flag = reg_flag.groupby(["date", "state"], as_index=False)["HReg"].max()

    # Local: por city
    loc = loc.rename(columns={"locale_name": "city"})
    loc_flag = loc[["date", "city", "type"]].copy()
    loc_flag["HLoc"] = 1.0
    loc_flag = loc_flag.groupby(["date", "city"], as_index=False)["HLoc"].max()

    # Tipos especiales por fecha (para todos)
    type_by_date = hol[["date", "type"]].copy()
    type_by_date["is_bridge"] = type_by_date["type"].astype(str).str.lower().eq("bridge").astype(float)
    type_by_date["is_additional"] = type_by_date["type"].astype(str).str.lower().eq("additional").astype(float)
    type_by_date["is_work_day"] = type_by_date["type"].astype(str).str.lower().eq("work day").astype(float)
    type_flags = type_by_date.groupby("date", as_index=False)[["is_bridge", "is_additional", "is_work_day"]].max()

    return nat_flag, reg_flag, loc_flag, type_flags


def build_weekly_proxies(transactions: pd.DataFrame, oil: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye proxies semanales:
      - Fsw = sum(transactions) por tienda-semana (log1p)
      - Ow  = promedio semanal del precio del petróleo
    """
    tr = transactions.copy()
    tr["year"] = tr["date"].dt.isocalendar().year.astype(int)
    tr["week"] = tr["date"].dt.isocalendar().week.astype(int)
    tr["year_week"] = tr["year"].astype(str) + "-" + tr["week"].astype(str).str.zfill(2)

    trw = tr.groupby(["year_week", "store_nbr"], as_index=False)["transactions"].sum()
    trw["Fsw_log1p"] = np.log1p(trw["transactions"].clip(lower=0.0))
    trw = trw[["year_week", "store_nbr", "Fsw_log1p"]]

    oi = oil.copy()
    oi["year"] = oi["date"].dt.isocalendar().year.astype(int)
    oi["week"] = oi["date"].dt.isocalendar().week.astype(int)
    oi["year_week"] = oi["year"].astype(str) + "-" + oi["week"].astype(str).str.zfill(2)
    oiw = oi.groupby("year_week", as_index=False)["oil_price"].mean()
    oiw = oiw.rename(columns={"oil_price": "Ow"})

    return trw, oiw


def add_temporal_features(panel: pd.DataFrame, lags_days: Sequence[int], fourier_k: int) -> pd.DataFrame:
    """
    Agrega rezagos por unidad (store,item) y términos de Fourier anuales al panel.
    """
    df = panel.copy()

    # Fourier anual
    day_of_year = df["date"].dt.dayofyear.astype(float)
    period = 365.25
    for k in range(1, fourier_k + 1):
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / period)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / period)

    # Rezagos por (store_nbr, item_nbr)
    df = df.sort_values(["store_nbr", "item_nbr", "date"])
    grouped = df.groupby(["store_nbr", "item_nbr"], group_keys=False)
    for L in lags_days:
        df[f"lag_{L}d"] = grouped["sales"].shift(L)
    return df


def _stl_trend_and_availability(y: pd.Series, period: int = 7, robust: bool = True, use_stl: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica STL sobre una serie y retorna (trend, availability_bool) con criterio T_t > 0.5.
    Fallback: si STL no está disponible o la serie es muy corta, usa media móvil.
    """
    vals = y.to_numpy(dtype=float)
    n = len(vals)
    if (not use_stl) or n < max(14, period * 2) or not _HAS_STL:
        trend = pd.Series(vals).rolling(window=7, min_periods=1, center=True).mean().to_numpy()
    else:
        try:
            stl = STL(vals, period=period, robust=robust)
            res = stl.fit()
            trend = res.trend
        except Exception:
            trend = pd.Series(vals).rolling(window=7, min_periods=1, center=True).mean().to_numpy()
    avail = (trend > 0.5).astype(float)
    return trend, avail


def add_availability_and_store_trend(panel: pd.DataFrame, use_stl: bool) -> pd.DataFrame:
    """
    Calcula:
      - T_it y A_it por (store,item) vía STL semanal (period=7) o fallback
      - Q_store_t: STL sobre ventas agregadas por tienda (period=7)
    """
    df = panel.copy()
    df = df.sort_values(["store_nbr", "item_nbr", "date"])

    # Tendencia y disponibilidad por unidad
    trends = []
    avails = []
    for (s, it), g in df.groupby(["store_nbr", "item_nbr"]):
        t, a = _stl_trend_and_availability(g["sales"], period=7, robust=True, use_stl=use_stl)
        trends.append(pd.Series(t, index=g.index))
        avails.append(pd.Series(a, index=g.index))
    df["trend_T"] = pd.concat(trends).sort_index()
    df["available_A"] = pd.concat(avails).sort_index()

    # Tendencia por tienda (agregado de ventas diarias)
    store_sum = df.groupby(["store_nbr", "date"], as_index=False)["sales"].sum()
    store_sum = store_sum.sort_values(["store_nbr", "date"])
    store_trend_parts = []
    for s, g in store_sum.groupby("store_nbr"):
        t, _ = _stl_trend_and_availability(g["sales"], period=7, robust=True, use_stl=use_stl)
        part = g[["store_nbr", "date"]].copy()
        part["Q_store_trend"] = t
        store_trend_parts.append(part)
    store_trend = pd.concat(store_trend_parts, ignore_index=True)

    df = df.merge(store_trend, on=["store_nbr", "date"], how="left")
    return df


def add_store_metadata(panel: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """
    Añade metadatos de tienda (type, cluster, state) y dummies mínimas (one-hot compacta).
    """
    df = panel.merge(stores[["store_nbr", "type", "cluster", "state", "city"]], on="store_nbr", how="left")
    # Dummies compactas
    for col in ["type", "cluster", "state"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            if dummies.shape[1] > 0:
                dcols = sorted(dummies.columns)
                dummies = dummies[dcols[1:]]  # efecto de referencia
                df = pd.concat([df, dummies], axis=1)
    # Interacción suave con mes
    df["month"] = df["date"].dt.month.astype(np.int16)
    return df


# -------------------------------
# Construcción del universo a procesar
# -------------------------------

def load_episodes(episodes_path: Path) -> pd.DataFrame:
    eps = pd.read_csv(episodes_path)
    _ensure_cols(eps, ["i_store", "i_item", "j_store", "j_item", "pre_start", "treat_start", "post_start", "post_end"], "pairs_windows")
    for col in ["pre_start", "treat_start", "post_start", "post_end"]:
        eps[col] = _parse_date(eps[col])
    eps["episode_id"] = eps.apply(_episode_id, axis=1)
    return eps


def load_donors(donors_path: Path, donor_kind: Optional[str] = None) -> pd.DataFrame:
    dons = pd.read_csv(donors_path)
    _ensure_cols(dons, ["j_store", "j_item", "donor_store", "donor_item", "donor_kind", "rank"], "donors_per_victim")
    if donor_kind is not None:
        dons = dons.loc[dons["donor_kind"] == donor_kind].copy()
    dons = dons.sort_values(["j_store", "j_item", "rank"])
    return dons


def universe_from_episodes_and_donors(episodes: pd.DataFrame, donors: pd.DataFrame, top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Devuelve tres dataframes con pares (store,item) a considerar:
      - victims: víctimas j de todos los episodios
      - donors_top: donantes top_k por víctima (filtrados por rank)
      - cannibals: caníbales i de todos los episodios
    """
    victims = episodes[["j_store", "j_item"]].drop_duplicates().rename(columns={"j_store": "store_nbr", "j_item": "item_nbr"})
    cannibals = episodes[["i_store", "i_item"]].drop_duplicates().rename(columns={"i_store": "store_nbr", "i_item": "item_nbr"})
    donors_top = (donors
                  .sort_values(["j_store", "j_item", "rank"])
                  .groupby(["j_store", "j_item"], as_index=False)
                  .head(top_k))
    donors_top = donors_top.rename(columns={"donor_store": "store_nbr", "donor_item": "item_nbr"})
    donors_top = donors_top[["store_nbr", "item_nbr", "j_store", "j_item", "rank"]]
    return victims, donors_top, cannibals


def compute_global_date_window(episodes: pd.DataFrame, max_lag: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    min_date = (episodes["pre_start"].min() - pd.Timedelta(days=max_lag)).normalize()
    max_date = episodes["post_end"].max().normalize()
    return min_date, max_date


# -------------------------------
# Filtrado del panel y construcción de features
# -------------------------------

def filter_panel_to_universe(panel: pd.DataFrame,
                             victims: pd.DataFrame,
                             donors_top: pd.DataFrame,
                             date_min: pd.Timestamp,
                             date_max: pd.Timestamp) -> pd.DataFrame:
    """
    Filtra el panel a víctimas + donantes top_k y fechas en [date_min, date_max].
    Implementación optimizada vía merge en lugar de .isin sobre MultiIndex.
    """
    key_units = pd.concat([
        victims[["store_nbr", "item_nbr"]],
        donors_top[["store_nbr", "item_nbr"]].drop_duplicates()
    ], ignore_index=True).drop_duplicates()

    p = panel.loc[panel["date"].between(date_min, date_max)].copy()
    p = p.merge(key_units.assign(_keep=1), on=["store_nbr", "item_nbr"], how="inner")
    p = p.drop(columns=["_keep"], errors="ignore")

    p = _downcast_numeric(p, float_cols=["sales", "onpromotion"], int_cols=["store_nbr", "item_nbr"])
    logging.info(f"Panel filtrado (universo mínimo): {p.shape[0]:,} filas, {key_units.shape[0]} unidades")
    return p


def attach_controls(panel: pd.DataFrame,
                    stores: pd.DataFrame,
                    transactions: pd.DataFrame,
                    oil: pd.DataFrame,
                    holidays_nat: pd.DataFrame,
                    holidays_reg: pd.DataFrame,
                    holidays_loc: pd.DataFrame,
                    holidays_types: pd.DataFrame) -> pd.DataFrame:
    """
    Une todos los controles al panel diario filtrado.
    """
    df = panel.copy()

    # Semanas ISO para joins semanales
    df["year"] = df["date"].dt.isocalendar().year.astype(int)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["year_week"] = df["year"].astype(str) + "-" + df["week"].astype(str).str.zfill(2)

    # Holidays por tienda y fecha
    df = df.merge(holidays_nat, on="date", how="left")
    df = df.merge(stores[["store_nbr", "state", "city"]], on="store_nbr", how="left")
    df = df.merge(holidays_reg, on=["date", "state"], how="left")
    df = df.merge(holidays_loc, on=["date", "city"], how="left")
    df = df.merge(holidays_types, on="date", how="left")
    for c in ["HNat", "HReg", "HLoc", "is_bridge", "is_additional", "is_work_day"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Proxies semanales
    trw, oiw = build_weekly_proxies(transactions, oil)
    df = df.merge(trw, on=["year_week", "store_nbr"], how="left")
    df = df.merge(oiw, on="year_week", how="left")

    # Fill & tipos
    df["Fsw_log1p"] = df["Fsw_log1p"].fillna(method="ffill").fillna(0.0)
    df["Ow"] = df["Ow"].fillna(method="ffill").fillna(0.0)

    # Limpiar columnas temporales
    df = df.drop(columns=["year", "week"], errors="ignore")
    return df


# -------------------------------
# Construcción de datasets GSC y Meta
# -------------------------------

def _window_mask(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return df["date"].between(start, end)

def _compute_availability_share(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["available_A"].mean())

def _compute_promo_share(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float((df["onpromotion"] > 0).mean())


def build_gsc_and_meta_for_episode(ep_row: pd.Series,
                                   base_panel: pd.DataFrame,
                                   donors_map: pd.DataFrame,
                                   cfg: PrepConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Construye para un episodio:
      - panel GSC long con víctima + donantes elegidos y etiquetas de tratamiento
      - panel Meta long (víctimas y donantes con D_t)
    Aplica filtros de calidad de donantes (promos superpuestas y disponibilidad).
    """
    i_store, i_item = int(ep_row["i_store"]), int(ep_row["i_item"])
    j_store, j_item = int(ep_row["j_store"]), int(ep_row["j_item"])
    pre_start = ep_row["pre_start"]
    treat_start = ep_row["treat_start"]
    post_end = ep_row["post_end"]

    episode_id = ep_row["episode_id"]

    # Unidades de interés
    dons = donors_map.loc[(donors_map["j_store"] == j_store) & (donors_map["j_item"] == j_item)].copy()
    dons = dons.sort_values("rank").drop_duplicates(subset=["store_nbr", "item_nbr"]).head(cfg.top_k_donors)
    donors_units = list(map(tuple, dons[["store_nbr", "item_nbr"]].to_numpy()))
    units = [(j_store, j_item)] + donors_units

    # Subpanel ventana [pre_start, post_end] para esas unidades
    sub = base_panel.loc[
        base_panel["date"].between(pre_start, post_end) &
        base_panel.set_index(["store_nbr", "item_nbr"]).index.isin(units)
    ].copy()

    # Filtros de calidad por donante en la ventana
    g_list = []
    donor_quality = []
    for (s, it), g in sub.groupby(["store_nbr", "item_nbr"]):
        is_victim = (s == j_store) and (it == j_item)
        promo_share = _compute_promo_share(g)
        avail_share = _compute_availability_share(g)
        keep = True
        reasons = []
        if (not is_victim) and (promo_share > cfg.max_donor_promo_share):
            keep = False
            reasons.append(f"promo_share={promo_share:.3f}>{cfg.max_donor_promo_share}")
        if (not is_victim) and (avail_share < cfg.min_availability_share):
            keep = False
            reasons.append(f"avail_share={avail_share:.3f}<{cfg.min_availability_share}")
        donor_quality.append({
            "episode_id": episode_id,
            "store_nbr": s, "item_nbr": it,
            "is_victim": is_victim,
            "promo_share": promo_share,
            "avail_share": avail_share,
            "keep": keep,
            "reason": "; ".join(reasons)
        })
        if keep:
            g_list.append(g)

    if not g_list:
        # Si todos fueron descartados, mantener al menos la víctima
        g_list = [sub.loc[(sub["store_nbr"] == j_store) & (sub["item_nbr"] == j_item)]]

    subf = pd.concat(g_list, ignore_index=True)

    # Etiquetas de tratamiento para GSC/Meta
    subf["unit_id"] = subf["store_nbr"].astype(str) + ":" + subf["item_nbr"].astype(str)
    subf["treated_unit"] = ((subf["store_nbr"] == j_store) & (subf["item_nbr"] == j_item)).astype(int)
    subf["treated_time"] = (subf["date"] >= treat_start).astype(int)
    subf["D"] = ((subf["treated_unit"] == 1) & (subf["treated_time"] == 1)).astype(int)

    # Paneles
    gsc_panel = subf.copy()
    meta_panel = subf.copy()

    # Métricas
    n_donors_kept = int(meta_panel.loc[meta_panel["treated_unit"] == 0, ["store_nbr", "item_nbr"]]
                        .drop_duplicates().shape[0])

    meta_info = {
        "episode_id": episode_id,
        "victim": {"store_nbr": j_store, "item_nbr": j_item},
        "cannibal": {"store_nbr": i_store, "item_nbr": i_item},
        "n_donors_input": len(donors_units),
        "n_donors_kept": n_donors_kept,
        "quality": donor_quality
    }
    return gsc_panel, meta_panel, meta_info


def prepare_datasets(cfg: PrepConfig) -> None:
    """
    Orquesta el pipeline de preprocesamiento end-to-end.
    """
    _setup_logging(cfg.log_level)
    cfg = _coerce_cfg(cfg)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "gsc").mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "intermediate").mkdir(parents=True, exist_ok=True)

    # 1) Lectura de entradas
    logging.info("Leyendo episodios y donantes...")
    episodes = load_episodes(cfg.episodes_path)
    donors = load_donors(cfg.donors_path, donor_kind=cfg.donor_kind)

    if cfg.max_episodes is not None:
        episodes = episodes.head(int(cfg.max_episodes)).copy()
        logging.info(f"Limite de episodios activado: {episodes.shape[0]} episodios")

    # 2) Universo mínimo y ventana global
    victims, donors_top, cannibals = universe_from_episodes_and_donors(episodes, donors, cfg.top_k_donors)
    date_min, date_max = compute_global_date_window(episodes, max_lag=max(cfg.lags_days))
    logging.info(f"Ventana global: {date_min.date()} → {date_max.date()}")

    # 3) Panel filtrado a universo
    panel_all = load_panel(cfg.raw_dir)
    stores = load_stores(cfg.raw_dir)
    transactions = load_transactions(cfg.raw_dir)
    oil = load_oil(cfg.raw_dir)
    holidays = load_holidays(cfg.raw_dir)

    panel = filter_panel_to_universe(panel_all, victims, donors_top, date_min, date_max)

    # 4) Controles: holidays + proxies semanales + temporales + disponibilidad + metadatos
    logging.info("Construyendo controles y features...")
    hol_nat, hol_reg, hol_loc, hol_types = build_holiday_controls(holidays, stores)
    panel = attach_controls(panel, stores, transactions, oil, hol_nat, hol_reg, hol_loc, hol_types)
    panel = add_temporal_features(panel, lags_days=cfg.lags_days, fourier_k=cfg.fourier_k)
    panel = add_availability_and_store_trend(panel, use_stl=(cfg.use_stl and _HAS_STL))
    panel = add_store_metadata(panel, stores)

    if cfg.drop_city and "city" in panel.columns:
        panel = panel.drop(columns=["city"], errors="ignore")

    # 5) Persistir intermedio filtrado (opcional) para iterar rápido
    if cfg.save_intermediate:
        inter_path = cfg.out_dir / "intermediate" / "panel_features.parquet"
        panel.to_parquet(inter_path, index=False)
        logging.info(f"Intermedio guardado: {inter_path}")

    # 6) Construcción por episodio
    meta_panels = []
    index_rows = []
    episode_infos: List[Dict] = []

    logging.info("Procesando episodios...")
    for idx, (_, ep) in enumerate(episodes.iterrows(), start=1):
        try:
            gsc_panel, meta_panel, meta_info = build_gsc_and_meta_for_episode(ep, panel, donors_top, cfg)

            # Guardar por episodio (GSC)
            if not cfg.dry_run:
                out_gsc = cfg.out_dir / "gsc" / f"{ep['episode_id']}.parquet"
                gsc_panel.to_parquet(out_gsc, index=False)

            # Acumular para Meta
            meta_panels.append(meta_panel)

            # Índice de episodio
            index_rows.append({
                "episode_id": ep["episode_id"],
                "i_store": ep["i_store"],
                "i_item": ep["i_item"],
                "j_store": ep["j_store"],
                "j_item": ep["j_item"],
                "pre_start": ep["pre_start"],
                "treat_start": ep["treat_start"],
                "post_start": ep["post_start"],
                "post_end": ep["post_end"],
                "n_gsc_rows": len(gsc_panel),
                "n_meta_rows": len(meta_panel),
                "n_donors_input": meta_info["n_donors_input"],
                "n_donors_kept": meta_info["n_donors_kept"]
            })
            episode_infos.append(meta_info)

            logging.info(f"[{idx}/{episodes.shape[0]}] Ep {ep['episode_id']} | donors kept: {meta_info['n_donors_kept']}/{meta_info['n_donors_input']} | rows: gsc={len(gsc_panel):,}, meta={len(meta_panel):,}")

        except Exception as e:
            logging.exception(f"Error en episodio {ep.get('episode_id','?')}: {e}")
            if cfg.fail_fast:
                raise

    # Guardar índice
    ep_index = pd.DataFrame(index_rows)
    ep_index_path = cfg.out_dir / "episodes_index.parquet"
    ep_index.to_parquet(ep_index_path, index=False)
    logging.info(f"Índice de episodios guardado: {ep_index_path} ({ep_index.shape[0]} filas)")

    # Guardar panel meta global
    if not cfg.dry_run and meta_panels:
        meta_all = pd.concat(meta_panels, ignore_index=True)
        meta_all_path = cfg.out_dir / "meta" / "all_units.parquet"
        meta_all.to_parquet(meta_all_path, index=False)
        logging.info(f"Panel meta global guardado: {meta_all_path} ({meta_all.shape[0]:,} filas)")
    elif cfg.dry_run:
        logging.info("DRY-RUN: se omitió la escritura de GSC y META.")

    # Guardar log de calidad de donantes por episodio
    quality_rows = []
    for info in episode_infos:
        for q in info["quality"]:
            row = dict(q)  # copia defensiva
            # evita duplicar episode_id
            row.setdefault("episode_id", info["episode_id"])
            quality_rows.append(row)
    if quality_rows:
        donor_quality_df = pd.DataFrame(quality_rows)
        donor_quality_path = cfg.out_dir / "gsc" / "donor_quality.parquet"
        donor_quality_df.to_parquet(donor_quality_path, index=False)
        logging.info(f"Log de calidad de donantes guardado: {donor_quality_path} ({donor_quality_df.shape[0]:,} filas)")


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> PrepConfig:
    p = argparse.ArgumentParser(description="Preprocesamiento para GSC y Meta-learners (retail canibalización).")
    p.add_argument("--episodes", type=str, default=str(PrepConfig.episodes_path), help="Ruta a pairs_windows.csv")
    p.add_argument("--donors", type=str, default=str(PrepConfig.donors_path), help="Ruta a donors_per_victim.csv")
    p.add_argument("--raw", type=str, default=str(PrepConfig.raw_dir), help="Directorio con datos raw")
    p.add_argument("--out", type=str, default=str(PrepConfig.out_dir), help="Directorio de salida")

    p.add_argument("--top_k", type=int, default=PrepConfig.top_k_donors, help="Top K donantes por víctima")
    p.add_argument("--donor_kind", type=str, default=PrepConfig.donor_kind, help="Filtro por tipo de donante")
    p.add_argument("--lags", type=str, default="7,14,28,56", help="Rezagos en días separados por coma")
    p.add_argument("--fourier_k", type=int, default=PrepConfig.fourier_k, help="Número de armónicos anuales")

    p.add_argument("--max_donor_promo_share", type=float, default=PrepConfig.max_donor_promo_share, help="Máx % días en promo permitido a donantes")
    p.add_argument("--min_availability_share", type=float, default=PrepConfig.min_availability_share, help="Mín fracción de días disponibles en ventana")

    p.add_argument("--no_intermediate", action="store_true", help="No guardar intermedios")
    p.add_argument("--skip_stl", action="store_true", help="Forzar fallback simple (sin STL)")
    p.add_argument("--keep_city", action="store_true", help="Conservar columna 'city' al guardar")
    p.add_argument("--dry_run", action="store_true", help="No escribir GSC/meta (rápido para pruebas)")
    p.add_argument("--max_episodes", type=int, default=None, help="Limitar número de episodios a procesar")
    p.add_argument("--log_level", type=str, default=PrepConfig.log_level, help="Nivel de logs (DEBUG, INFO, WARNING, ...)")
    p.add_argument("--fail_fast", action="store_true", help="Abortar al primer error por episodio")

    args = p.parse_args()
    cfg = PrepConfig(
        episodes_path=Path(args.episodes),
        donors_path=Path(args.donors),
        raw_dir=Path(args.raw),
        out_dir=Path(args.out),
        top_k_donors=args.top_k,
        donor_kind=args.donor_kind,
        lags_days=tuple(int(x.strip()) for x in args.lags.split(",") if x.strip()),
        fourier_k=args.fourier_k,
        max_donor_promo_share=args.max_donor_promo_share,
        min_availability_share=args.min_availability_share,
        save_intermediate=(not args.no_intermediate),
        use_stl=(not args.skip_stl),
        drop_city=(not args.keep_city),
        dry_run=args.dry_run,
        max_episodes=args.max_episodes,
        log_level=args.log_level,
        fail_fast=args.fail_fast
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    prepare_datasets(cfg)