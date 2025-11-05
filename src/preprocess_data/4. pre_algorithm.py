# -*- coding: utf-8 -*-
"""
pre_algorithm.py
=================

Módulo de preprocesamiento para construir datasets listos para:
  (1) Control Sintético Generalizado (GSC) -- opcional y sólo para un subconjunto de episodios
  (2) Meta-learners adaptados a series temporales (modo masivo y rápido)

Novedades clave
---------------
- Modo **meta_only**: genera el stack de ventanas y features para Meta sin requerir donantes.
- Selección consistente de episodios para evaluación/contrafactual:
  * --gsc_eval_ids (CSV con episode_id) o
  * --gsc_eval_n + --gsc_eval_selection {head,random} (+ --gsc_eval_seed)
- Salida separada para meta-training: **out_meta_dir** (default: ./data/processed_meta).
- Control fino de qué unidades incluye el stack de Meta:
  * --meta_units {victims_only,victims_plus_donors}
- Si no hay donantes, se desactiva automáticamente donor-blend (sc_hat).

Confounders y señales (activables con flags; por defecto activas):
- DOW (dummies), paydays/fin de mes
- Proxies semanales (tráfico tienda y petróleo)
- Proxy regional (tráfico semanal por estado excluyendo tienda)
- Presión promocional (store,class) + versión excluyente del SKU + lags
- Índice de clase excluyendo el SKU + lags
- Intermitencia (zero_streak, ADI, CV²)
- Donor-blend (sc_hat) sólo si hay donantes para el episodio y PRE suficiente

Entradas por defecto:
- ./data/processed_data/pairs_windows.csv
- ./data/processed_data/donors_per_victim.csv (opcional en modo meta_only)
- ./data/raw_data/{train,items,stores,transactions,oil,holidays_events}.csv

Salidas:
- data/processed/episodes_index.parquet
- data/processed/gsc/<episode_id>.parquet               (sólo si el episodio está en el set de evaluación)
- data/processed/gsc/donor_quality.parquet              (si hubo GSC)
- data/processed/meta/all_units.parquet                 (compatibilidad; opcional)
- data/processed_meta/windows.parquet                   (stack TS para Meta)
- data/processed_meta/episodes_index.parquet            (índice de episodios para Meta)
- data/processed/intermediate/panel_features.parquet    (opcional)

CLI (ejemplos):
  # Generar masivo para Meta (sin GSC):
  python -m src.preprocess_data.pre_algorithm --episodes ./data/processed_data/pairs_windows.csv \
      --raw ./data/raw_data --out ./data/processed --out_meta ./data/processed_meta --meta_only

  # Igual que arriba, pero además calcular GSC para 10 episodios tomados al comienzo:
  python -m src.preprocess_data.pre_algorithm --episodes ./data/processed_data/pairs_windows.csv \
      --donors ./data/processed_data/donors_per_victim.csv \
      --raw ./data/raw_data --out ./data/processed --out_meta ./data/processed_meta \
      --meta_only --gsc_eval_n 10 --gsc_eval_selection head

Cambios aplicados (2025-11-03)
------------------------------
- [FIX] El filtrado del `base_panel` **incluye donantes** cuando habrá GSC, independientemente de `meta_units`.
- [FIX] `cluster_id` ahora es único por episodio: `i_store_i_item_j_store_j_item_fecha`.
- [NEW] En el *stack* de Meta se escriben `y_raw` (= `sales`) y `y_log1p` (= `np.log1p(sales)`).
- [NEW] Chequeo opcional de coherencia Observado Meta vs GSC (warning/AssertionError).

Autor original: GPT-5 Pro. Cambios: GPT-5 Pro.
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Union

import numpy as np
import pandas as pd

# STL para tendencia/disponibilidad
try:
    from statsmodels.tsa.seasonal import STL
    _HAS_STL = True
except Exception:
    _HAS_STL = False
    warnings.warn("statsmodels no disponible: la disponibilidad (STL) usará un fallback simple.", RuntimeWarning)

# Ridge para donor-blend (con fallback propio si no hay sklearn)
try:
    from sklearn.linear_model import Ridge as _SklearnRidge
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
    _SklearnRidge = None
    warnings.warn("scikit-learn no disponible: se usará Ridge de respaldo en NumPy.", RuntimeWarning)

# -------------------------------
# Configuración
# -------------------------------

PathLike = Union[str, Path]

def _to_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _coerce_cfg(cfg: "PrepConfig") -> "PrepConfig":
    # Coaccionar rutas a Path
    cfg.episodes_path = _to_path(cfg.episodes_path)
    cfg.raw_dir       = _to_path(cfg.raw_dir)
    cfg.out_dir       = _to_path(cfg.out_dir)
    cfg.out_meta_dir  = _to_path(cfg.out_meta_dir)
    # donors_path puede ser opcional en meta_only
    cfg.donors_path   = _to_path(cfg.donors_path) if cfg.donors_path else None
    # Asegurar tipo de lags
    if isinstance(cfg.lags_days, str):
        cfg.lags_days = tuple(int(x.strip()) for x in cfg.lags_days.split(",") if x.strip())
    elif isinstance(cfg.lags_days, (list, tuple)):
        cfg.lags_days = tuple(int(x) for x in cfg.lags_days)
    else:
        cfg.lags_days = tuple([int(cfg.lags_days)]) if cfg.lags_days is not None else (7, 14, 28, 56)
    # Normalizar meta_units
    cfg.meta_units = (cfg.meta_units or "victims_only").strip().lower()
    if cfg.meta_units not in {"victims_only", "victims_plus_donors"}:
        cfg.meta_units = "victims_only"
    return cfg

@dataclass
class PrepConfig:
    # Rutas
    episodes_path: Path = Path("./.data/processed_data/pairs_windows.csv")
    donors_path: Optional[Path] = Path(".data/processed_data/donors_per_victim.csv")  # opcional en meta_only
    raw_dir: Path       = Path("./.data/raw_data")
    out_dir: Path       = Path("./.data/processed")
    out_meta_dir: Path  = Path("./.data/processed_meta")   # NUEVO: ruta separada para meta-training

    # Selección de donantes (si se usan)
    top_k_donors: int   = 10
    donor_kind: str     = "same_item"

    # Temporalidad
    lags_days: Tuple[int, ...] = (7, 14, 28, 56)
    fourier_k: int      = 3

    # Filtros de calidad donantes (sólo aplica si hay donantes)
    max_donor_promo_share: float = 0.02
    min_availability_share: float = 0.90

    # Guardado / ejecución
    save_intermediate: bool = True
    use_stl: bool = True
    drop_city: bool = True
    dry_run: bool = False
    max_episodes: Optional[int] = None
    log_level: str = "INFO"
    fail_fast: bool = False

    # ---- Nuevos flags de modo ----
    meta_only: bool = False                           # procesa masivo para Meta sin GSC
    meta_units: str = "victims_only"                  # {"victims_only","victims_plus_donors"}

    # Selección de episodios para GSC (subconjunto)
    gsc_eval_ids_path: Optional[Path] = None          # CSV con columna episode_id
    gsc_eval_n: Optional[int] = None                  # si se especifica, toma N episodios
    gsc_eval_selection: str = "head"                  # {"head","random"}
    gsc_eval_seed: int = 42

    # ---- Señales extra (maestra + subflags) ----
    confounds_plus: bool = True
    add_dow: bool = True
    add_paydays: bool = True
    use_regional_proxy: bool = True
    add_promo_pressure: bool = True
    add_class_index_excl: bool = True
    add_intermitency_feats: bool = True
    add_donor_blend_sc: bool = True
    donor_blend_alpha: float = 1.0
    donor_blend_min_pre_T: int = 10

    # Compatibilidad: además de escribir a out_meta_dir/windows.parquet,
    # se puede duplicar la salida en out_dir/meta/all_units.parquet
    write_legacy_meta_copy: bool = True

    # ---- Verificación de coherencia Observado (Meta vs GSC) ----
    check_observed_consistency: bool = True
    strict_obs_check: bool = False


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

def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Ajusta Ridge con regularización L2 (fit_intercept=False) y devuelve coeficientes.
    Usa scikit-learn si está disponible; de lo contrario, resuelve (X'X + αI)w = X'y.
    """
    if _HAS_SKLEARN:
        try:
            model = _SklearnRidge(alpha=float(alpha), fit_intercept=False, positive=False)
            model.fit(X, y)
            return model.coef_.astype(float)
        except Exception:
            pass
    XtX = X.T @ X
    n = XtX.shape[0]
    XtX_reg = XtX + (alpha * np.eye(n))
    Xty = X.T @ y
    try:
        w = np.linalg.solve(XtX_reg, Xty)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(XtX_reg) @ Xty
    return w.astype(float)

# ---- NUEVO: utilidades anti-duplicados / merges seguros ----

def _assert_unique(df: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    """Lanza si df no es único por keys; muestra algunos casos para depurar."""
    dup_mask = df.duplicated(keys, keep=False)
    if dup_mask.any():
        ex = (df.loc[dup_mask, list(keys)]
                .value_counts()
                .head(10))
        raise AssertionError(f"{name} no es único por {list(keys)}. Ejemplos (top):\n{ex}")

def _safe_merge(left: pd.DataFrame,
                right: Optional[pd.DataFrame],
                on: Union[str, List[str], Tuple[str, ...]],
                how: str = "left",
                validate: str = "many_to_one",
                name: str = "") -> pd.DataFrame:
    """Wrapper de merge con validate= para evitar many-to-many silenciosos."""
    if right is None:
        return left
    return left.merge(right, on=on, how=how, validate=validate)


# -------------------------------
# Lectura de datos
# -------------------------------

def load_panel(raw_dir: Path) -> pd.DataFrame:
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

    # merge seguro (items debe ser único por item_nbr)
    df = _safe_merge(df, items, on="item_nbr", how="left", validate="many_to_one", name="panel×items")

    df = _downcast_numeric(df, float_cols=["sales", "onpromotion"], int_cols=["store_nbr", "item_nbr", "class", "perishable"])
    logging.info(f"Panel cargado: {df.shape[0]:,} filas")
    return df


def load_stores(raw_dir: Path) -> pd.DataFrame:
    stores_path = _first_existing(raw_dir, ["stores.csv"])
    stores = pd.read_csv(stores_path)
    _ensure_cols(stores, ["store_nbr", "city", "state", "type", "cluster"], "stores")
    # garantizar unicidad por store_nbr
    _assert_unique(stores, ["store_nbr"], "stores.csv")
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
# Controles por confusión (holidays)
# -------------------------------

def _effective_holidays(hol: pd.DataFrame) -> pd.DataFrame:
    hol = hol.copy()
    transfer_orig = hol.loc[hol["type"].astype(str).str.lower().eq("transfer")]
    originals = set(transfer_orig["description"].astype(str).unique().tolist())
    transferred_new = hol.loc[hol["transferred"] == True]
    keep_mask = (hol["type"].astype(str).str.lower() != "transfer")
    if not transferred_new.empty:
        keep_mask &= (~hol["description"].astype(str).isin(originals)) | hol["transferred"]
    eff = hol.loc[keep_mask].copy()
    eff = eff.drop_duplicates(subset=["date", "type", "locale", "locale_name", "description", "transferred"])
    return eff


def build_holiday_controls(hol: pd.DataFrame, stores: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hol = _effective_holidays(hol)
    nat = hol.loc[hol["locale"].astype(str).str.lower().eq("national")].copy()
    reg = hol.loc[hol["locale"].astype(str).str.lower().eq("regional")].copy()
    loc = hol.loc[hol["locale"].astype(str).str.lower().eq("local")].copy()

    nat_flag = nat[["date", "type"]].copy()
    nat_flag["HNat"] = 1.0
    nat_flag = nat_flag.groupby("date", as_index=False)["HNat"].max()
    _assert_unique(nat_flag, ["date"], "holidays_nat")

    reg = reg.rename(columns={"locale_name": "state"})
    reg_flag = reg[["date", "state", "type"]].copy()
    reg_flag["HReg"] = 1.0
    reg_flag = reg_flag.groupby(["date", "state"], as_index=False)["HReg"].max()
    _assert_unique(reg_flag, ["date", "state"], "holidays_reg")

    loc = loc.rename(columns={"locale_name": "city"})
    loc_flag = loc[["date", "city", "type"]].copy()
    loc_flag["HLoc"] = 1.0
    loc_flag = loc_flag.groupby(["date", "city"], as_index=False)["HLoc"].max()
    _assert_unique(loc_flag, ["date", "city"], "holidays_loc")

    type_by_date = hol[["date", "type"]].copy()
    type_by_date["is_bridge"] = type_by_date["type"].astype(str).str.lower().eq("bridge").astype(float)
    type_by_date["is_additional"] = type_by_date["type"].astype(str).str.lower().eq("additional").astype(float)
    type_by_date["is_work_day"] = type_by_date["type"].astype(str).str.lower().eq("work day").astype(float)
    type_flags = type_by_date.groupby("date", as_index=False)[["is_bridge", "is_additional", "is_work_day"]].max()
    _assert_unique(type_flags, ["date"], "holidays_types")

    return nat_flag, reg_flag, loc_flag, type_flags


# -------------------------------
# Proxies semanales
# -------------------------------

def build_weekly_proxies(transactions: pd.DataFrame, oil: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = transactions.copy()
    tr["year"] = tr["date"].dt.isocalendar().year.astype(int)
    tr["week"] = tr["date"].dt.isocalendar().week.astype(int)
    tr["year_week"] = tr["year"].astype(str) + "-" + tr["week"].astype(str).str.zfill(2)

    trw = tr.groupby(["year_week", "store_nbr"], as_index=False)["transactions"].sum()
    trw["Fsw_log1p"] = np.log1p(trw["transactions"].clip(lower=0.0))
    trw = trw[["year_week", "store_nbr", "Fsw_log1p"]]
    _assert_unique(trw, ["year_week", "store_nbr"], "weekly_store_proxy")

    oi = oil.copy()
    oi["year"] = oi["date"].dt.isocalendar().year.astype(int)
    oi["week"] = oi["date"].dt.isocalendar().week.astype(int)
    oi["year_week"] = oi["year"].astype(str) + "-" + oi["week"].astype(str).str.zfill(2)
    oiw = oi.groupby("year_week", as_index=False)["oil_price"].mean()
    oiw = oiw.rename(columns={"oil_price": "Ow"})
    _assert_unique(oiw, ["year_week"], "weekly_oil_proxy")

    return trw, oiw


def build_regional_proxies(transactions: pd.DataFrame,
                           stores: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas:
      ['year_week', 'store_nbr', 'regional_proxy']
    donde regional_proxy = transacciones_semana_tienda / transacciones_semana_estado.
    Garantiza unicidad por (year_week, store_nbr).
    """
    tr = transactions.copy()

    # Normalización temporal
    if "date" in tr.columns:
        tr["date"] = pd.to_datetime(tr["date"], errors="coerce").dt.tz_localize(None)
    if "year_week" not in tr.columns:
        iw = tr["date"].dt.isocalendar()
        tr["year_week"] = iw["year"].astype(str) + "-" + iw["week"].astype(str).str.zfill(2)

    # Mapa tienda -> estado (limpio y 1:1)
    st_map = (stores[["store_nbr", "state"]]
              .copy()
              .assign(state=lambda d: d["state"].astype(str).str.strip()))
    # Si por alguna razón hubiera repeticiones con estados distintos, avisar y colapsar.
    multi_state = (st_map.groupby("store_nbr")["state"].nunique() > 1)
    if multi_state.any():
        bad = multi_state[multi_state].index.tolist()
        logging.warning("Stores con múltiples 'state' detectados (se tomará la 1ª aparición): %s", bad)
        st_map = (st_map.sort_values(["store_nbr", "state"])
                         .drop_duplicates("store_nbr", keep="first"))

    _assert_unique(st_map, ["store_nbr"], "stores_map (store->state)")

    # Unir estado a transacciones por tienda
    tr = tr.merge(st_map, on="store_nbr", how="left", validate="m:1")

    # Agregaciones semanales
    reg_w = (tr.groupby(["year_week", "state"], observed=False, as_index=False)["transactions"]
               .sum()
               .rename(columns={"transactions": "trans_state"}))
    store_w = (tr.groupby(["year_week", "store_nbr", "state"], observed=False, as_index=False)["transactions"]
                 .sum()
                 .rename(columns={"transactions": "trans_store"}))

    _assert_unique(reg_w, ["year_week", "state"], "reg_w (semana-estado)")
    _assert_unique(store_w, ["year_week", "store_nbr", "state"], "store_w (semana-tienda-estado)")

    # Join correcto: (semana, estado)
    out = store_w.merge(reg_w, on=["year_week", "state"], how="left", validate="m:1")

    # Proxy: cuota de la tienda dentro del estado esa semana
    out["regional_proxy"] = np.where(out["trans_state"] > 0,
                                     out["trans_store"] / out["trans_state"], np.nan)

    out = (out[["year_week", "store_nbr", "regional_proxy"]]
             .sort_values(["year_week", "store_nbr"])
             .reset_index(drop=True))

    # Debe ser único por (semana, tienda)
    _assert_unique(out, ["year_week", "store_nbr"], "regional_proxy")
    return out


# -------------------------------
# Features temporales
# -------------------------------

def add_temporal_features(panel: pd.DataFrame,
                          lags_days: Sequence[int],
                          fourier_k: int,
                          add_dow: bool = True,
                          add_paydays: bool = True) -> pd.DataFrame:
    df = panel.copy()
    day_of_year = df["date"].dt.dayofyear.astype(float)
    period = 365.25
    for k in range(1, fourier_k + 1):
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / period)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / period)

    if add_dow:
        df["dow"] = df["date"].dt.dayofweek.astype("int8")
        dow_dum = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)
        df = pd.concat([df, dow_dum], axis=1)

    if add_paydays:
        d = df["date"]
        df["is_month_end"] = (d.dt.is_month_end).astype("int8")
        df["is_month_start"] = (d.dt.day <= 3).astype("int8")
        df["is_15th"] = (d.dt.day == 15).astype("int8")
        df["days_to_month_end"] = (d.dt.daysinmonth - d.dt.day).astype("int16")

    df = df.sort_values(["store_nbr", "item_nbr", "date"])
    grouped = df.groupby(["store_nbr", "item_nbr"], group_keys=False)
    for L in lags_days:
        df[f"lag_{L}d"] = grouped["sales"].shift(L)
    return df


def _stl_trend_and_availability(y: pd.Series, period: int = 7, robust: bool = True, use_stl: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    CÁLCULO CAUSAL: evitamos usar 'futuro'. Si usamos rolling, 'center=False'.
    Aunque exista STL, por seguridad devolvemos una media móvil CAUSAL.
    """
    vals = y.to_numpy(dtype=float)
    # tendencia one-sided (solo pasado)
    trend = pd.Series(vals).rolling(window=max(7, period), min_periods=1, center=False).mean().to_numpy()
    avail = (trend > 0.5).astype(float)
    return trend, avail


def add_availability_and_store_trend(panel: pd.DataFrame, use_stl: bool) -> pd.DataFrame:
    """
    Calcula disponibilidad y tendencias de ítem/tienda.
    IMPORTANTE: solo se usan como features las versiones rezagadas (_l1) para evitar fuga.
    """
    df = panel.copy().sort_values(["store_nbr", "item_nbr", "date"])

    trends = []
    avails = []
    for (s, it), g in df.groupby(["store_nbr", "item_nbr"]):
        t, a = _stl_trend_and_availability(g["sales"], period=7, robust=True, use_stl=use_stl)
        trends.append(pd.Series(t, index=g.index))
        avails.append(pd.Series(a, index=g.index))
    df["trend_T"] = pd.concat(trends).sort_index()
    df["available_A"] = pd.concat(avails).sort_index()
    # Usar SOLO tendencia rezagada como feature
    df["trend_T_l1"] = df.groupby(["store_nbr", "item_nbr"], observed=False)["trend_T"].shift(1)

    store_sum = df.groupby(["store_nbr", "date"], as_index=False)["sales"].sum().sort_values(["store_nbr", "date"])
    store_trend_parts = []
    for s, g in store_sum.groupby("store_nbr"):
        t, _ = _stl_trend_and_availability(g["sales"], period=7, robust=True, use_stl=use_stl)
        part = g[["store_nbr", "date"]].copy()
        part["Q_store_trend"] = t
        store_trend_parts.append(part)
    store_trend = pd.concat(store_trend_parts, ignore_index=True)

    df = _safe_merge(df, store_trend, on=["store_nbr", "date"], how="left", validate="many_to_one", name="panel×store_trend")
    df["Q_store_trend_l1"] = df.groupby("store_nbr", observed=False)["Q_store_trend"].shift(1)
    return df


def add_intermitency_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula:
      - zero_streak: racha de días consecutivos con ventas == 0 (por store,item)
      - ADI y CV2 por unidad
    Implementación robusta: evita Series de dtype 'object' con arrays
    usando transform y devolviendo Series indexadas (no 'np.ndarray').
    """
    df = panel.sort_values(["store_nbr", "item_nbr", "date"]).copy()

    def _zero_streak_1d(flags: np.ndarray) -> np.ndarray:
        out = np.zeros(flags.shape[0], dtype=np.int32)
        c = 0
        for i, f in enumerate(flags):
            if f:
                c += 1
            else:
                c = 0
            out[i] = c
        return out

    df["zero_streak"] = (
        df.groupby(["store_nbr", "item_nbr"], observed=False)["sales"]
          .transform(lambda s: pd.Series(
              _zero_streak_1d((s.to_numpy(dtype=float) <= 0.0)),
              index=s.index
          ))
          .astype("int16")
    )

    agg_rows = []
    for (s, it), g in df.groupby(["store_nbr", "item_nbr"], observed=False):
        y = g["sales"].to_numpy(dtype=float)
        pos = np.flatnonzero(y > 0)
        if pos.size == 0:
            ADI = float(len(y))
        else:
            diffs = np.diff(np.r_[-1, pos, len(y)])
            ADI = float(np.maximum(diffs - 1, 0).mean())
        mu = y.mean()
        CV2 = float((y.std(ddof=0) / (mu + 1e-6)) ** 2) if mu > 0 else 0.0
        agg_rows.append({"store_nbr": s, "item_nbr": it, "ADI": ADI, "CV2": CV2})
    agg = pd.DataFrame(agg_rows)

    df = _safe_merge(df, agg, on=["store_nbr", "item_nbr"], how="left", validate="many_to_one", name="panel×ADI_CV2")
    return df


def add_store_metadata(panel: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    df = _safe_merge(panel, stores[["store_nbr", "type", "cluster", "state", "city"]],
                     on="store_nbr", how="left", validate="many_to_one", name="panel×stores")
    for col in ["type", "cluster", "state"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            if dummies.shape[1] > 0:
                dcols = sorted(dummies.columns)
                dummies = dummies[dcols[1:]]
                df = pd.concat([df, dummies], axis=1)
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


def universe_from_episodes_and_donors(episodes: pd.DataFrame,
                                      donors: Optional[pd.DataFrame],
                                      top_k: int,
                                      include_cannibals: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - victims: víctimas j de todos los episodios
      - donors_top: donantes top_k (si hay donantes; DataFrame vacío si no)
      - cannibals: caníbales i (si include_cannibals=True)
    """
    victims = episodes[["j_store", "j_item"]].drop_duplicates().rename(columns={"j_store": "store_nbr", "j_item": "item_nbr"})
    cannibals = episodes[["i_store", "i_item"]].drop_duplicates().rename(columns={"i_store": "store_nbr", "i_item": "item_nbr"}) if include_cannibals else pd.DataFrame(columns=["store_nbr","item_nbr"])
    if donors is None or donors.empty:
        donors_top = pd.DataFrame(columns=["store_nbr","item_nbr","j_store","j_item","rank"])
    else:
        donors_top = (donors.sort_values(["j_store", "j_item", "rank"])
                      .groupby(["j_store", "j_item"], as_index=False)
                      .head(top_k))
        donors_top = donors_top.rename(columns={"donor_store": "store_nbr", "donor_item": "item_nbr"})
        donors_top = donors_top[["store_nbr", "item_nbr", "j_store", "j_item", "rank"]]
        # Anti‑trivial: impedir que víctima/caníbal aparezcan como donantes
        di = set(map(tuple, episodes[["i_store","i_item"]].itertuples(index=False, name=None)))
        dj = set(map(tuple, episodes[["j_store","j_item"]].itertuples(index=False, name=None)))
        donors_top = donors_top[~donors_top[["store_nbr","item_nbr"]].apply(tuple, axis=1).isin(di|dj)]
    return victims, donors_top, cannibals


def compute_global_date_window(episodes: pd.DataFrame, max_lag: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    min_date = (episodes["pre_start"].min() - pd.Timedelta(days=max_lag)).normalize()
    max_date = episodes["post_end"].max().normalize()
    return min_date, max_date


# -------------------------------
# Filtrado del panel y construcción de features base
# -------------------------------

def filter_panel_to_universe(panel: pd.DataFrame,
                             victims: pd.DataFrame,
                             donors_top: Optional[pd.DataFrame],
                             cannibals: Optional[pd.DataFrame],
                             date_min: pd.Timestamp,
                             date_max: pd.Timestamp) -> pd.DataFrame:
    """
    Filtra el panel a víctimas + (opc) donantes top_k + (opc) caníbales y fechas en [date_min, date_max].
    """
    parts = [victims[["store_nbr","item_nbr"]]]
    if donors_top is not None and not donors_top.empty:
        parts.append(donors_top[["store_nbr","item_nbr"]].drop_duplicates())
    if cannibals is not None and not cannibals.empty:
        parts.append(cannibals[["store_nbr","item_nbr"]])
    key_units = pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame(columns=["store_nbr","item_nbr"])

    p = panel.loc[panel["date"].between(date_min, date_max)].copy()
    if not key_units.empty:
        p = p.merge(key_units.assign(_keep=1), on=["store_nbr", "item_nbr"], how="inner", validate="many_to_one").drop(columns=["_keep"], errors="ignore")

    p = _downcast_numeric(p, float_cols=["sales", "onpromotion"], int_cols=["store_nbr", "item_nbr"])
    logging.info(f"Panel filtrado (universo): {p.shape[0]:,} filas, {key_units.shape[0] if not key_units.empty else 0} unidades")
    return p


# -------------------------------
# Confounders desde panel completo
# -------------------------------

def promo_pressure_from_panel(panel_all: pd.DataFrame) -> pd.DataFrame:
    p = panel_all.copy()
    p["onpromotion"] = (pd.to_numeric(p["onpromotion"], errors="coerce").fillna(0.0) > 0).astype("int8")
    g = (p.groupby(["store_nbr", "class", "date"], as_index=False)
           .agg(n_items=("item_nbr", "nunique"),
                n_on=("onpromotion", "sum")))
    g["promo_share_sc"] = (g["n_on"] / g["n_items"]).astype("float32")
    _assert_unique(g, ["store_nbr","class","date"], "promo_pressure")
    return g[["store_nbr", "class", "date", "n_items", "n_on", "promo_share_sc"]]


def class_index_excluding_item(panel_all: pd.DataFrame) -> pd.DataFrame:
    tot = (panel_all.groupby(["store_nbr", "class", "date"], as_index=False)["sales"]
                   .sum().rename(columns={"sales": "class_sales"}))
    p = panel_all.merge(tot, on=["store_nbr", "class", "date"], how="left", validate="many_to_one")
    p["class_index_excl"] = (p["class_sales"] - p["sales"]).clip(lower=0.0)
    out = p[["store_nbr", "item_nbr", "date", "class_index_excl"]]
    _assert_unique(out, ["store_nbr","item_nbr","date"], "class_index_excluding_item")
    return out


def attach_controls(panel: pd.DataFrame,
                    stores: pd.DataFrame,
                    transactions: pd.DataFrame,
                    oil: pd.DataFrame,
                    holidays_nat: pd.DataFrame,
                    holidays_reg: pd.DataFrame,
                    holidays_loc: pd.DataFrame,
                    holidays_types: pd.DataFrame,
                    *,
                    weekly_store_proxy: Optional[pd.DataFrame] = None,
                    weekly_oil_proxy: Optional[pd.DataFrame] = None,
                    regional_proxy: Optional[pd.DataFrame] = None,
                    promo_ctx: Optional[pd.DataFrame] = None,
                    class_index: Optional[pd.DataFrame] = None,
                    cfg: PrepConfig) -> pd.DataFrame:
    df = panel.copy()

    df["year"] = df["date"].dt.isocalendar().year.astype(int)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["year_week"] = df["year"].astype(str) + "-" + df["week"].astype(str).str.zfill(2)

    df = _safe_merge(df, holidays_nat, on="date", how="left", validate="many_to_one", name="panel×hol_nat")
    df = _safe_merge(df, stores[["store_nbr", "state", "city"]], on="store_nbr", how="left", validate="many_to_one", name="panel×stores_state_city")
    df = _safe_merge(df, holidays_reg, on=["date", "state"], how="left", validate="many_to_one", name="panel×hol_reg")
    df = _safe_merge(df, holidays_loc, on=["date", "city"], how="left", validate="many_to_one", name="panel×hol_loc")
    df = _safe_merge(df, holidays_types, on="date", how="left", validate="many_to_one", name="panel×hol_types")
    for c in ["HNat", "HReg", "HLoc", "is_bridge", "is_additional", "is_work_day"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    if weekly_store_proxy is not None:
        df = _safe_merge(df, weekly_store_proxy, on=["year_week", "store_nbr"], how="left", validate="many_to_one", name="panel×weekly_store_proxy")
        df["Fsw_log1p"] = df["Fsw_log1p"].fillna(method="ffill").fillna(0.0)
    if weekly_oil_proxy is not None:
        df = _safe_merge(df, weekly_oil_proxy, on="year_week", how="left", validate="many_to_one", name="panel×weekly_oil_proxy")
        df["Ow"] = df["Ow"].fillna(method="ffill").fillna(0.0)

    if cfg.use_regional_proxy and regional_proxy is not None:
        df = _safe_merge(df, regional_proxy, on=["year_week", "store_nbr"], how="left", validate="many_to_one", name="panel×regional_proxy")
        df["F_state_excl_store_log1p"] = df["F_state_excl_store_log1p"].fillna(method="ffill").fillna(0.0)

    if cfg.add_promo_pressure and promo_ctx is not None:
        df = _safe_merge(df, promo_ctx, on=["store_nbr", "class", "date"], how="left", validate="many_to_one", name="panel×promo_pressure")
        df[["n_items", "n_on", "promo_share_sc"]] = df[["n_items", "n_on", "promo_share_sc"]].fillna(0.0)
        denom = np.maximum(df["n_items"] - 1.0, 1.0)
        df["promo_share_sc_excl"] = ((df["n_on"] - (df["onpromotion"] > 0).astype(float)) / denom).clip(lower=0.0)
        df = df.sort_values(["store_nbr", "class", "date"])
        df["promo_share_sc_l7"]  = df.groupby(["store_nbr", "class"])["promo_share_sc"].shift(7)
        df["promo_share_sc_l14"] = df.groupby(["store_nbr", "class"])["promo_share_sc"].shift(14)
        df["promo_share_sc_excl_l7"]  = df.groupby(["store_nbr", "class"])["promo_share_sc_excl"].shift(7)
        df["promo_share_sc_excl_l14"] = df.groupby(["store_nbr", "class"])["promo_share_sc_excl"].shift(14)

    if cfg.add_class_index_excl and class_index is not None:
        df = _safe_merge(df, class_index, on=["store_nbr", "item_nbr", "date"], how="left", validate="many_to_one", name="panel×class_index_excl")
        df = df.sort_values(["store_nbr", "item_nbr", "date"])
        df["class_index_excl_l7"]  = df.groupby(["store_nbr", "item_nbr"])["class_index_excl"].shift(7)
        df["class_index_excl_l14"] = df.groupby(["store_nbr", "item_nbr"])["class_index_excl"].shift(14)

    df = df.drop(columns=["year", "week"], errors="ignore")
    return df


# -------------------------------
# Construcción de datasets por episodio
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


def _make_cluster_id(i_store: int, i_item: int, j_store: int, j_item: int, treat_start: pd.Timestamp) -> str:
    """
    Cluster único por episodio (evita colisiones cuando un mismo caníbal i afecta múltiples víctimas j en la misma fecha).
    """
    return f"{i_store}_{i_item}_{j_store}_{j_item}_{pd.to_datetime(treat_start).date()}"

def _write_observed_series(out_dir: Path, episode_id: str, df_victim: pd.DataFrame) -> None:
    """Exporta serie observada (víctima) para facilitar que la EDA la encuentre con clave robusta."""
    obs_dir = out_dir / "gsc" / "observed_series"
    obs_dir.mkdir(parents=True, exist_ok=True)
    (df_victim[["date", "sales"]]
        .sort_values("date")
        .to_parquet(obs_dir / f"{episode_id}__observed.parquet", index=False))

def build_meta_for_episode_fast(ep_row: pd.Series,
                                base_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Versión liviana para Meta:
      - Sólo víctima j
      - Etiquetas de ventana y tratamiento (sin donantes)
      - Añade y_raw y y_log1p para trazado/modelo explícito
    """
    i_store, i_item = int(ep_row["i_store"]), int(ep_row["i_item"])
    j_store, j_item = int(ep_row["j_store"]), int(ep_row["j_item"])
    pre_start, treat_start, post_end = ep_row["pre_start"], ep_row["treat_start"], ep_row["post_end"]
    episode_id = ep_row["episode_id"]
    cluster_id = _make_cluster_id(i_store, i_item, j_store, j_item, treat_start)

    sub = base_panel.loc[
        base_panel["date"].between(pre_start, post_end) &
        (base_panel["store_nbr"] == j_store) &
        (base_panel["item_nbr"] == j_item)
    ].copy()

    # Garantizar unicidad por fecha (víctima)
    _assert_unique(sub, ["store_nbr","item_nbr","date"], f"subpanel víctima {episode_id}")

    sub["unit_id"] = sub["store_nbr"].astype(str) + ":" + sub["item_nbr"].astype(str)
    sub["treated_unit"] = 1
    sub["treated_time"] = (sub["date"] >= treat_start).astype(int)
    sub["D"] = sub["treated_time"].astype(int)
    sub["episode_id"] = episode_id
    sub["cluster_id"] = cluster_id
    sub["sc_hat"] = np.nan  # no donors en modo rápido

    # Variables objetivo explícitas
    sub["y_raw"] = sub["sales"].astype(float)
    sub["y_log1p"] = np.log1p(sub["y_raw"].clip(lower=0.0))

    # Máscaras (entrenamiento temporal: PRE estricto)
    sub["is_pre"] = (sub["date"] < treat_start).astype(int)
    sub["train_mask_time"] = sub["is_pre"].astype(int)
    sub["train_mask"] = sub["train_mask_time"]  # compat

    return sub


def build_gsc_and_meta_for_episode(ep_row: pd.Series,
                                   base_panel: pd.DataFrame,
                                   donors_map: pd.DataFrame,
                                   cfg: PrepConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Construye para un episodio:
      - panel GSC long con víctima + donantes elegidos y etiquetas de tratamiento
      - panel Meta long (víctima y opcionalmente donantes)
    """

    i_store, i_item = int(ep_row["i_store"]), int(ep_row["i_item"])
    j_store, j_item = int(ep_row["j_store"]), int(ep_row["j_item"])
    pre_start = ep_row["pre_start"]; treat_start = ep_row["treat_start"]; post_end = ep_row["post_end"]

    episode_id = ep_row["episode_id"]
    cluster_id = _make_cluster_id(i_store, i_item, j_store, j_item, treat_start)

    # Donantes del episodio
    dons = donors_map.loc[(donors_map["j_store"] == j_store) & (donors_map["j_item"] == j_item)].copy()
    dons = dons.sort_values("rank").drop_duplicates(subset=["store_nbr", "item_nbr"]).head(cfg.top_k_donors)
    # Nunca permitir que la víctima o el caníbal sean donantes
    dons = dons[~((dons["store_nbr"] == j_store) & (dons["item_nbr"] == j_item))]
    dons = dons[~((dons["store_nbr"] == i_store) & (dons["item_nbr"] == i_item))]
    donors_units = list(map(tuple, dons[["store_nbr", "item_nbr"]].to_numpy()))
    units = [(j_store, j_item)] + donors_units

    # Subpanel (víctima + donantes)
    sub = base_panel.loc[
        base_panel["date"].between(pre_start, post_end) &
        base_panel.set_index(["store_nbr", "item_nbr"]).index.isin(units)
    ].copy()

    # Garantizar unicidad por (unidad, fecha)
    _assert_unique(sub, ["store_nbr","item_nbr","date"], f"subpanel episodio {episode_id}")

    # Calidad donantes
    g_list = []
    donor_quality = []
    for (s, it), g in sub.groupby(["store_nbr", "item_nbr"]):
        is_victim = (s == j_store) and (it == j_item)
        promo_share = _compute_promo_share(g)
        avail_share = _compute_availability_share(g)
        keep = True
        reasons = []
        if (not is_victim) and (promo_share > cfg.max_donor_promo_share):
            keep = False; reasons.append(f"promo_share={promo_share:.3f}>{cfg.max_donor_promo_share}")
        if (not is_victim) and (avail_share < cfg.min_availability_share):
            keep = False; reasons.append(f"avail_share={avail_share:.3f}<{cfg.min_availability_share}")
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
        g_list = [sub.loc[(sub["store_nbr"] == j_store) & (sub["item_nbr"] == j_item)]]

    subf = pd.concat(g_list, ignore_index=True)
    subf["unit_id"] = subf["store_nbr"].astype(str) + ":" + subf["item_nbr"].astype(str)
    subf["treated_unit"] = ((subf["store_nbr"] == j_store) & (subf["item_nbr"] == j_item)).astype(int)
    subf["treated_time"] = (subf["date"] >= treat_start).astype(int)
    subf["D"] = ((subf["treated_unit"] == 1) & (subf["treated_time"] == 1)).astype(int)
    subf["episode_id"] = episode_id
    subf["cluster_id"] = cluster_id

    # Donor-blend (si aplica)
    if cfg.add_donor_blend_sc:
        try:
            vict = subf[(subf["store_nbr"] == j_store) & (subf["item_nbr"] == j_item)].copy()
            don = subf[subf["treated_unit"] == 0].copy()
            don["donor_unit"] = don["store_nbr"].astype(str) + ":" + don["item_nbr"].astype(str)

            Dpre = don[don["date"] < treat_start][["date", "donor_unit", "sales"]]
            if not Dpre.empty:
                Xpre = Dpre.pivot(index="date", columns="donor_unit", values="sales").fillna(0.0)
                ypre = (vict[vict["date"] < treat_start][["date", "sales"]]
                        .set_index("date").reindex(Xpre.index)["sales"].fillna(0.0))
                if Xpre.shape[1] >= 1 and Xpre.shape[0] >= max(3, int(cfg.donor_blend_min_pre_T)):
                    w = _ridge_fit(Xpre.to_numpy(dtype=float), ypre.to_numpy(dtype=float), alpha=float(cfg.donor_blend_alpha))
                    W = pd.Series(w, index=Xpre.columns)
                    Xall = don.pivot(index="date", columns="donor_unit", values="sales").fillna(0.0)
                    sc_hat = (Xall * W.reindex(Xall.columns).fillna(0.0)).sum(axis=1)
                    sc_hat = sc_hat.reindex(subf["date"].unique(), fill_value=np.nan)
                    scdf = pd.DataFrame({"date": sc_hat.index, "sc_hat": sc_hat.values})
                    subf = subf.merge(scdf, on="date", how="left", validate="many_to_one")
                    # NUEVO: no propagar sc_hat a donantes (claridad para EDA)
                    subf.loc[subf["treated_unit"] == 0, "sc_hat"] = np.nan
                else:
                    subf["sc_hat"] = np.nan
            else:
                subf["sc_hat"] = np.nan
        except Exception as e:
            logging.warning(f"[{episode_id}] donor-blend falló: {e}")
            subf["sc_hat"] = np.nan
    else:
        subf["sc_hat"] = np.nan

    # Paneles de salida
    gsc_panel = subf.copy()
    if cfg.meta_units == "victims_only":
        meta_panel = subf[subf["treated_unit"] == 1].copy()
    else:
        meta_panel = subf.copy()

    # Variables objetivo explícitas para Meta
    meta_panel["y_raw"] = meta_panel["sales"].astype(float)
    meta_panel["y_log1p"] = np.log1p(meta_panel["y_raw"].clip(lower=0.0))

    # Máscaras (entrenamiento temporal: PRE estricto)
    subf["is_pre"] = (subf["date"] < treat_start).astype(int)
    subf["train_mask_time"] = subf["is_pre"].astype(int)
    subf["train_mask"] = subf["train_mask_time"]  # compat
    meta_panel["is_pre"] = (meta_panel["date"] < treat_start).astype(int)
    meta_panel["train_mask_time"] = meta_panel["is_pre"].astype(int)
    meta_panel["train_mask"] = meta_panel["train_mask_time"]

    n_donors_kept = int(meta_panel.loc[meta_panel["treated_unit"] == 0, ["store_nbr", "item_nbr"]]
                        .drop_duplicates().shape[0])

    logging.debug(f"[{episode_id}] GSC: {len(gsc_panel)} filas, Meta: {len(meta_panel)} filas")
    logging.debug(f"[{episode_id}] Fechas GSC: {gsc_panel['date'].min()} - {gsc_panel['date'].max()}")
    logging.debug(f"[{episode_id}] Fechas Meta: {meta_panel['date'].min()} - {meta_panel['date'].max()}")

    meta_info = {
        "episode_id": episode_id,
        "victim": {"store_nbr": j_store, "item_nbr": j_item},
        "cannibal": {"store_nbr": i_store, "item_nbr": i_item},
        "n_donors_input": len(donors_units),
        "n_donors_kept": n_donors_kept,
        "quality": donor_quality
    }

    # Chequeo de consistencia observado entre GSC y Meta (víctima) **SIN trend_T**
    victim_gsc = (gsc_panel[gsc_panel["treated_unit"] == 1][["date", "sales"]]
                  .sort_values("date").reset_index(drop=True))
    victim_meta = (meta_panel[meta_panel["treated_unit"] == 1][["date", "sales"]]
                   .sort_values("date").reset_index(drop=True))
    assert victim_gsc.equals(victim_meta), f"Mismatch en observado víctima entre GSC y Meta para {episode_id}"

    # Smoke test: sc_hat idéntico al observado
    try:
        vict = gsc_panel.query("treated_unit==1")[["date", "sales", "sc_hat"]].dropna()
        if not vict.empty:
            diff = np.abs(vict["sales"].to_numpy(dtype=float) - vict["sc_hat"].to_numpy(dtype=float))
            if np.allclose(diff, 0.0, atol=1e-12):
                logging.warning(f"[{episode_id}] sc_hat == sales (posible fuga/autodonante).")
    except Exception:
        pass

    # Exportar observado para EDA (evita 'sin observado')
    try:
        _write_observed_series(cfg.out_dir, episode_id, victim_gsc)
    except Exception as e:
        logging.warning(f"[{episode_id}] No se pudo exportar observed.parquet: {e}")

    return gsc_panel, meta_panel, meta_info


# -------------------------------
# Chequeo de coherencia Observado Meta vs GSC
# -------------------------------

def _check_observed_consistency(meta_df: pd.DataFrame,
                                gsc_df: pd.DataFrame,
                                episode_id: str,
                                strict: bool,
                                fail_fast: bool) -> None:
    try:
        m = (meta_df.query("treated_unit == 1")[["date", "sales"]]
                     .sort_values("date").reset_index(drop=True))
        g = (gsc_df.query("treated_unit == 1")[["date", "sales"]]
                     .sort_values("date").reset_index(drop=True))
        ok = (len(m) == len(g)) and np.allclose(
            m["sales"].to_numpy(dtype=float),
            g["sales"].to_numpy(dtype=float),
            atol=1e-8, equal_nan=True
        )
        if not ok:
            msg = f"[{episode_id}] Inconsistencia: 'Observado' difiere entre Meta y GSC para la víctima."
            if strict or fail_fast:
                raise AssertionError(msg)
            else:
                logging.warning(msg)
    except Exception as e:
        if strict or fail_fast:
            raise
        logging.warning(f"[{episode_id}] No se pudo verificar coherencia Observado Meta vs GSC: {e}")


# -------------------------------
# Orquestación
# -------------------------------

def prepare_datasets(cfg: PrepConfig) -> None:
    _setup_logging(cfg.log_level)
    cfg = _coerce_cfg(cfg)

    logging.info(f"Configuración:")
    logging.info(f"  - meta_only: {cfg.meta_only}")
    logging.info(f"  - donors_path: {cfg.donors_path}")
    logging.info(f"  - gsc_eval_n: {cfg.gsc_eval_n}")
    logging.info(f"  - gsc_eval_ids_path: {cfg.gsc_eval_ids_path}")
    logging.info(f"  - gsc_eval_selection: {cfg.gsc_eval_selection}")

    # Bandera maestra
    if not cfg.confounds_plus:
        cfg.add_dow = False; cfg.add_paydays = False; cfg.use_regional_proxy = False
        cfg.add_promo_pressure = False; cfg.add_class_index_excl = False
        cfg.add_intermitency_feats = False; cfg.add_donor_blend_sc = False

    # Rutas
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "gsc").mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "gsc" / "observed_series").mkdir(parents=True, exist_ok=True)  # para EDA
    (cfg.out_dir / "meta").mkdir(parents=True, exist_ok=True)
    cfg.out_meta_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "intermediate").mkdir(parents=True, exist_ok=True)

    # 1) Lectura base
    logging.info("Leyendo episodios...")
    episodes = load_episodes(cfg.episodes_path)
    if cfg.max_episodes is not None:
        episodes = episodes.head(int(cfg.max_episodes)).copy()
        logging.info(f"Limite de episodios activado: {episodes.shape[0]} episodios")

    # Determinar set de evaluación para GSC
    eval_ids: Optional[set] = None
    if (not cfg.meta_only) or (cfg.gsc_eval_n and cfg.gsc_eval_n > 0) or (cfg.gsc_eval_ids_path is not None):
        if cfg.gsc_eval_ids_path and Path(cfg.gsc_eval_ids_path).exists():
            ids_df = pd.read_csv(cfg.gsc_eval_ids_path)
            if "episode_id" in ids_df.columns:
                eval_ids = set(ids_df["episode_id"].astype(str).tolist())
        elif cfg.gsc_eval_n and cfg.gsc_eval_n > 0:
            if cfg.gsc_eval_selection.lower() == "random":
                eval_ids = set(episodes["episode_id"].sample(n=min(cfg.gsc_eval_n, len(episodes)),
                                                             random_state=int(cfg.gsc_eval_seed)).astype(str).tolist())
            else:
                eval_ids = set(episodes["episode_id"].head(cfg.gsc_eval_n).astype(str).tolist())
        if eval_ids:
            sel_path = cfg.out_dir / "episodes_eval.csv"
            pd.DataFrame({"episode_id": sorted(list(eval_ids))}).to_csv(sel_path, index=False)
            logging.info(f"Episodios seleccionados para GSC: {len(eval_ids)} (guardado {sel_path})")

    # 2) Donantes (opcional)
    donors: Optional[pd.DataFrame] = None
    if (not cfg.meta_only) or (eval_ids is not None) or cfg.add_donor_blend_sc:
        if cfg.donors_path and Path(cfg.donors_path).exists():
            donors = load_donors(cfg.donors_path, donor_kind=cfg.donor_kind)
        else:
            donors = None
            if cfg.add_donor_blend_sc:
                logging.warning("No hay donors_per_victim.csv; se desactiva donor-blend (sc_hat).")
                cfg.add_donor_blend_sc = False

    # 3) Universo y ventana
    victims, donors_top, cannibals = universe_from_episodes_and_donors(
        episodes, donors, cfg.top_k_donors,
        include_cannibals=True  # incluimos i para posibles features/store proxies
    )
    date_min, date_max = compute_global_date_window(episodes, max_lag=max(cfg.lags_days))
    logging.info(f"Ventana global: {date_min.date()} → {date_max.date()}")

    # 4) Panel completo y confounders basados en panel_all
    panel_all = load_panel(cfg.raw_dir)
    stores = load_stores(cfg.raw_dir)
    transactions = load_transactions(cfg.raw_dir)
    oil = load_oil(cfg.raw_dir)
    holidays = load_holidays(cfg.raw_dir)

    promo_ctx = promo_pressure_from_panel(panel_all) if cfg.add_promo_pressure else None
    class_idx = class_index_excluding_item(panel_all) if cfg.add_class_index_excl else None

    # 5) Filtrar al universo requerido por modo
    need_gsc = ((not cfg.meta_only) or (eval_ids is not None and len(eval_ids) > 0))
    donors_for_filter = donors_top if need_gsc or (cfg.meta_units == "victims_plus_donors") else None
    panel = filter_panel_to_universe(panel_all, victims, donors_for_filter, cannibals, date_min, date_max)

    # 6) Adjuntar controles + features
    logging.info("Construyendo controles y features...")
    hol_nat, hol_reg, hol_loc, hol_types = build_holiday_controls(holidays, stores)
    trw, oiw = build_weekly_proxies(transactions, oil)
    reg_proxy = build_regional_proxies(transactions, stores) if cfg.use_regional_proxy else None

    panel = attach_controls(panel, stores, transactions, oil,
                            hol_nat, hol_reg, hol_loc, hol_types,
                            weekly_store_proxy=trw, weekly_oil_proxy=oiw,
                            regional_proxy=reg_proxy,
                            promo_ctx=promo_ctx, class_index=class_idx, cfg=cfg)

    panel = add_temporal_features(panel, lags_days=cfg.lags_days, fourier_k=cfg.fourier_k,
                                  add_dow=cfg.add_dow, add_paydays=cfg.add_paydays)
    # Forzamos cálculo causal y usaremos SOLO rezagos de tendencias
    panel = add_availability_and_store_trend(panel, use_stl=cfg.use_stl)
    if cfg.add_intermitency_feats:
        panel = add_intermitency_features(panel)
    panel = add_store_metadata(panel, stores)
    if cfg.drop_city and "city" in panel.columns:
        panel = panel.drop(columns=["city"], errors="ignore")
    # Evitar fuga por contemporáneos de tendencia
    panel = panel.drop(columns=[c for c in ["trend_T", "Q_store_trend"] if c in panel.columns], errors="ignore")

    # ---- NUEVO: garantizar unicidad del panel por (store,item,date)
    _assert_unique(panel, ["store_nbr","item_nbr","date"], "panel con features")

    if cfg.save_intermediate:
        inter_path = cfg.out_dir / "intermediate" / "panel_features.parquet"
        panel.to_parquet(inter_path, index=False)
        logging.info(f"Intermedio guardado: {inter_path}")

    # 7) Por episodio
    meta_panels = []
    gsc_quality_rows = []
    index_rows = []

    logging.info("Procesando episodios...")
    for idx, (_, ep) in enumerate(episodes.iterrows(), start=1):
        ep_id = str(ep["episode_id"])
        try:
            do_gsc = (eval_ids is not None) and (ep_id in eval_ids) and (donors is not None) and (not donors.empty)
            if do_gsc:
                gsc_panel, meta_panel, meta_info = build_gsc_and_meta_for_episode(ep, panel, donors_top, cfg)
                # Chequeo de coherencia de Observado (víctima) entre Meta y GSC
                if cfg.check_observed_consistency:
                    _check_observed_consistency(meta_panel, gsc_panel, ep_id, cfg.strict_obs_check, cfg.fail_fast)
                if not cfg.dry_run:
                    out_gsc = cfg.out_dir / "gsc" / f"{ep_id}.parquet"
                    gsc_panel.to_parquet(out_gsc, index=False)
                # calidad donantes
                for q in meta_info["quality"]:
                    row = dict(q); row.setdefault("episode_id", ep_id); gsc_quality_rows.append(row)
            else:
                # Meta rápido (víctima únicamente)
                meta_panel = build_meta_for_episode_fast(ep, panel)
                meta_info = {
                    "episode_id": ep_id,
                    "victim": {"store_nbr": int(ep["j_store"]), "item_nbr": int(ep["j_item"])},
                    "cannibal": {"store_nbr": int(ep["i_store"]), "item_nbr": int(ep["i_item"])},
                    "n_donors_input": 0, "n_donors_kept": 0, "quality": []
                }

            # Para Meta siempre acumulamos
            meta_panels.append(meta_panel)

            index_rows.append({
                "episode_id": ep_id,
                "i_store": ep["i_store"], "i_item": ep["i_item"],
                "j_store": ep["j_store"], "j_item": ep["j_item"],
                "pre_start": ep["pre_start"], "treat_start": ep["treat_start"],
                "post_start": ep["post_start"], "post_end": ep["post_end"],
                "n_meta_rows": len(meta_panel),
                "did_gsc": int(do_gsc),
                "n_donors_input": meta_info.get("n_donors_input", 0),
                "n_donors_kept": meta_info.get("n_donors_kept", 0),
            })

            logging.info(f"[{idx}/{len(episodes)}] Ep {ep_id} | meta_rows={len(meta_panel):,} | GSC={'yes' if do_gsc else 'no'}")
        except Exception as e:
            logging.exception(f"Error en episodio {ep_id}: {e}")
            if cfg.fail_fast:
                raise

    # 8) Índices y salidas
    ep_index = pd.DataFrame(index_rows)
    # Índice general (compat y meta)
    ep_index_path = cfg.out_dir / "episodes_index.parquet"
    ep_index.to_parquet(ep_index_path, index=False)
    # Índice específico para meta (en carpeta meta separada)
    ep_index_meta_path = cfg.out_meta_dir / "episodes_index.parquet"
    ep_index.to_parquet(ep_index_meta_path, index=False)
    logging.info(f"Índice guardado: {ep_index_path} y {ep_index_meta_path} ({ep_index.shape[0]} filas)")

    # Meta stack
    if not cfg.dry_run and meta_panels:
        meta_all = pd.concat(meta_panels, ignore_index=True)
        # Peso por episodio (inverso del tamaño)
        n_by_ep = ep_index.set_index("episode_id")["n_meta_rows"].to_dict()
        w_ep = {k: 1.0 / max(v, 1) for k, v in n_by_ep.items()}
        meta_all["episode_id"] = meta_all["episode_id"].astype(str)
        meta_all["w_episode"] = meta_all["episode_id"].map(w_ep).astype("float32")

        # Ruta nueva (entrenamiento Meta)
        meta_train_path = cfg.out_meta_dir / "windows.parquet"
        meta_all.to_parquet(meta_train_path, index=False)
        logging.info(f"Meta-training stack guardado: {meta_train_path} ({meta_all.shape[0]:,} filas)")

        # --- NUEVO: separación explícita X/Y y blacklist anti-fuga ---
        id_cols = ["episode_id","unit_id","date","store_nbr","item_nbr",
                   "treated_unit","treated_time","D","is_pre","train_mask","train_mask_time","cluster_id"]
        leak_cols = {"sales","y_raw","y_log1p","trend_T","Q_store_trend","sc_hat"}
        X_cols = [c for c in meta_all.columns if c not in leak_cols and c not in id_cols]
        X_safe = meta_all[id_cols + X_cols]
        Y_cols = meta_all[id_cols + ["sales","y_raw","y_log1p"]]

        X_path = cfg.out_meta_dir / "windows_X.parquet"
        Y_path = cfg.out_meta_dir / "windows_Y.parquet"
        X_safe.to_parquet(X_path, index=False)
        Y_cols.to_parquet(Y_path, index=False)
        logging.info(f"Meta X/Y escritos: {X_path} y {Y_path}")

        # Copia de compatibilidad (opcional)
        if cfg.write_legacy_meta_copy:
            legacy_path = cfg.out_dir / "meta" / "all_units.parquet"
            meta_all.to_parquet(legacy_path, index=False)
            logging.info(f"Copia de compatibilidad escrita: {legacy_path}")

    elif cfg.dry_run:
        logging.info("DRY-RUN: se omitió la escritura de stacks.")

    # Calidad donantes si hubo GSC
    if gsc_quality_rows:
        donor_quality_df = pd.DataFrame(gsc_quality_rows)
        donor_quality_path = cfg.out_dir / "gsc" / "donor_quality.parquet"
        donor_quality_df.to_parquet(donor_quality_path, index=False)
        logging.info(f"Log de calidad de donantes guardado: {donor_quality_path} ({donor_quality_df.shape[0]:,} filas)")


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> PrepConfig:
    p = argparse.ArgumentParser(description="Preprocesamiento para Meta y GSC (opcional y acotado).")

    # Rutas
    p.add_argument("--episodes", type=str, default=str(PrepConfig.episodes_path), help="Ruta a pairs_windows.csv")
    p.add_argument("--donors", type=str, default="", help="Ruta a donors_per_victim.csv (opcional en meta_only)")
    p.add_argument("--raw", type=str, default=str(PrepConfig.raw_dir), help="Directorio con datos raw")
    p.add_argument("--out", type=str, default=str(PrepConfig.out_dir), help="Directorio de salida (índices/GSC)")
    p.add_argument("--out_meta", type=str, default=str(PrepConfig.out_meta_dir), help="Directorio de salida para Meta-training")

    # Donantes (si se usan)
    p.add_argument("--top_k", type=int, default=PrepConfig.top_k_donors, help="Top K donantes por víctima")
    p.add_argument("--donor_kind", type=str, default=PrepConfig.donor_kind, help="Filtro por tipo de donante")

    # Temporalidad
    p.add_argument("--lags", type=str, default="7,14,28,56", help="Rezagos en días separados por coma")
    p.add_argument("--fourier_k", type=int, default=PrepConfig.fourier_k, help="Número de armónicos anuales")

    # Calidad donantes
    p.add_argument("--max_donor_promo_share", type=float, default=PrepConfig.max_donor_promo_share, help="Máx % días en promo permitido a donantes")
    p.add_argument("--min_availability_share", type=float, default=PrepConfig.min_availability_share, help="Mín fracción de días disponibles en ventana")

    # Ejecución
    p.add_argument("--no_intermediate", action="store_true", help="No guardar intermedios")
    p.add_argument("--skip_stl", action="store_true", help="Forzar fallback simple (sin STL)")
    p.add_argument("--keep_city", action="store_true", help="Conservar columna 'city' al guardar")
    p.add_argument("--dry_run", action="store_true", help="No escribir salidas (pruebas)")
    p.add_argument("--max_episodes", type=int, default=None, help="Limitar número de episodios a procesar")
    p.add_argument("--log_level", type=str, default=PrepConfig.log_level, help="Nivel de logs (DEBUG, INFO, WARNING, ...)")
    p.add_argument("--fail_fast", action="store_true", help="Abortar al primer error por episodio")

    # Modo meta / selección unidades
    p.add_argument("--meta_only", action="store_true", help="Generar sólo base de Meta (masivo) y saltar GSC salvo set de evaluación")
    p.add_argument("--meta_units", type=str, default=PrepConfig.meta_units, choices=["victims_only","victims_plus_donors"],
                   help="Unidades en el stack de Meta")

    # Selección de evaluación para GSC
    p.add_argument("--gsc_eval_ids", type=str, default="", help="CSV con columna episode_id a evaluar con GSC")
    p.add_argument("--gsc_eval_n", type=int, default=None, help="Número de episodios a evaluar con GSC")
    p.add_argument("--gsc_eval_selection", type=str, default=PrepConfig.gsc_eval_selection, choices=["head","random"],
                   help="Estrategia de selección cuando se usa --gsc_eval_n")
    p.add_argument("--gsc_eval_seed", type=int, default=PrepConfig.gsc_eval_seed, help="Semilla si selección aleatoria")

    # Señales extra
    p.add_argument("--no_confounds_plus", action="store_true", help="Desactiva todas las nuevas señales de confusión")
    p.add_argument("--no_dow", action="store_true", help="No añadir dummies de día de semana")
    p.add_argument("--no_paydays", action="store_true", help="No añadir paydays/fin de mes")
    p.add_argument("--no_regional_proxy", action="store_true", help="No añadir proxy regional de tráfico")
    p.add_argument("--no_promo_pressure", action="store_true", help="No añadir presión promocional (lags/excluyente)")
    p.add_argument("--no_class_index_excl", action="store_true", help="No añadir índice de clase excluyente (lags)")
    p.add_argument("--no_intermitency_feats", action="store_true", help="No añadir ADI/CV²/zero_streak")
    p.add_argument("--no_donor_blend_sc", action="store_true", help="No calcular sc_hat (donor-blend)")
    p.add_argument("--donor_blend_alpha", type=float, default=PrepConfig.donor_blend_alpha, help="Alpha de Ridge para donor-blend")
    p.add_argument("--donor_blend_min_pre_T", type=int, default=PrepConfig.donor_blend_min_pre_T, help="Mínimo PRE T para donor-blend")
    p.add_argument("--no_legacy_meta_copy", action="store_true", help="No duplicar salida de Meta en out/meta/all_units.parquet")

    # Chequeo de coherencia Observado
    p.add_argument("--no_obs_check", action="store_true", help="Desactiva check de coherencia Observado Meta vs GSC")
    p.add_argument("--strict_obs_check", action="store_true", help="Lanza AssertionError si difiere Observado Meta vs GSC")

    args = p.parse_args()
    cfg = PrepConfig(
        episodes_path=Path(args.episodes),
        donors_path=Path(args.donors) if args.donors else None,
        raw_dir=Path(args.raw),
        out_dir=Path(args.out),
        out_meta_dir=Path(args.out_meta),

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
        fail_fast=args.fail_fast,

        meta_only=args.meta_only,
        meta_units=args.meta_units,

        gsc_eval_ids_path=(Path(args.gsc_eval_ids) if args.gsc_eval_ids else None),
        gsc_eval_n=(args.gsc_eval_n if args.gsc_eval_n is not None and int(args.gsc_eval_n)>0 else None),
        gsc_eval_selection=args.gsc_eval_selection,
        gsc_eval_seed=int(args.gsc_eval_seed),

        confounds_plus=(not args.no_confounds_plus),
        add_dow=(not args.no_dow),
        add_paydays=(not args.no_paydays),
        use_regional_proxy=(not args.no_regional_proxy),
        add_promo_pressure=(not args.no_promo_pressure),
        add_class_index_excl=(not args.no_class_index_excl),
        add_intermitency_feats=(not args.no_intermitency_feats),
        add_donor_blend_sc=(not args.no_donor_blend_sc),
        donor_blend_alpha=args.donor_blend_alpha,
        donor_blend_min_pre_T=args.donor_blend_min_pre_T,
        write_legacy_meta_copy=(not args.no_legacy_meta_copy),

        check_observed_consistency=(not args.no_obs_check),
        strict_obs_check=args.strict_obs_check,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    prepare_datasets(cfg)