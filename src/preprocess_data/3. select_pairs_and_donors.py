# -*- coding: utf-8 -*-
"""
select_pairs_and_donors.py  (versión optimizada V3+: vectorización masiva + modo rápido + filtros por (store,class))

- Mantiene:
  * Misma firma de select_pairs_and_donors(H, train, items, stores, outdir)
  * Mismos nombres de funciones públicas/privadas (compatibilidad)
  * Mismos archivos de salida y logs/progreso clave
- Acelera:
  * Scoring ΔH_cls: carga única por TIENDA, partición por CLASE en memoria y vectorización por todos los i de la clase
  * Evita relecturas del shard por clase; el shard se lee una sola vez por tienda
  * Reutiliza item_stats para donantes (evita re-agregación de H)
- Añade:
  * Modo FAST_SCORING=2 (aprox. con media de clase S/C) para pruebas muy rápidas
  * Sampling en j (víctimas) y relajar/exponer filtros vía ENV
  * Permitir superposición de promoción en j (opcional) para aumentar cobertura
  * Volcar tablas intermedias de depuración
  * **NUEVO**: Filtros por (store,class): `SPD_MIN_ITEMS_PER_SC` y `SPD_REQUIRE_MIN_J_PER_SC`
"""

from __future__ import annotations

import os
import sys
import time
import json
import gzip
import math
import traceback
import multiprocessing as mp
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import logging
from collections import OrderedDict, defaultdict

# ------------------- Parámetros (idénticos + override por ENV) -------------------
def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "").strip()
        return int(v) if v else default
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name, "").strip()
        return float(v) if v else default
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    try:
        v = os.environ.get(name, None)
        return v if (v is not None and str(v).strip() != "") else default
    except Exception:
        return default

# Tamaños (defaults más razonables para I/O; puedes sobreescribir por ENV)
CHUNK_H = _env_int("SPD_CHUNK_H", 10_000_000)
CHUNK_TRAIN = _env_int("SPD_CHUNK_TRAIN", 10_000_000)

# Selección de universo base (overrideables por ENV)
MIN_ITEM_OBS = _env_int("SPD_MIN_ITEM_OBS", 200)
H_SD_MIN     = _env_float("SPD_H_SD_MIN", 0.004)
P_ANY_MIN    = _env_float("SPD_P_ANY_MIN", 0.02)
P_ANY_MAX    = _env_float("SPD_P_ANY_MAX", 0.98)

# Caníbales i
P_PROMO_I_MIN = _env_float("SPD_P_PROMO_I_MIN", 0.03)
P_PROMO_I_MAX = _env_float("SPD_P_PROMO_I_MAX", 0.25)
WINDOW_START  = pd.Timestamp(_env_str("SPD_WINDOW_START", "2016-01-01"))
WINDOW_END    = pd.Timestamp(_env_str("SPD_WINDOW_END",   "2017-06-30"))
MIN_RUN_DAYS  = _env_int("SPD_MIN_RUN_DAYS", 3)
# Override por env
N_CANNIBALS = _env_int("SPD_N_CANNIBALS", 10)

# Víctimas j
P_PROMO_J_MAX    = _env_float("SPD_P_PROMO_J_MAX", 0.10)
N_VICTIMS_PER_I  = _env_int("SPD_N_VICTIMS_PER_I", 12)
# Sampling en j
SPD_SAMPLE_J_FRAC = _env_float("SPD_SAMPLE_J_FRAC", 1.0)  # 0<frac<=1
SPD_SAMPLE_J_CAP  = _env_int("SPD_SAMPLE_J_CAP", 0)       # 0 = sin tope

# Donantes
N_DONORS_PER_J = _env_int("SPD_N_DONORS_PER_J", 10)

# Ventanas GSC
PRE_DAYS  = _env_int("SPD_PRE_DAYS", 90)
PRE_GAP   = _env_int("SPD_PRE_GAP", 7)
TREAT_MAX = _env_int("SPD_TREAT_MAX", 14)
POST_DAYS = _env_int("SPD_POST_DAYS", 30)

# Paralelismo (por defecto moderado para no saturar I/O)
N_WORKERS = max(1, min(_env_int("SPD_N_CORES", 10), os.cpu_count() or 1))
MP_IMAP_CHUNKSIZE = _env_int("SPD_MP_CHUNKSIZE", 8)

# Frecuencia de flush incremental
FLUSH_CANNIBALS_EVERY = 100
FLUSH_VICTIMS_EVERY   = 5
FLUSH_DONORS_EVERY    = 50

# Progreso
SPD_PROGRESS_COUNT_ROWS = _env_int("SPD_PROGRESS_COUNT_ROWS", 11)

# Sampling / límites (opcionales)
SPD_SAMPLE_CI_PER_SC = _env_int("SPD_SAMPLE_CI_PER_SC", 50)   # cap por (store,class)
SPD_SAMPLE_FRAC_CI   = _env_float("SPD_SAMPLE_FRAC_CI", 0.25) # fracción por (store,class)
SPD_MAX_TASKS_SCORE  = _env_int("SPD_MAX_TASKS_SCORE", 5000)  # tope global
SPD_SEED             = _env_int("SPD_SEED", 42)

# Caché límites
SPD_MAX_HSC_CACHE = _env_int("SPD_MAX_HSC_CACHE", 4)

# Sharding
SPD_SHARD_H = bool(_env_int("SPD_SHARD_H", 1))  # 1=on
_SHARD_DIR: Optional[str] = None

# Modos rápidos / permisividad
SPD_FAST_SCORING          = _env_int("SPD_FAST_SCORING", 1)  # 0=off, 1=exacto vectorizado, 2=ultra-rápido (S/C)
SPD_ALLOW_J_PROMO_OVERLAP = bool(_env_int("SPD_ALLOW_J_PROMO_OVERLAP", 0))
SPD_SAVE_DEBUG_TABLES     = bool(_env_int("SPD_SAVE_DEBUG_TABLES", 0))

# NUEVOS filtros por (store,class) — valores por defecto según recomendación
SPD_MIN_ITEMS_PER_SC     = _env_int("SPD_MIN_ITEMS_PER_SC", 2)  # mínimo de items válidos por (store,class)
SPD_REQUIRE_MIN_J_PER_SC = _env_int("SPD_REQUIRE_MIN_J_PER_SC", 4)  # mínimo de potenciales j por (store,class)

# ------------------- Progreso y Logging -------------------
_PHASE_ORDER = [
    "init",
    "aggregate_item_stats",
    "promo_stats",
    "build_base",
    "score_cannibals",
    "select_victims",
    "write_pairs",
    "select_donors",
    "write_donors"
]
_PHASE_WEIGHTS = {
    "init": 5,
    "aggregate_item_stats": 15,
    "promo_stats": 10,
    "build_base": 5,
    "score_cannibals": 20,
    "select_victims": 20,
    "write_pairs": 5,
    "select_donors": 15,
    "write_donors": 5,
}
_PREFIX_SUM = {}
_acc = 0
for ph in _PHASE_ORDER:
    _PREFIX_SUM[ph] = _acc
    _acc += _PHASE_WEIGHTS.get(ph, 0)

PROGRESS_FLAGS = {"mute_internal_progress": False}
_LOGGER = logging.getLogger("select_pairs_and_donors"); _LOGGER.addHandler(logging.NullHandler())
_CHECKPOINT_PATH = None
_T0 = time.time()

def _setup_logging(outdir: str, level: int = None) -> None:
    global _LOGGER
    lvl_name = os.environ.get("SPD_LOGLEVEL", "").upper()
    level = level if level is not None else (getattr(logging, lvl_name, logging.INFO) if lvl_name else logging.INFO)
    for h in list(_LOGGER.handlers):
        _LOGGER.removeHandler(h)
    _LOGGER.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s][%(processName)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setLevel(level); sh.setFormatter(fmt); _LOGGER.addHandler(sh)
    os.makedirs(outdir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(outdir, "select_pairs_and_donors.log"), mode="a", encoding="utf-8")
    fh.setLevel(level); fh.setFormatter(fmt); _LOGGER.addHandler(fh)

def _json_default(o):
    if isinstance(o, (pd.Timestamp, pd.Timedelta, np.generic)):
        try:
            return o.item()
        except Exception:
            return str(o)
    return str(o)

def _checkpoint_write(phase: str, progress: float, extra: Optional[Dict] = None):
    global _CHECKPOINT_PATH
    if not _CHECKPOINT_PATH: return
    rec = {"ts": time.time(), "elapsed_s": round(time.time()-_T0,3), "phase": phase,
           "progress": float(progress), "value": float(progress)/100.0}
    if extra: rec.update(extra)
    try:
        with open(_CHECKPOINT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")
    except Exception as e:
        _LOGGER.debug(f"No se pudo escribir checkpoint: {e}")

def _progress_emit(phase: str, within_phase_pct: float, message: str = "", *, force_log: bool=False):
    within_phase_pct = max(0.0, min(100.0, float(within_phase_pct)))
    if phase not in _PHASE_ORDER:
        overall = within_phase_pct
    else:
        base = _PREFIX_SUM.get(phase, 0); weight = _PHASE_WEIGHTS.get(phase, 0)
        overall = base + (within_phase_pct * weight / 100.0)
    overall = max(0.0, min(100.0, overall))
    msg = message or f"{phase}: {within_phase_pct:.2f}% (acumulado {overall:.2f}%)"
    if force_log or (int(overall) % 2 == 0):
        _LOGGER.info(f"[PROGRESS] {msg}")
    _checkpoint_write(phase, overall, extra={"within_phase_pct": within_phase_pct, "message": message})
    try:
        PROGRESS  # noqa
        try:
            if hasattr(PROGRESS, "update") and callable(PROGRESS.update):
                PROGRESS.update(phase=phase, progress=overall, percent=overall, value=overall/100.0, message=message)
            else:
                PROGRESS(phase=phase, progress=overall, percent=overall, value=overall/100.0, message=message)
        except Exception:
            pass
    except NameError:
        pass

def _count_csv_rows(path: str) -> Optional[int]:
    if not SPD_PROGRESS_COUNT_ROWS: return None
    try:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            n = -1
            for n, _ in enumerate(f, 0):
                pass
        return max(0, n)
    except Exception as e:
        _LOGGER.debug(f"No se pudo contar filas de {path}: {e}")
        return None

def _bytes_df(df: pd.DataFrame) -> int:
    try:
        return int(df.memory_usage(deep=True).sum())
    except Exception:
        return 0

def _append_partial_csv(path: str, df: pd.DataFrame, header_written: bool) -> bool:
    if df is None or df.empty: return header_written
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "a" if os.path.exists(path) or header_written else "w"
    need_header = not header_written and mode == "w"
    df.to_csv(path, mode=mode, index=False, header=need_header)
    return header_written or need_header

def _normalize_onpromotion(series: pd.Series) -> pd.Series:
    if series.dtype == bool: return series.astype("int8")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).clip(0,1).astype("int8")
    return series.fillna("False").astype(str).str.lower().isin(["true","1","t","yes"]).astype("int8")

# ------------------- Lecturas base -------------------
def _read_items(items_csv: str) -> pd.DataFrame:
    _LOGGER.info(f"Leyendo items desde {items_csv} ...")
    t0 = time.time()
    cols = pd.read_csv(items_csv, nrows=0).columns.tolist()
    use = [c for c in ["item_nbr","class","family"] if c in cols]
    df = pd.read_csv(items_csv, usecols=use)
    df["item_nbr"] = pd.to_numeric(df["item_nbr"], errors="coerce").astype("int32")
    if "class" in df.columns: df["class"] = pd.to_numeric(df["class"], errors="coerce").astype("int16")
    if "family" in df.columns: df["family"] = df["family"].astype("category")
    _LOGGER.info(f"Items: {len(df):,} filas ({_bytes_df(df)/1e6:.1f} MB) en {time.time()-t0:.2f}s")
    return df

def _read_stores(stores_csv: str) -> pd.DataFrame:
    _LOGGER.info(f"Leyendo stores desde {stores_csv} ...")
    t0 = time.time()
    cols = pd.read_csv(stores_csv, nrows=0).columns.tolist()
    use = [c for c in ["store_nbr","type","city","state"] if c in cols]
    df = pd.read_csv(stores_csv, usecols=use)
    df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").astype("int16")
    if "type" in df.columns: df["type"] = df["type"].astype("category")
    if "city" in df.columns: df["city"] = df["city"].astype("category")
    if "state" in df.columns: df["state"] = df["state"].astype("category")
    _LOGGER.info(f"Stores: {len(df):,} filas ({_bytes_df(df)/1e6:.1f} MB) en {time.time()-t0:.2f}s")
    return df

def _ensure_class_in_H(H_csv: str, items: pd.DataFrame, chunksize: int = CHUNK_H) -> str:
    cols = pd.read_csv(H_csv, nrows=0).columns.tolist()
    if "class" in cols:
        _LOGGER.info("H ya contiene 'class'; no se crea temporal.")
        return H_csv
    _LOGGER.info("H no contiene 'class'. Generando archivo temporal con 'class' ...")
    tmp = os.path.splitext(H_csv)[0] + "_with_class.tmp.csv"
    if os.path.exists(tmp): os.remove(tmp)
    open(tmp, "w").close()
    total_rows = _count_csv_rows(H_csv)
    processed = 0
    reader = pd.read_csv(
        H_csv,
        usecols=[c for c in ["date","store_nbr","item_nbr","H_prop","H_bin"] if c in cols],
        parse_dates=["date"],
        dtype={"store_nbr":"int16","item_nbr":"int32","H_bin":"int8"},
        chunksize=chunksize, low_memory=False
    )
    header = False
    for ch in reader:
        processed += len(ch)
        ch = ch.merge(items[["item_nbr","class"]], on="item_nbr", how="left")
        ch.to_csv(tmp, mode="a", index=False, header=(not header))
        header = True
        if total_rows:
            pct = 100.0 * processed / max(1, total_rows)
            _progress_emit("init", min(99.0, pct), f"Anexando 'class' a H: {processed:,}/{total_rows:,}")
    _progress_emit("init", 100.0, "Archivo temporal de H con 'class' generado.")
    return tmp

def _aggregate_item_stats(H_use: str) -> pd.DataFrame:
    t0 = time.time()
    _LOGGER.info("Agregando métricas por (store,item) desde H ...")
    dtypes = {"store_nbr":"int16","item_nbr":"int32","H_bin":"int8"}
    total_rows = _count_csv_rows(H_use)
    processed = 0
    reader = pd.read_csv(H_use, usecols=["date","store_nbr","item_nbr","H_prop","H_bin"],
                         parse_dates=["date"], dtype=dtypes, chunksize=CHUNK_H, low_memory=False)
    parts = []
    for ch in reader:
        processed += len(ch)
        g = ch.groupby(["store_nbr","item_nbr"], as_index=False).agg(
            n_obs=("H_prop","size"),
            sum_bin=("H_bin","sum"),
            sum_prop=("H_prop","sum"),
            sum_prop2=("H_prop", lambda s: np.square(s).sum())
        )
        parts.append(g)
        if total_rows and not PROGRESS_FLAGS["mute_internal_progress"]:
            pct = 100.0 * processed / max(1, total_rows)
            _progress_emit("aggregate_item_stats", min(99.0, pct), f"H procesadas: {processed:,}/{total_rows:,}")
    agg = (pd.concat(parts, ignore_index=True)
             .groupby(["store_nbr","item_nbr"], as_index=False)
             .agg(n_obs=("n_obs","sum"), sum_bin=("sum_bin","sum"),
                  sum_prop=("sum_prop","sum"), sum_prop2=("sum_prop2","sum")))
    agg["p_any"] = (agg["sum_bin"]/agg["n_obs"]).astype("float32")
    agg["h_mean"] = (agg["sum_prop"]/agg["n_obs"]).astype("float32")
    m2_over_n = (agg["sum_prop2"]/agg["n_obs"]).astype("float64")
    agg["h_sd"] = np.sqrt(np.maximum(m2_over_n - np.square(agg["h_mean"].astype("float64")), 0.0)).astype("float32")
    if not PROGRESS_FLAGS["mute_internal_progress"]:
        _progress_emit("aggregate_item_stats", 100.0, f"Listo en {time.time()-t0:.2f}s")
    return agg

def _promo_stats(train_csv: str) -> pd.DataFrame:
    t0 = time.time()
    _LOGGER.info("Agregando métricas de onpromotion por (store,item) ...")
    dtypes={"store_nbr":"int16","item_nbr":"int32"}
    total_rows = _count_csv_rows(train_csv)
    processed = 0
    reader = pd.read_csv(train_csv, usecols=["date","store_nbr","item_nbr","onpromotion"],
                         parse_dates=["date"], dtype=dtypes, chunksize=CHUNK_TRAIN, low_memory=False)
    parts=[]
    for ch in reader:
        processed += len(ch)
        ch["onpromotion"] = _normalize_onpromotion(ch["onpromotion"])
        g = ch.groupby(["store_nbr","item_nbr"], as_index=False).agg(
            n_obs=("onpromotion","size"),
            n_on=("onpromotion","sum"),
            first_date=("date","min"),
            last_date=("date","max")
        )
        parts.append(g)
        if total_rows and not PROGRESS_FLAGS["mute_internal_progress"]:
            pct = 100.0 * processed / max(1, total_rows)
            _progress_emit("promo_stats", min(99.0, pct), f"train procesado: {processed:,}/{total_rows:,}")
    agg = (pd.concat(parts, ignore_index=True)
             .groupby(["store_nbr","item_nbr"], as_index=False)
             .agg(n_obs=("n_obs","sum"), n_on=("n_on","sum"),
                  first_date=("first_date","min"), last_date=("last_date","max")))
    agg["p_promo"] = (agg["n_on"]/agg["n_obs"]).astype("float32")
    if not PROGRESS_FLAGS["mute_internal_progress"]:
        _progress_emit("promo_stats", 100.0, f"Listo en {time.time()-t0:.2f}s")
    return agg

# ------------------- Cachés por proceso -------------------
_PROC_CACHE_TRAIN: Dict[Tuple[str,int], Dict] = {}
_PROC_CACHE_HSC: "OrderedDict[Tuple[str,int,int], Dict]" = OrderedDict()
def _lru_touch(key):
    try:
        _PROC_CACHE_HSC.move_to_end(key)
    except Exception:
        pass
def _lru_prune():
    while len(_PROC_CACHE_HSC) > max(1, SPD_MAX_HSC_CACHE):
        _PROC_CACHE_HSC.popitem(last=False)

def _runs_to_mask(runs: List[Tuple[pd.Timestamp, pd.Timestamp]], dates_np: np.ndarray) -> np.ndarray:
    if not runs or len(dates_np)==0:
        return np.zeros(len(dates_np), dtype=bool)
    mask = np.zeros(len(dates_np), dtype=bool)
    for a, b in runs:
        a64 = np.datetime64(a.normalize())
        b64 = np.datetime64(b.normalize())
        i = np.searchsorted(dates_np, a64, side="left")
        j = np.searchsorted(dates_np, b64, side="right")
        if i < j:
            mask[i:j] = True
    return mask

def _get_train_cache(train_csv: str, store_nbr: int) -> Dict:
    key = (train_csv, int(store_nbr))
    if key in _PROC_CACHE_TRAIN:
        return _PROC_CACHE_TRAIN[key]

    dtypest = {"store_nbr":"int16","item_nbr":"int32"}
    readerT = pd.read_csv(train_csv, usecols=["date","store_nbr","item_nbr","onpromotion"],
                          parse_dates=["date"], dtype=dtypest, chunksize=CHUNK_TRAIN, low_memory=False)
    parts=[]
    for ch in readerT:
        ch = ch[ch["store_nbr"]==store_nbr]
        if ch.empty: continue
        ch["onpromotion"] = _normalize_onpromotion(ch["onpromotion"])
        ch = ch[(ch["date"]>=WINDOW_START) & (ch["date"]<=WINDOW_END)]
        if not ch.empty:
            parts.append(ch[["date","item_nbr","onpromotion"]])
    if not parts:
        cache = {"runs_by_item": {}, "p_promo_by_item": {}}
        _PROC_CACHE_TRAIN[key] = cache
        return cache

    T_s = pd.concat(parts, ignore_index=True)
    promo = (T_s.groupby("item_nbr")["onpromotion"]
             .agg(n_obs="size", n_on="sum")
             .assign(p_promo=lambda d: (d["n_on"]/d["n_obs"]).astype("float32")))
    p_promo_by_item = promo["p_promo"].to_dict()

    runs_by_item: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for item, grp in T_s.groupby("item_nbr"):
        s = grp.sort_values("date")[["date","onpromotion"]].drop_duplicates("date")
        s["run_id"] = (s["onpromotion"].ne(s["onpromotion"].shift())).cumsum()
        runs = s[s["onpromotion"]==1].groupby("run_id")["date"].agg(["min","max","count"]).reset_index()
        runs = runs[runs["count"]>=MIN_RUN_DAYS]
        if not runs.empty:
            runs_by_item[int(item)] = [(r["min"], r["max"]) for _, r in runs.iterrows()]

    cache = {"runs_by_item": runs_by_item, "p_promo_by_item": p_promo_by_item}
    _PROC_CACHE_TRAIN[key] = cache
    return cache

# ------------------- Sharding por tienda -------------------
def _shard_path(store_nbr: int) -> str:
    global _SHARD_DIR
    return os.path.join(_SHARD_DIR, f"H_store_{int(store_nbr)}.csv.gz")

def _build_H_store_shards(H_use: str, stores_of_interest: List[int]) -> None:
    """Crea shards gzip por tienda (solo para tiendas relevantes). Lee H una sola vez (streaming)."""
    if not SPD_SHARD_H:
        return
    os.makedirs(_SHARD_DIR, exist_ok=True)
    done = set()
    for s in stores_of_interest:
        if os.path.exists(_shard_path(s)):
            done.add(int(s))
    to_do = sorted(set(map(int, stores_of_interest)) - done)
    if not to_do:
        _LOGGER.info("Shards por tienda ya existen; se reutilizarán.")
        return

    _LOGGER.info(f"Construyendo shards por tienda en {_SHARD_DIR} para {len(to_do)} tiendas ...")
    total_rows = _count_csv_rows(H_use)
    processed = 0
    dtypesH = {"store_nbr":"int16","item_nbr":"int32","class":"int16","H_bin":"int8"}
    readerH = pd.read_csv(
        H_use,
        usecols=["date","store_nbr","item_nbr","class","H_prop","H_bin"],
        parse_dates=["date"], dtype=dtypesH,
        chunksize=CHUNK_H, low_memory=False
    )
    headers_written: Dict[int,bool] = defaultdict(lambda: False)
    stores_set = set(to_do)
    for ch in readerH:
        processed += len(ch)
        ch = ch[ch["store_nbr"].isin(stores_set)]
        if ch.empty:
            if total_rows:
                pct = 100.0 * processed / max(1, total_rows)
                _progress_emit("score_cannibals", min(10.0, pct*0.10), f"Preparando shards: {processed:,}/{total_rows:,}")
            continue
        for s, df_s in ch.groupby("store_nbr"):
            path = _shard_path(int(s))
            mode = "ab" if os.path.exists(path) or headers_written[int(s)] else "wb"
            with gzip.open(path, mode) as gz:
                df_s.to_csv(gz, index=False, header=(not headers_written[int(s)]))
            headers_written[int(s)] = True

        if total_rows:
            pct = 100.0 * processed / max(1, total_rows)
            _progress_emit("score_cannibals", min(10.0, pct*0.10), f"Preparando shards: {processed:,}/{total_rows:,}")
    _LOGGER.info("Shards por tienda listos.")

# ------------------- Utilidades para leer shard de tienda UNA VEZ -------------------
def _read_H_store_df(H_use: str, store_nbr: int) -> pd.DataFrame:
    """Lee el shard de la tienda si existe; de lo contrario filtra H_use por la tienda. Devuelve DataFrame con date,item_nbr,class,H_prop,H_bin."""
    dtypesH = {"store_nbr":"int16","item_nbr":"int32","class":"int16","H_bin":"int8"}
    usecols = ["date","store_nbr","item_nbr","class","H_prop","H_bin"]
    parse_dates = ["date"]
    path = _shard_path(store_nbr) if _SHARD_DIR and os.path.exists(_shard_path(store_nbr)) else H_use
    parts=[]
    reader = pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, dtype=dtypesH, chunksize=CHUNK_H, low_memory=False)
    for ch in reader:
        ch = ch[ch["store_nbr"]==store_nbr]
        if not ch.empty:
            parts.append(ch[["date","item_nbr","class","H_prop","H_bin"]])
    if not parts:
        return pd.DataFrame(columns=["date","item_nbr","class","H_prop","H_bin"])
    H_s = pd.concat(parts, ignore_index=True)
    # Consolidar por (date,item) si hay duplicados
    H_s = (H_s.groupby(["date","item_nbr","class"], as_index=False)
               .agg(H_prop=("H_prop","mean"), H_bin=("H_bin","max")))
    return H_s

def _build_group_cache_from_store_df(H_s: pd.DataFrame, cls: int) -> Dict:
    """Replica _get_H_group_cache pero usando H_s (DataFrame de tienda ya en memoria)."""
    df = H_s[H_s["class"]==cls]
    if df.empty:
        return {"dates_np": np.array([], dtype="datetime64[D]"),
                "items": np.array([], dtype="int32"),
                "Hmat": np.empty((0,0), dtype="float32"),
                "S": np.array([], dtype="float32"),
                "C": np.array([], dtype="int32"),
                "item_metrics": pd.DataFrame(columns=["item_nbr","n_obs_H","h_mean","h_sd","p_any"])}
    dates = pd.Index(np.sort(df["date"].unique()))
    items = np.sort(df["item_nbr"].unique().astype("int32"))
    date_to_pos = {d:i for i,d in enumerate(dates)}
    item_to_pos = {itm:i for i,itm in enumerate(items)}
    T = len(dates); N = len(items)

    Hmat = np.full((T, N), np.nan, dtype="float32")
    r = df["date"].map(date_to_pos).to_numpy(np.int64, copy=False)
    c = df["item_nbr"].map(item_to_pos).to_numpy(np.int64, copy=False)
    v = df["H_prop"].astype("float32").to_numpy(copy=False)
    Hmat[r, c] = v

    S = np.nan_to_num(Hmat, nan=0.0).sum(axis=1, dtype="float32")
    C = np.sum(~np.isnan(Hmat), axis=1, dtype="int32")

    item_stats = (df.groupby("item_nbr").agg(
        n_obs_H=("H_prop","size"),
        h_mean=("H_prop","mean"),
        h_sd=("H_prop","std"),
        p_any=("H_bin","mean"))
        .reset_index())
    item_stats["h_sd"] = item_stats["h_sd"].fillna(0.0).astype("float32")
    item_stats["h_mean"] = item_stats["h_mean"].astype("float32")
    item_stats["p_any"] = item_stats["p_any"].astype("float32")
    item_stats["item_nbr"] = item_stats["item_nbr"].astype("int32")

    return {
        "dates_np": dates.to_numpy(dtype="datetime64[D]"),
        "items": items,
        "Hmat": Hmat,
        "S": S,
        "C": C,
        "item_metrics": item_stats,
    }

# ------------------- Cache de grupo (legacy, preferimos el de tienda) -------------------
def _get_H_group_cache(H_use: str, store_nbr: int, cls: int) -> Dict:
    key = (H_use, int(store_nbr), int(cls))
    if key in _PROC_CACHE_HSC:
        _lru_touch(key)
        return _PROC_CACHE_HSC[key]

    # Fuente: shard de tienda (si existe) o CSV grande
    dtypesH = {"store_nbr":"int16","item_nbr":"int32","class":"int16","H_bin":"int8"}
    usecols = ["date","store_nbr","item_nbr","class","H_prop","H_bin"]
    parse_dates = ["date"]
    path = _shard_path(store_nbr) if _SHARD_DIR and os.path.exists(_shard_path(store_nbr)) else H_use

    parts=[]
    reader = pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, dtype=dtypesH, chunksize=CHUNK_H, low_memory=False)
    for ch in reader:
        ch = ch[(ch["store_nbr"]==store_nbr) & (ch["class"]==cls)]
        if not ch.empty:
            parts.append(ch[["date","item_nbr","H_prop","H_bin"]])
    if not parts:
        cache = {"dates_np": np.array([], dtype="datetime64[D]"),
                 "items": np.array([], dtype="int32"),
                 "Hmat": np.empty((0,0), dtype="float32"),
                 "S": np.array([], dtype="float32"),
                 "C": np.array([], dtype="int32"),
                 "item_metrics": pd.DataFrame(columns=["item_nbr","n_obs_H","h_mean","h_sd","p_any"])}
        _PROC_CACHE_HSC[key] = cache; _lru_touch(key); _lru_prune()
        return cache

    H_sc = pd.concat(parts, ignore_index=True)
    H_sc = (H_sc.groupby(["date","item_nbr"], as_index=False)
                  .agg(H_prop=("H_prop","mean"), H_bin=("H_bin","max")))

    dates = pd.Index(np.sort(H_sc["date"].unique()))
    items = np.sort(H_sc["item_nbr"].unique().astype("int32"))
    date_to_pos = {d:i for i,d in enumerate(dates)}
    item_to_pos = {itm:i for i,itm in enumerate(items)}
    T = len(dates); N = len(items)

    Hmat = np.full((T, N), np.nan, dtype="float32")
    r = H_sc["date"].map(date_to_pos).to_numpy(np.int64, copy=False)
    c = H_sc["item_nbr"].map(item_to_pos).to_numpy(np.int64, copy=False)
    v = H_sc["H_prop"].astype("float32").to_numpy(copy=False)
    Hmat[r, c] = v

    S = np.nan_to_num(Hmat, nan=0.0).sum(axis=1, dtype="float32")
    C = np.sum(~np.isnan(Hmat), axis=1, dtype="int32")

    item_stats = (H_sc.groupby("item_nbr").agg(
        n_obs_H=("H_prop","size"),
        h_mean=("H_prop","mean"),
        h_sd=("H_prop","std"),
        p_any=("H_bin","mean"))
        .reset_index())
    item_stats["h_sd"] = item_stats["h_sd"].fillna(0.0).astype("float32")
    item_stats["h_mean"] = item_stats["h_mean"].astype("float32")
    item_stats["p_any"] = item_stats["p_any"].astype("float32")
    item_stats["item_nbr"] = item_stats["item_nbr"].astype("int32")

    cache = {
        "dates_np": dates.to_numpy(dtype="datetime64[D]"),
        "items": items,
        "Hmat": Hmat,
        "S": S,
        "C": C,
        "item_metrics": item_stats,
    }
    _PROC_CACHE_HSC[key] = cache; _lru_touch(key); _lru_prune()
    return cache

def _ensure_J_ON_in_group_cache(H_use: str, train_csv: str, store_nbr: int, cls: int) -> Dict:
    g = _get_H_group_cache(H_use, store_nbr, cls)
    if g["Hmat"].size == 0:
        return g
    if "J_ON" in g:
        return g
    train = _get_train_cache(train_csv, store_nbr)
    runs_map = train["runs_by_item"]
    dates_np = g["dates_np"]; items = g["items"]
    T, N = g["Hmat"].shape
    J_ON = np.zeros((T, N), dtype=bool)
    for j_idx, item in enumerate(items):
        runs = runs_map.get(int(item), [])
        if runs:
            J_ON[:, j_idx] = _runs_to_mask(runs, dates_np)
    g["J_ON"] = J_ON
    return g

# ------------------- API compatibles (se mantienen) -------------------
def _runs_for_item(train_csv: str, store_nbr: int, item_nbr: int,
                   start: pd.Timestamp = WINDOW_START,
                   end: pd.Timestamp = WINDOW_END) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    cache = _get_train_cache(train_csv, store_nbr)
    return cache["runs_by_item"].get(int(item_nbr), [])

def _class_mean_H_excluding_i(H_use: str, s: int, c: int, i_item: int) -> pd.DataFrame:
    g = _get_H_group_cache(H_use, s, c)
    if g["Hmat"].size == 0:
        return pd.DataFrame(columns=["date","H_prop"])
    items = g["items"]; H = g["Hmat"]; S = g["S"]; C = g["C"]; dates_np = g["dates_np"]
    try:
        j = int(np.where(items == int(i_item))[0][0])
    except Exception:
        return pd.DataFrame({"date": pd.to_datetime(dates_np), "H_prop": np.nan})
    col = H[:, j]
    denom = (C - 1).astype("float32")
    valid = (denom > 0) & ~np.isnan(col)
    ex_mean = np.full_like(col, np.nan, dtype="float32")
    ex_mean[valid] = (S[valid] - col[valid]) / denom[valid]
    return pd.DataFrame({"date": pd.to_datetime(dates_np), "H_prop": ex_mean})

def _compute_windows(treat_start: pd.Timestamp, run_end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    treat_end = min(run_end, treat_start + pd.Timedelta(days=TREAT_MAX-1))
    pre_start = treat_start - pd.Timedelta(days=PRE_GAP + PRE_DAYS)
    post_start = treat_end + pd.Timedelta(days=1)
    post_end = post_start + pd.Timedelta(days=POST_DAYS-1)
    return pre_start.normalize(), treat_start.normalize(), post_start.normalize(), post_end.normalize()

def _aggregate_item_profile(H_use: str) -> pd.DataFrame:
    # Legacy: se mantiene pero en V3 reutilizamos el item_stats de la fase inicial para donantes.
    agg = _aggregate_item_stats(H_use)[["store_nbr","item_nbr","h_mean","h_sd","p_any"]]
    return agg

# ------------------- Workers originales (compatibilidad) -------------------
def _score_cannibal_worker(args) -> Optional[Dict]:
    # (no usado en V3, se mantiene por compatibilidad)
    H_use, train_csv, s, i_item, c = args
    try:
        g = _get_H_group_cache(H_use, s, c)
        if g["Hmat"].size == 0:  return None
        items = g["items"]; H = g["Hmat"]; S = g["S"]; C = g["C"]; dates_np = g["dates_np"]
        train = _get_train_cache(train_csv, s)
        runs = train["runs_by_item"].get(int(i_item), [])
        if not runs: return None
        try:
            j = int(np.where(items == int(i_item))[0][0])
        except Exception:
            return None
        col = H[:, j]
        denom = (C - 1).astype("float32")
        valid = (denom > 0) & ~np.isnan(col)
        if not valid.any(): return None
        i_on = _runs_to_mask(runs, dates_np)
        n_on = int((i_on & valid).sum()); n_off = int((~i_on & valid).sum())
        if n_on == 0 or n_off == 0: return None
        ex_mean = np.full_like(col, np.nan, dtype="float32")
        ex_mean[valid] = (S[valid] - col[valid]) / denom[valid]
        mean_on  = float(np.nanmean(ex_mean[i_on]))
        mean_off = float(np.nanmean(ex_mean[~i_on]))
        delta_cls = mean_on - mean_off
        fr_start, fr_end = runs[0]
        return {"store_nbr": int(s), "item_nbr": int(i_item), "class": int(c),
                "delta_H_cls": float(delta_cls),
                "first_run_start": fr_start, "first_run_end": fr_end}
    except Exception as e:
        _LOGGER.debug(f"[Worker score_i] Error s={s}, i={i_item}, c={c}: {e}")
        return None

def _victims_for_i_worker(args) -> List[Dict]:
    H_use, train_csv, s, i_item, c, n_victims = args
    try:
        g = _ensure_J_ON_in_group_cache(H_use, train_csv, s, c)
        if g["Hmat"].size == 0:
            return []
        items = g["items"]; H = g["Hmat"]; J_ON = g["J_ON"]; dates_np = g["dates_np"]

        train = _get_train_cache(train_csv, s)
        runs_i = train["runs_by_item"].get(int(i_item), [])
        if not runs_i:
            return []
        fr_start, fr_end = runs_i[0]
        pre_start, treat_start, post_start, post_end = _compute_windows(fr_start, fr_end)
        try:
            i_col = int(np.where(items == int(i_item))[0][0])
        except Exception:
            return []

        i_on = _runs_to_mask(runs_i, dates_np)

        base_sc = g["item_metrics"].copy()
        p_promo_map = train["p_promo_by_item"]
        base_sc["p_promo"] = base_sc["item_nbr"].map(lambda x: p_promo_map.get(int(x), 0.0)).astype("float32")

        base_sc = base_sc[(base_sc["n_obs_H"] >= MIN_ITEM_OBS) &
                          (base_sc["h_sd"]   >= H_SD_MIN) &
                          (base_sc["p_any"].between(P_ANY_MIN, P_ANY_MAX))]
        cand_j = base_sc[(base_sc["item_nbr"] != int(i_item)) & (base_sc["p_promo"] <= P_PROMO_J_MAX)]
        if cand_j.empty:
            return []

        # --- Muestreo en j para acelerar ---
        if SPD_SAMPLE_J_FRAC < 1.0 or SPD_SAMPLE_J_CAP > 0:
            np.random.seed(SPD_SEED)
            n_frac = math.ceil(len(cand_j) * max(0.0, min(1.0, SPD_SAMPLE_J_FRAC)))
            if SPD_SAMPLE_J_CAP > 0:
                n_frac = min(n_frac, SPD_SAMPLE_J_CAP)
            if n_frac > 0 and n_frac < len(cand_j):
                cand_j = cand_j.sample(n=n_frac, replace=False, random_state=SPD_SEED)

        j_items = cand_j["item_nbr"].to_numpy(dtype="int32")
        idx_map = {int(itm): int(pos) for pos, itm in enumerate(items)}
        j_cols = np.array([idx_map.get(int(itm), -1) for itm in j_items], dtype=int)
        valid_cols_mask = (j_cols >= 0)
        if not valid_cols_mask.any():
            return []
        j_items = j_items[valid_cols_mask]
        j_cols  = j_cols[valid_cols_mask]

        Hj = H[:, j_cols]
        Jj_on = J_ON[:, j_cols]
        valid_val = np.isfinite(Hj)

        if SPD_ALLOW_J_PROMO_OVERLAP:
            m_on  = (i_on[:, None]) & valid_val
            m_off = (~i_on[:, None]) & valid_val
        else:
            m_on  = (i_on[:, None]) & (~Jj_on) & valid_val
            m_off = (~i_on[:, None]) & (~Jj_on) & valid_val

        cnt_on  = m_on.sum(axis=0)
        cnt_off = m_off.sum(axis=0)

        safe_on  = np.where(m_on, Hj, np.nan)
        safe_off = np.where(m_off, Hj, np.nan)

        with np.errstate(invalid="ignore"):
            mean_on  = np.nanmean(safe_on,  axis=0)
            mean_off = np.nanmean(safe_off, axis=0)
        delta = mean_on - mean_off

        order = np.argsort(-delta)
        take = int(min(n_victims, order.size))
        sel = order[:take]

        victims_rows: List[Dict] = []
        for k in sel:
            if not (cnt_on[k] > 0 and cnt_off[k] > 0):
                continue
            victims_rows.append({
                "i_store": int(s),
                "i_item": int(i_item),
                "class": int(c),
                "j_store": int(s),
                "j_item": int(j_items[k]),
                "delta_H_j": float(delta[k]),
                "n_obs_i_on": int(cnt_on[k]),
                "n_obs_i_off": int(cnt_off[k]),
                "pre_start": pre_start.date(),
                "treat_start": treat_start.date(),
                "post_start": post_start.date(),
                "post_end": post_end.date()
            })
        return victims_rows
    except Exception as e:
        _LOGGER.debug(f"[Worker victims] Error s={s}, i={i_item}, c={c}: {e}")
        return []

# ------------------- NUEVO worker por TIENDA (ultra-vectorizado) -------------------
def _score_cannibals_store_worker(args) -> List[Dict]:
    """
    Worker por TIENDA:
      - Lee shard de H de esa tienda UNA VEZ
      - Secciona por clase en memoria
      - Para cada clase -> computa ΔH_cls para TODOS los i candidatos de esa clase (vectorizado)
    args: (H_use, train_csv, store_nbr, class_to_items_map)
    """
    H_use, train_csv, s, class_to_items = args
    out_rows: List[Dict] = []
    try:
        # Cache de runs/p_promo para la tienda
        train = _get_train_cache(train_csv, s)
        runs_map = train["runs_by_item"]

        # Lee H de la tienda una vez
        H_s = _read_H_store_df(H_use, s)
        if H_s.empty:
            return out_rows

        # Recorre SOLO las clases con items candidatos i
        for c, items_i in class_to_items.items():
            if not items_i:
                continue
            g = _build_group_cache_from_store_df(H_s, c)
            if g["Hmat"].size == 0:
                continue

            dates_np = g["dates_np"]; items = g["items"]; H = g["Hmat"]; S = g["S"]; C = g["C"]
            denom = (C - 1).astype("float32")

            # Mapeo item->col y selección de i presentes con runs válidos
            idx_map = {int(itm): int(pos) for pos, itm in enumerate(items)}
            present = []
            for i_item in items_i:
                col_idx = idx_map.get(int(i_item), -1)
                if col_idx < 0:
                    continue
                runs = runs_map.get(int(i_item), [])
                if not runs:
                    continue
                present.append((i_item, col_idx, runs))
            if not present:
                continue

            # Construir matrices para TODOS los i de la clase
            I = len(present)
            col_idx_arr = np.array([p[1] for p in present], dtype=int)  # columnas de H de cada i
            H_i = H[:, col_idx_arr]  # (T, I)

            # Matriz de "encendido" por i
            Tt = len(dates_np)
            OnMat = np.zeros((Tt, I), dtype=bool)
            for k, (_, _, runs) in enumerate(present):
                OnMat[:, k] = _runs_to_mask(runs, dates_np)

            # Máscara de validez para cada (t,i)
            denom_ok = (denom > 0)
            valid = denom_ok[:, None] & np.isfinite(H_i)

            if SPD_FAST_SCORING >= 2:
                # Aproximación ultrarrápida: usar media de clase M = S/C (incluye i)
                with np.errstate(divide="ignore", invalid="ignore"):
                    M = np.where(C>0, S / C.astype("float32"), np.nan).astype("float32")
                # mean_on/mean_off por i usando OnMat
                M_on  = np.nanmean(np.where(OnMat,  M[:, None], np.nan), axis=0)
                M_off = np.nanmean(np.where(~OnMat, M[:, None], np.nan), axis=0)
                delta_vec = (M_on - M_off).astype("float32")
            else:
                # Exacto vectorizado: ex_mean = (S - H_i)/(C - 1)
                ex = np.full_like(H_i, np.nan, dtype="float32")
                ex[valid] = (S[:, None][valid] - H_i[valid]) / (denom[:, None][valid])
                mean_on  = np.nanmean(np.where(OnMat,  ex, np.nan), axis=0)
                mean_off = np.nanmean(np.where(~OnMat, ex, np.nan), axis=0)
                delta_vec = (mean_on - mean_off).astype("float32")

            # Derivar también primer episodio (para ventanas)
            for k, (i_item, _, runs) in enumerate(present):
                i_on = OnMat[:, k]
                if SPD_FAST_SCORING >= 2:
                    valid_ref = np.isfinite(M)
                else:
                    valid_ref = np.isfinite(H_i[:, k])
                n_on  = int(np.sum(i_on & valid_ref))
                n_off = int(np.sum((~i_on) & valid_ref))
                if n_on == 0 or n_off == 0:
                    continue
                fr_start, fr_end = runs[0]
                out_rows.append({
                    "store_nbr": int(s),
                    "item_nbr": int(i_item),
                    "class": int(c),
                    "delta_H_cls": float(delta_vec[k]),
                    "first_run_start": fr_start,
                    "first_run_end": fr_end
                })
        return out_rows
    except Exception as e:
        _LOGGER.debug(f"[Worker score_store] Error s={s}: {e}")
        return out_rows

# ------------------- Pipeline principal -------------------
def select_pairs_and_donors(H_csv: str, train_csv: str, items_csv: str, stores_csv: str,
                            outdir: str = "data") -> Tuple[str, str]:
    t0_all = time.time()
    os.makedirs(outdir, exist_ok=True)
    _setup_logging(outdir)
    _LOGGER.info("=== Inicio select_pairs_and_donors (optimizado V3+) ===")
    _LOGGER.info(f"Parámetros: N_WORKERS={N_WORKERS}, CHUNK_H={CHUNK_H:,}, CHUNK_TRAIN={CHUNK_TRAIN:,}")
    _LOGGER.info(f"ENV sampling: SPD_SAMPLE_CI_PER_SC={SPD_SAMPLE_CI_PER_SC}, SPD_SAMPLE_FRAC_CI={SPD_SAMPLE_FRAC_CI}, "
                 f"SPD_MAX_TASKS_SCORE={SPD_MAX_TASKS_SCORE}, SPD_FAST_SCORING={SPD_FAST_SCORING}, "
                 f"SPD_SAMPLE_J_FRAC={SPD_SAMPLE_J_FRAC}, SPD_SAMPLE_J_CAP={SPD_SAMPLE_J_CAP}, "
                 f"SPD_ALLOW_J_PROMO_OVERLAP={int(SPD_ALLOW_J_PROMO_OVERLAP)}, "
                 f"SPD_MIN_ITEMS_PER_SC={SPD_MIN_ITEMS_PER_SC}, SPD_REQUIRE_MIN_J_PER_SC={SPD_REQUIRE_MIN_J_PER_SC}")

    global _CHECKPOINT_PATH, _SHARD_DIR
    _CHECKPOINT_PATH = os.path.join(outdir, "progress.jsonl")
    _SHARD_DIR = os.path.join(outdir, "shards")
    try:
        if os.path.exists(_CHECKPOINT_PATH):
            os.remove(_CHECKPOINT_PATH)
    except Exception:
        pass

    # --- INIT
    _progress_emit("init", 0.0, "Inicializando insumos ...", force_log=True)
    items  = _read_items(items_csv)
    stores = _read_stores(stores_csv)
    _progress_emit("init", 35.0, "Items y Stores cargados.")
    H_use  = _ensure_class_in_H(H_csv, items)
    _progress_emit("init", 100.0, "H listo para uso.")

    # --- 1) Universo base
    item_stats = _aggregate_item_stats(H_use)
    promo = _promo_stats(train_csv)

    # Guardar depuración antes de filtros
    if SPD_SAVE_DEBUG_TABLES:
        dbg_dir = os.path.join(outdir, "debug"); os.makedirs(dbg_dir, exist_ok=True)
        base_pre = (item_stats.merge(promo[["store_nbr","item_nbr","p_promo","first_date","last_date","n_obs"]],
                                     on=["store_nbr","item_nbr"], how="left")
                            .merge(items[["item_nbr","class"]], on="item_nbr", how="left"))
        base_pre.to_csv(os.path.join(dbg_dir, "base_universe_pre_filters.csv"), index=False)

    _progress_emit("build_base", 0.0, "Construyendo universo base y filtros ...")
    base = (item_stats.merge(promo[["store_nbr","item_nbr","p_promo","first_date","last_date","n_obs"]],
                             on=["store_nbr","item_nbr"], how="left")
                     .merge(items[["item_nbr","class"]], on="item_nbr", how="left"))
    base = base[(base["n_obs_x"]>=MIN_ITEM_OBS) & (base["h_sd"]>=H_SD_MIN) &
                (base["p_any"].between(P_ANY_MIN, P_ANY_MAX))].rename(
                    columns={"n_obs_x":"n_obs_H","n_obs_y":"n_obs_train"})
    _LOGGER.info(f"Base filtrada: {len(base):,} (store,item) válidos.")

    # -------- NUEVO: Filtro por tamaño de grupo (store,class) y pool mínimo de j --------
    # Tamaño de grupo
    if "n_items_sc" not in base.columns:
        sc_counts = (base.groupby(["store_nbr","class"])
                     .size().rename("n_items_sc").reset_index())
        base = base.merge(sc_counts, on=["store_nbr","class"], how="left")

    if SPD_MIN_ITEMS_PER_SC > 0:
        before = len(base)
        base = base[base["n_items_sc"] >= SPD_MIN_ITEMS_PER_SC]
        _LOGGER.info(f"Filtro SPD_MIN_ITEMS_PER_SC={SPD_MIN_ITEMS_PER_SC}: {before:,} -> {len(base):,}")

    # Pool de j por grupo según P_PROMO_J_MAX
    if SPD_REQUIRE_MIN_J_PER_SC > 0:
        j_pool = (base[base["p_promo"] <= P_PROMO_J_MAX]
                  .groupby(["store_nbr","class"])
                  .size().rename("n_j_sc").reset_index())
        base = base.merge(j_pool, on=["store_nbr","class"], how="left")
        base["n_j_sc"] = base["n_j_sc"].fillna(0).astype(int)
        before = len(base)
        base = base[base["n_j_sc"] >= SPD_REQUIRE_MIN_J_PER_SC]
        _LOGGER.info(f"Filtro SPD_REQUIRE_MIN_J_PER_SC={SPD_REQUIRE_MIN_J_PER_SC}: {before:,} -> {len(base):,}")

    if SPD_SAVE_DEBUG_TABLES:
        dbg_dir = os.path.join(outdir, "debug"); os.makedirs(dbg_dir, exist_ok=True)
        cols = ["store_nbr","class","n_items_sc"] + (["n_j_sc"] if "n_j_sc" in base.columns else [])
        base[cols].drop_duplicates().to_csv(os.path.join(dbg_dir, "sc_group_counts.csv"), index=False)
        base.to_csv(os.path.join(dbg_dir, "base_after_filters.csv"), index=False)

    # --- Derivar candidatos i tras filtros de grupo
    cand_i = base[base["p_promo"].between(P_PROMO_I_MIN, P_PROMO_I_MAX)].copy()

    # Ajuste por i: asegura pool j efectivo (descarta si i contaría como j y deja < requeridos)
    if SPD_REQUIRE_MIN_J_PER_SC > 0:
        if "n_j_sc" not in cand_i.columns:
            j_pool = (base[base["p_promo"] <= P_PROMO_J_MAX]
                      .groupby(["store_nbr","class"])
                      .size().rename("n_j_sc").reset_index())
            cand_i = cand_i.merge(j_pool, on=["store_nbr","class"], how="left")
            cand_i["n_j_sc"] = cand_i["n_j_sc"].fillna(0).astype(int)

        cand_i["n_j_effective"] = np.where(
            cand_i["p_promo"] <= P_PROMO_J_MAX,
            cand_i["n_j_sc"] - 1,
            cand_i["n_j_sc"]
        )
        before_ci = len(cand_i)
        cand_i = cand_i[cand_i["n_j_effective"] >= SPD_REQUIRE_MIN_J_PER_SC].drop(columns=["n_j_effective"])
        _LOGGER.info(f"Ajuste por i (pool j efectivo ≥ {SPD_REQUIRE_MIN_J_PER_SC}): {before_ci:,} -> {len(cand_i):,}")

    if SPD_SAVE_DEBUG_TABLES:
        dbg_dir = os.path.join(outdir, "debug")
        cand_i.to_csv(os.path.join(dbg_dir, "cand_i_before_sampling.csv"), index=False)

    # --- Muestreo estratificado (sin groupby.apply)
    if SPD_SAMPLE_FRAC_CI < 1.0 or SPD_SAMPLE_CI_PER_SC > 0:
        np.random.seed(SPD_SEED)
        groups = cand_i.groupby(["store_nbr","class"], sort=False)
        sampled_parts = []
        for (s, c), gdf in groups:
            df = gdf
            if SPD_SAMPLE_FRAC_CI < 1.0 and len(df) > 0:
                n_frac = max(1, int(math.ceil(len(df) * SPD_SAMPLE_FRAC_CI)))
                df = df.sample(n=n_frac, replace=False, random_state=SPD_SEED)
            if SPD_SAMPLE_CI_PER_SC > 0 and len(df) > SPD_SAMPLE_CI_PER_SC:
                df = df.sample(n=SPD_SAMPLE_CI_PER_SC, replace=False, random_state=SPD_SEED)
            sampled_parts.append(df)
        cand_i = pd.concat(sampled_parts, ignore_index=True)
        _LOGGER.info(f"Candidatos i tras muestreo: {len(cand_i):,}")

    if SPD_SAVE_DEBUG_TABLES:
        dbg_dir = os.path.join(outdir, "debug")
        cand_i.to_csv(os.path.join(dbg_dir, "cand_i_after_sampling.csv"), index=False)

    # Ordenar por (store,class) para mejorar cachés
    cand_i = cand_i.sort_values(["store_nbr","class","item_nbr"]).reset_index(drop=True)

    if SPD_MAX_TASKS_SCORE > 0 and len(cand_i) > SPD_MAX_TASKS_SCORE:
        cand_i = cand_i.iloc[:SPD_MAX_TASKS_SCORE].copy()
        _LOGGER.info(f"Candidatos i truncados por SPD_MAX_TASKS_SCORE={SPD_MAX_TASKS_SCORE}: {len(cand_i):,}")

    _LOGGER.info(f"Candidatos i: {len(cand_i):,} (criterios p_promo en [{P_PROMO_I_MIN}, {P_PROMO_I_MAX}])")
    _progress_emit("build_base", 100.0, "Universo base construido.")

    # --- Sharding por TIENDA (una sola pasada de H)
    stores_of_interest = sorted(cand_i["store_nbr"].unique().astype(int).tolist())
    if SPD_SHARD_H and stores_of_interest:
        _progress_emit("score_cannibals", 0.0, "Preparando shards de H por tienda ...", force_log=True)
        _build_H_store_shards(H_use, stores_of_interest)

    # --- 2) Scoring ΔH_cls por TIENDA (vectorizado)
    _progress_emit("score_cannibals", 12.0, "Scoring caníbales (ΔH_cls) ...", force_log=True)
    cannibals: List[Dict] = []

    # Mapa tienda -> { clase -> [items i] }
    store_map: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for _, r in cand_i.iterrows():
        store_map[int(r["store_nbr"])][int(r["class"])].append(int(r["item_nbr"]))

    store_tasks = [(H_use, train_csv, s, class_items) for s, class_items in store_map.items()]
    total_i = int(len(cand_i))
    processed_i_est = 0

    n_workers = max(1, min(N_WORKERS, len(store_tasks))) if store_tasks else 1
    if store_tasks:
        with mp.Pool(processes=n_workers) as pool:
            for rows in pool.imap_unordered(_score_cannibals_store_worker, store_tasks, chunksize=1):
                if rows:
                    cannibals.extend(rows)
                    processed_i_est += len(rows)  # aproximación
                pct = 12.0 + (processed_i_est / max(1, total_i)) * 88.0
                _progress_emit("score_cannibals", min(99.0, pct),
                               f"Caníbales procesados (aprox ítems): {min(processed_i_est,total_i)}/{total_i}")

    if not cannibals:
        _progress_emit("score_cannibals", 100.0, "No se encontraron caníbales con episodios válidos.", force_log=True)
        raise RuntimeError("No se encontraron caníbales con episodios y ΔH_cls en la ventana.")

    can_all_df = pd.DataFrame(cannibals).sort_values("delta_H_cls", ascending=False).reset_index(drop=True)
    if SPD_SAVE_DEBUG_TABLES:
        os.makedirs(os.path.join(outdir, "debug"), exist_ok=True)
        can_all_df.to_csv(os.path.join(outdir, "debug", "cannibals_scored_all.csv"), index=False)

    can_df = can_all_df.head(N_CANNIBALS).copy()
    _LOGGER.info(f"Top caníbales: {len(can_df)} (N_CANNIBALS={N_CANNIBALS})")
    _progress_emit("score_cannibals", 100.0, "Scoring caníbales completado.")

    # --- 3) Víctimas por i (paralelo, N_CANNIBALS suele ser pequeño)
    _progress_emit("select_victims", 0.0, "Buscando víctimas por caníbal i ...", force_log=True)
    victims_rows: List[Dict] = []
    pairs_partial_path = os.path.join(outdir, "pairs_windows.partial.csv")
    try:
        if os.path.exists(pairs_partial_path):
            os.remove(pairs_partial_path)
    except Exception:
        pass
    pairs_header_written = False

    if not can_df.empty:
        v_tasks = [(H_use, train_csv, int(r["store_nbr"]), int(r["item_nbr"]), int(r["class"]), N_VICTIMS_PER_I)
                   for _, r in can_df.iterrows()]
        total_v = len(v_tasks)
        done_v = 0
        v_tasks.sort(key=lambda t: (t[2], t[4], t[3]))  # (store, class, item)
        with mp.Pool(processes=max(1, min(N_WORKERS, total_v))) as pool:
            for rows in pool.imap_unordered(_victims_for_i_worker, v_tasks, chunksize=1):
                done_v += 1
                if rows:
                    victims_rows.extend(rows)
                    chunk_df = pd.DataFrame(rows)
                    pairs_header_written = _append_partial_csv(pairs_partial_path, chunk_df, pairs_header_written)
                pct = 100.0 * done_v / max(1, total_v)
                _progress_emit("select_victims", min(99.0, pct),
                               f"Víctimas procesadas para i: {done_v}/{total_v}")

    pairs_df = pd.DataFrame(victims_rows)
    if pairs_df.empty:
        _LOGGER.warning("No se lograron víctimas; se generará CSV vacío.")
        pairs_df = pd.DataFrame(columns=[
            "i_store","i_item","class","j_store","j_item","delta_H_j",
            "n_obs_i_on","n_obs_i_off","pre_start","treat_start","post_start","post_end"
        ])
    else:
        pairs_df = pairs_df.sort_values(["i_store","i_item","delta_H_j"], ascending=[True,True,False])
    pairs_path = os.path.join(outdir, "pairs_windows.csv")

    _progress_emit("write_pairs", 0.0, "Escribiendo pairs_windows.csv ...")
    pairs_df.to_csv(pairs_path, index=False)
    _progress_emit("write_pairs", 100.0, f"pairs_windows.csv listo ({len(pairs_df):,} filas).")

    # --- 4) Donantes (reutiliza item_stats para no recalcular)
    _progress_emit("select_donors", 0.0, "Seleccionando donantes por víctima ...", force_log=True)
    donors_rows=[]
    PROGRESS_FLAGS["mute_internal_progress"] = True
    profiles = item_stats[["store_nbr","item_nbr","h_mean","h_sd","p_any"]].copy()
    PROGRESS_FLAGS["mute_internal_progress"] = False

    meta = _read_stores(stores_csv)
    w = np.array([0.5, 0.2, 0.3], dtype=float)

    donors_partial_path = os.path.join(outdir, "donors_per_victim.partial.csv")
    try:
        if os.path.exists(donors_partial_path):
            os.remove(donors_partial_path)
    except Exception:
        pass
    donors_header_written = False

    total_pairs = len(pairs_df)
    processed_pairs = 0
    for _, v in pairs_df.iterrows():
        s, j, c = int(v["j_store"]), int(v["j_item"]), int(v["class"])
        processed_pairs += 1
        cand_same_item = profiles[profiles["item_nbr"]==j].merge(meta, on="store_nbr", how="left")
        cand_same_item = cand_same_item[cand_same_item["store_nbr"]!=s]
        if not cand_same_item.empty:
            pj = profiles[(profiles["store_nbr"]==s) & (profiles["item_nbr"]==j)]
            if not pj.empty:
                pj = pj.iloc[0]
                tgt = np.array([pj["h_mean"], pj["h_sd"], pj["p_any"]], dtype=float)
                X = cand_same_item[["h_mean","h_sd","p_any"]].to_numpy(dtype=float)
                d = np.sqrt(((X - tgt) ** 2 * w).sum(axis=1))
                cand_same_item = cand_same_item.assign(distance=d)
                cand_same_item = cand_same_item.sort_values(["distance","state","city"], ascending=[True,True,True])
                top_si = cand_same_item.head(N_DONORS_PER_J).copy()
                rank = 1
                for _, r in top_si.iterrows():
                    donors_rows.append({
                        "j_store": s, "j_item": j,
                        "donor_store": int(r["store_nbr"]), "donor_item": j,
                        "donor_kind": "same_item",
                        "distance": float(r["distance"]),
                        "store_type": str(r.get("type","")), "city": str(r.get("city","")), "state": str(r.get("state","")),
                        "rank": rank
                    })
                    rank += 1
                if rank <= N_DONORS_PER_J:
                    ci = items[items["class"]==c][["item_nbr"]].merge(profiles, on="item_nbr", how="inner").merge(meta, on="store_nbr", how="left")
                    ci = ci[ci["store_nbr"]!=s]
                    if not ci.empty:
                        Xc = ci[["h_mean","h_sd","p_any"]].to_numpy(dtype=float)
                        dc = np.sqrt(((Xc - tgt) ** 2 * w).sum(axis=1))
                        ci = ci.assign(distance=dc)
                        ci = ci.sort_values(["distance","state","city"], ascending=[True,True,True])
                        for _, r in ci.head(N_DONORS_PER_J - (rank - 1)).iterrows():
                            donors_rows.append({
                                "j_store": s, "j_item": j,
                                "donor_store": int(r["store_nbr"]), "donor_item": int(r["item_nbr"]),
                                "donor_kind": "same_class",
                                "distance": float(r["distance"]),
                                "store_type": str(r.get("type","")), "city": str(r.get("city","")), "state": str(r.get("state","")),
                                "rank": rank
                            })
                            rank += 1

        if processed_pairs % FLUSH_DONORS_EVERY == 0 and donors_rows:
            batch = pd.DataFrame(donors_rows[-FLUSH_DONORS_EVERY*N_DONORS_PER_J:])
            donors_header_written = _append_partial_csv(donors_partial_path, batch, donors_header_written)

        pct = 100.0 * processed_pairs / max(1, total_pairs) if total_pairs else 100.0
        _progress_emit("select_donors", min(99.0, pct),
                       f"Víctimas con donantes: {processed_pairs}/{total_pairs}")

    donors_df = pd.DataFrame(donors_rows)
    if donors_df.empty:
        _LOGGER.warning("No se hallaron donantes; se generará CSV vacío.")
        donors_df = pd.DataFrame(columns=["j_store","j_item","donor_store","donor_item","donor_kind","distance","store_type","city","state","rank"])
    else:
        donors_df = donors_df.sort_values(["j_store","j_item","rank"])
    donors_path = os.path.join(outdir, "donors_per_victim.csv")

    _progress_emit("write_donors", 0.0, "Escribiendo donors_per_victim.csv ...")
    donors_df.to_csv(donors_path, index=False)
    _progress_emit("write_donors", 100.0, f"donors_per_victim.csv listo ({len(donors_df):,} filas).")

    # limpieza temporal si se creó H+class
    if H_use.endswith("_with_class.tmp.csv"):
        try:
            os.remove(H_use)
            _LOGGER.info("Temporal de H con 'class' eliminado.")
        except Exception:
            _LOGGER.warning("No se pudo limpiar temporal H_with_class.")

    _LOGGER.info(f"=== Fin OK en {time.time()-t0_all:.2f}s ===")
    return pairs_path, donors_path


# ------------------- CLI -------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Genera 2 salidas: pairs_windows.csv y donors_per_victim.csv")
    ap.add_argument("--H", required=True, help="features_h_exposure.csv (con o sin 'class')")
    ap.add_argument("--train", required=True, help="train.csv")
    ap.add_argument("--items", required=True, help="items.csv")
    ap.add_argument("--stores", required=True, help="stores.csv")
    ap.add_argument("--outdir", default="data", help="Directorio de salida")
    args = ap.parse_args()
    try:
        p1, p2 = select_pairs_and_donors(args.H, args.train, args.items, args.stores, args.outdir)
        print("[OK] Generados:")
        print(" -", p1)
        print(" -", p2)
    except Exception as e:
        _LOGGER.error(f"Fallo en ejecución: {e}\n{traceback.format_exc()}")
        sys.exit(1)