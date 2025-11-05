# -*- coding: utf-8 -*-
"""
donors_selector.py
------------------
Módulo autónomo para seleccionar donantes por episodio (GSC), optimizado y paralelizable.

- Lee H (con o sin shards por tienda).
- Usa perfiles agregados por (store,item) (h_mean, h_sd, p_any) + clase.
- Para cada víctima (s,j,c):
    1) SAME_ITEM: busca el mismo SKU en otras tiendas, score por distancia de perfil
       + cercanía de nivel PRE (media PRE). Balancea por "above/below".
    2) SAME_CLASS: busca dentro de la clase c en otras tiendas. Vectoriza el cálculo
       de estadísticas PRE por tienda para reducir lecturas.

- Paraleliza a nivel episodio con initializer para compartir en memoria:
  H_use path, perfiles, items, stores y shard_dir.

Entrada principal: `build_donors_for_pairs(...)`.

"""
# models/donors_selector.py
from __future__ import annotations
import os
import math
import logging
import time
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# ENV helpers
# ---------------------------------------------------------------------
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

def _env_bool(name: str, default: bool) -> bool:
    return bool(_env_int(name, 1 if default else 0))

# ---------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------
CHUNK_H                = _env_int("SPD_CHUNK_H", 10_000_000)
PRE_GAP                = _env_int("SPD_PRE_GAP", 7)

DONORS_PARALLEL        = _env_bool("SPD_DONORS_PARALLEL", True)
DONORS_N_CORES         = max(1, min(_env_int("SPD_DONORS_N_CORES", 12), os.cpu_count() or 1))
DONORS_CHUNKSIZE       = _env_int("SPD_DONORS_CHUNKSIZE", 1)

N_DONORS_PER_J         = _env_int("SPD_N_DONORS_PER_J", 10)
SPD_DONOR_PRESELECT_TOP_K  = _env_int("SPD_DONOR_PRESELECT_TOP_K", 120)
SPD_DONOR_SAME_ITEM_RATIO  = _env_float("SPD_DONOR_SAME_ITEM_RATIO", 0.6)
SPD_DONOR_MIN_ABOVE        = _env_int("SPD_DONOR_MIN_ABOVE", 3)
SPD_DONOR_MIN_BELOW        = _env_int("SPD_DONOR_MIN_BELOW", 3)
SPD_DONOR_PRE_SD_MIN       = _env_float("SPD_DONOR_PRE_SD_MIN", 0.003)
SPD_DONOR_PRE_COVERAGE_MIN = _env_float("SPD_DONOR_PRE_COVERAGE_MIN", 0.55)
SPD_DONOR_PRE_MIN_LEVEL    = _env_float("SPD_DONOR_PRE_MIN_LEVEL", 0.0)
SPD_DONOR_LEVEL_BAND_PCT   = _env_float("SPD_DONOR_LEVEL_BAND_PCT", 0.5)
SPD_DONOR_COMBO_ALPHA      = _env_float("SPD_DONOR_COMBO_ALPHA", 0.5)

# Evitar donantes en la tienda caníbal (misma CLASE) si está disponible esa info
SPD_AVOID_CANNIBAL_STORE   = _env_bool("SPD_AVOID_CANNIBAL_STORE", True)

SPD_MAX_HSC_CACHE      = _env_int("SPD_MAX_HSC_CACHE", 32)

# ---------------------------------------------------------------------
# Logging / progreso
# ---------------------------------------------------------------------
_LOGGER = logging.getLogger("donors_selector"); _LOGGER.addHandler(logging.NullHandler())
def _log_setup_from_parent(logger: Optional[logging.Logger]) -> logging.Logger:
    if logger is not None:
        return logger
    if not _LOGGER.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("[%(asctime)s][%(processName)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        _LOGGER.addHandler(sh)
        _LOGGER.setLevel(logging.INFO)
    return _LOGGER

def _emit_progress(cb: Optional[Callable[[float, str], None]], pct: float, msg: str, logger: logging.Logger):
    pct = max(0.0, min(100.0, float(pct)))
    if cb:
        try:
            cb(pct, msg); return
        except Exception:
            pass
    logger.info(f"[DONORS] {pct:5.1f}%  {msg}")

# ---------------------------------------------------------------------
# Shards y caches por proceso
# ---------------------------------------------------------------------
_GLOBALS: Dict[str, object] = {}
_PROC_HSC_CACHE: "Dict[Tuple[str,int,int], Dict]" = {}
_PROC_HSC_ORDER: List[Tuple[str,int,int]] = []

def _init_pool(h_use: str, profiles_df: pd.DataFrame, items_df: pd.DataFrame, stores_df: pd.DataFrame,
               shard_dir: Optional[str]) -> None:
    _GLOBALS["H_USE"] = h_use
    _GLOBALS["PROFILES"] = profiles_df
    _GLOBALS["ITEMS"] = items_df
    _GLOBALS["STORES"] = stores_df
    _GLOBALS["SHARD_DIR"] = shard_dir

def _shard_path(store_nbr: int) -> Optional[str]:
    sd = _GLOBALS.get("SHARD_DIR")
    if not sd:
        return None
    return os.path.join(sd, f"H_store_{int(store_nbr)}.csv.gz")

def _lru_touch(key: Tuple[str,int,int]):
    if key in _PROC_HSC_CACHE:
        try:
            _PROC_HSC_ORDER.remove(key)
        except ValueError:
            pass
        _PROC_HSC_ORDER.append(key)
        while len(_PROC_HSC_ORDER) > max(1, SPD_MAX_HSC_CACHE):
            old = _PROC_HSC_ORDER.pop(0)
            _PROC_HSC_CACHE.pop(old, None)

# ---------------------------------------------------------------------
# Lecturas/grupo
# ---------------------------------------------------------------------
def _read_H_store_df(h_use: str, store_nbr: int) -> pd.DataFrame:
    dtypesH = {"store_nbr":"int16","item_nbr":"int32","class":"int16","H_bin":"int8"}
    usecols = ["date","store_nbr","item_nbr","class","H_prop","H_bin"]
    parse_dates = ["date"]
    path = _shard_path(store_nbr)
    if path and os.path.exists(path):
        reader = pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, dtype=dtypesH,
                             chunksize=CHUNK_H, low_memory=False)
    else:
        reader = pd.read_csv(h_use, usecols=usecols, parse_dates=parse_dates, dtype=dtypesH,
                             chunksize=CHUNK_H, low_memory=False)
    parts=[]
    for ch in reader:
        ch = ch[ch["store_nbr"]==store_nbr]
        if not ch.empty:
            parts.append(ch[["date","item_nbr","class","H_prop","H_bin"]])
    if not parts:
        return pd.DataFrame(columns=["date","item_nbr","class","H_prop","H_bin"])
    H_s = pd.concat(parts, ignore_index=True)
    H_s = (H_s.groupby(["date","item_nbr","class"], as_index=False)
               .agg(H_prop=("H_prop","mean"), H_bin=("H_bin","max")))
    return H_s

def _build_group_cache_from_store_df(H_s: pd.DataFrame, cls: int) -> Dict:
    df = H_s[H_s["class"]==cls]
    if df.empty:
        return {"dates_np": np.array([], dtype="datetime64[D]"),
                "items": np.array([], dtype="int32"),
                "Hmat": np.empty((0,0), dtype="float32")}
    dates = pd.Index(np.sort(df["date"].unique()))
    items = np.sort(df["item_nbr"].unique().astype("int32"))
    r = df["date"].map({d:i for i,d in enumerate(dates)}).to_numpy(np.int64, copy=False)
    c = df["item_nbr"].map({it:i for i,it in enumerate(items)}).to_numpy(np.int64, copy=False)
    v = df["H_prop"].astype("float32").to_numpy(copy=False)

    Hmat = np.full((len(dates), len(items)), np.nan, dtype="float32")
    Hmat[r, c] = v
    return {"dates_np": dates.to_numpy(dtype="datetime64[D]"),
            "items": items,
            "Hmat": Hmat}

def _get_H_group_cache(h_use: str, store_nbr: int, cls: int) -> Dict:
    key = (h_use, int(store_nbr), int(cls))
    if key in _PROC_HSC_CACHE:
        _lru_touch(key)
        return _PROC_HSC_CACHE[key]
    H_s = _read_H_store_df(h_use, store_nbr)
    cache = _build_group_cache_from_store_df(H_s, cls)
    _PROC_HSC_CACHE[key] = cache
    _lru_touch(key)
    return cache

# ---------------------------------------------------------------------
# Stats PRE y score
# ---------------------------------------------------------------------
def _pre_stats_batch(g: Dict, items_list: List[int],
                     pre_start: pd.Timestamp, treat_start: pd.Timestamp) -> pd.DataFrame:
    if g["Hmat"].size == 0 or not items_list:
        return pd.DataFrame(columns=["item_nbr","pre_mean","pre_sd","pre_min","coverage"])
    dates_np = g["dates_np"]; H = g["Hmat"]; all_items = g["items"]
    pre_end = (pd.Timestamp(treat_start) - pd.Timedelta(days=PRE_GAP+1)).normalize()
    s = np.datetime64(pd.Timestamp(pre_start).normalize())
    e = np.datetime64(pre_end)
    mask = np.zeros(len(dates_np), dtype=bool)
    i = np.searchsorted(dates_np, s, side="left")
    j = np.searchsorted(dates_np, e, side="right")
    if i < j:
        mask[i:j] = True
    if not mask.any():
        return pd.DataFrame(columns=["item_nbr","pre_mean","pre_sd","pre_min","coverage"])

    pos_map = {int(itm): int(pos) for pos, itm in enumerate(all_items)}
    cols = [pos_map.get(int(x), -1) for x in items_list]
    valid = np.array([c >= 0 for c in cols], dtype=bool)
    if not valid.any():
        return pd.DataFrame(columns=["item_nbr","pre_mean","pre_sd","pre_min","coverage"])
    items_arr = np.array(items_list, dtype=np.int32)[valid]
    col_idx = np.array([c for c in cols if c >= 0], dtype=int)

    X = H[:, col_idx]
    val = np.isfinite(X)
    pre_val = val & mask[:, None]
    n_in = int(mask.sum())
    n_ok = pre_val.sum(axis=0)
    safe = np.where(pre_val, X, np.nan)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(safe, axis=0)
        sds   = np.nanstd(safe,  axis=0)
    mins = np.nanmin(np.where(pre_val, X, np.inf), axis=0)
    mins[n_ok == 0] = np.nan
    cov  = np.where(n_in > 0, n_ok / float(n_in), 0.0)

    return pd.DataFrame({
        "item_nbr": items_arr.astype("int32"),
        "pre_mean": means.astype("float32"),
        "pre_sd":   sds.astype("float32"),
        "pre_min":  mins.astype("float32"),
        "coverage": cov.astype("float32")
    })

def _score_donor_candidate(tgt_mean: float, cand_stats: Dict, profile_dist: float) -> Tuple[float, str, float]:
    m = float(cand_stats["pre_mean"])
    side = "above" if m >= tgt_mean else "below"
    denom = max(1e-6, abs(tgt_mean))
    d_pre_norm = abs(m - tgt_mean) / denom
    score = SPD_DONOR_COMBO_ALPHA * d_pre_norm + (1.0 - SPD_DONOR_COMBO_ALPHA) * float(profile_dist)
    if abs(m - tgt_mean) <= SPD_DONOR_LEVEL_BAND_PCT * abs(tgt_mean):
        score *= 0.85
    return float(score), side, float(d_pre_norm)

# ---------------------------------------------------------------------
# Worker por episodio
# ---------------------------------------------------------------------
def _donors_for_episode_worker(ep: Dict) -> List[Dict]:
    H_use: str = _GLOBALS["H_USE"]  # type: ignore
    profiles: pd.DataFrame = _GLOBALS["PROFILES"]  # type: ignore
    items: pd.DataFrame = _GLOBALS["ITEMS"]  # type: ignore
    stores: pd.DataFrame = _GLOBALS["STORES"]  # type: ignore

    s = int(ep["j_store"]); j = int(ep["j_item"]); c = int(ep["class"])
    pre_start = pd.Timestamp(ep["pre_start"])
    treat_start = pd.Timestamp(ep["treat_start"])
    # opcionales (para evitar contaminación espacial)
    i_store = int(ep["i_store"]) if "i_store" in ep and pd.notna(ep["i_store"]) else None

    rows: List[Dict] = []

    # Perfil target
    pj = profiles[(profiles["store_nbr"]==s) & (profiles["item_nbr"]==j)]
    if pj.empty:
        return rows
    pj = pj.iloc[0]
    tgt = np.array([pj["h_mean"], pj["h_sd"], pj["p_any"]], dtype=float)
    w = np.array([0.5, 0.2, 0.3], dtype=float)

    # Media PRE víctima (para nivel)
    g_j = _get_H_group_cache(H_use, s, c)
    st_victim = _pre_stats_batch(g_j, [j], pre_start, treat_start)
    j_pre_mean = None if st_victim.empty else float(st_victim["pre_mean"].iloc[0])

    # SAME_ITEM (mismo SKU en otras tiendas)
    cand_si = profiles[profiles["item_nbr"]==j].merge(stores, on="store_nbr", how="left")
    cand_si = cand_si[cand_si["store_nbr"] != s]
    if SPD_AVOID_CANNIBAL_STORE and i_store is not None:
        cand_si = cand_si[cand_si["store_nbr"] != int(i_store)]
    same_item_selected = pd.DataFrame()

    if not cand_si.empty:
        X = cand_si[["h_mean","h_sd","p_any"]].to_numpy(dtype=float, copy=False)
        d = np.sqrt(((X - tgt) ** 2 * w).sum(axis=1))
        cand_si = cand_si.assign(profile_distance=d)
        pre_si = cand_si.sort_values(["profile_distance","state","city"], ascending=[True,True,True])\
                        .head(max(N_DONORS_PER_J * 5, SPD_DONOR_PRESELECT_TOP_K))
        scored = []
        if j_pre_mean is None:
            for _, r in pre_si.iterrows():
                scored.append({
                    "store_nbr": int(r["store_nbr"]), "item_nbr": j,
                    "score": float(r["profile_distance"]), "side": "na", "d_pre_norm": np.inf,
                    "store_type": str(r.get("type","")), "city": str(r.get("city","")), "state": str(r.get("state",""))
                })
        else:
            for s_d, grp in pre_si.groupby("store_nbr", sort=False):
                g_d = _get_H_group_cache(H_use, int(s_d), c)
                st_df = _pre_stats_batch(g_d, [j], pre_start, treat_start)
                if st_df.empty:
                    continue
                st = st_df.iloc[0].to_dict()
                if (st["coverage"] < SPD_DONOR_PRE_COVERAGE_MIN) or \
                   (st["pre_sd"]   < SPD_DONOR_PRE_SD_MIN) or \
                   (st["pre_min"]  < SPD_DONOR_PRE_MIN_LEVEL):
                    continue
                pdist = float(grp["profile_distance"].iloc[0])
                score, side, dnorm = _score_donor_candidate(float(j_pre_mean), st, pdist)
                scored.append({
                    "store_nbr": int(s_d), "item_nbr": j,
                    "score": float(score), "side": side, "d_pre_norm": float(dnorm),
                    "store_type": str(grp["type"].iloc[0]) if "type" in grp.columns else "",
                    "city": str(grp["city"].iloc[0]) if "city" in grp.columns else "",
                    "state": str(grp["state"].iloc[0]) if "state" in grp.columns else "",
                })
        if scored:
            df_si = pd.DataFrame(scored)
            n_target_si = int(math.ceil(N_DONORS_PER_J * SPD_DONOR_SAME_ITEM_RATIO))
            above = df_si[df_si["side"]=="above"].sort_values("score").head(max(SPD_DONOR_MIN_ABOVE, n_target_si//2))
            below = df_si[df_si["side"]=="below"].sort_values("score").head(max(SPD_DONOR_MIN_BELOW, n_target_si//2))
            same_item_selected = pd.concat([above, below], ignore_index=True).drop_duplicates(subset=["store_nbr"]).sort_values("score")
            if len(same_item_selected) < n_target_si:
                rest = df_si[~df_si["store_nbr"].isin(same_item_selected["store_nbr"])].sort_values("score")
                need = n_target_si - len(same_item_selected)
                if need > 0:
                    same_item_selected = pd.concat([same_item_selected, rest.head(need)], ignore_index=True)
            same_item_selected = same_item_selected.head(n_target_si)

    rank = 1
    if not same_item_selected.empty:
        for _, r in same_item_selected.iterrows():
            rows.append({
                "j_store": s, "j_item": j,
                "donor_store": int(r["store_nbr"]), "donor_item": j,
                "donor_kind": "same_item",
                "distance": float(r["score"]),
                "store_type": str(r.get("store_type","")), "city": str(r.get("city","")), "state": str(r.get("state","")),
                "rank": rank
            })
            rank += 1

    # SAME_CLASS (misma clase, otras tiendas; además evitamos la TIENDA CANÍBAL si se conoce)
    remaining = max(0, N_DONORS_PER_J - (rank - 1))
    if remaining > 0:
        prof_sc = profiles[(profiles["class"]==c) & (profiles["store_nbr"]!=s) & (profiles["item_nbr"]!=j)] \
                    .merge(_GLOBALS["STORES"], on="store_nbr", how="left")  # type: ignore
        if SPD_AVOID_CANNIBAL_STORE and i_store is not None:
            prof_sc = prof_sc[prof_sc["store_nbr"] != int(i_store)]
        if not prof_sc.empty:
            Xc = prof_sc[["h_mean","h_sd","p_any"]].to_numpy(dtype=float, copy=False)
            dc = np.sqrt(((Xc - tgt) ** 2 * w).sum(axis=1))
            prof_sc = prof_sc.assign(profile_distance=dc)
            preselect_sc = prof_sc.sort_values(["profile_distance","state","city"], ascending=[True,True,True])\
                                  .head(max(remaining*6, SPD_DONOR_PRESELECT_TOP_K))
            chosen_parts: List[pd.DataFrame] = []
            for s_d, grp in preselect_sc.groupby("store_nbr", sort=False):
                g_d = _get_H_group_cache(H_use, int(s_d), c)
                st_df = _pre_stats_batch(g_d, grp["item_nbr"].astype(int).tolist(), pre_start, treat_start)
                if st_df.empty:
                    continue
                merged = grp.merge(st_df, on="item_nbr", how="left")
                merged = merged[(merged["coverage"] >= SPD_DONOR_PRE_COVERAGE_MIN) &
                                (merged["pre_sd"]   >= SPD_DONOR_PRE_SD_MIN) &
                                (merged["pre_min"]  >= SPD_DONOR_PRE_MIN_LEVEL)]
                if merged.empty:
                    continue
                if j_pre_mean is not None:
                    denom = max(1e-6, abs(float(j_pre_mean)))
                    d_pre_norm = np.abs(merged["pre_mean"].to_numpy(dtype=float) - float(j_pre_mean)) / denom
                    score = SPD_DONOR_COMBO_ALPHA * d_pre_norm + (1.0 - SPD_DONOR_COMBO_ALPHA) * merged["profile_distance"].to_numpy(dtype=float)
                    close_band = np.abs(merged["pre_mean"].to_numpy(dtype=float) - float(j_pre_mean)) <= (SPD_DONOR_LEVEL_BAND_PCT * abs(float(j_pre_mean)))
                    score = np.where(close_band, score * 0.85, score)
                    side = np.where(merged["pre_mean"].to_numpy(dtype=float) >= float(j_pre_mean), "above", "below")
                    merged["score"] = score
                    merged["side"] = side
                    merged["d_pre_norm"] = d_pre_norm
                else:
                    merged["score"] = merged["profile_distance"].astype(float)
                    merged["side"] = "na"
                    merged["d_pre_norm"] = np.inf
                chosen_parts.append(merged)

            if chosen_parts:
                df_sc = pd.concat(chosen_parts, ignore_index=True)
                above_sc = df_sc[df_sc["side"]=="above"].sort_values("score")
                below_sc = df_sc[df_sc["side"]=="below"].sort_values("score")
                take_above = remaining//2
                take_below = remaining - take_above
                chosen_sc = pd.concat([above_sc.head(max(SPD_DONOR_MIN_ABOVE, take_above)),
                                       below_sc.head(max(SPD_DONOR_MIN_BELOW, take_below))], ignore_index=True)
                if len(chosen_sc) < remaining:
                    rest = df_sc[~df_sc.index.isin(chosen_sc.index)].sort_values("score")
                    chosen_sc = pd.concat([chosen_sc, rest.head(remaining-len(chosen_sc))], ignore_index=True)
                for _, r in chosen_sc.iterrows():
                    rows.append({
                        "j_store": s, "j_item": j,
                        "donor_store": int(r["store_nbr"]), "donor_item": int(r["item_nbr"]),
                        "donor_kind": "same_class",
                        "distance": float(r["score"]),
                        "store_type": str(r.get("type","")), "city": str(r.get("city","")), "state": str(r.get("state","")),
                        "rank": rank
                    })
                    rank += 1
    return rows

# ---------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------
def build_donors_for_pairs(*,
                           H_use: str,
                           pairs_df_gsc: pd.DataFrame,
                           items_df: pd.DataFrame,
                           stores_df: pd.DataFrame,
                           item_stats_df: pd.DataFrame,
                           outdir: Optional[str] = None,
                           shard_dir: Optional[str] = None,
                           logger: Optional[logging.Logger] = None,
                           progress_cb: Optional[Callable[[float, str], None]] = None,
                           exp_tag: Optional[str] = None,
                           ) -> Tuple[pd.DataFrame, str]:
    """
    Devuelve (donors_df, donors_csv_path) y escribe donors_per_victim.csv
    para los episodios contenidos en pairs_df_gsc.
    """
    if exp_tag:
        outdir = os.path.join("figures", str(exp_tag), "tables")
    outdir = outdir or os.path.join("figures", "default", "tables")
    os.makedirs(outdir, exist_ok=True)
    log = _log_setup_from_parent(logger)
    t0 = time.time()

    # Perfiles agregados (h_mean, h_sd, p_any) + clase
    profiles = item_stats_df[["store_nbr","item_nbr","h_mean","h_sd","p_any"]].merge(
        items_df[["item_nbr","class"]], on="item_nbr", how="left"
    )

    # Episodios como dicts (conservar i_store/i_item si están)
    eps = []
    for _, row in pairs_df_gsc.iterrows():
        d = row.to_dict()
        for k in ["pre_start","treat_start","post_start","post_end"]:
            if k in d:
                d[k] = pd.Timestamp(d[k]).date()
        eps.append(d)

    donors_rows_all: List[List[Dict]] = []

    if DONORS_PARALLEL and len(eps) > 0:
        import multiprocessing as mp
        n_proc = max(1, min(DONORS_N_CORES, len(eps)))
        with mp.Pool(processes=n_proc, initializer=_init_pool,
                     initargs=(H_use, profiles, items_df, stores_df, shard_dir)) as pool:
            for k, rows in enumerate(pool.imap_unordered(_donors_for_episode_worker, eps, chunksize=DONORS_CHUNKSIZE), start=1):
                donors_rows_all.append(rows or [])
                _emit_progress(progress_cb, 100.0 * k / max(1, len(eps)), f"Episodios GSC con donantes: {k}/{len(eps)}", log)
    else:
        _init_pool(H_use, profiles, items_df, stores_df, shard_dir)
        for k, ep in enumerate(eps, start=1):
            donors_rows_all.append(_donors_for_episode_worker(ep) or [])
            _emit_progress(progress_cb, 100.0 * k / max(1, len(eps)), f"Episodios GSC con donantes: {k}/{len(eps)}", log)

    donors_rows = [r for chunk in donors_rows_all for r in chunk]
    donors_df = pd.DataFrame(donors_rows)
    if donors_df.empty:
        log.warning("No se hallaron donantes; se generará CSV vacío.")
        donors_df = pd.DataFrame(columns=["j_store","j_item","donor_store","donor_item","donor_kind","distance","store_type","city","state","rank"])
    else:
        donors_df = donors_df.sort_values(["j_store","j_item","rank"])

    donors_path = os.path.join(outdir, "donors_per_victim.csv")
    donors_df.to_csv(donors_path, index=False)
    log.info(f"donors_per_victim.csv listo ({len(donors_df):,} filas) en {time.time()-t0:.2f}s")
    return donors_df, donors_path