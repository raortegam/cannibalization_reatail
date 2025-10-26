# -*- coding: utf-8 -*-
"""
synthetic_control.py
====================

Implementación de Control Sintético Generalizado (GSC) para el estudio de
canibalización en retail sobre paneles episodios (víctima + donantes).

Características clave
---------------------
- Estimación por *interactive fixed effects* (factores latentes) vía SoftImpute
  (Mazumder et al., 2010): descomposición de bajo rango con *shrinkage* nuclear.
- Opción de covariables X_it (proxies semanales, Fourier, rezagos, feriados,
  tendencia/disponibilidad STL, dummies tienda, etc.) con actualización alternada:
      Y_it ≈ L_it + X_it'β    con M_ij como máscara de observación
- Selección de hiperparámetros (rango, τ, α) con validación cruzada en el tiempo
  sobre la unidad tratada usando únicamente datos *pre*.
- Contrafactual para la víctima en *post*, efectos diarios y agregados (ATT).
- Validación causal: placebos en el espacio (treat cada donante) e in-time
  (ventana pre desplazada), *p-values* por ranking.
- Robustez: *leave-one-out* retirando controles; sensibilidad global (muestreo
  sobre hiperparámetros/especificaciones).
- Salidas listas para comparar con Meta-learners.

Entradas esperadas
------------------
- Carpeta con parquets por episodio (panel largo) generados por `pre_algorithm.py`:
    ./.data/processed_data/gsc/<episode_id>.parquet
  con columnas al menos:
    ['date','store_nbr','item_nbr','sales','treated_unit','treated_time','D','unit_id', ... features ...]

- (Opcional) Índice de episodios:
    ./.data/processed_data/episodes_index.parquet

- Donantes por víctima:
    ./.data/processed_data/donors_per_victim.csv

Salidas
-------
- cf_series/<episode_id>_cf.parquet         : serie diaria y_hat_cf, efecto, etc.
- placebos/<episode_id>_space.parquet       : ATTs placebo en espacio (por donante)
- placebos/<episode_id>_time.parquet        : ATT placebo in-time
- loo/<episode_id>_loo.parquet              : ATTs al excluir 1 control
- reports/<episode_id>.json                 : reporte por episodio (hiperparámetros, métricas)
- gsc_metrics.parquet                       : resumen por episodio (para comparar con Meta)

CLI (ejemplo)
-------------
python -m src.models.synthetic_control \
  --gsc_dir ./.data/processed_data/gsc \
  --episodes_index ./.data/processed_data/episodes_index.parquet \
  --donors_csv ./.data/processed_data/donors_per_victim.csv \
  --out_dir ./.data/processed_data/gsc_outputs \
  --max_episodes 50 --cv_folds 3 --cv_holdout 21 \
  --grid_ranks "1,2,3" --grid_tau "0.5,1.0,2.0" --grid_alpha "0.0,0.01,0.1" \
  --include_covariates

Limitaciones
------------
- SoftImpute implementado con SVD denso (NumPy). Para paneles muy grandes,
  considerar un solver esparso/convexo dedicado o *warm starts*.
- La inferencia (p-values) se basa en placebos (randomización en espacio)
  y no en errores estándar asintóticos.

Autor: generado por GPT-5 Pro.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------

def _to_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _sorted_unique(a: Iterable) -> List:
    return sorted(list(dict.fromkeys(a)))

def _as_datetime(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def _nanrmspe(y, yhat) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return np.nan
    e = y[m] - yhat[m]
    denom = np.maximum(1.0, np.nanmean(np.square(y[m])) ** 0.5)
    return float(np.sqrt(np.nanmean(e**2)) / denom)

def _rmspe(y, yhat) -> float:
    e = y - yhat
    denom = max(1.0, float(np.sqrt(np.mean(y**2))))
    return float(np.sqrt(np.mean(e**2)) / denom)

def _mae(y, yhat) -> float:
    return float(np.mean(np.abs(y - yhat)))

def _safe_pct(a: float, b: float) -> float:
    if abs(b) < 1e-8:
        return np.nan
    return float(a / b)

def _augment_ridge(X: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Construye el sistema aumentado para resolver ridge con lstsq."""
    n, p = X.shape
    if alpha <= 0:
        return X, np.eye(0)  # sin regularización
    A = np.sqrt(alpha) * np.eye(p)
    return np.vstack([X, A]), A  # A devuelto sólo por compatibilidad


# ---------------------------------------------------------------------
# Selección de covariables y armado de matrices wide
# ---------------------------------------------------------------------

EXCLUDE_BASE_COLS = {
    # identificadores / etiquetas
    "id", "date", "year_week", "store_nbr", "item_nbr", "unit_id",
    "family_name",
    # objetivo y tratamiento
    "sales", "treated_unit", "treated_time", "D",
    # calidad donante / auxiliares
    "episode_id", "is_victim", "promo_share", "avail_share", "keep", "reason",
    # potencialmente post-tratamiento o no exógena
    "onpromotion",
}

def select_feature_cols(df: pd.DataFrame) -> List[str]:
    """Elige columnas numéricas 'seguras' como X_it."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = []
    for c in num:
        if c in EXCLUDE_BASE_COLS:
            continue
        # Heurística: mantener Fourier/rezagos/proxies/feriados/dummies tienda/mes/tendencias
        if (
            c.startswith("fourier_") or c.startswith("lag_") or
            c in {"Fsw_log1p", "Ow", "HNat", "HReg", "HLoc",
                  "is_bridge", "is_additional", "is_work_day",
                  "trend_T", "available_A", "Q_store_trend",
                  "month"} or
            c.startswith("type_") or c.startswith("cluster_") or c.startswith("state_")
        ):
            feats.append(c)
    # Asegurar orden estable
    feats = sorted(list(dict.fromkeys(feats)))
    return feats


@dataclass
class EpisodeWide:
    Y: np.ndarray                # (U,T) ventas
    M: np.ndarray                # (U,T) máscara de observación (True si observado)
    X: Optional[np.ndarray]      # (U,T,P) covariables (o None)
    units: List[str]             # unit_id ordenadas (víctima va primero)
    dates: List[pd.Timestamp]    # fechas ordenadas
    treated_row: int             # índice de fila de la víctima (0)
    treated_post_mask: np.ndarray# (T,) bool, columnas post para víctima
    pre_mask: np.ndarray         # (T,) bool, columnas pre para víctima
    feats: List[str]             # nombres de features
    meta: Dict                   # metadatos varios


def long_to_wide_for_episode(df: pd.DataFrame, include_covariates: bool = True) -> EpisodeWide:
    """Convierte panel largo del episodio a matrices wide (unidades x fecha)."""
    d = df.copy()
    d["date"] = _as_datetime(d["date"])
    d = d.sort_values(["date", "store_nbr", "item_nbr"])

    # Asegurar victim primero
    d["__is_victim__"] = (d["treated_unit"] == 1).astype(int)
    units = _sorted_unique(d.sort_values(["__is_victim__", "unit_id"], ascending=[False, True])["unit_id"])
    dates = _sorted_unique(d["date"])

    U, T = len(units), len(dates)

    # Pivot ventas
    Y = np.full((U, T), np.nan, dtype=float)
    treated_time_vec = np.zeros(T, dtype=bool)

    # Features
    feats = select_feature_cols(d) if include_covariates else []
    P = len(feats)
    X = np.zeros((U, T, P), dtype=float) if include_covariates and P > 0 else None

    # Mapa índice
    unit2idx = {u: i for i, u in enumerate(units)}
    date2idx = {dt: j for j, dt in enumerate(dates)}

    for _, r in d.iterrows():
        i = unit2idx[r["unit_id"]]
        j = date2idx[r["date"]]
        Y[i, j] = float(r["sales"])
        if int(r["treated_unit"]) == 1 and int(r["treated_time"]) == 1:
            treated_time_vec[j] = True
        if include_covariates and P > 0:
            for k, c in enumerate(feats):
                val = r.get(c, np.nan)
                X[i, j, k] = 0.0 if pd.isna(val) else float(val)

    # Máscara: observado todo excepto víctima en post
    treated_row = 0 if d.loc[d["treated_unit"] == 1, "unit_id"].head(1).tolist() and units[0] == d.loc[d["treated_unit"] == 1, "unit_id"].iloc[0] else units.index(
        d.loc[d["treated_unit"] == 1, "unit_id"].iloc[0]
    )
    M = np.isfinite(Y)
    # Celdas no observadas: víctima en post
    M[treated_row, treated_time_vec] = False

    pre_mask = ~treated_time_vec

    meta = {
        "n_units": U,
        "n_time": T,
        "victim_unit": units[treated_row],
        "n_controls": U - 1,
        "n_pre": int(pre_mask.sum()),
        "n_post": int(treated_time_vec.sum()),
    }
    return EpisodeWide(Y=Y, M=M, X=X, units=units, dates=dates,
                       treated_row=treated_row,
                       treated_post_mask=treated_time_vec,
                       pre_mask=pre_mask,
                       feats=feats,
                       meta=meta)


# ---------------------------------------------------------------------
# Núcleo GSC: SoftImpute + Ridge alternado
# ---------------------------------------------------------------------

@dataclass
class GSCConfig:
    rank: int = 2
    tau: float = 1.0
    alpha: float = 0.01        # ridge para β
    max_inner: int = 100       # iteraciones SoftImpute
    max_outer: int = 10        # alternancias (L <-> β)
    tol: float = 1e-4
    include_covariates: bool = True
    random_state: int = 42


class SoftImpute:
    """SoftImpute con SVD denso y máscara M (True si observado)."""

    def __init__(self, tau: float, rank: int, max_iters: int = 100, tol: float = 1e-4):
        self.tau = float(tau)
        self.rank = int(rank)
        self.max_iters = int(max_iters)
        self.tol = float(tol)

    def fit(self, W: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Resuelve: min_L 0.5||M*(W - L)||_F^2 + tau * ||L||_*
        Iterando: L = S_tau( SVD( W_tilde ) ), con W_tilde = M*W + (1-M)*L_{t-1}
        """
        # Inicialización: cero para missing
        L = np.nan_to_num(W, nan=0.0).copy()
        prev_missing = L.copy()

        for it in range(self.max_iters):
            # SVD de la matriz "completa" actual
            try:
                U, s, Vt = np.linalg.svd(L, full_matrices=False)
            except np.linalg.LinAlgError:
                # fallback: pequeña perturbación
                L = L + 1e-6 * np.random.randn(*L.shape)
                U, s, Vt = np.linalg.svd(L, full_matrices=False)
            # Soft-threshold a los singulares
            s_shrunk = np.maximum(s - self.tau, 0.0)
            r = int(min(self.rank, np.sum(s_shrunk > 0)))
            if r <= 0:
                r = 1  # asegurar al menos rango 1
            L_new = (U[:, :r] * s_shrunk[:r]) @ Vt[:r, :]

            # Mantener observados como W, missing como L_new
            L = np.where(M, W, L_new)

            # Convergencia en missing
            miss_mask = ~M
            num = np.linalg.norm((L - prev_missing)[miss_mask])
            den = np.linalg.norm(prev_missing[miss_mask]) + 1e-8
            rel = num / den
            if rel < self.tol:
                break
            prev_missing = L.copy()
        # Reconstrucción de L* (sin "pegar" observados)
        # Rehacer SVD y devolver L_shrunk de la última iteración:
        U, s, Vt = np.linalg.svd(L, full_matrices=False)
        s_shrunk = np.maximum(s - self.tau, 0.0)
        r = int(min(self.rank, np.sum(s_shrunk > 0)))
        if r <= 0:
            r = 1
        L_star = (U[:, :r] * s_shrunk[:r]) @ Vt[:r, :]
        return L_star


class GSCModel:
    """GSC con alternancia entre L (factores) y β (covariables) sobre máscara M."""

    def __init__(self, cfg: GSCConfig):
        self.cfg = cfg
        self.beta_: Optional[np.ndarray] = None
        self.L_: Optional[np.ndarray] = None
        self.history_: Dict = {}

    def fit(self, Y: np.ndarray, M: np.ndarray, X: Optional[np.ndarray] = None) -> "GSCModel":
        rng = np.random.default_rng(self.cfg.random_state)
        include_X = self.cfg.include_covariates and X is not None and X.ndim == 3 and X.shape[2] > 0
        P = 0 if not include_X else X.shape[2]

        # Inicialización
        beta = np.zeros(P, dtype=float) if include_X else None
        residual = Y.copy()

        if include_X:
            # Estándar: centramos X por columna (p) para estabilidad
            X_flat = X.reshape(-1, P)
            X_means = X_flat.mean(axis=0)
            X_std = X_flat.std(axis=0) + 1e-8
            self._x_means, self._x_std = X_means, X_std
        else:
            self._x_means, self._x_std = None, None

        def xbeta_all(beta_vec: Optional[np.ndarray]) -> np.ndarray:
            if not include_X or beta_vec is None:
                return np.zeros_like(Y)
            # normalizar X y multiplicar
            Xn = (X - self._x_means.reshape(1, 1, -1)) / self._x_std.reshape(1, 1, -1)
            return np.tensordot(Xn, beta_vec, axes=(2, 0))  # (U,T,P)·(P,) -> (U,T)

        # Alternancia
        hist = {"outer_obj": [], "rmspe_obs": []}
        L = np.zeros_like(Y)
        for outer in range(self.cfg.max_outer):
            # Paso 1: SoftImpute sobre residuales observados
            R = Y - (xbeta_all(beta) if include_X else 0.0)
            W = np.where(M, R, np.nan)
            L_est = SoftImpute(self.cfg.tau, self.cfg.rank, self.cfg.max_inner, self.cfg.tol).fit(W, M)
            L = L_est

            # Paso 2: Ridge para β con observados
            if include_X:
                R_obs = (Y - L)[M]
                Xn = (X - self._x_means.reshape(1, 1, -1)) / self._x_std.reshape(1, 1, -1)
                X_obs = Xn[M, :]  # filas (i,t) observadas
                # Resolver ridge con sistema aumentado
                if X_obs.size == 0:
                    beta = np.zeros(P, dtype=float)
                else:
                    A, _ = _augment_ridge(X_obs, self.cfg.alpha)
                    y_aug = np.concatenate([R_obs, np.zeros(X_obs.shape[1])]) if self.cfg.alpha > 0 else R_obs
                    beta, *_ = np.linalg.lstsq(A, y_aug, rcond=None)

            # Métrica de seguimiento (RMSPE sobre observados)
            yhat_obs = (L + (xbeta_all(beta) if include_X else 0.0))[M]
            hist["rmspe_obs"].append(_rmspe(Y[M], yhat_obs))
            # Criterio simple
            if outer > 0 and abs(hist["rmspe_obs"][-1] - hist["rmspe_obs"][-2]) < 1e-5:
                break

        self.beta_ = beta
        self.L_ = L
        self.history_ = hist
        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        assert self.L_ is not None, "Modelo no ajustado."
        if self.cfg.include_covariates and self.beta_ is not None and X is not None and X.ndim == 3:
            Xn = (X - self._x_means.reshape(1, 1, -1)) / self._x_std.reshape(1, 1, -1)
            Xb = np.tensordot(Xn, self.beta_, axes=(2, 0))
            return self.L_ + Xb
        return self.L_


# ---------------------------------------------------------------------
# Validación en el tiempo (selección de hiperparámetros)
# ---------------------------------------------------------------------

@dataclass
class CVCfg:
    folds: int = 3
    holdout_days: int = 21
    grid_ranks: Tuple[int, ...] = (1, 2, 3)
    grid_tau: Tuple[float, ...] = (0.5, 1.0, 2.0)
    grid_alpha: Tuple[float, ...] = (0.0, 0.01, 0.1)

def time_cv_select(ep: EpisodeWide, cv: CVCfg, base_cfg: GSCConfig) -> Tuple[GSCConfig, Dict]:
    """Realiza validación cruzada sobre el periodo pre de la víctima."""
    T = len(ep.dates)
    pre_idx = np.where(ep.pre_mask)[0]
    if len(pre_idx) < max(14, cv.holdout_days + 7):
        # pre muy corto: usar configuración base
        return base_cfg, {"cv_used": False, "score": np.nan, "best": asdict(base_cfg)}

    # Construir folds por ventanas deslizantes al final del pre
    H = int(cv.holdout_days)
    fold_ends = list(range(pre_idx[-1] - H + 1, pre_idx[-1] - H * cv.folds + 1, -H))
    fold_ends = [e for e in fold_ends if e >= pre_idx[0] + H - 1][:cv.folds]
    if not fold_ends:
        return base_cfg, {"cv_used": False, "score": np.nan, "best": asdict(base_cfg)}

    results = []
    for rnk in cv.grid_ranks:
        for tau in cv.grid_tau:
            for alpha in cv.grid_alpha:
                cfg_try = GSCConfig(rank=rnk, tau=float(tau), alpha=float(alpha),
                                    max_inner=base_cfg.max_inner, max_outer=base_cfg.max_outer,
                                    tol=base_cfg.tol, include_covariates=base_cfg.include_covariates,
                                    random_state=base_cfg.random_state)
                fold_scores = []
                for e in fold_ends:
                    # Holdout última ventana H del pre para la víctima
                    M_train = ep.M.copy()
                    hold_j = np.arange(e - H + 1, e + 1, dtype=int)
                    M_train[ep.treated_row, hold_j] = False  # ocultar como missing (validación)
                    model = GSCModel(cfg_try).fit(ep.Y, M_train, ep.X)
                    Yhat = model.predict(ep.X)
                    y_true = ep.Y[ep.treated_row, hold_j]
                    y_hat = Yhat[ep.treated_row, hold_j]
                    fold_scores.append(_rmspe(y_true, y_hat))
                results.append({
                    "rank": rnk, "tau": float(tau), "alpha": float(alpha),
                    "score": float(np.nanmedian(fold_scores)),
                    "scores": fold_scores
                })
    # Selección por score mínimo (mediana)
    best = min(results, key=lambda z: z["score"])
    best_cfg = GSCConfig(rank=int(best["rank"]), tau=float(best["tau"]), alpha=float(best["alpha"]),
                         max_inner=base_cfg.max_inner, max_outer=base_cfg.max_outer,
                         tol=base_cfg.tol, include_covariates=base_cfg.include_covariates,
                         random_state=base_cfg.random_state)
    return best_cfg, {"cv_used": True, "score": best["score"], "best": best, "all": results}


# ---------------------------------------------------------------------
# Placebos, LOO y sensibilidad
# ---------------------------------------------------------------------

def placebo_space(ep: EpisodeWide, cfg: GSCConfig, max_units: int = 20) -> pd.DataFrame:
    """Trata cada control como 'víctima' (misma ventana temporal) y estima ATT placebo."""
    rows = []
    control_rows = [i for i in range(len(ep.units)) if i != ep.treated_row]
    control_rows = control_rows[:max_units]  # limitar costo
    for r in control_rows:
        M_alt = ep.M.copy()
        # "finge" tratamiento en mismas columnas post pero para unidad r
        M_alt[r, ep.treated_post_mask] = False
        model = GSCModel(cfg).fit(ep.Y, M_alt, ep.X)
        Yhat = model.predict(ep.X)
        eff = ep.Y[r, ep.treated_post_mask] - Yhat[r, ep.treated_post_mask]
        rows.append({
            "unit_id": ep.units[r],
            "att_placebo_mean": float(np.nanmean(eff)),
            "att_placebo_sum": float(np.nansum(eff)),
            "rmspe_pre": float(_rmspe(ep.Y[r, ep.pre_mask], Yhat[r, ep.pre_mask]))
        })
    out = pd.DataFrame(rows)
    return out


def placebo_time(ep: EpisodeWide, cfg: GSCConfig) -> pd.DataFrame:
    """In-time placebo: desplaza ventana (mismo largo) dentro del pre de la víctima."""
    pre_idx = np.where(ep.pre_mask)[0]
    Lpre = len(pre_idx)
    Lpost = int(ep.treated_post_mask.sum())
    H = min(Lpost, max(7, Lpre // 2))
    if Lpre <= H + 7:
        return pd.DataFrame([{"att_placebo_mean": np.nan, "att_placebo_sum": np.nan, "used": False}])

    # Tomar la última ventana pre de longitud H como pseudo-post
    hold = pre_idx[-H:]
    M_alt = ep.M.copy()
    M_alt[ep.treated_row, hold] = False
    model = GSCModel(cfg).fit(ep.Y, M_alt, ep.X)
    Yhat = model.predict(ep.X)
    eff = ep.Y[ep.treated_row, hold] - Yhat[ep.treated_row, hold]
    return pd.DataFrame([{
        "att_placebo_mean": float(np.nanmean(eff)),
        "att_placebo_sum": float(np.nansum(eff)),
        "used": True,
        "H": int(H)
    }])


def leave_one_out(ep: EpisodeWide, cfg: GSCConfig, max_units: int = 20) -> pd.DataFrame:
    """Robustez: excluir un control a la vez."""
    rows = []
    control_rows = [i for i in range(len(ep.units)) if i != ep.treated_row][:max_units]
    for r in control_rows:
        # Eliminar fila r
        keep = [i for i in range(len(ep.units)) if i != r]
        Y = ep.Y[keep, :]
        M = ep.M[keep, :]
        X = ep.X[keep, :, :] if ep.X is not None else None
        treated_row = next(i for i, k in enumerate(keep) if k == ep.treated_row)
        model = GSCModel(cfg).fit(Y, M, X)
        Yhat = model.predict(X)

        eff = Y[treated_row, ep.treated_post_mask] - Yhat[treated_row, ep.treated_post_mask]
        rows.append({
            "excluded_unit": ep.units[r],
            "att_mean": float(np.nanmean(eff)),
            "att_sum": float(np.nansum(eff))
        })
    return pd.DataFrame(rows)


def global_sensitivity(ep: EpisodeWide,
                       base_cfg: GSCConfig,
                       n_samples: int = 24) -> pd.DataFrame:
    """Barrido global simple sobre hiperparámetros/especificaciones."""
    rng = np.random.default_rng(base_cfg.random_state)
    rows = []
    for _ in range(n_samples):
        rank = int(rng.integers(1, max(2, base_cfg.rank + 2)))
        tau = float(10 ** rng.uniform(-0.3, 0.7))  # ~ [0.5, 5]
        alpha = float(10 ** rng.uniform(-6, -1))   # ~ [1e-6, 1e-1]
        incX = bool(rng.integers(0, 2)) if base_cfg.include_covariates else False
        cfg = GSCConfig(rank=rank, tau=tau, alpha=alpha,
                        max_inner=base_cfg.max_inner, max_outer=base_cfg.max_outer,
                        tol=base_cfg.tol, include_covariates=incX,
                        random_state=base_cfg.random_state)
        model = GSCModel(cfg).fit(ep.Y, ep.M, ep.X if incX else None)
        Yhat = model.predict(ep.X if incX else None)
        eff = ep.Y[ep.treated_row, ep.treated_post_mask] - Yhat[ep.treated_row, ep.treated_post_mask]
        rows.append({
            "rank": rank, "tau": tau, "alpha": alpha, "include_covariates": incX,
            "att_mean": float(np.nanmean(eff)),
            "att_sum": float(np.nansum(eff))
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Pipeline por episodio y batch
# ---------------------------------------------------------------------

@dataclass
class RunConfig:
    gsc_dir: Path
    donors_csv: Path
    episodes_index: Optional[Path]
    out_dir: Path
    include_covariates: bool = True
    max_episodes: Optional[int] = None
    log_level: str = "INFO"
    # CV
    cv_folds: int = 3
    cv_holdout: int = 21
    grid_ranks: Tuple[int, ...] = (1, 2, 3)
    grid_tau: Tuple[float, ...] = (0.5, 1.0, 2.0)
    grid_alpha: Tuple[float, ...] = (0.0, 0.01, 0.1)
    # Placebos / Robustez / Sensibilidad
    max_placebos: int = 20
    max_loo: int = 20
    do_placebo_space: bool = True
    do_placebo_time: bool = True
    do_loo: bool = True
    do_sensitivity: bool = True
    sens_samples: int = 24
    # Solver
    rank: int = 2
    tau: float = 1.0
    alpha: float = 0.01
    max_inner: int = 100
    max_outer: int = 10
    tol: float = 1e-4
    random_state: int = 42


def _discover_episode_files(gsc_dir: Path) -> List[Path]:
    files = [p for p in gsc_dir.glob("*.parquet") if p.name != "donor_quality.parquet"]
    return sorted(files)

def _episode_id_from_path(p: Path) -> str:
    return p.stem  # coincide con convención: i-i_j-j_yyyymmdd

def _load_episode_df(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    # Normalizar tipos
    if "date" in df.columns:
        df["date"] = _as_datetime(df["date"])
    return df


def run_episode(p: Path, cfg: RunConfig) -> Dict:
    ep_id = _episode_id_from_path(p)
    df = _load_episode_df(p)

    # Armar wide
    epw = long_to_wide_for_episode(df, include_covariates=cfg.include_covariates)

    # Config base + CV
    base = GSCConfig(rank=cfg.rank, tau=cfg.tau, alpha=cfg.alpha,
                     max_inner=cfg.max_inner, max_outer=cfg.max_outer,
                     tol=cfg.tol, include_covariates=cfg.include_covariates,
                     random_state=cfg.random_state)
    cv_cfg = CVCfg(folds=cfg.cv_folds, holdout_days=cfg.cv_holdout,
                   grid_ranks=cfg.grid_ranks, grid_tau=cfg.grid_tau, grid_alpha=cfg.grid_alpha)
    best_cfg, cv_info = time_cv_select(epw, cv_cfg, base)

    # Ajuste final
    model = GSCModel(best_cfg).fit(epw.Y, epw.M, epw.X)
    Yhat = model.predict(epw.X)

    # Efectos y métricas clave
    y_obs_post = epw.Y[epw.treated_row, epw.treated_post_mask]
    y_hat_post = Yhat[epw.treated_row, epw.treated_post_mask]
    eff_post = y_obs_post - y_hat_post

    y_obs_pre = epw.Y[epw.treated_row, epw.pre_mask]
    y_hat_pre = Yhat[epw.treated_row, epw.pre_mask]

    rmspe_pre = _rmspe(y_obs_pre, y_hat_pre)
    mae_pre = _mae(y_obs_pre, y_hat_pre)
    att_mean = float(np.nanmean(eff_post))
    att_sum = float(np.nansum(eff_post))
    base_level = float(np.nanmean(y_obs_pre))
    rel_att = _safe_pct(att_mean, base_level)

    # Guardar serie contrafactual
    cf_dir = cfg.out_dir / "cf_series"
    _ensure_dir(cf_dir)
    cf = pd.DataFrame({
        "episode_id": ep_id,
        "date": [epw.dates[j] for j in range(len(epw.dates))],
        "y_obs": epw.Y[epw.treated_row, :],
        "y_hat": Yhat[epw.treated_row, :],
        "treated_time": epw.treated_post_mask.astype(int),
    })
    cf["effect"] = cf["y_obs"] - cf["y_hat"]
    cf["cum_effect"] = cf["effect"].cumsum()
    cf_path = cf_dir / f"{ep_id}_cf.parquet"
    cf.to_parquet(cf_path, index=False)

    # Placebos y robustez
    plac_dir = cfg.out_dir / "placebos"; _ensure_dir(plac_dir)
    loo_dir = cfg.out_dir / "loo"; _ensure_dir(loo_dir)

    if cfg.do_placebo_space:
        ps = placebo_space(epw, best_cfg, max_units=cfg.max_placebos)
        ps["episode_id"] = ep_id
        ps_path = plac_dir / f"{ep_id}_space.parquet"
        ps.to_parquet(ps_path, index=False)
        # p-value (rank) para ATT_sum
        if ps.shape[0] > 0 and np.isfinite(att_sum):
            pval_space = (1 + np.sum(np.abs(ps["att_placebo_sum"].to_numpy()) >= abs(att_sum))) / (1 + ps.shape[0])
        else:
            pval_space = np.nan
    else:
        pval_space, ps_path = np.nan, None

    if cfg.do_placebo_time:
        pt = placebo_time(epw, best_cfg)
        pt["episode_id"] = ep_id
        pt_path = plac_dir / f"{ep_id}_time.parquet"
        pt.to_parquet(pt_path, index=False)
    else:
        pt_path = None

    if cfg.do_loo:
        loo = leave_one_out(epw, best_cfg, max_units=cfg.max_loo)
        loo["episode_id"] = ep_id
        loo_path = loo_dir / f"{ep_id}_loo.parquet"
        loo.to_parquet(loo_path, index=False)
        loo_sd = float(np.nanstd(loo["att_sum"])) if not loo.empty else np.nan
        loo_range = float(np.nanmax(loo["att_sum"]) - np.nanmin(loo["att_sum"])) if not loo.empty else np.nan
    else:
        loo_sd, loo_range, loo_path = np.nan, np.nan, None

    # Sensibilidad global
    if cfg.do_sensitivity:
        sens = global_sensitivity(epw, best_cfg, n_samples=cfg.sens_samples)
        sens["episode_id"] = ep_id
        sens_dir = cfg.out_dir / "sensitivity"; _ensure_dir(sens_dir)
        sens_path = sens_dir / f"{ep_id}_sens.parquet"
        sens.to_parquet(sens_path, index=False)
        sens_sd = float(np.nanstd(sens["att_sum"])) if not sens.empty else np.nan
    else:
        sens_sd, sens_path = np.nan, None

    # Reporte JSON por episodio
    rep = {
        "episode_id": ep_id,
        "n_units": epw.meta["n_units"], "n_controls": epw.meta["n_controls"],
        "n_time": epw.meta["n_time"], "n_pre": epw.meta["n_pre"], "n_post": epw.meta["n_post"],
        "victim_unit": epw.meta["victim_unit"],
        "include_covariates": cfg.include_covariates,
        "features_used": epw.feats,
        "cv": cv_info,
        "fit": {
            "rmspe_pre": rmspe_pre, "mae_pre": mae_pre,
            "att_mean": att_mean, "att_sum": att_sum, "rel_att_vs_pre_mean": rel_att
        },
        "p_value_placebo_space": pval_space,
        "paths": {
            "cf": str(cf_path),
            "placebo_space": str(ps_path) if cfg.do_placebo_space else None,
            "placebo_time": str(pt_path) if cfg.do_placebo_time else None,
            "loo": str(loo_path) if cfg.do_loo else None,
            "sensitivity": str(sens_path) if cfg.do_sensitivity else None
        },
        "robustness": {
            "loo_sd_att_sum": loo_sd, "loo_range_att_sum": loo_range,
            "sens_sd_att_sum": sens_sd
        },
        "best_cfg": asdict(best_cfg)
    }
    rep_dir = cfg.out_dir / "reports"; _ensure_dir(rep_dir)
    with open(rep_dir / f"{ep_id}.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)

    # Resumen para gsc_metrics.parquet
    summary = {
        "episode_id": ep_id,
        "victim_unit": epw.meta["victim_unit"],
        "n_controls": epw.meta["n_controls"],
        "n_pre": epw.meta["n_pre"], "n_post": epw.meta["n_post"],
        "rmspe_pre": rmspe_pre, "mae_pre": mae_pre,
        "att_mean": att_mean, "att_sum": att_sum, "rel_att_vs_pre_mean": rel_att,
        "p_value_placebo_space": pval_space,
        "loo_sd_att_sum": rep["robustness"]["loo_sd_att_sum"],
        "loo_range_att_sum": rep["robustness"]["loo_range_att_sum"],
        "sens_sd_att_sum": rep["robustness"]["sens_sd_att_sum"],
        "rank": best_cfg.rank, "tau": best_cfg.tau, "alpha": best_cfg.alpha,
        "include_covariates": best_cfg.include_covariates,
        "cv_used": bool(rep["cv"]["cv_used"])
    }
    return summary


def run_batch(cfg: RunConfig) -> None:
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    # Descubrir episodios
    gsc_dir = cfg.gsc_dir
    if not gsc_dir.exists():
        # fallback a layout alternativo
        alt = Path("./data/processed/gsc")
        if alt.exists():
            gsc_dir = alt
        else:
            raise FileNotFoundError(f"No existe la carpeta GSC: {cfg.gsc_dir}")

    files = _discover_episode_files(gsc_dir)
    if cfg.max_episodes is not None:
        files = files[: int(cfg.max_episodes)]

    logging.info(f"Episodios detectados: {len(files)} en {gsc_dir}")

    _ensure_dir(cfg.out_dir)
    metrics = []
    for k, p in enumerate(files, start=1):
        try:
            logging.info(f"[{k}/{len(files)}] Ejecutando GSC en {p.name} ...")
            summ = run_episode(p, cfg)
            metrics.append(summ)
            logging.info(f"OK {p.stem} | ATT_sum={summ['att_sum']:.3f} | RMSPE_pre={summ['rmspe_pre']:.4f}")
        except Exception as e:
            logging.exception(f"Error en episodio {p.name}: {e}")

    # Guardar resumen global
    if metrics:
        mdf = pd.DataFrame(metrics)
        mpath = cfg.out_dir / "gsc_metrics.parquet"
        mdf.to_parquet(mpath, index=False)
        logging.info(f"Métricas globales guardadas: {mpath} ({mdf.shape[0]} episodios)")
    else:
        logging.warning("No se generaron métricas (¿sin episodios válidos?).")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="GSC (Generalized Synthetic Control) para episodios de canibalización.")
    p.add_argument("--gsc_dir", type=str, default="./.data/processed_data/gsc", help="Directorio con parquets de episodios GSC.")
    p.add_argument("--donors_csv", type=str, default="./.data/processed_data/donors_per_victim.csv", help="CSV donantes por víctima (opcional).")
    p.add_argument("--episodes_index", type=str, default="./.data/processed_data/episodes_index.parquet", help="Índice de episodios (opcional).")
    p.add_argument("--out_dir", type=str, default="./.data/processed_data/gsc_outputs", help="Directorio de salida.")
    p.add_argument("--include_covariates", action="store_true", help="Usar covariables X_it en el ajuste.")
    p.add_argument("--max_episodes", type=int, default=None, help="Limitar # episodios a procesar.")
    p.add_argument("--log_level", type=str, default="INFO")

    # CV
    p.add_argument("--cv_folds", type=int, default=3)
    p.add_argument("--cv_holdout", type=int, default=21)
    p.add_argument("--grid_ranks", type=str, default="1,2,3")
    p.add_argument("--grid_tau", type=str, default="0.5,1.0,2.0")
    p.add_argument("--grid_alpha", type=str, default="0.0,0.01,0.1")

    # Placebos / Robustez / Sensibilidad
    p.add_argument("--max_placebos", type=int, default=20)
    p.add_argument("--max_loo", type=int, default=20)
    p.add_argument("--no_placebo_space", action="store_true")
    p.add_argument("--no_placebo_time", action="store_true")
    p.add_argument("--no_loo", action="store_true")
    p.add_argument("--no_sensitivity", action="store_true")
    p.add_argument("--sens_samples", type=int, default=24)

    # Solver / hiperparámetros base
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--max_inner", type=int, default=100)
    p.add_argument("--max_outer", type=int, default=10)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--random_state", type=int, default=42)

    a = p.parse_args()
    grid_ranks = tuple(int(x.strip()) for x in a.grid_ranks.split(",") if x.strip())
    grid_tau = tuple(float(x.strip()) for x in a.grid_tau.split(",") if x.strip())
    grid_alpha = tuple(float(x.strip()) for x in a.grid_alpha.split(",") if x.strip())

    return RunConfig(
        gsc_dir=_to_path(a.gsc_dir),
        donors_csv=_to_path(a.donors_csv),
        episodes_index=_to_path(a.episodes_index) if a.episodes_index else None,
        out_dir=_to_path(a.out_dir),
        include_covariates=bool(a.include_covariates),
        max_episodes=a.max_episodes,
        log_level=a.log_level,
        cv_folds=a.cv_folds, cv_holdout=a.cv_holdout,
        grid_ranks=grid_ranks, grid_tau=grid_tau, grid_alpha=grid_alpha,
        max_placebos=a.max_placebos, max_loo=a.max_loo,
        do_placebo_space=(not a.no_placebo_space),
        do_placebo_time=(not a.no_placebo_time),
        do_loo=(not a.no_loo),
        do_sensitivity=(not a.no_sensitivity),
        sens_samples=a.sens_samples,
        rank=a.rank, tau=a.tau, alpha=a.alpha,
        max_inner=a.max_inner, max_outer=a.max_outer, tol=a.tol,
        random_state=a.random_state
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_batch(cfg)