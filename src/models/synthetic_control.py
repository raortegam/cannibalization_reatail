# -*- coding: utf-8 -*-
"""
synthetic_control.py
====================

Implementación de Control Sintético Generalizado (GSC) con mejoras de
validación temporal y robustez numérica para contrafactuales en paneles.

Cambios clave respecto a la versión previa
------------------------------------------
1) **CV temporal purgada + embargo (gap)**:
   - La validación oculta **todas las unidades** en la(s) columna(s) de *holdout*
     (no solo la víctima) y aplica un *gap* antes del holdout para evitar filtraciones
     de información temporal (concepto "purged/embargoed CV").
   - Resultado: evita sobreajuste en pre y contrafactuales “no orgánicos”.

2) **Espacios de búsqueda saneados y adaptativos**:
   - Los rangos de `rank`, `tau` y `alpha` se ajustan dinámicamente a los límites
     válidos del episodio: `rank ≤ min(U-1, T-1)`.
   - `tau` se acota con base en la escala de `Y` (singular mayor del pre).
   - `alpha` evita valores demasiado pequeños (≥ 1e-5) para estabilidad.

3) **HPO a prueba de NaNs / out-of-range**:
   - La función objetivo **nunca** devuelve NaN/Inf (penaliza con un número grande).
   - Se registran pistas de frontera cuando el óptimo cae cerca de límites.

4) **Estandarización estable**:
   - `Y` se estandariza con media/desvío sobre celdas observadas de entrenamiento.
   - `X` se normaliza usando sólo celdas observadas (respetando la máscara M).

5) **Entrenamiento final con gap previo al tratamiento**:
   - Se aplica un pequeño embargo (configurable) antes del inicio del tratamiento
     al ajustar el modelo final, reduciendo el calce “perfecto” justo al borde.

6) **Métricas NaN‑safe + columnas plot‑ready**:
   - `_rmspe` y `_mae` ignoran filas con NaN al evaluar.
   - Se exportan `y_obs_plot`, `obs_mask`, `effect_plot` y `cum_effect_obs` en la
     serie contrafactual por episodio (para graficar en continuo sin alterar la evidencia).

7) **(NUEVO) Predicciones no negativas y menos planos**:
   - `link` opcional para `Y` (`identity` o `log1p`) + `pred_clip_min` (por defecto 0.0).
   - Calibración post‑ajuste de la víctima en PRE (`calibrate_victim`: `none|level|level_and_trend`).
   - Suavizado opcional del contrafactual de la víctima por ventana móvil.

La API de alto nivel se mantiene (misma estructura de funciones/clases y CLI),
por lo que el *pipeline* existente no se rompe.

"""
# models/synthetic_control.py
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Importar módulo de métricas causales
try:
    from .causal_metrics import CausalMetricsCalculator
    CAUSAL_METRICS_AVAILABLE = True
except ImportError:
    CAUSAL_METRICS_AVAILABLE = False
    logging.warning("causal_metrics.py no disponible; métricas causales comparativas desactivadas.")

# ---------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------
from dataclasses import field  # si el archivo define dataclasses con default_factory
def _data_root() -> Path:
    """
    Prefiere ./.data; si no existe, intenta ./data; y por último crea ./.data.
    """
    for cand in (Path("./.data"), Path("./data")):
        try:
            if cand.exists():
                return cand
        except Exception:
            pass
    return Path("./.data")

def _fig_root() -> Path:
    return Path("./figures")

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

# ---- Métricas NaN-safe usadas en todo el pipeline -----------
def _rmspe(y, yhat) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return float("nan")
    e = y[m] - yhat[m]
    denom = max(1.0, float(np.sqrt(np.mean((y[m])**2))))
    return float(np.sqrt(np.mean(e**2)) / denom)

def _mae(y, yhat) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(y[m] - yhat[m])))

def _safe_pct(a: float, b: float) -> float:
    if abs(b) < 1e-8:
        return np.nan
    return float(a / b)

def _augment_ridge(X: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Construye el sistema aumentado para resolver ridge con lstsq."""
    n, p = X.shape
    if alpha <= 0:
        return X, np.eye(0)
    A = np.sqrt(alpha) * np.eye(p)
    return np.vstack([X, A]), A

def _check_boundary_float(val: float, lo: float, hi: float, name: str, tol: float = 0.05) -> None:
    if not np.isfinite(val) or hi <= lo:
        return
    rel = (val - lo) / (hi - lo)
    if rel <= tol:
        logging.info(f"[HPO-GSC] '{name}' cerca del límite inferior ({lo:.6g}); considera ampliar el rango hacia abajo.")
    if rel >= 1 - tol:
        logging.info(f"[HPO-GSC] '{name}' cerca del límite superior ({hi:.6g}); considera ampliar el rango hacia arriba.")

def _check_boundary_int(val: int, lo: int, hi: int, name: str) -> None:
    if val == lo:
        logging.info(f"[HPO-GSC] '{name}' en el mínimo ({lo}); considera ampliar hacia abajo.")
    if val == hi:
        logging.info(f"[HPO-GSC] '{name}' en el máximo ({hi}); considera ampliar hacia arriba.")

def _finite_or_big(x: float, big: float = 1e6) -> float:
    """Devuelve x si es finito; en caso contrario, 'big'. Evita NaN/Inf en Optuna."""
    if not np.isfinite(x):
        return float(big)
    return float(x)


# ---------------------------------------------------------------------
# Transformación (link) de Y para reforzar no‑negatividad
# ---------------------------------------------------------------------

class _Link:
    def __init__(self, name: str, eps: float = 0.0):
        name = (name or "identity").strip().lower()
        if name not in {"identity", "log1p"}:
            raise ValueError("link debe ser 'identity' o 'log1p'")
        self.name = name
        self.eps = float(max(0.0, eps))

    def forward(self, Y: np.ndarray) -> np.ndarray:
        if self.name == "identity":
            return Y
        # ventas >= 0; por seguridad, clip mínimo para log1p
        return np.log1p(np.maximum(Y, -1.0 + self.eps))

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        if self.name == "identity":
            return Z
        out = np.expm1(Z)
        out[out < 0] = 0.0
        return out


# ---------------------------------------------------------------------
# Selección de covariables y armado de matrices wide (anti‑fuga)
# ---------------------------------------------------------------------

BLACKLIST_COLS = {
    "sales", "y_raw", "y_log1p",
    "sc_hat", "trend_T", "Q_store_trend", "available_A",
    "class_index_excl", "promo_share_sc", "promo_share_sc_excl",
    "treated_unit", "treated_time", "D", "episode_id", "unit_id",
    "store_nbr", "item_nbr", "year_week", "id", "family_name",
    "is_victim", "promo_share", "avail_share", "keep", "reason",
}
SAFE_PREFIXES = ("fourier_", "lag_", "promo_share_sc_l", "promo_share_sc_excl_l",
                 "class_index_excl_l", "type_", "cluster_", "state_")
SAFE_SINGLE_COLS = {"Fsw_log1p", "Ow", "HNat", "HReg", "HLoc",
                    "is_bridge", "is_additional", "is_work_day",
                    "trend_T_l1", "Q_store_trend_l1", "month"}

def _is_safe_feature(col: str) -> bool:
    if col in BLACKLIST_COLS:
        return False
    if col in SAFE_SINGLE_COLS:
        return True
    for p in SAFE_PREFIXES:
        if col.startswith(p):
            return True
    return False

def select_feature_cols(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if _is_safe_feature(c)]
    feats = sorted(list(dict.fromkeys(feats)))
    present_black = sorted([c for c in BLACKLIST_COLS if c in df.columns])
    if present_black:
        logging.warning(f"[FEATS] Blacklist presente y EXCLUIDA de X: {present_black}")
    logging.info(f"[FEATS] #{len(feats)} features seleccionadas.")
    if len(feats) <= 5:
        logging.info(f"[FEATS] Features: {feats}")
    return feats


@dataclass
class EpisodeWide:
    Y: np.ndarray
    M: np.ndarray
    X: Optional[np.ndarray]
    units: List[str]
    dates: List[pd.Timestamp]
    treated_row: int
    treated_post_mask: np.ndarray
    pre_mask: np.ndarray
    feats: List[str]
    meta: Dict


def long_to_wide_for_episode(df: pd.DataFrame, include_covariates: bool = True) -> EpisodeWide:
    """
    Convierte panel largo del episodio a matrices wide (unidades x fecha) y
    aplica una máscara **a prueba de fugas**: la víctima en POST siempre queda oculta.
    """
    d = df.copy()
    d["date"] = _as_datetime(d["date"])
    d = d.sort_values(["date", "store_nbr", "item_nbr"])

    # Víctima primero
    d["__is_victim__"] = (d["treated_unit"] == 1).astype(int)
    units = _sorted_unique(d.sort_values(["__is_victim__", "unit_id"], ascending=[False, True])["unit_id"])
    dates = _sorted_unique(d["date"])
    U, T = len(units), len(dates)

    unit2idx = {u: i for i, u in enumerate(units)}
    date2idx = {dt: j for j, dt in enumerate(dates)}

    Y = np.full((U, T), np.nan, dtype=float)
    TM = np.ones((U, T), dtype=bool)
    treated_time_vec = np.zeros(T, dtype=bool)

    feats = select_feature_cols(d) if include_covariates else []
    P = len(feats)
    X = np.zeros((U, T, P), dtype=float) if include_covariates and P > 0 else None

    has_train_mask = "train_mask" in d.columns
    # poblar matrices
    for _, r in d.iterrows():
        i = unit2idx[r["unit_id"]]
        j = date2idx[r["date"]]
        Y[i, j] = float(r["sales"])
        if int(r["treated_unit"]) == 1 and int(r.get("treated_time", 0)) == 1:
            treated_time_vec[j] = True
        if include_covariates and P > 0:
            for k, c in enumerate(feats):
                val = r.get(c, np.nan)
                X[i, j, k] = 0.0 if pd.isna(val) else float(val)
        if has_train_mask:
            TM[i, j] = bool(int(r["train_mask"]))

    # fila víctima y reconstrucción robusta de POST si hiciera falta
    victim_unit = d.loc[d["treated_unit"] == 1, "unit_id"].iloc[0]
    treated_row = units.index(victim_unit)
    if not treated_time_vec.any():
        # fallback: usa D para la víctima
        try:
            d_v = d[d["treated_unit"] == 1].sort_values("date")
            idx_first_one = d_v.index[d_v["D"].astype(int).to_numpy() == 1]
            if len(idx_first_one) > 0:
                first_date = d_v.loc[idx_first_one[0], "date"]
                j0 = date2idx[first_date]
                treated_time_vec[j0:] = True
                logging.warning("[MASK] 'treated_time' vacío; POST reconstruido desde 'D' para la víctima.")
        except Exception:
            pass

    pre_mask = ~treated_time_vec

    # máscara final M
    M = TM if has_train_mask else np.isfinite(Y)
    # Forzar SIEMPRE ocultamiento de víctima en POST
    n_post_before = int(np.sum(M[treated_row, treated_time_vec]))
    M[treated_row, treated_time_vec] = False
    n_post_after = int(np.sum(M[treated_row, treated_time_vec]))
    if n_post_before > 0:
        logging.warning(f"[MASK] Víctima tenía {n_post_before} celdas POST visibles; ahora {n_post_after}.")

    meta = {
        "n_units": U, "n_time": T,
        "victim_unit": units[treated_row],
        "n_controls": U - 1,
        "n_pre": int(pre_mask.sum()),
        "n_post": int(treated_time_vec.sum()),
        "n_train_victim_post_before": n_post_before,
        "n_train_victim_post_after": n_post_after,
    }
    return EpisodeWide(Y=Y, M=M, X=X, units=units, dates=dates,
                       treated_row=treated_row,
                       treated_post_mask=treated_time_vec,
                       pre_mask=pre_mask,
                       feats=feats,
                       meta=meta)


# ---------------------------------------------------------------------
# Núcleo GSC: SoftImpute + Ridge alternado (con estandarización)
# ---------------------------------------------------------------------

@dataclass
class GSCConfig:
    rank: int = 2
    tau: float = 1.0
    alpha: float = 0.01
    max_inner: int = 120
    max_outer: int = 10
    tol: float = 1e-4
    include_covariates: bool = True
    random_state: int = 42
    standardize_y: bool = True
    # robustez y no‑negatividad
    link: str = "identity"
    link_eps: float = 0.0
    pred_clip_min: Optional[float] = 0.0
    calibrate_victim: str = "level"        # 'none' | 'level' | 'level_and_trend'
    post_smooth_window: int = 1            # 1 => sin suavizado

class SoftImpute:
    """SoftImpute con SVD denso y máscara M (True si observado/entrenable)."""

    def __init__(self, tau: float, rank: int, max_iters: int = 100, tol: float = 1e-4):
        self.tau = float(tau)
        self.rank = int(rank)
        self.max_iters = int(max_iters)
        self.tol = float(tol)

    def fit(self, W: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        min_L 0.5||M*(W - L)||_F^2 + tau * ||L||_*  por soft-thresholding.
        """
        L = np.nan_to_num(W, nan=0.0).copy()
        prev_missing = L.copy()

        for _ in range(self.max_iters):
            try:
                U, s, Vt = np.linalg.svd(L, full_matrices=False)
            except np.linalg.LinAlgError:
                L = L + 1e-6 * np.random.randn(*L.shape)
                U, s, Vt = np.linalg.svd(L, full_matrices=False)

            s_shrunk = np.maximum(s - self.tau, 0.0)
            r = int(min(self.rank, max(1, np.sum(s_shrunk > 0))))
            L_new = (U[:, :r] * s_shrunk[:r]) @ Vt[:r, :]

            # Mantener observados (M=True) en W; missing como L_new
            L = np.where(M, W, L_new)

            miss_mask = ~M
            num = np.linalg.norm((L - prev_missing)[miss_mask]) if np.any(miss_mask) else 0.0
            den = np.linalg.norm(prev_missing[miss_mask]) + 1e-8 if np.any(miss_mask) else 1.0
            rel = num / den
            if rel < self.tol:
                break
            prev_missing = L.copy()

        # Reconstrucción final
        U, s, Vt = np.linalg.svd(L, full_matrices=False)
        s_shrunk = np.maximum(s - self.tau, 0.0)
        r = int(min(self.rank, max(1, np.sum(s_shrunk > 0))))
        L_star = (U[:, :r] * s_shrunk[:r]) @ Vt[:r, :]
        return L_star


class GSCModel:
    """GSC con alternancia entre L (factores) y β (covariables) sobre máscara M."""

    def __init__(self, cfg: GSCConfig):
        self.cfg = cfg
        self.beta_: Optional[np.ndarray] = None
        self.L_: Optional[np.ndarray] = None
        self.history_: Dict = {}
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._x_means: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._link: _Link = _Link(cfg.link, cfg.link_eps)

    @staticmethod
    def _normalize_X_on_mask(X: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normaliza X usando sólo celdas con M=True."""
        U, T, P = X.shape
        mask3 = np.repeat(M[:, :, None], P, axis=2)
        X_obs = np.where(mask3, X, np.nan)
        means = np.nanmean(X_obs, axis=(0, 1))
        stds = np.nanstd(X_obs, axis=(0, 1)) + 1e-8
        Xn = (X - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)
        return Xn, means, stds

    def fit(self, Y: np.ndarray, M: np.ndarray, X: Optional[np.ndarray] = None) -> "GSCModel":
        include_X = self.cfg.include_covariates and X is not None and X.ndim == 3 and X.shape[2] > 0
        P = 0 if not include_X else X.shape[2]

        # Transformación + estandarización de Y (sólo celdas M=True)
        Yt = self._link.forward(Y)
        if self.cfg.standardize_y:
            obs_vals = Yt[M]
            self._y_mean = float(np.mean(obs_vals))
            self._y_std = float(np.std(obs_vals) + 1e-8)
            Ys = (Yt - self._y_mean) / self._y_std
        else:
            self._y_mean, self._y_std = 0.0, 1.0
            Ys = Yt.copy()

        # Normalizar X (si aplica)
        if include_X:
            Xn, xm, xs = self._normalize_X_on_mask(X, M)
            self._x_means, self._x_std = xm, xs
        else:
            Xn, self._x_means, self._x_std = None, None, None

        def xbeta_all(beta_vec: Optional[np.ndarray]) -> np.ndarray:
            if not include_X or beta_vec is None:
                return np.zeros_like(Ys)
            return np.tensordot(Xn, beta_vec, axes=(2, 0))  # (U,T,P)·(P,) -> (U,T)

        beta = np.zeros(P, dtype=float) if include_X else None
        hist = {"rmspe_obs": []}
        L = np.zeros_like(Ys)

        for _ in range(self.cfg.max_outer):
            # Paso 1: SoftImpute en residuales (sobre M)
            R = Ys - (xbeta_all(beta) if include_X else 0.0)
            W = np.where(M, R, np.nan)
            L_est = SoftImpute(self.cfg.tau, self.cfg.rank, self.cfg.max_inner, self.cfg.tol).fit(W, M)
            L = L_est

            # Paso 2: Ridge para β con observados
            if include_X:
                R_obs = (Ys - L)[M]
                X_obs = Xn[M, :]
                if X_obs.size == 0:
                    beta = np.zeros(P, dtype=float)
                else:
                    A, _ = _augment_ridge(X_obs, self.cfg.alpha)
                    y_aug = np.concatenate([R_obs, np.zeros(X_obs.shape[1])]) if self.cfg.alpha > 0 else R_obs
                    beta, *_ = np.linalg.lstsq(A, y_aug, rcond=None)

            yhat_obs = (L + (xbeta_all(beta) if include_X else 0.0))[M]
            hist["rmspe_obs"].append(_rmspe(Ys[M], yhat_obs))

            if len(hist["rmspe_obs"]) >= 2 and abs(hist["rmspe_obs"][-1] - hist["rmspe_obs"][-2]) < 1e-5:
                break

        self.beta_ = beta
        self.L_ = L
        self.history_ = hist
        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        assert self.L_ is not None, "Modelo no ajustado."
        include_X = self.cfg.include_covariates and self.beta_ is not None and X is not None and X.ndim == 3
        if include_X:
            Xn = (X - self._x_means.reshape(1, 1, -1)) / self._x_std.reshape(1, 1, -1)
            Xb = np.tensordot(Xn, self.beta_, axes=(2, 0))
            Yhat_std = self.L_ + Xb
        else:
            Yhat_std = self.L_

        # desestandarizar + inversa del link
        Yhat_t = Yhat_std * self._y_std + self._y_mean
        Yhat = self._link.inverse(Yhat_t)

        # Clip mínimo (última salvaguarda)
        if self.cfg.pred_clip_min is not None:
            Yhat = np.maximum(Yhat, float(self.cfg.pred_clip_min))

        return Yhat


# ---------------------------------------------------------------------
# CV temporal y utilidades
# ---------------------------------------------------------------------

@dataclass
class CVCfg:
    folds: int = 3
    holdout_days: int = 21
    gap_days: int = 7
    grid_ranks: Tuple[int, ...] = (1, 2, 3)
    grid_tau: Tuple[float, ...] = (0.5, 1.0, 2.0)
    grid_alpha: Tuple[float, ...] = (0.001, 0.01, 0.1)

def _svd_scale_pre(ep: EpisodeWide) -> float:
    """Escala típica para tau basada en el mayor singular del PRE."""
    Ypre = ep.Y[:, ep.pre_mask]
    Mpre = ep.M[:, ep.pre_mask]
    Ytmp = np.where(Mpre, Ypre, 0.0)
    try:
        s0 = float(np.linalg.svd(Ytmp, full_matrices=False, compute_uv=False)[0])
    except Exception:
        s0 = float(np.sqrt(np.nanmean(Ypre**2)) * np.sqrt(min(Ypre.shape)))
    if not np.isfinite(s0) or s0 <= 1e-8:
        s0 = 1.0
    return s0

def _sanitize_grids(ep: EpisodeWide, cv: CVCfg) -> Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[float, ...], Dict]:
    U, T = ep.Y.shape
    max_rank = max(1, min(U - 1, T - 1))
    ranks = sorted(set(int(np.clip(r, 1, max_rank)) for r in cv.grid_ranks))
    if not ranks:
        ranks = tuple(int(x) for x in range(1, min(4, max_rank) + 1))

    s0 = _svd_scale_pre(ep)
    tau_lo = max(1e-3, 0.05 * s0)
    tau_hi = max(tau_lo * 1.25, 0.8 * s0)
    taus = [float(np.clip(t, tau_lo, tau_hi)) for t in cv.grid_tau if np.isfinite(t) and t > 0]
    if not taus:
        taus = list(np.geomspace(tau_lo, tau_hi, num=5))
    taus = tuple(sorted(set(taus)))

    alphas = [float(np.clip(a, 1e-5, 1e-1)) for a in cv.grid_alpha if np.isfinite(a)]
    if not alphas:
        alphas = list(np.geomspace(1e-4, 1e-2, num=5))
    alphas = tuple(sorted(set(alphas)))

    info = {"tau_lo": tau_lo, "tau_hi": tau_hi, "max_rank": max_rank, "s0_pre": s0}
    logging.info(f"[CV-SANITIZE] rank≤{max_rank}, tau∈[{tau_lo:.4g},{tau_hi:.4g}], alpha∈[1e-5,1e-1]")
    return tuple(ranks), tuple(taus), tuple(alphas), info

def _cv_folds_pre_indices(ep: EpisodeWide, holdout_days: int, gap_days: int, k_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    pre_idx = np.where(ep.pre_mask)[0]
    if len(pre_idx) == 0:
        return []
    H = int(max(1, holdout_days))
    G = int(max(0, gap_days))
    end_max = pre_idx[-1] - G
    if end_max < pre_idx[0] + H - 1:
        return []

    folds = []
    for f in range(k_folds):
        hold_end = end_max - f * H
        hold_start = hold_end - H + 1
        if hold_start < pre_idx[0]:
            break
        gap_start = max(pre_idx[0], hold_start - G)
        gap_idx = np.arange(gap_start, hold_start) if G > 0 else np.array([], dtype=int)
        hold_idx = np.arange(hold_start, hold_end + 1)
        folds.append((gap_idx, hold_idx))
    return folds

def _apply_time_mask(M: np.ndarray, gap_idx: np.ndarray, hold_idx: np.ndarray) -> np.ndarray:
    M_train = M.copy()
    if hold_idx.size > 0:
        M_train[:, hold_idx] = False
    if gap_idx.size > 0:
        M_train[:, gap_idx] = False
    return M_train

def time_cv_select(ep: EpisodeWide, cv: CVCfg, base_cfg: GSCConfig) -> Tuple[GSCConfig, Dict]:
    folds = _cv_folds_pre_indices(ep, cv.holdout_days, cv.gap_days, cv.folds)
    if not folds:
        logging.info("CV temporal omitida (pre insuficiente); se usa configuración base.")
        return base_cfg, {"cv_used": False, "score": np.nan, "best": asdict(base_cfg), "method": "grid"}

    grid_ranks, grid_tau, grid_alpha, san_info = _sanitize_grids(ep, cv)

    results = []
    for rnk in grid_ranks:
        for tau in grid_tau:
            for alpha in grid_alpha:
                cfg_try = GSCConfig(rank=int(rnk), tau=float(tau), alpha=float(alpha),
                                    max_inner=base_cfg.max_inner, max_outer=base_cfg.max_outer,
                                    tol=base_cfg.tol, include_covariates=base_cfg.include_covariates,
                                    standardize_y=base_cfg.standardize_y,
                                    random_state=base_cfg.random_state,
                                    link=base_cfg.link, link_eps=base_cfg.link_eps,
                                    pred_clip_min=base_cfg.pred_clip_min,
                                    calibrate_victim=base_cfg.calibrate_victim,
                                    post_smooth_window=base_cfg.post_smooth_window)
                fold_scores = []
                for gap_idx, hold_idx in folds:
                    M_train = _apply_time_mask(ep.M, gap_idx, hold_idx)
                    model = GSCModel(cfg_try).fit(ep.Y, M_train, ep.X if cfg_try.include_covariates else None)
                    Yhat = model.predict(ep.X if cfg_try.include_covariates else None)
                    y_true = ep.Y[ep.treated_row, hold_idx]
                    y_hat = Yhat[ep.treated_row, hold_idx]
                    fold_scores.append(_rmspe(y_true, y_hat))
                score = float(np.nanmedian(fold_scores))
                score = _finite_or_big(score)
                results.append({"rank": rnk, "tau": float(tau), "alpha": float(alpha),
                                "score": score, "scores": fold_scores})

    best = min(results, key=lambda z: z["score"])
    best_cfg = GSCConfig(rank=int(best["rank"]), tau=float(best["tau"]), alpha=float(best["alpha"]),
                         max_inner=base_cfg.max_inner, max_outer=base_cfg.max_outer,
                         tol=base_cfg.tol, include_covariates=base_cfg.include_covariates,
                         standardize_y=base_cfg.standardize_y,
                         random_state=base_cfg.random_state,
                         link=base_cfg.link, link_eps=base_cfg.link_eps,
                         pred_clip_min=base_cfg.pred_clip_min,
                         calibrate_victim=base_cfg.calibrate_victim,
                         post_smooth_window=base_cfg.post_smooth_window)
    logging.info(f"[GRID-GSC] best_score(RMSPE)={best['score']:.6f} | best_params="
                 f"rank={best['rank']}, tau={best['tau']}, alpha={best['alpha']}")
    try:
        if len(grid_ranks) > 0 and best["rank"] in (min(grid_ranks), max(grid_ranks)):
            which = "mínimo" if best["rank"] == min(grid_ranks) else "máximo"
            logging.info(f"[GRID-GSC] 'rank' está en el {which} de la grilla; considera ampliar el rango.")
        if len(grid_tau) > 0 and best["tau"] in (min(grid_tau), max(grid_tau)):
            which = "mínimo" if best["tau"] == min(grid_tau) else "máximo"
            logging.info(f"[GRID-GSC] 'tau' está en el {which} de la grilla; considera ampliar el rango.")
        if len(grid_alpha) > 0 and best["alpha"] in (min(grid_alpha), max(grid_alpha)):
            which = "mínimo" if best["alpha"] == min(grid_alpha) else "máximo"
            logging.info(f"[GRID-GSC] 'alpha' está en el {which} de la grilla; considera ampliar el rango.")
    except Exception:
        pass

    return best_cfg, {"cv_used": True, "score": float(best["score"]), "best": best,
                      "all": results, "method": "grid", **san_info}

def optuna_cv_select(ep: EpisodeWide, cv: CVCfg, base_cfg: GSCConfig,
                     trials: int = 300, seed: int = 42) -> Tuple[GSCConfig, Dict]:
    try:
        import optuna  # type: ignore
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        from optuna.pruners import MedianPruner  # type: ignore
    except Exception as e:
        logging.warning(f"Optuna no disponible ({e}); se usa búsqueda en grilla.")
        return time_cv_select(ep, cv, base_cfg)

    folds = _cv_folds_pre_indices(ep, cv.holdout_days, cv.gap_days, cv.folds)
    if not folds:
        logging.info("HPO (Optuna) omitido (pre insuficiente); se usa configuración base.")
        return base_cfg, {"cv_used": False, "score": np.nan, "best": asdict(base_cfg), "method": "optuna"}

    U, T = ep.Y.shape
    # Rango amplio para rank - permitir exploración completa
    max_rank = int(max(1, min(U - 1, T - 1, 6)))  # Máximo 6 (más opciones)

    s0 = _svd_scale_pre(ep)
    # Rangos amplios para tau - explorar desde muy bajo hasta moderado
    tau_lo = float(max(5e-5, 0.005 * s0))   # Mínimo más bajo
    tau_hi = float(max(tau_lo * 10.0, 0.4 * s0))  # Rango más amplio (10x)
    # Rangos amplios para alpha - explorar desde muy bajo hasta moderado
    alpha_lo, alpha_hi = 5e-5, 1e-2  # Rango más amplio

    pruner = MedianPruner(n_warmup_steps=max(1, len(folds) // 2))
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, study_name="gsc_hpo")

    def objective(trial) -> float:
        rank = trial.suggest_int("rank", 1, max_rank)
        tau = trial.suggest_float("tau", tau_lo, tau_hi, log=True)
        alpha = trial.suggest_float("alpha", alpha_lo, alpha_hi, log=True)
        incX = base_cfg.include_covariates and (ep.X is not None and ep.X.ndim == 3 and ep.X.shape[2] > 0)
        if incX:
            incX = trial.suggest_categorical("include_covariates", [True, False])
        max_inner = trial.suggest_int("max_inner", 60, 200)

        cfg_try = GSCConfig(rank=rank, tau=tau, alpha=alpha,
                            max_inner=max_inner, max_outer=base_cfg.max_outer,
                            tol=base_cfg.tol, include_covariates=bool(incX),
                            standardize_y=base_cfg.standardize_y,
                            random_state=base_cfg.random_state,
                            link=base_cfg.link, link_eps=base_cfg.link_eps,
                            pred_clip_min=base_cfg.pred_clip_min,
                            calibrate_victim=base_cfg.calibrate_victim,
                            post_smooth_window=base_cfg.post_smooth_window)

        scores = []
        cv_ratios = []  # Para penalizar contrafactuales planos
        for step, (gap_idx, hold_idx) in enumerate(folds):
            M_train = _apply_time_mask(ep.M, gap_idx, hold_idx)
            model = GSCModel(cfg_try).fit(ep.Y, M_train, ep.X if cfg_try.include_covariates else None)
            Yhat = model.predict(ep.X if cfg_try.include_covariates else None)
            y_true = ep.Y[ep.treated_row, hold_idx]
            y_hat = Yhat[ep.treated_row, hold_idx]
            sc = _finite_or_big(_rmspe(y_true, y_hat))
            scores.append(sc)
            
            # Calcular ratio de variabilidad para penalizar contrafactuales planos
            cv_true = np.std(y_true) / (np.mean(np.abs(y_true)) + 1e-8)
            cv_hat = np.std(y_hat) / (np.mean(np.abs(y_hat)) + 1e-8)
            cv_ratio = cv_hat / (cv_true + 1e-8) if cv_true > 0.05 else 1.0
            cv_ratios.append(cv_ratio)
            
            trial.report(sc, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        median_score = float(np.median(scores))
        median_cv_ratio = float(np.median(cv_ratios))
        
        # Penalizar si el contrafactual es demasiado plano (cv_ratio < 0.3)
        # o demasiado variable (cv_ratio > 1.5)
        if median_cv_ratio < 0.3:
            penalty = (0.3 - median_cv_ratio) * 0.5  # Penalización por ser plano
            median_score += penalty
        elif median_cv_ratio > 1.5:
            penalty = (median_cv_ratio - 1.5) * 0.3  # Penalización por ser muy variable
            median_score += penalty
            
        return median_score

    logging.info(f"Iniciando HPO (Optuna) para GSC con {int(trials)} trials...")
    study.optimize(objective, n_trials=int(trials), show_progress_bar=False)

    bp = study.best_params
    best_cfg = GSCConfig(
        rank=int(bp.get("rank", base_cfg.rank)),
        tau=float(bp.get("tau", base_cfg.tau)),
        alpha=float(bp.get("alpha", base_cfg.alpha)),
        max_inner=int(bp.get("max_inner", base_cfg.max_inner)),
        max_outer=base_cfg.max_outer,
        tol=base_cfg.tol,
        include_covariates=bool(bp.get("include_covariates", base_cfg.include_covariates)),
        standardize_y=base_cfg.standardize_y,
        random_state=base_cfg.random_state,
        link=base_cfg.link, link_eps=base_cfg.link_eps,
        pred_clip_min=base_cfg.pred_clip_min,
        calibrate_victim=base_cfg.calibrate_victim,
        post_smooth_window=base_cfg.post_smooth_window
    )
    info = {
        "cv_used": True,
        "score": float(study.best_value),
        "best": {"method": "optuna", **bp},
        "method": "optuna",
        "tau_lo": tau_lo, "tau_hi": tau_hi, "max_rank": max_rank, "s0_pre": s0
    }

    logging.info(f"[HPO-GSC] best_score(RMSPE)={study.best_value:.6f} | best_params={bp}")
    _check_boundary_int(int(bp.get("rank", best_cfg.rank)), 1, max_rank, "rank")
    _check_boundary_float(float(bp.get("tau", best_cfg.tau)), tau_lo, tau_hi, "tau")
    _check_boundary_float(float(bp.get("alpha", best_cfg.alpha)), 1e-5, 1e-1, "alpha")
    _check_boundary_int(int(bp.get("max_inner", best_cfg.max_inner)), 60, 200, "max_inner")

    return best_cfg, info


# ---------------------------------------------------------------------
# Placebos, LOO y sensibilidad
# ---------------------------------------------------------------------

def placebo_space(ep: EpisodeWide, cfg: GSCConfig, max_units: int = 20) -> pd.DataFrame:
    rows = []
    control_rows = [i for i in range(len(ep.units)) if i != ep.treated_row][:max_units]
    for r in control_rows:
        M_alt = ep.M.copy()
        M_alt[r, ep.treated_post_mask] = False
        model = GSCModel(cfg).fit(ep.Y, M_alt, ep.X if cfg.include_covariates else None)
        Yhat = model.predict(ep.X if cfg.include_covariates else None)
        eff = ep.Y[r, ep.treated_post_mask] - Yhat[r, ep.treated_post_mask]
        rows.append({
            "unit_id": ep.units[r],
            "att_placebo_mean": float(np.nanmean(eff)),
            "att_placebo_sum": float(np.nansum(eff)),
            "rmspe_pre": float(_rmspe(ep.Y[r, ep.pre_mask], Yhat[r, ep.pre_mask]))
        })
    return pd.DataFrame(rows)

def placebo_time(ep: EpisodeWide, cfg: GSCConfig) -> pd.DataFrame:
    pre_idx = np.where(ep.pre_mask)[0]
    Lpre = len(pre_idx)
    Lpost = int(ep.treated_post_mask.sum())
    H = min(Lpost, max(7, Lpre // 2))
    if Lpre <= H + 7:
        return pd.DataFrame([{"att_placebo_mean": np.nan, "att_placebo_sum": np.nan, "used": False}])

    hold = pre_idx[-H:]
    M_alt = ep.M.copy()
    M_alt[ep.treated_row, hold] = False
    model = GSCModel(cfg).fit(ep.Y, M_alt, ep.X if cfg.include_covariates else None)
    Yhat = model.predict(ep.X if cfg.include_covariates else None)
    eff = ep.Y[ep.treated_row, hold] - Yhat[ep.treated_row, hold]
    return pd.DataFrame([{
        "att_placebo_mean": float(np.nanmean(eff)),
        "att_placebo_sum": float(np.nansum(eff)),
        "used": True,
        "H": int(H)
    }])

def leave_one_out(ep: EpisodeWide, cfg: GSCConfig, max_units: int = 20) -> pd.DataFrame:
    rows = []
    control_rows = [i for i in range(len(ep.units)) if i != ep.treated_row][:max_units]
    for r in control_rows:
        keep = [i for i in range(len(ep.units)) if i != r]
        Y = ep.Y[keep, :]
        M = ep.M[keep, :]
        X = ep.X[keep, :, :] if ep.X is not None else None
        treated_row = next(i for i, k in enumerate(keep) if k == ep.treated_row)
        model = GSCModel(cfg).fit(Y, M, X if cfg.include_covariates else None)
        Yhat = model.predict(X if cfg.include_covariates else None)
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
    rng = np.random.default_rng(base_cfg.random_state)
    U, T = ep.Y.shape
    max_rank = int(max(1, min(U - 1, T - 1, 10)))
    s0 = _svd_scale_pre(ep)
    tau_lo = float(max(1e-3, 0.05 * s0))
    tau_hi = float(max(tau_lo * 1.25, 0.8 * s0))
    rows = []
    for _ in range(n_samples):
        rank = int(rng.integers(1, max_rank + 1))
        tau = float(10 ** rng.uniform(np.log10(tau_lo), np.log10(tau_hi)))
        alpha = float(10 ** rng.uniform(-4, -2))  # [1e-4,1e-2]
        incX = bool(rng.integers(0, 2)) if base_cfg.include_covariates else False
        cfg = GSCConfig(rank=rank, tau=tau, alpha=alpha,
                        max_inner=base_cfg.max_inner, max_outer=base_cfg.max_outer,
                        tol=base_cfg.tol, include_covariates=incX,
                        standardize_y=base_cfg.standardize_y,
                        random_state=base_cfg.random_state,
                        link=base_cfg.link, link_eps=base_cfg.link_eps,
                        pred_clip_min=base_cfg.pred_clip_min,
                        calibrate_victim=base_cfg.calibrate_victim,
                        post_smooth_window=base_cfg.post_smooth_window)
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
    include_covariates: bool = True
    max_episodes: Optional[int] = None
    log_level: str = "INFO"
    # CV
    cv_folds: int = 3
    cv_holdout: int = 21
    cv_gap: int = 7
    grid_ranks: Tuple[int, ...] = (1, 2, 3)
    grid_tau: Tuple[float, ...] = (0.5, 1.0, 2.0)
    grid_alpha: Tuple[float, ...] = (0.001, 0.01, 0.1)
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
    max_inner: int = 120
    max_outer: int = 10
    tol: float = 1e-4
    random_state: int = 42
    # Gap de entrenamiento final antes del tratamiento
    train_gap: int = 7
    # HPO
    hpo_trials: int = 300  # 0 => off
    # robustez/no-negatividad
    link: str = "identity"
    link_eps: float = 0.0
    pred_clip_min: Optional[float] = 0.0
    calibrate_victim: str = "level"
    post_smooth_window: int = 1
    # Salidas
    out_dir: Path = field(default_factory=lambda: _data_root() / "processed" / "gsc")
    # NUEVO: etiquetar experimento y resolver raíz figures/<exp_tag>
    exp_tag: str = "A_base"
    fig_root: Path = Path(default_factory=_fig_root)
    gsc_dir: Path = field(default_factory=lambda: _data_root() / "processed" / "gsc")
    donors_csv: Path = field(default_factory=lambda: _data_root() / "processed" / "donors.csv")
    episodes_index: Optional[Path] = field(default_factory=lambda: _data_root() / "processed" / "episodes_index.parquet")



# --- helper local (colócalo cerca de otros helpers de paths) ---
def _exp_root(exp_tag: Optional[str]) -> Optional[Path]:
    return (Path("figures") / str(exp_tag)) if exp_tag else None

def _discover_episode_files(gsc_dir: Path) -> List[Path]:
    # Excluir archivos de métricas/calidad que no son episodios
    exclude = {"donor_quality.parquet", "gsc_metrics.parquet"}
    files = [p for p in gsc_dir.glob("*.parquet") if p.name not in exclude]
    return sorted(files)

def _episode_id_from_path(p: Path) -> str:
    return p.stem

def _load_episode_df(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = _as_datetime(df["date"])
    return df


def run_episode(p: Path, cfg: RunConfig) -> Dict:
    ep_id = _episode_id_from_path(p)
    df = _load_episode_df(p)
    epw = long_to_wide_for_episode(df, include_covariates=cfg.include_covariates)

    # Config base y CV
    base = GSCConfig(rank=cfg.rank, tau=cfg.tau, alpha=cfg.alpha,
                     max_inner=cfg.max_inner, max_outer=cfg.max_outer,
                     tol=cfg.tol, include_covariates=cfg.include_covariates,
                     standardize_y=True,
                     random_state=cfg.random_state,
                     link=cfg.link, link_eps=cfg.link_eps,
                     pred_clip_min=cfg.pred_clip_min,
                     calibrate_victim=cfg.calibrate_victim,
                     post_smooth_window=cfg.post_smooth_window)
    cv_cfg = CVCfg(folds=cfg.cv_folds, holdout_days=cfg.cv_holdout, gap_days=cfg.cv_gap,
                   grid_ranks=cfg.grid_ranks, grid_tau=cfg.grid_tau, grid_alpha=cfg.grid_alpha)
    if cfg.hpo_trials and cfg.hpo_trials > 0:
        best_cfg, cv_info = optuna_cv_select(epw, cv_cfg, base, trials=cfg.hpo_trials, seed=cfg.random_state)
    else:
        best_cfg, cv_info = time_cv_select(epw, cv_cfg, base)

    # Entrenamiento final con embargo en PRE
    M_final = epw.M.copy()
    pre_idx = np.where(epw.pre_mask)[0]
    if len(pre_idx) > 0 and cfg.train_gap > 0:
        gap_idx = pre_idx[-min(cfg.train_gap, len(pre_idx)):]
        M_final[:, gap_idx] = False

    model = GSCModel(best_cfg).fit(epw.Y, M_final, epw.X if best_cfg.include_covariates else None)
    Yhat = model.predict(epw.X if best_cfg.include_covariates else None)

    # Serie víctima
    y_obs_vec = epw.Y[epw.treated_row, :]
    y_hat_vec = Yhat[epw.treated_row, :]
    obs_mask  = np.isfinite(y_obs_vec)

    # Calibración exacta en PRE (nivel o nivel+trend)
    mode = (best_cfg.calibrate_victim or "none").lower()
    if mode in {"level", "level_and_trend"}:
        jj = np.where(epw.pre_mask & obs_mask)[0]
        if jj.size > 0:
            res = (y_obs_vec[jj] - y_hat_vec[jj]).astype(float)
            if mode == "level":
                bias = float(np.nanmean(res))
                y_hat_vec = y_hat_vec + bias
            else:
                t = jj.astype(float)
                t0 = float(np.mean(t))
                t_c = t - t0
                A = np.vstack([np.ones_like(t_c), t_c]).T
                try:
                    ab, *_ = np.linalg.lstsq(A, res, rcond=None)
                    a, b = float(ab[0]), float(ab[1])
                except Exception:
                    a, b = float(np.nanmean(res)), 0.0
                full_t = np.arange(len(y_hat_vec), dtype=float)
                full_t_c = full_t - t0
                y_hat_vec = y_hat_vec + a + b * full_t_c

    # Suavizado opcional
    W = int(max(1, best_cfg.post_smooth_window))
    if W > 1:
        k = np.ones(W, dtype=float) / float(W)
        base_val = float(np.nanmean(y_hat_vec)) if np.isfinite(np.nanmean(y_hat_vec)) else 0.0
        row_conv = np.convolve(np.nan_to_num(y_hat_vec, nan=base_val), k, mode="same")
        y_hat_vec = row_conv

    # No-negatividad
    if best_cfg.pred_clip_min is not None:
        y_hat_vec = np.maximum(y_hat_vec, float(best_cfg.pred_clip_min))

    # Inyectar la fila calibrada en la matriz final
    Yhat[epw.treated_row, :] = y_hat_vec

    # Métricas PRE
    y_obs_pre = y_obs_vec[epw.pre_mask]
    y_hat_pre = Yhat[epw.treated_row, epw.pre_mask]
    rmspe_pre = _rmspe(y_obs_pre, y_hat_pre)
    mae_pre   = _mae(y_obs_pre, y_hat_pre)

    # ---- Anti-overfit: re‑entrena si PRE ≈ copiado ----
    overfit_refit = False
    # Detección multi-criterio de overfitting:
    # 1. RMSPE extremadamente bajo (< 0.05)
    # 2. Correlación casi perfecta (> 0.99)
    # 3. Valores casi idénticos (allclose)
    corr_pre = np.corrcoef(y_obs_pre, y_hat_pre)[0, 1] if len(y_obs_pre) > 1 else 0.0
    if (np.isfinite(rmspe_pre) and rmspe_pre < 0.05) or \
       (np.isfinite(corr_pre) and corr_pre > 0.99) or \
       np.allclose(y_obs_pre, y_hat_pre, atol=1e-3, rtol=0.05):
        overfit_refit = True
        logging.warning(f"[{ep_id}] Overfitting detectado (RMSPE_pre={rmspe_pre:.4f} < 0.10). Re‑entrenando con mayor regularización.")
        s0 = _svd_scale_pre(epw)
        # AUMENTAR regularización agresivamente para combatir overfitting
        # - Reducir rank drásticamente (capacidad del modelo)
        # - Aumentar tau significativamente (penalización nuclear)
        # - Aumentar alpha (penalización de covariables)
        # - Solo calibración de nivel (no tendencia que puede sobreajustar)
        safer_cfg = GSCConfig(
            rank=max(1, best_cfg.rank // 3),              # Reducir a 1/3 del original
            tau=max(best_cfg.tau * 10.0, 0.5 * s0),       # 10x tau o 50% del singular mayor
            alpha=max(best_cfg.alpha * 5.0, 0.05),        # 5x alpha, mínimo 0.05
            max_inner=best_cfg.max_inner,
            max_outer=best_cfg.max_outer,
            tol=best_cfg.tol,
            include_covariates=best_cfg.include_covariates,
            random_state=best_cfg.random_state,
            standardize_y=True,
            link=best_cfg.link, link_eps=best_cfg.link_eps,
            pred_clip_min=best_cfg.pred_clip_min,
            calibrate_victim="level",                     # Solo nivel, NO tendencia
            post_smooth_window=1,                         # Sin suavizado
        )
        model = GSCModel(safer_cfg).fit(epw.Y, M_final, epw.X if safer_cfg.include_covariates else None)
        Yhat = model.predict(epw.X if safer_cfg.include_covariates else None)
        y_hat_vec = Yhat[epw.treated_row, :]
        # recalibración ligera de nivel
        jj = np.where(epw.pre_mask & obs_mask)[0]
        if jj.size > 0:
            bias = float(np.nanmean(y_obs_vec[jj] - y_hat_vec[jj]))
            y_hat_vec = y_hat_vec + bias
            if safer_cfg.pred_clip_min is not None:
                y_hat_vec = np.maximum(y_hat_vec, float(safer_cfg.pred_clip_min))
            Yhat[epw.treated_row, :] = y_hat_vec
        y_hat_pre = Yhat[epw.treated_row, epw.pre_mask]
        rmspe_pre = _rmspe(y_obs_pre, y_hat_pre)
        mae_pre   = _mae(y_obs_pre, y_hat_pre)
    
    # ---- Detección de contrafactuales planos (sobre-regularización) ----
    flat_cf_refit = False
    # Calcular coeficiente de variación del contrafactual en PRE
    y_hat_pre_cv = float(np.nanstd(y_hat_pre) / (np.nanmean(np.abs(y_hat_pre)) + 1e-8))
    y_obs_pre_cv = float(np.nanstd(y_obs_pre) / (np.nanmean(np.abs(y_obs_pre)) + 1e-8))
    
    # Solo aplicar si NO hubo overfit_refit (para evitar conflictos)
    # Si el CF tiene mucha menos variabilidad que el observado, está sobre-regularizado
    # Pero también verificar que RMSPE no sea demasiado bajo (evitar casos de overfit)
    if not overfit_refit and y_hat_pre_cv < 0.5 * y_obs_pre_cv and y_obs_pre_cv > 0.1 and rmspe_pre > 0.20:
        flat_cf_refit = True
        logging.warning(f"[{ep_id}] Contrafactual plano detectado (CV_cf={y_hat_pre_cv:.3f} vs CV_obs={y_obs_pre_cv:.3f}). "
                       f"Re-entrenando con MENOR regularización para capturar variabilidad.")
        s0 = _svd_scale_pre(epw)
        # REDUCIR tau y AUMENTAR rank para más flexibilidad
        less_reg_cfg = GSCConfig(
            rank=min(10, max(best_cfg.rank + 2, 5)),  # Aumentar capacidad
            tau=max(best_cfg.tau * 0.3, 0.01 * s0),   # Reducir regularización nuclear
            alpha=max(best_cfg.alpha * 0.5, 1e-4),    # Reducir regularización de covariables
            max_inner=best_cfg.max_inner,
            max_outer=best_cfg.max_outer,
            tol=best_cfg.tol,
            include_covariates=best_cfg.include_covariates,
            random_state=best_cfg.random_state,
            standardize_y=True,
            link=best_cfg.link, link_eps=best_cfg.link_eps,
            pred_clip_min=best_cfg.pred_clip_min,
            calibrate_victim="level",  # Solo calibración de nivel, no tendencia
            post_smooth_window=1,      # Sin suavizado para preservar variabilidad
        )
        model = GSCModel(less_reg_cfg).fit(epw.Y, M_final, epw.X if less_reg_cfg.include_covariates else None)
        Yhat = model.predict(epw.X if less_reg_cfg.include_covariates else None)
        y_hat_vec = Yhat[epw.treated_row, :]
        # Calibración solo de nivel (no tendencia)
        jj = np.where(epw.pre_mask & obs_mask)[0]
        if jj.size > 0:
            bias = float(np.nanmean(y_obs_vec[jj] - y_hat_vec[jj]))
            y_hat_vec = y_hat_vec + bias
            if less_reg_cfg.pred_clip_min is not None:
                y_hat_vec = np.maximum(y_hat_vec, float(less_reg_cfg.pred_clip_min))
            Yhat[epw.treated_row, :] = y_hat_vec
        y_hat_pre = Yhat[epw.treated_row, epw.pre_mask]
        rmspe_pre = _rmspe(y_obs_pre, y_hat_pre)
        mae_pre   = _mae(y_obs_pre, y_hat_pre)
        # Recalcular CV después del refit
        y_hat_pre_cv_new = float(np.nanstd(y_hat_pre) / (np.nanmean(np.abs(y_hat_pre)) + 1e-8))
        logging.info(f"[{ep_id}] Después de refit: CV_cf={y_hat_pre_cv_new:.3f}, RMSPE_pre={rmspe_pre:.4f}")

    # Filtrar solo las fechas donde la víctima tiene datos reales
    # CRÍTICO: epw.dates contiene TODAS las fechas del panel (víctima + donantes)
    # pero y_obs_vec solo tiene datos válidos en las fechas de la víctima
    # Necesitamos exportar SOLO las fechas con datos reales (no NaNs)
    valid_mask = np.isfinite(y_obs_vec)
    valid_indices = np.where(valid_mask)[0]
    
    # Extraer solo datos válidos
    dates_valid = [epw.dates[j] for j in valid_indices]
    y_obs_valid = y_obs_vec[valid_indices]
    y_hat_valid = y_hat_vec[valid_indices]
    treated_time_valid = epw.treated_post_mask[valid_indices]
    
    # Efectos (calculados sobre datos válidos)
    eff_post = y_obs_vec[epw.treated_post_mask & valid_mask] - y_hat_vec[epw.treated_post_mask & valid_mask]
    att_mean  = float(np.nanmean(eff_post))
    att_sum   = float(np.nansum(eff_post))
    base_level = float(np.nanmean(y_obs_pre))
    rel_att   = _safe_pct(att_mean, base_level)

    # Guardar serie contrafactual + columnas plot‑ready
    # IMPORTANTE: Exportar 'sales' (datos reales) y 'mu0_hat' (contrafactual) 
    # Solo para fechas donde la víctima tiene datos válidos
    
    cf_dir = cfg.out_dir / "cf_series"
    _ensure_dir(cf_dir)
    effect_valid = y_obs_valid - y_hat_valid
    cum_effect_valid = pd.Series(effect_valid).cumsum()
    
    cf = pd.DataFrame({
        "episode_id": ep_id,
        "date": dates_valid,
        "sales": y_obs_valid,             # Serie observada REAL (sin NaNs ni ceros artificiales)
        "y_obs": y_obs_valid,             # Alias
        "mu0_hat": y_hat_valid,           # Contrafactual (nombre estándar para EDA)
        "y_hat": y_hat_valid,             # Alias para compatibilidad hacia atrás
        "treated_time": treated_time_valid.astype(int),
        "obs_mask": np.ones(len(dates_valid), dtype=int),  # Todos son válidos por construcción
        "y_obs_plot": y_obs_valid,        # Alias
        "effect": effect_valid,
        "effect_plot": effect_valid,
        "cum_effect": cum_effect_valid,
        "cum_effect_obs": cum_effect_valid
    })
    cf_path = cf_dir / f"{ep_id}_cf.parquet"
    cf.to_parquet(cf_path, index=False)

    # Placebos / LOO / Sensibilidad
    plac_dir = cfg.out_dir / "placebos"; _ensure_dir(plac_dir)
    loo_dir = cfg.out_dir / "loo"; _ensure_dir(loo_dir)

    if cfg.do_placebo_space:
        ps = placebo_space(epw, best_cfg, max_units=cfg.max_placebos)
        ps["episode_id"] = ep_id
        ps_path = plac_dir / f"{ep_id}_space.parquet"
        ps.to_parquet(ps_path, index=False)
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

    if cfg.do_sensitivity:
        sens = global_sensitivity(epw, best_cfg, n_samples=cfg.sens_samples)
        sens["episode_id"] = ep_id
        sens_dir = cfg.out_dir / "sensitivity"; _ensure_dir(sens_dir)
        sens_path = sens_dir / f"{ep_id}_sens.parquet"
        sens.to_parquet(sens_path, index=False)
        sens_sd = float(np.nanstd(sens["att_sum"])) if not sens.empty else np.nan
    else:
        sens_sd, sens_path = np.nan, None

    # ===== MÉTRICAS CAUSALES COMPARATIVAS =====
    causal_metrics_report = None
    if CAUSAL_METRICS_AVAILABLE:
        try:
            calc = CausalMetricsCalculator(ep_id, "gsc")
            
            # Preparar datos para balance de covariables (si hay X)
            X_treat_pre, X_control_pre = None, None
            if epw.X is not None and epw.X.ndim == 3:
                # Víctima en PRE
                X_treat_pre = epw.X[epw.treated_row, epw.pre_mask, :]
                # Controles en PRE (promedio)
                control_rows = [i for i in range(len(epw.units)) if i != epw.treated_row]
                if control_rows:
                    X_control_pre = epw.X[control_rows, :, :][:, epw.pre_mask, :].reshape(-1, epw.X.shape[2])
            
            # Cargar placebos si existen
            ps_df = pd.read_parquet(ps_path) if ps_path and Path(ps_path).exists() else None
            pt_df = pd.read_parquet(pt_path) if pt_path and Path(pt_path).exists() else None
            sens_df = pd.read_parquet(sens_path) if sens_path and Path(sens_path).exists() else None
            
            # Calcular todas las métricas
            causal_metrics_report = calc.compute_all(
                y_obs_pre=y_obs_pre,
                y_hat_pre=y_hat_pre,
                tau_post=eff_post,
                att_base=att_sum,
                X_treat_pre=X_treat_pre,
                X_control_pre=X_control_pre,
                feature_names=epw.feats if epw.X is not None else None,
                sensitivity_df=sens_df,
                placebo_space_df=ps_df,
                placebo_time_df=pt_df,
                n_control_units=epw.meta["n_controls"]
            )
            
            # Guardar reporte de métricas causales
            causal_dir = cfg.out_dir / "causal_metrics"; _ensure_dir(causal_dir)
            causal_path = causal_dir / f"{ep_id}_causal.parquet"
            pd.DataFrame([causal_metrics_report.to_flat_dict()]).to_parquet(causal_path, index=False)
            
        except Exception as e:
            logging.warning(f"[{ep_id}] Error calculando métricas causales: {e}")

    # Reporte
    rep = {
        "episode_id": ep_id,
        "n_units": epw.meta["n_units"], "n_controls": epw.meta["n_controls"],
        "n_time": epw.meta["n_time"], "n_pre": epw.meta["n_pre"], "n_post": epw.meta["n_post"],
        "victim_unit": epw.meta["victim_unit"],
        "include_covariates": best_cfg.include_covariates,
        "features_used": epw.feats,
        "cv": cv_info,
        "fit": {
            "rmspe_pre": rmspe_pre, "mae_pre": mae_pre,
            "att_mean": att_mean, "att_sum": att_sum, "rel_att_vs_pre_mean": rel_att
        },
        "mask_debug": {
            "train_victim_post_before": epw.meta["n_train_victim_post_before"],
            "train_victim_post_after": epw.meta["n_train_victim_post_after"]
        },
        "overfit_refit": bool(overfit_refit),
        "flat_cf_refit": bool(flat_cf_refit),
        "variability": {
            "cv_obs_pre": float(y_obs_pre_cv),
            "cv_cf_pre": float(y_hat_pre_cv)
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
        "cv_used": bool(rep["cv"]["cv_used"]),
        "link": best_cfg.link,
        "pred_clip_min": best_cfg.pred_clip_min,
        "calibrate_victim": best_cfg.calibrate_victim,
        "post_smooth_window": best_cfg.post_smooth_window,
        "overfit_refit": bool(overfit_refit),
        "flat_cf_refit": bool(flat_cf_refit),
        "cv_obs_pre": float(y_obs_pre_cv),
        "cv_cf_pre": float(y_hat_pre_cv),
    }
    return summary


def run_batch(cfg: RunConfig) -> None:
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
                        
    
    # Solo construir out_dir desde exp_tag si out_dir no está definido y exp_tag existe
    if cfg.out_dir is None and cfg.exp_tag:
        cfg.out_dir = (cfg.gsc_dir / cfg.exp_tag / "gsc").resolve()
    elif cfg.out_dir is None:
        cfg.out_dir = cfg.gsc_dir.parent / "gsc_output"
    
    gsc_dir = cfg.gsc_dir
    if not gsc_dir.exists():
        alt = Path("./data/processed/gsc")
        if alt.exists():
            gsc_dir = alt
        else:
            raise FileNotFoundError(f"No existe la carpeta GSC: {cfg.gsc_dir}")

    files = _discover_episode_files(gsc_dir)
    if cfg.max_episodes is not None:
        files = files[: int(cfg.max_episodes)]

    logging.info(f"Episodios detectados: {len(files)} en {gsc_dir}")
    if cfg.hpo_trials and cfg.hpo_trials > 0:
        logging.info(f"HPO activado: {cfg.hpo_trials} trials de Optuna (fallback a grilla si Optuna no está disponible).")
    else:
        logging.info("HPO desactivado; se usará búsqueda en grilla (rank, tau, alpha).")

    _ensure_dir(cfg.out_dir)
    metrics = []
    for k, p in enumerate(files, start=1):
        try:
            logging.info(f"[{k}/{len(files)}] Ejecutando GSC en {p.name} ...")
            summ = run_episode(p, cfg)
            metrics.append(summ)
            refit_flags = []
            if summ['overfit_refit']:
                refit_flags.append("overfit")
            if summ['flat_cf_refit']:
                refit_flags.append("flat_cf")
            refit_str = f" | refit=[{','.join(refit_flags)}]" if refit_flags else ""
            logging.info(f"OK {p.stem} | ATT_sum={summ['att_sum']:.3f} | RMSPE_pre={summ['rmspe_pre']:.4f} | CV_cf={summ['cv_cf_pre']:.3f}{refit_str}")
        except Exception as e:
            logging.exception(f"Error en episodio {p.name}: {e}")

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
    p.add_argument("--gsc_dir", type=str, default=str(_data_root() / "processed_data/gsc"))
    p.add_argument("--donors_csv", type=str, default=str(_data_root() / "processed_data/donors_per_victim.csv"))
    p.add_argument("--episodes_index", type=str, default=str(_data_root() / "processed_data/episodes_index.parquet"))
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--include_covariates", action="store_true", help="Usar covariables X_it en el ajuste (por defecto: off).")
    p.add_argument("--max_episodes", type=int, default=None, help="Limitar # episodios a procesar.")
    p.add_argument("--log_level", type=str, default="INFO")

    # CV
    p.add_argument("--cv_folds", type=int, default=3)
    p.add_argument("--cv_holdout", type=int, default=21)
    p.add_argument("--cv_gap", type=int, default=7, help="Embargo (días) antes del holdout en CV temporal.")

    # Grillas
    p.add_argument("--grid_ranks", type=str, default="1,2,3")
    p.add_argument("--grid_tau", type=str, default="0.5,1.0,2.0")
    p.add_argument("--grid_alpha", type=str, default="0.001,0.01,0.1")

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
    p.add_argument("--max_inner", type=int, default=120)
    p.add_argument("--max_outer", type=int, default=10)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--train_gap", type=int, default=7, help="Embargo previo al tratamiento en el ajuste final.")

    # HPO
    p.add_argument("--hpo_trials", type=int, default=300,
                   help="N° de trials de Optuna para rank/tau/alpha/include_covariates/max_inner (0=off).")

    # robustez/no‑negatividad
    p.add_argument("--link", type=str, default="identity", choices=["identity", "log1p"], help="Transformación del objetivo Y.")
    p.add_argument("--link_eps", type=float, default=0.0, help="Epsilon para log1p cuando hay ceros.")
    p.add_argument("--pred_clip_min", type=str, default="0", help="Mínimo de predicción (float) o 'none' para desactivar clipping.")
    p.add_argument("--calibrate_victim", type=str, default="level", choices=["none", "level", "level_and_trend"],
                   help="Corrección post‑ajuste en PRE para la víctima.")
    p.add_argument("--post_smooth_window", type=int, default=1, help="Tamaño de ventana de suavizado (1 = off).")

    p.add_argument("--exp_tag", type=str, default=None,
                   help="Etiqueta del experimento. Si se define, todo se guarda en figures/<exp_tag>/gsc")
    
    a = p.parse_args()
    grid_ranks = tuple(int(x.strip()) for x in a.grid_ranks.split(",") if x.strip())
    grid_tau = tuple(float(x.strip()) for x in a.grid_tau.split(",") if x.strip())
    grid_alpha = tuple(float(x.strip()) for x in a.grid_alpha.split(",") if x.strip())

    clip_min: Optional[float]
    if isinstance(a.pred_clip_min, str) and a.pred_clip_min.strip().lower() == "none":
        clip_min = None
    else:
        try:
            clip_min = float(a.pred_clip_min)
        except Exception:
            clip_min = 0.0

    exp_root = _exp_root(a.exp_tag)
    out_dir = (exp_root / "gsc") if exp_root else _to_path(a.out_dir)

    return RunConfig(
        gsc_dir=_to_path(a.gsc_dir),
        donors_csv=_to_path(a.donors_csv),
        episodes_index=_to_path(a.episodes_index) if a.episodes_index else None,

        include_covariates=bool(a.include_covariates),
        max_episodes=a.max_episodes,
        log_level=a.log_level,
        cv_folds=a.cv_folds, cv_holdout=a.cv_holdout, cv_gap=a.cv_gap,
        grid_ranks=grid_ranks, grid_tau=grid_tau, grid_alpha=grid_alpha,
        max_placebos=a.max_placebos, max_loo=a.max_loo,
        do_placebo_space=(not a.no_placebo_space),
        do_placebo_time=(not a.no_placebo_time),
        do_loo=(not a.no_loo),
        do_sensitivity=(not a.no_sensitivity),
        sens_samples=a.sens_samples,
        rank=a.rank, tau=a.tau, alpha=a.alpha,
        max_inner=a.max_inner, max_outer=a.max_outer, tol=a.tol,
        random_state=a.random_state,
        train_gap=a.train_gap,
        hpo_trials=a.hpo_trials,
        link=a.link, link_eps=a.link_eps,
        pred_clip_min=clip_min,
        calibrate_victim=a.calibrate_victim,
        post_smooth_window=max(1, int(a.post_smooth_window)),
        out_dir=out_dir,
        exp_tag=a.exp_tag,
        fig_root=exp_root
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_batch(cfg)