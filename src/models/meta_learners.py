# -*- coding: utf-8 -*-
"""
meta_learners.py
================

Entrena meta‑learners (T/S/X) en el **dataset grande** `meta/all_units.parquet`
y calcula contrafactuales + métricas causales **solo sobre el set pequeño de episodios**
que usarás en GSC (los mismos `episodes_index.parquet` y `donors_per_victim.csv`
que alimentarás a tu Control Sintético Generalizado).

Motivación y cambios clave
--------------------------
- **Cross‑fitting temporal con embargo (gap=7d)** para evitar fugas temporales.
- **Enriquecimiento por fold** con *unit stats* (media, sd, slope, medias últimas 7/28).
- **Reponderación** por unidad y recencia (exp(-λ*days_to_vstart), λ=0.004).
- **Objetivo Poisson** cuando el target luce cuasi‑entero/no‑negativo.
- **Calibración de baseline en PRE (OLS)**: y ≈ a + b·μ̂₀ corrige
  el “aplastamiento” del contrafactual observado en tus gráficos Meta‑X
  (p.ej. páginas 1, 3–4 y 14–17 muestran RMSPE_pre≈0.35–0.56 y líneas
  contrafactuales demasiado planas). :contentReference[oaicite:0]{index=0}

Cómo usarlo (ejemplo de dos corridas)
-------------------------------------
1) **Construir el dataset grande** para entrenar Meta:
   - Ejecuta `pre_algorithm.py` con TODOS los episodios de entrenamiento
     para producir `./data/processed/meta/all_units.parquet`.

2) **Preparar el set pequeño de evaluación** (los mismos episodios que irá a GSC):
   - Ejecuta `pre_algorithm.py` SOLO con esos episodios (p.ej. 10) para generar
     `./data/processed/episodes_index.parquet` y `./data/processed_data/donors_per_victim.csv`.

3) **Entrenar y evaluar Meta en los episodios de GSC** (entrenando con el dataset grande):
   - `python -m meta_learners \
        --learner x \
        --meta_parquet ./data/processed/meta/all_units.parquet \
        --episodes_index ./data/processed/episodes_index.parquet \
        --donors_csv ./data/processed_data/donors_per_victim.csv \
        --out_dir ./data/processed/meta_outputs/x \
        --hpo_trials 0 --cv_folds 3 --cv_holdout 21`

Salidas principales
-------------------
- `meta_outputs/<learner>/cf_series/<episode_id>_cf.parquet`:
  serie Observado vs μ̂₀ (calibrada) y efecto diario/cumulado.
- `meta_outputs/<learner>/reports/<episode_id>.json`: métricas por episodio
  (RMSPE_pre, ATT_sum, ATT_mean, p‑value placebo en el espacio, etc.).
- `meta_outputs/<learner>/meta_metrics_<learner>.parquet`: resumen por episodio.

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
# Utilidades generales y selección de features (anti-fuga)
# ---------------------------------------------------------------------

EXCLUDE_BASE_COLS = {
    # Identificadores / etiquetas
    "id", "date", "year_week", "store_nbr", "item_nbr", "unit_id",
    "family_name", "cluster_id", "w_episode",

    # Objetivo y tratamiento
    "sales", "treated_unit", "treated_time", "D", "is_pre", "train_mask",

    # Info de episodio / calidad de donantes
    "episode_id", "is_victim", "promo_share", "avail_share", "keep", "reason",

    # Mediadores inmediatos (no usar como confounders)
    "onpromotion",
}

# --- Hyper‑parámetros internos ---
_CV_GAP_DAYS = 7
_LAMBDA_RECENCY = 0.004  # peso temporal: exp(-λ * days_to_vstart)
_MIN_TRAIN_FOLD = 30     # mínimo “blando” por grupo antes de caer al modelo constante


def select_feature_cols(df: pd.DataFrame, extra_exclude: Sequence[str] | None = None) -> List[str]:
    """
    Selecciona columnas numéricas 'seguras' como covariables X.
    Incluye automáticamente confounders generados en pre (lags, Fourier, dummies, proxies, etc.).
    Excluye explícitamente etiquetas/mediadores y columnas extra solicitadas.
    """
    ex = set(EXCLUDE_BASE_COLS)
    if extra_exclude:
        ex |= set([c for c in extra_exclude if c in df.columns])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats: List[str] = []
    for c in num_cols:
        if c in ex:
            continue
        feats.append(c)

    # Mantener columnas típicas de confounders/feats
    keep = set([
        "Fsw_log1p", "F_state_excl_store_log1p", "Ow",
        "HNat", "HReg", "HLoc", "is_bridge", "is_additional", "is_work_day",
        "available_A", "month", "ADI", "CV2", "zero_streak", "sc_hat",
        "promo_share_sc_l7", "promo_share_sc_l14",
        "promo_share_sc_excl_l7", "promo_share_sc_excl_l14",
        "class_index_excl_l7", "class_index_excl_l14",
    ])
    feats = [c for c in feats if (c.startswith(("fourier_", "lag_", "dow_", "type_", "cluster_", "state_")) or c in keep)]
    feats = sorted(list(dict.fromkeys(feats)))
    if not feats:
        raise ValueError("No se encontraron features válidos. Revisa las columnas del parquet meta.")
    return feats

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

def _as_datetime(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s.dt.tz_localize(None)
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _sorted_unique(a: Iterable) -> List:
    return sorted(list(dict.fromkeys(a)))


def _rmspe(y: np.ndarray, yhat: np.ndarray) -> float:
    e = y - yhat
    denom = max(1.0, float(np.sqrt(np.mean(y**2))))
    return float(np.sqrt(np.mean(e**2)) / denom)


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def _safe_pct(a: float, b: float) -> float:
    if b is None or not np.isfinite(b) or abs(b) < 1e-8:
        return np.nan
    return float(a / b)


def _finite_or_big(x: float, big: float = 1e6) -> float:
    return float(x) if np.isfinite(x) else float(big)


def _is_count_target(y: np.ndarray) -> bool:
    """Heurístico: target no-negativo y cuasi-entero -> usar pérdida Poisson."""
    if y.size == 0:
        return False
    if np.nanmin(y) < -1e-12:
        return False
    frac_int = np.nanmean(np.abs(y - np.round(y)) <= 0.05)
    return bool(frac_int >= 0.9)

# ---------------------------------------------------------------------
# Modelos base (regresores/clasificadores) con fallbacks
# ---------------------------------------------------------------------

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline


def make_regressor(name: str = "lgbm", random_state: int = 42, **kwargs):
    """
    Crea un regresor robusto por defecto. Opciones:
      - 'lgbm' (LightGBM) [default]
      - 'hgbt' (HistGradientBoostingRegressor)
      - 'rf'   (RandomForestRegressor)
      - 'ridge' (Ridge con estandarización)
    Acepta kwargs: objective (LGBM), loss (HGBT), reg_lambda (LGBM), etc.
    """
    name = (name or "lgbm").lower()

    if name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        alpha = float(kwargs.get("alpha", 1.0))
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=alpha, random_state=random_state))
        ])

    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=int(kwargs.get("n_estimators", 300)),
            min_samples_leaf=int(kwargs.get("min_samples_leaf", 5)),
            random_state=random_state, n_jobs=-1
        )

    if name == "lgbm":
        # Intento con LightGBM y fallback a HGBT si no está instalado
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                num_leaves=int(kwargs.get("num_leaves", 127)),
                max_depth=int(kwargs.get("max_depth", 8)),
                learning_rate=float(kwargs.get("learning_rate", 0.05)),
                n_estimators=int(kwargs.get("max_iter", 600)),
                min_child_samples=int(kwargs.get("min_child_samples", kwargs.get("min_samples_leaf", 10))),
                feature_fraction=float(kwargs.get("feature_fraction", 0.8)),
                subsample=float(kwargs.get("subsample", 0.8)),
                reg_lambda=float(kwargs.get("reg_lambda", kwargs.get("l2", 0.0))),
                objective=kwargs.get("objective", None),
                random_state=random_state,
                n_jobs=-1
            )
        except Exception as e:
            logging.warning(f"LightGBM no disponible ({e}); usando HistGradientBoostingRegressor como fallback.")
            name = "hgbt"  # cae al bloque siguiente

    # Por defecto: HGBT
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_depth=int(kwargs.get("max_depth", 8)),
            learning_rate=float(kwargs.get("learning_rate", 0.05)),
            max_iter=int(kwargs.get("max_iter", 800)),
            min_samples_leaf=int(kwargs.get("min_samples_leaf", 10)),
            l2_regularization=float(kwargs.get("l2", kwargs.get("reg_lambda", 0.0))),
            loss=str(kwargs.get("loss", "squared_error")),
            random_state=random_state
        )
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=int(kwargs.get("n_estimators", 300)),
            min_samples_leaf=int(kwargs.get("min_samples_leaf", 5)),
            random_state=random_state, n_jobs=-1
        )


def make_classifier(name: str = "logit", random_state: int = 42):
    """
    Clasificador de propensión:
      - 'hgbc' (HistGradientBoostingClassifier)
      - fallback: LogisticRegression (liblinear/saga)
    """
    if name == "hgbc":
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(random_state=random_state)
        except Exception:
            pass
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=200, solver="liblinear", random_state=random_state)


# Modelo constante (fallback si no hay suficientes observaciones)
class _ConstantRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, c: float = 0.0):
        self.c = float(c)
    def fit(self, X, y):
        self.c = float(np.nanmean(y)) if len(y) else 0.0
        return self
    def predict(self, X):
        return np.full(shape=(X.shape[0],), fill_value=self.c, dtype=float)


def _fit_with_weights(model, X, y, sample_weight: Optional[np.ndarray] = None):
    """Intenta pasar sample_weight de forma genérica (incluye Pipeline/Ridge)."""
    if sample_weight is None:
        return model.fit(X, y)
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        # Pipeline: intentar con último step
        if isinstance(model, Pipeline):
            last_name = model.steps[-1][0]
            return model.fit(X, y, **{f"{last_name}__sample_weight": sample_weight})
        # si no, entrenar sin peso
        return model.fit(X, y)

# ---------------------------------------------------------------------
# Split temporal (forward chaining con H días de holdout + embargo)
# ---------------------------------------------------------------------

def make_time_folds(dates: pd.Series, holdout_days: int, folds: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Devuelve lista de (valid_start, valid_end) para 'folds' ventanas consecutivas
    al final del horizonte temporal. Entrenamiento para el fold k: fechas < valid_start_k.
    """
    d_unique = _sorted_unique(_as_datetime(dates))
    if not d_unique:
        return []
    H = int(max(1, holdout_days))
    total = H * int(max(1, folds))
    start_idx = max(0, len(d_unique) - total)
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for k in range(folds):
        s = start_idx + k * H
        e = min(len(d_unique) - 1, s + H - 1)
        windows.append((d_unique[s], d_unique[e]))
    return windows

# ---------------------------------------------------------------------
# Enriquecimiento por fold: unit stats (solo con TRAIN del fold)
# ---------------------------------------------------------------------

def _build_unit_stats(df: pd.DataFrame, train_mask: np.ndarray) -> Tuple[Dict[str, Tuple[float, float, float, float, float, int]], Tuple[float, float, float, float, float, int]]:
    """
    Calcula estadísticas por unidad usando SOLO filas del train del fold:
    (mean, std, slope, last7_mean, last28_mean, n_obs)
    Devuelve (dict uid->tuple, defaults_tuple)
    """
    cols = ["unit_id", "date", "sales"]
    gdf = df.loc[train_mask, cols].copy()
    stats: Dict[str, Tuple[float, float, float, float, float, int]] = {}
    if gdf.empty:
        default_mu = 0.0; default_sd = 0.0; default_slope = 0.0; default_m7 = 0.0; default_m28 = 0.0; default_n = 1
        return {}, (default_mu, default_sd, default_slope, default_m7, default_m28, default_n)

    default_mu = float(np.nanmean(gdf["sales"]))
    default_sd = float(np.nanstd(gdf["sales"]))
    default_m7 = default_mu
    default_m28 = default_mu

    for uid, sub in gdf.groupby("unit_id", sort=False):
        sub = sub.sort_values("date")
        y = sub["sales"].to_numpy(dtype=float)
        n = int(y.size)
        mu = float(np.nanmean(y))
        sd = float(np.nanstd(y))
        # slope: cov(t,y)/var(t)
        t = (sub["date"] - sub["date"].min()).dt.days.to_numpy(dtype=float)
        var_t = float(np.var(t)) + 1e-8
        slope = float(np.cov(t, y, bias=True)[0, 1] / var_t) if n >= 2 else 0.0
        m7 = float(np.nanmean(y[-min(7, n):])) if n > 0 else mu
        m28 = float(np.nanmean(y[-min(28, n):])) if n > 0 else mu
        stats[str(uid)] = (mu, sd, slope, m7, m28, n)

    default = (default_mu, default_sd, 0.0, default_m7, default_m28, 1)
    return stats, default


def _map_stats_to_rows(unit_ids: pd.Series,
                       stats: Dict[str, Tuple[float, float, float, float, float, int]],
                       default: Tuple[float, float, float, float, float, int]) -> np.ndarray:
    """Convierte el dict de stats en matriz Z (n_rows x 5) para las filas dadas."""
    mu0, sd0, slope0, m70, m280, _ = default
    out = np.empty((unit_ids.shape[0], 5), dtype=float)
    for i, uid in enumerate(unit_ids.astype(str).to_numpy()):
        t = stats.get(uid, (mu0, sd0, slope0, m70, m280, 1))
        out[i, 0] = t[0]; out[i, 1] = t[1]; out[i, 2] = t[2]; out[i, 3] = t[3]; out[i, 4] = t[4]
    return out


def _compute_sample_weights(df: pd.DataFrame, train_mask: np.ndarray, v_start: pd.Timestamp) -> np.ndarray:
    """w_unit * w_time (normalizado a media ~1)."""
    dtr = df.loc[train_mask, ["unit_id", "date"]].copy()
    if dtr.empty:
        return np.ones(np.count_nonzero(train_mask), dtype=float)
    counts = dtr["unit_id"].value_counts()
    w_unit = dtr["unit_id"].map(lambda u: 1.0 / float(max(1, counts.get(u, 1)))).to_numpy(dtype=float)
    # recency respecto al inicio de validación del fold
    rec_days = (pd.Series(v_start, index=dtr.index) - dtr["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
    w_time = np.exp(-_LAMBDA_RECENCY * rec_days)
    w = w_unit * w_time
    m = float(np.mean(w)) if np.isfinite(np.mean(w)) and np.mean(w) > 0 else 1.0
    return w / m


def _build_augmented_matrices(df: pd.DataFrame,
                              feats: List[str],
                              train_mask: np.ndarray,
                              valid_mask: np.ndarray,
                              v_start: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Retorna:
      X_tr_aug, X_va_aug, y_tr, y_va, sample_weight_tr, is_count
    """
    X = df[feats].values.astype(float)
    X[np.isnan(X)] = 0.0
    y = df["sales"].to_numpy(dtype=float)
    dates = _as_datetime(df["date"])
    uids = df["unit_id"]

    # Embargo temporal: excluir gap en TRAIN
    train_mask = train_mask & (dates < (v_start - pd.Timedelta(days=_CV_GAP_DAYS)))

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[valid_mask], y[valid_mask]

    stats, default = _build_unit_stats(df, train_mask)
    Z_tr = _map_stats_to_rows(uids[train_mask], stats, default)
    Z_va = _map_stats_to_rows(uids[valid_mask], stats, default)

    X_tr_aug = np.column_stack([X_tr, Z_tr])
    X_va_aug = np.column_stack([X_va, Z_va])

    sample_weight_tr = _compute_sample_weights(df, train_mask, v_start)
    is_count = _is_count_target(y_tr)

    return X_tr_aug, X_va_aug, y_tr, y_va, sample_weight_tr, is_count

# ---------------------------------------------------------------------
# HPO con Optuna (global, antes del cross-fitting) usando CV purgada
# ---------------------------------------------------------------------

def _time_cv_score(meta_df: pd.DataFrame,
                   feats: List[str],
                   cfg_cv_folds: int,
                   cfg_holdout_days: int,
                   build_model_fn,
                   **model_kwargs) -> float:
    """
    Score promedio RMSPE en validación usando ventanas temporales con embargo (gap).
    """
    dates = _as_datetime(meta_df["date"])
    windows = make_time_folds(dates, cfg_holdout_days, cfg_cv_folds)
    if not windows:
        return np.inf

    scores: List[float] = []
    for (v_start, v_end) in windows:
        valid_mask = (dates >= v_start) & (dates <= v_end)
        train_mask = (dates < v_start)

        X_tr_aug, X_va_aug, y_tr, y_va, sw, is_count = _build_augmented_matrices(meta_df, feats, train_mask, valid_mask, v_start)

        if X_tr_aug.shape[0] < max(_MIN_TRAIN_FOLD, 10) or X_va_aug.shape[0] < 1:
            continue

        # Objetivo según la naturaleza del target
        model_params = dict(model_kwargs)
        if "objective" not in model_params and "loss" not in model_params:
            if build_model_fn.__name__.endswith("make_regressor"):
                if is_count:
                    model_params.update({"objective": "poisson", "loss": "poisson"})
        model = build_model_fn(**model_params)

        try:
            _fit_with_weights(model, X_tr_aug, y_tr, sw)
            y_hat = model.predict(X_va_aug)
            scores.append(_rmspe(y_va, y_hat))
        except Exception as e:
            logging.debug(f"HPO fold falló: {e}")
            scores.append(1e3)  # penaliza

    return float(np.mean(scores)) if scores else np.inf

def tune_hyperparams(meta_df: pd.DataFrame, feats: List[str], cfg_model: str,
                     random_state: int, cv_folds: int, cv_holdout_days: int,
                     base_defaults: Dict) -> Dict:
    """
    Optimiza hiperparámetros con Optuna (si está disponible).
    Retorna diccionario con los mejores hiperparámetros.
    """
    try:
        import optuna  # type: ignore
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    except Exception as e:
        logging.warning(f"Optuna no disponible ({e}); se omite HPO y se usan hiperparámetros por defecto.")
        return {}

    model_name = (cfg_model or "lgbm").lower()

    def objective(trial):
        if model_name == "lgbm":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 63, 255),
                "max_depth": trial.suggest_int("max_depth", 6, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                # heredados
                "max_iter": int(base_defaults.get("max_iter", 600)),
            }
        elif model_name == "hgbt":
            params = {
                "max_depth": trial.suggest_int("max_depth", 6, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "max_iter": trial.suggest_int("max_iter", 400, 2000, step=50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
                "l2": trial.suggest_float("l2", 1e-6, 10.0, log=True),
            }
        else:
            return np.inf

        score = _time_cv_score(
            meta_df, feats,
            cv_folds, cv_holdout_days,
            lambda **kw: make_regressor(model_name, random_state, **kw),
            **params
        )
        return _finite_or_big(score)

    study = optuna.create_study(direction="minimize", study_name=f"hpo_{model_name}")
    study.optimize(objective, n_trials=int(base_defaults.get("hpo_trials", 300)), show_progress_bar=False)

    best = dict(study.best_params)
    logging.info(f"[HPO] Modelo={model_name} | best_score(RMSPE)={study.best_value:.6f}")
    logging.info(f"[HPO] Mejores hiperparámetros: {best}")

    if model_name == "lgbm":
        keys = ["num_leaves", "max_depth", "learning_rate", "min_child_samples", "feature_fraction", "subsample", "reg_lambda", "max_iter"]
    elif model_name == "hgbt":
        keys = ["max_depth", "learning_rate", "max_iter", "min_samples_leaf", "l2"]
    else:
        keys = []

    return {k: best[k] for k in keys if k in best}

# ---------------------------------------------------------------------
# Núcleo de aprendizaje: T-, S-, X- learners con cross-fitting temporal
# ---------------------------------------------------------------------

@dataclass
class TrainCfg:
    learner: str = "x"                # 't' | 's' | 'x'
    model: str = "lgbm"               # base regressor (default LGBM)
    prop_model: str = "logit"         # clasificador p(X) para X-learner
    random_state: int = 42
    cv_folds: int = 3
    cv_holdout_days: int = 21
    min_train_samples: int = 50       # mínimo nominal; se usa max(_MIN_TRAIN_FOLD, min_train_samples//2)
    # hiperparámetros genéricos / HGBT
    max_depth: int = 8
    learning_rate: float = 0.05
    max_iter: int = 600
    min_samples_leaf: int = 10
    l2: float = 0.0
    # LightGBM (mapeados; si no se setean, se infieren de los genéricos cuando procede)
    num_leaves: int = 127
    min_child_samples: int = 10
    feature_fraction: float = 0.8
    subsample: float = 0.8
    # sensibilidad
    sens_samples: int = 0
    # --- Tratamientos configurables ---
    treat_col_s: str = "D"            # S-learner: por defecto usa D
    s_ref: float = 0.0                # baseline de referencia para S-learner
    treat_col_b: str = "D"            # T/X: por defecto usa D
    bin_threshold: float = 0.0        # umbral para binarizar treat_col_b si no es 0/1
    cover_all_time: bool = True  # NUEVO


def _build_X(df: pd.DataFrame, feats: List[str]) -> np.ndarray:
    X = df[feats].values.astype(float)
    X[np.isnan(X)] = 0.0
    return X

def make_time_folds(dates: pd.Series, holdout_days: int, folds: int, cover_all: bool = False) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    d_unique = _sorted_unique(_as_datetime(dates))
    if not d_unique: return []
    H = int(max(1, holdout_days))
    if not cover_all:
        total = H * int(max(1, folds))
        start_idx = max(0, len(d_unique) - total)
        return [(d_unique[start_idx + k*H], d_unique[min(len(d_unique)-1, start_idx + (k+1)*H - 1)])
                for k in range(int(max(1, folds)))]
    windows = []
    for s in range(0, max(1, len(d_unique) - H + 1), H):
        e = min(len(d_unique) - 1, s + H - 1)
        windows.append((d_unique[s], d_unique[e]))
    return windows

def crossfit_predictions(df: pd.DataFrame, feats: List[str], cfg: TrainCfg) -> Dict[str, np.ndarray]:
    """
    Entrena modelos por ventanas temporales (CV purgada con gap) y devuelve
    predicciones cross-fitted:
      mu0_cf, mu1_cf, tau_cf
    y refits completos (para uso operativo, no para métricas):
      mu0_full, mu1_full, tau_full
    """
    dates = _as_datetime(df["date"])
    y = df["sales"].to_numpy(dtype=float)

    # Tratamiento binario (T/X)
    if cfg.treat_col_b in df.columns:
        Db_raw = pd.to_numeric(df[cfg.treat_col_b], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        Db_raw = df["D"].to_numpy(dtype=float)
    D_bin = (Db_raw > cfg.bin_threshold).astype(int)

    # Tratamiento continuo (S)
    if cfg.treat_col_s in df.columns:
        D_cont = pd.to_numeric(df[cfg.treat_col_s], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        D_cont = df["D"].to_numpy(dtype=float)

    mu0_cf = np.full_like(y, np.nan, dtype=float)
    mu1_cf = np.full_like(y, np.nan, dtype=float)
    tau_cf = np.full_like(y, np.nan, dtype=float)

    windows = make_time_folds(dates, cfg.cv_holdout_days, cfg.cv_folds, cover_all=cfg.cover_all_time)
    if not windows:
        logging.warning("No se pudieron construir ventanas de CV temporal; se harán refits completos sin cross-fitting.")
        windows = []

    for (v_start, v_end) in windows:
        valid_mask = (dates >= v_start) & (dates <= v_end)
        train_mask = (dates < v_start)

        X_tr_aug, X_va_aug, y_tr, y_va, sw, is_count = _build_augmented_matrices(df, feats, train_mask, valid_mask, v_start)

        # --- T-learner ---
        if cfg.learner.lower() == "t":
            D_tr = D_bin[train_mask & (dates < (v_start - pd.Timedelta(days=_CV_GAP_DAYS)))]
            mask0 = (D_tr == 0); mask1 = (D_tr == 1)

            min_needed = max(_MIN_TRAIN_FOLD, cfg.min_train_samples // 2)

            # m0: E[Y|X, D=0]
            if mask0.sum() >= min_needed:
                m0 = make_regressor(cfg.model, cfg.random_state,
                                    max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                    max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                    num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                    feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                    reg_lambda=cfg.l2,
                                    objective="poisson" if is_count else None, loss="poisson" if is_count else "squared_error")
                _fit_with_weights(m0, X_tr_aug[mask0], y_tr[mask0], sw[mask0])
            else:
                m0 = _ConstantRegressor(np.nanmean(y_tr[mask0]) if mask0.sum() else np.nanmean(y_tr)).fit(X_tr_aug, y_tr)

            # m1: E[Y|X, D=1]
            if mask1.sum() >= min_needed:
                m1 = make_regressor(cfg.model, cfg.random_state,
                                    max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                    max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                    num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                    feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                    reg_lambda=cfg.l2,
                                    objective="poisson" if is_count else None, loss="poisson" if is_count else "squared_error")
                _fit_with_weights(m1, X_tr_aug[mask1], y_tr[mask1], sw[mask1])
            else:
                m1 = _ConstantRegressor(np.nanmean(y_tr[mask1]) if mask1.sum() else np.nanmean(y_tr)).fit(X_tr_aug, y_tr)

            mu0_cf[valid_mask] = m0.predict(X_va_aug)
            mu1_cf[valid_mask] = m1.predict(X_va_aug)
            tau_cf[valid_mask] = mu1_cf[valid_mask] - mu0_cf[valid_mask]

        # --- S-learner: f(X_aug, D_cont) ---
        elif cfg.learner.lower() == "s":
            d_tr = D_cont[train_mask & (dates < (v_start - pd.Timedelta(days=_CV_GAP_DAYS)))]
            d_va = D_cont[valid_mask]
            model = make_regressor(cfg.model, cfg.random_state,
                                   max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                   max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                   num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                   feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                   reg_lambda=cfg.l2,
                                   objective="poisson" if is_count else None, loss="poisson" if is_count else "squared_error")
            X_tr_s = np.column_stack([X_tr_aug, d_tr.reshape(-1, 1)])
            _fit_with_weights(model, X_tr_s, y_tr, sw)

            X_va_ref = np.column_stack([X_va_aug, np.full((X_va_aug.shape[0], 1), cfg.s_ref)])
            X_va_obs = np.column_stack([X_va_aug, d_va.reshape(-1, 1)])

            mu0_cf[valid_mask] = model.predict(X_va_ref)
            mu1_cf[valid_mask] = model.predict(X_va_obs)
            tau_cf[valid_mask] = mu1_cf[valid_mask] - mu0_cf[valid_mask]

        # --- X-learner (binario) ---
        else:
            D_tr = D_bin[train_mask & (dates < (v_start - pd.Timedelta(days=_CV_GAP_DAYS)))]
            mask0 = (D_tr == 0); mask1 = (D_tr == 1)
            min_needed = max(_MIN_TRAIN_FOLD, cfg.min_train_samples // 2)

            m0 = make_regressor(cfg.model, cfg.random_state,
                                max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                reg_lambda=cfg.l2,
                                objective="poisson" if is_count else None, loss="poisson" if is_count else "squared_error")
            m1 = make_regressor(cfg.model, cfg.random_state,
                                max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                reg_lambda=cfg.l2,
                                objective="poisson" if is_count else None, loss="poisson" if is_count else "squared_error")

            if mask0.sum() >= min_needed:
                _fit_with_weights(m0, X_tr_aug[mask0], y_tr[mask0], sw[mask0])
            else:
                m0 = _ConstantRegressor(np.nanmean(y_tr[mask0]) if mask0.sum() else np.nanmean(y_tr)).fit(X_tr_aug, y_tr)

            if mask1.sum() >= min_needed:
                _fit_with_weights(m1, X_tr_aug[mask1], y_tr[mask1], sw[mask1])
            else:
                m1 = _ConstantRegressor(np.nanmean(y_tr[mask1]) if mask1.sum() else np.nanmean(y_tr)).fit(X_tr_aug, y_tr)

            mu0_on_t = m0.predict(X_tr_aug[mask1])  # para tratados
            mu1_on_c = m1.predict(X_tr_aug[mask0])  # para controles
            d1 = y_tr[mask1] - mu0_on_t
            d0 = mu1_on_c - y_tr[mask0]

            g1 = make_regressor(cfg.model, cfg.random_state,
                                max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                reg_lambda=cfg.l2,
                                objective=None, loss="squared_error")
            g0 = make_regressor(cfg.model, cfg.random_state,
                                max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                reg_lambda=cfg.l2,
                                objective=None, loss="squared_error")

            if mask1.sum() >= min_needed:
                _fit_with_weights(g1, X_tr_aug[mask1], d1, sw[mask1])
            else:
                g1 = _ConstantRegressor(np.nanmean(d1) if len(d1) else 0.0).fit(X_tr_aug, y_tr)

            if mask0.sum() >= min_needed:
                _fit_with_weights(g0, X_tr_aug[mask0], d0, sw[mask0])
            else:
                g0 = _ConstantRegressor(np.nanmean(d0) if len(d0) else 0.0).fit(X_tr_aug, y_tr)

            # Propensión
            clf = make_classifier(cfg.prop_model, cfg.random_state)
            try:
                clf.fit(X_tr_aug, D_tr)
                p_va = np.clip(getattr(clf, "predict_proba")(X_va_aug)[:, 1], 1e-3, 1 - 1e-3)
            except Exception:
                p_va = np.full(X_va_aug.shape[0], fill_value=float(max(1e-3, min(0.999, np.mean(D_tr)))))

            tau0 = g0.predict(X_va_aug)
            tau1 = g1.predict(X_va_aug)
            tau_cf[valid_mask] = p_va * tau0 + (1.0 - p_va) * tau1

            mu0_cf[valid_mask] = m0.predict(X_va_aug)
            mu1_cf[valid_mask] = m1.predict(X_va_aug)

    # -------- Refit completo (operativo) --------
    # Construcción de features enriquecidas a nivel global (sin gap)
    X_full = _build_X(df, feats)
    X_full[np.isnan(X_full)] = 0.0
    y_full = y
    dates_full = dates
    uids_full = df["unit_id"]

    # Stats globales (sobre todo el set)
    stats_all, default_all = _build_unit_stats(df, np.ones(len(df), dtype=bool))
    Z_all = _map_stats_to_rows(uids_full, stats_all, default_all)
    X_all_aug = np.column_stack([X_full, Z_all])
    is_count_full = _is_count_target(y_full)

    # Pesos “globales” (recencia hacia el final del panel)
    vstart_full = dates_full.max() + pd.Timedelta(days=1)
    sw_full = _compute_sample_weights(df, np.ones(len(df), dtype=bool), vstart_full)

    if cfg.learner.lower() == "t":
        D_all = D_bin
        m0_full = make_regressor(cfg.model, cfg.random_state,
                                 max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                 max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                 num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                 feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                 reg_lambda=cfg.l2,
                                 objective="poisson" if is_count_full else None, loss="poisson" if is_count_full else "squared_error")
        m1_full = make_regressor(cfg.model, cfg.random_state,
                                 max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                 max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                 num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                 feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                 reg_lambda=cfg.l2,
                                 objective="poisson" if is_count_full else None, loss="poisson" if is_count_full else "squared_error")
        mask0_all, mask1_all = (D_all == 0), (D_all == 1)
        if mask0_all.sum() >= _MIN_TRAIN_FOLD:
            _fit_with_weights(m0_full, X_all_aug[mask0_all], y_full[mask0_all], sw_full[mask0_all])
        else:
            m0_full = _ConstantRegressor(np.nanmean(y_full[mask0_all]) if mask0_all.sum() else np.nanmean(y_full)).fit(X_all_aug, y_full)
        if mask1_all.sum() >= _MIN_TRAIN_FOLD:
            _fit_with_weights(m1_full, X_all_aug[mask1_all], y_full[mask1_all], sw_full[mask1_all])
        else:
            m1_full = _ConstantRegressor(np.nanmean(y_full[mask1_all]) if mask1_all.sum() else np.nanmean(y_full)).fit(X_all_aug, y_full)

        mu0_full = m0_full.predict(X_all_aug)
        mu1_full = m1_full.predict(X_all_aug)
        tau_full = mu1_full - mu0_full

    elif cfg.learner.lower() == "s":
        D_all = D_cont
        model_full = make_regressor(cfg.model, cfg.random_state,
                                    max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                    max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                    num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                    feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                    reg_lambda=cfg.l2,
                                    objective="poisson" if is_count_full else None, loss="poisson" if is_count_full else "squared_error")
        XD = np.column_stack([X_all_aug, D_all.reshape(-1, 1)])
        _fit_with_weights(model_full, XD, y_full, sw_full)

        X_ref = np.column_stack([X_all_aug, np.full((X_all_aug.shape[0], 1), cfg.s_ref)])
        X_obs = np.column_stack([X_all_aug, D_all.reshape(-1, 1)])
        mu0_full = model_full.predict(X_ref)
        mu1_full = model_full.predict(X_obs)
        tau_full = mu1_full - mu0_full

    else:
        D_all = D_bin
        m0_full = make_regressor(cfg.model, cfg.random_state,
                                 max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                 max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                 num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                 feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                 reg_lambda=cfg.l2,
                                 objective="poisson" if is_count_full else None, loss="poisson" if is_count_full else "squared_error")
        m1_full = make_regressor(cfg.model, cfg.random_state,
                                 max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                 max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                 num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                 feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                 reg_lambda=cfg.l2,
                                 objective="poisson" if is_count_full else None, loss="poisson" if is_count_full else "squared_error")
        mask0_all, mask1_all = (D_all == 0), (D_all == 1)
        if mask0_all.sum() >= _MIN_TRAIN_FOLD:
            _fit_with_weights(m0_full, X_all_aug[mask0_all], y_full[mask0_all], sw_full[mask0_all])
        else:
            m0_full = _ConstantRegressor(np.nanmean(y_full[mask0_all]) if mask0_all.sum() else np.nanmean(y_full)).fit(X_all_aug, y_full)
        if mask1_all.sum() >= _MIN_TRAIN_FOLD:
            _fit_with_weights(m1_full, X_all_aug[mask1_all], y_full[mask1_all], sw_full[mask1_all])
        else:
            m1_full = _ConstantRegressor(np.nanmean(y_full[mask1_all]) if mask1_all.sum() else np.nanmean(y_full)).fit(X_all_aug, y_full)

        mu0_full = m0_full.predict(X_all_aug)
        mu1_full = m1_full.predict(X_all_aug)

        # pseudo-outcomes
        d1_all = y_full[mask1_all] - mu0_full[mask1_all]
        d0_all = mu1_full[mask0_all] - y_full[mask0_all]

        g1_full = make_regressor(cfg.model, cfg.random_state); g0_full = make_regressor(cfg.model, cfg.random_state)
        if mask1_all.sum() >= _MIN_TRAIN_FOLD:
            _fit_with_weights(g1_full, X_all_aug[mask1_all], d1_all, sw_full[mask1_all])
        else:
            g1_full = _ConstantRegressor(np.nanmean(d1_all) if len(d1_all) else 0.0).fit(X_all_aug, y_full)
        if mask0_all.sum() >= _MIN_TRAIN_FOLD:
            _fit_with_weights(g0_full, X_all_aug[mask0_all], d0_all, sw_full[mask0_all])
        else:
            g0_full = _ConstantRegressor(np.nanmean(d0_all) if len(d0_all) else 0.0).fit(X_all_aug, y_full)

        clf_full = make_classifier("logit", cfg.random_state)
        try:
            clf_full.fit(X_all_aug, D_all)
            p_all = np.clip(getattr(clf_full, "predict_proba")(X_all_aug)[:, 1], 1e-3, 1 - 1e-3)
        except Exception:
            p_all = np.full(X_all_aug.shape[0], fill_value=float(max(1e-3, min(0.999, np.mean(D_all)))))

        tau_full = p_all * g0_full.predict(X_all_aug) + (1.0 - p_all) * g1_full.predict(X_all_aug)

    # Completar huecos CF con refit si falta
    if np.any(~np.isfinite(mu0_cf)):
        idx = ~np.isfinite(mu0_cf)
        mu0_cf[idx] = mu0_full[idx]
    if np.any(~np.isfinite(mu1_cf)):
        idx = ~np.isfinite(mu1_cf)
        mu1_cf[idx] = mu1_full[idx]
    if np.any(~np.isfinite(tau_cf)):
        idx = ~np.isfinite(tau_cf)
        tau_cf[idx] = tau_full[idx]

    return {"mu0_cf": mu0_cf, "mu1_cf": mu1_cf, "tau_cf": tau_cf,
            "mu0_full": mu0_full, "mu1_full": mu1_full, "tau_full": tau_full}

# ---------------------------------------------------------------------
# Placebos, LOO y sensibilidad
# ---------------------------------------------------------------------

def placebo_space_meta(ep_rows: pd.DataFrame,
                       donors_map: pd.DataFrame,
                       victim: Tuple[int, int],
                       treat_start: pd.Timestamp,
                       post_end: pd.Timestamp) -> pd.DataFrame:
    """
    Calcula ATT placebo para donantes de la víctima en la ventana post,
    usando la columna 'tau_hat' (método-agnóstica: T/S/X).
    """
    j_store, j_item = victim
    dons = donors_map.loc[(donors_map["j_store"] == j_store) & (donors_map["j_item"] == j_item)]
    if dons.empty:
        return pd.DataFrame(columns=["unit_id", "att_placebo_mean", "att_placebo_sum"])

    donor_uids = set(dons.assign(uid=dons["donor_store"].astype(str) + ":" + dons["donor_item"].astype(str))["uid"].tolist())
    mask = ep_rows["unit_id"].isin(donor_uids) & ep_rows["date"].between(treat_start, post_end, inclusive="both")
    g = ep_rows.loc[mask].copy()
    if g.empty:
        return pd.DataFrame(columns=["unit_id", "att_placebo_mean", "att_placebo_sum"])

    out = (g.groupby("unit_id", as_index=False)["tau_hat"]
             .agg(att_placebo_sum=lambda x: float(np.nansum(x)),
                  att_placebo_mean=lambda x: float(np.nanmean(x))))
    return out


def placebo_time_victim(ep_rows: pd.DataFrame,
                        treat_start: pd.Timestamp,
                        post_end: pd.Timestamp,
                        victim_uid: str) -> pd.DataFrame:
    """
    Placebo in-time: usa la última ventana del pre con longitud igual al post.
    Efecto estimado con tau_hat (debería ~ 0).
    """
    post_len = int((post_end - treat_start).days + 1)
    v = ep_rows.loc[ep_rows["unit_id"] == victim_uid].sort_values("date")
    pre = v.loc[v["date"] < treat_start]
    if pre.shape[0] < max(7, post_len):
        return pd.DataFrame([{"used": False, "att_placebo_sum": np.nan, "att_placebo_mean": np.nan, "H": 0}])

    win = pre.tail(post_len)
    return pd.DataFrame([{
        "used": True,
        "H": int(post_len),
        "att_placebo_sum": float(np.nansum(win["tau_hat"])),
        "att_placebo_mean": float(np.nanmean(win["tau_hat"]))
    }])


def loo_retrain_if_requested(meta_df: pd.DataFrame,
                             feats: List[str],
                             cfg: TrainCfg,
                             donors_map: pd.DataFrame,
                             episode_filter: pd.DataFrame,
                             victim: Tuple[int, int],
                             max_loo: int = 5) -> pd.DataFrame:
    """
    LOO opcional: reentrena quitando cada donante del *entrenamiento global*.
    Devuelve variación de ATT_sum post para la víctima.
    """
    j_store, j_item = victim
    dons = donors_map.loc[(donors_map["j_store"] == j_store) & (donors_map["j_item"] == j_item)].copy()
    dons = dons.head(int(max_loo))
    if dons.empty:
        return pd.DataFrame(columns=["excluded_unit", "att_sum"])

    victim_uid = f"{j_store}:{j_item}"
    ep_dates = episode_filter[["pre_start", "treat_start", "post_end"]].iloc[0]
    vmask_post = (meta_df["unit_id"] == victim_uid) & meta_df["date"].between(ep_dates["treat_start"], ep_dates["post_end"], inclusive="both")

    rows = []

    # Tratamientos globales
    if cfg.treat_col_b in meta_df.columns:
        Db_raw_all = pd.to_numeric(meta_df[cfg.treat_col_b], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        Db_raw_all = meta_df["D"].to_numpy(dtype=float)
    D_bin_all = (Db_raw_all > cfg.bin_threshold).astype(int)

    if cfg.treat_col_s in meta_df.columns:
        D_cont_all = pd.to_numeric(meta_df[cfg.treat_col_s], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        D_cont_all = meta_df["D"].to_numpy(dtype=float)

    X_all = _build_X(meta_df, feats)
    dates_all = _as_datetime(meta_df["date"])
    y_all = meta_df["sales"].to_numpy(dtype=float)

    for _, d in dons.iterrows():
        excl_uid = f"{int(d['donor_store'])}:{int(d['donor_item'])}"
        train_df = meta_df.loc[meta_df["unit_id"] != excl_uid].copy()

        X_train = _build_X(train_df, feats)
        y_train = train_df["sales"].to_numpy(dtype=float)
        dates_train = _as_datetime(train_df["date"])
        uids_train = train_df["unit_id"]

        # Stats y features aumentadas (globales)
        stats, default = _build_unit_stats(train_df, np.ones(len(train_df), dtype=bool))
        Z_tr = _map_stats_to_rows(uids_train, stats, default)
        X_train_aug = np.column_stack([X_train, Z_tr])
        Z_all = _map_stats_to_rows(meta_df["unit_id"], stats, default)
        X_all_aug = np.column_stack([X_all, Z_all])

        # Tratamientos en train_df
        if cfg.treat_col_b in train_df.columns:
            Db_raw_tr = pd.to_numeric(train_df[cfg.treat_col_b], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            Db_raw_tr = train_df["D"].to_numpy(dtype=float)
        D_bin_tr = (Db_raw_tr > cfg.bin_threshold).astype(int)

        if cfg.treat_col_s in train_df.columns:
            D_cont_tr = pd.to_numeric(train_df[cfg.treat_col_s], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            D_cont_tr = train_df["D"].to_numpy(dtype=float)

        if cfg.learner.lower() == "t":
            mask0, mask1 = (D_bin_tr == 0), (D_bin_tr == 1)
            m0 = make_regressor(cfg.model, cfg.random_state); m1 = make_regressor(cfg.model, cfg.random_state)
            if mask0.sum() >= _MIN_TRAIN_FOLD: _fit_with_weights(m0, X_train_aug[mask0], y_train[mask0], None)
            else: m0 = _ConstantRegressor(np.nanmean(y_train[mask0]) if mask0.sum() else np.nanmean(y_train)).fit(X_train_aug, y_train)
            if mask1.sum() >= _MIN_TRAIN_FOLD: _fit_with_weights(m1, X_train_aug[mask1], y_train[mask1], None)
            else: m1 = _ConstantRegressor(np.nanmean(y_train[mask1]) if mask1.sum() else np.nanmean(y_train)).fit(X_train_aug, y_train)
            mu0 = m0.predict(X_all_aug); mu1 = m1.predict(X_all_aug); tau_all = mu1 - mu0

        elif cfg.learner.lower() == "s":
            model = make_regressor(cfg.model, cfg.random_state)
            XD = np.column_stack([X_train_aug, D_cont_tr.reshape(-1, 1)])
            model.fit(XD, y_train)
            X0 = np.column_stack([X_all_aug, np.full((X_all_aug.shape[0], 1), cfg.s_ref)])
            X1 = np.column_stack([X_all_aug, D_cont_all.reshape(-1, 1)])
            tau_all = model.predict(X1) - model.predict(X0)

        else:
            mask0, mask1 = (D_bin_tr == 0), (D_bin_tr == 1)
            m0 = make_regressor(cfg.model, cfg.random_state); m1 = make_regressor(cfg.model, cfg.random_state)
            if mask0.sum() >= _MIN_TRAIN_FOLD: m0.fit(X_train_aug[mask0], y_train[mask0])
            else: m0 = _ConstantRegressor(np.nanmean(y_train[mask0]) if mask0.sum() else np.nanmean(y_train)).fit(X_train_aug, y_train)
            if mask1.sum() >= _MIN_TRAIN_FOLD: m1.fit(X_train_aug[mask1], y_train[mask1])
            else: m1 = _ConstantRegressor(np.nanmean(y_train[mask1]) if mask1.sum() else np.nanmean(y_train)).fit(X_train_aug, y_train)

            mu0_tr = m0.predict(X_train_aug)
            mu1_tr = m1.predict(X_train_aug)
            d1 = y_train[mask1] - mu0_tr[mask1]
            d0 = mu1_tr[mask0] - y_train[mask0]

            g1 = make_regressor(cfg.model, cfg.random_state); g0 = make_regressor(cfg.model, cfg.random_state)
            if mask1.sum() >= _MIN_TRAIN_FOLD: g1.fit(X_train_aug[mask1], d1)
            else: g1 = _ConstantRegressor(np.nanmean(d1) if len(d1) else 0.0).fit(X_train_aug, y_train)
            if mask0.sum() >= _MIN_TRAIN_FOLD: g0.fit(X_train_aug[mask0], d0)
            else: g0 = _ConstantRegressor(np.nanmean(d0) if len(d0) else 0.0).fit(X_train_aug, y_train)

            from sklearn.linear_model import LogisticRegression
            clf = make_classifier("logit", cfg.random_state)
            try:
                clf.fit(X_train_aug, D_bin_tr)
                p_all = np.clip(getattr(clf, "predict_proba")(X_all_aug)[:, 1], 1e-3, 1 - 1e-3)
            except Exception:
                p_all = np.full(X_all_aug.shape[0], fill_value=float(max(1e-3, min(0.999, np.mean(D_bin_tr)))))
            tau_all = p_all * g0.predict(X_all_aug) + (1.0 - p_all) * g1.predict(X_all_aug)

        att_sum = float(np.nansum(tau_all[vmask_post]))
        rows.append({"excluded_unit": excl_uid, "att_sum": att_sum})

    return pd.DataFrame(rows)

def sensitivity_sweep(meta_df: pd.DataFrame,
                      feats: List[str],
                      cfg: TrainCfg,
                      victim_uid: str,
                      treat_start: pd.Timestamp,
                      post_end: pd.Timestamp,
                      n_samples: int = 8) -> pd.DataFrame:
    """
    Sensibilidad global simple: barremos pequeñas variaciones en hiperparámetros/modelo
    y registramos la variación de ATT_sum post en la víctima.
    """
    rng = np.random.default_rng(cfg.random_state)
    rows = []
    for _ in range(int(max(0, n_samples))):
        model = rng.choice(["lgbm", "hgbt", "rf", "ridge"])
        max_depth = int(rng.integers(6, 12))
        lr = float(10 ** rng.uniform(-2.0, -0.7))     # ~ [0.01, 0.2]
        l2 = float(10 ** rng.uniform(-3.0, 0.0))      # ~ [1e-3, 1]
        min_leaf = int(rng.integers(10, 30))
        cfg2 = TrainCfg(**{**asdict(cfg), "model": model, "max_depth": max_depth,
                           "learning_rate": lr, "l2": l2, "min_samples_leaf": min_leaf})

        pred = crossfit_predictions(meta_df, feats, cfg2)
        df_pred = meta_df[["unit_id", "date", "sales", "D"]].copy()
        df_pred["tau_hat"] = pred["tau_cf"]
        m = (df_pred["unit_id"] == victim_uid) & df_pred["date"].between(treat_start, post_end, inclusive="both")
        att_sum = float(np.nansum(df_pred.loc[m, "tau_hat"]))
        rows.append({
            "model": model, "max_depth": max_depth, "learning_rate": lr, "l2": l2, "min_samples_leaf": min_leaf,
            "att_sum": att_sum
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Pipeline por episodio y batch
# ---------------------------------------------------------------------

@dataclass
class RunCfg:
    learner: str = "x"  # 't'|'s'|'x'
    # Entrena SIEMPRE con este parquet grande:
    meta_parquet: Path = field(default_factory=_data_root) / "meta/all_units.parquet"
    # Evalúa SOLO sobre los episodios de este índice (pequeños, usados por GSC):
    episodes_index: Path = field(default_factory=_data_root) / "episodes_index.parquet"
    donors_csv: Path = field(default_factory=_data_root) / "donors_per_victim.csv"
    out_dir: Optional[Path] = None
    log_level: str = "INFO"
    max_episodes: Optional[int] = None

    # entrenamiento / CV temporal
    cv_folds: int = 3
    cv_holdout_days: int = 21
    min_train_samples: int = 50

    # modelo base (default 'lgbm')
    model: str = "lgbm"
    prop_model: str = "logit"
    random_state: int = 42
    max_depth: int = 8
    learning_rate: float = 0.05
    max_iter: int = 600
    min_samples_leaf: int = 10
    l2: float = 0.0

    # LightGBM específicos
    num_leaves: int = 127
    min_child_samples: int = 10
    feature_fraction: float = 0.8
    subsample: float = 0.8

    # HPO
    hpo_trials: int = 10  # 0 para desactivar

    # diagnósticos
    do_placebo_space: bool = True
    do_placebo_time: bool = True
    do_loo: bool = False
    max_loo: int = 5
    sens_samples: int = 0   # 0 => desactivado

    # --- Tratamientos configurables ---
    treat_col_s: str = "D"   # S-learner (continuo)
    s_ref: float = 0.0
    treat_col_b: str = "D"   # T/X (binario)
    bin_threshold: float = 0.0
    exp_tag: Optional[str] = None
    fig_root: Path = field(default_factory=_fig_root)


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

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_meta_df(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = _as_datetime(df["date"])
    if "D" in df.columns:
        df["D"] = pd.to_numeric(df["D"], errors="coerce").fillna(0).astype(int)
    if "unit_id" not in df.columns:
        df["unit_id"] = df["store_nbr"].astype(str) + ":" + df["item_nbr"].astype(str)

    # Unicidad por (unit_id,date)
    before = df.shape[0]
    df = df.sort_values(["unit_id", "date"]).drop_duplicates(["unit_id", "date"], keep="last")
    after = df.shape[0]
    if after < before:
        logging.warning("Se detectaron y limpiaron %d duplicados por (unit_id,date).", before - after)

    # NUEVO: rezago seguro de disponibilidad
    if "available_A" in df.columns and "available_A_l1" not in df.columns:
        df["available_A_l1"] = (
            df.groupby("unit_id", observed=False)["available_A"].shift(1).astype("float32")
        )

    return df


def _load_episodes(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    for c in ["pre_start", "treat_start", "post_start", "post_end"]:
        if c in df.columns:
            df[c] = _as_datetime(df[c])
    return df


def _load_donors(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _ols_calibrate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """OLS y_true ≈ a + b*y_pred (con guardas)."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 5:
        return 0.0, 1.0
    X = np.column_stack([np.ones(m.sum()), y_pred[m]])
    try:
        coef, *_ = np.linalg.lstsq(X, y_true[m], rcond=None)
        a, b = float(coef[0]), float(coef[1])
        if not np.isfinite(a) or not np.isfinite(b):
            return 0.0, 1.0
        return a, b
    except Exception:
        return 0.0, 1.0

def run_episode(meta_df: pd.DataFrame,
                feats: List[str],
                ep_row: pd.Series,
                donors_df: pd.DataFrame,
                cfg: RunCfg,
                pred_cache: Optional[Dict[str, np.ndarray]] = None) -> Dict:
    """
    Ejecuta el learner elegido y genera artefactos/métricas por episodio.
    Usa pred_cache si ya existe (para no recalcular cross-fitting global).

    --- FIX CRÍTICO ---
    Filtramos PRIMERO por episode_id y alineamos las predicciones usando el MISMO índice,
    para evitar mezclar filas de otros episodios del mismo unit_id en las series del CF.
    Este bug explicaba por qué la línea azul (observado) parecía "de otro episodio"
    aunque el contrafactual se viera razonable; se observaba en los reportes Meta con
    grandes valores de 'dup_dates_fixed' frente a GSC donde era 0.  :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}
    """

    # --- Identificadores del episodio / víctima ---
    j_store, j_item = int(ep_row["j_store"]), int(ep_row["j_item"])
    victim_uid = f"{j_store}:{j_item}"
    treat_start, post_end = ep_row["treat_start"], ep_row["post_end"]
    ep_id = str(ep_row["episode_id"]) if "episode_id" in ep_row else f"{j_store}-{j_item}_{pd.to_datetime(treat_start).strftime('%Y%m%d')}"

    # --- Config de entrenamiento (incluyendo columnas de tratamiento) ---
    tcfg = TrainCfg(learner=cfg.learner, model=cfg.model, prop_model=cfg.prop_model,
                    random_state=cfg.random_state, cv_folds=cfg.cv_folds,
                    cv_holdout_days=cfg.cv_holdout_days, min_train_samples=cfg.min_train_samples,
                    max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                    max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                    num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                    feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                    sens_samples=cfg.sens_samples,
                    treat_col_s=cfg.treat_col_s, s_ref=cfg.s_ref,
                    treat_col_b=cfg.treat_col_b, bin_threshold=cfg.bin_threshold)

    # --- Predicciones cross-fitted globales (una sola vez por corrida) ---
    if pred_cache is None or not pred_cache:
        pred_cache = crossfit_predictions(meta_df, feats, tcfg)

    # --- NUEVO: recortar el parquet GRANDE AL EPISODIO ACTUAL y alinear predicciones por índice ---
    if "episode_id" not in meta_df.columns:
        logging.warning(f"[{ep_id}] meta_df no tiene 'episode_id'; se caerá a filtro legacy por unit_id+fechas (menos robusto).")
        ep_mask = (meta_df["unit_id"] == victim_uid) & meta_df["date"].between(ep_row["pre_start"], post_end, inclusive="both")
    else:
        ep_mask = meta_df["episode_id"].astype(str).eq(ep_id)

    if not np.any(ep_mask):
        raise ValueError(f"[{ep_id}] No hay filas en meta_df para este episode_id (revisa origen de windows.parquet).")

    idx = meta_df.index[ep_mask]
    dfp = meta_df.loc[idx, ["episode_id", "unit_id", "date", "sales", "D", "treated_unit", "treated_time"]].copy()

    # Inyectar predicciones usando el MISMO índice del subset del episodio
    dfp["mu0_hat"] = pred_cache["mu0_cf"][idx]
    dfp["mu1_hat"] = pred_cache["mu1_cf"][idx]
    dfp["tau_hat"] = pred_cache["tau_cf"][idx]

    # --- Serie de la víctima dentro del episodio/ventana ---
    vmask = (dfp["unit_id"] == victim_uid) & dfp["date"].between(ep_row["pre_start"], post_end, inclusive="both")
    vpre = vmask & (dfp["date"] < treat_start)
    vpost = vmask & (dfp["date"] >= treat_start)

    # Guardia: unicidad por fecha (no debería haber duplicados tras filtrar por episodio)
    dup_count = int(dfp.loc[vmask, ["date"]].duplicated().sum())
    if dup_count > 0:
        logging.warning(f"[{ep_id}] (unit {victim_uid}) fechas duplicadas dentro del episodio: {dup_count}. Se resolverán conservando la última por fecha.")
        # Resolver por fecha
        ord_cols = ["date", "treated_time", "D"]
        keep_cols = [c for c in ["episode_id", "unit_id", "date", "sales", "D", "treated_unit", "treated_time",
                                 "mu0_hat", "mu1_hat", "tau_hat"] if c in dfp.columns]
        dfp = (dfp.sort_values(ord_cols)
                  .groupby(["episode_id", "unit_id", "date"], as_index=False)[keep_cols].last())

        # Recalcular máscaras tras el groupby
        vmask = (dfp["unit_id"] == victim_uid) & dfp["date"].between(ep_row["pre_start"], post_end, inclusive="both")
        vpre = vmask & (dfp["date"] < treat_start)
        vpost = vmask & (dfp["date"] >= treat_start)

    # --- Calibración baseline en PRE (OLS: y_obs ≈ a + b * mu0_hat) ---
    y_obs_pre = dfp.loc[vpre, "sales"].to_numpy(dtype=float)
    y_hat_pre = dfp.loc[vpre, "mu0_hat"].to_numpy(dtype=float)
    a_cal, b_cal = _ols_calibrate(y_obs_pre, y_hat_pre)
    dfp.loc[vmask, "mu0_hat"] = a_cal + b_cal * dfp.loc[vmask, "mu0_hat"]

    # --- Métricas (con baseline calibrada) ---
    y_obs_pre = dfp.loc[vpre, "sales"].to_numpy(dtype=float)
    y_hat_pre = dfp.loc[vpre, "mu0_hat"].to_numpy(dtype=float)
    y_obs_post = dfp.loc[vpost, "sales"].to_numpy(dtype=float)
    y_hat_post = dfp.loc[vpost, "mu0_hat"].to_numpy(dtype=float)

    rmspe_pre = _rmspe(y_obs_pre, y_hat_pre) if y_obs_pre.size else np.nan
    mae_pre = _mae(y_obs_pre, y_hat_pre) if y_obs_pre.size else np.nan
    effect_post = y_obs_post - y_hat_post
    att_sum = float(np.nansum(effect_post))
    att_mean = float(np.nanmean(effect_post)) if effect_post.size else np.nan
    rel_att = _safe_pct(att_mean, float(np.nanmean(y_obs_pre)) if y_obs_pre.size else np.nan)

    # --- Guardar serie CF (víctima, con episode_id para trazabilidad) ---
    cf_dir = Path(cfg.out_dir) / "cf_series"
    _ensure_dir(cf_dir)

    cf = dfp.loc[vmask, ["episode_id", "unit_id", "date", "sales", "mu0_hat", "mu1_hat", "tau_hat", "D", "treated_time"]].copy()
    cf = cf.sort_values("date")
    # (por seguridad) unicidad por fecha dentro del episodio
    if not cf["date"].is_unique:
        cf = cf.groupby("date", as_index=False).last()

    cf["effect"] = cf["sales"] - cf["mu0_hat"]
    is_post = cf["date"] >= treat_start
    cf["cum_effect"] = (cf["effect"].where(is_post, 0.0)).cumsum()

    cf_path = cf_dir / f"{ep_id}_cf.parquet"
    cf.to_parquet(cf_path, index=False)

    # --- Placebo en el espacio (usando sólo filas del episodio recortado) ---
    plac_dir = Path(cfg.out_dir) / "placebos"
    _ensure_dir(plac_dir)
    pval_space = np.nan
    if cfg.do_placebo_space:
        # ep_rows: todo el episodio (víctima + (opcional) donantes si están en meta_units)
        ep_rows = dfp.loc[dfp["date"].between(ep_row["pre_start"], post_end, inclusive="both")].copy()
        ps = placebo_space_meta(ep_rows, donors_df, (j_store, j_item), treat_start, post_end)
        ps["episode_id"] = ep_id
        ps_path = plac_dir / f"{ep_id}_space.parquet"
        ps.to_parquet(ps_path, index=False)
        if not ps.empty and np.isfinite(att_sum):
            pval_space = (1 + np.sum(np.abs(ps["att_placebo_sum"].to_numpy()) >= abs(att_sum))) / (1 + ps.shape[0])
    else:
        ps_path = None

    # --- Placebo in-time (sobre la víctima) ---
    if cfg.do_placebo_time:
        pt = placebo_time_victim(ep_rows if cfg.do_placebo_space else dfp, treat_start, post_end, victim_uid)
        pt["episode_id"] = ep_id
        pt_path = plac_dir / f"{ep_id}_time.parquet"
        pt.to_parquet(pt_path, index=False)
    else:
        pt_path = None

    # --- LOO (opcional y costoso) ---
    loo_dir = Path(cfg.out_dir) / "loo"
    _ensure_dir(loo_dir)
    loo_sd = np.nan
    loo_range = np.nan
    if cfg.do_loo:
        ep_f = pd.DataFrame([{"pre_start": ep_row["pre_start"], "treat_start": treat_start, "post_end": post_end}])
        tcfg_local = TrainCfg(**asdict(tcfg))
        loo = loo_retrain_if_requested(meta_df, feats, tcfg_local, donors_df, ep_f, (j_store, j_item), cfg.max_loo)
        if not loo.empty:
            loo["episode_id"] = ep_id
            loo_path = loo_dir / f"{ep_id}_loo.parquet"
            loo.to_parquet(loo_path, index=False)
            loo_sd = float(np.nanstd(loo["att_sum"]))
            loo_range = float(np.nanmax(loo["att_sum"]) - np.nanmin(loo["att_sum"]))
        else:
            loo_path = None
    else:
        loo_path = None

    # --- Sensibilidad (opcional) ---
    sens_dir = Path(cfg.out_dir) / "sensitivity"
    _ensure_dir(sens_dir)
    sens_sd = np.nan
    if int(cfg.sens_samples) > 0:
        sens = sensitivity_sweep(meta_df, feats, tcfg, victim_uid, treat_start, post_end, n_samples=int(cfg.sens_samples))
        if not sens.empty:
            sens["episode_id"] = ep_id
            sens_path = sens_dir / f"{ep_id}_sens.parquet"
            sens.to_parquet(sens_path, index=False)
            sens_sd = float(np.nanstd(sens["att_sum"]))
        else:
            sens_path = None
    else:
        sens_path = None

    # --- Reporte JSON por episodio ---
    rep_dir = Path(cfg.out_dir) / "reports"
    _ensure_dir(rep_dir)
    rep = {
        "episode_id": ep_id,
        "victim_unit": victim_uid,
        "window": {"pre_start": str(ep_row["pre_start"]), "treat_start": str(treat_start), "post_end": str(post_end)},
        "learner": cfg.learner,
        "model": cfg.model,
        "cv": {"folds": cfg.cv_folds, "holdout_days": cfg.cv_holdout_days, "gap_days": _CV_GAP_DAYS},
        "fit": {
            "rmspe_pre": rmspe_pre, "mae_pre": mae_pre,
            "att_mean": att_mean, "att_sum": att_sum, "rel_att_vs_pre_mean": rel_att
        },
        "p_value_placebo_space": pval_space,
        "robustness": {"loo_sd_att_sum": loo_sd, "loo_range_att_sum": loo_range, "sens_sd_att_sum": sens_sd},
        "paths": {"cf": str(cf_path), "placebo_space": str(ps_path) if cfg.do_placebo_space else None,
                  "placebo_time": str(pt_path) if cfg.do_placebo_time else None,
                  "loo": str(loo_path) if cfg.do_loo else None,
                  "sensitivity": str(sens_path) if int(cfg.sens_samples) > 0 else None},
        "calibration": {"a": a_cal, "b": b_cal}
    }
    with open(rep_dir / f"{ep_id}.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # --- Resumen para meta_metrics_<learner>.parquet ---
    n_pre_days = int(pd.Series(dfp.loc[vpre, "date"]).nunique())
    n_post_days = int(pd.Series(dfp.loc[vpost, "date"]).nunique())
    # Si hubo groupby por duplicados, n_pre_days/n_post_days ya reflejan unicidad

    summary = {
        "episode_id": ep_id,
        "victim_unit": victim_uid,
        "learner": cfg.learner,
        "n_pre_days": n_pre_days,
        "n_post_days": n_post_days,
        "rmspe_pre": rmspe_pre,
        "mae_pre": mae_pre,
        "att_mean": att_mean,
        "att_sum": att_sum,
        "rel_att_vs_pre_mean": rel_att,
        "p_value_placebo_space": pval_space,
        "loo_sd_att_sum": loo_sd,
        "loo_range_att_sum": loo_range,
        "sens_sd_att_sum": sens_sd
    }
    return summary


def run_batch(cfg: RunCfg) -> None:
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    if cfg.out_dir is None:
        cfg.out_dir = (cfg.meta_dir / cfg.exp_tag / "meta" / cfg.learner).resolve()
    # Cargar insumos
    meta_df = _load_meta_df(Path(cfg.meta_parquet))
    episodes = _load_episodes(Path(cfg.episodes_index))
    if (episodes is None or episodes.empty) and cfg.exp_tag:
        alt_ep = Path("figures") / cfg.exp_tag / "tables" / "episodes_index.parquet"
        if alt_ep.exists():
            episodes = _load_episodes(alt_ep)

    donors = _load_donors(Path(cfg.donors_csv))
    if (donors is None or donors.empty) and cfg.exp_tag:
        alt_dn = Path("figures") / cfg.exp_tag / "tables" / "donors_per_victim.csv"
        if alt_dn.exists():
            donors = _load_donors(alt_dn)

    # Selección de features (excluye columnas de tratamiento/etiquetas para evitar leakage)
    feats = select_feature_cols(meta_df, extra_exclude=[cfg.treat_col_s, cfg.treat_col_b])
    logging.info(f"Features seleccionadas ({len(feats)}): {feats[:8]}{' ...' if len(feats) > 8 else ''}")

    # Asegurar unit_id, date ordenados
    meta_df = meta_df.sort_values(["date", "unit_id"]).reset_index(drop=True)

    # Filtrar episodios si se solicita
    if cfg.max_episodes is not None:
        episodes = episodes.head(int(cfg.max_episodes)).copy()

    # --------------------- HPO (Optuna) antes del cross-fitting ---------------------
    if int(cfg.hpo_trials) > 0 and cfg.model.lower() in {"lgbm", "hgbt"}:
        logging.info(f"Iniciando HPO (Optuna) para modelo '{cfg.model}' con {int(cfg.hpo_trials)} trials...")
        best = tune_hyperparams(
            meta_df, feats, cfg.model, cfg.random_state, cfg.cv_folds, cfg.cv_holdout_days,
            {
                "max_iter": cfg.max_iter,
                "hpo_trials": cfg.hpo_trials
            }
        )
        if best:
            # Inyectar los hiperparámetros óptimos en cfg
            for k, v in best.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    if k == "min_child_samples":
                        cfg.min_child_samples = int(v)
                    elif k == "feature_fraction":
                        cfg.feature_fraction = float(v)
                    elif k == "num_leaves":
                        cfg.num_leaves = int(v)
                    elif k == "subsample":
                        cfg.subsample = float(v)
                    elif k == "reg_lambda":
                        cfg.l2 = float(v)
                    else:
                        logging.debug(f"HPO: parámetro '{k}' no es atributo directo de cfg; se ignora.")
            logging.info(f"HPO finalizado. Hiperparámetros aplicados al entrenamiento: {best}")
        else:
            logging.info("HPO no produjo mejoras o fue omitido; se entrenará con hiperparámetros por defecto.")
    else:
        if int(cfg.hpo_trials) <= 0:
            logging.info("HPO desactivado por configuración (hpo_trials=0).")
        else:
            logging.info(f"Modelo '{cfg.model}' no soporta HPO en este pipeline; se omite HPO.")

    # Entrenar cross-fitting global una sola vez y cachear (SIEMPRE con el meta parquet grande)
    tcfg = TrainCfg(learner=cfg.learner, model=cfg.model, prop_model=cfg.prop_model,
                    random_state=cfg.random_state, cv_folds=cfg.cv_folds,
                    cv_holdout_days=cfg.cv_holdout_days, min_train_samples=cfg.min_train_samples,
                    max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                    max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                    num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                    feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                    sens_samples=cfg.sens_samples,
                    treat_col_s=cfg.treat_col_s, s_ref=cfg.s_ref,
                    treat_col_b=cfg.treat_col_b, bin_threshold=cfg.bin_threshold)
    logging.info(f"Entrenando {cfg.learner.upper()}-learner con cross-fitting temporal (gap={_CV_GAP_DAYS} días) sobre el dataset grande...")
    pred_cache = crossfit_predictions(meta_df, feats, tcfg)

    # Directorios de salida
    _ensure_dir(Path(cfg.out_dir) / "cf_series")
    _ensure_dir(Path(cfg.out_dir) / "placebos")
    _ensure_dir(Path(cfg.out_dir) / "loo")
    _ensure_dir(Path(cfg.out_dir) / "sensitivity")
    _ensure_dir(Path(cfg.out_dir) / "reports")

    # Iterar episodios (SOLO los del set pequeño que usarás en GSC)
    summaries = []
    order_cols = [c for c in ["treat_start", "j_store", "j_item"] if c in episodes.columns]
    if order_cols:
        episodes = episodes.sort_values(order_cols, kind="mergesort").reset_index(drop=True)

    for k, (_, ep) in enumerate(episodes.iterrows(), start=1):
        logging.info(f"[{k}/{episodes.shape[0]}] Episodio {ep.get('episode_id', '?')} | víctima {int(ep['j_store'])}:{int(ep['j_item'])}")
        try:
            summ = run_episode(meta_df, feats, ep, donors, cfg, pred_cache)
            summaries.append(summ)
            logging.info(f"OK episodio {summ['episode_id']} | ATT_sum={summ['att_sum']:.3f} | RMSPE_pre={summ['rmspe_pre']:.4f}")
        except Exception as e:
            logging.exception(f"Error en episodio {ep.get('episode_id','?')}: {e}")

    # Guardar métricas globales
    if summaries:
        mdf = pd.DataFrame(summaries)
        _ensure_dir(Path(cfg.out_dir))
        out_path = Path(cfg.out_dir) / f"meta_metrics_{cfg.learner.lower()}.parquet"
        mdf.to_parquet(out_path, index=False)
        logging.info(f"Métricas globales guardadas: {out_path} ({mdf.shape[0]} episodios)")
    else:
        logging.warning("No se generaron métricas (¿sin episodios válidos o fallos previos?).")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> RunCfg:
    p = argparse.ArgumentParser(
        description=(
            "T/S/X-learners (time-series aware) para canibalización en retail.\n"
            "Entrena SIEMPRE con --meta_parquet (dataset grande) y evalúa SOLO "
            "sobre --episodes_index (set pequeño que usarás en GSC)."
        )
    )

    p.add_argument("--learner", type=str, default="x", choices=["t", "s", "x"], help="Tipo de meta-learner.")
    p.add_argument("--meta_parquet", type=str, default=str(_data_dir / "meta" / "all_units.parquet"))
    p.add_argument("--episodes_index", type=str, default=str(_data_dir / "meta" / "episodes_index.parquet"))
    p.add_argument("--donors_csv", type=str, default=str(_data_dir / "meta" / "donors_per_victim.csv"))
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--max_episodes", type=int, default=None)

    # CV temporal
    p.add_argument("--cv_folds", type=int, default=3)
    p.add_argument("--cv_holdout", type=int, default=21)
    p.add_argument("--min_train_samples", type=int, default=50)

    # modelo base (incluye lgbm)
    p.add_argument("--model", type=str, default="lgbm", choices=["lgbm", "hgbt", "rf", "ridge"])
    p.add_argument("--prop_model", type=str, default="logit")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_depth", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--max_iter", type=int, default=600)
    p.add_argument("--min_samples_leaf", type=int, default=10)
    p.add_argument("--l2", type=float, default=0.0)

    # LightGBM específicos opcionales
    p.add_argument("--num_leaves", type=int, default=127)
    p.add_argument("--min_child_samples", type=int, default=10)
    p.add_argument("--feature_fraction", type=float, default=0.8)
    p.add_argument("--subsample", type=float, default=0.8)

    # HPO
    p.add_argument("--hpo_trials", type=int, default=52, help="N° de trials de Optuna (0 para desactivar).")

    # diagnósticos
    p.add_argument("--do_placebo_space", action="store_true")
    p.add_argument("--do_placebo_time", action="store_true")
    p.add_argument("--do_loo", action="store_true")
    p.add_argument("--max_loo", type=int, default=5)
    p.add_argument("--sens_samples", type=int, default=0)

    # Tratamientos configurables
    p.add_argument("--treat_col_s", type=str, default="D", help="Columna de tratamiento continuo para S-learner (por defecto D).")
    p.add_argument("--s_ref", type=float, default=0.0, help="Valor de referencia (baseline) para S-learner.")
    p.add_argument("--treat_col_b", type=str, default="D", help="Columna de tratamiento binario para T/X-learners (por defecto D).")
    p.add_argument("--bin_threshold", type=float, default=0.0, help="Umbral para binarizar treat_col_b si no es 0/1.")
    p.add_argument("--exp_tag", type=str, default="A_base")

    a = p.parse_args()
    return RunCfg(
        learner=a.learner,
        meta_parquet=Path(a.meta_parquet),
        episodes_index=Path(a.episodes_index),
        donors_csv=Path(a.donors_csv),
        out_dir=out_dir,
        log_level=a.log_level,
        max_episodes=a.max_episodes,
        cv_folds=a.cv_folds,
        cv_holdout_days=a.cv_holdout,
        min_train_samples=a.min_train_samples,
        model=a.model,
        prop_model=a.prop_model,
        random_state=a.random_state,
        max_depth=a.max_depth,
        learning_rate=a.learning_rate,
        max_iter=a.max_iter,
        min_samples_leaf=a.min_samples_leaf,
        l2=a.l2,
        # LGBM
        num_leaves=a.num_leaves,
        min_child_samples=a.min_child_samples,
        feature_fraction=a.feature_fraction,
        subsample=a.subsample,
        # HPO
        hpo_trials=a.hpo_trials,
        # diagnósticos
        do_placebo_space=a.do_placebo_space,
        do_placebo_time=a.do_placebo_time,
        do_loo=a.do_loo,
        max_loo=a.max_loo,
        sens_samples=a.sens_samples,
        # tratamientos
        treat_col_s=a.treat_col_s,
        s_ref=a.s_ref,
        treat_col_b=a.treat_col_b,
        bin_threshold=a.bin_threshold,
        exp_tag=a.exp_tag,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_batch(cfg)