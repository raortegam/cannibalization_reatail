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
# meta_learners.py
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
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

# =============================================================================
# Rutas por defecto
# =============================================================================

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

def _prefer_existing(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        try:
            if c and c.exists():
                return c
        except Exception:
            pass
    return None

def _hpo_registry_path(exp_tag: Optional[str], learner: str, model_name: str) -> Optional[Path]:
    if not exp_tag:
        return None
    return Path("figures") / str(exp_tag) / "tables" / f"meta_hpo_{learner.lower()}_{model_name.lower()}.json"

def _load_json_silent(p: Path) -> Optional[Dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_json_silent(p: Path, data: Dict) -> None:
    try:
        _ensure_dir(p.parent)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# =============================================================================
# Utilidades generales y selección de features (anti-fuga)
# =============================================================================

EXCLUDE_BASE_COLS = {
    # Identificadores / etiquetas
    "id", "date", "year_week", "store_nbr", "item_nbr", "unit_id",
    "family_name", "cluster_id", "w_episode",

    # Objetivo y tratamiento
    "sales", "treated_unit", "treated_time", "D", "is_pre", "train_mask",

    # Info de episodio / calidad de donantes
    "episode_id", "is_victim", "promo_share", "avail_share", "keep", "reason",

    # Mediadores inmediatos
    "onpromotion",
}

# --- Hiperparámetros internos ---
_CV_GAP_DAYS = 7
_LAMBDA_RECENCY = 0.004  # peso temporal: exp(-λ * days_to_vstart)
_MIN_TRAIN_FOLD = 30     # mínimo “blando” por grupo

def select_feature_cols(df: pd.DataFrame, extra_exclude: Sequence[str] | None = None) -> List[str]:
    """
    Selecciona columnas numéricas seguras como X.
    Compatible con 'regional_proxy' y el alias antiguo 'F_state_excl_store_log1p'.
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

    keep = {
        # proxies/señales base
        "Fsw_log1p", "Ow", "regional_proxy", "F_state_excl_store_log1p",
        # holidays/types
        "HNat", "HReg", "HLoc", "is_bridge", "is_additional", "is_work_day",
        # disponibilidad / intermitencia
        "available_A", "available_A_l1", "ADI", "CV2", "zero_streak",
        # donor blend
        "sc_hat",
        # promo/class (lags)
        "promo_share_sc_l7", "promo_share_sc_l14",
        "promo_share_sc_excl_l7", "promo_share_sc_excl_l14",
        "class_index_excl_l7", "class_index_excl_l14",
        # metadatos discretos
        "month",
    }

    feats = [
        c for c in feats
        if (c.startswith(("fourier_", "lag_", "dow_", "type_", "cluster_", "state_"))
            or c in keep)
    ]

    # Si solo existe el alias viejo, renombramos on-the-fly a regional_proxy (no modifica el df)
    if "regional_proxy" not in feats and "F_state_excl_store_log1p" in feats:
        feats.append("regional_proxy")  # modelo verá columna si está en df; si no, se ignora (cero)
    feats = sorted(list(dict.fromkeys(feats)))
    if not feats:
        raise ValueError("No se encontraron features válidos en el parquet meta.")
    return feats

# =============================================================================
# Pequeñas utilidades
# =============================================================================

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

def _is_count_target(y: np.ndarray) -> bool:
    """Heurístico: target no-negativo y cuasi-entero -> pérdida Poisson."""
    if y.size == 0:
        return False
    if np.nanmin(y) < -1e-12:
        return False
    frac_int = np.nanmean(np.abs(y - np.round(y)) <= 0.05)
    return bool(frac_int >= 0.9)

# =============================================================================
# Modelos base (regresores/clasificadores) con fallbacks
# =============================================================================

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

def make_regressor(name: str = "lgbm", random_state: int = 42, **kwargs):
    """
    Regresor robusto:
      - 'lgbm' (LightGBM) [default]
      - 'hgbt' (HistGradientBoostingRegressor)
      - 'rf'   (RandomForestRegressor)
      - 'ridge'
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
            name = "hgbt"

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

def make_classifier(name: str = "logit", random_state: int = 42):
    """
    Clasificador de propensión:
      - 'hgbc' (HistGradientBoostingClassifier)
      - fallback: LogisticRegression
    """
    if name == "hgbc":
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(random_state=random_state)
        except Exception:
            pass
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=200, solver="liblinear", random_state=random_state)

# Modelo constante
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
        if isinstance(model, Pipeline):
            last_name = model.steps[-1][0]
            return model.fit(X, y, **{f"{last_name}__sample_weight": sample_weight})
        return model.fit(X, y)

# =============================================================================
# Split temporal (forward chaining con H días de holdout + embargo)
# =============================================================================

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

# =============================================================================
# Enriquecimiento por fold: unit stats (solo con TRAIN del fold)
# =============================================================================

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
    Retorna: X_tr_aug, X_va_aug, y_tr, y_va, sample_weight_tr, is_count
    """
    X = df[feats].values.astype(float)
    X[np.isnan(X)] = 0.0
    y = df["sales"].to_numpy(dtype=float)
    dates = _as_datetime(df["date"])
    uids = df["unit_id"]

    # Embargo temporal
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

# =============================================================================
# HPO con Optuna (global, antes del cross-fitting)
# =============================================================================

def _finite_or_big(x: float, big: float = 1e6) -> float:
    return float(x) if np.isfinite(x) else float(big)

def _time_cv_score(meta_df: pd.DataFrame,
                   feats: List[str],
                   cfg_cv_folds: int,
                   cfg_holdout_days: int,
                   build_model_fn,
                   **model_kwargs) -> float:
    dates = _as_datetime(meta_df["date"])
    windows = make_time_folds(dates, cfg_holdout_days, cfg_cv_folds)
    if not windows:
        return np.inf

    scores: List[float] = []
    for (v_start, v_end) in windows:
        valid_mask = (dates >= v_start) & (dates <= v_end)
        train_mask = (dates < v_start)
        X_tr_aug, X_va_aug, y_tr, y_va, sw, is_count = _build_augmented_matrices(
            meta_df, feats, train_mask, valid_mask, v_start
        )

        if X_tr_aug.shape[0] < max(_MIN_TRAIN_FOLD, 10) or X_va_aug.shape[0] < 1:
            continue

        model_params = dict(model_kwargs)
        if "objective" not in model_params and "loss" not in model_params:
            if is_count:
                model_params.update({"objective": "poisson", "loss": "poisson"})
        model = build_model_fn(**model_params)

        try:
            _fit_with_weights(model, X_tr_aug, y_tr, sw)
            y_hat = model.predict(X_va_aug)
            scores.append(_rmspe(y_va, y_hat))
        except Exception as e:
            logging.debug(f"HPO fold falló: {e}")
            scores.append(1e3)

    return float(np.mean(scores)) if scores else np.inf

def tune_hyperparams(meta_df: pd.DataFrame, feats: List[str], cfg_model: str,
                     random_state: int, cv_folds: int, cv_holdout_days: int,
                     base_defaults: Dict) -> Dict:
    try:
        import optuna  # type: ignore
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    except Exception as e:
        logging.warning(f"Optuna no disponible ({e}); se omite HPO.")
        return {}

    model_name = (cfg_model or "lgbm").lower()
    warm_params = dict(base_defaults.get("hpo_warm_params", {}) or {})

    def objective(trial):
        if model_name == "lgbm":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 31, 511),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "learning_rate": trial.suggest_float("learning_rate", 5e-4, 0.3, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 3, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 20.0),
                "max_iter": trial.suggest_int("max_iter", 400, 1500, step=100),
            }
        elif model_name == "hgbt":
            params = {
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "learning_rate": trial.suggest_float("learning_rate", 5e-4, 0.3, log=True),
                "max_iter": trial.suggest_int("max_iter", 400, 3000, step=100),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 100),
                "l2": trial.suggest_float("l2", 1e-7, 20.0, log=True),
            }
        else:
            return np.inf

        score = _time_cv_score(
            meta_df, feats, cv_folds, cv_holdout_days,
            lambda **kw: make_regressor(model_name, random_state, **kw),
            **params
        )
        return _finite_or_big(score)

    sampler = optuna.samplers.TPESampler(seed=int(random_state))
    study = optuna.create_study(direction="minimize", study_name=f"hpo_{model_name}", sampler=sampler)
    if warm_params:
        try:
            study.enqueue_trial({k: warm_params[k] for k in warm_params})
        except Exception:
            pass
    study.optimize(objective, n_trials=int(base_defaults.get("hpo_trials", 100)), show_progress_bar=False)
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

# =============================================================================
# Núcleo: cross-fitting + learners
# =============================================================================

@dataclass
class TrainCfg:
    learner: str = "x"                # 't' | 's' | 'x'
    model: str = "lgbm"
    prop_model: str = "logit"
    random_state: int = 42
    cv_folds: int = 3
    cv_holdout_days: int = 21
    min_train_samples: int = 50

    # hiperparámetros genéricos / HGBT
    max_depth: int = 8
    learning_rate: float = 0.05
    max_iter: int = 600
    min_samples_leaf: int = 10
    l2: float = 0.0

    # LightGBM
    num_leaves: int = 127
    min_child_samples: int = 10
    feature_fraction: float = 0.8
    subsample: float = 0.8

    # sensibilidad
    sens_samples: int = 0

    # Tratamientos configurables
    treat_col_s: str = "D"
    s_ref: float = 0.0
    treat_col_b: str = "D"
    bin_threshold: float = 0.0

def _build_X(df: pd.DataFrame, feats: List[str]) -> np.ndarray:
    X = df[feats].values.astype(float)
    X[np.isnan(X)] = 0.0
    return X

def crossfit_predictions(df: pd.DataFrame, feats: List[str], cfg: TrainCfg) -> Dict[str, np.ndarray]:
    """
    Predicciones cross-fitted:
      mu0_cf, mu1_cf, tau_cf
    y refits completos:
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

    windows = make_time_folds(dates, cfg.cv_holdout_days, cfg.cv_folds)
    if not windows:
        logging.warning("No se pudieron construir ventanas de CV temporal; se harán refits completos sin cross-fitting.")

    for (v_start, v_end) in windows:
        valid_mask = (dates >= v_start) & (dates <= v_end)
        train_mask = (dates < v_start)

        X_tr_aug, X_va_aug, y_tr, y_va, sw, is_count = _build_augmented_matrices(
            df, feats, train_mask, valid_mask, v_start
        )
        if X_tr_aug.shape[0] < max(_MIN_TRAIN_FOLD, 10) or X_va_aug.shape[0] < 1:
            continue

        # --- T-learner ---
        if cfg.learner.lower() == "t":
            D_tr = D_bin[train_mask & (dates < (v_start - pd.Timedelta(days=_CV_GAP_DAYS)))]
            mask0 = (D_tr == 0); mask1 = (D_tr == 1)
            min_needed = max(_MIN_TRAIN_FOLD, cfg.min_train_samples // 2)

            # m0
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

            # m1
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

        # --- S-learner ---
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

        # --- X-learner ---
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

            # Degradar a T si falta variación en TRAIN
            if (mask0.sum() == 0) or (mask1.sum() == 0):
                mu0_cf[valid_mask] = m0.predict(X_va_aug)
                mu1_cf[valid_mask] = m1.predict(X_va_aug)
                tau_cf[valid_mask] = mu1_cf[valid_mask] - mu0_cf[valid_mask]
                continue

            mu0_on_t = m0.predict(X_tr_aug[mask1])
            mu1_on_c = m1.predict(X_tr_aug[mask0])
            d1 = y_tr[mask1] - mu0_on_t
            d0 = mu1_on_c - y_tr[mask0]

            g1 = make_regressor(cfg.model, cfg.random_state,
                                max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                reg_lambda=cfg.l2, objective=None, loss="squared_error")
            g0 = make_regressor(cfg.model, cfg.random_state,
                                max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                                max_iter=cfg.max_iter, min_samples_leaf=cfg.min_samples_leaf, l2=cfg.l2,
                                num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
                                feature_fraction=cfg.feature_fraction, subsample=cfg.subsample,
                                reg_lambda=cfg.l2, objective=None, loss="squared_error")

            if mask1.sum() >= min_needed:
                _fit_with_weights(g1, X_tr_aug[mask1], d1, sw[mask1])
            else:
                g1 = _ConstantRegressor(np.nanmean(d1) if len(d1) else 0.0).fit(X_tr_aug, y_tr)

            if mask0.sum() >= min_needed:
                _fit_with_weights(g0, X_tr_aug[mask0], d0, sw[mask0])
            else:
                g0 = _ConstantRegressor(np.nanmean(d0) if len(d0) else 0.0).fit(X_tr_aug, y_tr)

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
    X_full = _build_X(df, feats); X_full[np.isnan(X_full)] = 0.0
    y_full = y
    dates_full = dates
    uids_full = df["unit_id"]

    stats_all, default_all = _build_unit_stats(df, np.ones(len(df), dtype=bool))
    Z_all = _map_stats_to_rows(uids_full, stats_all, default_all)
    X_all_aug = np.column_stack([X_full, Z_all])
    is_count_full = _is_count_target(y_full)

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
        X_ref = np.column_stack([X_all_aug, np.full((X_all_aug.shape[0], 1), 0.0)])
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

        mu0_full = m0_full.predict(X_all_aug);  mu1_full = m1_full.predict(X_all_aug)

        d1_all = y_full[mask1_all] - mu0_full[mask1_all]
        d0_all = mu1_full[mask0_all] - y_full[mask0_all]

        g1_full = make_regressor(cfg.model, cfg.random_state);  g0_full = make_regressor(cfg.model, cfg.random_state)
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
    if np.any(~np.isfinite(mu0_cf)): mu0_cf[~np.isfinite(mu0_cf)] = mu0_full[~np.isfinite(mu0_cf)]
    if np.any(~np.isfinite(mu1_cf)): mu1_cf[~np.isfinite(mu1_cf)] = mu1_full[~np.isfinite(mu1_cf)]
    if np.any(~np.isfinite(tau_cf)): tau_cf[~np.isfinite(tau_cf)] = tau_full[~np.isfinite(tau_cf)]

    return {"mu0_cf": mu0_cf, "mu1_cf": mu1_cf, "tau_cf": tau_cf,
            "mu0_full": mu0_full, "mu1_full": mu1_full, "tau_full": tau_full}

# =============================================================================
# Placebos, LOO y sensibilidad
# =============================================================================

def placebo_space_meta(ep_rows: pd.DataFrame,
                       donors_map: pd.DataFrame,
                       victim: Tuple[int, int],
                       treat_start: pd.Timestamp,
                       post_end: pd.Timestamp) -> pd.DataFrame:
    j_store, j_item = victim
    if donors_map is None or donors_map.empty:
        return pd.DataFrame(columns=["unit_id", "att_placebo_mean", "att_placebo_sum"])

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
    post_len = int((post_end - treat_start).days + 1)
    v = ep_rows.loc[ep_rows["unit_id"] == victim_uid].sort_values("date")
    pre = v.loc[v["date"] < treat_start]
    if pre.shape[0] < max(7, post_len):
        return pd.DataFrame([{"used": False, "att_placebo_sum": np.nan, "att_placebo_mean": np.nan, "H": 0}])
    win = pre.tail(post_len)
    return pd.DataFrame([{
        "used": True, "H": int(post_len),
        "att_placebo_sum": float(np.nansum(win["tau_hat"])),
        "att_placebo_mean": float(np.nanmean(win["tau_hat"]))
    }])

# =============================================================================
# Pipeline por episodio y batch
# =============================================================================

@dataclass
class RunCfg:
    learner: str = "x"  # 't'|'s'|'x'

    # Entrena SIEMPRE con este parquet grande:
    meta_parquet: Path = field(default_factory=lambda: (
        _prefer_existing(
            _data_root() / "processed_meta" / "windows.parquet",
            _data_root() / "processed" / "meta" / "all_units.parquet",
            _data_root() / "processed_data" / "meta" / "all_units.parquet"
        ) or (_data_root() / "processed_meta" / "windows.parquet")
    ))
    # Evalúa SOLO sobre los episodios de este índice:
    episodes_index: Path = field(default_factory=lambda: (
        _prefer_existing(
            _data_root() / "processed_data" / "episodes_index.parquet",
            _data_root() / "processed" / "episodes_index.parquet"
        ) or (_data_root() / "processed_data" / "episodes_index.parquet")
    ))
    # Donantes por víctima (fallback a figures/<exp_tag>/tables si no existe)
    donors_csv: Path = field(default_factory=lambda: _data_root() / "processed_data" / "A_base" / "donors_per_victim.csv")

    # Salidas
    out_dir: Optional[Path] = None
    exp_tag: str = "A_base"
    fig_root: Path = field(default_factory=_fig_root)

    # logging / ejecución
    log_level: str = "INFO"
    max_episodes: Optional[int] = None

    # CV temporal
    cv_folds: int = 3
    cv_holdout_days: int = 21
    min_train_samples: int = 50

    # modelo base
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
    hpo_trials: int = 100  # 0 para desactivar; aumentado para mejor exploración

    # diagnósticos
    do_placebo_space: bool = True
    do_placebo_time: bool = True
    do_loo: bool = False
    max_loo: int = 5
    sens_samples: int = 0   # 0 => desactivado

    # Tratamientos configurables
    treat_col_s: str = "D"
    s_ref: float = 0.0
    treat_col_b: str = "D"
    bin_threshold: float = 0.0

def _default_out_dir(cfg: RunCfg) -> Path:
    """
    Carpeta de salidas: ./.data/processed_data/meta_outputs/<exp_tag>/<learner>
    y espejos útiles en figures/<exp_tag>/tables cuando aplica.
    """
    base = _data_root() / "processed_data" / "meta_outputs" / cfg.exp_tag / cfg.learner
    return base

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

    # Rezago seguro de disponibilidad
    if "available_A" in df.columns and "available_A_l1" not in df.columns:
        df["available_A_l1"] = (
            df.groupby("unit_id", observed=False)["available_A"].shift(1).astype("float32")
        )

    # Si existe alias viejo de proxy, crear columna canónica
    if "regional_proxy" not in df.columns and "F_state_excl_store_log1p" in df.columns:
        df["regional_proxy"] = df["F_state_excl_store_log1p"]

    return df

def _load_episodes(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_parquet(p)
    for c in ["pre_start", "treat_start", "post_start", "post_end"]:
        if c in df.columns:
            df[c] = _as_datetime(df[c])
    return df

def _load_donors(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=["j_store","j_item","donor_store","donor_item","rank"])

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
    *Clave*: primero filtramos por episode_id; si no hay filas, caemos a victim_uid+ventana.
    """
    j_store, j_item = int(ep_row["j_store"]), int(ep_row["j_item"])
    victim_uid = f"{j_store}:{j_item}"
    treat_start, post_end = ep_row["treat_start"], ep_row["post_end"]
    ep_id = str(ep_row.get("episode_id", f"{j_store}-{j_item}_{pd.to_datetime(treat_start).strftime('%Y%m%d')}"))

    # Config de entrenamiento
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

    # Predicciones cross-fitted (cache global)
    if pred_cache is None or not pred_cache:
        pred_cache = crossfit_predictions(meta_df, feats, tcfg)

    # --- Subset por episodio (preferido) y fallback robusto ---
    if "episode_id" in meta_df.columns:
        ep_mask = meta_df["episode_id"].astype(str).eq(ep_id)
    else:
        ep_mask = pd.Series(False, index=meta_df.index)

    if not np.any(ep_mask):
        # Fallback: por víctima + ventana
        ep_mask = (
            (meta_df["unit_id"] == victim_uid) &
            meta_df["date"].between(ep_row["pre_start"], post_end, inclusive="both")
        )

    if not np.any(ep_mask):
        raise ValueError(f"[{ep_id}] No hay filas en meta_df para este episode_id (revisa origen de windows.parquet).")

    idx = meta_df.index[ep_mask]
    dfp = meta_df.loc[idx, ["episode_id", "unit_id", "date", "sales", "D", "treated_unit", "treated_time"]].copy()

    # Alinear predicciones por índice del subset
    dfp["mu0_hat"] = pred_cache["mu0_cf"][idx]
    dfp["mu1_hat"] = pred_cache["mu1_cf"][idx]
    dfp["tau_hat"] = pred_cache["tau_cf"][idx]

    # Serie víctima
    vmask = (dfp["unit_id"] == victim_uid) & dfp["date"].between(ep_row["pre_start"], post_end, inclusive="both")
    vpre = vmask & (dfp["date"] < treat_start)
    vpost = vmask & (dfp["date"] >= treat_start)

    # Unicidad por fecha (dentro del episodio)
    dup_count = int(dfp.loc[vmask, ["date"]].duplicated().sum())
    if dup_count > 0:
        logging.warning(f"[{ep_id}] (unit {victim_uid}) fechas duplicadas dentro del episodio: {dup_count}. Se resolverán por última ocurrencia.")
        ord_cols = ["date", "treated_time", "D"]
        keep_cols = [c for c in ["episode_id", "unit_id", "date", "sales", "D", "treated_unit", "treated_time",
                                 "mu0_hat", "mu1_hat", "tau_hat"] if c in dfp.columns]
        dfp = (dfp.sort_values(ord_cols)
                  .groupby(["episode_id", "unit_id", "date"], as_index=False)[keep_cols].last())
        vmask = (dfp["unit_id"] == victim_uid) & dfp["date"].between(ep_row["pre_start"], post_end, inclusive="both")
        vpre = vmask & (dfp["date"] < treat_start)
        vpost = vmask & (dfp["date"] >= treat_start)

    # Calibración PRE
    y_obs_pre = dfp.loc[vpre, "sales"].to_numpy(dtype=float)
    y_hat_pre = dfp.loc[vpre, "mu0_hat"].to_numpy(dtype=float)
    a_cal, b_cal = _ols_calibrate(y_obs_pre, y_hat_pre)
    dfp.loc[vmask, "mu0_hat"] = a_cal + b_cal * dfp.loc[vmask, "mu0_hat"]

    # Métricas
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

    # Guardado CF (víctima)
    cf_dir = Path(cfg.out_dir) / "cf_series"
    _ensure_dir(cf_dir)
    cf = dfp.loc[vmask, ["episode_id", "unit_id", "date", "sales", "mu0_hat", "mu1_hat", "tau_hat", "D", "treated_time"]].copy()
    cf = cf.sort_values("date")
    if not cf["date"].is_unique:
        cf = cf.groupby("date", as_index=False).last()
    cf["effect"] = cf["sales"] - cf["mu0_hat"]
    is_post = cf["date"] >= treat_start
    cf["cum_effect"] = (cf["effect"].where(is_post, 0.0)).cumsum()
    cf_path = cf_dir / f"{ep_id}_cf.parquet"
    cf.to_parquet(cf_path, index=False)

    # Placebos
    plac_dir = Path(cfg.out_dir) / "placebos"
    _ensure_dir(plac_dir)
    pval_space = np.nan
    if cfg.do_placebo_space:
        ep_rows = dfp.loc[dfp["date"].between(ep_row["pre_start"], post_end, inclusive="both")].copy()
        ps = placebo_space_meta(ep_rows, donors_df, (j_store, j_item), treat_start, post_end)
        ps["episode_id"] = ep_id
        ps_path = plac_dir / f"{ep_id}_space.parquet"
        ps.to_parquet(ps_path, index=False)
        if not ps.empty and np.isfinite(att_sum):
            pval_space = (1 + np.sum(np.abs(ps["att_placebo_sum"].to_numpy()) >= abs(att_sum))) / (1 + ps.shape[0])
    else:
        ps_path = None

    if cfg.do_placebo_time:
        pt = placebo_time_victim(ep_rows if cfg.do_placebo_space else dfp, treat_start, post_end, victim_uid)
        pt["episode_id"] = ep_id
        pt_path = plac_dir / f"{ep_id}_time.parquet"
        pt.to_parquet(pt_path, index=False)
    else:
        pt_path = None

    # ===== MÉTRICAS CAUSALES COMPARATIVAS =====
    causal_metrics_report = None
    if CAUSAL_METRICS_AVAILABLE:
        try:
            calc = CausalMetricsCalculator(ep_id, f"meta-{cfg.learner}")
            
            # Cargar placebos si existen
            ps_df = pd.read_parquet(ps_path) if ps_path and Path(ps_path).exists() else None
            pt_df = pd.read_parquet(pt_path) if pt_path and Path(pt_path).exists() else None
            
            # Para meta-learners, no tenemos análisis de sensibilidad por defecto
            # pero podríamos agregarlo en el futuro
            
            # Calcular métricas
            causal_metrics_report = calc.compute_all(
                y_obs_pre=y_obs_pre,
                y_hat_pre=y_hat_pre,
                tau_post=effect_post,
                att_base=att_sum,
                placebo_space_df=ps_df,
                placebo_time_df=pt_df,
                n_control_units=0  # Meta-learners no tienen concepto directo de unidades control
            )
            
            # Guardar reporte de métricas causales
            causal_dir = Path(cfg.out_dir) / "causal_metrics"; _ensure_dir(causal_dir)
            causal_path = causal_dir / f"{ep_id}_causal.parquet"
            pd.DataFrame([causal_metrics_report.to_flat_dict()]).to_parquet(causal_path, index=False)
            
        except Exception as e:
            logging.warning(f"[{ep_id}] Error calculando métricas causales: {e}")

    # Reporte JSON
    rep_dir = Path(cfg.out_dir) / "reports"; _ensure_dir(rep_dir)
    rep = {
        "episode_id": ep_id,
        "victim_unit": victim_uid,
        "window": {"pre_start": str(ep_row["pre_start"]), "treat_start": str(treat_start), "post_end": str(post_end)},
        "learner": cfg.learner,
        "model": cfg.model,
        "cv": {"folds": cfg.cv_folds, "holdout_days": cfg.cv_holdout_days, "gap_days": _CV_GAP_DAYS},
        "fit": {"rmspe_pre": rmspe_pre, "mae_pre": mae_pre, "att_mean": att_mean, "att_sum": att_sum, "rel_att_vs_pre_mean": rel_att},
        "p_value_placebo_space": pval_space,
        "paths": {"cf": str(cf_path), "placebo_space": str(ps_path) if cfg.do_placebo_space else None,
                  "placebo_time": str(pt_path) if cfg.do_placebo_time else None},
        "calibration": {"a": a_cal, "b": b_cal}
    }
    with open(rep_dir / f"{ep_id}.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # Resumen para meta_metrics_<learner>.parquet
    n_pre_days = int(pd.Series(dfp.loc[vpre, "date"]).nunique())
    n_post_days = int(pd.Series(dfp.loc[vpost, "date"]).nunique())
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
        "p_value_placebo_space": pval_space
    }
    return summary

def run_batch(cfg: RunCfg) -> None:
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    # Salidas
    if cfg.out_dir is None:
        cfg.out_dir = _default_out_dir(cfg)
    _ensure_dir(Path(cfg.out_dir))
    _ensure_dir(Path(cfg.out_dir) / "cf_series")
    _ensure_dir(Path(cfg.out_dir) / "placebos")
    _ensure_dir(Path(cfg.out_dir) / "reports")

    # Cargar insumos
    meta_df = _load_meta_df(Path(cfg.meta_parquet))
    episodes = _load_episodes(Path(cfg.episodes_index))

    # Fallbacks a figures/<exp_tag>/tables si falta algún insumo
    if (episodes is None or episodes.empty) and cfg.exp_tag:
        alt_ep = Path("figures") / cfg.exp_tag / "tables" / "episodes_index.parquet"
        if alt_ep.exists():
            episodes = _load_episodes(alt_ep)

    donors = _load_donors(Path(cfg.donors_csv))
    if (donors is None or donors.empty) and cfg.exp_tag:
        alt_dn = Path("figures") / cfg.exp_tag / "tables" / "donors_per_victim.csv"
        if alt_dn.exists():
            donors = _load_donors(alt_dn)

    # Selección de features (excluye columnas de tratamiento para evitar leakage)
    feats = select_feature_cols(meta_df, extra_exclude=[cfg.treat_col_s, cfg.treat_col_b])
    logging.info(f"Features seleccionadas ({len(feats)}): {feats[:8]}{' ...' if len(feats) > 8 else ''}")

    # Orden base
    meta_df = meta_df.sort_values(["date", "unit_id"]).reset_index(drop=True)

    # Filtrar episodios si se solicita
    if cfg.max_episodes is not None:
        episodes = episodes.head(int(cfg.max_episodes)).copy()

    # --------------------- HPO (Optuna) ---------------------
    if int(cfg.hpo_trials) > 0 and cfg.model.lower() in {"lgbm", "hgbt"}:
        logging.info(f"Iniciando HPO (Optuna) para modelo '{cfg.model}' con {int(cfg.hpo_trials)} trials...")
        reg_path = _hpo_registry_path(cfg.exp_tag, cfg.learner, cfg.model)
        warm = {}
        if reg_path and reg_path.exists():
            prev = _load_json_silent(reg_path)
            if isinstance(prev, dict):
                warm = {k: prev[k] for k in prev.keys()}
        best = tune_hyperparams(
            meta_df, feats, cfg.model, cfg.random_state, cfg.cv_folds, cfg.cv_holdout_days,
            {"max_iter": cfg.max_iter, "hpo_trials": cfg.hpo_trials, "hpo_warm_params": warm}
        )
        if best:
            for k, v in best.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            logging.info(f"[HPO] Hiperparámetros aplicados: {best}")
            if reg_path is not None:
                _save_json_silent(reg_path, best)
        else:
            logging.info("HPO omitido o sin mejoras.")
    else:
        if int(cfg.hpo_trials) <= 0:
            logging.info("HPO desactivado (hpo_trials=0).")

    # Cross-fitting global (cache)
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

    # Iterar episodios (set pequeño)
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
        out_path = Path(cfg.out_dir) / f"meta_metrics_{cfg.learner.lower()}.parquet"
        mdf.to_parquet(out_path, index=False)
        logging.info(f"Métricas globales guardadas: {out_path} ({mdf.shape[0]} episodios)")
        # espejo útil para EDA en figures/<exp_tag>/tables
        tables_dir = Path("figures") / cfg.exp_tag / "tables"
        _ensure_dir(tables_dir)
        try:
            mdf.to_parquet(tables_dir / f"meta_metrics_{cfg.learner.lower()}.parquet", index=False)
        except Exception:
            pass
    else:
        logging.warning("No se generaron métricas (¿sin episodios válidos o fallos previos?).")

# =============================================================================
# CLI
# =============================================================================

def parse_args() -> RunCfg:
    p = argparse.ArgumentParser(
        description=(
            "T/S/X-learners (time-series aware) para canibalización en retail.\n"
            "Entrena SIEMPRE con --meta_parquet (dataset grande) y evalúa SOLO "
            "sobre --episodes_index (set pequeño usado por GSC)."
        )
    )

    p.add_argument("--learner", type=str, default="x", choices=["t", "s", "x"], help="Tipo de meta-learner.")
    p.add_argument("--meta_parquet", type=str, default=str(_data_root() / "processed_meta" / "windows.parquet"))
    p.add_argument("--episodes_index", type=str, default=str(_data_root() / "processed_data" / "episodes_index.parquet"))
    p.add_argument("--donors_csv", type=str, default=str(_data_root() / "processed_data" / "A_base" / "donors_per_victim.csv"))
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--exp_tag", type=str, default="A_base")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--max_episodes", type=int, default=None)

    # CV temporal
    p.add_argument("--cv_folds", type=int, default=3)
    p.add_argument("--cv_holdout", type=int, default=21)
    p.add_argument("--min_train_samples", type=int, default=50)

    # modelo base
    p.add_argument("--model", type=str, default="lgbm", choices=["lgbm", "hgbt", "rf", "ridge"])
    p.add_argument("--prop_model", type=str, default="logit")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_depth", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--max_iter", type=int, default=600)
    p.add_argument("--min_samples_leaf", type=int, default=10)
    p.add_argument("--l2", type=float, default=0.0)

    # LightGBM
    p.add_argument("--num_leaves", type=int, default=127)
    p.add_argument("--min_child_samples", type=int, default=10)
    p.add_argument("--feature_fraction", type=float, default=0.8)
    p.add_argument("--subsample", type=float, default=0.8)

    # HPO
    p.add_argument("--hpo_trials", type=int, default=10, help="0 para desactivar.")

    # diagnósticos
    p.add_argument("--do_placebo_space", action="store_true")
    p.add_argument("--do_placebo_time", action="store_true")
    p.add_argument("--do_loo", action="store_true")
    p.add_argument("--max_loo", type=int, default=5)
    p.add_argument("--sens_samples", type=int, default=0)

    # Tratamientos
    p.add_argument("--treat_col_s", type=str, default="D")
    p.add_argument("--s_ref", type=float, default=0.0)
    p.add_argument("--treat_col_b", type=str, default="D")
    p.add_argument("--bin_threshold", type=float, default=0.0)

    a = p.parse_args()
    return RunCfg(
        learner=a.learner,
        meta_parquet=Path(a.meta_parquet),
        episodes_index=Path(a.episodes_index),
        donors_csv=Path(a.donors_csv),
        out_dir=(Path(a.out_dir) if a.out_dir else None),
        exp_tag=a.exp_tag,
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
        num_leaves=a.num_leaves,
        min_child_samples=a.min_child_samples,
        feature_fraction=a.feature_fraction,
        subsample=a.subsample,
        hpo_trials=a.hpo_trials,
        do_placebo_space=a.do_placebo_space,
        do_placebo_time=a.do_placebo_time,
        do_loo=a.do_loo,
        max_loo=a.max_loo,
        sens_samples=a.sens_samples,
        treat_col_s=a.treat_col_s,
        s_ref=a.s_ref,
        treat_col_b=a.treat_col_b,
        bin_threshold=a.bin_threshold,
    )

if __name__ == "__main__":
    cfg = parse_args()
    run_batch(cfg)