from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class CausalLearnResult:
    """Resultado común para T-learner / X-learner.

    Alineado con los campos más usados para comparar con GSC.
    - att: promedio de efectos en celdas tratadas post
    - att_t: efecto promedio por tiempo (solo tiempos post)
    - att_i: efecto promedio por unidad tratada (solo periodos post por unidad)
    - y0_hat: matriz id x time con Y(0) imputado en tratadas post (NaN en otras celdas)
    - tau_it: matriz id x time con efecto observado - contrafactual (NaN fuera de tratadas post)
    - metrics: dict con diagnósticos (ej. pre_rmse, n_units, n_times, n_treated)
    - models: dict con modelos entrenados, útil para depurar/reproducir
    """

    att: float
    att_t: pd.Series
    att_i: pd.Series
    y0_hat: pd.DataFrame
    tau_it: pd.DataFrame
    metrics: Dict[str, Any]
    models: Dict[str, Any]


# ------------------------------
# Helpers de preparación de datos
# ------------------------------

def _sorted_unique(values: Iterable[Any]) -> List[Any]:
    return sorted(pd.unique(pd.Series(list(values))).tolist())


def _derive_treat_from_adoption(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    adoption_time_col: str,
    out_treat_col: str = "__D__",
) -> pd.DataFrame:
    d = df.copy()
    adop = (
        d[[id_col, adoption_time_col]]
        .drop_duplicates(subset=[id_col])
        .set_index(id_col)[adoption_time_col]
    )
    d = d.merge(adop.rename("__adopt__"), how="left", left_on=id_col, right_index=True)
    d[out_treat_col] = (d[time_col] >= d["__adopt__"]) & d["__adopt__"].notna()
    d.drop(columns=["__adopt__"], inplace=True)
    return d


def _ensure_periods(
    df: pd.DataFrame,
    time_col: str,
    adoption_time_col: Optional[str],
    pre_period: Optional[Tuple[Any, Any]],
    post_period: Optional[Tuple[Any, Any]],
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
    """Retorna (pre_period, post_period). Si no se pasan, intenta derivar de adoption_time_col.
    Regla: pre = t < min(t0_i), post = t >= min(t0_i).
    """
    times = _sorted_unique(df[time_col])
    if pre_period is not None and post_period is not None:
        return pre_period, post_period
    if adoption_time_col is None or adoption_time_col not in df.columns:
        raise ValueError("Si no pasas pre/post period, necesitas 'adoption_time_col' en df.")
    adop = df[[adoption_time_col]].dropna()
    if adop.empty:
        raise ValueError("'adoption_time_col' no tiene adopciones válidas para definir post.")
    t0_min = adop[adoption_time_col].min()
    pre_times = [t for t in times if t < t0_min]
    post_times = [t for t in times if t >= t0_min]
    if not post_times:
        raise ValueError("No hay tiempos 'post' detectados.")
    pre_period = (pre_times[0], pre_times[-1]) if pre_times else (times[0], times[0])
    post_period = (post_times[0], post_times[-1])
    return pre_period, post_period


def _one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica one-hot a columnas object/category; deja el resto como está."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return df.copy()
    return pd.get_dummies(df, columns=cat_cols, drop_first=False)


def _wide_frames(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    value: pd.Series,
    ids_order: Optional[Sequence[Any]] = None,
    times_order: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """Convierte una serie alineada a df en matriz ancha id x time."""
    tmp = pd.DataFrame({id_col: df[id_col].values, time_col: df[time_col].values, "val": value.values})
    wide = tmp.pivot(index=id_col, columns=time_col, values="val")
    if ids_order is not None:
        wide = wide.reindex(index=list(ids_order))
    if times_order is not None:
        wide = wide.reindex(columns=list(times_order))
    return wide


def _compute_att_objects(
    tau_wide: pd.DataFrame,
    d_wide: pd.DataFrame,
    time_col: str,
    post_period: Tuple[Any, Any],
) -> Tuple[float, pd.Series, pd.Series]:
    post_start, post_end = post_period
    post_cols_mask = [post_start <= t <= post_end for t in tau_wide.columns]
    post_cols = [t for t, m in zip(tau_wide.columns, post_cols_mask) if m]

    # Mask tratadas post
    treated_post = (d_wide.loc[:, post_cols] == 1)
    tau_post = tau_wide.loc[:, post_cols]

    # ATT global
    att = float(np.nanmean(tau_post.values[treated_post.values])) if treated_post.any().any() else np.nan

    # ATT por tiempo
    att_t_vals = []
    att_t_idx = []
    for t in post_cols:
        mask = treated_post[t]
        if mask.any():
            att_t_idx.append(t)
            att_t_vals.append(float(np.nanmean(tau_post.loc[mask, t])))
    att_t = pd.Series(att_t_vals, index=pd.Index(att_t_idx, name=time_col), name="att_t")

    # ATT por unidad
    att_i_vals = []
    att_i_idx = []
    for uid, row_mask in treated_post.iterrows():
        if row_mask.any():
            effs = tau_post.loc[uid, row_mask[row_mask].index]
            att_i_idx.append(uid)
            att_i_vals.append(float(np.nanmean(effs.values)))
    att_i = pd.Series(att_i_vals, index=pd.Index(att_i_idx, name=tau_wide.index.name or "id"), name="att_i")

    return att, att_t, att_i


def _build_masks_and_orders(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    treat_col: Optional[str],
    adoption_time_col: Optional[str],
    pre_period: Tuple[Any, Any],
    post_period: Tuple[Any, Any],
) -> Tuple[pd.Series, pd.Series, List[Any], List[Any]]:
    """Devuelve:
    - D: serie binaria 0/1 alineada a df
    - treated_post_mask: serie booleana alineada a df
    - ids_order, times_order: órdenes globales para matrices anchas
    """
    if treat_col is None or treat_col not in df.columns:
        if adoption_time_col is None or adoption_time_col not in df.columns:
            raise ValueError("Necesitas 'treat_col' o 'adoption_time_col'.")
        d2 = _derive_treat_from_adoption(df, id_col, time_col, adoption_time_col, out_treat_col="__D__")
        D = d2["__D__"].astype(int)
    else:
        D = df[treat_col].astype(int)

    pre_start, pre_end = pre_period
    post_start, post_end = post_period
    treated_post_mask = (D == 1) & df[time_col].apply(lambda t: (post_start <= t <= post_end))

    ids_order = _sorted_unique(df[id_col])
    times_order = _sorted_unique(df[time_col])
    return D, treated_post_mask, ids_order, times_order


def _pre_rmse_for_m0(
    y_true: pd.Series,
    y_pred: pd.Series,
    df: pd.DataFrame,
    time_col: str,
    pre_period: Tuple[Any, Any],
    d_series: pd.Series,
) -> float:
    pre_start, pre_end = pre_period
    mask_pre = df[time_col].apply(lambda t: (pre_start <= t <= pre_end))
    mask_eval = mask_pre & (d_series == 0)
    if mask_eval.any():
        err = (y_pred[mask_eval] - y_true[mask_eval]) ** 2
        return float(np.sqrt(np.nanmean(err)))
    return float("nan")


# ------------------------------------
# Implementación con LightGBM
# ------------------------------------

class LightGBMLearner:
    """Implementa T-learner y X-learner usando LightGBM.

    Notas:
    - Se asume que las columnas en `features` son numéricas o convertibles vía one-hot.
    - Para evitar fugas, Y(0) se imputa sólo en celdas tratadas post; m0 se entrena con D=0.
    - Si LightGBM no está instalado, se levanta un ImportError con mensaje claro.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _import(self):
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Falta instalar 'lightgbm'. Agrega 'lightgbm' a pyproject.toml y reinstala."
            ) from e
        return LGBMRegressor, LGBMClassifier

    def t_learner(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        outcome_col: str,
        features: List[str],
        treat_col: Optional[str] = None,
        adoption_time_col: Optional[str] = None,
        pre_period: Optional[Tuple[Any, Any]] = None,
        post_period: Optional[Tuple[Any, Any]] = None,
        lgbm_params_m0: Optional[Dict[str, Any]] = None,
    ) -> CausalLearnResult:
        LGBMRegressor, _ = self._import()

        # Validaciones básicas
        needed = {id_col, time_col, outcome_col}
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        for c in features:
            if c not in df.columns:
                raise ValueError(f"Feature faltante en df: {c}")

        d = df.copy()
        pre_period, post_period = _ensure_periods(d, time_col, adoption_time_col, pre_period, post_period)
        D, treated_post_mask, ids_order, times_order = _build_masks_and_orders(
            d, id_col, time_col, treat_col, adoption_time_col, pre_period, post_period
        )

        y = d[outcome_col].astype(float)
        X = _one_hot(d[features])

        # Entrena m0 con D=0
        params_m0 = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
        }
        if lgbm_params_m0:
            params_m0.update(lgbm_params_m0)
        m0 = LGBMRegressor(**params_m0)
        m0.fit(X[D == 0], y[D == 0])

        # Predicciones sólo en tratadas post
        y0_pred_series = pd.Series(np.nan, index=d.index, dtype=float)
        idx_tp = treated_post_mask[treated_post_mask].index
        if len(idx_tp) == 0:
            raise ValueError("No hay celdas tratadas en periodo post para imputar contrafactual.")
        y0_pred_series.loc[idx_tp] = m0.predict(X.loc[idx_tp])

        # Efectos y outputs anchos
        tau_series = pd.Series(np.nan, index=d.index, dtype=float)
        tau_series.loc[idx_tp] = y.loc[idx_tp] - y0_pred_series.loc[idx_tp]

        y0_wide = _wide_frames(d, id_col, time_col, y0_pred_series, ids_order, times_order)
        tau_wide = _wide_frames(d, id_col, time_col, tau_series, ids_order, times_order)

        # Construye D en wide para métricas/ATT
        d_wide = _wide_frames(d, id_col, time_col, D.astype(int), ids_order, times_order).fillna(0).astype(int)

        att, att_t, att_i = _compute_att_objects(tau_wide, d_wide, time_col, post_period)

        pre_rmse = _pre_rmse_for_m0(
            y_true=y,
            y_pred=pd.Series(m0.predict(X), index=d.index, dtype=float),
            df=d,
            time_col=time_col,
            pre_period=pre_period,
            d_series=D,
        )

        metrics: Dict[str, Any] = {
            "pre_rmse": pre_rmse,
            "n_units": int(len(ids_order)),
            "n_times": int(len(times_order)),
            "n_treated": int((d.groupby(id_col)[D.name if D.name else treat_col or "D"].max()).sum()) if D.name in d.columns else int((d_wide.max(axis=1) > 0).sum()),
            "learner": "T",
            "library": "lightgbm",
        }
        models = {"m0": m0}

        return CausalLearnResult(
            att=att,
            att_t=att_t,
            att_i=att_i,
            y0_hat=y0_wide,
            tau_it=tau_wide,
            metrics=metrics,
            models=models,
        )

    def x_learner(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        outcome_col: str,
        features: List[str],
        treat_col: Optional[str] = None,
        adoption_time_col: Optional[str] = None,
        pre_period: Optional[Tuple[Any, Any]] = None,
        post_period: Optional[Tuple[Any, Any]] = None,
        lgbm_params_outcome: Optional[Dict[str, Any]] = None,
        lgbm_params_propensity: Optional[Dict[str, Any]] = None,
        lgbm_params_tau: Optional[Dict[str, Any]] = None,
    ) -> CausalLearnResult:
        LGBMRegressor, LGBMClassifier = self._import()

        # Validaciones
        needed = {id_col, time_col, outcome_col}
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        for c in features:
            if c not in df.columns:
                raise ValueError(f"Feature faltante en df: {c}")

        d = df.copy()
        pre_period, post_period = _ensure_periods(d, time_col, adoption_time_col, pre_period, post_period)
        D, treated_post_mask, ids_order, times_order = _build_masks_and_orders(
            d, id_col, time_col, treat_col, adoption_time_col, pre_period, post_period
        )

        y = d[outcome_col].astype(float)
        X = _one_hot(d[features])

        # Outcome models m0 y m1
        params_out = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
        }
        if lgbm_params_outcome:
            params_out.update(lgbm_params_outcome)
        m0 = LGBMRegressor(**params_out)
        m1 = LGBMRegressor(**params_out)
        m0.fit(X[D == 0], y[D == 0])
        if (D == 1).any():
            m1.fit(X[D == 1], y[D == 1])
        else:
            # Sin datos tratados para entrenar m1: duplicamos m0 como fallback
            m1 = m0

        # Propensity e(x)
        params_clf = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
        }
        if lgbm_params_propensity:
            params_clf.update(lgbm_params_propensity)
        e_clf = LGBMClassifier(**params_clf)
        e_clf.fit(X, D)

        # Pseudo-outcomes
        m0_all = pd.Series(m0.predict(X), index=d.index, dtype=float)
        m1_all = pd.Series(m1.predict(X), index=d.index, dtype=float)
        e_all = pd.Series(np.clip(e_clf.predict_proba(X)[:, 1], 1e-3, 1 - 1e-3), index=d.index, dtype=float)

        D1 = pd.Series(np.nan, index=d.index, dtype=float)
        D0 = pd.Series(np.nan, index=d.index, dtype=float)
        D1[D == 1] = y[D == 1] - m0_all[D == 1]
        D0[D == 0] = m1_all[D == 0] - y[D == 0]

        # Tau models g1 (en tratados) y g0 (en controles)
        params_tau = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
        }
        if lgbm_params_tau:
            params_tau.update(lgbm_params_tau)
        g1 = LGBMRegressor(**params_tau)
        g0 = LGBMRegressor(**params_tau)
        if (D == 1).any():
            g1.fit(X[D == 1], D1[D == 1])
        if (D == 0).any():
            g0.fit(X[D == 0], D0[D == 0])

        # Estima tau en tratadas post (mezcla de X-learner)
        idx_tp = treated_post_mask[treated_post_mask].index
        tau_series = pd.Series(np.nan, index=d.index, dtype=float)
        if len(idx_tp) == 0:
            raise ValueError("No hay celdas tratadas en periodo post para estimar efectos.")

        tau_g1 = pd.Series(0.0, index=d.index, dtype=float)
        tau_g0 = pd.Series(0.0, index=d.index, dtype=float)
        if hasattr(g1, "predict") and (D == 1).any():
            tau_g1.loc[idx_tp] = g1.predict(X.loc[idx_tp])
        if hasattr(g0, "predict") and (D == 0).any():
            tau_g0.loc[idx_tp] = g0.predict(X.loc[idx_tp])

        tau_series.loc[idx_tp] = (1.0 - e_all.loc[idx_tp]) * tau_g1.loc[idx_tp] + e_all.loc[idx_tp] * tau_g0.loc[idx_tp]

        # y0_hat = Y - tau en tratadas post
        y0_pred_series = pd.Series(np.nan, index=d.index, dtype=float)
        y0_pred_series.loc[idx_tp] = y.loc[idx_tp] - tau_series.loc[idx_tp]

        # Salidas anchas
        y0_wide = _wide_frames(d, id_col, time_col, y0_pred_series, ids_order, times_order)
        tau_wide = _wide_frames(d, id_col, time_col, tau_series, ids_order, times_order)
        d_wide = _wide_frames(d, id_col, time_col, D.astype(int), ids_order, times_order).fillna(0).astype(int)

        att, att_t, att_i = _compute_att_objects(tau_wide, d_wide, time_col, post_period)

        pre_rmse = _pre_rmse_for_m0(
            y_true=y,
            y_pred=m0_all,
            df=d,
            time_col=time_col,
            pre_period=pre_period,
            d_series=D,
        )

        metrics: Dict[str, Any] = {
            "pre_rmse": pre_rmse,
            "n_units": int(len(ids_order)),
            "n_times": int(len(times_order)),
            "n_treated": int((d_wide.max(axis=1) > 0).sum()),
            "learner": "X",
            "library": "lightgbm",
        }
        models = {"m0": m0, "m1": m1, "e": e_clf, "g0": g0, "g1": g1}

        return CausalLearnResult(
            att=att,
            att_t=att_t,
            att_i=att_i,
            y0_hat=y0_wide,
            tau_it=tau_wide,
            metrics=metrics,
            models=models,
        )


# ------------------------------------
# Implementación con CatBoost
# ------------------------------------

class CatBoostLearner:
    """Implementa T-learner y X-learner usando CatBoost.

    Notas:
    - Se usa one-hot simple para asegurar entrada numérica.
    - Si CatBoost no está instalado, se levanta un ImportError con mensaje claro.
    """

    def __init__(self, random_seed: int = 42, verbose: bool = False):
        self.random_seed = random_seed
        self.verbose = verbose

    def _import(self):
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Falta instalar 'catboost'. Agrega 'catboost' a pyproject.toml y reinstala."
            ) from e
        return CatBoostRegressor, CatBoostClassifier

    def t_learner(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        outcome_col: str,
        features: List[str],
        treat_col: Optional[str] = None,
        adoption_time_col: Optional[str] = None,
        pre_period: Optional[Tuple[Any, Any]] = None,
        post_period: Optional[Tuple[Any, Any]] = None,
        cb_params_m0: Optional[Dict[str, Any]] = None,
    ) -> CausalLearnResult:
        CatBoostRegressor, _ = self._import()

        needed = {id_col, time_col, outcome_col}
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        for c in features:
            if c not in df.columns:
                raise ValueError(f"Feature faltante en df: {c}")

        d = df.copy()
        pre_period, post_period = _ensure_periods(d, time_col, adoption_time_col, pre_period, post_period)
        D, treated_post_mask, ids_order, times_order = _build_masks_and_orders(
            d, id_col, time_col, treat_col, adoption_time_col, pre_period, post_period
        )

        y = d[outcome_col].astype(float)
        X = _one_hot(d[features])

        params_m0 = {
            "iterations": 800,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "loss_function": "RMSE",
        }
        if cb_params_m0:
            params_m0.update(cb_params_m0)
        m0 = CatBoostRegressor(**params_m0)
        m0.fit(X[D == 0], y[D == 0])

        # Predicciones en tratadas post
        y0_pred_series = pd.Series(np.nan, index=d.index, dtype=float)
        idx_tp = treated_post_mask[treated_post_mask].index
        if len(idx_tp) == 0:
            raise ValueError("No hay celdas tratadas en periodo post para imputar contrafactual.")
        y0_pred_series.loc[idx_tp] = m0.predict(X.loc[idx_tp])

        tau_series = pd.Series(np.nan, index=d.index, dtype=float)
        tau_series.loc[idx_tp] = y.loc[idx_tp] - y0_pred_series.loc[idx_tp]

        y0_wide = _wide_frames(d, id_col, time_col, y0_pred_series, ids_order, times_order)
        tau_wide = _wide_frames(d, id_col, time_col, tau_series, ids_order, times_order)
        d_wide = _wide_frames(d, id_col, time_col, D.astype(int), ids_order, times_order).fillna(0).astype(int)

        att, att_t, att_i = _compute_att_objects(tau_wide, d_wide, time_col, post_period)

        pre_rmse = _pre_rmse_for_m0(
            y_true=y,
            y_pred=pd.Series(m0.predict(X), index=d.index, dtype=float),
            df=d,
            time_col=time_col,
            pre_period=pre_period,
            d_series=D,
        )

        metrics: Dict[str, Any] = {
            "pre_rmse": pre_rmse,
            "n_units": int(len(ids_order)),
            "n_times": int(len(times_order)),
            "n_treated": int((d_wide.max(axis=1) > 0).sum()),
            "learner": "T",
            "library": "catboost",
        }
        models = {"m0": m0}

        return CausalLearnResult(
            att=att,
            att_t=att_t,
            att_i=att_i,
            y0_hat=y0_wide,
            tau_it=tau_wide,
            metrics=metrics,
            models=models,
        )

    def x_learner(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        outcome_col: str,
        features: List[str],
        treat_col: Optional[str] = None,
        adoption_time_col: Optional[str] = None,
        pre_period: Optional[Tuple[Any, Any]] = None,
        post_period: Optional[Tuple[Any, Any]] = None,
        cb_params_outcome: Optional[Dict[str, Any]] = None,
        cb_params_propensity: Optional[Dict[str, Any]] = None,
        cb_params_tau: Optional[Dict[str, Any]] = None,
    ) -> CausalLearnResult:
        CatBoostRegressor, CatBoostClassifier = self._import()

        needed = {id_col, time_col, outcome_col}
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        for c in features:
            if c not in df.columns:
                raise ValueError(f"Feature faltante en df: {c}")

        d = df.copy()
        pre_period, post_period = _ensure_periods(d, time_col, adoption_time_col, pre_period, post_period)
        D, treated_post_mask, ids_order, times_order = _build_masks_and_orders(
            d, id_col, time_col, treat_col, adoption_time_col, pre_period, post_period
        )

        y = d[outcome_col].astype(float)
        X = _one_hot(d[features])

        params_out = {
            "iterations": 800,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "loss_function": "RMSE",
        }
        if cb_params_outcome:
            params_out.update(cb_params_outcome)
        m0 = CatBoostRegressor(**params_out)
        m1 = CatBoostRegressor(**params_out)
        m0.fit(X[D == 0], y[D == 0])
        if (D == 1).any():
            m1.fit(X[D == 1], y[D == 1])
        else:
            m1 = m0

        # Propensity
        params_clf = {
            "iterations": 600,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "loss_function": "Logloss",
        }
        if cb_params_propensity:
            params_clf.update(cb_params_propensity)
        e_clf = CatBoostClassifier(**params_clf)
        e_clf.fit(X, D)

        m0_all = pd.Series(m0.predict(X), index=d.index, dtype=float)
        m1_all = pd.Series(m1.predict(X), index=d.index, dtype=float)
        e_all = pd.Series(np.clip(e_clf.predict_proba(X)[:, 1], 1e-3, 1 - 1e-3), index=d.index, dtype=float)

        D1 = pd.Series(np.nan, index=d.index, dtype=float)
        D0 = pd.Series(np.nan, index=d.index, dtype=float)
        D1[D == 1] = y[D == 1] - m0_all[D == 1]
        D0[D == 0] = m1_all[D == 0] - y[D == 0]

        params_tau = {
            "iterations": 600,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "loss_function": "RMSE",
        }
        if cb_params_tau:
            params_tau.update(cb_params_tau)
        g1 = CatBoostRegressor(**params_tau)
        g0 = CatBoostRegressor(**params_tau)
        if (D == 1).any():
            g1.fit(X[D == 1], D1[D == 1])
        if (D == 0).any():
            g0.fit(X[D == 0], D0[D == 0])

        idx_tp = treated_post_mask[treated_post_mask].index
        if len(idx_tp) == 0:
            raise ValueError("No hay celdas tratadas en periodo post para estimar efectos.")

        tau_series = pd.Series(np.nan, index=d.index, dtype=float)
        tau_g1 = pd.Series(0.0, index=d.index, dtype=float)
        tau_g0 = pd.Series(0.0, index=d.index, dtype=float)
        if hasattr(g1, "predict") and (D == 1).any():
            tau_g1.loc[idx_tp] = g1.predict(X.loc[idx_tp])
        if hasattr(g0, "predict") and (D == 0).any():
            tau_g0.loc[idx_tp] = g0.predict(X.loc[idx_tp])
        tau_series.loc[idx_tp] = (1.0 - e_all.loc[idx_tp]) * tau_g1.loc[idx_tp] + e_all.loc[idx_tp] * tau_g0.loc[idx_tp]

        y0_pred_series = pd.Series(np.nan, index=d.index, dtype=float)
        y0_pred_series.loc[idx_tp] = y.loc[idx_tp] - tau_series.loc[idx_tp]

        y0_wide = _wide_frames(d, id_col, time_col, y0_pred_series, ids_order, times_order)
        tau_wide = _wide_frames(d, id_col, time_col, tau_series, ids_order, times_order)
        d_wide = _wide_frames(d, id_col, time_col, D.astype(int), ids_order, times_order).fillna(0).astype(int)

        att, att_t, att_i = _compute_att_objects(tau_wide, d_wide, time_col, post_period)

        pre_rmse = _pre_rmse_for_m0(
            y_true=y,
            y_pred=m0_all,
            df=d,
            time_col=time_col,
            pre_period=pre_period,
            d_series=D,
        )

        metrics: Dict[str, Any] = {
            "pre_rmse": pre_rmse,
            "n_units": int(len(ids_order)),
            "n_times": int(len(times_order)),
            "n_treated": int((d_wide.max(axis=1) > 0).sum()),
            "learner": "X",
            "library": "catboost",
        }
        models = {"m0": m0, "m1": m1, "e": e_clf, "g0": g0, "g1": g1}

        return CausalLearnResult(
            att=att,
            att_t=att_t,
            att_i=att_i,
            y0_hat=y0_wide,
            tau_it=tau_wide,
            metrics=metrics,
            models=models,
        )


__all__ = [
    "CausalLearnResult",
    "LightGBMLearner",
    "CatBoostLearner",
]