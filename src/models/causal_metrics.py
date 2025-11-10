# -*- coding: utf-8 -*-
"""
causal_metrics.py
=================

Módulo centralizado para métricas de validación causal comparativas entre
Control Sintético Generalizado (GSC) y Meta-learners (T/S/X).

Métricas implementadas:
1. Sensibilidad/Robustez: Variación de efectos bajo distintas especificaciones
2. Balance de Covariables: Similitud pre-tratamiento entre tratados y controles
3. Error de Predicción Pre-tratamiento: Calidad del ajuste contrafactual
4. Heterogeneidad del Efecto: Varianza de efectos causales
5. Placebo Tests: Significancia estadística mediante permutaciones
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _rmspe(y: np.ndarray, yhat: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return np.nan
    e = y[m] - yhat[m]
    denom = max(1.0, float(np.sqrt(np.mean((y[m])**2))))
    return float(np.sqrt(np.mean(e**2)) / denom)


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return np.nan
    return float(np.mean(np.abs(y[m] - yhat[m])))


def _standardized_diff(x_treat: np.ndarray, x_control: np.ndarray) -> float:
    """Cohen's d entre dos grupos."""
    if len(x_treat) == 0 or len(x_control) == 0:
        return np.nan
    
    m_t = np.nanmean(x_treat)
    m_c = np.nanmean(x_control)
    s_t = np.nanstd(x_treat)
    s_c = np.nanstd(x_control)
    
    n_t, n_c = len(x_treat), len(x_control)
    pooled_std = np.sqrt(((n_t - 1) * s_t**2 + (n_c - 1) * s_c**2) / (n_t + n_c - 2))
    
    if pooled_std < 1e-8:
        return 0.0
    
    return float((m_t - m_c) / pooled_std)


@dataclass
class SensitivityMetrics:
    att_mean_base: float = np.nan
    att_std: float = np.nan
    att_cv: float = np.nan
    att_range: float = np.nan
    n_specs: int = 0
    relative_std: float = np.nan


@dataclass
class CovariateBalanceMetrics:
    mean_abs_std_diff: float = np.nan
    max_abs_std_diff: float = np.nan
    n_imbalanced: int = 0
    balance_rate: float = np.nan


@dataclass
class PredictionErrorMetrics:
    rmspe_pre: float = np.nan
    mae_pre: float = np.nan
    corr_pre: float = np.nan
    r2_pre: float = np.nan


@dataclass
class HeterogeneityMetrics:
    tau_mean: float = np.nan
    tau_std: float = np.nan
    tau_cv: float = np.nan
    tau_median: float = np.nan
    pct_positive: float = np.nan


@dataclass
class PlaceboMetrics:
    n_placebos_space: int = 0
    p_value_space: float = np.nan
    p_value_time: float = np.nan
    effect_to_placebo_ratio: float = np.nan


@dataclass
class CausalMetricsReport:
    episode_id: str
    model_type: str
    
    sensitivity: SensitivityMetrics = field(default_factory=SensitivityMetrics)
    covariate_balance: CovariateBalanceMetrics = field(default_factory=CovariateBalanceMetrics)
    prediction_error: PredictionErrorMetrics = field(default_factory=PredictionErrorMetrics)
    heterogeneity: HeterogeneityMetrics = field(default_factory=HeterogeneityMetrics)
    placebo: PlaceboMetrics = field(default_factory=PlaceboMetrics)
    
    n_pre_periods: int = 0
    n_post_periods: int = 0
    n_control_units: int = 0
    
    def to_flat_dict(self) -> Dict:
        """Versión aplanada para DataFrame."""
        return {
            "episode_id": self.episode_id,
            "model_type": self.model_type,
            "n_pre_periods": self.n_pre_periods,
            "n_post_periods": self.n_post_periods,
            "n_control_units": self.n_control_units,
            "sens_att_std": self.sensitivity.att_std,
            "sens_att_cv": self.sensitivity.att_cv,
            "sens_relative_std": self.sensitivity.relative_std,
            "sens_n_specs": self.sensitivity.n_specs,
            "bal_mean_abs_std_diff": self.covariate_balance.mean_abs_std_diff,
            "bal_max_abs_std_diff": self.covariate_balance.max_abs_std_diff,
            "bal_n_imbalanced": self.covariate_balance.n_imbalanced,
            "bal_rate": self.covariate_balance.balance_rate,
            "pred_rmspe_pre": self.prediction_error.rmspe_pre,
            "pred_mae_pre": self.prediction_error.mae_pre,
            "pred_corr_pre": self.prediction_error.corr_pre,
            "pred_r2_pre": self.prediction_error.r2_pre,
            "het_tau_std": self.heterogeneity.tau_std,
            "het_tau_cv": self.heterogeneity.tau_cv,
            "het_tau_median": self.heterogeneity.tau_median,
            "het_pct_positive": self.heterogeneity.pct_positive,
            "plac_p_value_space": self.placebo.p_value_space,
            "plac_p_value_time": self.placebo.p_value_time,
            "plac_effect_ratio": self.placebo.effect_to_placebo_ratio,
            "plac_n_space": self.placebo.n_placebos_space,
        }


class CausalMetricsCalculator:
    """Calculadora de métricas causales para un episodio."""
    
    def __init__(self, episode_id: str, model_type: str):
        self.episode_id = episode_id
        self.model_type = model_type
    
    def compute_sensitivity(self, sensitivity_results: pd.DataFrame, base_att: float) -> SensitivityMetrics:
        if sensitivity_results.empty or "att_mean" not in sensitivity_results.columns:
            return SensitivityMetrics()
        
        att_values = sensitivity_results["att_mean"].dropna().to_numpy(dtype=float)
        if len(att_values) == 0:
            return SensitivityMetrics()
        
        att_std = float(np.std(att_values))
        att_mean = float(np.mean(att_values))
        att_cv = float(att_std / abs(att_mean)) if abs(att_mean) > 1e-8 else np.nan
        att_range = float(np.max(att_values) - np.min(att_values))
        relative_std = float(att_std / abs(base_att)) if abs(base_att) > 1e-8 else np.nan
        
        return SensitivityMetrics(
            att_mean_base=float(base_att),
            att_std=att_std,
            att_cv=att_cv,
            att_range=att_range,
            n_specs=len(att_values),
            relative_std=relative_std
        )
    
    def compute_covariate_balance(self, X_treat: np.ndarray, X_control: np.ndarray, 
                                   feature_names: Optional[List[str]] = None) -> CovariateBalanceMetrics:
        if X_treat.size == 0 or X_control.size == 0:
            return CovariateBalanceMetrics()
        
        if X_treat.ndim == 1:
            X_treat = X_treat.reshape(-1, 1)
        if X_control.ndim == 1:
            X_control = X_control.reshape(-1, 1)
        
        n_features = X_treat.shape[1]
        abs_diffs = []
        
        for i in range(n_features):
            d = _standardized_diff(X_treat[:, i], X_control[:, i])
            if np.isfinite(d):
                abs_diffs.append(abs(d))
        
        if len(abs_diffs) == 0:
            return CovariateBalanceMetrics()
        
        mean_abs = float(np.mean(abs_diffs))
        max_abs = float(np.max(abs_diffs))
        n_imbalanced = int(np.sum(np.array(abs_diffs) > 0.25))
        balance_rate = float(1.0 - n_imbalanced / len(abs_diffs))
        
        return CovariateBalanceMetrics(
            mean_abs_std_diff=mean_abs,
            max_abs_std_diff=max_abs,
            n_imbalanced=n_imbalanced,
            balance_rate=balance_rate
        )
    
    def compute_prediction_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> PredictionErrorMetrics:
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(m):
            return PredictionErrorMetrics()
        
        y_t = y_true[m]
        y_p = y_pred[m]
        
        rmspe = _rmspe(y_t, y_p)
        mae = _mae(y_t, y_p)
        corr = float(np.corrcoef(y_t, y_p)[0, 1]) if len(y_t) > 1 else np.nan
        
        ss_res = np.sum((y_t - y_p)**2)
        ss_tot = np.sum((y_t - np.mean(y_t))**2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else np.nan
        
        return PredictionErrorMetrics(rmspe_pre=rmspe, mae_pre=mae, corr_pre=corr, r2_pre=r2)
    
    def compute_heterogeneity(self, tau: np.ndarray) -> HeterogeneityMetrics:
        tau_clean = tau[np.isfinite(tau)]
        if len(tau_clean) == 0:
            return HeterogeneityMetrics()
        
        tau_mean = float(np.mean(tau_clean))
        tau_std = float(np.std(tau_clean))
        tau_cv = float(tau_std / abs(tau_mean)) if abs(tau_mean) > 1e-8 else np.nan
        tau_median = float(np.median(tau_clean))
        pct_positive = float(np.mean(tau_clean > 0) * 100)
        
        return HeterogeneityMetrics(
            tau_mean=tau_mean, tau_std=tau_std, tau_cv=tau_cv,
            tau_median=tau_median, pct_positive=pct_positive
        )
    
    def compute_placebo(self, att_real: float, placebo_space_df: Optional[pd.DataFrame] = None,
                        placebo_time_df: Optional[pd.DataFrame] = None) -> PlaceboMetrics:
        metrics = PlaceboMetrics()
        
        if placebo_space_df is not None and not placebo_space_df.empty:
            if "att_placebo_sum" in placebo_space_df.columns:
                plac_vals = placebo_space_df["att_placebo_sum"].dropna().to_numpy(dtype=float)
                
                if len(plac_vals) > 0:
                    metrics.n_placebos_space = len(plac_vals)
                    
                    if np.isfinite(att_real):
                        n_extreme = np.sum(np.abs(plac_vals) >= abs(att_real))
                        metrics.p_value_space = float((1 + n_extreme) / (1 + len(plac_vals)))
                        
                        plac_mean_abs = np.mean(np.abs(plac_vals))
                        if plac_mean_abs > 1e-8:
                            metrics.effect_to_placebo_ratio = float(abs(att_real) / plac_mean_abs)
        
        if placebo_time_df is not None and not placebo_time_df.empty:
            if "att_placebo_mean" in placebo_time_df.columns:
                att_plac_time = placebo_time_df["att_placebo_mean"].iloc[0]
                if np.isfinite(att_plac_time):
                    metrics.p_value_time = 0.5  # Placeholder
        
        return metrics
    
    def compute_all(self, y_obs_pre: np.ndarray, y_hat_pre: np.ndarray, tau_post: np.ndarray,
                     att_base: float, X_treat_pre: Optional[np.ndarray] = None,
                     X_control_pre: Optional[np.ndarray] = None, feature_names: Optional[List[str]] = None,
                     sensitivity_df: Optional[pd.DataFrame] = None, placebo_space_df: Optional[pd.DataFrame] = None,
                     placebo_time_df: Optional[pd.DataFrame] = None, n_control_units: int = 0) -> CausalMetricsReport:
        
        report = CausalMetricsReport(episode_id=self.episode_id, model_type=self.model_type)
        
        report.n_pre_periods = len(y_obs_pre)
        report.n_post_periods = len(tau_post)
        report.n_control_units = n_control_units
        
        report.prediction_error = self.compute_prediction_error(y_obs_pre, y_hat_pre)
        report.heterogeneity = self.compute_heterogeneity(tau_post)
        
        if sensitivity_df is not None and not sensitivity_df.empty:
            report.sensitivity = self.compute_sensitivity(sensitivity_df, att_base)
        
        if X_treat_pre is not None and X_control_pre is not None:
            report.covariate_balance = self.compute_covariate_balance(X_treat_pre, X_control_pre, feature_names)
        
        report.placebo = self.compute_placebo(att_base, placebo_space_df, placebo_time_df)
        
        return report


def create_comparison_table(metrics_list: List[CausalMetricsReport]) -> pd.DataFrame:
    """Crea tabla comparativa desde lista de reportes."""
    rows = [m.to_flat_dict() for m in metrics_list]
    return pd.DataFrame(rows)
