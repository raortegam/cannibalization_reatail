from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import json
import concurrent.futures as _cf
try:  # GPU opcional
    import cupy as cp  # type: ignore
    _CUPY_OK = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    _CUPY_OK = False

from app.causal.learners import CausalLearnResult


@dataclass
class PlaceboResult:
    """Resultados de un test placebo en espacio (unidades).

    Atributos
    ---------
    att_obs: float
        ATT observado con la asignación real.
    atts_placebo: pd.Series
        Serie con los ATT bajo reasignaciones placebo (índice = iteración 1..B).
    p_value_two_sided: float
        p-valor de aleatorización (dos colas) basado en |ATT_b|.
    summary: Dict[str, float]
        Resumen: mean, std, q05, q50, q95 de la distribución placebo.
    details: Dict[str, Any]
        Metadatos: B, seed, n_treated, strategy, etc.
    """

    att_obs: float
    atts_placebo: pd.Series
    p_value_two_sided: float
    summary: Dict[str, float]
    details: Dict[str, Any]


@dataclass
class PreErrorResult:
    """Errores de predicción en pre-tratamiento.

    - metrics_overall: métricas globales (RMSE, MAE, MAPE si aplica) en celdas de pre.
    - by_time: métricas por tiempo en pre.
    - by_unit: métricas por unidad en pre.
    - details: info contextual (n_obs, periodos, columnas usadas).
    """

    metrics_overall: Dict[str, float]
    by_time: pd.DataFrame
    by_unit: pd.DataFrame
    details: Dict[str, Any]


@dataclass
class SensitivityResult:
    """Resultados de un análisis de sensibilidad por configuración del estimador.

    - results: tabla con filas por configuración (params) y columnas con métricas clave.
    - best_idx_by_pre_rmse: índice de la mejor configuración por menor pre_rmse (si disponible).
    - details: metadatos (tamaño de grilla, etc.).
    """

    results: pd.DataFrame
    best_idx_by_pre_rmse: Optional[int]
    details: Dict[str, Any]


@dataclass
class HeterogeneityGPUResult:
    """Salida de análisis de heterogeneidad acelerado (GPU si disponible).

    - summary: métricas por estimador (estadísticas de tau_it, att_i y att_t).
    - pairwise_corr: matriz de correlaciones de tau_it entre estimadores (en celdas comunes tratadas post).
    - subgroups: métricas por bins de features a nivel unidad (si se pasan).
    - details: backend (cupy/numpy), device (gpu/cpu), tamaños y metadatos.
    """

    summary: pd.DataFrame
    pairwise_corr: Optional[pd.DataFrame]
    subgroups: Dict[str, pd.DataFrame]
    details: Dict[str, Any]


class CounterfactualEvaluator:
    """Utilidades para evaluar y comparar estimadores de contrafactual.

    Incluye placebo tests en espacio (unidades) y funciones de comparación.
    """

    def __init__(self, random_state: int = 42):
        self._rng = np.random.default_rng(random_state)

    # ------------------------------
    # Helpers internos
    # ------------------------------

    @staticmethod
    def _sorted_unique(values: Sequence[Any]) -> List[Any]:
        return sorted(pd.unique(pd.Series(list(values))).tolist())

    @staticmethod
    def _one_hot_df(df: pd.DataFrame) -> pd.DataFrame:
        """One-hot simple para columnas categóricas."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            return df.copy()
        return pd.get_dummies(df, columns=cat_cols, drop_first=False)

    @staticmethod
    def _derive_treat_from_adoption(
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        adoption_time_col: str,
        out_treat_col: str = "__D__",
    ) -> pd.Series:
        d = df[[id_col, time_col, adoption_time_col]].copy()
        adop = (
            d[[id_col, adoption_time_col]]
            .drop_duplicates(subset=[id_col])
            .set_index(id_col)[adoption_time_col]
        )
        d = d.merge(adop.rename("__adopt__"), how="left", left_on=id_col, right_index=True)
        D = ((d[time_col] >= d["__adopt__"]) & d["__adopt__"].notna()).astype(int)
        return pd.Series(D.values, index=df.index, name=out_treat_col)

    @staticmethod
    def _ensure_periods(
        df: pd.DataFrame,
        time_col: str,
        adoption_time_col: Optional[str],
        pre_period: Optional[Tuple[Any, Any]],
        post_period: Optional[Tuple[Any, Any]],
    ) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
        times = CounterfactualEvaluator._sorted_unique(df[time_col])
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

    # ------------------------------
    # Placebo en espacio (unidades)
    # ------------------------------

    def placebo_in_space(
        self,
        estimator_fn: Callable[..., CausalLearnResult],
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        outcome_col: str,
        features: List[str],
        treat_col: Optional[str] = None,
        adoption_time_col: Optional[str] = None,
        pre_period: Optional[Tuple[Any, Any]] = None,
        post_period: Optional[Tuple[Any, Any]] = None,
        B: int = 200,
        strategy: str = "controls_only",  # "controls_only" | "all_units"
        placebo_treat_col: str = "__PLACEBO_D__",
        estimator_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        n_jobs: Optional[int] = None,
        parallel_backend: str = "thread",  # "thread" | (futuro: "process")
    ) -> PlaceboResult:
        """Ejecuta un placebo test en espacio reasignando tratamiento a unidades de control.

        Parámetros
        ----------
        estimator_fn: Callable
            Función/método que entrena y retorna `CausalLearnResult` (p.ej., `LightGBMLearner().x_learner`).
        strategy: str
            "controls_only": muestrea placebo sólo de unidades de control.  
            "all_units": muestrea de todas las unidades, permutando la identidad de tratadas.
        placebo_treat_col: str
            Nombre de columna booleana/int que se creará para marcar tratadas en post.
        estimator_kwargs: Dict
            Parámetros adicionales a pasar a `estimator_fn`.
        seed: Optional[int]
            Semilla para la aleatorización (por defecto usa la del evaluador).
        n_jobs: Optional[int]
            Número de workers para paralelizar placebos (por defecto = 1, secuencial). Recomendado: <= número de cores.
        parallel_backend: str
            "thread" (por defecto). El backend "process" no está soportado actualmente para métodos ligados; se hará fallback a "thread".
        """
        if estimator_kwargs is None:
            estimator_kwargs = {}

        d = df.copy()
        # Determina periodos pre/post
        pre_period, post_period = self._ensure_periods(d, time_col, adoption_time_col, pre_period, post_period)
        pre_start, pre_end = pre_period
        post_start, post_end = post_period

        # Construye D real para contar tratadas
        if treat_col is not None and treat_col in d.columns:
            D_real = d[treat_col].astype(int)
        elif adoption_time_col is not None and adoption_time_col in d.columns:
            D_real = self._derive_treat_from_adoption(d, id_col, time_col, adoption_time_col, out_treat_col="__D__")
        else:
            raise ValueError("Necesitas 'treat_col' o 'adoption_time_col' para el ATT observado.")

        # Unidades tratadas y de control (reales)
        treated_units = d.groupby(id_col)[D_real.name].max()
        treated_units = treated_units[treated_units > 0].index.tolist()
        all_units = self._sorted_unique(d[id_col])
        control_units = [u for u in all_units if u not in set(treated_units)]
        n_treated = len(treated_units)
        if n_treated == 0:
            raise ValueError("No hay unidades tratadas en los datos reales.")

        # ATT observado
        res_obs = estimator_fn(
            d,
            id_col=id_col,
            time_col=time_col,
            outcome_col=outcome_col,
            features=features,
            treat_col=treat_col,
            adoption_time_col=adoption_time_col,
            pre_period=pre_period,
            post_period=post_period,
            **estimator_kwargs,
        )
        att_obs = float(res_obs.att)

        # Preparación aleatoria y muestreo de placebos
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        if strategy == "controls_only":
            pool = control_units
        elif strategy == "all_units":
            pool = all_units
        else:
            raise ValueError("strategy debe ser 'controls_only' o 'all_units'.")
        if len(pool) == 0:
            raise ValueError("El pool de unidades para placebo está vacío.")

        # Precalcula las listas de unidades placebo a usar en cada iteración (deterministas con la semilla)
        placebo_units_list: List[List[Any]] = []
        for _ in range(B):
            if len(pool) >= n_treated:
                units = rng.choice(pool, size=n_treated, replace=False).tolist()
            else:
                units = rng.choice(pool, size=n_treated, replace=True).tolist()
            placebo_units_list.append(units)

        # Precalcula máscara de post (compartida por todas las iteraciones)
        is_post = d[time_col].apply(lambda t: (post_start <= t <= post_end))

        def _run_one(b_idx_units: Tuple[int, List[Any]]) -> Tuple[int, float]:
            b_idx, units = b_idx_units
            d_b = d.copy()
            is_placebo_unit = d[id_col].isin(units)
            D_placebo = (is_placebo_unit & is_post).astype(int)
            d_b[placebo_treat_col] = D_placebo.values
            res_b = estimator_fn(
                d_b,
                id_col=id_col,
                time_col=time_col,
                outcome_col=outcome_col,
                features=features,
                treat_col=placebo_treat_col,
                adoption_time_col=None,
                pre_period=pre_period,
                post_period=post_period,
                **estimator_kwargs,
            )
            return b_idx, float(res_b.att)

        # Ejecuta en paralelo (threads). Si n_jobs<=1, ejecuta secuencial.
        workers = max(1, int(n_jobs)) if n_jobs is not None else 1
        results: Dict[int, float] = {}
        errors: Dict[int, str] = {}
        if workers == 1:
            for b_idx, units in enumerate(placebo_units_list, start=1):
                try:
                    k, val = _run_one((b_idx, units))
                    results[k] = val
                except Exception as ex:  # pragma: no cover
                    errors[b_idx] = str(ex)
        else:
            if parallel_backend != "thread":
                # Fallback seguro
                parallel_backend = "thread"
            with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_run_one, (b_idx, units)): b_idx for b_idx, units in enumerate(placebo_units_list, start=1)}
                for fut in _cf.as_completed(futs):
                    b_idx = futs[fut]
                    try:
                        k, val = fut.result()
                        results[k] = val
                    except Exception as ex:  # pragma: no cover
                        errors[b_idx] = str(ex)

        # Reconstruye serie de ATTs en el orden 1..B y filtra NaN/errores
        atts: List[float] = [results.get(i, np.nan) for i in range(1, B + 1)]

        atts_series = pd.Series(atts, index=pd.RangeIndex(1, B + 1, name="iter"), name="att_placebo")
        # p-valor de aleatorización (dos colas)
        valid = atts_series.dropna()
        B_eff = len(valid)
        p_val = float((1 + np.sum(np.abs(valid.values) >= abs(att_obs))) / (B_eff + 1)) if B_eff > 0 else np.nan

        summary = {
            "mean": float(valid.mean()) if B_eff > 0 else np.nan,
            "std": float(valid.std(ddof=1)) if B_eff > 1 else 0.0,
            "q05": float(valid.quantile(0.05)) if B_eff > 0 else np.nan,
            "q50": float(valid.quantile(0.50)) if B_eff > 0 else np.nan,
            "q95": float(valid.quantile(0.95)) if B_eff > 0 else np.nan,
        }
        details: Dict[str, Any] = {
            "B": int(B),
            "seed": int(seed if seed is not None else 0),
            "n_treated": int(n_treated),
            "strategy": strategy,
            "n_jobs": int(workers),
            "parallel_backend": parallel_backend,
            "B_effective": int(B_eff),
            "n_errors": int(len(errors)),
        }

        return PlaceboResult(
            att_obs=att_obs,
            atts_placebo=atts_series,
            p_value_two_sided=p_val,
            summary=summary,
            details=details,
        )

    # ------------------------------
    # Heterogeneidad (GPU-accelerated cuando sea posible)
    # ------------------------------

    def heterogeneity_analysis_gpu(
        self,
        results: Dict[str, CausalLearnResult] | CausalLearnResult,
        df: Optional[pd.DataFrame] = None,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        subgroup_features: Optional[Dict[str, int]] = None,  # feature -> n_bins (qcut)
        prefer_gpu: bool = True,
        force_gpu: bool = False,
    ) -> HeterogeneityGPUResult:
        """Analiza heterogeneidad de efectos usando operaciones vectorizadas en GPU si hay CuPy.

        - Acepta un `CausalLearnResult` o un dict nombre->resultado para comparar múltiples estimadores.
        - Usa tau_it (NaN fuera de tratadas post) para computar estadísticas y correlaciones.
        - Subgrupos opcionales por feature a nivel unidad (bins por cuantiles con qcut).
        """
        # Selección de backend
        if prefer_gpu and not _CUPY_OK and force_gpu:
            raise RuntimeError("GPU solicitada pero CuPy no está disponible.")
        use_gpu = bool(prefer_gpu and _CUPY_OK)
        xp = cp if use_gpu else np  # type: ignore

        def _to_np(a):
            if use_gpu:
                return cp.asnumpy(a)  # type: ignore
            return a

        # Normaliza input a dict
        if isinstance(results, CausalLearnResult):
            results_dict: Dict[str, CausalLearnResult] = {"estimator": results}
        else:
            results_dict = results

        # Resumen por estimador
        rows: List[Dict[str, Any]] = []
        for name, res in results_dict.items():
            tau_np = res.tau_it.to_numpy(dtype=float)
            mask = np.isfinite(tau_np)
            vals = tau_np[mask]
            if vals.size == 0:
                rows.append({
                    "estimator": name,
                    "tau_mean": np.nan,
                    "tau_std": np.nan,
                    "tau_q25": np.nan,
                    "tau_q50": np.nan,
                    "tau_q75": np.nan,
                    "tau_iqr": np.nan,
                    "att_i_mean": np.nan,
                    "att_i_std": np.nan,
                    "att_i_cv": np.nan,
                    "att_i_q10": np.nan,
                    "att_i_q50": np.nan,
                    "att_i_q90": np.nan,
                    "att_t_std": np.nan,
                    "att_t_iqr": np.nan,
                    "n_units": res.tau_it.shape[0],
                    "n_times": res.tau_it.shape[1],
                    "n_cells_post": 0,
                    "backend": "cupy" if use_gpu else "numpy",
                })
                continue

            arr = xp.asarray(vals)
            q25, q50, q75 = [float(_to_np(x)) for x in xp.percentile(arr, [25, 50, 75])]
            tau_mean = float(_to_np(xp.mean(arr)))
            tau_std = float(_to_np(xp.std(arr)))
            tau_iqr = q75 - q25

            # att_i y att_t desde tau_it con NaN fuera de tratadas-post
            tau_xp = xp.asarray(tau_np)
            att_i_x = xp.nanmean(tau_xp, axis=1)
            att_t_x = xp.nanmean(tau_xp, axis=0)
            att_i_vals = att_i_x[~xp.isnan(att_i_x)]
            att_t_vals = att_t_x[~xp.isnan(att_t_x)]

            if att_i_vals.size > 0:
                ai_mean = float(_to_np(xp.mean(att_i_vals)))
                ai_std = float(_to_np(xp.std(att_i_vals)))
                ai_cv = float(ai_std / (abs(ai_mean) + 1e-12))
                q10, q50i, q90 = [float(_to_np(x)) for x in xp.percentile(att_i_vals, [10, 50, 90])]
            else:
                ai_mean = ai_std = ai_cv = q10 = q50i = q90 = np.nan

            if att_t_vals.size > 0:
                at_std = float(_to_np(xp.std(att_t_vals)))
                qt25, qt75 = [float(_to_np(x)) for x in xp.percentile(att_t_vals, [25, 75])]
                at_iqr = qt75 - qt25
            else:
                at_std = at_iqr = np.nan

            rows.append({
                "estimator": name,
                "tau_mean": tau_mean,
                "tau_std": tau_std,
                "tau_q25": q25,
                "tau_q50": q50,
                "tau_q75": q75,
                "tau_iqr": tau_iqr,
                "att_i_mean": ai_mean,
                "att_i_std": ai_std,
                "att_i_cv": ai_cv,
                "att_i_q10": q10,
                "att_i_q50": q50i,
                "att_i_q90": q90,
                "att_t_std": at_std,
                "att_t_iqr": at_iqr,
                "n_units": res.tau_it.shape[0],
                "n_times": res.tau_it.shape[1],
                "n_cells_post": int(vals.size),
                "backend": "cupy" if use_gpu else "numpy",
            })

        summary_df = pd.DataFrame(rows).set_index("estimator")

        # Correlaciones pareadas entre estimadores (en celdas comunes tratadas post)
        pairwise_df: Optional[pd.DataFrame] = None
        keys = list(results_dict.keys())
        if len(keys) >= 2:
            # Intersección de índices y columnas
            idx_common = results_dict[keys[0]].tau_it.index
            col_common = results_dict[keys[0]].tau_it.columns
            for k in keys[1:]:
                idx_common = idx_common.intersection(results_dict[k].tau_it.index)
                col_common = col_common.intersection(results_dict[k].tau_it.columns)
            if len(idx_common) > 0 and len(col_common) > 0:
                vecs = []
                for k in keys:
                    sub = results_dict[k].tau_it.loc[idx_common, col_common].to_numpy(dtype=float)
                    m = np.isfinite(sub)
                    vecs.append(sub[m])
                min_len = min(len(v) for v in vecs)
                if min_len > 1:
                    X = xp.stack([xp.asarray(v[:min_len]) for v in vecs], axis=0)
                    C = _to_np(xp.corrcoef(X))  # (k x k)
                    pairwise_df = pd.DataFrame(C, index=keys, columns=keys)

        # Subgrupos por feature (bins a nivel unidad)
        subgroups: Dict[str, pd.DataFrame] = {}
        if subgroup_features and df is not None and id_col is not None:
            for feat, q in subgroup_features.items():
                if feat not in df.columns:
                    continue
                feat_unit = df[[id_col, feat]].groupby(id_col)[feat].mean()
                try:
                    bins = pd.qcut(feat_unit, q=q, duplicates="drop")
                except Exception:
                    bins = pd.cut(feat_unit, bins=q, duplicates="drop")
                tables = []
                for name, res in results_dict.items():
                    att_i = res.att_i
                    df_join = pd.DataFrame({"bin": bins}).join(att_i.rename("att_i"), how="inner")
                    tab = df_join.groupby("bin")["att_i"].agg(["count", "mean", "std"])
                    tab.columns = pd.MultiIndex.from_product([[name], tab.columns])
                    tables.append(tab)
                if tables:
                    subgroups[feat] = pd.concat(tables, axis=1)

        details = {
            "backend": "cupy" if use_gpu else "numpy",
            "device": "gpu" if use_gpu else "cpu",
            "n_estimators": len(results_dict),
        }

        return HeterogeneityGPUResult(
            summary=summary_df,
            pairwise_corr=pairwise_df,
            subgroups=subgroups,
            details=details,
        )

    # ------------------------------
    # Comparación simple de estimadores
    # ------------------------------


    # ------------------------------
    # Sensibilidad del algoritmo (grid de hiperparámetros)
    # ------------------------------

    def sensitivity_analysis(
        self,
        estimator_fn: Callable[..., CausalLearnResult],
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        outcome_col: str,
        features: List[str],
        treat_col: Optional[str] = None,
        adoption_time_col: Optional[str] = None,
        pre_period: Optional[Tuple[Any, Any]] = None,
        post_period: Optional[Tuple[Any, Any]] = None,
        param_grid: Optional[List[Dict[str, Any]]] = None,
        base_kwargs: Optional[Dict[str, Any]] = None,
        names: Optional[List[str]] = None,
        n_jobs: Optional[int] = None,
        parallel_backend: str = "thread",
    ) -> SensitivityResult:
        """Corre una grilla de configuraciones y resume métricas clave (ATT, pre_rmse, etc.).

        - `param_grid`: lista de dicts que se mezclan sobre `base_kwargs` y se pasan a `estimator_fn`.
        - `names`: etiquetas opcionales por configuración para la salida.
        - `n_jobs`: número de workers para ejecutar configuraciones en paralelo (threads por defecto).
        - `parallel_backend`: "thread" (recomendado para LightGBM/CatBoost). "process" no soportado aquí.
        """
        d = df.copy()
        pre_period, post_period = self._ensure_periods(d, time_col, adoption_time_col, pre_period, post_period)
        if param_grid is None or len(param_grid) == 0:
            param_grid = [{}]
        if base_kwargs is None:
            base_kwargs = {}
        if names is not None and len(names) != len(param_grid):
            raise ValueError("Si pasas 'names', debe tener la misma longitud que 'param_grid'.")

        # Tareas a ejecutar
        tasks = [(i, params, names[i - 1] if names is not None else f"cfg_{i}") for i, params in enumerate(param_grid, start=1)]

        def _run_one(idx: int, params: Dict[str, Any], name: str) -> Dict[str, Any]:
            kw = {**base_kwargs, **params}
            try:
                res = estimator_fn(
                    d,
                    id_col=id_col,
                    time_col=time_col,
                    outcome_col=outcome_col,
                    features=features,
                    treat_col=treat_col,
                    adoption_time_col=adoption_time_col,
                    pre_period=pre_period,
                    post_period=post_period,
                    **kw,
                )
                row: Dict[str, Any] = {
                    "idx": idx,
                    "name": name,
                    "params": json.dumps(params, sort_keys=True),
                    "att": float(res.att),
                    "pre_rmse": float(res.metrics.get("pre_rmse", np.nan)),
                    "n_units": int(res.metrics.get("n_units", np.nan)),
                    "n_times": int(res.metrics.get("n_times", np.nan)),
                    "n_treated": int(res.metrics.get("n_treated", np.nan)),
                    "learner": res.metrics.get("learner", ""),
                    "library": res.metrics.get("library", ""),
                    "error": "",
                }
                try:
                    row["att_t_std"] = float(res.att_t.std()) if res.att_t is not None and len(res.att_t) > 0 else np.nan
                except Exception:
                    row["att_t_std"] = np.nan
                return row
            except Exception as ex:  # pragma: no cover
                return {
                    "idx": idx,
                    "name": name,
                    "params": json.dumps(params, sort_keys=True),
                    "att": np.nan,
                    "pre_rmse": np.nan,
                    "n_units": np.nan,
                    "n_times": np.nan,
                    "n_treated": np.nan,
                    "learner": "",
                    "library": "",
                    "att_t_std": np.nan,
                    "error": str(ex),
                }

        # Ejecuta paralelo (threads) o secuencial
        workers = max(1, int(n_jobs)) if n_jobs is not None else 1
        rows: List[Dict[str, Any]] = []
        if workers == 1 or parallel_backend != "thread":
            for idx, params, name in tasks:
                rows.append(_run_one(idx, params, name))
        else:
            with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_run_one, idx, params, name): idx for idx, params, name in tasks}
                for fut in _cf.as_completed(futs):
                    rows.append(fut.result())

        results_df = pd.DataFrame(rows).set_index("idx")
        # Mejor por menor pre_rmse (si existe)
        if results_df["pre_rmse"].notna().any():
            best_idx = int(results_df["pre_rmse"].idxmin())
        else:
            best_idx = None

        details = {
            "grid_size": len(param_grid),
            "pre_period": pre_period,
            "post_period": post_period,
            "n_jobs": int(workers),
            "parallel_backend": parallel_backend if workers > 1 else "none",
            "n_errors": int(results_df["error"].astype(bool).sum()) if "error" in results_df.columns else 0,
        }

        return SensitivityResult(
            results=results_df,
            best_idx_by_pre_rmse=best_idx,
            details=details,
        )

    # ------------------------------
    # Comparación simple de estimadores
    # ------------------------------

    @staticmethod
    def compare_att(results: Dict[str, CausalLearnResult]) -> pd.DataFrame:
        """Compara ATT y métricas básicas entre estimadores.

        Parámetros
        ----------
        results: Dict[str, CausalLearnResult]
            Mapa nombre_estimator -> resultado.
        """
        rows = []
        for name, res in results.items():
            rows.append(
                {
                    "estimator": name,
                    "att": float(res.att),
                    "pre_rmse": float(res.metrics.get("pre_rmse", np.nan)),
                    "n_units": int(res.metrics.get("n_units", np.nan)),
                    "n_times": int(res.metrics.get("n_times", np.nan)),
                    "n_treated": int(res.metrics.get("n_treated", np.nan)),
                    "learner": res.metrics.get("learner", ""),
                    "library": res.metrics.get("library", ""),
                }
            )
        return pd.DataFrame(rows)


__all__ = [
    "PlaceboResult",
    "PreErrorResult",
    "SensitivityResult",
    "HeterogeneityGPUResult",
    "CounterfactualEvaluator",
]