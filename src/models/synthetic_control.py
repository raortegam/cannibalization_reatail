from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as _np
import pandas as pd

# Intento opcional de usar CuPy (GPU). Si no está disponible, cae a NumPy.
try:  # pragma: no cover
    import cupy as _cp  # type: ignore
    _CUPY_OK = True
except Exception:  # pragma: no cover
    _cp = None  # type: ignore
    _CUPY_OK = False


# ==============================
# Resultados y helpers públicos
# ==============================

@dataclass
class GSCResult:
    """Resultados de Generalized Synthetic Control (matrix completion baja-rango).

    Atributos
    ---------
    att: float
        Efecto promedio del tratamiento (ATT) en el periodo post (promedio sobre celdas tratadas post).
    att_t: pd.Series
        ATT por tiempo (index = valores de `time` en post). Promedia sobre unidades tratadas con D_it=1.
    att_i: pd.Series
        ATT por unidad tratada (index = ids tratados). Promedia sobre periodos post donde D_it=1.
    y0_hat: pd.DataFrame
        Contrafactual imputado Y(0) solo para celdas tratadas en post (NaN en otras celdas).
    tau_it: pd.DataFrame
        Efecto por celda tratada en post (= Y - y0_hat). NaN fuera de celdas tratadas post.
    full_y_hat: pd.DataFrame
        Reconstrucción baja-rango completa (todas las celdas). Útil para diagnóstico.
    r: int
        Rango/factores efectivos utilizados.
    factors: pd.DataFrame
        Factores en el tiempo (t x r), indexado por time.
    loadings: pd.DataFrame
        Cargas por unidad (i x r), indexado por id.
    metrics: Dict[str, Any]
        Métricas y diagnósticos (p.ej., pre_rmse, n_iter, converged, backend, device).
    """

    att: float
    att_t: pd.Series
    att_i: pd.Series
    y0_hat: pd.DataFrame
    tau_it: pd.DataFrame
    full_y_hat: pd.DataFrame
    r: int
    factors: pd.DataFrame
    loadings: pd.DataFrame
    metrics: Dict[str, Any]


# ------------------------------
# Helpers de preparación de datos
# ------------------------------

def derive_treat_from_adoption(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    adoption_time_col: str,
    out_treat_col: str = "D",
) -> pd.DataFrame:
    """Deriva D_it = 1[t >= t0_i] a partir de fecha de adopción por unidad.

    No sobrescribe columnas existentes; retorna un nuevo DataFrame con `out_treat_col`.
    """
    d = df.copy()
    adop = d[[id_col, adoption_time_col]].drop_duplicates(subset=[id_col]).set_index(id_col)[adoption_time_col]
    d = d.merge(adop.rename("__adopt__"), how="left", left_on=id_col, right_index=True)
    d[out_treat_col] = (d[time_col] >= d["__adopt__"]) & d["__adopt__"].notna()
    d.drop(columns=["__adopt__"], inplace=True)
    return d


def prepare_panel(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    y_col: str,
    treat_col: Optional[str],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Construye matrices anchas Y (id x tiempo) y D (si existe)."""
    Y = df.pivot(index=id_col, columns=time_col, values=y_col).sort_index(axis=0).sort_index(axis=1)
    D = None
    if treat_col is not None and treat_col in df.columns:
        D = df.pivot(index=id_col, columns=time_col, values=treat_col).sort_index(axis=0).sort_index(axis=1)
    return Y, D


def _sorted_unique(values: Iterable[Any]) -> List[Any]:
    return sorted(pd.unique(pd.Series(list(values))).tolist())


# ------------------------------
# Selección de donantes (opcional, CPU)
# ------------------------------

def find_donors(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    y_col: str,
    treat_col: str,
    pre_start: Any,
    pre_end: Any,
    K: int = 10,
    min_pre_periods: int = 8,
    extra_filters: Optional[Dict[str, Any]] = None,
    method: str = "cosine",  # "cosine" | "corr" | "euclidean"
) -> Dict[Any, List[Any]]:
    """Selecciona K donantes por unidad tratada usando similitud en pre (CPU/NumPy)."""
    d = df.copy()
    if extra_filters:
        for k, v in extra_filters.items():
            d = d[d[k] == v]

    d_pre = d[(d[time_col] >= pre_start) & (d[time_col] <= pre_end)].copy()
    Y = d_pre.pivot(index=id_col, columns=time_col, values=y_col)
    counts = (~Y.isna()).sum(axis=1)
    ok_ids = counts[counts >= min_pre_periods].index
    Y = Y.loc[ok_ids]

    treated_flag = d.groupby(id_col)[treat_col].max().reindex(Y.index).fillna(0).astype(int)
    treated_ids = treated_flag[treated_flag > 0].index.tolist()
    control_ids = treated_flag[treated_flag == 0].index.tolist()
    if not treated_ids or not control_ids:
        return {}

    col_means = Y.mean(axis=0, skipna=True)
    Y_dm = (Y - col_means).fillna(0.0)
    row_mean = Y_dm.mean(axis=1)
    row_std = Y_dm.std(axis=1).replace(0, 1.0)
    Y_std = ((Y_dm.sub(row_mean, axis=0)).div(row_std, axis=0)).astype(float)

    T = Y_std.loc[treated_ids].to_numpy(dtype=float)
    C = Y_std.loc[control_ids].to_numpy(dtype=float)

    donors: Dict[Any, List[Any]] = {}
    if method in ("cosine", "corr"):
        def _normalize_rows(A: _np.ndarray) -> _np.ndarray:
            nrm = _np.linalg.norm(A, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            return A / nrm
        Tn = _normalize_rows(T)
        Cn = _normalize_rows(C)
        sim = Tn @ Cn.T
        rank = _np.argsort(-sim, axis=1)
    else:
        T2 = (T ** 2).sum(axis=1, keepdims=True)
        C2 = (C ** 2).sum(axis=1, keepdims=True).T
        cross = T @ C.T
        dist = _np.sqrt(_np.maximum(T2 + C2 - 2.0 * cross, 0.0))
        rank = _np.argsort(dist, axis=1)

    ctrl_list = control_ids
    for i, tid in enumerate(treated_ids):
        idx = rank[i, : min(K, len(ctrl_list))]
        donors[tid] = [ctrl_list[j] for j in idx]
    return donors


# ==============================
# Núcleo GSC (GPU con CuPy, fallback CPU)
# ==============================

class _XP:
    """Pequeño wrapper para escoger backend NumPy/CuPy y mover datos."""

    def __init__(self, prefer_gpu: bool = True, force_gpu: bool = False):
        self.using_gpu = False
        if prefer_gpu and _CUPY_OK:
            try:  # pragma: no cover
                _cp.cuda.runtime.getDeviceCount()  # valida acceso a GPU
                self.xp = _cp
                self.using_gpu = True
            except Exception:
                if force_gpu:
                    raise RuntimeError("No se pudo inicializar GPU/CuPy.")
                self.xp = _np
        else:
            if force_gpu:
                raise RuntimeError("'cupy' no está instalado para ejecución en GPU.")
            self.xp = _np

    def to_xp(self, a, dtype=_np.float64):
        if self.using_gpu:
            return self.xp.asarray(a, dtype=dtype)
        return _np.asarray(a, dtype=dtype)

    def to_np(self, a):
        if self.using_gpu:
            return _cp.asnumpy(a)
        return a


def _initial_fill_xp(xp, Y, obs_mask):
    # Basado en medias de fila/columna sobre observados
    filled = Y.copy()
    gmean = xp.nanmean(Y[obs_mask]) if bool(obs_mask.any()) else xp.array(0.0, dtype=Y.dtype)
    row_means = xp.where(
        obs_mask.sum(axis=1, keepdims=True) > 0,
        xp.nansum(xp.where(obs_mask, Y, 0.0), axis=1, keepdims=True)
        / xp.clip(obs_mask.sum(axis=1, keepdims=True), 1, None),
        gmean,
    )
    col_means = xp.where(
        obs_mask.sum(axis=0, keepdims=True) > 0,
        xp.nansum(xp.where(obs_mask, Y, 0.0), axis=0, keepdims=True)
        / xp.clip(obs_mask.sum(axis=0, keepdims=True), 1, None),
        gmean,
    )
    base = row_means + col_means - gmean
    filled = filled.copy()
    filled[~obs_mask] = base[~obs_mask]
    filled = xp.where(xp.isfinite(filled), filled, gmean)
    return filled


def _choose_rank_via_ic_xp(xp, U, S, Vt, Y_true, pre_mask, r_grid: Sequence[int]) -> int:
    n, t = Y_true.shape
    n_obs = int(pre_mask.sum())
    n_obs = max(n_obs, 1)
    best_r, best_obj = 1, float("inf")
    for r in r_grid:
        r = max(1, int(r))
        Ur, Sr, Vtr = U[:, :r], S[:r], Vt[:r, :]
        Yhat = Ur @ xp.diag(Sr) @ Vtr
        err = Y_true - Yhat
        mse_pre = float((err[pre_mask] ** 2).mean()) if n_obs > 0 else float("inf")
        pen = _np.log(max(n * t, 2)) * r * (n + t) / max(n * t, 2)
        obj = mse_pre + float(pen)
        if obj < best_obj:
            best_obj, best_r = obj, r
    return best_r


def _em_svd_xp(
    xp,
    Y,
    obs_mask,
    pre_mask,
    r: Optional[int],
    r_grid: Tuple[int, int] = (1, 6),
    max_iter: int = 200,
    tol: float = 1e-4,
) -> Tuple[Any, int, Dict[str, Any]]:
    """EM con SVD truncado sobre backend xp (CuPy/NumPy)."""
    n, t = Y.shape
    filled = _initial_fill_xp(xp, Y, obs_mask)

    chosen_r: Optional[int] = r
    converged = False
    it = 0
    prev_missing = filled.copy()

    r_candidates = list(range(r_grid[0], r_grid[1] + 1)) if r is None else [r]

    while it < max_iter:
        it += 1
        U, S, Vt = xp.linalg.svd(filled, full_matrices=False)
        if chosen_r is None:
            chosen_r = _choose_rank_via_ic_xp(xp, U, S, Vt, Y_true=Y, pre_mask=pre_mask, r_grid=r_candidates)
        r_eff = max(1, int(chosen_r))

        Ur, Sr, Vtr = U[:, :r_eff], S[:r_eff], Vt[:r_eff, :]
        Yhat = Ur @ xp.diag(Sr) @ Vtr

        filled[~obs_mask] = Yhat[~obs_mask]

        diff = Yhat[~obs_mask] - prev_missing[~obs_mask]
        denom = float(xp.linalg.norm(prev_missing[~obs_mask])) + 1e-12
        rel_change = float(xp.linalg.norm(diff)) / denom if denom > 0 else 0.0
        prev_missing[~obs_mask] = Yhat[~obs_mask]

        if rel_change < tol:
            converged = True
            break

    metrics = {
        "n_iter": it,
        "converged": converged,
        "rank": int(chosen_r if chosen_r is not None else r_eff),
    }
    return filled, int(chosen_r if chosen_r is not None else r_eff), metrics


# ==============================
# API pública: generalized_synth_gpu
# ==============================

def generalized_synth_gpu(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    outcome_col: str,
    treat_col: Optional[str] = None,
    adoption_time_col: Optional[str] = None,
    treated_units: Optional[Sequence[Any]] = None,
    control_units: Optional[Sequence[Any]] = None,
    pre_period: Optional[Tuple[Any, Any]] = None,
    post_period: Optional[Tuple[Any, Any]] = None,
    r: Optional[int] = None,
    r_grid: Tuple[int, int] = (1, 6),
    max_iter: int = 200,
    tol: float = 1e-4,
    prefer_gpu: bool = True,
    force_gpu: bool = False,
) -> GSCResult:
    """GSC vía matrix completion (EM+SVD) ejecutado en GPU con CuPy cuando está disponible.

    - Si `prefer_gpu=True` y hay GPU/CuPy, todo el núcleo iterativo corre en GPU.
    - Si no hay GPU, cae a NumPy (CPU) salvo que `force_gpu=True` (entonces lanza error).
    """
    if treat_col is None and adoption_time_col is None:
        raise ValueError("Debe especificar `treat_col` o `adoption_time_col`.")

    d = df.copy()
    if adoption_time_col is not None and (treat_col is None or treat_col not in d.columns):
        d = derive_treat_from_adoption(d, id_col, time_col, adoption_time_col, out_treat_col="__D__")
        treat_col_eff = "__D__"
    else:
        treat_col_eff = treat_col  # type: ignore

    needed = {id_col, time_col, outcome_col, treat_col_eff}
    missing_cols = [c for c in needed if c not in d.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")

    all_times = _sorted_unique(d[time_col])
    if pre_period is None or post_period is None:
        if adoption_time_col is not None and adoption_time_col in d.columns:
            adop_map = d[[id_col, adoption_time_col]].dropna().drop_duplicates(subset=[id_col])
            if adop_map.empty:
                raise ValueError("`adoption_time_col` no tiene adopciones válidas para definir post.")
            t0_min = adop_map[adoption_time_col].min()
            pre_period = (all_times[0], min([t for t in all_times if t < t0_min])) if any(t < t0_min for t in all_times) else None
            post_period = (min([t for t in all_times if t >= t0_min]), all_times[-1]) if any(t >= t0_min for t in all_times) else None
        else:
            raise ValueError("Si no pasa `pre_period`/`post_period`, necesita `adoption_time_col`.")
    if pre_period is None or post_period is None:
        raise ValueError("No se pudieron determinar `pre_period` y `post_period`.")

    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    # Paneles anchos
    Y_w, D_w = prepare_panel(d, id_col, time_col, outcome_col, treat_col_eff)
    units = Y_w.index.tolist()
    times = Y_w.columns.tolist()

    # Define conjuntos
    if treated_units is None:
        treated_units = (
            D_w.max(axis=1) if D_w is not None else d.groupby(id_col)[treat_col_eff].max()
        )
        treated_units = treated_units[treated_units > 0].index.tolist()
    if control_units is None:
        control_units = [u for u in units if u not in set(treated_units)]

    keep_units = [u for u in units if u in set(treated_units) or u in set(control_units)]
    Y_np = Y_w.loc[keep_units, :].to_numpy(dtype=float)
    if D_w is None:
        D_w2 = d.pivot(index=id_col, columns=time_col, values=treat_col_eff).reindex(index=keep_units, columns=times)
        D_np = D_w2.to_numpy(dtype=float)
    else:
        D_np = D_w.loc[keep_units, :].to_numpy(dtype=float)

    n, tdim = Y_np.shape

    # Backend (GPU/CPU)
    xp_wrap = _XP(prefer_gpu=prefer_gpu, force_gpu=force_gpu)
    xp = xp_wrap.xp

    # A GPU
    Y = xp_wrap.to_xp(Y_np, dtype=_np.float64)
    D = xp_wrap.to_xp(D_np, dtype=_np.float64)

    # Máscaras
    y_obs_mask = xp.isfinite(Y)
    pre_cols_np = _np.array([(pre_start <= tt <= pre_end) for tt in times], dtype=bool)
    post_cols_np = _np.array([(post_start <= tt <= post_end) for tt in times], dtype=bool)
    pre_cols = xp_wrap.to_xp(pre_cols_np, dtype=bool)
    post_cols = xp_wrap.to_xp(post_cols_np, dtype=bool)

    treated_post_mask = (D == 1.0) & post_cols.reshape(1, -1)
    obs_mask = y_obs_mask & (~treated_post_mask)

    # EM-SVD en GPU/CPU
    Y_filled, r_eff, em_metrics = _em_svd_xp(
        xp=xp,
        Y=Y,
        obs_mask=obs_mask,
        pre_mask=obs_mask & pre_cols.reshape(1, -1),
        r=r,
        r_grid=r_grid,
        max_iter=max_iter,
        tol=tol,
    )

    # Outputs principales
    Y_hat = Y_filled
    y0 = xp.full_like(Y_hat, xp.nan)
    y0[treated_post_mask] = Y_hat[treated_post_mask]

    tau = xp.full_like(Y_hat, xp.nan)
    tau[treated_post_mask] = Y[treated_post_mask] - y0[treated_post_mask]

    # A NumPy para pandas
    y0_np = xp_wrap.to_np(y0)
    tau_np = xp_wrap.to_np(tau)
    full_hat_np = xp_wrap.to_np(Y_hat)

    y0_df = pd.DataFrame(y0_np, index=keep_units, columns=times)
    tau_df = pd.DataFrame(tau_np, index=keep_units, columns=times)
    full_hat_df = pd.DataFrame(full_hat_np, index=keep_units, columns=times)

    # ATT global y por tiempo/unidad (CPU)
    treated_post_mask_np = (D_np == 1.0) & post_cols_np.reshape(1, -1)
    att = float(_np.nanmean(tau_np[treated_post_mask_np])) if _np.any(treated_post_mask_np) else _np.nan

    att_t_vals: List[float] = []
    att_t_idx: List[Any] = []
    for j, tt in enumerate(times):
        if not post_cols_np[j]:
            continue
        mask_col = treated_post_mask_np[:, j]
        if mask_col.any():
            att_t_idx.append(tt)
            att_t_vals.append(float(_np.nanmean(tau_np[mask_col, j])))
    att_t = pd.Series(att_t_vals, index=pd.Index(att_t_idx, name=time_col), name="att_t")

    att_i_vals: List[float] = []
    att_i_idx: List[Any] = []
    for i, uid in enumerate(keep_units):
        mask_row = treated_post_mask_np[i, :]
        if mask_row.any():
            att_i_idx.append(uid)
            att_i_vals.append(float(_np.nanmean(tau_np[i, mask_row])))
    att_i = pd.Series(att_i_vals, index=pd.Index(att_i_idx, name=id_col), name="att_i")

    # Factores/cargas desde SVD final (GPU/CPU y luego a NumPy)
    Yhat_no_nan = xp.nan_to_num(Y_hat, copy=False, nan=0.0)
    U, S, Vt = xp.linalg.svd(Yhat_no_nan, full_matrices=False)
    r_show = int(min(r_eff, U.shape[1], Vt.shape[0]))
    Ur, Sr, Vtr = U[:, :r_show], S[:r_show], Vt[:r_show, :]

    loadings = Ur @ xp.diag(xp.sqrt(Sr))
    factors = (Vtr.T) @ xp.diag(xp.sqrt(Sr))
    loadings_df = pd.DataFrame(xp_wrap.to_np(loadings), index=keep_units, columns=[f"f{k+1}" for k in range(r_show)])
    factors_df = pd.DataFrame(xp_wrap.to_np(factors), index=times, columns=[f"f{k+1}" for k in range(r_show)])

    # Métricas
    pre_mask_eval = obs_mask & pre_cols.reshape(1, -1)
    pre_rmse = float(_np.sqrt(_np.nanmean((full_hat_np[pre_mask_eval.get() if xp is _cp else pre_mask_eval] - Y_np[pre_mask_eval.get() if xp is _cp else pre_mask_eval]) ** 2)))) if bool(pre_mask_eval.any()) else _np.nan  # type: ignore

    metrics: Dict[str, Any] = {
        **em_metrics,
        "pre_rmse": pre_rmse,
        "n_units": int(n),
        "n_times": int(tdim),
        "n_treated": int(len([u for u in keep_units if u in set(treated_units)])),
        "backend": "cupy" if xp_wrap.using_gpu else "numpy",
        "device": "gpu" if xp_wrap.using_gpu else "cpu",
    }

    return GSCResult(
        att=att,
        att_t=att_t,
        att_i=att_i,
        y0_hat=y0_df,
        tau_it=tau_df,
        full_y_hat=full_hat_df,
        r=r_eff,
        factors=factors_df,
        loadings=loadings_df,
        metrics=metrics,
    )


def att_by_event_time(
    tau_it: pd.DataFrame,
    adoption_times: Dict[Any, Any],
    time_index: Sequence[Any],
) -> pd.Series:
    """Calcula ATT por "event time" k = t - t0_i usando la matriz tau_it (CPU)."""
    time_pos = {t: idx for idx, t in enumerate(list(time_index))}
    contrib: Dict[int, List[float]] = {}
    for i, uid in enumerate(tau_it.index):
        t0 = adoption_times.get(uid, None)
        if pd.isna(t0) or t0 is None:
            continue
        pos0 = time_pos.get(t0, None)
        if pos0 is None:
            continue
        row = tau_it.loc[uid]
        for tval, eff in row.items():
            if pd.isna(eff):
                continue
            k = time_pos[tval] - pos0
            contrib.setdefault(k, []).append(float(eff))
    if not contrib:
        return pd.Series(dtype=float, name="att_event_time")
    ks = sorted(contrib.keys())
    vals = [float(_np.mean(contrib[k])) for k in ks]
    return pd.Series(vals, index=pd.Index(ks, name="event_time"), name="att_event_time")


__all__ = [
    "GSCResult",
    "derive_treat_from_adoption",
    "prepare_panel",
    "find_donors",
    "generalized_synth_gpu",
    "att_by_event_time",
]