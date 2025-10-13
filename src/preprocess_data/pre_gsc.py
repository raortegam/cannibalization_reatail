# pre_gsc.py
# -*- coding: utf-8 -*-
"""
Prepara los datasets necesarios para aplicar Generalized Synthetic Control (GSC)
a partir de las salidas: panel_labeled, episodes, windows_long.

Salidas clave:
- gsc_global:
    Y:            DataFrame (unidades x w_idx) con el outcome (p.ej. y_isw)
    T:            DataFrame (unidades x w_idx) con el tratamiento usado (treatment_used)
    controls_cube: dict[str -> DataFrame] (opcional) controles (unidades x w_idx)
    unit_index:   Index con (store_nbr,item_nbr)
    time_index:   Index de w_idx
    week_start_map: Series w_idx -> week_start
    unit_meta:    DataFrame con metadatos por unidad (family, etc.)

- episodes_data: list[dict] – uno por episodio, con:
    episode_meta:   dict metadatos del episodio
    treated_unit:   tuple (store_nbr,item_nbr)
    pre_idx:        list[int] w_idx en pre
    treat_idx:      list[int] w_idx en treat
    post_idx:       list[int] w_idx en post
    donor_units:    list[tuple] unidades candidatas a donantes según política
    Y_pre, Y_treat, Y_post:    submatrices outcome
    X_pre, X_treat, X_post:    dict[str-> DataFrame] controles (si se solicitaron)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Permite imports tipo src.*
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.conf import config  # type: ignore


# =========================
# Configuración (con fallbacks)
# =========================
OUTCOME_COL_DEFAULT: str = getattr(config, "OUTCOME_COL_DEFAULT_GSC", "y_isw")
ID_COLS: Tuple[str, str] = tuple(getattr(config, "EPISODE_GROUPBY", ("store_nbr", "item_nbr")))  # type: ignore
TIME_COL: str = "w_idx"
WEEK_START_COL: str = "week_start"

MIN_PRE_WEEKS: int = getattr(config, "GSC_MIN_PRE_WEEKS", 6)

# Controles por defecto si no se pasan explícitamente:
GSC_CONTROLS_DEFAULT: Optional[List[str]] = getattr(
    config, "GSC_CONTROLS_DEFAULT", None
)

# Política de donantes
DONOR_POLICY: str = getattr(config, "GSC_DONOR_POLICY", "not_treated_in_window")  # {"not_treated_in_window", "all"}
DONOR_RESTRICT_SAME_FAMILY: bool = getattr(config, "GSC_DONOR_RESTRICT_SAME_FAMILY", True)
DONOR_RESTRICT_SAME_STORE: bool = getattr(config, "GSC_DONOR_RESTRICT_SAME_STORE", False)
EXCLUDE_ROLES_FOR_DONORS: Tuple[str, ...] = tuple(
    getattr(config, "GSC_EXCLUDE_ROLES_FOR_DONORS", ("pre", "guard_pre", "treat", "guard_post"))
)


# =========================
# Utils internos
# =========================
def _infer_controls_columns(panel: pd.DataFrame) -> List[str]:
    if GSC_CONTROLS_DEFAULT is not None:
        return [c for c in GSC_CONTROLS_DEFAULT if c in panel.columns]

    # Heurística: columnas frecuentes en tu pipeline
    candidates = []
    for c in ["H_nat", "H_reg", "H_loc", "log_F_sw", "O_w", "eq_shock", "perishable", "cluster"]:
        if c in panel.columns:
            candidates.append(c)

    # Fourier y lags
    candidates += [c for c in panel.columns if c.startswith("fourier_s")]
    candidates += [c for c in panel.columns if c.startswith("fourier_c")]
    candidates += [c for c in panel.columns if c.startswith("lag")]

    # Evitar duplicados y ordenar
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _create_wide(panel: pd.DataFrame, value_col: str,
                 id_cols: Sequence[str] = ID_COLS, time_col: str = TIME_COL) -> pd.DataFrame:
    tmp = panel[list(id_cols) + [time_col, value_col]].copy()
    tmp[time_col] = tmp[time_col].astype(int)
    wide = tmp.pivot_table(index=list(id_cols), columns=time_col, values=value_col, aggfunc="first")
    wide = wide.sort_index(axis=1)
    return wide


def _week_map(panel: pd.DataFrame, time_col: str = TIME_COL, week_col: str = WEEK_START_COL) -> pd.Series:
    s = (panel[[time_col, week_col]]
         .drop_duplicates(time_col)
         .set_index(time_col)[week_col]
         .sort_index())
    return s


def _build_unit_meta(panel: pd.DataFrame, id_cols: Sequence[str] = ID_COLS) -> pd.DataFrame:
    meta_cols = [c for c in ["family", "class", "perishable", "cluster", "city", "state", "type"] if c in panel.columns]
    if not meta_cols:
        return (panel[list(id_cols)]
                .drop_duplicates()
                .set_index(list(id_cols)))
    meta = (panel[list(id_cols) + meta_cols]
            .dropna(subset=[id_cols[0], id_cols[1]])
            .sort_values(list(id_cols))
            .groupby(list(id_cols), as_index=True)
            .agg(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan))
    return meta


def _episode_indices(windows_long: pd.DataFrame,
                     keys: Tuple,
                     episode_id: int) -> Tuple[List[int], List[int], List[int]]:
    filt = np.ones(len(windows_long), dtype=bool)
    for k, v in zip(ID_COLS, keys):
        filt &= (windows_long[k] == v).to_numpy()
    sub = windows_long[filt & (windows_long["episode_id"] == episode_id)]
    pre = sorted(sub.loc[sub["role"] == "pre", "w_idx"].astype(int).unique().tolist())
    treat = sorted(sub.loc[sub["role"] == "treat", "w_idx"].astype(int).unique().tolist())
    post = sorted(sub.loc[sub["role"] == "post", "w_idx"].astype(int).unique().tolist())
    return pre, treat, post


def _donor_units_for_episode(panel_labeled: pd.DataFrame,
                             T_wide: pd.DataFrame,
                             unit_keys: Tuple,
                             pre_idx: List[int],
                             treat_idx: List[int],
                             post_idx: List[int],
                             unit_meta: pd.DataFrame) -> List[Tuple]:
    """Selecciona donantes según política y restricciones."""
    all_units = list(T_wide.index)
    treated_unit = tuple(unit_keys)
    candidates = [u for u in all_units if tuple(u) != treated_unit]

    # Restricciones de homogeneidad
    if DONOR_RESTRICT_SAME_STORE and "store_nbr" in ID_COLS:
        same_store = [u for u in candidates if u[ID_COLS.index("store_nbr")] == treated_unit[ID_COLS.index("store_nbr")]]
        candidates = same_store or candidates
    if DONOR_RESTRICT_SAME_FAMILY and "family" in unit_meta.columns:
        fam = unit_meta.loc[treated_unit, "family"] if treated_unit in unit_meta.index else np.nan
        if pd.notna(fam):
            same_family = [u for u in candidates
                           if (u in unit_meta.index and unit_meta.loc[u, "family"] == fam)]
            candidates = same_family or candidates

    if DONOR_POLICY == "all":
        return candidates

    # Excluir candidatos que tengan tratamiento en pre/guard_pre/treat/guard_post del episodio
    # Construimos el set de semanas a vetar (pre + guard_pre + treat + guard_post)
    # Lo hacemos desde panel_labeled para el episodio del unit_keys.
    ep_mask = np.ones(len(panel_labeled), dtype=bool)
    for k, v in zip(ID_COLS, unit_keys):
        ep_mask &= (panel_labeled[k] == v).to_numpy()
    ep_w = panel_labeled.loc[ep_mask & (panel_labeled["in_window"] == 1)]
    veto_roles = set(EXCLUDE_ROLES_FOR_DONORS)
    veto_weeks = set(ep_w.loc[ep_w["window_role"].isin(veto_roles), TIME_COL].astype(int).tolist())

    donors = []
    for u in candidates:
        t_series = T_wide.loc[u, sorted(veto_weeks)]
        # Si no hay columnas (p.ej., windows truncadas fuera del rango de T), consideramos válido
        if t_series.shape[0] == 0:
            donors.append(u)
        else:
            # Debe ser todo 0 en semanas vetadas
            val = np.nan_to_num(t_series.to_numpy(), nan=0.0)
            if (val == 0).all():
                donors.append(u)
    return donors


def _subset_controls(cube: Optional[Dict[str, pd.DataFrame]],
                     units: List[Tuple],
                     times: List[int]) -> Optional[Dict[str, pd.DataFrame]]:
    if cube is None:
        return None
    times_sorted = sorted(times)
    return {k: v.loc[units, times_sorted] for k, v in cube.items() if isinstance(v, pd.DataFrame)}


# =========================
# API principal
# =========================
def prepare_gsc_datasets(
    panel_labeled: pd.DataFrame,
    episodes: pd.DataFrame,
    windows_long: pd.DataFrame,
    outcome_col: str = OUTCOME_COL_DEFAULT,
    controls_cols: Optional[List[str]] = None,
    min_pre_weeks: int = MIN_PRE_WEEKS,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Construye los artefactos para GSC.

    Parameters
    ----------
    panel_labeled : DataFrame
        Panel con etiquetas (salida de build_treatment_design).
    episodes : DataFrame
        Una fila por episodio.
    windows_long : DataFrame
        Ventanas por episodio y semana.
    outcome_col : str
        Columna de outcome a usar (por defecto 'y_isw').
    controls_cols : list[str] | None
        Lista de controles a pivotear (unidades x w_idx). Si None, se infiere.
    min_pre_weeks : int
        Mínimo de semanas en 'pre' para mantener el episodio.

    Returns
    -------
    gsc_global : dict
        Artefactos globales (Y, T, controles, índices, week_map, unit_meta).
    episodes_data : list[dict]
        Lista de estructuras por episodio con submatrices y metadatos.
    """
    assert outcome_col in panel_labeled.columns, f"Outcome {outcome_col} no existe en panel."

    # Outcome y tratamiento en formato ancho (unidades x tiempo)
    Y = _create_wide(panel_labeled, outcome_col, ID_COLS, TIME_COL)
    T = _create_wide(panel_labeled, "treatment_used", ID_COLS, TIME_COL)
    week_map = _week_map(panel_labeled, TIME_COL, WEEK_START_COL)
    unit_meta = _build_unit_meta(panel_labeled, ID_COLS)

    # Controles
    if controls_cols is None:
        controls_cols = _infer_controls_columns(panel_labeled)
    controls_cube = None
    if controls_cols:
        controls_cube = {c: _create_wide(panel_labeled, c, ID_COLS, TIME_COL) for c in controls_cols}

    episodes_data: List[Dict[str, object]] = []
    # Iterar episodios
    for row in episodes.itertuples(index=False):
        unit_keys = tuple(getattr(row, c) for c in ID_COLS)
        episode_id = int(getattr(row, "episode_id"))
        pre_idx, treat_idx, post_idx = _episode_indices(windows_long, unit_keys, episode_id)

        # Validaciones mínimas
        if len(treat_idx) == 0 or len(pre_idx) < min_pre_weeks:
            continue

        # Donantes
        donors = _donor_units_for_episode(panel_labeled, T, unit_keys, pre_idx, treat_idx, post_idx, unit_meta)
        units_subset = donors + [unit_keys]

        # Submatrices outcome (ordenadas por tiempo)
        pre_sorted = sorted([t for t in pre_idx if t in Y.columns])
        treat_sorted = sorted([t for t in treat_idx if t in Y.columns])
        post_sorted = sorted([t for t in post_idx if t in Y.columns])

        Y_pre = Y.loc[units_subset, pre_sorted]
        Y_treat = Y.loc[units_subset, treat_sorted]
        Y_post = Y.loc[units_subset, post_sorted] if post_sorted else pd.DataFrame(index=units_subset, columns=[])

        # Controles (opcional)
        X_pre = _subset_controls(controls_cube, units_subset, pre_sorted)
        X_treat = _subset_controls(controls_cube, units_subset, treat_sorted)
        X_post = _subset_controls(controls_cube, units_subset, post_sorted) if post_sorted else None

        # Empaquetar
        episode_meta = {c: getattr(row, c) for c in episodes.columns}
        episodes_data.append({
            "episode_meta": episode_meta,
            "treated_unit": unit_keys,
            "pre_idx": pre_sorted,
            "treat_idx": treat_sorted,
            "post_idx": post_sorted,
            "donor_units": donors,
            "Y_pre": Y_pre,
            "Y_treat": Y_treat,
            "Y_post": Y_post,
            "X_pre": X_pre,
            "X_treat": X_treat,
            "X_post": X_post,
            "week_start_map": week_map,
        })

    gsc_global: Dict[str, object] = {
        "Y": Y, "T": T,
        "controls_cube": controls_cube,
        "unit_index": Y.index,
        "time_index": Y.columns,
        "week_start_map": week_map,
        "unit_meta": unit_meta,
        "config_used": {
            "outcome_col": outcome_col,
            "controls_cols": controls_cols,
            "donor_policy": DONOR_POLICY,
            "donor_restrict_same_family": DONOR_RESTRICT_SAME_FAMILY,
            "donor_restrict_same_store": DONOR_RESTRICT_SAME_STORE,
            "min_pre_weeks": min_pre_weeks,
        }
    }
    return gsc_global, episodes_data


# =========================
# Ejemplo mínimo (opcional)
# =========================
if __name__ == "__main__":
    # panel_labeled, episodes, windows_long = ... # cargar
    # gsc_global, episodes_data = prepare_gsc_datasets(panel_labeled, episodes, windows_long)
    # print(len(episodes_data), "episodios listos")
    pass