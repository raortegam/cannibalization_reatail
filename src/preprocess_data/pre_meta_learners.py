# pre_metalearners.py
# -*- coding: utf-8 -*-
"""
Prepara datasets listos para Meta-learners (T- y X-Learner) a partir de:
panel_labeled, episodes, windows_long.

Salidas clave:
- ml_global:
    ml_df:        DataFrame base con X, y, T y metadatos
    features:     lista de columnas de features tras codificación
    target_col:   nombre de la columna objetivo (outcome)
    treat_col:    columna binaria de tratamiento
    encoders:     dict de codificación de categóricas (si ordinal)
    folds:        lista de splits forward-chaining (train_idx, valid_idx)
    config_used:  parámetros efectivos

- t_learner_views:
    treated:      DataFrame de entrenamiento para el modelo de tratados
    control:      DataFrame de entrenamiento para el modelo de control

- x_learner_proto:
    folds:        mismos folds para cross-fitting
    groups:       dict con índices de tratados/control por fold (útil para entrenar modelos 1 y 0)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.conf import config  # type: ignore

# =========================
# Configuración (con fallbacks)
# =========================
OUTCOME_COL_DEFAULT: str = getattr(config, "OUTCOME_COL_DEFAULT_ML", "y_isw")
TREATMENT_COL_DEFAULT: str = getattr(config, "TREATMENT_COL_DEFAULT", "E_bin_isw")  # se usa 'treatment_used' del panel_labeled
ID_COLS: Tuple[str, str] = tuple(getattr(config, "EPISODE_GROUPBY", ("store_nbr", "item_nbr")))  # type: ignore
TIME_COL: str = "w_idx"

# Roles a excluir por defecto para entrenar (evitar contaminación)
EXCLUDE_ROLES_FOR_TRAIN: Tuple[str, ...] = tuple(getattr(config, "ML_EXCLUDE_ROLES_FOR_TRAIN", ("guard_pre", "guard_post")))
INCLUDE_ROLES_FOR_TRAIN: Optional[Tuple[str, ...]] = None  # si se quiere restringir a un subconjunto explícito

# Codificación de categóricas
ENCODING_METHOD: str = getattr(config, "ML_ENCODING_METHOD", "ordinal")  # {"ordinal","onehot"}

# Cross-validation temporal (forward-chaining global por w_idx)
INIT_TRAIN_WEEKS: int = getattr(config, "ML_INIT_TRAIN_WEEKS", 52)
TEST_WEEKS: int = getattr(config, "ML_TEST_WEEKS", 8)
STEP_WEEKS: int = getattr(config, "ML_STEP_WEEKS", 4)
MIN_TRAIN_WEEKS: int = getattr(config, "ML_MIN_TRAIN_WEEKS", 32)


# =========================
# Utils internos
# =========================
def _infer_feature_columns(panel: pd.DataFrame,
                           target_col: str,
                           treat_col: str) -> Tuple[List[str], List[str]]:
    """
    Devuelve (num_features, cat_features) inferidos del panel.
    Excluye: IDs, tiempo, target, tratamiento y columnas triviales.
    """
    drop = set(ID_COLS) | {TIME_COL, "week_start", target_col, treat_col,
                           "Y_isw", "Ptilde_isw", "treatment_used", "episode_id_primary",
                           "event_time", "in_window", "window_role", "episode_start_w_idx"}
    num_cols, cat_cols = [], []
    for c in panel.columns:
        if c in drop:
            continue
        dt = panel[c].dtype
        if pd.api.types.is_numeric_dtype(dt):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    # Reordenar: lags, fourier primero (suelen ser útiles)
    num_cols = sorted(num_cols, key=lambda x: (not (x.startswith("lag") or x.startswith("fourier_")), x))
    return num_cols, cat_cols


def _encode_categoricals(df: pd.DataFrame, cat_cols: List[str], method: str = "ordinal") -> Tuple[pd.DataFrame, Dict[str, List]]:
    """
    Devuelve df codificado y diccionario {col: categorias}.
    - ordinal: df[col+"_enc"] con codes; elimina original.
    - onehot:  get_dummies (drop_first=False), elimina original.
    """
    encoders: Dict[str, List] = {}
    out = df.copy()
    if not cat_cols:
        return out, encoders

    if method == "ordinal":
        new_cols = []
        for c in cat_cols:
            as_cat = pd.Categorical(out[c])
            out[c + "_enc"] = as_cat.codes.astype("int32")
            encoders[c] = list(as_cat.categories)
            new_cols.append(c + "_enc")
        out = out.drop(columns=cat_cols)
        return out, encoders

    # onehot
    out = pd.get_dummies(out, columns=cat_cols, dummy_na=False)
    for c in cat_cols:
        # guardar base para reproducibilidad, aunque en onehot no se usan categorías explícitas
        encoders[c] = []
    return out, encoders


def _time_based_folds(df: pd.DataFrame,
                      time_col: str = TIME_COL,
                      init_train_weeks: int = INIT_TRAIN_WEEKS,
                      test_weeks: int = TEST_WEEKS,
                      step_weeks: int = STEP_WEEKS,
                      min_train_weeks: int = MIN_TRAIN_WEEKS) -> List[Dict[str, np.ndarray]]:
    """
    Folds forward-chaining a nivel global (por w_idx).
    Cada fold k define:
      - train: w_idx <= train_end_k
      - valid: train_end_k < w_idx <= train_end_k + test_weeks
    """
    weeks = np.sort(df[time_col].dropna().unique().astype(int))
    if weeks.size < (init_train_weeks + test_weeks):
        return []

    folds = []
    start = weeks.min()
    train_end = start + init_train_weeks - 1
    last_week = weeks.max()

    while (train_end + test_weeks) <= last_week:
        valid_end = train_end + test_weeks
        train_mask = df[time_col].between(start, train_end)
        valid_mask = df[time_col].between(train_end + 1, valid_end)
        # Chequeo min_train
        if df.loc[train_mask].shape[0] >= min_train_weeks * 10:  # heurística ~10 unidades
            folds.append({
                "train_idx": train_mask.to_numpy().nonzero()[0],
                "valid_idx": valid_mask.to_numpy().nonzero()[0],
                "train_range": (int(start), int(train_end)),
                "valid_range": (int(train_end + 1), int(valid_end)),
            })
        train_end += step_weeks

    return folds


def _filter_roles_for_train(df: pd.DataFrame,
                            exclude_roles: Optional[Iterable[str]] = EXCLUDE_ROLES_FOR_TRAIN,
                            include_roles: Optional[Iterable[str]] = INCLUDE_ROLES_FOR_TRAIN) -> pd.DataFrame:
    out = df.copy()
    if include_roles is not None:
        out = out[out["window_role"].isin(list(include_roles))]
    if exclude_roles is not None:
        out = out[~out["window_role"].isin(list(exclude_roles))]
    return out


# =========================
# API principal
# =========================
def prepare_ml_datasets(
    panel_labeled: pd.DataFrame,
    episodes: pd.DataFrame,
    windows_long: pd.DataFrame,
    outcome_col: str = OUTCOME_COL_DEFAULT,
    treat_col: str = "treatment_used",
    encoding_method: str = ENCODING_METHOD,
    exclude_roles_for_train: Optional[Iterable[str]] = EXCLUDE_ROLES_FOR_TRAIN,
    include_roles_for_train: Optional[Iterable[str]] = INCLUDE_ROLES_FOR_TRAIN,
    drop_na: bool = True,
) -> Tuple[Dict[str, object], Dict[str, pd.DataFrame], Dict[str, object]]:
    """
    Construye datasets para T- y X-Learner (estructura de splits y vistas por grupo).

    Returns
    -------
    ml_global : dict
        - ml_df:       DataFrame base con X,y,T + metadatos (IDs, w_idx, window_role, event_time)
        - features:    lista de columnas finales de features (tras encodings)
        - target_col:  outcome a modelar
        - treat_col:   nombre de columna de tratamiento (binaria)
        - encoders:    mapeos de categóricas (si ordinal)
        - folds:       lista de folds forward-chaining (train_idx, valid_idx)
        - config_used: parámetros efectivos
    t_learner_views : dict
        - treated:     df de entrenamiento para modelo de tratados (T=1)
        - control:     df de entrenamiento para modelo de control (T=0)
    x_learner_proto : dict
        - folds:       folds para cross-fitting
        - groups:      {'treated': list[np.ndarray], 'control': list[np.ndarray]} índices por fold
    """
    assert outcome_col in panel_labeled.columns, f"Outcome {outcome_col} no existe."
    assert treat_col in panel_labeled.columns, "Se requiere 'treatment_used' en panel_labeled."

    # 1) Base y selección de columnas
    base_cols = list(ID_COLS) + [TIME_COL, "week_start", "window_role", "event_time", treat_col, outcome_col]
    others = [c for c in panel_labeled.columns if c not in base_cols]
    df = panel_labeled[base_cols + others].copy()

    # 2) Filtrado por roles (evitar guard bands por defecto)
    df = _filter_roles_for_train(df,
                                 exclude_roles=exclude_roles_for_train,
                                 include_roles=include_roles_for_train)

    # 3) Inferir features num/cat
    num_feats, cat_feats = _infer_feature_columns(df, outcome_col, treat_col)

    # 4) Codificar categóricas
    df_enc, encoders = _encode_categoricals(df, cat_feats, method=encoding_method)

    # 5) Armar lista final de features (excluir target y tratamiento por si quedaron en num_feats)
    feature_cols = [c for c in num_feats if c in df_enc.columns]
    # Si codificación onehot, agregar nuevas columnas
    if encoding_method == "onehot":
        added = [c for c in df_enc.columns if c not in df.columns and c != outcome_col]
        feature_cols += [c for c in added if not c.endswith(treat_col)]

    # 6) Limpieza de NAs si se solicita
    if drop_na:
        keep = base_cols + [c for c in feature_cols if c not in base_cols]
        df_enc = df_enc.dropna(subset=[outcome_col, treat_col])
        df_enc = df_enc.dropna(subset=feature_cols, how="any")

    # 7) Folds temporales forward-chaining
    folds = _time_based_folds(df_enc, time_col=TIME_COL)

    # 8) Vistas T-Learner (dos datasets)
    treated_df = df_enc[df_enc[treat_col] == 1].copy()
    control_df = df_enc[df_enc[treat_col] == 0].copy()

    # 9) Protocolo X-Learner (estructura para cross-fitting)
    #    Para cada fold, entrenar modelos m1 en tratados y m0 en controles sobre 'train_idx';
    #    luego predecir OOF en 'valid_idx' y construir pseudo-outcomes.
    treated_groups_by_fold = []
    control_groups_by_fold = []
    for f in folds:
        tr_idx = f["train_idx"]
        treated_groups_by_fold.append(treated_df.index.intersection(df_enc.iloc[tr_idx].index).to_numpy())
        control_groups_by_fold.append(control_df.index.intersection(df_enc.iloc[tr_idx].index).to_numpy())

    ml_global: Dict[str, object] = {
        "ml_df": df_enc,
        "features": feature_cols,
        "target_col": outcome_col,
        "treat_col": treat_col,
        "encoders": encoders,
        "folds": folds,
        "config_used": {
            "encoding_method": encoding_method,
            "exclude_roles_for_train": tuple(exclude_roles_for_train) if exclude_roles_for_train is not None else None,
            "include_roles_for_train": tuple(include_roles_for_train) if include_roles_for_train is not None else None,
            "init_train_weeks": INIT_TRAIN_WEEKS,
            "test_weeks": TEST_WEEKS,
            "step_weeks": STEP_WEEKS,
            "min_train_weeks": MIN_TRAIN_WEEKS,
        }
    }

    t_learner_views = {
        "treated": treated_df,
        "control": control_df,
    }

    x_learner_proto = {
        "folds": folds,
        "groups": {
            "treated": treated_groups_by_fold,
            "control": control_groups_by_fold,
        }
    }

    return ml_global, t_learner_views, x_learner_proto


# =========================
# Ejemplo mínimo (opcional)
# =========================
if __name__ == "__main__":
    # panel_labeled, episodes, windows_long = ...
    # ml_global, t_views, x_proto = prepare_ml_datasets(panel_labeled, episodes, windows_long)
    # print(ml_global["ml_df"].shape, len(ml_global["folds"]))
    pass