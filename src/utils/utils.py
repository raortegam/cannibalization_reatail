# utils.py
import os
import numpy as np
import pandas as pd
from typing import Iterable, List
from src.conf import config

def read_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    """
    Lee un CSV desde config.DATA_DIR con pd.read_csv.
    """
    full_path = os.path.join(config.DATA_DIR, path)
    return pd.read_csv(full_path, **kwargs)

def add_week_start(df: pd.DataFrame, date_col: str = "date", out_col: str = "week_start") -> pd.DataFrame:
    """
    Agrega la columna 'week_start' con el lunes de cada semana ISO.
    """
    df[out_col] = df[date_col] - pd.to_timedelta(df[date_col].dt.weekday, unit="D")
    return df

def safe_bool_to_int(s: pd.Series) -> pd.Series:
    """
    Convierte {True/False, "True"/"False", 1/0, NaN} → {1,0} (int8) de forma robusta.
    """
    if s.dtype == bool:
        return s.astype("int8")
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype("int8").clip(0, 1)
    # objeto/cadena
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "t", "yes"]).astype("int8")

def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Asegura que existan todas las columnas listadas en 'cols'. Si falta alguna, la crea con NaN.
    """
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def cast_if_exists(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    """
    Castea tipos solo si la columna existe (evita errores de KeyError).
    """
    for col, dt in dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dt)
    return df

import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple, Dict, Optional

def ensure_binary_series(s: pd.Series, name: str) -> pd.Series:
    """
    Asegura que la serie sea {0,1} (o booleana) y devuelve int8 con NaNs→0.
    """
    if s.dtype == bool:
        out = s.fillna(False).astype("int8")
    else:
        out = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0, upper=1).astype("int8")
    out.name = name
    return out

def pick_treatment_column(df: pd.DataFrame,
                          preferred: str,
                          fallbacks: Iterable[str] = ("Ptilde_isw",)) -> str:
    """
    Elige la columna de tratamiento binario existente en df.
    Prioriza 'preferred'; si no está, prueba fallbacks; si ninguna existe, levanta error.
    """
    if preferred in df.columns:
        return preferred
    for c in fallbacks:
        if c in df.columns:
            return c
    raise KeyError(
        f"No se encontró columna de tratamiento. Probé: {preferred} y {list(fallbacks)}."
    )

def rle_runs_on_widx(w_idx: np.ndarray, treat: np.ndarray,
                     min_run: int = 1, max_gap: int = 0) -> List[Tuple[int, int]]:
    """
    Detecta episodios como runs de semanas con treat==1 permitiendo 'max_gap' ceros entre 1's.
    - w_idx: índices semanales (enteros, crecientes, consecutivos a escala global)
    - treat: vector {0,1} alineado con w_idx
    Devuelve lista de (start_w, end_w) en escala w_idx para cada episodio detectado.
    Nota: el episodio abarca desde el primer '1' hasta el último '1' del run fusionado.
    """
    ones = w_idx[treat == 1]
    if ones.size == 0:
        return []
    episodes: List[Tuple[int, int]] = []
    start = int(ones[0])
    prev  = int(ones[0])
    for x in ones[1:]:
        x = int(x)
        if x - prev <= (max_gap + 1):
            # sigue el run (casi consecutivo)
            prev = x
        else:
            # termina episodio anterior
            if (prev - start + 1) >= min_run:
                episodes.append((start, prev))
            # nuevo episodio
            start = x
            prev  = x
    # último episodio
    if (prev - start + 1) >= min_run:
        episodes.append((start, prev))
    return episodes

def make_windows_for_episode(start_w: int, end_w: int,
                             K_pre: int, G_pre: int,
                             K_post: int, G_post: int) -> Dict[str, Tuple[int, int]]:
    """
    Construye ventanas (rango inclusive de w_idx) alrededor de un episodio (start_w..end_w).
      pre:        [start_w - G_pre - K_pre, start_w - G_pre - 1]
      guard_pre:  [start_w - G_pre,        start_w - 1]
      treat:      [start_w,                end_w]            (solo semanas con treat==1 se etiquetan 'treat')
      guard_post: [end_w + 1,              end_w + G_post]
      post:       [end_w + G_post + 1,     end_w + G_post + K_post]
    Si K_* o G_* = 0, el rango correspondiente puede ser inválido (iniciar > terminar) y se ignora luego.
    """
    windows = {
        "pre":        (start_w - G_pre - K_pre, start_w - G_pre - 1),
        "guard_pre":  (start_w - G_pre,         start_w - 1),
        "treat":      (start_w,                 end_w),
        "guard_post": (end_w + 1,               end_w + G_post),
        "post":       (end_w + G_post + 1,      end_w + G_post + K_post),
    }
    return windows

def expand_range_to_set(r: Tuple[int, int]) -> set:
    a, b = int(r[0]), int(r[1])
    if a > b:
        return set()
    return set(range(a, b + 1))

def apply_role_priority(existing: Optional[str], candidate: str,
                        priority: List[str]) -> str:
    """
    Devuelve el rol con mayor prioridad (índice más bajo en 'priority').
    """
    if existing is None:
        return candidate
    return candidate if priority.index(candidate) < priority.index(existing) else existing