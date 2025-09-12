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