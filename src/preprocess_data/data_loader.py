import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
from src.conf import config

# data_loader.py

from typing import Dict
from src.utils.utils import read_csv_safe


def load_ventas() -> pd.DataFrame:
    return read_csv_safe(
        "train.csv",
        usecols=config.VENTAS_COLS,
        parse_dates=config.VENTAS_PARSE_DATES
    )

def load_items() -> pd.DataFrame:
    return read_csv_safe(
        "items.csv",
        usecols=config.ITEMS_COLS
    )

def load_stores() -> pd.DataFrame:
    return read_csv_safe("stores.csv")

def load_holidays() -> pd.DataFrame:
    return read_csv_safe("holidays_events.csv", parse_dates=config.HOLIDAYS_PARSE_DATES)

def load_transactions() -> pd.DataFrame:
    return read_csv_safe("transactions.csv", parse_dates=config.TRANSACTIONS_PARSE_DATES)

def load_oil() -> pd.DataFrame:
    return read_csv_safe("oil.csv", parse_dates=config.OIL_PARSE_DATES)

def load_all_raw() -> Dict[str, pd.DataFrame]:
    """
    Carga todos los datasets crudos.
    """
    return {
        "ventas": load_ventas(),
        "items": load_items(),
        "stores": load_stores(),
        "hol": load_holidays(),
        "trans": load_transactions(),
        "oil": load_oil(),
    }