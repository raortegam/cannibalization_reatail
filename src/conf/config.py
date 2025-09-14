# config.py
from datetime import datetime

# Directorio base de datos (cámbialo si lo necesitas)
DATA_DIR = r"D:\repos\cannibalization_reatail\.data\raw_data"

# Parámetros del capítulo / modelado
THETA_DIAS_PROMO_SEMANA: float = 2 / 7     # θ: al menos 2 días de promo en la semana
FOURIER_K: int = 2                         # Nº de armónicos anuales (~52 semanas)
LAGS = [1, 2, 4, 8]                        # Lags semanales de la variable dependiente

# Choque institucional (terremoto)
EQ_DATE = datetime(2016, 4, 16)            # 2016-04-16

# Columnas a leer en bruto (por si quieres centralizarlas)
VENTAS_COLS = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
VENTAS_PARSE_DATES = ["date"]
ITEMS_COLS = ["item_nbr", "family", "class", "perishable"]
HOLIDAYS_PARSE_DATES = ["date"]
TRANSACTIONS_PARSE_DATES = ["date"]
OIL_PARSE_DATES = ["date"]
OUTPUT_DIR = r"D:\repos\cannibalization_reatail\.data\processed_data"

from datetime import datetime

# --------- Expectativas de validación (puedes ajustarlas) ---------
EXPECTED_DATE_MIN = datetime(2013, 1, 1)
EXPECTED_DATE_MAX = datetime(2017, 8, 15)

# Claves / columnas esperadas por dataset
SCHEMA_EXPECTED = {
    "ventas": {
        "required_cols": ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        "key": ["date", "store_nbr", "item_nbr"],
    },
    "items": {
        "required_cols": ["item_nbr", "family", "class", "perishable"],
        "key": ["item_nbr"],
    },
    "stores": {
        "required_cols": ["store_nbr", "city", "state", "type", "cluster"],
        "key": ["store_nbr"],
    },
    "hol": {
        "required_cols": ["date", "type", "locale", "locale_name", "description", "transferred"],
        "key": ["date", "locale", "locale_name", "description"],
    },
    "trans": {
        "required_cols": ["date", "store_nbr", "transactions"],
        "key": ["date", "store_nbr"],
    },
    "oil": {
        "required_cols": ["date", "dcoilwtico"],
        "key": ["date"],
    },
}

# Expectativas agregadas
EXPECTED_MIN_FAMILIES = 33
EXPECTED_MIN_STORES = 54

# Directorio de salida para reportes de calidad
REPORT_DIR = DATA_DIR  # o usa otro path si prefieres