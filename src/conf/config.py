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