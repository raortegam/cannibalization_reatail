# config.py
from datetime import datetime

# Directorio base de datos (cámbialo si lo necesitas)
DATA_DIR = r"D:\repos\cannibalization_reatail\.data\raw_data"

# Parámetros del capítulo / modelado
THETA_DIAS_PROMO_SEMANA: float = 2 / 7     # θ: al menos 2 días de promo en la semana
FOURIER_K: int = 2                         # Nº de armónicos anuales (~52 semanas)
LAGS = [1, 2, 4, 13, 26, 52]                        # Lags semanales de la variable dependiente

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

# =========================
# Tratamiento y episodios
# =========================
# Columna binaria a usar como tratamiento por defecto
TREATMENT_COL_DEFAULT = "E_bin_isw"  # fallback a Ptilde_isw si no existe

# Agrupación para detectar episodios (unidades tratables)
EPISODE_GROUPBY = ("store_nbr", "item_nbr")

# Definición de episodios (runs de tratamiento)
EPISODE_MIN_RUN = 1     # mínimo de semanas "1" dentro del episodio
EPISODE_MAX_GAP = 0     # permitir huecos de hasta 'g' semanas entre 1's para fusionar runs

# Ventanas alrededor del episodio con bandas de protección
K_PRE  = 8   # semanas de pre-periodo (para ajuste/validación)
G_PRE  = 1   # guard band antes del episodio (excluida del ajuste)
K_POST = 12  # semanas de post-periodo (para estimar efectos)
G_POST = 2   # guard band después del episodio (excluida de la estimación)

# Prioridad al colapsar etiquetas si un (i,s,w) cae en varias ventanas de distintos episodios
WINDOW_ROLE_PRIORITY = ["treat", "guard_pre", "pre", "guard_post", "post", "none"]

# Política ante solapamientos entre episodios dentro de la misma unidad
EPISODE_OVERLAP_POLICY = "priority"  # "priority" (usar prioridad), "merge" (fusionar) [no implementado], "drop" (descartar solapadas)

# =========================
# Exposición competitiva
# =========================
# Nivel de competencia para medir exposición (puede ser "family" o "class")
EXPOSURE_SCOPE_COL = "family"

# ¿Excluir al propio ítem al medir la exposición de competidores?
EXPOSURE_EXCLUDE_SELF = True

# Umbral para binarizar exposición competitiva: E_bin_isw = 1(E_cat_isw >= THETA_E_CAT)
THETA_E_CAT = 0.20  # 20% de los demás ítems de la familia en promo



# =========================
# GSC (pre_gsc.py)
# =========================
OUTCOME_COL_DEFAULT_GSC = "y_isw"

# Debe ser <= K_PRE; con K_PRE=8, fija 8 para no descartar episodios válidos.
GSC_MIN_PRE_WEEKS = 8

# Evita donantes contaminados: no deben estar tratados en las semanas pre/guard_pre/treat/guard_post del episodio.
GSC_DONOR_POLICY = "not_treated_in_window"

# Arranca restringiendo por familia (productos comparables) y NO por tienda (necesitas masa de donantes).
GSC_DONOR_RESTRICT_SAME_FAMILY = True
GSC_DONOR_RESTRICT_SAME_STORE = False

# Roles que vetan al donante si estuvo tratado en esas semanas del episodio focal.
GSC_EXCLUDE_ROLES_FOR_DONORS = ("pre", "guard_pre", "treat", "guard_post")

# Controles: deja None para que pre_gsc los infiera (H_*, log_F_sw, O_w, eq_shock, fourier_*, lag*,
# y si existen: perishable, cluster). Más robusto al cambiar K y LAGS.
GSC_CONTROLS_DEFAULT = None


# =========================
# Meta-learners (pre_metalearners.py)
# =========================
OUTCOME_COL_DEFAULT_ML = "y_isw"

# Evita entrenamiento en semanas de transición (anticipos y ajustes pos-tratamiento).
ML_EXCLUDE_ROLES_FOR_TRAIN = ("guard_pre", "guard_post")

# Para modelos tree-based (GBM, RF, XGBoost/LightGBM), 'ordinal' es simple y funciona bien.
# Si vas con lineales o GLM, cambia a 'onehot'.
ML_ENCODING_METHOD = "ordinal"

# CV temporal (forward-chaining) — captura una estacionalidad anual completa al inicio.
ML_INIT_TRAIN_WEEKS = 52
ML_TEST_WEEKS = 8
ML_STEP_WEEKS = 4
ML_MIN_TRAIN_WEEKS = 32
