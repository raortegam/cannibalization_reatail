from pathlib import Path
import pandas as pd, numpy as np

# 1. Lista todos los competitive_exposure.* que existan
for p in Path(".data/processed_data").rglob("competitive_exposure.*"):
    print(">>", p)

# 2. Lee el archivo de la iteración A_base (ajusta si el exp_id es otro)
H_PATH = r".data/processed_data/A_base/competitive_exposure.csv"  # o .parquet si lo guardas así
read = (pd.read_parquet if H_PATH.lower().endswith((".parquet",".pq")) else pd.read_csv)
H = read(H_PATH, parse_dates=["date"])

# 3. Deben estar ambas columnas
assert {"H_prop","H_prop_raw"} <= set(H.columns), f"Columnas presentes: {set(H.columns)}"

# 4. Verifica que H_prop ≠ H_prop_raw (suavizado+winsor debe cambiar la serie)
import numpy as np
diff = float(np.nanmean(np.abs(H["H_prop"] - H["H_prop_raw"])))
print("mean(|H_prop - H_prop_raw|) =", diff)