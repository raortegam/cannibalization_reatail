import pandas as pd, numpy as np

H_PATH = r".data\processed_data\A_base\competitive_exposure.csv"  # o .csv — usa la ruta real que escribió el paso 2
read = (pd.read_parquet if H_PATH.lower().endswith((".parquet",".pq")) else pd.read_csv)
H = read(H_PATH, parse_dates=["date"])

# 1.a) ¿existen ambas columnas?
assert {"H_prop","H_prop_raw"} <= set(H.columns), set(H.columns)

# 1.b) ¿H_prop difiere de H_prop_raw?  (si es ~0, no capturaste el preprocesamiento)
diff = np.nanmean(np.abs(H["H_prop"] - H["H_prop_raw"]))
print("mean(|H_prop - H_prop_raw|) =", float(diff))
