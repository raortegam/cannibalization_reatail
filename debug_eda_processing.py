import pandas as pd
from pathlib import Path

print('\n=== DEBUG: PROCESAMIENTO DE COLUMNAS EN EDA ===')

# Episodio de la imagen
ep_id = '2-913363_2-911989_20170628'

# Cargar archivo Meta-X
meta_file = Path('.data/processed_data/A_base/meta_outputs/x/cf_series') / f'{ep_id}_cf.parquet'
df_raw = pd.read_parquet(meta_file)

print(f'\n--- Archivo Original ---')
print(f'Columnas: {df_raw.columns.tolist()}')

# Simular el procesamiento de _read_cf_file
_OBS_COLS = ["sales", "y_obs", "observed", "Y", "y"]
_CF_COLS  = ["mu0_hat", "y0_hat", "y_hat", "y_cf", "counterfactual", "Y0", "cf"]
_EFF_COLS = ["effect", "tau_hat", "att", "delta"]

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

obs_c = _pick_col(df_raw, _OBS_COLS)
cf_c  = _pick_col(df_raw, _CF_COLS)
eff_c = _pick_col(df_raw, _EFF_COLS)

print(f'\n--- Columnas Seleccionadas ---')
print(f'  obs_c (observado): {obs_c}')
print(f'  cf_c (contrafactual): {cf_c}')
print(f'  eff_c (efecto): {eff_c}')

# Verificar valores
print(f'\n--- Valores de las primeras 10 filas ---')
sample = df_raw[['date', obs_c, cf_c, eff_c, 'tau_hat']].head(10).copy()
sample['effect_correcto'] = sample[obs_c] - sample[cf_c]
sample['diferencia'] = sample[eff_c] - sample['effect_correcto']

print(sample)

print(f'\n--- Diagnóstico ---')
if eff_c == 'effect':
    print(f'✅ Se está usando la columna "effect" (correcto)')
    if sample['diferencia'].abs().max() < 0.01:
        print(f'✅ La columna "effect" está calculada correctamente como sales - mu0_hat')
    else:
        print(f'⚠️  La columna "effect" NO es sales - mu0_hat')
elif eff_c == 'tau_hat':
    print(f'❌ Se está usando "tau_hat" en lugar de "effect"')
    print(f'   tau_hat = mu1_hat - mu0_hat (efecto de tratamiento)')
    print(f'   effect = sales - mu0_hat (efecto observado)')
    print(f'   ¡Estos son diferentes!')

print('\n' + '='*60)
