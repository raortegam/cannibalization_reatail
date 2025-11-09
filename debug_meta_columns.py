import pandas as pd
from pathlib import Path

print('\n=== VERIFICACIÓN DE COLUMNAS EN ARCHIVOS META ===')

# Episodio de la imagen
ep_id = '2-913363_2-911989_20170628'

# Buscar archivo Meta-X
meta_dir = Path('.data/processed_data/A_base/meta_outputs/x/cf_series')
meta_file = meta_dir / f'{ep_id}_cf.parquet'

if meta_file.exists():
    df = pd.read_parquet(meta_file)
    print(f'\nArchivo: {meta_file.name}')
    print(f'Columnas: {df.columns.tolist()}')
    print(f'\nPrimeras 5 filas:')
    print(df.head())
    
    # Verificar si effect está calculado correctamente
    if {'y', 'y0_hat', 'effect'}.issubset(df.columns):
        print(f'\n--- Verificación de cálculo de effect ---')
        df_check = df[['date', 'y', 'y0_hat', 'effect']].head(10)
        df_check['effect_correcto'] = df_check['y'] - df_check['y0_hat']
        df_check['diferencia'] = df_check['effect'] - df_check['effect_correcto']
        print(df_check)
        
        if df_check['diferencia'].abs().max() > 0.01:
            print('\n⚠️  PROBLEMA: La columna "effect" NO está calculada correctamente!')
            print(f'   Debería ser: y - y0_hat')
            print(f'   Pero parece ser: {df["effect"].describe()}')
        else:
            print('\n✅ La columna "effect" está calculada correctamente')
    
    # Verificar nombres de columnas
    print(f'\n--- Mapeo de columnas ---')
    if 'y' in df.columns:
        print(f'  Observado: "y"')
    if 'sales' in df.columns:
        print(f'  Observado alternativo: "sales"')
    if 'y0_hat' in df.columns:
        print(f'  Contrafactual: "y0_hat"')
    if 'mu0_hat' in df.columns:
        print(f'  Contrafactual alternativo: "mu0_hat"')
    if 'effect' in df.columns:
        print(f'  Efecto: "effect"')
        
else:
    print(f'\n❌ Archivo no existe: {meta_file}')

print('\n' + '='*60)
