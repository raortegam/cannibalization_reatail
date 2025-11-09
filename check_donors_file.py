import pandas as pd
from pathlib import Path

print('\n=== VERIFICACIÓN DE ARCHIVO DE DONANTES ===')

# Verificar si existe el archivo de donantes
donors_file = Path('.data/processed_data/A_base/donors_per_victim.csv')
if donors_file.exists():
    print(f'\n✅ Archivo de donantes existe: {donors_file}')
    donors_df = pd.read_csv(donors_file)
    print(f'\nShape: {donors_df.shape}')
    print(f'Columnas: {donors_df.columns.tolist()}')
    print(f'\nPrimeras 5 filas:')
    print(donors_df.head())
    
    # Buscar el episodio problemático
    ep_id = '7-463901_7-1017791_20160915'
    if 'episode_id' in donors_df.columns:
        ep_donors = donors_df[donors_df['episode_id'] == ep_id]
        print(f'\n\nDonantes para episodio {ep_id}:')
        print(f'  N donantes: {len(ep_donors)}')
        if len(ep_donors) > 0:
            print(ep_donors)
        else:
            print('  ⚠️  No hay donantes registrados para este episodio')
    else:
        print('\n⚠️  Columna "episode_id" no encontrada')
else:
    print(f'\n❌ Archivo de donantes NO existe: {donors_file}')

# Verificar archivo alternativo
alt_file = Path('.data/processed/A_base/donors_per_victim.csv')
if alt_file.exists():
    print(f'\n✅ Archivo alternativo existe: {alt_file}')
    donors_df = pd.read_csv(alt_file)
    print(f'Shape: {donors_df.shape}')
else:
    print(f'\n❌ Archivo alternativo NO existe: {alt_file}')
