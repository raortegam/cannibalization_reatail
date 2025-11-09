import pandas as pd
from pathlib import Path

print('\n=== VERIFICACIÓN DE EPISODIOS GSC ===')

# Listar archivos en el directorio GSC
gsc_dir = Path('.data/processed/A_base/gsc')
gsc_files = sorted(gsc_dir.glob('*.parquet'))

print(f'\nArchivos en {gsc_dir}:')
print(f'Total: {len(gsc_files)} archivos')

# Buscar los episodios sospechosos
ep1_id = '7-1576236_7-1576263_20161006'
ep2_id = '7-1576330_7-1576263_20161006'

ep1_file = gsc_dir / f'{ep1_id}.parquet'
ep2_file = gsc_dir / f'{ep2_id}.parquet'

print(f'\n--- Episodios Sospechosos ---')
print(f'Episodio 1 ({ep1_id}):')
print(f'  Archivo: {ep1_file}')
print(f'  Existe: {ep1_file.exists()}')

print(f'\nEpisodio 2 ({ep2_id}):')
print(f'  Archivo: {ep2_file}')
print(f'  Existe: {ep2_file.exists()}')

# Buscar episodios con la misma víctima
victim_pattern = '7-1576263'
print(f'\n--- Episodios con víctima {victim_pattern} ---')
matching_files = [f for f in gsc_files if victim_pattern in f.stem]
print(f'Total: {len(matching_files)} episodios')
for f in matching_files:
    print(f'  {f.stem}')

# Verificar índice de episodios
try:
    episodes_index = pd.read_parquet('.data/processed/A_base/episodes_index.parquet')
    print(f'\n--- Índice de Episodios ---')
    print(f'Total episodios en índice: {len(episodes_index)}')
    
    # Buscar episodios con la víctima 7-1576263
    victim_episodes = episodes_index[
        (episodes_index['j_store'] == 7) & 
        (episodes_index['j_item'] == 1576263)
    ]
    
    print(f'\nEpisodios con víctima 7:1576263 en índice:')
    print(f'Total: {len(victim_episodes)}')
    if len(victim_episodes) > 0:
        print(victim_episodes[['episode_id', 'i_store', 'i_item', 'j_store', 'j_item', 'treat_start']].to_string())
        
except Exception as e:
    print(f'\n❌ Error al cargar índice: {e}')

# Verificar archivos de salida de GSC
gsc_output_dir = Path('.data/processed_data/A_base/gsc/cf_series')
if gsc_output_dir.exists():
    print(f'\n--- Archivos de Series Contrafactuales ---')
    cf_files = sorted(gsc_output_dir.glob('*_cf.parquet'))
    print(f'Total: {len(cf_files)} archivos')
    
    # Buscar los episodios sospechosos
    ep1_cf = gsc_output_dir / f'{ep1_id}_cf.parquet'
    ep2_cf = gsc_output_dir / f'{ep2_id}_cf.parquet'
    
    print(f'\nEpisodio 1 CF: {ep1_cf.exists()}')
    print(f'Episodio 2 CF: {ep2_cf.exists()}')
    
    # Buscar episodios con la víctima
    matching_cf = [f for f in cf_files if victim_pattern in f.stem]
    print(f'\nArchivos CF con víctima {victim_pattern}:')
    for f in matching_cf:
        print(f'  {f.stem}')
else:
    print(f'\n❌ Directorio no existe: {gsc_output_dir}')

print('\n' + '='*60)
print('DIAGNÓSTICO:')
print('Si el episodio 1 NO existe en disco pero SÍ aparece en las gráficas,')
print('entonces hay un problema en el código de EDA que está generando')
print('gráficas con IDs incorrectos o duplicados.')
print('='*60)
