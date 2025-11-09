import pandas as pd
import numpy as np

print('\n=== INVESTIGACIÓN DE EPISODIOS DUPLICADOS ===')

# IDs de los episodios sospechosos
ep1_id = '7-1576236_7-1576263_20161006'
ep2_id = '7-1576330_7-1576263_20161006'

print(f'\nEpisodio 1: {ep1_id}')
print(f'Episodio 2: {ep2_id}')

# Parsear IDs
def parse_episode_id(ep_id):
    parts = ep_id.split('_')
    cannibal = parts[0]  # store-item del caníbal
    victim = parts[1]    # store-item de la víctima
    treat_date = parts[2]  # fecha de tratamiento
    
    cannibal_store, cannibal_item = cannibal.split('-')
    victim_store, victim_item = victim.split('-')
    
    return {
        'cannibal_store': int(cannibal_store),
        'cannibal_item': int(cannibal_item),
        'victim_store': int(victim_store),
        'victim_item': int(victim_item),
        'treat_date': treat_date
    }

ep1_info = parse_episode_id(ep1_id)
ep2_info = parse_episode_id(ep2_id)

print(f'\n--- Episodio 1 ---')
print(f'  Caníbal: Store {ep1_info["cannibal_store"]}, Item {ep1_info["cannibal_item"]}')
print(f'  Víctima: Store {ep1_info["victim_store"]}, Item {ep1_info["victim_item"]}')
print(f'  Fecha tratamiento: {ep1_info["treat_date"]}')

print(f'\n--- Episodio 2 ---')
print(f'  Caníbal: Store {ep2_info["cannibal_store"]}, Item {ep2_info["cannibal_item"]}')
print(f'  Víctima: Store {ep2_info["victim_store"]}, Item {ep2_info["victim_item"]}')
print(f'  Fecha tratamiento: {ep2_info["treat_date"]}')

print(f'\n--- Comparación ---')
print(f'  Misma víctima: {ep1_info["victim_store"] == ep2_info["victim_store"] and ep1_info["victim_item"] == ep2_info["victim_item"]}')
print(f'  Mismo caníbal: {ep1_info["cannibal_store"] == ep2_info["cannibal_store"] and ep1_info["cannibal_item"] == ep2_info["cannibal_item"]}')
print(f'  Misma fecha: {ep1_info["treat_date"] == ep2_info["treat_date"]}')

# Cargar datos de los episodios
try:
    df1 = pd.read_parquet(f'.data/processed/A_base/gsc/{ep1_id}.parquet')
    df2 = pd.read_parquet(f'.data/processed/A_base/gsc/{ep2_id}.parquet')
    
    print(f'\n--- Datos de Episodio 1 ---')
    print(f'  N filas: {len(df1)}')
    print(f'  N unidades: {df1["unit_id"].nunique()}')
    print(f'  Unidades: {sorted(df1["unit_id"].unique())}')
    
    print(f'\n--- Datos de Episodio 2 ---')
    print(f'  N filas: {len(df2)}')
    print(f'  N unidades: {df2["unit_id"].nunique()}')
    print(f'  Unidades: {sorted(df2["unit_id"].unique())}')
    
    # Comparar series de la víctima
    victim_id = f'{ep1_info["victim_store"]}:{ep1_info["victim_item"]}'
    
    victim1 = df1[df1['unit_id'] == victim_id][['date', 'sales']].sort_values('date').reset_index(drop=True)
    victim2 = df2[df2['unit_id'] == victim_id][['date', 'sales']].sort_values('date').reset_index(drop=True)
    
    print(f'\n--- Comparación de Series de Víctima ({victim_id}) ---')
    print(f'  Episodio 1: {len(victim1)} fechas')
    print(f'  Episodio 2: {len(victim2)} fechas')
    
    if victim1.equals(victim2):
        print(f'  ✅ Las series de la víctima son IDÉNTICAS')
    else:
        print(f'  ❌ Las series de la víctima son DIFERENTES')
        
    # Verificar si los donantes son los mismos
    donors1 = set(df1[df1['treated_unit'] == 0]['unit_id'].unique())
    donors2 = set(df2[df2['treated_unit'] == 0]['unit_id'].unique())
    
    print(f'\n--- Donantes ---')
    print(f'  Episodio 1: {len(donors1)} donantes')
    print(f'  Episodio 2: {len(donors2)} donantes')
    print(f'  Donantes en común: {len(donors1.intersection(donors2))}')
    print(f'  Solo en ep1: {donors1 - donors2}')
    print(f'  Solo en ep2: {donors2 - donors1}')
    
except Exception as e:
    print(f'\n❌ Error al cargar datos: {e}')

# Verificar en el índice de episodios
try:
    episodes_index = pd.read_parquet('.data/processed/A_base/episodes_index.parquet')
    
    ep1_row = episodes_index[episodes_index['episode_id'] == ep1_id]
    ep2_row = episodes_index[episodes_index['episode_id'] == ep2_id]
    
    print(f'\n--- Índice de Episodios ---')
    if not ep1_row.empty:
        print(f'\nEpisodio 1:')
        print(ep1_row[['episode_id', 'i_store', 'i_item', 'j_store', 'j_item', 'treat_start']].to_string())
    
    if not ep2_row.empty:
        print(f'\nEpisodio 2:')
        print(ep2_row[['episode_id', 'i_store', 'i_item', 'j_store', 'j_item', 'treat_start']].to_string())
        
except Exception as e:
    print(f'\n❌ Error al cargar índice: {e}')

print('\n' + '='*60)
print('CONCLUSIÓN:')
print('Si la víctima es la misma y las series son idénticas,')
print('entonces es CORRECTO que ambos episodios compartan la serie observada.')
print('Cada episodio representa un evento de canibalización diferente')
print('(canibales distintos) sobre la misma víctima en la misma fecha.')
print('='*60)
