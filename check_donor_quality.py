import pandas as pd
import numpy as np

ep_id = '7-463901_7-1017791_20160915'

# Cargar datos
print('\n=== ANÁLISIS DE CALIDAD DE DONANTES ===')

# Verificar si existe archivo de calidad
quality_file = '.data/processed/A_base/gsc/donor_quality.parquet'
try:
    quality_df = pd.read_parquet(quality_file)
    ep_quality = quality_df[quality_df['episode_id'] == ep_id]
    
    print(f'\nCalidad de donantes para {ep_id}:')
    print(f'Total unidades: {len(ep_quality)}')
    print(f'Víctima: {ep_quality[ep_quality["is_victim"]==True].shape[0]}')
    print(f'Donantes: {ep_quality[ep_quality["is_victim"]==False].shape[0]}')
    print(f'Donantes KEEP=True: {ep_quality[(ep_quality["is_victim"]==False) & (ep_quality["keep"]==True)].shape[0]}')
    print(f'Donantes KEEP=False: {ep_quality[(ep_quality["is_victim"]==False) & (ep_quality["keep"]==False)].shape[0]}')
    
    print(f'\nDonantes filtrados (keep=False):')
    filtered = ep_quality[(ep_quality["is_victim"]==False) & (ep_quality["keep"]==False)]
    if len(filtered) > 0:
        print(filtered[['store_nbr', 'item_nbr', 'promo_share', 'avail_share', 'keep', 'reason']].to_string())
    else:
        print('  Ninguno')
    
    print(f'\nDonantes aceptados (keep=True):')
    accepted = ep_quality[(ep_quality["is_victim"]==False) & (ep_quality["keep"]==True)]
    if len(accepted) > 0:
        print(accepted[['store_nbr', 'item_nbr', 'promo_share', 'avail_share', 'keep']].to_string())
    else:
        print('  Ninguno - ⚠️ TODOS LOS DONANTES FUERON FILTRADOS')
        
except FileNotFoundError:
    print(f'❌ Archivo no encontrado: {quality_file}')
except Exception as e:
    print(f'❌ Error: {e}')
