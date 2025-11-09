import pandas as pd

ep_id = '7-463901_7-1017791_20160915'
df = pd.read_parquet(f'.data/processed/A_base/gsc/{ep_id}.parquet')
df['date'] = pd.to_datetime(df['date'])
treat_date = pd.to_datetime('2016-09-15')

print('\n=== TRAIN_MASK ANALYSIS ===')

if 'train_mask' in df.columns:
    victim = df[df['treated_unit']==1].sort_values('date')
    
    print(f'\nVíctima train_mask:')
    print(f'  Total filas: {len(victim)}')
    print(f'  train_mask=1: {(victim["train_mask"]==1).sum()}')
    print(f'  train_mask=0: {(victim["train_mask"]==0).sum()}')
    
    pre_victim = victim[victim['date'] < treat_date]
    post_victim = victim[victim['date'] >= treat_date]
    
    print(f'\nPRE (antes {treat_date.date()}):')
    print(f'  train_mask=1: {(pre_victim["train_mask"]==1).sum()} / {len(pre_victim)}')
    print(f'  train_mask=0: {(pre_victim["train_mask"]==0).sum()} / {len(pre_victim)}')
    
    print(f'\nPOST (desde {treat_date.date()}):')
    print(f'  train_mask=1: {(post_victim["train_mask"]==1).sum()} / {len(post_victim)}')
    print(f'  train_mask=0: {(post_victim["train_mask"]==0).sum()} / {len(post_victim)}')
    
    print(f'\nÚltimas 20 filas PRE:')
    print(pre_victim[['date', 'sales', 'train_mask']].tail(20).to_string())
else:
    print('No hay columna train_mask en el dataset')
