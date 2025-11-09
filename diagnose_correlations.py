import pandas as pd
import numpy as np

ep_id = '7-463901_7-1017791_20160915'
df = pd.read_parquet(f'.data/processed/A_base/gsc/{ep_id}.parquet')
df['date'] = pd.to_datetime(df['date'])

victim_sales = df[df['treated_unit']==1].set_index('date')['sales']

print('\n=== CORRELACIÓN VÍCTIMA-DONANTES ===')
corrs = []
for donor_id in df[df['treated_unit']==0]['unit_id'].unique()[:10]:
    donor_sales = df[df['unit_id']==donor_id].set_index('date')['sales']
    common_dates = victim_sales.index.intersection(donor_sales.index)
    if len(common_dates) > 10:
        v = victim_sales.loc[common_dates]
        d = donor_sales.loc[common_dates]
        corr = np.corrcoef(v, d)[0,1]
        corrs.append(corr)
        print(f'  Donante {donor_id}: corr={corr:.3f}, n_dates={len(common_dates)}')

if corrs:
    print(f'\nCorrelación promedio: {np.mean(corrs):.3f}')
    print(f'Correlación máxima: {np.max(corrs):.3f}')
    print(f'Correlación mínima: {np.min(corrs):.3f}')
else:
    print('No se pudieron calcular correlaciones')
