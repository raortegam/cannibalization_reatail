import pandas as pd

ep_id = '7-463901_7-1017791_20160915'
df = pd.read_parquet(f'.data/processed/A_base/gsc/{ep_id}.parquet')
df['date'] = pd.to_datetime(df['date'])

print('\n=== ANÁLISIS DE FECHAS ===')

victim = df[df['treated_unit']==1].sort_values('date')
donors = df[df['treated_unit']==0]

print(f'\nVÍCTIMA:')
print(f'  Unit ID: {victim["unit_id"].iloc[0]}')
print(f'  N filas: {len(victim)}')
print(f'  Fechas: {victim["date"].min().date()} a {victim["date"].max().date()}')
print(f'  Fechas únicas: {victim["date"].nunique()}')

print(f'\nDONANTES:')
print(f'  N donantes: {donors["unit_id"].nunique()}')
print(f'  N filas totales: {len(donors)}')
print(f'  Fechas: {donors["date"].min().date()} a {donors["date"].max().date()}')

# Verificar overlap de fechas
victim_dates = set(victim['date'])
donor_dates = set(donors['date'])
common_dates = victim_dates.intersection(donor_dates)

print(f'\nOVERLAP DE FECHAS:')
print(f'  Fechas de víctima: {len(victim_dates)}')
print(f'  Fechas de donantes: {len(donor_dates)}')
print(f'  Fechas en común: {len(common_dates)}')
print(f'  % overlap: {100*len(common_dates)/len(victim_dates):.1f}%')

if len(common_dates) == 0:
    print('\n⚠️  PROBLEMA CRÍTICO: No hay fechas en común entre víctima y donantes!')
    print('     GSC no puede funcionar sin overlap temporal.')
else:
    print(f'\nFechas en común: {min(common_dates).date()} a {max(common_dates).date()}')

# Verificar estructura del panel
print(f'\nESTRUCTURA DEL PANEL:')
print(f'  Columnas: {df.columns.tolist()[:10]}...')
print(f'\nPrimeras 5 filas de víctima:')
print(victim[['date', 'unit_id', 'sales', 'treated_unit', 'treated_time']].head())
print(f'\nPrimeras 5 filas de donantes:')
print(donors[['date', 'unit_id', 'sales', 'treated_unit']].head())
