import pandas as pd

d = pd.read_csv(r'.data/processed_data/_shared_base/donors_per_victim.csv')
self_donors = ((d['j_store'] == d['donor_store']) & (d['j_item'] == d['donor_item']))

print('Self-donors encontrados:', int(self_donors.sum()))
print('Total filas:', len(d))
print('Porcentaje:', f'{100*self_donors.mean():.2f}%')

if self_donors.any():
    print('\nEjemplos de self-donors:')
    print(d[self_donors][['j_store','j_item','donor_store','donor_item','donor_kind','rank']].head(10).to_string(index=False))
else:
    print('\n✓ No hay self-donors (víctima como su propio donante)')
