import pandas as pd

print('\n=== VERIFICACIÓN DE VÍCTIMA 7:1576285 ===')

# Cargar índice de episodios
episodes = pd.read_parquet('.data/processed/A_base/episodes_index.parquet')

# Buscar episodios con esta víctima
victim_episodes = episodes[
    (episodes['j_store'] == 7) & 
    (episodes['j_item'] == 1576285)
]

print(f'\nEpisodios con víctima 7:1576285:')
print(f'Total: {len(victim_episodes)}')
print()
print(victim_episodes[['episode_id', 'i_store', 'i_item', 'j_store', 'j_item', 'treat_start']].to_string())

print('\n' + '='*60)
print('INTERPRETACIÓN:')
print('Cada fila es un EVENTO DE CANIBALIZACIÓN diferente.')
print('El item 1576285 en la tienda 7 fue canibalizado por:')
for _, row in victim_episodes.iterrows():
    print(f'  - Item {row["i_item"]} (caníbal) en fecha {row["treat_start"]}')
print()
print('Cada episodio estima el efecto causal de ESE caníbal específico')
print('sobre la víctima. Es correcto que compartan la serie observada.')
print('='*60)
