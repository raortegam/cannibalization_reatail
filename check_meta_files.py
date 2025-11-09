from pathlib import Path

print('\n=== VERIFICACIÓN DE ARCHIVOS META vs GSC ===')

# Directorios
meta_dir = Path('.data/processed_data/A_base/meta_outputs/x/cf_series')
gsc_dir = Path('.data/processed_data/A_base/gsc/cf_series')

# Archivos Meta
if meta_dir.exists():
    meta_files = sorted(meta_dir.glob('*_cf.parquet'))
    meta_eps = set([f.stem.replace('_cf', '') for f in meta_files])
    print(f'\n--- Archivos META (x-learner) ---')
    print(f'Total: {len(meta_files)} archivos')
    print(f'\nEpisodios:')
    for ep in sorted(meta_eps):
        print(f'  {ep}')
else:
    print(f'\n❌ Directorio Meta no existe: {meta_dir}')
    meta_eps = set()

# Archivos GSC
if gsc_dir.exists():
    gsc_files = sorted(gsc_dir.glob('*_cf.parquet'))
    gsc_eps = set([f.stem.replace('_cf', '') for f in gsc_files])
    print(f'\n--- Archivos GSC ---')
    print(f'Total: {len(gsc_files)} archivos')
    print(f'\nEpisodios:')
    for ep in sorted(gsc_eps):
        print(f'  {ep}')
else:
    print(f'\n❌ Directorio GSC no existe: {gsc_dir}')
    gsc_eps = set()

# Comparación
print(f'\n--- COMPARACIÓN ---')
print(f'Episodios en Meta: {len(meta_eps)}')
print(f'Episodios en GSC: {len(gsc_eps)}')
print(f'Episodios en ambos: {len(meta_eps.intersection(gsc_eps))}')

# Episodios solo en Meta
only_meta = meta_eps - gsc_eps
if only_meta:
    print(f'\n⚠️  Solo en Meta ({len(only_meta)}):')
    for ep in sorted(only_meta):
        print(f'  {ep}')

# Episodios solo en GSC
only_gsc = gsc_eps - meta_eps
if only_gsc:
    print(f'\n⚠️  Solo en GSC ({len(only_gsc)}):')
    for ep in sorted(only_gsc):
        print(f'  {ep}')

# Buscar el episodio problemático
victim_pattern = '7-1576263'
print(f'\n--- Episodios con víctima {victim_pattern} ---')
meta_victim = [ep for ep in meta_eps if victim_pattern in ep]
gsc_victim = [ep for ep in gsc_eps if victim_pattern in ep]

print(f'\nEn Meta:')
for ep in sorted(meta_victim):
    print(f'  {ep}')

print(f'\nEn GSC:')
for ep in sorted(gsc_victim):
    print(f'  {ep}')

print('\n' + '='*60)
print('DIAGNÓSTICO:')
print('Si hay episodios en Meta que NO están en GSC,')
print('EDA intentará plotear GSC con IDs que no existen,')
print('causando gráficas con títulos incorrectos o vacías.')
print('='*60)
