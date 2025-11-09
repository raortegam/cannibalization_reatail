from pathlib import Path
import re

print('\n=== VERIFICACIÓN DE ARCHIVOS PNG ===')

# Directorio de figuras
fig_dir = Path('figures/A_base/algorithms')

if not fig_dir.exists():
    print(f'❌ Directorio no existe: {fig_dir}')
    exit(1)

# Buscar PNGs de series GSC
gsc_pngs = sorted(fig_dir.glob('series__gsc_*.png'))

print(f'\nArchivos PNG de GSC: {len(gsc_pngs)}')

# Buscar los episodios con víctima 7-1576263
victim_pattern = '7-1576263'
matching_pngs = [f for f in gsc_pngs if victim_pattern in f.stem]

print(f'\nPNGs con víctima {victim_pattern}:')
for png in matching_pngs:
    print(f'  {png.name}')
    # Extraer episode_id del nombre del archivo
    match = re.search(r'series__gsc_(.+)\.png', png.name)
    if match:
        ep_id = match.group(1)
        print(f'    Episode ID: {ep_id}')

# Buscar el episodio problemático
ep1_png = fig_dir / 'series__gsc_7-1576236_7-1576263_20161006.png'
ep2_png = fig_dir / 'series__gsc_7-1576326_7-1576263_20161006.png'
ep3_png = fig_dir / 'series__gsc_7-1576330_7-1576263_20161006.png'

print(f'\n--- Verificación de Archivos Específicos ---')
print(f'7-1576236 (INCORRECTO): {ep1_png.exists()}')
print(f'7-1576326 (CORRECTO):   {ep2_png.exists()}')
print(f'7-1576330 (CORRECTO):   {ep3_png.exists()}')

# Listar TODOS los PNGs de GSC
print(f'\n--- Todos los PNGs de GSC ---')
for png in gsc_pngs:
    print(f'  {png.name}')

print('\n' + '='*60)
print('CONCLUSIÓN:')
print('Si existe un PNG con 7-1576236, entonces el problema está')
print('en una ejecución anterior de EDA que generó archivos incorrectos.')
print('Solución: Re-ejecutar EDA para regenerar las imágenes correctas.')
print('='*60)
