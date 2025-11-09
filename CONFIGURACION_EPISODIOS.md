# Configuración de Episodios - Resumen de Cambios

## 🎯 Problema Identificado

El pipeline solo estaba generando **10 episodios** en total, cuando se esperaban ~100-150 episodios para comparación robusta entre algoritmos.

### 🔍 Causa Raíz

El parámetro `MAX_EPISODES_FOR_DONORS` en `3. select_pairs_and_donors.py` estaba limitado a **10 episodios** por defecto.

## ✅ Cambios Realizados

### 1. **`src/preprocess_data/3. select_pairs_and_donors.py`** (Línea 240)

```python
# ANTES
MAX_EPISODES_FOR_DONORS = _env_int("SPD_MAX_EPISODES_FOR_DONORS", 10)

# DESPUÉS
MAX_EPISODES_FOR_DONORS = _env_int("SPD_MAX_EPISODES_FOR_DONORS", 150)
```

**Impacto:** Ahora se seleccionarán hasta 150 episodios para GSC (y por ende, para todo el pipeline).

### 2. **`pipeline_config.yaml`** - Límites Removidos

```yaml
# ANTES
max_episodes: 50
gsc_max_episodes: 50
meta_max_episodes: 50
eda_alg_max_episodes_gsc: 40
eda_alg_max_episodes_meta: 40

# DESPUÉS
max_episodes: null              # Sin límite
gsc_max_episodes: null          # Sin límite
meta_max_episodes: null         # Sin límite
eda_alg_max_episodes_gsc: null  # Sin límite (plotea todos)
eda_alg_max_episodes_meta: null # Sin límite (plotea todos)
```

**Impacto:** Todos los pasos del pipeline procesarán y visualizarán todos los episodios disponibles.

## 📊 Episodios Esperados Ahora

### Flujo Completo

```
Step 3: select_pairs_and_donors.py
├── Dataset Meta: ~5,000 episodios (100 caníbales × 50 víctimas)
├── Selección GSC: 150 episodios (top por delta_abs)
└── episodes_index.parquet: ~150 episodios

Step 4: pre_algorithm.py
├── Procesa ~150 episodios
└── Aplica filtros de calidad → ~100-150 episodios válidos

Step 5: GSC
└── Procesa ~100-150 episodios

Step 6: Meta-learners (X, S, T)
└── Cada learner procesa ~100-150 episodios

EDA: EDA_algorithms.py
└── Plotea TODOS los episodios disponibles (~100-150 por algoritmo)
```

### Por Experimento

| Algoritmo | Episodios Esperados | Archivos Generados |
|-----------|--------------------|--------------------|
| **GSC** | ~100-150 | `gsc_metrics.parquet` + ~100-150 PNGs |
| **X-Learner** | ~100-150 | `meta_metrics_x.parquet` + ~100-150 PNGs |
| **S-Learner** | ~100-150 | `meta_metrics_s.parquet` + ~100-150 PNGs |
| **T-Learner** | ~100-150 | `meta_metrics_t.parquet` + ~100-150 PNGs |
| **TOTAL** | **~400-600** | Por experimento |

### En 8 Experimentos

```
Total episodios procesados: ~3,200-4,800
Total gráficos generados: ~3,200-4,800 PNGs
Total métricas: 32 archivos parquet (4 por experimento × 8)
```

## 🔧 Parámetros Configurables

Si necesitas ajustar el número de episodios, puedes modificar:

### 1. Variable de Entorno (Temporal)

```bash
# Windows
set SPD_MAX_EPISODES_FOR_DONORS=200
python 01_run_sweep.py --experiments experiments.yaml

# Linux/Mac
export SPD_MAX_EPISODES_FOR_DONORS=200
python 01_run_sweep.py --experiments experiments.yaml
```

### 2. Código Fuente (Permanente)

Editar línea 240 en `src/preprocess_data/3. select_pairs_and_donors.py`:

```python
MAX_EPISODES_FOR_DONORS = _env_int("SPD_MAX_EPISODES_FOR_DONORS", 150)  # Cambiar 150 por el valor deseado
```

### 3. Límites por Paso (En `pipeline_config.yaml`)

```yaml
max_episodes: 100              # Límite en Step 4
gsc_max_episodes: 100          # Límite para GSC
meta_max_episodes: 100         # Límite para Meta-learners
eda_alg_max_episodes_gsc: 50   # Límite de ploteo GSC
eda_alg_max_episodes_meta: 50  # Límite de ploteo Meta
```

## ⚠️ Consideraciones

### Tiempo de Ejecución

Con 150 episodios:

| Paso | Tiempo Estimado |
|------|----------------|
| **Step 3** | ~30-45 min (genera episodios y donantes) |
| **Step 4** | ~20-30 min (preprocesamiento) |
| **Step 5 (GSC)** | ~30-60 min (150 episodios) |
| **Step 6 (Meta)** | ~2-3 horas (150 episodios × 3 learners × HPO) |
| **EDA** | ~15-30 min (genera ~600 gráficos) |
| **TOTAL por experimento** | **~3-5 horas** |

**Total para 8 experimentos:** ~24-40 horas

### Espacio en Disco

| Tipo | Tamaño Estimado |
|------|----------------|
| **Datos procesados** | ~500 MB - 1 GB por experimento |
| **Gráficos PNG** | ~200-400 MB por experimento |
| **PDFs** | ~50-100 MB por experimento |
| **TOTAL por experimento** | ~750 MB - 1.5 GB |
| **TOTAL 8 experimentos** | **~6-12 GB** |

### Memoria RAM

- **Mínimo:** 8 GB
- **Recomendado:** 16 GB
- **Óptimo:** 32 GB (para HPO paralelo)

## 🎯 Estrategia de Selección de Episodios

El parámetro `EPISODE_SELECTION_STRATEGY` controla cómo se seleccionan los 150 episodios:

```python
EPISODE_SELECTION_STRATEGY = "top_delta_abs"  # Opciones: top_delta_abs | random | first
```

### Estrategias Disponibles

1. **`top_delta_abs`** (ACTUAL): Selecciona episodios con mayor cambio absoluto en ventas
   - ✅ Episodios más interesantes (mayor efecto potencial)
   - ✅ Mejor para validar algoritmos
   - ⚠️ Puede sesgar hacia casos extremos

2. **`random`**: Selección aleatoria
   - ✅ Muestra representativa
   - ✅ Sin sesgo
   - ⚠️ Puede incluir episodios poco informativos

3. **`first`**: Primeros N episodios
   - ✅ Rápido y determinista
   - ⚠️ Puede tener sesgo temporal

## 📝 Verificación

Para verificar cuántos episodios se generaron:

```bash
# Ver episodios en el índice
python -c "import pandas as pd; df = pd.read_parquet('.data/processed_data/A_base/episodes_index.parquet'); print(f'Total: {len(df)}')"

# Ver episodios procesados por GSC
python -c "import pandas as pd; df = pd.read_parquet('.data/processed_data/A_base/gsc/gsc_metrics.parquet'); print(f'GSC: {len(df)}')"

# Ver episodios procesados por Meta-X
python -c "import pandas as pd; df = pd.read_parquet('.data/processed_data/A_base/meta_outputs/x/meta_metrics_x.parquet'); print(f'Meta-X: {len(df)}')"
```

## 🚀 Próximos Pasos

1. ✅ Cambios aplicados
2. 🔄 Ejecutar pipeline: `python 01_run_sweep.py --experiments experiments.yaml`
3. 📊 Verificar número de episodios generados
4. 📈 Analizar resultados con ~100-150 episodios por algoritmo
5. 🎯 Comparar rendimiento entre configuraciones

---

**Fecha:** 2025-01-08  
**Cambios:** Aumentado límite de episodios de 10 → 150  
**Impacto:** ~10x más episodios para comparación robusta
