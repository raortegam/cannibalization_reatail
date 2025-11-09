# Variación de Parámetros por Experimento

## 🎯 Resumen Ejecutivo

**El Step 3 (`select_pairs_and_donors.py`) NO varía entre experimentos** - todos los experimentos usan el mismo dataset base generado en el Step 3.

## 📊 Qué Varía y Qué NO Varía

### ❌ NO Varía Entre Experimentos

#### Step 3: `select_pairs_and_donors.py`

**Todos estos parámetros son FIJOS** (solo configurables por variables de entorno, no por `experiments.yaml`):

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `N_CANNIBALS_META` | **100** | Número de caníbales para meta-learners |
| `N_VICTIMS_PER_I_META` | **50** | Víctimas por caníbal para meta |
| `MAX_EPISODES_FOR_DONORS` | **150** | Episodios seleccionados para GSC |
| `EPISODE_SELECTION_STRATEGY` | `"top_delta_abs"` | Estrategia de selección |
| `P_PROMO_I_MIN` | 0.03 | Mínimo % de promoción para caníbales |
| `P_PROMO_I_MAX` | 0.25 | Máximo % de promoción para caníbales |
| `P_PROMO_J_MAX` | 0.10 | Máximo % de promoción para víctimas |
| `PRE_DAYS` | 90 | Días de período PRE |
| `POST_DAYS` | 30 | Días de período POST |
| `PRE_GAP` | 7 | Gap entre PRE y tratamiento |
| `WINDOW_START` | "2016-01-01" | Inicio de ventana temporal |
| `WINDOW_END` | "2017-06-30" | Fin de ventana temporal |

**Resultado:** Todos los experimentos parten del **mismo dataset base** de ~150 episodios.

### ✅ SÍ Varía Entre Experimentos

#### Step 4: `pre_algorithm.py` (Preprocesamiento)

| Parámetro | A_base | B_donors30 | C_donors5 | Otros |
|-----------|--------|------------|-----------|-------|
| `top_k_donors` | **20** | **30** | **5** | 10 |
| `lags_days` | [7,14,28,56] | [7,14,28,56] | [7,14,28,56] | **[7,14,28,56,84]** (D) |
| `fourier_k` | 3 | 3 | 3 | **6** (D,E) |
| `use_stl` | true | true | true | **false** (E) |
| `max_donor_promo_share` | 0.02 | 0.02 | **0.01** | 0.02 |
| `min_availability_share` | 0.90 | 0.90 | **0.95** | 0.90 |

#### Step 5: GSC (Synthetic Control)

| Parámetro | A_base | H_gsc_rank8 | Otros |
|-----------|--------|-------------|-------|
| `gsc_rank` | 5 | **8** | 5 |
| `gsc_tau` | 0.0001 | **0.00001** | 0.0001 |
| `gsc_alpha` | 0.0 | **0.001** | 0.0 |
| `gsc_cv_folds` | 3 | 3 | **5** (G) |
| `gsc_cv_holdout` | 21 | 21 | **35** (G) |

#### Step 6: Meta-learners

| Parámetro | Todos | F_treat_continuous |
|-----------|-------|-------------------|
| `meta_learners` | **["x","s","t"]** | ["x","s","t"] |
| `meta_hpo_trials` | **100** | 100 |
| `meta_max_iter` | **1000** | 1000 |
| `meta_cv_folds` | 3 | **5** (G) |
| `meta_cv_holdout` | 21 | **35** (G) |
| `treat_col_s` | "H_disc" | **"H_prop"** (F) |
| `treat_col_b` | "H_prop" | **"H_prop"** (F) |

## 🔄 Flujo de Datos por Experimento

```
┌─────────────────────────────────────────────────────────────┐
│ Step 3: select_pairs_and_donors.py                         │
│ ❌ NO VARÍA - Se ejecuta IGUAL para todos los experimentos │
├─────────────────────────────────────────────────────────────┤
│ Dataset generado:                                           │
│ • ~5,000 episodios meta (100 caníbales × 50 víctimas)      │
│ • 150 episodios GSC (top_delta_abs)                         │
│ • episodes_index.parquet (150 episodios)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: pre_algorithm.py                                    │
│ ✅ SÍ VARÍA - Diferentes configuraciones de preprocesamiento│
├─────────────────────────────────────────────────────────────┤
│ Experimento A_base:                                         │
│ • 20 donantes, lags [7,14,28,56], fourier=3, STL=on        │
│ → Genera features específicas para A_base                   │
│                                                             │
│ Experimento B_donors30:                                     │
│ • 30 donantes, lags [7,14,28,56], fourier=3, STL=on        │
│ → Genera features diferentes (más donantes)                 │
│                                                             │
│ Experimento C_donors5_hiqual:                               │
│ • 5 donantes (alta calidad), lags [7,14,28,56]             │
│ → Genera features con menos donantes pero mejor calidad     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Steps 5 & 6: Algoritmos (GSC + Meta)                       │
│ ✅ SÍ VARÍA - Diferentes hiperparámetros y configuraciones  │
├─────────────────────────────────────────────────────────────┤
│ Cada experimento usa:                                       │
│ • Mismo dataset base (150 episodios)                        │
│ • Features diferentes (por Step 4)                          │
│ • Hiperparámetros diferentes (GSC rank, Meta HPO, etc.)     │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Implicaciones

### ✅ Ventajas de NO Variar Step 3

1. **Comparación Justa**
   - Todos los experimentos evalúan los mismos episodios
   - Las diferencias en resultados se deben SOLO a configuraciones de algoritmos
   - No hay sesgo por selección de episodios diferentes

2. **Eficiencia**
   - Step 3 es costoso (~30-45 min)
   - Se ejecuta una vez por experimento pero con mismos criterios
   - Ahorra tiempo total de ejecución

3. **Reproducibilidad**
   - Mismo dataset base garantiza reproducibilidad
   - Fácil de comparar métricas entre experimentos

### ⚠️ Limitaciones

1. **No se puede evaluar impacto de selección de episodios**
   - No puedes comparar "top_delta_abs" vs "random"
   - No puedes variar el número de caníbales/víctimas por experimento

2. **Filtros de calidad fijos**
   - `P_PROMO_I_MIN/MAX`, `P_PROMO_J_MAX` son fijos
   - No puedes experimentar con diferentes criterios de selección

3. **Ventanas temporales fijas**
   - `PRE_DAYS`, `POST_DAYS`, `PRE_GAP` son fijos
   - No puedes experimentar con diferentes longitudes de ventana

## 🔧 Cómo Variar Step 3 (Si Necesitas)

Si quieres experimentar con diferentes configuraciones de Step 3, tienes que usar **variables de entorno**:

### Opción 1: Por Experimento (Manual)

```bash
# Experimento con más caníbales
set SPD_N_CANNIBALS_META=200
set SPD_MAX_EPISODES_FOR_DONORS=200
python 00_run_pipeline.py --config pipeline_config.yaml

# Experimento con selección aleatoria
set SPD_EPISODE_SELECTION=random
python 00_run_pipeline.py --config pipeline_config.yaml
```

### Opción 2: Modificar Código (Permanente)

Editar `src/preprocess_data/3. select_pairs_and_donors.py`:

```python
# Línea 226
N_CANNIBALS_META = _env_int("SPD_N_CANNIBALS_META", 200)  # Cambiar default

# Línea 240
MAX_EPISODES_FOR_DONORS = _env_int("SPD_MAX_EPISODES_FOR_DONORS", 200)

# Línea 241
EPISODE_SELECTION_STRATEGY = _env_str("SPD_EPISODE_SELECTION", "random")
```

### Opción 3: Crear Variantes de Experimentos

Podrías crear scripts wrapper que configuren variables de entorno antes de ejecutar:

```bash
# run_experiment_A_large.bat
set SPD_N_CANNIBALS_META=200
set SPD_MAX_EPISODES_FOR_DONORS=200
python 00_run_pipeline.py --config pipeline_config.yaml
```

## 📊 Resumen de Variación por Paso

| Paso | ¿Varía? | Qué Varía | Impacto |
|------|---------|-----------|---------|
| **Step 1** | ❌ NO | Filtrado de datos | Mismo dataset limpio |
| **Step 2** | ❌ NO | Cálculo de H (exposure) | Mismo H para todos |
| **Step 3** | ❌ NO | Selección episodios | **Mismo dataset base (150 eps)** |
| **Step 4** | ✅ SÍ | Donantes, lags, fourier, STL | **Features diferentes** |
| **Step 5** | ✅ SÍ | GSC rank, tau, alpha, CV | **Hiperparámetros GSC** |
| **Step 6** | ✅ SÍ | Meta HPO, CV, tratamiento | **Hiperparámetros Meta** |
| **EDA** | ✅ SÍ | Número de gráficos | Visualización |

## 🎯 Conclusión

**La variación entre experimentos ocurre principalmente en:**

1. **Preprocesamiento (Step 4):**
   - Número de donantes (5, 10, 20, 30)
   - Features temporales (lags, fourier)
   - Descomposición (STL on/off)
   - Filtros de calidad

2. **Algoritmos (Steps 5 & 6):**
   - Hiperparámetros GSC (rank, tau, alpha)
   - Hiperparámetros Meta (HPO, max_iter)
   - Cross-validation (folds, holdout)
   - Tipo de tratamiento (discreto vs continuo)

**El dataset base (Step 3) es el MISMO para todos**, lo cual es **ideal para comparación científica**.

---

**Recomendación:** Si necesitas variar Step 3, considera crear un conjunto separado de experimentos con prefijo diferente (ej: "Z_large_dataset", "Z_random_selection") para no mezclar con los experimentos actuales.
