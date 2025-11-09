# Mejoras en Meta-Learners para Estimación Causal

## Problema Identificado

Los resultados de meta-learners mostraban un ajuste pobre del contrafactual:
- El contrafactual no seguía bien el patrón observado en PRE
- RMSPE_pre alto (~0.35-0.56)
- Líneas contrafactuales demasiado planas

## Cambios Implementados

### 1. Optimización de Hiperparámetros (HPO) con Optuna

**Archivo**: `src/models/meta_learners.py`

#### Aumento de Trials
- **Antes**: 10 trials por defecto
- **Ahora**: 100 trials por defecto
- **Impacto**: Mejor exploración del espacio de hiperparámetros

#### Ampliación de Grillas de Búsqueda

##### LightGBM:
| Parámetro | Antes | Ahora | Mejora |
|-----------|-------|-------|--------|
| `num_leaves` | 63-255 | 31-511 | Mayor flexibilidad en complejidad del modelo |
| `max_depth` | 6-12 | 5-15 | Árboles más profundos para capturar patrones complejos |
| `learning_rate` | 1e-3 a 0.2 | 5e-4 a 0.3 | Rango más amplio para convergencia óptima |
| `min_child_samples` | 5-50 | 3-100 | Mejor control de overfitting |
| `feature_fraction` | 0.7-1.0 | 0.6-1.0 | Mayor exploración de subsampling |
| `subsample` | 0.7-1.0 | 0.6-1.0 | Mayor exploración de subsampling |
| `reg_lambda` | 0.0-10.0 | 0.0-20.0 | Mayor regularización disponible |
| `max_iter` | Fijo 600 | 400-1500 | Optimización del número de iteraciones |

##### HistGradientBoosting:
| Parámetro | Antes | Ahora | Mejora |
|-----------|-------|-------|--------|
| `max_depth` | 6-12 | 5-15 | Mayor capacidad de modelado |
| `learning_rate` | 1e-3 a 0.2 | 5e-4 a 0.3 | Rango más amplio |
| `max_iter` | 400-2000 | 400-3000 | Más iteraciones para convergencia |
| `min_samples_leaf` | 5-50 | 3-100 | Mejor control de complejidad |
| `l2` | 1e-6 a 10.0 | 1e-7 a 20.0 | Mayor rango de regularización |

### 2. Aumento del Dataset para Meta-Learners

**Archivo**: `src/preprocess_data/3. select_pairs_and_donors.py`

| Parámetro | Antes | Ahora | Mejora |
|-----------|-------|-------|--------|
| `N_CANNIBALS_META` | 30 | 100 | +233% más productos promocionados |
| `N_VICTIMS_PER_I` | 10 | 30 | +200% más víctimas por caníbal |
| `N_VICTIMS_PER_I_META` | 10 | 50 | +400% más víctimas para meta-learners |

**Impacto esperado**:
- Dataset de entrenamiento ~10x más grande
- Mejor generalización del modelo
- Reducción de varianza en las predicciones
- Mayor cobertura de patrones de canibalización

### 3. Actualización de Configuración de Experimentos

**Archivo**: `experiments.yaml`

Experimento `A_base` actualizado con:
- `meta_hpo_trials: 100` - Activar HPO con 100 trials
- `meta_max_iter: 1000` - Más iteraciones base para el modelo

## Resultados Esperados

1. **Mejor ajuste en PRE**: El contrafactual debería seguir más de cerca el patrón observado antes del tratamiento
2. **RMSPE_pre más bajo**: Reducción esperada de ~0.35-0.56 a <0.25
3. **Contrafactuales más realistas**: Líneas con mayor variabilidad que reflejen mejor la dinámica real
4. **Efectos causales más precisos**: ATT (Average Treatment Effect on Treated) más confiable

## Cómo Ejecutar

```bash
# Ejecutar con la nueva configuración
python 01_run_sweep.py

# O ejecutar solo el experimento A_base
python 00_run_pipeline.py --exp_tag A_base
```

## Monitoreo

Durante la ejecución, observar:
1. **Logs de HPO**: Verificar que Optuna explore 100 trials
2. **Tamaño del dataset meta**: Confirmar que `all_units.parquet` sea significativamente más grande
3. **Métricas por episodio**: Revisar `meta_metrics_x.parquet` para RMSPE_pre
4. **Gráficos**: Inspeccionar `figures/A_base/meta/` para calidad visual del ajuste

## Notas Técnicas

- **Cross-fitting temporal**: Se mantiene el embargo de 7 días para evitar fugas temporales
- **Reponderación**: Los pesos por unidad y recencia se mantienen
- **Calibración de baseline**: El ajuste OLS en PRE sigue activo para corregir el "aplastamiento"
- **Objetivo Poisson**: Se detecta automáticamente cuando el target es cuasi-entero y no-negativo

## Archivos Modificados

1. **`src/models/meta_learners.py`**
   - Líneas 473-492: Ampliación de grillas de optimización
   - Línea 504: Aumento de trials por defecto a 100
   - Línea 956: Cambio de `hpo_trials` por defecto

2. **`src/preprocess_data/3. select_pairs_and_donors.py`**
   - Líneas 225-228: Aumento de tamaño del dataset para meta-learners

3. **`00_run_pipeline.py`**
   - Línea 406: Agregado parámetro `meta_hpo_trials`
   - Línea 1282: Paso del parámetro a MetaRunCfg

4. **`experiments.yaml`**
   - Líneas 23-24: Configuración de HPO para experimento A_base

## Reversión (si es necesario)

Para volver a la configuración anterior:

```python
# En meta_learners.py línea 956
hpo_trials: int = 10

# En 3. select_pairs_and_donors.py líneas 226-228
N_CANNIBALS_META = 30
N_VICTIMS_PER_I = 10
N_VICTIMS_PER_I_META = 10

# En 00_run_pipeline.py línea 406
# Eliminar: meta_hpo_trials: int = 100

# En experiments.yaml líneas 23-24
# Eliminar: meta_hpo_trials y meta_max_iter
```

## Referencias

- Documentación de Optuna: https://optuna.readthedocs.io/
- LightGBM Parameters: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- Meta-Learners Theory: Künzel et al. (2019) "Metalearners for estimating heterogeneous treatment effects"
