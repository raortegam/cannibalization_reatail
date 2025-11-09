# Estrategia de Experimentos - Comparación GSC vs Meta-Learners

## 🎯 Objetivo

Obtener la **máxima cantidad de episodios comparables** entre GSC (Generalized Synthetic Control) y Meta-Learners (T/S/X) para evaluar qué algoritmo funciona mejor en diferentes configuraciones.

## ✅ Mejoras Aplicadas

### 1. **Todos los Experimentos Corren Ambos Algoritmos**
- ✅ **GSC**: Se ejecuta automáticamente en todos los experimentos
- ✅ **Meta-Learners**: Ahora todos corren `["x", "s", "t"]` (3 learners)
- **Resultado**: Cada experimento genera 4 conjuntos de resultados comparables

### 2. **HPO Mejorado Propagado a Todos**
Todos los experimentos ahora incluyen:
```yaml
meta_hpo_trials: 100  # vs. 10 anterior
meta_max_iter: 1000   # vs. 500-600 anterior
```

### 3. **Experimentos Rediseñados para Comparación**

| ID | Descripción | Objetivo de Comparación |
|----|-------------|------------------------|
| **A_base** | Baseline con HPO mejorado | Configuración estándar optimizada |
| **B_donors30** | 30 donantes (vs. 20) | ¿Más donantes mejora GSC? ¿Afecta Meta? |
| **C_donors5_hiqual** | 5 donantes alta calidad | ¿Calidad > cantidad? Filtros estrictos |
| **D_seasonal_rich** | Fourier=6, lags hasta 84 días | ¿Captura mejor estacionalidad? |
| **E_no_stl** | Sin descomposición STL | ¿STL es necesario o añade ruido? |
| **F_treat_continuous** | Tratamiento continuo H_prop | ¿Mejor que binarizado? Solo Meta puede usar esto |
| **G_cv_robust** | 5 folds, 35 días holdout | ¿CV más robusto reduce overfitting? |
| **H_gsc_rank8** | GSC rank=8, tau más bajo | ¿GSC más flexible mejora ajuste? |

## 📊 Episodios Esperados por Experimento

Con el dataset aumentado (~100 caníbales × 50 víctimas):

| Algoritmo | Episodios Esperados | Archivos de Salida |
|-----------|--------------------|--------------------|
| **GSC** | ~100-150 | `gsc_metrics.parquet` |
| **X-Learner** | ~100-150 | `meta_metrics_x.parquet` |
| **S-Learner** | ~100-150 | `meta_metrics_s.parquet` |
| **T-Learner** | ~100-150 | `meta_metrics_t.parquet` |
| **TOTAL** | **~400-600** | Por experimento |

**Total en 8 experimentos**: ~3,200-4,800 episodios procesados

## 🔍 Preguntas de Investigación por Experimento

### A_base (Baseline)
- ¿Cuál es el RMSPE_pre de cada algoritmo?
- ¿Qué algoritmo tiene mejor ajuste en PRE?
- ¿Cuál estima efectos causales más realistas?

### B_donors30 vs A_base
- ¿30 donantes mejora el ajuste de GSC?
- ¿Los meta-learners se benefician de más features?
- ¿Hay overfitting con más donantes?

### C_donors5_hiqual vs A_base
- ¿5 donantes de alta calidad superan a 20 promedio?
- ¿Filtros estrictos reducen episodios procesables?
- ¿Qué algoritmo es más robusto con menos donantes?

### D_seasonal_rich vs A_base
- ¿Fourier=6 captura mejor estacionalidad que 3?
- ¿Lags largos (84 días) mejoran predicción?
- ¿Hay trade-off entre complejidad y generalización?

### E_no_stl vs A_base
- ¿STL es necesario o añade ruido?
- ¿Qué algoritmo depende más de STL?
- ¿Fourier=6 compensa la falta de STL?

### F_treat_continuous vs A_base
- ¿Tratamiento continuo (H_prop) es mejor que binario?
- Solo Meta puede usar esto → ¿ventaja sobre GSC?
- ¿S/T-learners mejoran con tratamiento continuo?

### G_cv_robust vs A_base
- ¿CV más robusto (5 folds, 35 días) reduce overfitting?
- ¿Mejora la generalización a POST?
- ¿Hay trade-off con tiempo de ejecución?

### H_gsc_rank8 vs A_base
- ¿GSC rank=8 mejora ajuste vs. rank=5?
- ¿Tau más bajo (1e-5) reduce regularización excesiva?
- ¿Meta-learners mantienen ventaja con GSC optimizado?

## 📈 Métricas de Comparación

Para cada episodio y algoritmo, comparar:

### Ajuste en PRE (Calidad del Modelo)
- **RMSPE_pre**: Root Mean Squared Percentage Error
- **MAE_pre**: Mean Absolute Error
- **R²_pre**: Coeficiente de determinación
- **Bias_pre**: Sesgo sistemático

### Validez del Contrafactual
- **Placebo espacial**: ¿Detecta correctamente no-efecto?
- **Placebo temporal**: ¿Estable en períodos sin tratamiento?
- **Leave-One-Out**: ¿Robusto a exclusión de donantes?

### Efecto Causal Estimado
- **ATE (Average Treatment Effect)**: Efecto promedio
- **ATT (Average Treatment on Treated)**: Efecto en tratados
- **Intervalos de confianza**: Incertidumbre
- **Heterogeneidad**: Variación entre episodios

## 🚀 Ejecución

```bash
# Ejecutar todos los experimentos (8 configuraciones)
python 01_run_sweep.py

# Tiempo estimado: 16-32 horas (2-4h por experimento)
# Recomendación: Ejecutar en servidor overnight
```

## 📁 Estructura de Salidas

```
.data/processed_data/
├── A_base/
│   ├── gsc/gsc_metrics.parquet          # GSC
│   └── meta_outputs/
│       ├── x/meta_metrics_x.parquet     # X-Learner
│       ├── s/meta_metrics_s.parquet     # S-Learner
│       └── t/meta_metrics_t.parquet     # T-Learner
├── B_donors30/
│   └── ...
└── ... (C-H)

figures/
├── A_base/
│   ├── gsc/                             # Gráficos GSC
│   └── meta/                            # Gráficos Meta
└── ... (C-H)
```

## 📊 Análisis Post-Experimentos

### 1. Consolidar Métricas
```python
import pandas as pd
from pathlib import Path

results = []
for exp in ["A_base", "B_donors30", "C_donors5_hiqual", ...]:
    # GSC
    gsc = pd.read_parquet(f".data/processed_data/{exp}/gsc/gsc_metrics.parquet")
    gsc["algorithm"] = "GSC"
    gsc["experiment"] = exp
    results.append(gsc)
    
    # Meta-learners
    for learner in ["x", "s", "t"]:
        meta = pd.read_parquet(f".data/processed_data/{exp}/meta_outputs/{learner}/meta_metrics_{learner}.parquet")
        meta["algorithm"] = f"Meta-{learner.upper()}"
        meta["experiment"] = exp
        results.append(meta)

df_all = pd.concat(results, ignore_index=True)
```

### 2. Comparar Algoritmos
```python
# RMSPE_pre por algoritmo y experimento
comparison = df_all.groupby(["experiment", "algorithm"])["rmspe_pre"].agg(["mean", "median", "std", "count"])

# Mejor algoritmo por experimento
best = df_all.loc[df_all.groupby(["experiment", "episode_id"])["rmspe_pre"].idxmin()]
best["algorithm"].value_counts()
```

### 3. Visualizar
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot de RMSPE_pre
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_all, x="experiment", y="rmspe_pre", hue="algorithm")
plt.xticks(rotation=45)
plt.title("RMSPE_pre por Experimento y Algoritmo")
plt.tight_layout()
plt.savefig("comparison_rmspe.png", dpi=300)
```

## ⚠️ Consideraciones

### Tiempo de Ejecución
- **Por experimento**: 2-4 horas
- **Total (8 experimentos)**: 16-32 horas
- **Recomendación**: Ejecutar en servidor o dejar overnight

### Recursos Computacionales
- **RAM**: 8-16 GB recomendado
- **CPU**: Multi-core beneficia Optuna (paralelización)
- **Disco**: ~5-10 GB por experimento

### Episodios Fallidos
Algunos episodios pueden fallar por:
- Insuficientes datos en PRE/POST
- Donantes de baja calidad
- Convergencia de optimización

**Solución**: Los algoritmos continúan con los episodios válidos

## 🎯 Criterios de Éxito

Un experimento es exitoso si:
1. ✅ Procesa >80% de episodios esperados
2. ✅ RMSPE_pre < 0.30 en promedio
3. ✅ Placebos no detectan efectos espurios
4. ✅ Efectos causales son interpretables

## 📝 Próximos Pasos

1. ✅ Ejecutar `python 01_run_sweep.py`
2. ⏳ Monitorear logs durante ejecución
3. 📊 Consolidar métricas al finalizar
4. 🔍 Analizar qué algoritmo y configuración funciona mejor
5. 📈 Generar reporte comparativo
6. 🎯 Seleccionar configuración óptima para producción

---

**Última actualización**: 2025-01-08  
**Configuración**: 8 experimentos × 4 algoritmos = 32 configuraciones comparables
