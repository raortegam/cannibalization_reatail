# Experimento FINAL_PAPER - Documentación

## Resumen

Este experimento genera los resultados definitivos para el paper académico sobre canibalización en retail usando datos de Corporación Favorita (Kaggle).

## Configuración del Experimento

### Episodios y Víctimas
- **50 víctimas totales**: 10 productos caníbales × 5 víctimas por caníbal
- Selección determinista basada en frecuencia de pares caníbal-víctima

### Donantes (Control Sintético)
- **18 donantes por víctima** (rango 15-20)
- Criterio: mismo producto (`donor_kind: "same_item"`)
- Filtros de calidad:
  - Máximo 2% de días en promoción
  - Mínimo 90% de disponibilidad temporal

### Algoritmos Implementados

#### 1. Control Sintético Generalizado (GSC)
- **Hiperparámetros anti-overfitting**:
  - `rank = 3`: Capacidad reducida del modelo
  - `tau = 0.01`: Regularización nuclear (100× más que baseline)
  - `alpha = 0.005`: Regularización L2
- **Optimización de hiperparámetros**:
  - 500 trials de Optuna
  - Validación cruzada rolling: 5 folds, ventanas de 14 días, gap de 7 días
- **Características**:
  - Transformación log1p para estabilizar varianza
  - Covariables incluidas (lags, Fourier, STL)
  - Calibración de nivel del contrafactual
  - Tests placebo espaciales y temporales

#### 2. Meta-learners (X, S, T)
- **X-learner**: Efecto heterogéneo con propensity score
- **S-learner**: Modelo único con tratamiento como feature
- **T-learner**: Dos modelos separados (tratado/control)
- **Base model**: Histogram Gradient Boosting Trees (HGBT)
- **Optimización de hiperparámetros**:
  - 200 trials de Optuna por learner
  - Validación cruzada: 5 folds, ventanas de 14 días
- **Configuración**:
  - Max depth: 6
  - Learning rate: 0.05
  - Min samples per leaf: 20
  - Max iterations: 1000

### Features Engineered
- **Lags temporales**: 7, 14, 28, 56 días
- **Componentes de Fourier**: k=3 (estacionalidad)
- **Descomposición STL**: Trend + Seasonal + Residual
- **Tratamiento**: H_disc (discreto 0/1) y H_prop (continuo)

## Salidas Generadas

### Ubicación
```
figures/FINAL_PAPER/
```

### Archivos Principales

#### PDFs con Series Temporales (50 gráficas cada uno)
- `series_gsc.pdf`: Control Sintético Generalizado
- `series_meta_x.pdf`: X-learner
- `series_meta_s.pdf`: S-learner
- `series_meta_t.pdf`: T-learner
- `series_gsc_with_donors.pdf`: GSC con overlay de donantes

Cada gráfica incluye:
- Panel superior: Observado vs Contrafactual
- Panel inferior: Efecto acumulado
- Métricas: RMSPE(pre), ATT_sum, ATT_mean

#### Gráficas Comparativas de Métricas Causales
- `causal_metrics_prediction_quality.png`: Calidad predictiva (RMSPE, correlación, R²)
- `causal_metrics_heterogeneity_sensitivity.png`: Heterogeneidad y sensibilidad
- `causal_metrics_balance_placebo.png`: Balance de covariables y tests placebo
- `causal_metrics_radar_summary.png`: Comparación global en radar chart

#### Tablas de Métricas
```
figures/FINAL_PAPER/tables/
```
- `causal_metrics_comparison.csv`: Métricas causales por episodio y modelo
- `causal_metrics_summary_by_model.csv`: Resumen agregado por modelo
- `eda_algorithms_coverage.parquet`: Cobertura de episodios por algoritmo
- `top_episodes_by_abs_att.parquet`: Top 25 episodios por |ATT|

### PNGs Individuales
```
figures/FINAL_PAPER/series_*.png
```
- Una imagen PNG por episodio y algoritmo (200 archivos totales)
- Formato: `series_{algoritmo}_{episode_id}.png`

## Métricas Causales Calculadas

### Calidad Predictiva
- **RMSPE(pre)**: Root Mean Squared Percentage Error en período pre-tratamiento
- **Correlación(pre)**: Correlación observado vs predicho
- **R²(pre)**: Coeficiente de determinación

### Heterogeneidad del Efecto
- **CV(τ)**: Coeficiente de variación del efecto
- **σ(τ)**: Desviación estándar del efecto
- **% Positivos**: Porcentaje de efectos positivos

### Sensibilidad
- **Relative Std**: σ(ATT) / |ATT|
- **CV(ATT)**: Coeficiente de variación del ATT

### Balance de Covariables
- **Mean |Std. Diff|**: Diferencia estandarizada media
- **Balance Rate**: Proporción de covariables balanceadas

### Tests Placebo
- **P-value (espacial)**: Test placebo con unidades no tratadas
- **P-value (temporal)**: Test placebo en período pre-tratamiento

## Ejecución

### Opción 1: Script Batch (Windows)
```batch
run_final_paper.bat
```

### Opción 2: Línea de comandos
```bash
python 01_run_sweep.py --experiments experiments.yaml --only FINAL_PAPER
```

### Opción 3: Con limpieza de outputs previos
```bash
python 01_run_sweep.py --experiments experiments.yaml --only FINAL_PAPER --clean
```

## Tiempo de Ejecución Estimado

- **Total**: 2-4 horas (depende del hardware)
- **Step 4 (Features)**: ~10-15 min
- **Step 5 (GSC)**: ~1-2 horas (500 trials × 50 episodios)
- **Step 6 (Meta)**: ~1-2 horas (200 trials × 3 learners × 50 episodios)
- **EDA**: ~5-10 min

## Requisitos de Hardware

- **RAM**: Mínimo 8GB, recomendado 16GB
- **CPU**: Multi-core recomendado (Optuna paraleliza trials)
- **Disco**: ~2GB para outputs

## Validación de Resultados

### Checklist Post-Ejecución

1. **Archivos generados**:
   - [ ] `figures/FINAL_PAPER/series_gsc.pdf` (50 páginas)
   - [ ] `figures/FINAL_PAPER/series_meta_x.pdf` (50 páginas)
   - [ ] `figures/FINAL_PAPER/series_meta_s.pdf` (50 páginas)
   - [ ] `figures/FINAL_PAPER/series_meta_t.pdf` (50 páginas)
   - [ ] 4 gráficas PNG de métricas causales
   - [ ] Tablas en `figures/FINAL_PAPER/tables/`

2. **Calidad de ajuste** (revisar en tablas):
   - RMSPE(pre) < 0.25 para mayoría de episodios
   - Correlación(pre) > 0.80 para mayoría de episodios
   - Balance de covariables: Mean |Std. Diff| < 0.25

3. **Logs**:
   - Revisar `diagnostics/pipeline_FINAL_PAPER_*.log`
   - Verificar que no hay errores críticos
   - Confirmar 50 episodios procesados por cada algoritmo

## Interpretación de Resultados

### ATT (Average Treatment Effect on the Treated)
- **ATT_sum**: Efecto total acumulado en el período post-tratamiento
- **ATT_mean**: Efecto promedio diario
- **Negativo**: Indica canibalización (pérdida de ventas en víctima)
- **Positivo**: Indica efecto complementario (ganancia de ventas)

### Comparación entre Algoritmos
- **GSC**: Mejor para series temporales con patrones complejos
- **X-learner**: Mejor cuando hay heterogeneidad en el efecto
- **S-learner**: Más simple, útil como baseline
- **T-learner**: Robusto cuando hay diferencias claras entre tratado/control

### Gráficas Radar
- Compara 5 dimensiones normalizadas (0-1, donde 1 es mejor):
  - Calidad predictiva
  - Baja sensibilidad
  - Heterogeneidad del efecto
  - Balance de covariables
  - Tests placebo

## Troubleshooting

### Error: "No se encontraron episodios"
- Verificar que Steps 1-3 se ejecutaron previamente
- Revisar que existe `.data/processed_data/_shared_base/pairs_windows.csv`

### Error: "Out of memory"
- Reducir `gsc_hpo_trials` o `meta_hpo_trials`
- Reducir `n_victims_per_cannibal` para menos episodios

### Optuna muy lento
- Verificar que Optuna está usando paralelización
- Considerar reducir `gsc_cv_folds` de 5 a 3

### Gráficas no se generan
- Verificar que existe `figures/FINAL_PAPER/`
- Revisar logs para errores en EDA_algorithms

## Referencias

- **Datos**: Corporación Favorita (Kaggle)
- **GSC**: Athey et al. (2021) - Matrix Completion Methods for Causal Panel Data Models
- **Meta-learners**: Künzel et al. (2019) - Metalearners for estimating heterogeneous treatment effects
- **Optuna**: Akiba et al. (2019) - Optuna: A Next-generation Hyperparameter Optimization Framework

## Contacto y Soporte

Para preguntas sobre este experimento, revisar:
1. Logs en `diagnostics/`
2. Configuración en `experiments.yaml` (id: FINAL_PAPER)
3. Pipeline en `00_run_pipeline.py`
