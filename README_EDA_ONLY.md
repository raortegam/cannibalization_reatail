# EDA de Algoritmos - Ejecución Independiente

Este documento explica cómo ejecutar **solo el EDA de algoritmos** (visualización de resultados de GSC y Meta-learners) sin necesidad de correr todo el pipeline.

## 📋 Requisitos Previos

Antes de ejecutar el EDA, debes haber corrido al menos una vez el pipeline completo para generar:

1. **Episodes index**: `.data/processed_data/<exp_tag>/episodes_index.parquet`
2. **Outputs de GSC**: `.data/processed_data/<exp_tag>/gsc/`
3. **Outputs de Meta-learners**: `.data/processed_data/meta_outputs/<exp_tag>/`

## 🚀 Uso Rápido

### Opción 1: Script Batch (Windows - Más fácil)

```bash
# Ejecutar EDA para el experimento A_quick_smoke
run_eda_only.bat A_quick_smoke

# Ejecutar EDA para A_base
run_eda_only.bat A_base

# Con opciones adicionales
run_eda_only.bat A_quick_smoke --max_episodes_gsc 10 --learners x s t
```

### Opción 2: Script Python (Más control)

```bash
# Básico - solo experimento
python run_eda_only.py --exp_tag A_quick_smoke

# Con learners específicos
python run_eda_only.py --exp_tag A_base --learners x s t

# Limitar número de episodios
python run_eda_only.py --exp_tag A_quick_smoke --max_episodes_gsc 20 --max_episodes_meta 20

# Sin PDFs (solo PNGs)
python run_eda_only.py --exp_tag A_base --no_pdf

# Cambiar orientación y DPI
python run_eda_only.py --exp_tag A_quick_smoke --orientation portrait --dpi 150
```

## 📊 Outputs Generados

El EDA genera las siguientes visualizaciones en `figures/<exp_tag>/`:

### Series por Episodio (PNGs individuales)
- `series_GSC_<episode_id>.png` - Series de tiempo de GSC por episodio
- `series_meta-<learner>_<episode_id>.png` - Series de tiempo de Meta-learners por episodio

**Modificación reciente**: El panel inferior ahora muestra **solo el efecto acumulado punteado** (sin el efecto instantáneo).

### Resúmenes Comparativos
- `gsc_overview_summary_*.png` - Resumen de métricas de GSC
- `meta_<learner>_overview_summary_*.png` - Resumen de métricas de Meta-learners
- `compare_att_sum_gsc_vs_meta_<learner>.png` - Comparación entre métodos

### PDFs Consolidados (opcional)
- `gsc_report.pdf` - Reporte completo de GSC
- `meta_<learner>_report.pdf` - Reporte completo de Meta-learners

## 🔧 Opciones Disponibles

| Opción | Descripción | Default |
|--------|-------------|---------|
| `--exp_tag` | **[REQUERIDO]** Tag del experimento | - |
| `--base_dir` | Directorio base de datos procesados | `.data/processed_data` |
| `--figures_dir` | Directorio base para figuras | `figures` |
| `--learners` | Meta-learners a incluir (t, s, x) | `x` |
| `--max_episodes_gsc` | Máximo de episodios GSC a renderizar | Todos |
| `--max_episodes_meta` | Máximo de episodios Meta a renderizar | Todos |
| `--orientation` | Orientación (landscape/portrait) | `landscape` |
| `--dpi` | DPI de las figuras | `300` |
| `--no_pdf` | No exportar PDFs (solo PNGs) | `False` |

## 📁 Estructura de Archivos Esperada

```
.data/processed_data/
└── <exp_tag>/                          # e.g., A_quick_smoke
    ├── episodes_index.parquet          # REQUERIDO
    ├── gsc/                            # Outputs de GSC
    │   ├── gsc_metrics.parquet
    │   └── cf_series/
    │       └── *.parquet
    └── ...

.data/processed_data/meta_outputs/
└── <exp_tag>/                          # e.g., A_quick_smoke
    ├── x/                              # Meta X-learner
    │   ├── meta_metrics_x.parquet
    │   └── cf_series/
    │       └── *.parquet
    ├── s/                              # Meta S-learner (opcional)
    └── t/                              # Meta T-learner (opcional)

figures/
└── <exp_tag>/                          # Outputs del EDA
    ├── series_GSC_*.png
    ├── series_meta-x_*.png
    ├── gsc_report.pdf
    └── ...
```

## 💡 Ejemplos de Uso

### Ejemplo 1: Re-generar visualizaciones después de modificar EDA_algorithms.py

```bash
# Después de modificar el código de visualización
python run_eda_only.py --exp_tag A_quick_smoke
```

### Ejemplo 2: Generar solo primeros 10 episodios para revisión rápida

```bash
python run_eda_only.py --exp_tag A_base --max_episodes_gsc 10 --max_episodes_meta 10
```

### Ejemplo 3: Comparar todos los learners

```bash
python run_eda_only.py --exp_tag A_base --learners x s t
```

### Ejemplo 4: Generar solo PNGs (más rápido)

```bash
python run_eda_only.py --exp_tag A_quick_smoke --no_pdf
```

## 🐛 Solución de Problemas

### Error: "No existe episodes_index"
**Causa**: No se ha ejecutado el pipeline hasta Step 4.  
**Solución**: Ejecuta primero `python 01_run_sweep.py --experiments .\experiments.yaml --only <exp_tag>`

### Warning: "No se encontraron outputs de GSC ni Meta-learners"
**Causa**: No se ejecutaron los Steps 5 (GSC) y 6 (Meta-learners).  
**Solución**: Verifica que en `experiments.yaml` los toggles `step5_gsc` y `step6_meta` estén en `true`.

### Las gráficas se ven vacías o incompletas
**Causa**: Los archivos de series contrafactuales (`cf_series/*.parquet`) no existen.  
**Solución**: Ejecuta el pipeline completo al menos una vez para generar estos archivos.

## 📝 Notas

- Este script **no ejecuta los algoritmos**, solo genera visualizaciones de resultados existentes.
- Para ejecutar los algoritmos, usa `01_run_sweep.py` o `00_run_pipeline.py`.
- Los cambios en `EDA/EDA_algorithms.py` se reflejarán inmediatamente al re-ejecutar este script.
- El script es seguro: no modifica datos, solo lee y genera figuras.

## 🔄 Workflow Típico

1. **Primera vez**: Ejecutar pipeline completo
   ```bash
   python 01_run_sweep.py --experiments .\experiments.yaml --only A_quick_smoke
   ```

2. **Modificar visualizaciones**: Editar `EDA/EDA_algorithms.py`

3. **Re-generar solo EDA**: Usar este script
   ```bash
   python run_eda_only.py --exp_tag A_quick_smoke
   ```

4. **Revisar resultados**: Abrir figuras en `figures/A_quick_smoke/`

---

**Última actualización**: Nov 2025  
**Modificación reciente**: Panel inferior ahora muestra solo efecto acumulado punteado
