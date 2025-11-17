@echo off
REM ============================================================================
REM Script para ejecutar el experimento FINAL_PAPER
REM ============================================================================
REM 
REM Este experimento genera los resultados definitivos para el paper:
REM - 50 víctimas (10 caníbales x 5 víctimas cada uno)
REM - 3 Meta-learners: X-learner, S-learner, T-learner
REM - 15-20 donantes por víctima (18 promedio)
REM - 500 trials de Optuna para GSC
REM - 200 trials de Optuna por cada Meta-learner
REM 
REM Salidas esperadas en: figures/FINAL_PAPER/
REM - series_gsc.pdf: 50 gráficas de series temporales GSC
REM - series_meta_x.pdf: 50 gráficas de X-learner
REM - series_meta_s.pdf: 50 gráficas de S-learner
REM - series_meta_t.pdf: 50 gráficas de T-learner
REM - causal_metrics_*.png: Gráficas comparativas de métricas causales
REM - tables/: Tablas con métricas consolidadas
REM 
REM Tiempo estimado: 2-4 horas (depende del hardware)
REM ============================================================================

echo ========================================
echo EXPERIMENTO FINAL PARA PAPER
echo ========================================
echo.
echo Configuracion:
echo - 50 victimas totales
echo - Meta-learners: X, S, T
echo - Donantes: 18 por victima
echo - GSC HPO: 500 trials
echo - Meta HPO: 200 trials por learner
echo.
echo Tiempo estimado: 2-4 horas
echo.
echo Presiona Ctrl+C para cancelar, o cualquier tecla para continuar...
pause > nul

echo.
echo Iniciando experimento FINAL_PAPER...
echo.

python 01_run_sweep.py --experiments experiments.yaml --only FINAL_PAPER

echo.
echo ========================================
echo EXPERIMENTO COMPLETADO
echo ========================================
echo.
echo Revisa los resultados en:
echo   figures\FINAL_PAPER\
echo.
echo Archivos principales:
echo   - series_gsc.pdf (50 graficas GSC)
echo   - series_meta_x.pdf (50 graficas X-learner)
echo   - series_meta_s.pdf (50 graficas S-learner)
echo   - series_meta_t.pdf (50 graficas T-learner)
echo   - causal_metrics_*.png (comparaciones)
echo   - tables\causal_metrics_comparison.csv
echo.
pause
