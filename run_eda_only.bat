@echo off
REM ============================================================================
REM run_eda_only.bat
REM Script batch para ejecutar solo el EDA de algoritmos en Windows
REM ============================================================================

echo.
echo ========================================
echo   EDA de Algoritmos - Ejecucion rapida
echo ========================================
echo.

REM Verificar que se pase el experimento como argumento
if "%1"=="" (
    echo ERROR: Debes especificar el experimento
    echo.
    echo Uso:
    echo   run_eda_only.bat A_quick_smoke
    echo   run_eda_only.bat A_base
    echo   run_eda_only.bat A_quick_smoke --max_episodes_gsc 10
    echo.
    exit /b 1
)

REM Ejecutar el script Python con todos los argumentos
python run_eda_only.py --exp_tag %*

echo.
echo ========================================
echo   Proceso completado
echo ========================================
echo.
