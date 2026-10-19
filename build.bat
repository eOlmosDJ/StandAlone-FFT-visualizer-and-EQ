@echo off
echo ========================================================
echo   Compilando BDJ-FFT-EQ Standalone
echo ========================================================
echo.
echo 1. Verificando dependencias...
pip install -r requirements.txt
echo.
echo 2. Generando ejecutable standalone optimizado con PyInstaller...
pyinstaller --noconsole --onefile --icon "assets\icon.ico" --add-data "assets;assets" --name "FIRST RELEASE" main.py
echo.
echo ========================================================
echo   Proceso finalizado con exito!
echo   Ejecutable generado en: dist\FIRST RELEASE.exe
echo ========================================================
pause
