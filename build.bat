@echo off
echo ========================================================
echo   Compilando Pro-Q 4 Standalone Parametric ^& Dynamic EQ
echo ========================================================
echo.
echo 1. Verificando dependencias...
pip install -r requirements.txt
echo.
echo 2. Generando ejecutable standalone optimizado con PyInstaller...
pyinstaller --noconsole --onefile --name "ProQ4_Standalone" main.py
echo.
echo ========================================================
echo   Proceso finalizado con exito!
echo   Ejecutable generado en: dist\ProQ4_Standalone.exe
echo ========================================================
pause
