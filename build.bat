@echo off
echo ========================================================
echo   Compilando BDJ-FFT-EQ & Limiter Standalone
echo ========================================================
echo.
echo 1. Verificando dependencias...
pip install -r requirements.txt
echo.
echo 2. Generando ejecutable standalone optimizado con PyInstaller...
pyinstaller "FFTEQ&LIM.spec" --noconfirm
echo.
echo ========================================================
echo   Proceso finalizado con exito!
echo   Ejecutable generado en: dist\FFTEQ&LIM.exe
echo   Para compilar el instalador, compila: build_installer.iss
echo ========================================================
pause
