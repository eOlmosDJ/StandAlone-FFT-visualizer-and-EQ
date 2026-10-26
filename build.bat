@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   Compilando BDJ-FFT-EQ & Limiter Standalone
echo ========================================================
echo.
echo 1. Verificando dependencias...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    pause
    exit /b 1
)
echo.
echo 2. Generando ejecutable standalone optimizado con PyInstaller...
pyinstaller "FFTEQ&LIM.spec" --noconfirm
if errorlevel 1 (
    echo [ERROR] Fallo PyInstaller al compilar el ejecutable.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   Ejecutable generado con exito: dist\FFTEQ&LIM.exe
echo ========================================================
echo.
echo 3. Buscando compilador de Inno Setup (ISCC)...

set "ISCC_PATH="
if exist "%ProgramFiles%\Inno Setup 7\ISCC.exe" set "ISCC_PATH=%ProgramFiles%\Inno Setup 7\ISCC.exe"
if not defined ISCC_PATH if exist "%ProgramFiles(x86)%\Inno Setup 7\ISCC.exe" set "ISCC_PATH=%ProgramFiles(x86)%\Inno Setup 7\ISCC.exe"
if not defined ISCC_PATH if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC_PATH=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC_PATH if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC_PATH=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"

if not defined ISCC_PATH (
    for /f "delims=" %%I in ('where iscc 2^>nul') do set "ISCC_PATH=%%I"
)

if defined ISCC_PATH (
    echo Compilador Inno Setup encontrado en: "!ISCC_PATH!"
    echo Compilando instalador setup.exe con setup.iss...
    "!ISCC_PATH!" "setup.iss"
    if errorlevel 1 (
        echo [ADVERTENCIA] Fallo la compilacion de Inno Setup.
    ) else (
        echo.
        echo ========================================================
        echo   Instalador generado con exito: setup.exe
        echo ========================================================
    )
) else (
    echo [INFO] Inno Setup no fue encontrado en las rutas estandar ni en el PATH.
    echo Para compilar el instalador setup.exe:
    echo 1. Instala Inno Setup 6 o 7 (https://jrsoftware.org/isdl.php)
    echo 2. Abre setup.iss (o setup.ino) y pulsa Run/Compile (Ctrl+F9).
)

echo.
echo ========================================================
echo   Proceso de compilacion completo.
echo ========================================================
pause
