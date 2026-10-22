; Script de Inno Setup para BDJ Standalone Fast Fourier Transform analyzer, equalizer and limiter
; Generado automáticamente para empaquetado y despliegue profesional en Windows

[Setup]
AppName=BDJ Standalone Fast Fourier Transform analyzer, equalizer and limiter
AppVersion=1.0.0
AppPublisher=Bautista Eneas Olmos
DefaultDirName={autopf}\BDJ_FFT_EQ
DisableProgramGroupPage=yes
OutputDir=.\Instalador
OutputBaseFilename=Instalar_BDJ_FFT_EQ
Compression=lzma
SolidCompression=yes
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\FFTEQ&LIM.exe

[Files]
; Empaquetar el ejecutable principal
Source: "dist\FFTEQ&LIM.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Accesos directos en Menú Inicio y Escritorio
Name: "{autoprograms}\BDJ FFT-EQ"; Filename: "{app}\FFTEQ&LIM.exe"
Name: "{autodesktop}\BDJ FFT-EQ"; Filename: "{app}\FFTEQ&LIM.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Crear un acceso directo en el escritorio"; GroupDescription: "Accesos directos:"

[Registry]
; Agregar al inicio automático de Windows (Current User Run)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "BDJ_FFT_EQ"; ValueData: """{app}\FFTEQ&LIM.exe"""; Flags: uninsdeletevalue
