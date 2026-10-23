; Script de Inno Setup para BDJ Standalone Fast Fourier Transform analyzer, equalizer and limiter
; Generado para empaquetado y despliegue profesional en Windows

[Setup]
AppId={{BDJ-FFT-EQ-LIMITER-BAUTISTA-OLMOS}}
AppName=BDJ Standalone Fast Fourier Transform analyzer, equalizer and limiter
AppVersion=1.1.0
AppPublisher=Bautista Eneas Olmos
DefaultDirName={autopf}\BDJ_FFT_EQ
DisableProgramGroupPage=yes
OutputDir=.\Instalador
OutputBaseFilename=Instalar_BDJ_FFT_EQ
Compression=lzma
SolidCompression=yes
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\FFTEQ&LIM.exe

[Tasks]
Name: "desktopicon"; Description: "Crear un acceso directo en el escritorio"; GroupDescription: "Accesos directos:"
Name: "autostart"; Description: "Iniciar automáticamente con Windows"; GroupDescription: "Inicio automático:"

[Files]
; Empaquetar el ejecutable principal e icono
Source: "dist\FFTEQ&LIM.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Accesos directos en Menú Inicio y Escritorio
Name: "{autoprograms}\BDJ FFT-EQ"; Filename: "{app}\FFTEQ&LIM.exe"; IconFilename: "{app}\icon.ico"
Name: "{autodesktop}\BDJ FFT-EQ"; Filename: "{app}\FFTEQ&LIM.exe"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Registry]
; Agregar al inicio automático de Windows (Current User Run)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "BDJ_FFT_EQ"; ValueData: """{app}\FFTEQ&LIM.exe"""; Tasks: autostart; Flags: uninsdeletevalue

[Run]
; Opción para ejecutar la aplicación tras finalizar la instalación
Filename: "{app}\FFTEQ&LIM.exe"; Description: "Ejecutar BDJ FFT-EQ ahora"; Flags: nowait postinstall skipifsilent
