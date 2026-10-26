; =====================================================================
; Inno Setup Script para BDJ-FFT-EQ & Limiter
; Compilador: Inno Setup 6 / 7 (ISCC.exe)
; Genera: setup.exe
; =====================================================================

[Setup]
AppId={{BDJ-FFT-EQ-LIMITER-BAUTISTA-OLMOS}}
AppName=BDJ Standalone Fast Fourier Transform analyzer, equalizer and limiter
AppVersion=1.2.0
AppPublisher=Bautista Eneas Olmos
DefaultDirName={autopf}\BDJ_FFT_EQ
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=setup
Compression=lzma2/ultra64
SolidCompression=yes
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\FFTEQ&LIM.exe
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "autostart"; Description: "Iniciar automáticamente con Windows"; GroupDescription: "Inicio automático:"

[Files]
Source: "dist\FFTEQ&LIM.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{autoprograms}\BDJ FFT-EQ"; Filename: "{app}\FFTEQ&LIM.exe"; IconFilename: "{app}\icon.ico"
Name: "{autodesktop}\BDJ FFT-EQ"; Filename: "{app}\FFTEQ&LIM.exe"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "BDJ_FFT_EQ"; ValueData: """{app}\FFTEQ&LIM.exe"""; Tasks: autostart; Flags: uninsdeletevalue

[Run]
Filename: "{app}\FFTEQ&LIM.exe"; Description: "{cm:LaunchProgram,BDJ FFT-EQ}"; Flags: nowait postinstall skipifsilent
