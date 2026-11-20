; =====================================================================
; Inno Setup Script para Heaviside DSP
; Compilador: Inno Setup 6 / 7 (ISCC.exe)
; Genera: Heaviside DSP Setup.exe
; =====================================================================

[Setup]
AppId={{HEAVISIDE-DSP-BAUTISTA-OLMOS}}
AppName=Heaviside DSP
AppVersion=1.2.0
AppPublisher=Bautista Eneas Olmos
VersionInfoDescription=Heaviside DSP Setup
VersionInfoProductName=Heaviside DSP
VersionInfoProductVersion=1.2.0
VersionInfoVersion=1.2.0.0
VersionInfoCompany=Bautista Eneas Olmos
DefaultDirName={autopf}\Heaviside DSP
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=Heaviside DSP Setup
Compression=lzma2/ultra64
SolidCompression=yes
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\Heaviside DSP.exe
UninstallDisplayName=Heaviside DSP
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "autostart"; Description: "Iniciar automáticamente con Windows"; GroupDescription: "Inicio automático:"

[Dirs]
Name: "{app}\presets"; Permissions: users-modify
Name: "{userappdata}\Heaviside_DSP\presets"; Flags: uninsneveruninstall

[Files]
Source: "dist\Heaviside DSP.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{autoprograms}\Heaviside DSP"; Filename: "{app}\Heaviside DSP.exe"; IconFilename: "{app}\icon.ico"; AppUserModelID: "heaviside.dsp.1.0"
Name: "{autodesktop}\Heaviside DSP"; Filename: "{app}\Heaviside DSP.exe"; IconFilename: "{app}\icon.ico"; AppUserModelID: "heaviside.dsp.1.0"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "Heaviside_DSP"; ValueData: """{app}\Heaviside DSP.exe"""; Tasks: autostart; Flags: uninsdeletevalue

[Run]
Filename: "{app}\Heaviside DSP.exe"; Description: "{cm:LaunchProgram,Heaviside DSP}"; Flags: nowait postinstall skipifsilent
