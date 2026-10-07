; Inno Setup script for the AiOCR Windows installer.
; Built by CI (ISCC.exe) against the onedir PyInstaller output in dist\AiOCR.
;
; ISCC resolves paths relative to this .iss file's folder (build/), so the
; PyInstaller output one level up is addressed as ..\dist.
;
; The version comes from the git tag via /DAppVersion (fallback: 1.0.0);
; PyInstaller executables carry no VERSIONINFO resource, so deriving the
; version from the exe is not possible.

#define SourceDir "..\dist\AiOCR"

#ifndef AppVersion
#define AppVersion "1.0.0"
#endif

#define AppName "AiOCR"
#define AppPublisher "AiOCR contributors"
#define AppExeName "AiOCR.exe"

[Setup]
AppId={{8C41B78A-2E62-4B71-9C4B-AiOCR00000001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputBaseFilename=AiOCR-windows-installer
OutputDir=..\dist_installer
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#AppExeName}

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}";

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
