[Setup]
AppId={{37AF4A8E-6884-4DF0-BDA4-46FAF67C0009}
AppName=capta
AppVersion=1.1
AppPublisher=geltz
DefaultDirName={autopf}\capta
SetupIconFile=capta.ico
DefaultGroupName=capta
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir=.
OutputBaseFilename=capta_setup_1.1
WizardStyle=modern
UninstallDisplayIcon={app}\capta.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\capta\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\capta"; Filename: "{app}\capta.exe"
Name: "{group}\{cm:UninstallProgram,capta}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\capta"; Filename: "{app}\capta.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\capta.exe"; Description: "{cm:LaunchProgram,capta}"; Flags: nowait postinstall skipifsilent