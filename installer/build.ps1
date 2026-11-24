# Builds the installers from dist\AudioSR_Upscaler:
#   installer\out\AudioSR-Upscaler-<version>.msi (+ cab*.cab)  - the Windows Installer package
#   installer\out\AudioSR-Upscaler-Setup-<version>.exe         - single-file setup (ship this)
# Usage: powershell -File installer\build.ps1 [-Version 1.0.0]
param([string]$Version = "1.0.0")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$app = Join-Path $root "dist\AudioSR_Upscaler"
$assets = Join-Path $root "assets"
$out = Join-Path $PSScriptRoot "out"
if (-not (Test-Path (Join-Path $app "AudioSR_Upscaler.exe"))) { throw "Build the app first: $app is missing." }
New-Item -ItemType Directory -Force $out | Out-Null
Get-ChildItem $out -File | Remove-Item -Force   # stale cabs from an older build would ship too
$env:PATH = "$env:USERPROFILE\.dotnet\tools;$env:PATH"

$msi = Join-Path $out "AudioSR-Upscaler-$Version.msi"
wix build (Join-Path $PSScriptRoot "AudioSR.wxs") -arch x64 `
    -d "Version=$Version" -d "Assets=$assets" -bindpath "App=$app" -o $msi
if ($LASTEXITCODE -ne 0) { throw "MSI build failed ($LASTEXITCODE)" }

# The Bal extension is used by path: the wix CLI mislabels this 5.0.2 package as "damaged"
# because its DLL carries the newer BootstrapperApplications name.
$bal = Get-ChildItem "$env:USERPROFILE\.wix\extensions\WixToolset.Bal.wixext\5.0.2" -Recurse -Filter "*.wixext.dll" |
    Select-Object -First 1
if (-not $bal) { throw "WiX Bal extension missing: wix extension add -g WixToolset.Bal.wixext/5.0.2" }
$setup = Join-Path $out "AudioSR-Upscaler-Setup-$Version.exe"
wix build (Join-Path $PSScriptRoot "Setup.wxs") -arch x64 -ext $bal.FullName `
    -d "Version=$Version" -d "Assets=$assets" -d "Msi=$msi" -o $setup
if ($LASTEXITCODE -ne 0) { throw "Setup build failed ($LASTEXITCODE)" }

Get-ChildItem $out -File | Where-Object Extension -ne ".wixpdb" |
    Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB) } } | Format-Table -AutoSize
$gib = (Get-Item $setup).Length / 1GB
"Setup size: {0:N2} GiB (GitHub release files must be under 2 GiB: {1})" -f $gib, $(if ($gib -lt 2) { "OK" } else { "TOO BIG" })
