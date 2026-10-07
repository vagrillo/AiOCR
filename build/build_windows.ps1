# Build script for Windows: portable onefile exe + onedir for the installer.
# Usage:  powershell -ExecutionPolicy Bypass -File build\build_windows.ps1
$ErrorActionPreference = "Stop"

# Default: CPU-friendly torch for the portable bundle.
# For a CUDA bundle install torch from the cu126/cu128 index before running.
pip install -r requirements.txt pyinstaller

Write-Host "== Portable onefile build =="
$env:AIOCR_ONEDIR = "0"
pyinstaller build/AiOCR.spec --noconfirm --distpath dist_portable
Compress-Archive -Path dist_portable/AiOCR.exe -DestinationPath dist_portable/AiOCR-windows-portable.zip -Force

Write-Host "== Installer (onedir) build =="
$env:AIOCR_ONEDIR = "1"
pyinstaller build/AiOCR.spec --noconfirm --distpath dist
iscc build/AiOCR.iss

Write-Host "Artifacts:"
Get-ChildItem dist_portable, dist_installer
