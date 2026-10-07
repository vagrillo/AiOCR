#!/usr/bin/env bash
# Build script for macOS (Apple Silicon): single statically linked binary.
# Usage: bash build/build_macos.sh
set -euo pipefail

pip install -r requirements.txt -r requirements-macos.txt pyinstaller

AIOCR_ONEDIR=0 pyinstaller build/AiOCR.spec --noconfirm --distpath dist_portable
cd dist_portable
zip -y -r AiOCR-macos-arm64.zip AiOCR
cd ..

echo "Artifacts:"
ls -la dist_portable
