# PyInstaller spec for AiOCR.
#
# Default build is a statically linked, single-file executable (onefile).
# Set AIOCR_ONEDIR=1 to build a folder instead (used for the Windows
# installer, which prefers instant startup over a single portable binary).
#
#   pyinstaller build/AiOCR.spec --noconfirm
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

PROJECT_ROOT = Path(SPECPATH).resolve().parent  # noqa: F821
ONEFILE = os.environ.get("AIOCR_ONEDIR", "0") != "1"

hiddenimports = collect_submodules("transformers.models.auto") + [
    "einops",
    "addict",
    "easydict",
    "huggingface_hub",
    "pymupdf",
    "markdown",
]

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / "assets" / "sample" / "testmultipage.pdf"), "assets/sample"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "PyQt5",
        "PyQt6",
        "IPython",
        "jupyter",
        "pytest",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

icon = PROJECT_ROOT / "assets" / "icon.png"
icon_arg = f"--icon={icon}" if icon.exists() else ""

if ONEFILE:
    exe = EXE(  # noqa: F821
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="AiOCR",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    exe = EXE(  # noqa: F821
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="AiOCR",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(  # noqa: F821
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="AiOCR",
    )
