# AiOCR

A cross-platform desktop GUI for document OCR, powered by Baidu's
[Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) model.

AiOCR turns PDF documents into **Markdown** or **HTML** with a couple of
clicks and runs **fully offline** on your own hardware:

- **Tab 1 - PDF & Convert**: pick a PDF, preview its pages, choose the
  conversion density (DPI) and the precision, then run the OCR conversion.
- **Tab 2 - Result & Export**: read the converted document (rendered
  Markdown, HTML preview or source) and export it as `.md` or `.html`.

On the very first launch the app downloads the model into the current
directory (`./models/Unlimited-OCR`, resumable, with a live progress bar) and
then loads it on the best available backend.

## Features

| | |
|---|---|
| PDF preview | Built-in pager (first/prev/next/last) rendered with PyMuPDF |
| Conversion density | 96 / 150 / 200 / 300 DPI (PDF pages are rasterized before OCR) |
| Page range | Convert the whole document or just a range of pages |
| Live output | Generated text streams into the result tab while the model works |
| Export | `.md` (with cropped page images) and self-contained `.html` (images embedded as data URIs) |
| Sample document | A demo PDF (`assets/sample/testmultipage.pdf`) opens automatically on first run |
| Model management | Auto-download to the local folder with progress; if the disk is full the app asks where to store the model |

## Hardware backends

| Platform | Backend | Precision | Notes |
|---|---|---|---|
| Windows / Linux + NVIDIA (RTX 20xx+) | CUDA | bf16 | default `pip` wheels |
| Windows / Linux + NVIDIA (GTX 10xx/16xx, 4 GB cards) | CUDA | **8-bit (bitsandbytes)** | fits the ~3B model into 4 GB VRAM; use `requirements-cuda-legacy.txt` |
| Apple Silicon (M1/M2/M3/**M4**) | MPS | fp16 | native macOS wheels |
| Any machine | CPU | bf16/fp32 | slow fallback |

The backend is auto-detected at startup and shown in the UI. With `Precision:
Auto` the app enables 8-bit quantization automatically on GPUs with 6 GB of
VRAM or less; you can force it from the settings combo (`8-bit - fits 4 GB
GPUs`). Changing the precision takes effect on the next application start.

## Install and run from source

Python 3.10-3.12 is required (3.12 recommended).

```bash
# 1) core dependencies (all platforms)
pip install -r requirements.txt

# 2) PyTorch for your platform - pick ONE of:
pip install -r requirements-macos.txt         # macOS (Apple Silicon, MPS)
pip install -r requirements-cuda.txt          # NVIDIA RTX / CUDA 12.8
pip install -r requirements-cuda-legacy.txt   # NVIDIA GTX 10xx/16xx / CUDA 12.6

# 3) run
python main.py
```

### First launch: model download

The model (~6.7 GB, single safetensors shard + tokenizer + remote code) is
downloaded on first start into:

```
./models/Unlimited-OCR          (relative to the folder the app runs from)
```

The download is **resumable** - if it is interrupted, simply start the app
again. To store the model elsewhere, the app will ask you automatically when
the disk is short on space, or set:

```bash
export AIOCR_MODEL_DIR=/path/to/models/Unlimited-OCR   # optional override
```

If the Hugging Face CDN is unreachable from your network, a mirror can be
used via `export HF_ENDPOINT=https://hf-mirror.com`.

## Static binaries

The release artifacts are **statically linked** executables built with
PyInstaller (`--onefile`): no Python installation is required on the target
machine.

- `AiOCR-windows-portable.zip` - single `AiOCR.exe`, runs anywhere
- `AiOCR-windows-installer.exe` - classic Windows installer (Inno Setup)
- `AiOCR-macos-arm64.zip` - single `AiOCR` binary for Apple Silicon
  (on first run macOS may require `xattr -cr AiOCR` for unsigned binaries)

Bundled executables ship with CPU/MPS-capable PyTorch to keep the download
size sane. To run with **CUDA acceleration** on Windows/Linux, install from
source with one of the CUDA requirements files above - the app picks the GPU
up automatically.

Build them yourself:

```powershell
# Windows (PowerShell)
pip install -r requirements.txt -r requirements-cuda.txt pyinstaller
pyinstaller build/AiOCR.spec
```

```bash
# macOS
pip install -r requirements.txt -r requirements-macos.txt pyinstaller
pyinstaller build/AiOCR.spec
```

Continuous Integration builds both platforms on every push (see
`.github/workflows/`); tagged releases also publish the Windows installer.

## Usage

1. **Tab 1**: press *Browse...* (or use the pre-loaded sample PDF), check the
   preview, set the DPI density, the page range and the precision.
2. Press **Convert to Markdown / HTML**. Pages are rasterized at the chosen
   DPI and parsed by Unlimited-OCR; live output appears in Tab 2.
3. **Tab 2**: switch between *Rendered Markdown*, *HTML Preview* and
   *Markdown source*, then **Export .md** / **Export .html** / copy to
   clipboard. Exported `.md` files carry their `images/` folder along;
   exported `.html` files are single self-contained files.

### Conversion settings explained

- **Density (DPI)** - resolution used to rasterize PDF pages before OCR.
  200 DPI is a good default; use 300 DPI for small print, 96-150 DPI for
  speed on large documents.
- **Precision** - `8-bit` quantization (bitsandbytes) is what makes the model
  fit on 4 GB NVIDIA GPUs; `Auto` picks it for you on small-VRAM cards.
- **Page range** - convert only `From..To` pages of long documents.

## Project layout

```
aiocr/
  app.py            bootstrap: QSettings, model folder, main window
  constants.py      repo id, defaults, resource paths
  engine.py         backend detection (CUDA/MPS/CPU) + 8-bit model loading
  inference.py      device-agnostic multi-page parsing (mirrors upstream)
  model_manager.py  resumable snapshot download with progress callback
  pdf_utils.py      PyMuPDF rendering (preview + page rasterization)
  exporter.py       .md / self-contained .html export
  workers.py        Qt worker threads (download/load pipeline, conversion)
  ui/               convert tab, result tab, main window
build/              PyInstaller spec + Inno Setup installer script
assets/sample/      demo PDF opened on first run
docs/               original project brief
```

## Troubleshooting

- **CUDA out of memory** - select `Precision: 8-bit`, lower the DPI, or
  convert fewer pages at a time.
- **GTX 10xx/16xx GPUs** - install `requirements-cuda-legacy.txt`
  (CUDA 12.6 wheels); recent CUDA 12.8+ wheels dropped Pascal support.
- **8-bit is unavailable** - `pip install bitsandbytes` (CUDA only; on
  Apple Silicon the unified memory runs the 16-bit model comfortably).
- **Slow on CPU** - expected: the model is ~3B parameters. Use a GPU if you
  can, or convert small page ranges.
- **macOS blocks the binary** - run `xattr -cr AiOCR` once, or right-click >
  Open the first time.
- **Model download fails / is slow** - it resumes automatically on the next
  start; you can also point `HF_ENDPOINT` to a mirror.

## Credits and licenses

- Model & inference reference: [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) (MIT).
- AiOCR application code: MIT (see [LICENSE](LICENSE)). Model weights are
  governed by the license published in their Hugging Face repository.
- Built with PySide6, PyMuPDF, transformers, bitsandbytes and PyInstaller.
