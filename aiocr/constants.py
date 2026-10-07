"""AiOCR - desktop GUI for document OCR with Baidu's Unlimited-OCR model."""

__version__ = "1.0.0"

APP_NAME = "AiOCR"
ORG_NAME = "AiOCR"

# Hugging Face repository holding the model weights + remote modeling code.
MODEL_REPO_ID = "baidu/Unlimited-OCR"

# Only these file patterns are needed to run the model. The repository also
# ships a paper PDF, demo assets and SGLang wheels which we do not download.
MODEL_ALLOW_PATTERNS = ["*.json", "*.py", "*.safetensors"]

# Written into the model directory after a successful snapshot download so the
# app can skip the network check on subsequent starts.
DOWNLOAD_MARKER = ".aiocr_download_complete"

# Inference defaults (mirroring the upstream README for multi-page parsing).
DEFAULT_DPI = 200
DPI_CHOICES = [96, 150, 200, 300]
DEFAULT_IMAGE_SIZE = 1024          # "base" mode: image_size=1024, no cropping
DEFAULT_MAX_LENGTH = 32768
DEFAULT_NO_REPEAT_NGRAM = 35
DEFAULT_NGRAM_WINDOW = 1024        # multi-page window (128 is for single images)
MULTI_PAGE_PROMPT = "<image>Multi page parsing."

# Bits-and-bytes quantized inference for small-VRAM GPUs.
# The 8-bit weights alone need ~4.5 GB (LLM int8 + fp16 vision encoder and
# embeddings) and the SAM vision encoder adds a few hundred MB of activations,
# so: <=5 GB -> 4-bit NF4, <=7 GB -> 8-bit, otherwise 16-bit.
AUTO_4BIT_VRAM_THRESHOLD_GB = 5.0
AUTO_8BIT_VRAM_THRESHOLD_GB = 7.0

# Free disk space required before starting a model download (~6.7 GB + margin).
MODEL_DOWNLOAD_NEEDED_GB = 9.0

PRECISION_CHOICES = ["auto", "4bit", "8bit", "16bit", "32bit"]

import sys
from pathlib import Path


def resource_path(relative: str) -> Path:
    """Absolute path to a bundled resource (works frozen with PyInstaller)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / relative
    return Path(__file__).resolve().parent.parent / relative


def sample_pdf_path() -> Path:
    """Bundled demo PDF, loaded automatically on the very first launch."""
    return resource_path("assets/sample/testmultipage.pdf")
