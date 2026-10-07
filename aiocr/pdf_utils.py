"""PDF rendering helpers built on PyMuPDF (fitz)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

try:  # PyMuPDF >= 1.26 prefers the pymupdf name
    import pymupdf as fitz
except ImportError:  # pragma: no cover - older PyMuPDF
    import fitz
from PIL import Image


def open_pdf(path: str | Path) -> fitz.Document:
    return fitz.open(str(path))


def page_count(doc: fitz.Document) -> int:
    return doc.page_count


def render_page_preview(doc: fitz.Document, page_index: int, target_width: int = 900) -> Image.Image:
    """Render one page as a PIL image sized for on-screen preview."""
    page = doc.load_page(page_index)
    rect = page.rect
    zoom = max(0.2, min(3.0, target_width / max(1.0, rect.width)))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def preview_qimage(doc: fitz.Document, page_index: int, target_width: int = 900):
    """Render one page directly to a QImage (no intermediate PIL copy)."""
    from PySide6.QtGui import QImage

    page = doc.load_page(page_index)
    rect = page.rect
    zoom = max(0.2, min(3.0, target_width / max(1.0, rect.width)))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
    return img.copy()  # detach from fitz buffer before the pixmap is freed


def pdf_to_images(
    doc: fitz.Document,
    out_dir: str | Path,
    dpi: int = 200,
    page_range: Optional[Tuple[int, int]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Path]:
    """Rasterize the requested pages at the given DPI and return the PNG paths."""
    out_dir = Path(out_dir)
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)

    start, end = page_range or (0, doc.page_count - 1)
    start = max(0, start)
    end = min(doc.page_count - 1, end)
    indices: Iterable[int] = range(start, end + 1)

    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    paths: List[Path] = []
    total = max(1, end - start + 1)
    for done, i in enumerate(indices, start=1):
        page = doc.load_page(i)
        out = out_dir / "pages" / f"page_{i + 1:04d}.png"
        page.get_pixmap(matrix=mat, alpha=False).save(str(out))
        paths.append(out)
        if progress_cb:
            progress_cb(done, total)
    return paths
