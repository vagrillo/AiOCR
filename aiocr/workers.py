"""Qt worker threads: model download/load pipeline and PDF conversion."""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import QThread, Signal

from . import engine, model_manager
from .constants import DEFAULT_DPI
from .exporter import assemble_markdown
from .inference import infer_multi_device


class PipelineWorker(QThread):
    """Downloads (if needed) and loads the model, reporting progress."""

    progress = Signal(int, int, str)      # bytes_done, bytes_total, current file
    status = Signal(str)
    ready = Signal(str)                   # runtime label
    failed = Signal(str)

    def __init__(self, model_dir: Path, precision: str, parent=None):
        super().__init__(parent)
        self._model_dir = Path(model_dir)
        self._precision = precision
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            if not model_manager.is_download_complete(self._model_dir):
                self.status.emit(
                    f"Downloading model to {self._model_dir} (resumable, ~6.7 GB)..."
                )
                model_manager.download_model(
                    self._model_dir,
                    progress_cb=self.progress.emit,
                    should_cancel=lambda: self._cancelled,
                )
            elif self._cancelled:
                return
            else:
                self.status.emit(f"Model found in {self._model_dir}.")

            engine.ENGINE.load(
                str(self._model_dir), self._precision, status_cb=self.status.emit
            )
            self.ready.emit(engine.ENGINE.runtime.label)
        except Exception as exc:  # surface the full traceback in the UI log
            if self._cancelled:
                return
            traceback.print_exc()
            self.failed.emit(str(exc))


class ConvertWorker(QThread):
    """Renders PDF pages and runs OCR, streaming live text to the UI."""

    render_progress = Signal(int, int)
    status = Signal(str)
    live_text = Signal(str)
    page_done = Signal(int, int)          # pages completed, total pages
    finished_ok = Signal(str, str)        # markdown text, working directory
    failed = Signal(str)

    def __init__(
        self,
        pdf_path: str,
        out_dir: Path,
        dpi: int = DEFAULT_DPI,
        page_range: Optional[Tuple[int, int]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._pdf_path = pdf_path
        self._out_dir = Path(out_dir)
        self._dpi = dpi
        self._page_range = page_range
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: D102
        try:
            from . import pdf_utils
            from .pdf_utils import pdf_to_images

            fitz = pdf_utils.fitz

            eng = engine.ENGINE
            if not eng.is_ready:
                self.failed.emit("Model is not loaded yet.")
                return

            stamp = time.strftime("%Y%m%d_%H%M%S")
            work_dir = self._out_dir / f"{Path(self._pdf_path).stem}_{stamp}"
            self.status.emit(f"Rendering pages at {self._dpi} DPI...")

            doc = fitz.open(self._pdf_path)
            try:
                image_paths = pdf_to_images(
                    doc,
                    work_dir,
                    dpi=self._dpi,
                    page_range=self._page_range,
                    progress_cb=lambda done, total: self.render_progress.emit(done, total),
                )
            finally:
                doc.close()

            if self._cancelled:
                return
            if not image_paths:
                self.failed.emit("No pages selected for conversion.")
                return

            self.status.emit(
                f"Parsing {len(image_paths)} page(s) on {eng.runtime.label}..."
            )

            completed = {"n": 0}
            total_pages = len(image_paths)

            def on_text(chunk: str) -> None:
                if not self._cancelled:
                    self.live_text.emit(chunk)

            def on_page_done(page_md: str) -> None:
                completed["n"] += 1
                if not self._cancelled:
                    self.page_done.emit(completed["n"], total_pages)

            pages, n_tokens = infer_multi_device(
                eng.model,
                eng.tokenizer,
                image_files=[str(p) for p in image_paths],
                output_path=work_dir,
                device=eng.device,
                dtype=eng.dtype,
                autocast_ctx=eng.runtime.autocast,
                on_text=on_text,
            )

            if self._cancelled:
                return

            for p in pages:
                on_page_done(p)

            markdown = assemble_markdown(pages)
            (work_dir / "result.md").write_text(markdown, encoding="utf-8")

            self.status.emit(f"Done ({n_tokens} tokens generated).")
            self.finished_ok.emit(markdown, str(work_dir))
        except Exception as exc:
            if self._cancelled:
                return
            traceback.print_exc()
            self.failed.emit(str(exc))
