"""Tab 1: choose a PDF, preview it, configure conversion, run OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import Qt, QSettings, QUrl, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..constants import DPI_CHOICES, PRECISION_CHOICES, sample_pdf_path
from ..pdf_utils import open_pdf, page_count, preview_qimage
from ..workers import ConvertWorker, PipelineWorker

_PRECISION_LABELS = {
    "auto": "Auto (recommended)",
    "4bit": "4-bit NF4 - fits 4 GB GPUs",
    "8bit": "8-bit - fits 6-8 GB GPUs",
    "16bit": "16-bit",
    "32bit": "32-bit",
}


class ConvertTab(QWidget):
    """PDF selection + preview + conversion settings + progress."""

    conversion_finished = Signal(str, str)   # markdown, work_dir

    def __init__(self, model_dir: Path, precision: str, parent=None):
        super().__init__(parent)
        self._model_dir = model_dir
        self._precision = precision
        self._doc = None
        self._current_page = 0
        self._pipeline: Optional[PipelineWorker] = None
        self._converter: Optional[ConvertWorker] = None
        self._pdf_path: Optional[Path] = None

        self._build_ui()
        self._maybe_load_sample()
        self._start_pipeline()

    def _maybe_load_sample(self) -> None:
        """First launch: open the bundled sample PDF so the UI is not empty."""
        settings = QSettings()
        if settings.value("first_run_done", False, type=bool):
            return
        settings.setValue("first_run_done", True)
        p = sample_pdf_path()
        if p.exists():
            self.load_pdf(p)

    def _precision_changed(self) -> None:
        QSettings().setValue("precision", self.precision_combo.currentData())

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Model status box -------------------------------------------------
        model_box = QGroupBox("Model")
        mlay = QHBoxLayout(model_box)
        self.model_label = QLabel("Starting...")
        self.model_label.setWordWrap(True)
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 100)
        self.model_progress.setVisible(False)
        self.model_progress.setFormat("%p% (%v / %m MB)")
        mlay.addWidget(self.model_label, 1)
        mlay.addWidget(self.model_progress, 0)
        root.addWidget(model_box)

        # File selection ----------------------------------------------------
        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("Choose a PDF file...")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        sample = QPushButton("Open sample PDF")
        sample.clicked.connect(self._open_sample)
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(sample)
        file_row.addWidget(browse)
        root.addLayout(file_row)

        # Preview ------------------------------------------------------------
        self.preview_area = QScrollArea()
        self.preview_area.setWidgetResizable(True)
        self.preview_area.setAlignment(Qt.AlignCenter)
        self.preview_label = QLabel("No PDF loaded")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        self.preview_area.setWidget(self.preview_label)
        root.addWidget(self.preview_area, 1)

        nav_row = QHBoxLayout()
        self.btn_first = QPushButton("|<")
        self.btn_prev = QPushButton("<")
        self.btn_next = QPushButton(">")
        self.btn_last = QPushButton(">|")
        self.page_label = QLabel("page 0 / 0")
        for b in (self.btn_first, self.btn_prev, self.btn_next, self.btn_last):
            b.setEnabled(False)
        self.btn_first.clicked.connect(lambda: self._goto(0))
        self.btn_prev.clicked.connect(lambda: self._goto(self._current_page - 1))
        self.btn_next.clicked.connect(lambda: self._goto(self._current_page + 1))
        self.btn_last.clicked.connect(lambda: self._goto(10**9))
        nav_row.addWidget(self.btn_first)
        nav_row.addWidget(self.btn_prev)
        nav_row.addStretch(1)
        nav_row.addWidget(self.page_label)
        nav_row.addStretch(1)
        nav_row.addWidget(self.btn_next)
        nav_row.addWidget(self.btn_last)
        root.addLayout(nav_row)

        # Settings -----------------------------------------------------------
        settings_box = QGroupBox("Conversion settings")
        slay = QHBoxLayout(settings_box)

        slay.addWidget(QLabel("Density (DPI):"))
        self.dpi_combo = QComboBox()
        for dpi in DPI_CHOICES:
            self.dpi_combo.addItem(f"{dpi} DPI", dpi)
        self.dpi_combo.setCurrentIndex(2)
        slay.addWidget(self.dpi_combo)

        slay.addWidget(QLabel("From page:"))
        self.page_from = QSpinBox()
        self.page_from.setRange(1, 1)
        slay.addWidget(self.page_from)

        slay.addWidget(QLabel("To:"))
        self.page_to = QSpinBox()
        self.page_to.setRange(1, 1)
        slay.addWidget(self.page_to)

        slay.addWidget(QLabel("Precision:"))
        self.precision_combo = QComboBox()
        settings = QSettings()
        current = str(settings.value("precision", "auto") or "auto")
        for value in PRECISION_CHOICES:
            self.precision_combo.addItem(_PRECISION_LABELS.get(value, value), value)
        self.precision_combo.setCurrentIndex(
            max(0, PRECISION_CHOICES.index(current) if current in PRECISION_CHOICES else 0)
        )
        self.precision_combo.currentIndexChanged.connect(self._precision_changed)
        self.precision_combo.setToolTip(
            "8-bit quantization fits the model into ~4 GB NVIDIA GPUs.\n"
            "Changing this reloads the model on the next application start."
        )
        slay.addWidget(self.precision_combo)

        slay.addStretch(1)
        root.addWidget(settings_box)

        # Convert row ---------------------------------------------------------
        convert_row = QHBoxLayout()
        self.convert_btn = QPushButton("Convert to Markdown / HTML")
        self.convert_btn.setEnabled(False)
        self.convert_btn.setMinimumHeight(34)
        self.convert_btn.clicked.connect(self._convert)
        convert_row.addWidget(self.convert_btn, 1)
        root.addLayout(convert_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        root.addWidget(self.progress)

        self.status_label = QLabel("Initializing model...")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

    # ------------------------------------------------------------- pipeline
    def _start_pipeline(self) -> None:
        self._model_dir = self._ensure_model_location()
        self._pipeline = PipelineWorker(self._model_dir, self._precision)
        self._pipeline.status.connect(self._on_pipeline_status)
        self._pipeline.progress.connect(self._on_model_progress)
        self._pipeline.ready.connect(self._on_model_ready)
        self._pipeline.failed.connect(self._on_pipeline_failed)
        self._pipeline.start()

    def _ensure_model_location(self) -> Path:
        """Verify free disk space for the download; ask the user to relocate
        the model folder when the target drive does not have enough room."""
        from .. import model_manager
        from ..constants import MODEL_DOWNLOAD_NEEDED_GB

        model_dir = self._model_dir
        if model_manager.is_download_complete(model_dir):
            return model_dir

        needed = MODEL_DOWNLOAD_NEEDED_GB * 1024**3
        try:
            free = model_manager.free_bytes(model_dir)
        except Exception:
            return model_dir
        if free >= needed:
            return model_dir

        QMessageBox.warning(
            self,
            "Low disk space",
            f"The model needs about {MODEL_DOWNLOAD_NEEDED_GB:.0f} GB but the folder\n"
            f"{model_dir}\nonly has {free / 1024**3:.1f} GB free.\n\n"
            "Choose a different folder for the model.",
        )
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose where to store the model", str(Path.home())
        )
        if chosen:
            model_dir = Path(chosen) / "Unlimited-OCR"
            QSettings().setValue("model_dir", str(model_dir))
            self.model_label.setText(f"Model folder: {model_dir}")
        else:
            self.model_label.setText(
                "Warning: low disk space - the download may fail."
            )
        return model_dir

    def _on_pipeline_status(self, text: str) -> None:
        self.model_label.setText(text)
        self.status_label.setText(text)

    def _on_model_progress(self, done: int, total: int, label: str) -> None:
        if not self.model_progress.isVisible():
            self.model_progress.setVisible(True)
        self.model_progress.setMaximum(max(1, total // (1024 * 1024)))
        self.model_progress.setValue(done // (1024 * 1024))
        mb_done = done / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        if total > 0:
            self.model_label.setText(
                f"Downloading model: {mb_done:.0f} / {mb_total:.0f} MB - {label}"
            )

    def _on_model_ready(self, label: str) -> None:
        self.model_progress.setVisible(False)
        self.model_label.setText(f"Model ready - {label}")
        self.status_label.setText("Ready. Load a PDF and press Convert.")
        self.convert_btn.setEnabled(self._doc is not None)

    def _on_pipeline_failed(self, err: str) -> None:
        self.model_progress.setVisible(False)
        self.model_label.setText(f"Model pipeline failed: {err}")
        self.status_label.setText("See the console for details.")

    # ----------------------------------------------------------------- pdf
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF files (*.pdf)")
        if path:
            self.load_pdf(Path(path))

    def _open_sample(self) -> None:
        from ..constants import sample_pdf_path

        p = sample_pdf_path()
        if p and p.exists():
            self.load_pdf(p)
        else:
            self.status_label.setText("Sample PDF not found.")

    def load_pdf(self, path: Path) -> None:
        try:
            self._doc = open_pdf(path)
        except Exception as exc:
            self.status_label.setText(f"Cannot open PDF: {exc}")
            return
        self._pdf_path = Path(path)
        self.file_edit.setText(str(path))
        n = page_count(self._doc)
        self._current_page = 0
        self.page_label.setText(f"page 1 / {n}")
        for b in (self.btn_first, self.btn_prev, self.btn_next, self.btn_last):
            b.setEnabled(True)
        self.page_from.setRange(1, n)
        self.page_to.setRange(1, n)
        self.page_from.setValue(1)
        self.page_to.setValue(n)
        self.convert_btn.setEnabled(engine_ready := self._model_ready())
        self._render_current()
        self.status_label.setText(f"Loaded {path.name} ({n} pages).")

    def _model_ready(self) -> bool:
        from .. import engine

        return engine.ENGINE.is_ready

    def _render_current(self) -> None:
        if self._doc is None:
            return
        qimg = preview_qimage(self._doc, self._current_page)
        self.preview_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.preview_area.viewport().size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        n = page_count(self._doc)
        self.page_label.setText(f"page {self._current_page + 1} / {n}")

    def _goto(self, page: int) -> None:
        if self._doc is None:
            return
        n = page_count(self._doc)
        self._current_page = max(0, min(n - 1, page))
        self._render_current()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if self._doc is not None and self.preview_label.pixmap() is not None:
            self._render_current()

    # ------------------------------------------------------------ conversion
    def _convert(self) -> None:
        if self._doc is None or self._converter is not None:
            return
        if not self._model_ready():
            self.status_label.setText("Model is not ready yet.")
            return

        dpi = self.dpi_combo.currentData()
        from_page = self.page_from.value() - 1
        to_page = self.page_to.value() - 1
        if to_page < from_page:
            from_page, to_page = to_page, from_page

        out_dir = self._pdf_path.parent / "output" if self._pdf_path else Path.cwd() / "output"
        self._converter = ConvertWorker(
            str(self._pdf_path), out_dir, dpi=dpi, page_range=(from_page, to_page)
        )
        self._converter.render_progress.connect(self._on_render_progress)
        self._converter.status.connect(self.status_label.setText)
        self._converter.live_text.connect(self._on_live_text)
        self._converter.page_done.connect(self._on_page_done)
        self._converter.finished_ok.connect(self._on_finished)
        self._converter.failed.connect(self._on_failed)
        self.convert_btn.setEnabled(False)
        self.progress.setValue(0)
        self._converter.start()

    def _on_render_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.progress.setFormat("rendering pages %p%")

    def _on_live_text(self, chunk: str) -> None:
        self.status_label.setText("Generating... (live output in tab 2)")

    def _on_page_done(self, done: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.progress.setFormat("parsing %p%")

    def _on_finished(self, markdown: str, work_dir: str) -> None:
        self.progress.setValue(self.progress.maximum())
        self.status_label.setText(f"Conversion finished - results in {work_dir}")
        self.convert_btn.setEnabled(True)
        self._converter = None
        self.conversion_finished.emit(markdown, work_dir)

    def _on_failed(self, err: str) -> None:
        self.status_label.setText(f"Conversion failed: {err}")
        self.convert_btn.setEnabled(True)
        self._converter = None

    def shutdown(self) -> None:
        for w in (self._pipeline, self._converter):
            if w is not None:
                w.cancel()
        for w in (self._pipeline, self._converter):
            if w is not None:
                w.wait(2000)
