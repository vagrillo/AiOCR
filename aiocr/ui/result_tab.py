"""Tab 2: view the OCR result and export it as Markdown or HTML."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..exporter import export_html, export_markdown, markdown_to_html


class ResultTab(QWidget):
    """Rendered result (Markdown/HTML/source) + export buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._markdown: str = ""
        self._work_dir: Optional[Path] = None

        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Rendered Markdown", "HTML Preview", "Markdown source"])
        self.view_combo.currentIndexChanged.connect(self._refresh)
        bar.addWidget(self.view_combo)
        bar.addStretch(1)

        self.btn_md = QPushButton("Export .md...")
        self.btn_html = QPushButton("Export .html...")
        self.btn_copy = QPushButton("Copy all")
        self.btn_md.clicked.connect(self._export_md)
        self.btn_html.clicked.connect(self._export_html)
        self.btn_copy.clicked.connect(self._copy)
        for b in (self.btn_md, self.btn_html, self.btn_copy):
            bar.addWidget(b)
        root.addLayout(bar)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        root.addWidget(self.browser, 1)

        self.path_label = QLabel("No result yet - run a conversion in tab 1.")
        self.path_label.setStyleSheet("color: #666;")
        root.addWidget(self.path_label)

        self._set_enabled(False)

    # ----------------------------------------------------------------- state
    def _set_enabled(self, on: bool) -> None:
        for b in (self.btn_md, self.btn_html, self.btn_copy):
            b.setEnabled(on)
        self.view_combo.setEnabled(on)

    def set_result(self, markdown: str, work_dir: str) -> None:
        self._markdown = markdown
        self._work_dir = Path(work_dir)
        self.path_label.setText(f"Working folder: {work_dir}")
        self._set_enabled(True)
        if self.view_combo.currentIndex() == 0:
            self._refresh()
        else:
            self.view_combo.setCurrentIndex(0)

    def clear(self) -> None:
        self._markdown = ""
        self._work_dir = None
        self.browser.clear()
        self.path_label.setText("No result yet - run a conversion in tab 1.")
        self._set_enabled(False)

    # ----------------------------------------------------------------- views
    def _refresh(self) -> None:
        idx = self.view_combo.currentIndex()
        doc = self.browser.document()
        if self._work_dir is not None:
            # Let relative image paths (images/page_x_y.jpg) resolve on screen.
            doc.setBaseUrl(QUrl.fromLocalFile(str(self._work_dir) + "/"))
        if idx == 0:
            doc.setMarkdown(self._markdown)
        elif idx == 1:
            base = self._work_dir or Path.cwd()
            doc.setHtml(markdown_to_html(self._markdown, base, embed_images=False))
        else:
            doc.setPlainText(self._markdown)

    # ---------------------------------------------------------------- export
    def _default_dir(self) -> Path:
        if self._work_dir is not None:
            return self._work_dir.parent
        return Path.home()

    def _export_md(self) -> None:
        if not self._markdown:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Markdown", str(self._default_dir() / "ocr_result.md"),
            "Markdown (*.md)",
        )
        if not dest:
            return
        images_dir = self._work_dir / "images" if self._work_dir else None
        path = export_markdown(self._markdown, images_dir, dest)
        self.path_label.setText(f"Exported: {path}")

    def _export_html(self) -> None:
        if not self._markdown:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export HTML", str(self._default_dir() / "ocr_result.html"),
            "HTML (*.html)",
        )
        if not dest:
            return
        path = export_html(self._markdown, self._work_dir or Path.cwd(), dest)
        self.path_label.setText(f"Exported: {path}")

    def _copy(self) -> None:
        if self._markdown:
            QGuiApplication.clipboard().setText(self._markdown)
            self.path_label.setText("Copied to clipboard.")
