"""AiOCR main window: two tabs (convert / result)."""

from __future__ import annotations

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QStatusBar, QTabWidget

from ..constants import APP_NAME, __version__
from .convert_tab import ConvertTab
from .result_tab import ResultTab


class MainWindow(QMainWindow):
    def __init__(self, model_dir, precision: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.resize(1180, 860)

        self.tabs = QTabWidget()
        self.convert_tab = ConvertTab(model_dir, precision)
        self.result_tab = ResultTab()
        self.tabs.addTab(self.convert_tab, "1 - PDF && Convert")
        self.tabs.addTab(self.result_tab, "2 - Result && Export")
        self.convert_tab.conversion_finished.connect(self._on_conversion_finished)
        self.setCentralWidget(self.tabs)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Powered by baidu/Unlimited-OCR")

    def _on_conversion_finished(self, markdown: str, work_dir: str) -> None:
        self.result_tab.set_result(markdown, work_dir)
        self.tabs.setCurrentIndex(1)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.convert_tab.shutdown()
        super().closeEvent(event)
