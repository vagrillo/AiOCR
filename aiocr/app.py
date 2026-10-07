"""Application bootstrap: settings, model folder resolution, main window."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from .constants import APP_NAME, ORG_NAME, resource_path, sample_pdf_path
from .model_manager import default_model_dir
from .ui.main_window import MainWindow


def icon_path() -> Path:
    return resource_path("assets/icon.png")


def resolve_model_dir(settings: QSettings) -> Path:
    """Model folder: user setting -> $AIOCR_MODEL_DIR -> ./models/Unlimited-OCR."""
    custom = str(settings.value("model_dir", "") or "")
    return Path(custom) if custom else default_model_dir()


def resolve_precision(settings: QSettings) -> str:
    return str(settings.value("precision", "auto") or "auto")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    icon = icon_path()
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    settings = QSettings()

    model_dir = resolve_model_dir(settings)
    precision = resolve_precision(settings)

    window = MainWindow(model_dir, precision)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
