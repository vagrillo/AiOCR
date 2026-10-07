#!/usr/bin/env python3
"""Offscreen GUI smoke test: builds the window, loads the sample PDF,
fakes a ready engine, exports a canned result, saves screenshots."""
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QSettings, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import aiocr.engine as engine  # noqa: E402
from aiocr.app import MainWindow, sample_pdf_path  # noqa: E402

# Keep the smoke test away from the real 6.7 GB download.
import aiocr.model_manager as mm  # noqa: E402
mm.is_download_complete = lambda p: True


def _fake_load(self, d, precision="auto", status_cb=None):
    import torch
    from aiocr.engine import RuntimeInfo
    self.runtime = RuntimeInfo(
        device=torch.device("cpu"), dtype=torch.float32, dtype_name="float32",
        quantization=None, label="STUB (smoke test)", notes=[],
    )
    self.model = object()
    self.tokenizer = object()
    if status_cb:
        status_cb("stub model loaded")
    return self.runtime


engine.Engine.load = _fake_load

app = QApplication(sys.argv)
QSettings().clear()
win = MainWindow(ROOT / "models" / "Unlimited-OCR", "auto")
win.show()

results = {"exports": False}


def fail_and_quit():
    results["exports"] = False
    app.quit()


def step2():
    try:
        ct = win.convert_tab
        print("model label:", ct.model_label.text())
        print("file:", ct.file_edit.text())
        print("pages:", ct.page_label.text())
        ct._goto(1)
        ct.grab().save("build/smoke_tab1.png")
        win.grab().save("build/smoke_main.png")
        md = (
            "# Smoke test\n\nSome **bold** text and a table:\n\n"
            "| A | B |\n|---|---|\n| 1 | 2 |\n\n"
            "```python\nprint('hello AiOCR')\n```\n"
        )
        workdir = ROOT / "build" / "smoke_work"
        workdir.mkdir(exist_ok=True)
        (workdir / "images").mkdir(exist_ok=True)
        win.result_tab.set_result(md, str(workdir))
        from aiocr.exporter import export_html, export_markdown
        p1 = export_markdown(md, workdir / "images", workdir / "smoke.md")
        p2 = export_html(md, workdir, workdir / "smoke.html")
        assert p1.exists() and "<table>" in p2.read_text(encoding="utf-8"), "export failed"
        print("md export:", p1)
        print("html export: ok (table present)")
        results["exports"] = True
        win.result_tab.grab().save("build/smoke_tab2.png")
    except Exception:
        traceback.print_exc()
    finally:
        win.close()
        app.quit()


def step1():
    try:
        win.grab().save("build/smoke_first_run.png")
    except Exception:
        traceback.print_exc()
        fail_and_quit()
        return
    QTimer.singleShot(300, step2)


assert sample_pdf_path().exists(), "sample pdf missing"
QTimer.singleShot(300, step1)
app.exec()

print("RESULT:", "PASS" if results["exports"] else "FAIL")
sys.exit(0 if results["exports"] else 1)
