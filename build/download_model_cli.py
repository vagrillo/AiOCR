#!/usr/bin/env python3
"""Download the model using the app's own manager (same code path as the GUI)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiocr.model_manager import default_model_dir, download_model, is_download_complete

target = default_model_dir()
print(f"Model dir: {target}")
if is_download_complete(target):
    print("Already complete.")
    raise SystemExit(0)

state = {"last": 0.0, "t0": time.time(), "last_done": 0}


def cb(done: int, total: int, label: str) -> None:
    now = time.time()
    if now - state["last"] < 2 and done < total:
        return
    dt = now - state["last"]
    speed = (done - state["last_done"]) / max(dt, 1e-9) / 1e6 if dt else 0.0
    state["last"] = now
    state["last_done"] = done
    pct = 100.0 * done / total if total else 0.0
    print(f"[{pct:5.1f}%] {done/1e9:6.2f}/{total/1e9:6.2f} GB  {speed:6.1f} MB/s  {label}", flush=True)


download_model(target, progress_cb=cb)
print("DOWNLOAD COMPLETE")
