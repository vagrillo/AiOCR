"""Model download management.

Downloads the baidu/Unlimited-OCR snapshot (config, remote code, tokenizer and
safetensors weights) into a local directory with resumable progress reporting.
A marker file lets subsequent starts skip the network entirely.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

# The xet storage backend (hf-xet) can hang on its final commit phase for
# multi-GB shards; the classic HTTP backend resumes reliably, so we prefer it.
# An explicit user-provided value always wins.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from tqdm import tqdm

from .constants import (
    DOWNLOAD_MARKER,
    MODEL_ALLOW_PATTERNS,
    MODEL_REPO_ID,
)

ProgressCB = Callable[[int, int, str], None]  # bytes_done, bytes_total, current_file


def free_bytes(path: Path) -> int:
    """Free space on the volume holding `path` (works even if path is new)."""
    import shutil

    path = Path(path).resolve()
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def default_model_dir() -> Path:
    """Model storage location: $AIOCR_MODEL_DIR or ./models/Unlimited-OCR."""
    env = os.environ.get("AIOCR_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / "models" / MODEL_REPO_ID.split("/")[-1]


def is_download_complete(model_dir: Path) -> bool:
    return (Path(model_dir) / DOWNLOAD_MARKER).exists()


def _write_marker(model_dir: Path) -> None:
    (Path(model_dir) / DOWNLOAD_MARKER).write_text(
        f"repo={MODEL_REPO_ID}\ncompleted={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )


class _ProgressTqdm(tqdm):  # type: ignore[misc, valid-type]
    """tqdm subclass that aggregates every active bar into one callback.

    huggingface_hub instantiates one tqdm per downloaded file (and expects the
    full tqdm API, e.g. the ``get_lock`` classmethod); we register each live
    instance and emit throttled overall (done, total, filename) updates.
    """

    _bars: Dict[int, "_ProgressTqdm"] = {}
    _callback: Optional[ProgressCB] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_emit = 0.0
        if not self.disable:
            _ProgressTqdm._bars[id(self)] = self
            self._emit(force=True)

    def update(self, n=1):  # type: ignore[override]
        super().update(n)
        if not self.disable:
            self._emit()

    def close(self):  # type: ignore[override]
        super().close()
        _ProgressTqdm._bars.pop(id(self), None)
        self._emit(force=True)

    # -----------------------------------------------------------------------
    def _emit(self, force: bool = False) -> None:
        cb = _ProgressTqdm._callback
        if cb is None:
            return
        now = time.time()
        if not force and now - self._last_emit < 0.2:
            return
        self._last_emit = now
        done = sum(float(b.n) for b in _ProgressTqdm._bars.values())
        total = sum(float(b.total or 0) for b in _ProgressTqdm._bars.values())
        desc = str(self.desc or "")
        cb(int(done), int(total), desc)

    @classmethod
    def reset(cls, callback: Optional[ProgressCB]) -> None:
        cls._bars.clear()
        cls._callback = callback


def download_model(
    model_dir: Path,
    progress_cb: Optional[ProgressCB] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Path:
    """Download (or resume) the model snapshot into `model_dir`."""
    from huggingface_hub import snapshot_download

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if is_download_complete(model_dir):
        return model_dir

    _ProgressTqdm.reset(progress_cb)
    try:
        snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=str(model_dir),
            allow_patterns=MODEL_ALLOW_PATTERNS,
            max_workers=2,
            tqdm_class=_ProgressTqdm,
        )
    finally:
        _ProgressTqdm.reset(None)

    if should_cancel is not None and should_cancel():
        raise RuntimeError("Download cancelled")

    if not (model_dir / "config.json").exists():
        raise RuntimeError(
            "Download finished but config.json is missing; the model folder is incomplete."
        )
    _write_marker(model_dir)
    return model_dir
