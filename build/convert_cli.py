#!/usr/bin/env python3
"""Headless end-to-end conversion test: real model, real GPU backend.

Usage: python build/convert_cli.py [pdf] [pages] [precision]
       python build/convert_cli.py testmultipage.pdf 0-1 8bit
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aiocr import engine, model_manager  # noqa: E402
from aiocr.pdf_utils import pdf_to_images, open_pdf  # noqa: E402

pdf = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "testmultipage.pdf")
pages = sys.argv[2] if len(sys.argv) > 2 else "0-1"
precision = sys.argv[3] if len(sys.argv) > 3 else "auto"

model_dir = model_manager.default_model_dir()
if not model_manager.is_download_complete(model_dir):
    raise SystemExit(f"Model not downloaded yet: {model_dir}")

t0 = time.time()
runtime = engine.ENGINE.load(str(model_dir), precision, status_cb=lambda s: print("  [engine]", s))
print(f"Load time: {time.time() - t0:.1f}s -> {runtime.label} dtype={runtime.dtype_name} quant={runtime.quantization}")

a, b = (int(x) for x in pages.split("-"))
doc = open_pdf(pdf)
workdir = ROOT / "build" / "convert_work"
t0 = time.time()
imgs = pdf_to_images(doc, workdir, dpi=150, page_range=(a, b))
print(f"Rendered {len(imgs)} pages in {time.time() - t0:.1f}s")

state = {"last": 0.0, "chars": 0}


def on_text(chunk: str) -> None:
    state["chars"] += len(chunk)
    now = time.time()
    if now - state["last"] > 3:
        state["last"] = now
        print(f"  ... {state['chars']} chars generated", flush=True)


t0 = time.time()
from aiocr.inference import infer_multi_device  # noqa: E402

pages_md, n_tokens = infer_multi_device(
    engine.ENGINE.model,
    engine.ENGINE.tokenizer,
    image_files=[str(p) for p in imgs],
    output_path=workdir,
    device=runtime.device,
    dtype=runtime.dtype,
    autocast_ctx=runtime.autocast,
    on_text=on_text,
)
gen_t = time.time() - t0
print(f"Generation: {gen_t:.1f}s, {n_tokens} tokens, {n_tokens / max(gen_t, 1e-9):.1f} tok/s")

md = "\n\n---\n\n".join(pages_md)
print("=" * 60)
print(md[:2000])
print("=" * 60)
out = workdir / "converted.md"
out.write_text(md, encoding="utf-8")
img_files = list((workdir / "images").glob("*.jpg")) if (workdir / "images").is_dir() else []
print(f"Saved {out} ({len(md)} chars), cropped images: {len(img_files)}")
