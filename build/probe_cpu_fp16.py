#!/usr/bin/env python3
"""Probe: fp16 weights on CPU (no quantization, no autocast).

CPU fp16 kernels accumulate in fp32 internally, so overflow cannot happen:
coherent output => plumbing is correct and the GPU garbage comes from fp16
overflow; garbage => there is a plumbing bug independent of the backend."""
import contextlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from aiocr.inference import infer_multi_device  # noqa: E402

model_dir = str(ROOT / "models" / "Unlimited-OCR")
MAXLEN = int(__import__("os").environ.get("PROBE_MAXLEN", "300"))

print("Loading fp16 on CPU (no quantization, no autocast)...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model = model.to("cpu").eval()

img = str(ROOT / "build" / "convert_work" / "pages" / "page_0001.png")
t0 = time.time()
pages, n_tok = infer_multi_device(
    model, tokenizer, [img], str(ROOT / "build" / "probe_out"),
    device=torch.device("cpu"),
    dtype=torch.float16,
    autocast_ctx=contextlib.nullcontext,
    max_length=MAXLEN,
    on_text=lambda s: print(s, end="", flush=True),
)
print(f"\n\n[n_tokens={n_tok}, {time.time()-t0:.0f}s]")
