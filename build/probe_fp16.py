#!/usr/bin/env python3
"""Probe: fp16 weights (NO quantization), hybrid GPU+CPU offload, short
generation. If output is coherent, quantization was corrupting the LLM;
if still garbage, the issue is elsewhere (fp16 numerics / plumbing)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

model_dir = str(ROOT / "models" / "Unlimited-OCR")
MAXLEN = int(os.environ.get("PROBE_MAXLEN", "400"))
PRECISION = os.environ.get("PROBE_DTYPE", "float16")

print(f"Loading fp16 hybrid offload (dtype={PRECISION}, max_length={MAXLEN})...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=getattr(torch, PRECISION),
    low_cpu_mem_usage=True,
    device_map="auto",
    max_memory={0: "2.8GiB", "cpu": "5GiB"},
)
model.eval()
print("offload map sample:", {k: v for k, v in list(model.hf_device_map.items())[:5]}, "...")

# Same projector dtype normalization as aiocr.engine.load.
_target = getattr(model, "model", model)
_orig_fwd = _target.projector.forward
_dtype = torch.bfloat16 if PRECISION == "bfloat16" else torch.float16
_target.projector.forward = lambda x: _orig_fwd(x).to(_dtype)

from aiocr.inference import infer_multi_device  # noqa: E402

img = str(ROOT / "build" / "convert_work" / "pages" / "page_0001.png")

def autocast():
    return torch.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else torch.autocast("cpu", dtype=torch.bfloat16)

import time  # noqa: E402
t0 = time.time()
pages, n_tok = infer_multi_device(
    model, tokenizer, [img], str(ROOT / "build" / "probe_out"),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float16 if PRECISION == "float16" else torch.bfloat16,
    autocast_ctx=autocast,
    max_length=MAXLEN,
    on_text=lambda s: print(s, end="", flush=True),
)
print(f"\n\n[n_tokens={n_tok}, {time.time()-t0:.0f}s]")
