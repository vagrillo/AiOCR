#!/usr/bin/env python3
"""Probe: run the model's OWN upstream infer_multi (unmodified remote code)
on the same hybrid bf16 setup. Decides whether the garbage output comes from
AiOCR's device-agnostic wrapper or from the model/checkpoint/environment."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

model_dir = str(ROOT / "models" / "Unlimited-OCR")
MAXLEN = int(__import__("os").environ.get("PROBE_MAXLEN", "300"))

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    max_memory={0: "2.8GiB", "cpu": "5GiB"},
)
model.eval()

# Upstream patches the projector dtype nowhere, but the bf16+offload combo
# reintroduces the fp32 projector output; keep parity with our wrapper.
_target = getattr(model, "model", model)
_orig_fwd = _target.projector.forward
_target.projector.forward = lambda x: _orig_fwd(x).to(torch.bfloat16)

img = str(ROOT / "build" / "convert_work" / "pages" / "page_0001.png")
t0 = time.time()
outputs, n_tokens = model.infer_multi(
    tokenizer,
    prompt="<image>Multi page parsing.",
    image_files=[img],
    output_path=str(ROOT / "build" / "probe_upstream_out"),
    image_size=1024,
    max_length=MAXLEN,
    no_repeat_ngram_size=35,
    ngram_window=1024,
    save_results=False,
)
print("=" * 60)
print(outputs[:1200])
print("=" * 60)
print(f"[upstream infer_multi: n_tokens={n_tokens}, {time.time()-t0:.0f}s]")
