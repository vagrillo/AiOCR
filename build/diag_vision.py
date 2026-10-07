#!/usr/bin/env python3
"""Vision-tower health check: run SAM+CLIP+projector on a page image and
inspect the feature statistics (NaN/Inf, mean, std) that get injected into
the language model."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from aiocr import engine, model_manager  # noqa: E402

model_dir = model_manager.default_model_dir()
runtime = engine.ENGINE.load(str(model_dir), "auto", status_cb=lambda s: print("[engine]", s))

model = engine.ENGINE.model
tokenizer = engine.ENGINE.tokenizer
inner = model.model  # UnlimitedOCRModel (vision tower + projector live here)
mod = sys.modules[type(model).__module__]

img_path = str(ROOT / "build" / "convert_work" / "pages" / "page_0001.png")
if not Path(img_path).exists():
    raise SystemExit("page image missing - run convert_cli first")

conversation = [{"role": "<|User|>", "content": "<image>x", "images": [img_path]}]
pil = mod.load_pil_images(conversation)[0]
print("PIL image:", pil.size, pil.mode)

from PIL import ImageOps  # noqa: E402
transform = mod.BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
global_view = ImageOps.pad(pil, (1024, 1024), color=(127, 127, 127))
t = transform(global_view).to(runtime.dtype).to(runtime.device)
print("input tensor:", t.shape, t.dtype, "mean", t.float().mean().item())

with torch.no_grad():
    sam = inner.sam_model(t.unsqueeze(0))
    clip = inner.vision_model(t.unsqueeze(0), sam)
    feats = torch.cat((clip[:, 1:], sam.flatten(2).permute(0, 2, 1)), dim=-1)
    proj = inner.projector(feats)

for name, x in (("sam", sam), ("clip", clip), ("feats", feats), ("projector", proj)):
    xf = x.float()
    print(
        f"{name:10s} shape={tuple(x.shape)} dtype={x.dtype} "
        f"mean={xf.mean().item():+.4f} std={xf.std().item():.4f} "
        f"nan={torch.isnan(xf).sum().item()} inf={torch.isinf(xf).sum().item()} "
        f"absmax={xf.abs().max().item():.2f}"
    )
