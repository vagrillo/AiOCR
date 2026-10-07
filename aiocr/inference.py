"""Device-agnostic multi-page inference for the Unlimited-OCR model.

The model's remote code (modeling_unlimitedocr.py, loaded via
``trust_remote_code=True``) hardcodes CUDA (``.cuda()`` calls and
``torch.autocast("cuda")``). This module mirrors the upstream
``infer_multi`` implementation line by line but routes every tensor to the
backend resolved by :mod:`aiocr.engine` (CUDA, MPS or CPU), so the exact same
document-parsing pipeline runs on NVIDIA GPUs, Apple Silicon and CPU.

The post-processing step replicates the upstream ``save_results`` branch: it
turns ``<PAGE>`` separated raw output into per-page markdown, crops image
regions referenced by the model into ``images/`` and strips residual layout
markers.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .constants import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NO_REPEAT_NGRAM,
    DEFAULT_NGRAM_WINDOW,
    MULTI_PAGE_PROMPT,
)

TextCB = Callable[[str], None]

_STOP_STR = "<｜end▁of▁sentence｜>"
_RESIDUAL_DET = re.compile(r"<\|det\|>\s*[A-Za-z_][\w-]*\s*(?:\[[^\]]*\])?\s*<\|/det\|>")


def remote_module(model):
    """Return the dynamically loaded modeling module (trust_remote_code)."""
    name = type(model).__module__
    mod = sys.modules.get(name)
    if mod is None:
        raise RuntimeError(f"Remote modeling module '{name}' not found in sys.modules")
    return mod


class CallbackStreamer:
    """TextStreamer that forwards decoded chunks to a callback (no printing)."""

    def __init__(self, tokenizer, on_text: TextCB):
        from transformers import TextStreamer

        self._impl = TextStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=False
        )
        # Bind our handler in place of the default print behavior.
        self._impl.on_finalized_text = self._make_handler(on_text)  # type: ignore[method-assign]
        self._tokenizer = tokenizer
        self._on_text = on_text

    @staticmethod
    def _make_handler(on_text: TextCB):
        def handler(text: str, stream_end: bool = False) -> None:
            if text:
                on_text(text)
        return handler

    def put(self, value):
        self._impl.put(value)

    def end(self):
        self._impl.end()


def _strip_residual_markers(text: str) -> str:
    text = _RESIDUAL_DET.sub("", text)
    text = text.replace("<|ref|>", "").replace("<|/ref|>", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def infer_multi_device(
    model,
    tokenizer,
    image_files: List[str],
    output_path: str | Path,
    device=None,
    dtype=None,
    autocast_ctx: Optional[Callable[[], contextlib.AbstractContextManager]] = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM,
    ngram_window: int = DEFAULT_NGRAM_WINDOW,
    on_text: Optional[TextCB] = None,
) -> Tuple[List[str], int]:
    """Run multi-page document parsing; returns (per-page markdown, n_tokens)."""
    import torch
    from PIL import ImageOps

    mod = remote_module(model)
    if device is None:
        device = next(model.parameters()).device
    if dtype is None:
        dtype = torch.float32 if device.type == "cpu" else model.config.torch_dtype
    if autocast_ctx is None:
        def autocast_ctx():
            if device.type == "cuda":
                return torch.autocast("cuda", dtype=dtype)
            if device.type == "cpu" and dtype == torch.bfloat16:
                return torch.autocast("cpu", dtype=torch.bfloat16)
            return contextlib.nullcontext()

    model.disable_torch_init()

    if not image_files:
        raise ValueError("image_files must be a non-empty list for multi-image inference")

    output_path = str(output_path)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)

    conversation = [
        {
            "role": "<|User|>",
            "content": MULTI_PAGE_PROMPT,
            "images": list(image_files),
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    formatted_prompt = mod.format_messages(
        conversations=conversation, sft_format="plain", system_prompt=""
    )

    patch_size = 16
    downsample_ratio = 4

    images = mod.load_pil_images(conversation)
    image_transform = mod.BasicImageTransform(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True
    )

    image_token = "<image>"
    image_token_id = 128815

    text_splits = formatted_prompt.split(image_token)

    images_list: List = []
    images_seq_mask: List[bool] = []
    tokenized_str: List[int] = []
    images_spatial_crop: List[List[int]] = []

    num_queries = math.ceil((image_size // patch_size) / downsample_ratio)

    tokenized_sep = mod.text_encode(tokenizer, text_splits[0], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    for image in images:
        if image_size <= 640:
            image = image.resize((image_size, image_size))
        global_view = ImageOps.pad(
            image,
            (image_size, image_size),
            color=tuple(int(x * 255) for x in image_transform.mean),
        )
        images_list.append(image_transform(global_view).to(dtype))
        images_spatial_crop.append([1, 1])

        tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
        tokenized_image += [image_token_id]
        tokenized_str += tokenized_image
        images_seq_mask += [True] * len(tokenized_image)

    tokenized_sep = mod.text_encode(tokenizer, text_splits[1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    # Add bos token (upstream infer_multi does the same).
    tokenized_str = [0] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    input_ids = torch.LongTensor(tokenized_str)
    images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

    images_ori = torch.stack(images_list, dim=0).to(device)
    images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
    dummy_crop = torch.zeros((1, 3, image_size, image_size), dtype=dtype).to(device)

    streamer = CallbackStreamer(tokenizer, on_text) if on_text else None

    # Mirror upstream: disable config sliding_window so DynamicCache does not
    # truncate prefill tokens; the ring buffer reads _ring_window instead.
    orig_sw = getattr(model.config, "sliding_window_size", None) or getattr(
        model.config, "sliding_window", None
    )
    model.config._ring_window = orig_sw
    model.config.sliding_window = None

    try:
        with autocast_ctx():
            with torch.no_grad():
                gen_kwargs = dict(
                    input_ids=input_ids.unsqueeze(0).to(device),
                    images=[(dummy_crop, images_ori)],
                    images_seq_mask=images_seq_mask.unsqueeze(0).to(device),
                    images_spatial_crop=images_spatial_crop,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                    max_length=max_length,
                    use_cache=True,
                )
                if no_repeat_ngram_size > 0 and ngram_window > 0:
                    gen_kwargs["logits_processor"] = [
                        mod.SlidingWindowNoRepeatNgramProcessor(
                            no_repeat_ngram_size, ngram_window
                        )
                    ]
                elif no_repeat_ngram_size > 0:
                    gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
                output_ids = model.generate(**gen_kwargs)
    finally:
        model.config.sliding_window = orig_sw

    outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).shape[1]:])
    if outputs.endswith(_STOP_STR):
        outputs = outputs[: -len(_STOP_STR)]
    outputs = outputs.strip()

    output_tokens = len(mod.text_encode(tokenizer, outputs, bos=False, eos=False))

    pages = postprocess_output(model, images, outputs, output_path)
    return pages, output_tokens


def postprocess_output(model, images: list, raw_output: str, output_path: str | Path) -> List[str]:
    """Replicate the upstream ``save_results`` post-processing on CPU."""
    mod = remote_module(model)
    output_path = str(output_path)

    pages_raw = [p.strip() for p in raw_output.split("<PAGE>")[1:]]
    if not pages_raw:
        pages_raw = [raw_output.strip()]

    processed: List[str] = []
    for page_idx, page_output in enumerate(pages_raw):
        if page_idx >= len(images):
            processed.append(_strip_residual_markers(page_output))
            continue

        matches_ref, matches_images, matches_other = mod.re_match(page_output)
        image_prefix = f"page_{page_idx}_"
        try:
            mod.process_image_with_refs(
                images[page_idx].copy(), matches_ref, output_path, image_prefix=image_prefix
            )
        except Exception:
            pass  # cropping/drawing must never fail the text conversion

        for idx, match_image in enumerate(matches_images):
            page_output = page_output.replace(
                match_image, f"![](images/{image_prefix}{idx}.jpg)\n"
            )
        for match_other in matches_other:
            page_output = page_output.replace(match_other, "")
            page_output = page_output.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

        processed.append(_strip_residual_markers(page_output))

    return [p for p in processed if p]
