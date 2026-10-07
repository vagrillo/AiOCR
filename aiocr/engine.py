"""Model loading and hardware runtime selection.

Resolves the best execution backend for the host machine:

* NVIDIA CUDA  - bf16 on Ampere+, fp16 on older cards (incl. GTX 10xx/4 GB
                 cards), with optional bitsandbytes 8-bit quantization that
                 fits the ~3B parameter model into 4 GB of VRAM.
* Apple M-series - MPS in float16 (runs natively on M1/M2/M3/M4).
* CPU          - bfloat16 when available, otherwise float32 (slow fallback).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .constants import (
    AUTO_4BIT_VRAM_THRESHOLD_GB,
    AUTO_8BIT_VRAM_THRESHOLD_GB,
)

# Reduce fragmentation with 8-bit weights; must be set before CUDA context init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

StatusCB = Callable[[str], None]


@dataclass
class RuntimeInfo:
    device: object = None                # torch.device
    dtype: object = None                 # torch.dtype used for tensors/weights
    dtype_name: str = ""
    quantization: Optional[str] = None   # "4-bit" | "8-bit" | None
    label: str = ""                      # human readable, shown in the UI
    notes: List[str] = field(default_factory=list)

    def autocast(self):  # noqa: ANN201 - torch.autocast context manager
        """Context manager for mixed precision, mirroring upstream bf16 CUDA."""
        import contextlib
        import torch

        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=self.dtype)
        if self.device.type == "cpu" and self.dtype == torch.bfloat16:
            return torch.autocast("cpu", dtype=torch.bfloat16)
        return contextlib.nullcontext()


def resolve_runtime(precision: str = "auto") -> RuntimeInfo:
    """Pick device/dtype/quantization. `precision`: auto|8bit|16bit|32bit."""
    import torch

    precision = (precision or "auto").lower()

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / 1024**3
        cc = torch.cuda.get_device_capability(idx)

        dtype = torch.bfloat16 if cc >= (8, 0) else torch.float16
        quant: Optional[str] = None
        notes: List[str] = []

        if precision in ("4bit", "8bit"):
            quant = "4-bit" if precision == "4bit" else "8-bit"
        elif precision == "auto" and vram_gb <= AUTO_4BIT_VRAM_THRESHOLD_GB:
            quant = "4-bit"
            notes.append(
                f"{vram_gb:.0f} GB VRAM detected: 4-bit NF4 quantization enabled "
                "automatically (8-bit weights plus the vision encoder do not fit)."
            )
        elif precision == "auto" and vram_gb <= AUTO_8BIT_VRAM_THRESHOLD_GB:
            quant = "8-bit"
            notes.append(
                f"{vram_gb:.0f} GB VRAM detected: 8-bit quantization enabled automatically."
            )
        elif precision == "32bit":
            dtype = torch.float32

        if quant:
            try:
                import bitsandbytes  # noqa: F401
            except Exception:
                quant = None
                notes.append(
                    "bitsandbytes is not installed: falling back to 16-bit. "
                    "Install it with `pip install bitsandbytes`."
                )
                if precision == "32bit":
                    dtype = torch.float32

        label = f"CUDA - {name} ({vram_gb:.0f} GB, sm_{cc[0]}{cc[1]})"
        if quant:
            label += f" - {quant}"
        return RuntimeInfo(
            device=torch.device("cuda", idx),
            dtype=dtype,
            dtype_name=str(dtype).replace("torch.", ""),
            quantization=quant,
            label=label,
            notes=notes,
        )

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        notes: List[str] = []
        dtype = torch.float16
        if precision == "32bit":
            dtype = torch.float32
        if precision == "8bit":
            notes.append(
                "8-bit quantization (bitsandbytes) is not available on Apple MPS; "
                "using 16-bit instead. Unified memory on M-series Macs handles the full model."
            )
        return RuntimeInfo(
            device=torch.device("mps"),
            dtype=dtype,
            dtype_name=str(dtype).replace("torch.", ""),
            quantization=None,
            label="Apple Silicon GPU (MPS)",
            notes=notes,
        )

    notes = ["No GPU backend detected: running on CPU. Inference will be very slow."]
    dtype = torch.bfloat16 if precision != "32bit" else torch.float32
    return RuntimeInfo(
        device=torch.device("cpu"),
        dtype=dtype,
        dtype_name=str(dtype).replace("torch.", ""),
        quantization=None,
        label="CPU",
        notes=notes,
    )


class Engine:
    """Holds the loaded model so it is initialized exactly once per session."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.runtime: Optional[RuntimeInfo] = None
        self.model_dir: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self, model_dir: str, precision: str = "auto", status_cb: Optional[StatusCB] = None) -> RuntimeInfo:
        import torch
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        def status(msg: str) -> None:
            if status_cb:
                status_cb(msg)

        status("Resolving hardware backend...")
        runtime = resolve_runtime(precision)
        for note in runtime.notes:
            status(note)

        status(f"Loading tokenizer from {model_dir} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        load_kwargs = dict(
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=runtime.dtype,
            low_cpu_mem_usage=True,
        )
        if runtime.quantization == "8-bit":
            status("Loading model with 8-bit quantization (bitsandbytes)...")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = {"": runtime.device}
        elif runtime.quantization == "4-bit":
            status("Loading model with 4-bit NF4 quantization (bitsandbytes)...")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=runtime.dtype,
            )
            load_kwargs["device_map"] = {"": runtime.device}
        else:
            status(f"Loading model weights ({runtime.dtype_name})...")

        model = AutoModel.from_pretrained(model_dir, **load_kwargs)

        if runtime.quantization is None:
            model = model.to(runtime.device)
        model.eval()

        # The vision tower mixes fp32 ops (LayerNorm/CLIP tail) with the fp16
        # language path; bitsandbytes Linears also bypass autocast casting, so
        # the projector output can come out fp32 while the embedding table is
        # fp16. masked_scatter_ in the model's forward requires exact dtype
        # match - normalize the projector output to the runtime dtype.
        self._normalize_projector_dtype(model, runtime.dtype)

        self.model = model
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.model_dir = model_dir

        status(f"Model ready on {runtime.label}.")
        return runtime

    @staticmethod
    def _normalize_projector_dtype(model, dtype) -> None:
        # The projector lives on the inner UnlimitedOCRModel (model.model),
        # not on the ForCausalLM wrapper returned by AutoModel.
        for target in (model, getattr(model, "model", None)):
            if target is None:
                continue
            projector = getattr(target, "projector", None)
            if projector is None:
                continue

            original_forward = projector.forward

            def forward(x, _orig=original_forward, _dtype=dtype):
                return _orig(x).to(_dtype)

            projector.forward = forward  # type: ignore[method-assign]
            return

    # Convenience accessors used by the inference module --------------------
    @property
    def device(self):
        return self.runtime.device

    @property
    def dtype(self):
        return self.runtime.dtype


# Single shared engine instance for the whole application.
ENGINE = Engine()
