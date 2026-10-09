"""
One-off weight shrinking (optimisation step 4)
==============================================

What this script does
---------------------
Loads Qwen-Image-2.1's two big components from the Hugging Face cache,
shrinks them, and saves the results so the worker can load the small
versions directly every time it starts::

    uv run python scripts/quantise_models.py
    uv run python scripts/quantise_models.py --transformer nf4   # more VRAM headroom

* Generator (7.1B parameters): FP8 by default, about 7.5 GB instead of 14 GB.
  FP8 runs on the RTX 50 series' FP8 tensor cores. ``nf4`` is about 4.5 GB;
  in community tests on an RTX 5090 it was no faster but gave more headroom.
* Text encoder (Qwen3-VL-8B): NF4, about 5-6 GB instead of 16 GB, with its
  vision tower kept at full precision because it reads the room photo.

Output goes to ``editor.quantised_dir`` (``models/qwen-image-2.1-quantised``).

Why do it once
--------------
The full-precision checkpoint is about 31 GB, roughly as much as the system
RAM of the PC it was developed on. Shrinking on every start-up would be slow
and risks running out of memory.
Each component is loaded straight onto the GPU and shrunk one at a time,
with memory freed in between, to keep peak RAM low.

Run ``scripts/download_models.py`` first. Takes a few minutes.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, repo_path  # noqa: E402


def _free() -> None:
    """Release a component's memory before loading the next one, so peak usage stays low."""
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    cfg = load_config()
    e = cfg["editor"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transformer", choices=["fp8", "nf4"], default=e.get("transformer_quant", "fp8"))
    parser.add_argument("--text-encoder", choices=["nf4"], default=e.get("text_encoder_quant", "nf4"))
    parser.add_argument("--out", default=e["quantised_dir"], help="output folder")
    args = parser.parse_args()

    import torch
    from diffusers import QwenImage21Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration

    from src.diffusion.editor import text_encoder_quant_config, transformer_quant_config

    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = e["base_model_id"]

    print(f"Generator -> {args.transformer}")
    t = time.perf_counter()
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        base,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        quantization_config=transformer_quant_config(args.transformer),
        device_map="cuda",
    )
    # torchao (FP8) weights cannot be stored as safetensors, so they use PyTorch's format.
    transformer.save_pretrained(out / "transformer", safe_serialization=args.transformer != "fp8")
    print(f"  saved in {time.perf_counter() - t:.0f}s, VRAM {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    del transformer
    _free()

    print(f"Text encoder -> {args.text_encoder}")
    t = time.perf_counter()
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        base,
        subfolder="text_encoder",
        dtype=torch.bfloat16,
        quantization_config=text_encoder_quant_config(args.text_encoder),
        device_map="cuda",
    )
    text_encoder.save_pretrained(out / "text_encoder")
    print(f"  saved in {time.perf_counter() - t:.0f}s, VRAM {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    del text_encoder
    _free()

    manifest = {
        "base_model_id": base,
        "transformer": args.transformer,
        "text_encoder": args.text_encoder,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": {p: version(p) for p in ("torch", "diffusers", "transformers", "torchao", "bitsandbytes")},
    }
    (out / "quantisation.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done: {out}")
    if args.transformer != e.get("transformer_quant"):
        print(f"Remember to set editor.transformer_quant: {args.transformer} in config.yaml.")


if __name__ == "__main__":
    main()
