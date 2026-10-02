"""
Environment check
=================

What this script does
---------------------
Run it first, and again whenever something odd happens::

    uv run python scripts/check_env.py               # quick checks
    uv run python scripts/check_env.py --probe-sysmem

It reports:

* whether PyTorch is a CUDA build (the venv originally had ``+cpu``) and
  whether it supports the RTX 5080 (Blackwell, compute capability 12.0,
  needs CUDA 12.8 or newer);
* total and free VRAM. The Windows desktop and open apps typically hold
  about 1.4 GB before TensorRoom starts;
* library versions against what Qwen-Image-2.1 needs (diffusers with
  ``QwenImage21Pipeline``, transformers >= 5.17, torchao, bitsandbytes);
* whether you are logged in to Hugging Face (SAM 3 is gated);
* whether the downloads and pre-shrunk weights are in place.

``--probe-sysmem`` covers optimisation step 7. It deliberately allocates
slightly more than the card's VRAM. If that *succeeds*, Windows' "CUDA -
Sysmem Fallback Policy" is quietly spilling into system RAM, which makes
overflows several times slower instead of failing with an error. The script
then explains how to turn it off. The probe briefly uses about 17 GB of
system RAM, so close heavy apps first.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, repo_path  # noqa: E402

OK, WARN, FAIL = "[ ok ]", "[warn]", "[FAIL]"


def _ver(pkg: str) -> str | None:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def _major_minor(v: str) -> tuple[int, int]:
    parts = v.split("+")[0].split(".")
    return int(parts[0]), int(parts[1])


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        print(FAIL, "PyTorch is not installed.")
        return False
    print(OK if torch.version.cuda else FAIL, f"torch {torch.__version__} (CUDA build: {torch.version.cuda})")
    if not torch.version.cuda:
        print("       Reinstall a CUDA build, see the README section 'Room editing pipeline'.")
        return False
    if not torch.cuda.is_available():
        print(FAIL, "CUDA build installed but no GPU visible. Check the NVIDIA driver.")
        return False
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch_ok = f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list()
    print(OK if arch_ok else FAIL, f"GPU {name}, compute capability {cap[0]}.{cap[1]}, supported by this build: {arch_ok}")
    free, total = torch.cuda.mem_get_info()
    gb = 1024**3
    used_elsewhere = (total - free) / gb
    print(OK if used_elsewhere < 2.5 else WARN, f"VRAM {total / gb:.1f} GB total, {free / gb:.1f} GB free ({used_elsewhere:.1f} GB used by other apps)")
    return arch_ok


def check_libraries() -> None:
    want = {
        "diffusers": None,
        "transformers": (5, 17),
        "accelerate": None,
        "torchao": (0, 18),
        "bitsandbytes": (0, 50),
        "fastapi": None,
        "opencv-python": None,
    }
    for pkg, minimum in want.items():
        v = _ver(pkg)
        if v is None:
            print(FAIL, f"{pkg} not installed")
        elif minimum and _major_minor(v) < minimum:
            print(FAIL, f"{pkg} {v} (need >= {minimum[0]}.{minimum[1]})")
        else:
            print(OK, f"{pkg} {v}")
    try:
        importlib.import_module("diffusers").QwenImage21Pipeline  # noqa: B018
        print(OK, "diffusers has QwenImage21Pipeline")
    except (ImportError, AttributeError):
        print(FAIL, "diffusers lacks QwenImage21Pipeline: install diffusers from GitHub (see pyproject.toml).")


def check_files(cfg: dict) -> None:
    try:
        from huggingface_hub import get_token

        print(OK if get_token() else WARN, "Hugging Face token " + ("found" if get_token() else "missing: run `uv run hf auth login` (needed for SAM 3)"))
    except ImportError:
        print(WARN, "huggingface_hub not installed")
    e = cfg["editor"]
    qdir = repo_path(e["quantised_dir"])
    for part in ("transformer", "text_encoder"):
        p = qdir / part
        print(OK if p.exists() else WARN, f"pre-shrunk {part}: {p}" + ("" if p.exists() else " (run scripts/quantise_models.py)"))
    acc = repo_path(e["acceleration"]["local_dir"])
    has_acc = acc.exists() and any(acc.rglob("qwenimage21_pdd.py"))
    print(OK if has_acc else WARN, f"4-step adapter: {acc}" + ("" if has_acc else " (run scripts/download_models.py)"))


def probe_sysmem_fallback() -> None:
    import torch

    free, total = torch.cuda.mem_get_info()
    target = total + 1024**3  # 1 GB more than the card physically has
    chunk = 256 * 1024**2
    blocks, allocated = [], 0
    try:
        while allocated < target:
            blocks.append(torch.empty(chunk, dtype=torch.uint8, device="cuda"))
            allocated += chunk
    except torch.cuda.OutOfMemoryError:
        print(OK, f"Sysmem fallback is OFF: allocation stopped at {allocated / 1024**3:.1f} GB with a clear out-of-memory error.")
        return
    finally:
        del blocks
        torch.cuda.empty_cache()
    print(WARN, f"Sysmem fallback is ON: allocated {allocated / 1024**3:.1f} GB on a {total / 1024**3:.1f} GB card.")
    print("       NVIDIA Control Panel > Manage 3D settings > Program Settings > add this python.exe:")
    print(f"       {sys.executable}")
    print("       then set 'CUDA - Sysmem Fallback Policy' to 'Prefer No Sysmem Fallback'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-sysmem", action="store_true", help="test whether VRAM overflow silently spills into system RAM")
    args = parser.parse_args()

    cfg = load_config()
    print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
    gpu_ok = check_torch()
    print()
    check_libraries()
    print()
    check_files(cfg)
    if args.probe_sysmem:
        print()
        if gpu_ok:
            probe_sysmem_fallback()
        else:
            print(WARN, "Skipping the sysmem probe: no usable GPU.")


if __name__ == "__main__":
    main()
