"""
Shared configuration loader
===========================

What this file does
-------------------
Reads ``config.yaml`` from the repository root and turns relative paths in it
into absolute ones. Every new module (the editor, the segmenter, the worker,
the scripts) goes through here, so there is one place that knows where the
config lives.

Why it exists
-------------
``src/diffusion/generator.py`` and ``src/segmentation/sam.py`` each carry their
own copy of ``load_config``. Those are left alone so Jake's code keeps
working, but new code should import from here instead of adding a third copy.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Return the parsed config. Cached, so repeated calls are free."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Point the Hugging Face cache somewhere roomier if the config asks for it.
    # This must happen before diffusers/transformers are imported anywhere.
    hf_cache = (cfg.get("paths") or {}).get("hf_cache_dir")
    if hf_cache:
        os.environ.setdefault("HF_HOME", str(repo_path(hf_cache)))
    return cfg


def repo_path(path: str | os.PathLike) -> Path:
    """Resolve a path from the config: absolute paths are kept, relative ones hang off the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
