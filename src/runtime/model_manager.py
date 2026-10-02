"""
GPU model manager
=================

What this file does
-------------------
Owns every GPU model in the process (SAM 3 and Qwen-Image-2.1), loads each
one once, and decides what sits on the GPU at each stage:

* ``segment`` stage: SAM 3 on the GPU.
* ``edit`` stage: SAM 3 moved to system RAM (if
  ``segmentation.offload_during_edit``), then the image model runs. The
  editor itself swaps its text encoder in and out (step 5).

It also holds a lock so only one GPU job runs at a time, and reports VRAM use.

Why it exists (optimisation step 6)
-----------------------------------
Loading the models takes about 30 s, so they must not be reloaded per
request. One long-running worker (``server.py``) creates a single
``ModelManager`` at start-up and keeps everything warm.
"""

from __future__ import annotations

import gc
import logging
import threading
import time

import torch

from src.diffusion.editor import RoomEditor

log = logging.getLogger(__name__)


def make_segmenter(cfg: dict):
    """Pick the segmentation backend named in config.yaml (segmentation.backend)."""
    backend = cfg["segmentation"].get("backend", "sam3")
    if backend == "sam3":
        from src.segmentation.sam3_segmenter import Sam3Segmenter

        return Sam3Segmenter(cfg)
    if backend == "grounded_sam":
        from src.segmentation.grounded_segmenter import GroundedSegmenter

        return GroundedSegmenter(cfg)
    raise ValueError(f"Unknown segmentation.backend: {backend!r} (use 'sam3' or 'grounded_sam')")


class ModelManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.segmenter = make_segmenter(cfg)
        self.editor = RoomEditor(cfg)
        self.lock = threading.Lock()  # One GPU job at a time
        self._stage: str | None = None

    def load_all(self, warmup: bool = False) -> dict[str, float]:
        """Load both models up front and return how long each took, in seconds."""
        timings = {}
        t = time.perf_counter()
        self.segmenter.load()
        timings["load_segmenter"] = time.perf_counter() - t
        self._park_segmenter()

        t = time.perf_counter()
        self.editor.load()
        timings["load_editor"] = time.perf_counter() - t

        if warmup:
            t = time.perf_counter()
            self.editor.warmup()
            timings["warmup"] = time.perf_counter() - t
        log.info("Models ready: %s | %s", timings, self.vram_report())
        return timings

    # ---------------------------------------------------------------- stages
    def activate(self, stage: str) -> None:
        """Arrange GPU residency for ``stage`` ("segment" or "edit")."""
        if stage == self._stage:
            return
        if stage == "segment":
            self.segmenter.to(self.cfg["segmentation"].get("device", "cuda"))
        elif stage == "edit":
            self._park_segmenter()
        else:
            raise ValueError(f"Unknown stage {stage!r}")
        self._stage = stage

    def _park_segmenter(self) -> None:
        if self.cfg["segmentation"].get("offload_during_edit", True):
            self.segmenter.to("cpu")
            free_gpu_cache()

    # ---------------------------------------------------------------- reporting
    @staticmethod
    def vram_report() -> dict[str, float]:
        """VRAM in GB. ``device_used`` includes other apps (e.g. the Windows desktop)."""
        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        gb = 1024**3
        return {
            "allocated": round(torch.cuda.memory_allocated() / gb, 2),
            "peak_allocated": round(torch.cuda.max_memory_allocated() / gb, 2),
            "reserved": round(torch.cuda.memory_reserved() / gb, 2),
            "device_used": round((total - free) / gb, 2),
            "device_total": round(total / gb, 2),
        }


def free_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
