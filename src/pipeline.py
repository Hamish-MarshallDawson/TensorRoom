"""
Room edit pipeline
==================

What this file does
-------------------
Joins the pieces into the two operations the app needs:

``segment(photo, terms)``
    Finds every object matching the user's words (SAM 3) and returns them so
    the user can tick which ones to change.

``edit(photo, instances, instruction, quality)``
    Combines and grows the chosen masks, crops around them (step 2), edits
    the crop with Qwen-Image-2.1 (steps 1 and 5), and pastes the result back
    through a soft-edged mask (step 3). It also records how long each stage
    took and the VRAM peak, which ``scripts/benchmark.py`` reports on.

Both functions take the shared ``ModelManager`` and hold its lock, so they
are safe to call from the web worker's threads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from PIL import Image

from src.runtime.model_manager import ModelManager
from src.segmentation.masks import CropInfo, grow_mask, hull_mask, paste_back, prepare_crop, union_masks
from src.segmentation.sam3_segmenter import Instance


@dataclass
class EditResult:
    image: Image.Image                       # Full-resolution result
    crop_info: CropInfo
    timings: dict[str, float] = field(default_factory=dict)
    vram: dict[str, float] = field(default_factory=dict)


def segment(manager: ModelManager, photo: Image.Image, terms: list[str]) -> tuple[list[Instance], dict[str, float]]:
    """Find every object matching ``terms``. Returns the instances and the time the search took."""
    with manager.lock:
        t = time.perf_counter()
        manager.activate("segment")
        instances = manager.segmenter.segment(photo, terms)
        return instances, {"segment": time.perf_counter() - t}


def edit(
    manager: ModelManager,
    photo: Image.Image,
    instances: list[Instance],
    instruction: str,
    quality: str = "preview",
    seed: int = 42,
) -> EditResult:
    """
    Edit the chosen objects in ``photo`` and paste the result back into it.

    The chosen masks are combined, the crop around them is edited, and only the
    pixels inside the (feathered) mask change. Timings and VRAM are recorded
    along the way.
    """
    e = manager.cfg["editor"]
    timings: dict[str, float] = {}

    with manager.lock:
        t = time.perf_counter()
        # Each object is filled out to its convex hull separately, so two
        # chairs far apart do not merge into one big region.
        shape = e.get("mask_shape", "hull")
        object_masks = [hull_mask(i.mask) if shape == "hull" else i.mask for i in instances]
        mask = grow_mask(union_masks(object_masks), e["mask_grow_px"])
        crop, crop_mask, info = prepare_crop(
            photo,
            mask,
            context=e["crop_context"],
            min_side=e["min_crop_side"],
            max_side=e["max_side"],
            multiple=e["size_multiple"],
        )
        timings["prepare"] = time.perf_counter() - t

        t = time.perf_counter()
        manager.activate("edit")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        edited = manager.editor.edit(crop, crop_mask, instruction, quality=quality, seed=seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["edit"] = time.perf_counter() - t
        vram = manager.vram_report()

        t = time.perf_counter()
        result = paste_back(photo, edited, mask, info, e["feather_px"])
        timings["paste_back"] = time.perf_counter() - t

    return EditResult(image=result, crop_info=info, timings=timings, vram=vram)


def instances_by_id(instances: list[Instance], ids: list[int]) -> list[Instance]:
    """Pick the instances the user ticked (ids are list positions)."""
    chosen = [instances[i] for i in ids if 0 <= i < len(instances)]
    if not chosen:
        raise ValueError("Select at least one object to edit.")
    return chosen
