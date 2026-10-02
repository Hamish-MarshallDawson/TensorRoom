"""
Text-prompted segmentation with SAM 3
=====================================

What this file does
-------------------
Given a room photo and the words the user typed ("sofa", "coffee table"),
returns a mask, box, score and label for every matching object, using Meta's
SAM 3 through Hugging Face ``transformers``.

The "key/dictionary" step lives here too: ``expand_terms`` looks each user
word up in ``segmentation.vocabulary`` in ``config.yaml`` so "couch" also
searches for "sofa", and so on. Duplicate masks found by two synonyms are
merged.

Why SAM 3
---------
SAM 3 does detection and masking in one model from a text phrase and ships
inside modern ``transformers``.

Optimisations
-------------
* The photo is shrunk to ``segmentation.max_side`` before SAM 3 sees it
  (SAM 3 works at about 1008 px internally anyway); masks are still returned
  at the original resolution.
* Vision features are computed once per photo and reused for every phrase.
* ``to("cpu")`` lets the model manager park SAM 3 in system RAM while the
  image model needs the GPU.

Note: ``facebook/sam3`` is gated. Accept the terms on its Hugging Face page
and run ``uv run hf auth login`` before downloading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class Instance:
    """One detected object."""

    label: str                       # The user's term that found it
    phrase: str                      # The phrase SAM 3 was actually given
    score: float
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 in original pixels
    mask: np.ndarray                 # bool, H x W at original resolution


def expand_terms(terms: list[str], vocabulary: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Map user terms to (term, phrase) pairs via the config dictionary. Unknown terms are used as-is."""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in terms:
        term = raw.strip().lower()
        if not term:
            continue
        for phrase in vocabulary.get(term, [term]):
            if phrase not in seen:
                seen.add(phrase)
                pairs.append((term, phrase))
    return pairs


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def dedupe(instances: list[Instance], iou_threshold: float) -> list[Instance]:
    """Keep the higher-scoring mask whenever two masks overlap more than ``iou_threshold``."""
    kept: list[Instance] = []
    for inst in sorted(instances, key=lambda i: i.score, reverse=True):
        if all(_iou(inst.mask, k.mask) < iou_threshold for k in kept):
            kept.append(inst)
    return kept


class Sam3Segmenter:
    def __init__(self, cfg: dict):
        seg = cfg["segmentation"]
        self.model_id: str = seg["model_id"]
        self.device: str = seg.get("device", "cuda")
        self.dtype = getattr(torch, seg.get("dtype", "bfloat16"))
        self.threshold: float = seg.get("threshold", 0.5)
        self.mask_threshold: float = seg.get("mask_threshold", 0.5)
        self.max_side: int = seg.get("max_side", 1536)
        self.dedupe_iou: float = seg.get("dedupe_iou", 0.8)
        self.vocabulary: dict[str, list[str]] = {k.lower(): v for k, v in (seg.get("vocabulary") or {}).items()}
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import Sam3Model, Sam3Processor

        self.processor = Sam3Processor.from_pretrained(self.model_id)
        self.model = Sam3Model.from_pretrained(self.model_id, dtype=self.dtype).to(self.device).eval()

    def to(self, device: str) -> None:
        """Move SAM 3 between the GPU and system RAM."""
        if self.model is not None:
            self.model.to(device)

    @torch.inference_mode()
    def segment(self, image: Image.Image, terms: list[str]) -> list[Instance]:
        if self.model is None:
            self.load()
        image = image.convert("RGB")
        orig_w, orig_h = image.size

        # Shrink large photos first; masks are scaled back to full size in post-processing.
        scale = min(1.0, self.max_side / max(orig_w, orig_h))
        work = image if scale >= 1.0 else image.resize((round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)

        device = next(self.model.parameters()).device
        img_inputs = self.processor(images=work, return_tensors="pt").to(device, dtype=self.dtype)
        vision_embeds = self.model.get_vision_features(pixel_values=img_inputs.pixel_values)

        found: list[Instance] = []
        for term, phrase in expand_terms(terms, self.vocabulary):
            text_inputs = self.processor(text=phrase, return_tensors="pt").to(device)
            outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
            result = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=[[orig_h, orig_w]],
            )[0]
            for mask, box, score in zip(result["masks"], result["boxes"], result["scores"]):
                m = mask.detach().to("cpu").numpy().astype(bool)
                if not m.any():
                    continue
                x0, y0, x1, y1 = (int(round(v)) for v in box.detach().float().cpu().tolist())
                found.append(Instance(label=term, phrase=phrase, score=float(score), box=(x0, y0, x1, y1), mask=m))

        return dedupe(found, self.dedupe_iou)
