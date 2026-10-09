"""
Text-prompted segmentation with Grounding DINO + SAM (stand-in for SAM 3)
=========================================================================

What this file does
-------------------
Same job and same interface as ``sam3_segmenter.py``: given a room photo
and the user's words, return a mask, box, score and label for every match.
It does it in two stages:

1. Grounding DINO finds a box for each object matching the phrases.
2. SAM (the original Segment Anything) turns each box into a precise mask.

Both models are the versions built into modern ``transformers``, so they
run in the main environment alongside Qwen-Image-2.1.

Why it exists
-------------
SAM 3 is gated: Meta has to approve each Hugging Face account before it can
be downloaded. These models are open (Apache 2.0), so the platform can be
tested end to end while access is pending. Select it in ``config.yaml``::

    segmentation:
      backend: grounded_sam

SAM 3 is still the better long-term choice: one model instead of two, and
better at finding every instance of a concept. Switch back to
``backend: sam3`` once access is granted.

Optimisations
-------------
* The photo is shrunk to ``segmentation.max_side`` for both models; masks
  are scaled back to the original size at the end.
* All boxes go through SAM in a single batch.
* Both models run in float32: they are small (about 0.9B parameters
  together), and the model manager moves them to system RAM while the
  image model runs.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from src.segmentation.sam3_segmenter import Instance, dedupe, expand_terms


class GroundedSegmenter:
    """Text-prompted segmentation with Grounding DINO (boxes) + SAM (masks). Same interface as ``Sam3Segmenter``."""

    def __init__(self, cfg: dict):
        seg = cfg["segmentation"]
        g = seg.get("grounded_sam") or {}
        self.detector_id: str = g.get("detector_id", "IDEA-Research/grounding-dino-base")
        self.sam_id: str = g.get("sam_id", "facebook/sam-vit-huge")
        self.box_threshold: float = g.get("box_threshold", 0.3)
        self.text_threshold: float = g.get("text_threshold", 0.25)
        self.device: str = seg.get("device", "cuda")
        self.max_side: int = seg.get("max_side", 1536)
        self.dedupe_iou: float = seg.get("dedupe_iou", 0.8)
        self.vocabulary: dict[str, list[str]] = {k.lower(): v for k, v in (seg.get("vocabulary") or {}).items()}
        self.detector = self.det_processor = self.sam = self.sam_processor = None

    def load(self) -> None:
        """Download (if needed) and load both the detector and SAM."""
        from transformers import AutoProcessor, GroundingDinoForObjectDetection, SamModel, SamProcessor

        self.det_processor = AutoProcessor.from_pretrained(self.detector_id)
        self.detector = GroundingDinoForObjectDetection.from_pretrained(self.detector_id).to(self.device).eval()
        self.sam_processor = SamProcessor.from_pretrained(self.sam_id)
        self.sam = SamModel.from_pretrained(self.sam_id).to(self.device).eval()

    def to(self, device: str) -> None:
        """Move both models between the GPU and system RAM."""
        for model in (self.detector, self.sam):
            if model is not None:
                model.to(device)

    @torch.inference_mode()
    def segment(self, image: Image.Image, terms: list[str]) -> list[Instance]:
        if self.detector is None:
            self.load()
        pairs = expand_terms(terms, self.vocabulary)
        if not pairs:
            return []
        phrase_to_term = {phrase: term for term, phrase in pairs}

        image = image.convert("RGB")
        orig_w, orig_h = image.size
        scale = min(1.0, self.max_side / max(orig_w, orig_h))
        work = image if scale >= 1.0 else image.resize((round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)
        device = next(self.detector.parameters()).device

        # 1. Boxes from Grounding DINO, for all phrases in one pass.
        phrases = [phrase for _, phrase in pairs]
        det_inputs = self.det_processor(images=work, text=[phrases], return_tensors="pt").to(device)
        det_out = self.detector(**det_inputs)
        det = self.det_processor.post_process_grounded_object_detection(
            det_out,
            det_inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(work.height, work.width)],
        )[0]
        boxes = det["boxes"].detach().float().cpu()
        if len(boxes) == 0:
            return []
        scores = det["scores"].detach().float().cpu().tolist()
        # Prefer "text_labels"; only touch "labels" on older versions (it warns on newer ones).
        labels = det["text_labels"] if "text_labels" in det else det.get("labels")

        # 2. One mask per box from SAM, all boxes in one batch.
        sam_inputs = self.sam_processor(images=work, input_boxes=[boxes.tolist()], return_tensors="pt").to(device)
        sam_out = self.sam(**sam_inputs, multimask_output=False)
        masks = self.sam_processor.post_process_masks(
            sam_out.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu()
        )[0]  # (num_boxes, 1, H, W) at the shrunk size

        found: list[Instance] = []
        for i, box in enumerate(boxes.tolist()):
            m = masks[i, 0].numpy().astype(np.uint8)
            if scale < 1.0:
                m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            m = m.astype(bool)
            if not m.any():
                continue
            phrase = str(labels[i]).strip() if labels is not None else phrases[0]
            # Grounding DINO can return a fragment or a merge of phrases; map back to the user's word.
            term = phrase_to_term.get(phrase) or next((t for p, t in phrase_to_term.items() if p in phrase or phrase in p), phrase)
            x0, y0, x1, y1 = (int(round(v / scale)) for v in box)
            found.append(Instance(label=term, phrase=phrase, score=float(scores[i]), box=(x0, y0, x1, y1), mask=m))

        return dedupe(found, self.dedupe_iou)
