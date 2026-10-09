"""
Mask, crop and paste-back helpers
=================================

What this file does
-------------------
Pure image maths (NumPy + OpenCV + Pillow, no GPU) that sits between the
segmenter and the image model:

1. ``hull_mask`` / ``union_masks`` / ``grow_mask`` fill each chosen object
   out to its convex hull (so the gaps between legs can change shape too),
   combine them, and grow the mask slightly so edges and contact shadows are
   included.
2. ``prepare_crop`` cuts a box around the mask with some surrounding context
   and scales it so its long side is ``max_side`` (optimisation step 2: the
   model never sees a full 12 MP phone photo, because latency grows faster
   than pixel count).
3. ``paste_back`` scales the edited crop back down and blends it into the
   original through a soft-edged (feathered) mask (optimisation step 3).
   Pixels where the blend weight is zero are copied from the original
   untouched, so repeated edits do not slowly drift in colour.

Why it is separate
------------------
Keeping this free of model code means it can be unit-tested on any machine
(see ``tests/test_masks.py``) and reused by the worker and the scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class CropInfo:
    """Where the crop came from in the original photo and the size the model works at."""

    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in original pixels (x1/y1 exclusive)
    model_size: tuple[int, int]     # width, height sent to the model


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Combine several boolean masks of the same shape into one."""
    if not masks:
        raise ValueError("No masks were selected.")
    out = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        out |= m.astype(bool)
    return out


def hull_mask(mask: np.ndarray) -> np.ndarray:
    """
    Fill an object's outline out to its convex hull.

    A table's mask is its top plus thin legs; the rug between the legs is not
    included. If the new piece has a different shape (say a pedestal base),
    the model cannot paint there and paste-back cuts it off. The hull adds
    those gaps to the editable area.
    """
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return mask.astype(bool)
    hull = cv2.convexHull(points)
    out = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool) | mask.astype(bool)


def grow_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate a boolean mask by roughly ``pixels`` using an elliptical kernel."""
    if pixels <= 0:
        return mask.astype(bool)
    size = 2 * pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def feather_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """
    Turn a boolean mask into blend weights in [0, 1].

    The inside of the mask is always exactly 1 (fully replaced); the weights
    fade to 0 over roughly ``pixels`` outside it, and are exactly 0 beyond that.
    """
    hard = mask.astype(np.float32)
    if pixels <= 0:
        return hard
    # Grow first so the fade happens outside the mask rather than eating into it.
    grown = grow_mask(mask, pixels).astype(np.float32)
    sigma = max(pixels / 2.0, 0.5)
    soft = cv2.GaussianBlur(grown, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Anything outside the grown region is forced back to exactly zero so the
    # original pixels there are guaranteed to survive untouched.
    soft = np.where(grown > 0, soft, 0.0)
    return np.clip(np.maximum(soft, hard), 0.0, 1.0)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight bounding box (x0, y0, x1, y1), with x1/y1 exclusive."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("The mask is empty.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_box_for_mask(
    mask: np.ndarray, context: float, min_side: int
) -> tuple[int, int, int, int]:
    """
    Box around the mask plus ``context`` (a fraction of the mask's own size) on
    every side, at least ``min_side`` pixels each way, clamped to the image.
    """
    h, w = mask.shape
    x0, y0, x1, y1 = mask_bbox(mask)
    bw, bh = x1 - x0, y1 - y0
    pad_x, pad_y = int(round(bw * context)), int(round(bh * context))
    cx0, cy0, cx1, cy1 = x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y

    # Enforce a minimum size by expanding around the centre.
    def _expand(lo: int, hi: int, limit: int) -> tuple[int, int]:
        need = min(min_side, limit) - (hi - lo)
        if need > 0:
            lo -= need // 2
            hi += need - need // 2
        # Shift back inside the image without shrinking.
        if lo < 0:
            hi, lo = hi - lo, 0
        if hi > limit:
            lo, hi = max(0, lo - (hi - limit)), limit
        return lo, hi

    cx0, cx1 = _expand(cx0, cx1, w)
    cy0, cy1 = _expand(cy0, cy1, h)
    return max(0, cx0), max(0, cy0), min(w, cx1), min(h, cy1)


def fit_to_model_size(width: int, height: int, max_side: int, multiple: int) -> tuple[int, int]:
    """Scale (width, height) so the long side is ``max_side``, rounded to ``multiple``."""
    scale = max_side / max(width, height)
    tw = max(multiple, int(round(width * scale / multiple)) * multiple)
    th = max(multiple, int(round(height * scale / multiple)) * multiple)
    return tw, th


def prepare_crop(
    image: Image.Image,
    mask: np.ndarray,
    *,
    context: float,
    min_side: int,
    max_side: int,
    multiple: int,
) -> tuple[Image.Image, Image.Image, CropInfo]:
    """
    Cut the region the model should edit.

    Returns the RGB crop and an L-mode mask (white = change), both already at
    the model's working size, plus the information needed to paste back.
    """
    box = crop_box_for_mask(mask, context, min_side)
    x0, y0, x1, y1 = box
    model_size = fit_to_model_size(x1 - x0, y1 - y0, max_side, multiple)

    crop = image.convert("RGB").crop(box).resize(model_size, Image.LANCZOS)
    mask_img = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize(model_size, Image.NEAREST)
    return crop, mask_img, CropInfo(box=box, model_size=model_size)


def annotate_region(crop: Image.Image, mask_img: Image.Image, colour=(255, 0, 0), opacity=0.55) -> Image.Image:
    """Paint the masked area in translucent colour (used by the "annotated" mask mode)."""
    base = np.asarray(crop.convert("RGB"), dtype=np.float32)
    m = (np.asarray(mask_img) > 127)[..., None]
    tint = np.array(colour, dtype=np.float32)
    out = np.where(m, base * (1 - opacity) + tint * opacity, base)
    return Image.fromarray(out.round().astype(np.uint8))


def paste_back(
    original: Image.Image,
    edited_crop: Image.Image,
    mask: np.ndarray,
    info: CropInfo,
    feather_px: int,
) -> Image.Image:
    """
    Blend the edited crop into the original photo.

    ``mask`` is the full-size (already grown) mask. Only pixels with a
    non-zero blend weight change; everything else is copied byte-for-byte.
    """
    x0, y0, x1, y1 = info.box
    orig = np.asarray(original.convert("RGB")).copy()
    edited = np.asarray(edited_crop.convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS), dtype=np.float32)

    alpha = feather_mask(mask, feather_px)[y0:y1, x0:x1][..., None]
    region = orig[y0:y1, x0:x1].astype(np.float32)
    blended = region * (1.0 - alpha) + edited * alpha
    # Write back only where alpha > 0 so untouched pixels stay exact.
    orig[y0:y1, x0:x1] = np.where(alpha > 0, blended.round().clip(0, 255), region).astype(np.uint8)
    return Image.fromarray(orig)


def draw_overlay(image: Image.Image, masks: list[np.ndarray], labels: list[str], max_side: int = 1600) -> Image.Image:
    """Preview for the UI: each instance tinted in its own colour with a numbered label."""
    img = image.convert("RGB")
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    canvas = np.asarray(img, dtype=np.float32).copy()
    palette = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]
    label_spots = []
    for i, m in enumerate(masks):
        small = cv2.resize(m.astype(np.uint8), img.size, interpolation=cv2.INTER_NEAREST).astype(bool)
        colour = np.array(palette[i % len(palette)], dtype=np.float32)
        canvas[small] = canvas[small] * 0.5 + colour * 0.5
        if small.any():
            ys, xs = np.nonzero(small)
            label_spots.append((int(xs.min()), int(ys.min()), f"{i}: {labels[i]}", palette[i % len(palette)]))
    out = canvas.round().astype(np.uint8)
    for x, y, text, colour in label_spots:
        cv2.putText(out, text, (x + 4, max(18, y + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (x + 4, max(18, y + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
    return Image.fromarray(out)
