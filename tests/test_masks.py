"""
Tests for the mask, crop and paste-back helpers
===============================================

What these cover
----------------
``src/segmentation/masks.py`` is plain NumPy/OpenCV, so these run on any
machine without a GPU or model downloads::

    uv run pytest tests/test_masks.py

The most important test is ``test_paste_back_leaves_outside_untouched``: it
checks optimisation step 3's promise that pixels outside the soft-edged mask
come back byte-for-byte identical, which is what stops colour drift across
repeated edits.
"""

import numpy as np
from PIL import Image

from src.segmentation.masks import (
    crop_box_for_mask,
    feather_mask,
    fit_to_model_size,
    grow_mask,
    hull_mask,
    paste_back,
    prepare_crop,
    union_masks,
)


def _photo(w=1200, h=800):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def _box_mask(w=1200, h=800, box=(500, 300, 700, 450)):
    m = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


def test_union_and_grow():
    a = _box_mask(box=(10, 10, 20, 20))
    b = _box_mask(box=(100, 100, 110, 110))
    u = union_masks([a, b])
    assert u.sum() == a.sum() + b.sum()
    g = grow_mask(u, 5)
    assert g.sum() > u.sum() and g[u].all()


def test_hull_fills_gap_between_legs():
    # A crude "table": a top and two legs with a gap between them.
    m = np.zeros((100, 100), dtype=bool)
    m[20:30, 10:90] = True   # top
    m[30:80, 12:18] = True   # left leg
    m[30:80, 82:88] = True   # right leg
    h = hull_mask(m)
    assert h[m].all()
    assert not m[60, 50] and h[60, 50]  # Between the legs is now editable
    assert not h[90, 50]                 # Below the legs is not


def test_feather_is_one_inside_and_zero_far_away():
    m = _box_mask()
    alpha = feather_mask(m, 16)
    assert np.all(alpha[m] == 1.0)
    assert alpha[0, 0] == 0.0
    assert 0.0 < alpha[300, 490] < 1.0  # Soft edge just outside the box


def test_crop_box_stays_in_bounds_and_respects_min_side():
    m = _box_mask(box=(1190, 790, 1200, 800))  # Tiny object in the corner
    x0, y0, x1, y1 = crop_box_for_mask(m, context=0.5, min_side=384)
    assert 0 <= x0 < x1 <= 1200 and 0 <= y0 < y1 <= 800
    assert x1 - x0 >= 384 and y1 - y0 >= 384


def test_fit_to_model_size_rounds_to_multiple():
    w, h = fit_to_model_size(1500, 1000, max_side=1024, multiple=32)
    assert w == 1024 and h % 32 == 0 and abs(h - 683) <= 32


def test_prepare_crop_shapes():
    crop, mask_img, info = prepare_crop(_photo(), _box_mask(), context=0.6, min_side=384, max_side=1024, multiple=32)
    assert crop.size == mask_img.size == info.model_size
    assert max(info.model_size) == 1024
    assert mask_img.mode == "L" and np.asarray(mask_img).max() == 255


def test_paste_back_leaves_outside_untouched():
    photo = _photo()
    mask = _box_mask()
    _, _, info = prepare_crop(photo, mask, context=0.6, min_side=384, max_side=1024, multiple=32)
    white = Image.new("RGB", info.model_size, (255, 255, 255))
    result = paste_back(photo, white, mask, info, feather_px=16)

    before, after = np.asarray(photo), np.asarray(result)
    alpha = feather_mask(mask, 16)
    outside = alpha == 0
    assert np.array_equal(before[outside], after[outside])
    assert np.all(after[mask] == 255)
