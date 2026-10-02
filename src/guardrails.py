"""
Photo guardrails
================

What this file does
-------------------
Checks every newly uploaded photo for people (or characters: cartoons,
mascots, figures) before it is stored or edited. If the segmenter finds one
above ``guardrails.min_score``, the photo is rejected and the web app sends
the user back to the start with a warning.

It reuses the segmenter that is already loaded (Grounding DINO + SAM, or
SAM 3), asking it for the phrases in ``guardrails.block_terms``, so it costs
one extra search of about a second and no extra VRAM.

Pictures on the wall
--------------------
A portrait, poster or TV showing a person is part of the room, not someone
in it. So when a person is found, a second search looks for the phrases in
``guardrails.artwork_terms`` (picture frames, paintings, posters, screens).
A person whose box lies at least ``guardrails.inside_fraction`` inside one
of those is ignored. The second search is separate from the first because
the segmenter merges near-identical masks, and a portrait that fills its
frame could otherwise swallow the frame (or the other way round). Rooms with
nobody in them never pay for it.

Mirrors are deliberately not in the default list: a person in a mirror is
usually the photographer, who is really there.

Why it exists
-------------
TensorRoom is for redesigning rooms. Redrawing parts of photos of people is
outside that purpose and open to misuse, so those photos are refused up
front rather than relying on users only choosing furniture.

Only uploads are checked: edited results come from an already-checked photo
and stay inside the worker, so chained edits do not pay for the check again.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from PIL import Image

from src.segmentation.sam3_segmenter import Instance

log = logging.getLogger(__name__)

DEFAULT_TERMS = ["person", "human face", "cartoon character"]
DEFAULT_ARTWORK_TERMS = ["picture frame", "painting", "poster", "television screen"]
DEFAULT_MESSAGE = (
    "This photo looks like it shows a person or a character. TensorRoom only edits rooms, "
    "so take or choose a photo of the room with nobody in it."
)


@dataclass
class GuardResult:
    allowed: bool
    found: list[Instance] = field(default_factory=list)  # Matches that blocked the photo
    ignored: list[Instance] = field(default_factory=list)  # People inside pictures, let through
    message: str = ""
    seconds: float = 0.0


def check_photo(manager, photo: Image.Image) -> GuardResult:
    """Return whether ``photo`` may be edited, using the manager's segmenter."""
    g = manager.cfg.get("guardrails") or {}
    if not g.get("enabled", True):
        return GuardResult(allowed=True)

    t = time.perf_counter()
    ignored: list[Instance] = []
    with manager.lock:
        manager.activate("segment")
        matches = manager.segmenter.segment(photo, g.get("block_terms") or DEFAULT_TERMS)
        found = [m for m in matches if m.score >= g.get("min_score", 0.35)]
        artwork_terms = g.get("artwork_terms", DEFAULT_ARTWORK_TERMS)
        if found and artwork_terms:
            artwork = [
                a.box
                for a in manager.segmenter.segment(photo, artwork_terms)
                if a.score >= g.get("artwork_min_score", 0.35)
            ]
            fraction = g.get("inside_fraction", 0.85)
            in_picture = [inside_any(m.box, artwork, fraction) for m in found]
            ignored = [m for m, inside in zip(found, in_picture) if inside]
            found = [m for m, inside in zip(found, in_picture) if not inside]
    seconds = time.perf_counter() - t

    if ignored:
        log.info("Guardrail ignored people inside pictures: %s", ", ".join(f"{m.label} {m.score:.2f}" for m in ignored))
    if not found:
        return GuardResult(allowed=True, ignored=ignored, seconds=seconds)
    log.info("Guardrail rejected a photo: %s", ", ".join(f"{m.label} {m.score:.2f}" for m in found))
    return GuardResult(
        allowed=False, found=found, ignored=ignored, message=g.get("message") or DEFAULT_MESSAGE, seconds=seconds
    )


def inside_any(box: tuple[int, int, int, int], containers: list[tuple[int, int, int, int]], fraction: float) -> bool:
    """True if at least ``fraction`` of ``box``'s area lies inside one of ``containers`` (all x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    area = max(0, x1 - x0) * max(0, y1 - y0)
    if area == 0:
        return False
    for cx0, cy0, cx1, cy1 in containers:
        overlap = max(0, min(x1, cx1) - max(x0, cx0)) * max(0, min(y1, cy1) - max(y0, cy0))
        if overlap / area >= fraction:
            return True
    return False
