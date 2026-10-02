"""
Tests for the people guardrail
==============================

What these cover
----------------
``src/guardrails.py`` decides whether a photo may be edited from what the
segmenter finds. A fake segmenter stands in for the real models, so these run
on any machine::

    uv run pytest tests/test_guardrails.py
"""

import threading

import numpy as np
from PIL import Image

from src import guardrails
from src.segmentation.sam3_segmenter import Instance


PERSON_BOX = (40, 20, 60, 60)


class FakeSegmenter:
    """Finds people with ``scores`` (at PERSON_BOX) and wall art at ``artwork`` [(score, box), ...]."""

    def __init__(self, scores, artwork=()):
        self.scores = scores
        self.artwork = artwork
        self.asked = []

    def segment(self, image, terms):
        self.asked.append(terms)
        mask = np.ones((4, 4), dtype=bool)
        if terms == guardrails.DEFAULT_ARTWORK_TERMS:
            return [Instance(label=terms[0], phrase=terms[0], score=s, box=b, mask=mask) for s, b in self.artwork]
        return [Instance(label=terms[0], phrase=terms[0], score=s, box=PERSON_BOX, mask=mask) for s in self.scores]


class FakeManager:
    def __init__(self, scores, artwork=(), guard_cfg=None):
        self.cfg = {"guardrails": guard_cfg if guard_cfg is not None else {"block_terms": ["person"], "min_score": 0.35}}
        self.segmenter = FakeSegmenter(scores, artwork)
        self.lock = threading.Lock()
        self.stages = []

    def activate(self, stage):
        self.stages.append(stage)


PHOTO = Image.new("RGB", (4, 4))


def test_empty_room_is_allowed():
    manager = FakeManager(scores=[])
    result = guardrails.check_photo(manager, PHOTO)
    assert result.allowed
    assert manager.segmenter.asked == [["person"]]  # No artwork search when nobody was found
    assert manager.stages == ["segment"]


def test_person_is_refused_with_message():
    result = guardrails.check_photo(FakeManager(scores=[0.8]), PHOTO)
    assert not result.allowed
    assert [m.label for m in result.found] == ["person"]
    assert result.message == guardrails.DEFAULT_MESSAGE


def test_weak_match_below_min_score_is_allowed():
    assert guardrails.check_photo(FakeManager(scores=[0.2]), PHOTO).allowed


def test_disabled_guardrail_skips_the_search():
    manager = FakeManager(scores=[0.9], guard_cfg={"enabled": False})
    assert guardrails.check_photo(manager, PHOTO).allowed
    assert manager.segmenter.asked == []


def test_missing_config_uses_default_terms():
    manager = FakeManager(scores=[], guard_cfg={})
    guardrails.check_photo(manager, PHOTO)
    assert manager.segmenter.asked == [guardrails.DEFAULT_TERMS]


def test_portrait_in_a_frame_is_allowed():
    manager = FakeManager(scores=[0.8], artwork=[(0.6, (30, 10, 70, 70))])
    result = guardrails.check_photo(manager, PHOTO)
    assert result.allowed
    assert len(result.ignored) == 1
    assert manager.segmenter.asked[1] == guardrails.DEFAULT_ARTWORK_TERMS


def test_person_beside_a_painting_is_refused():
    # The painting only covers the top of the person's box.
    result = guardrails.check_photo(FakeManager(scores=[0.8], artwork=[(0.6, (30, 0, 70, 30))]), PHOTO)
    assert not result.allowed


def test_weak_artwork_match_does_not_excuse_a_person():
    result = guardrails.check_photo(FakeManager(scores=[0.8], artwork=[(0.1, (30, 10, 70, 70))]), PHOTO)
    assert not result.allowed


def test_inside_any():
    assert guardrails.inside_any((10, 10, 20, 20), [(0, 0, 100, 100)], 0.85)
    assert not guardrails.inside_any((10, 10, 20, 20), [(15, 0, 100, 100)], 0.85)  # Only half inside
    assert not guardrails.inside_any((10, 10, 20, 20), [], 0.85)
    assert not guardrails.inside_any((10, 10, 10, 20), [(0, 0, 100, 100)], 0.85)  # Empty box
