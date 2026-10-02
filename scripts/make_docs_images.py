"""
README image generator
======================

What this script does
---------------------
Makes the example images in ``docs/images/`` that the README shows:

1. Generates a realistic living room from text with Qwen-Image-2.1 itself,
   so the README uses no real home and no copyrighted photo.
2. Runs the real pipeline on it: finds the objects, then makes a few edits
   (restyled sofa, swapped rug and so on) with the full model.
3. Lays each result out next to the original in the same visual style as
   the web app, and records the timings in ``docs/images/results.json``.

Run it with the GPU worker stopped (they cannot share 16 GB)::

    uv run python scripts/make_docs_images.py

Images are saved as JPEG: they are much smaller than PNG for photos, and
``*.png`` is git-ignored in this repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.config import load_config, repo_path  # noqa: E402

ROOM_PROMPT = (
    "A realistic photo of a bright Scandinavian living room taken on a phone, eye level. "
    "A large light grey fabric three-seater sofa against a white wall, a rectangular oak coffee table "
    "in front of it on a large patterned wool rug, a mustard yellow armchair on the right, "
    "a black floor lamp, a potted fiddle leaf fig plant, a large window with soft natural daylight, "
    "oak floorboards. Natural colours, sharp focus, no people."
)

EDITS = [
    ("sofa", "a dark green velvet mid-century sofa"),
    ("rug", "a round natural jute rug"),
    ("coffee table", "a round white marble coffee table"),
    ("chair", "a cognac leather armchair"),
]

INK, PAPER, SIGNAL = (0, 0, 0), (255, 255, 255), (31, 31, 255)


def _font(names: list[str], size: int):
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


MONO = _font(["consolab.ttf", "DejaVuSansMono-Bold.ttf"], 22)
HEAD = _font(["arialbd.ttf", "DejaVuSans-Bold.ttf"], 30)


def tag(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, bg) -> None:
    """Mono uppercase label in a solid block, as used in the web app."""
    text = text.upper()
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=MONO)
    draw.rectangle([x, y, x + (r - l) + 20, y + (b - t) + 16], fill=bg)
    draw.text((x + 10 - l, y + 8 - t), text, font=MONO, fill=PAPER)


def before_after(before: Image.Image, after: Image.Image, caption: str, detail: str, width: int = 1600) -> Image.Image:
    """Side-by-side panel: thick black frame, BEFORE / AFTER tags, caption bar underneath."""
    half = width // 2
    scale = half / before.width
    h = round(before.height * scale)
    rule, bar = 6, 92
    canvas = Image.new("RGB", (width + rule * 3, h + rule * 2 + bar), INK)
    canvas.paste(before.resize((half, h), Image.LANCZOS), (rule, rule))
    canvas.paste(after.resize((half, h), Image.LANCZOS), (half + rule * 2, rule))
    d = ImageDraw.Draw(canvas)
    tag(d, (rule + 16, rule + 16), "Before", INK)
    tag(d, (half + rule * 2 + 16, rule + 16), "After", SIGNAL)
    d.rectangle([rule, h + rule * 2, width + rule * 2 - 1, h + rule * 2 + bar - rule], fill=PAPER)
    d.text((rule + 20, h + rule * 2 + 14), caption, font=HEAD, fill=INK)
    d.text((rule + 20, h + rule * 2 + 54), detail.upper(), font=MONO, fill=INK)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="docs/images")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import torch

    from src import pipeline
    from src.runtime.model_manager import ModelManager
    from src.segmentation.masks import draw_overlay

    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    manager = ModelManager(cfg)
    manager.load_all(warmup=False)
    results: dict = {"gpu": torch.cuda.get_device_name(0), "edits": []}

    print("Generating the example room...")
    t = time.perf_counter()
    with manager.lock:
        manager.activate("edit")
        room = manager.editor.generate(ROOM_PROMPT, width=1344, height=896, seed=args.seed)
    results["generate_s"] = round(time.perf_counter() - t, 1)
    room.save(out / "room.jpg", quality=90)

    # Step 2 of the app: find every object the edits will mention.
    terms = [term for term, _ in EDITS]
    instances, seg_t = pipeline.segment(manager, room, terms)
    results["segment_s"] = round(seg_t["segment"], 2)
    results["found"] = [i.label for i in instances]
    overlay = draw_overlay(room, [i.mask for i in instances], [i.label for i in instances], max_side=1600)
    overlay.convert("RGB").save(out / "find-objects.jpg", quality=88)
    print("Found:", results["found"])

    for n, (term, instruction) in enumerate(EDITS, start=1):
        chosen = [i for i in instances if i.label == term]
        if not chosen:
            print(f"Skipping '{term}': not found in the generated room.")
            continue
        r = pipeline.edit(manager, room, chosen, instruction, quality="final", seed=args.seed)
        name = f"edit-{n}-{term.replace(' ', '-')}.jpg"
        panel = before_after(
            room, r.image,
            caption=f"“{term}” → {instruction}",
            detail=f"Final render · {r.timings['edit']:.0f} s · peak {r.vram.get('peak_allocated')} GB VRAM · RTX 5080",
        )
        panel.save(out / name, quality=88)
        results["edits"].append({"term": term, "instruction": instruction, "file": name, "edit_s": round(r.timings["edit"], 1),
                                 "peak_vram_gb": r.vram.get("peak_allocated")})
        print(f"{name}: {r.timings['edit']:.1f}s")

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
