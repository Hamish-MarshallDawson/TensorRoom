"""
Feasibility benchmark (the go/no-go test)
=========================================

What this script does
---------------------
Runs the real pipeline (SAM 3, crop, Qwen-Image-2.1 edit, paste back) on a
folder of room photos and records, for each photo and setting:

* time for segmentation, the edit itself and paste-back;
* peak VRAM used by this process, and VRAM used on the whole card (which
  includes the Windows desktop);
* the crop size the model actually worked at.

Example::

    uv run python scripts/benchmark.py --images data/test_rooms --terms sofa \
        --instruction "a green velvet mid-century sofa" --max-side 768 1024 --quality preview final

Results go to ``data/benchmarks/<timestamp>/`` (``results.csv`` plus every
edited image so quality can be judged by eye). At the end it prints pass or
fail against ``benchmark.max_peak_vram_gb`` and
``benchmark.max_preview_seconds`` from ``config.yaml``.

What it is for
--------------
Several things in this design are informed estimates until measured on the
RTX 5080: FP8 + 4-step adapter compatibility, the mask prompt wording,
latency, and VRAM headroom. Run this before building further. If it fails,
try ``--max-side 768``, ``editor.transformer_quant: nf4``, or
``editor.mask_mode: annotated``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import branding  # noqa: E402
from src.config import load_config, repo_path  # noqa: E402


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="folder of room photos (jpg/png/webp)")
    parser.add_argument("--terms", required=True, help="comma-separated objects to change, e.g. 'sofa'")
    parser.add_argument("--instruction", required=True, help="what the objects should become")
    parser.add_argument("--max-side", type=int, nargs="+", default=[cfg["editor"]["max_side"]])
    parser.add_argument("--quality", nargs="+", choices=["preview", "final"], default=["preview"])
    parser.add_argument("--repeats", type=int, default=2, help="runs per setting; the first is often slower")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from PIL import Image, ImageOps

    from src import pipeline
    from src.runtime.model_manager import ModelManager

    photos = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not photos:
        sys.exit(f"No photos found in {args.images}")

    out_dir = repo_path(cfg["paths"]["benchmarks_dir"]) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    manager = ModelManager(cfg)
    load_times = manager.load_all(warmup=True)
    print(f"Loaded: {load_times}  VRAM: {manager.vram_report()}")

    rows = []
    # Segment each photo once, then time the edit for every (max_side, quality, repeat) combination.
    for photo_path in photos:
        photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
        instances, seg_t = pipeline.segment(manager, photo, args.terms.split(","))
        if not instances:
            print(f"{photo_path.name}: nothing matched '{args.terms}', skipped")
            continue
        for max_side in args.max_side:
            cfg["editor"]["max_side"] = max_side  # The pipeline reads this per call
            for quality in args.quality:
                for rep in range(args.repeats):
                    res = pipeline.edit(manager, photo, instances, args.instruction, quality=quality, seed=args.seed)
                    name = f"{photo_path.stem}_{max_side}_{quality}_{rep}.png"
                    branding.save_png(branding.watermark(res.image), out_dir / name)
                    row = {
                        "photo": photo_path.name,
                        "megapixels": round(photo.width * photo.height / 1e6, 1),
                        "objects": len(instances),
                        "max_side": max_side,
                        "model_size": f"{res.crop_info.model_size[0]}x{res.crop_info.model_size[1]}",
                        "quality": quality,
                        "repeat": rep,
                        "segment_s": round(seg_t["segment"], 2),
                        "edit_s": round(res.timings["edit"], 2),
                        "paste_back_s": round(res.timings["paste_back"], 2),
                        "peak_vram_gb": res.vram.get("peak_allocated"),
                        "device_used_gb": res.vram.get("device_used"),
                        "output": name,
                    }
                    rows.append(row)
                    print(row)

    if not rows:
        sys.exit("Nothing was edited.")
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Go/no-go, ignoring each setting's first (warm-up) run.
    b = cfg["benchmark"]
    steady = [r for r in rows if r["repeat"] > 0] or rows
    peak = max(r["peak_vram_gb"] or 0 for r in steady)
    previews = [r["edit_s"] for r in steady if r["quality"] == "preview" and r["max_side"] == max(args.max_side)]
    print(f"\nResults: {out_dir / 'results.csv'}")
    print(f"Peak VRAM (process): {peak:.2f} GB  -> {'PASS' if peak <= b['max_peak_vram_gb'] else 'FAIL'} (limit {b['max_peak_vram_gb']} GB)")
    if previews:
        worst = max(previews)
        print(f"Slowest preview edit: {worst:.1f}s -> {'PASS' if worst <= b['max_preview_seconds'] else 'FAIL'} (limit {b['max_preview_seconds']}s)")
    print("Now check the saved images by eye: edges, lighting, and whether the object looks right.")


if __name__ == "__main__":
    main()
