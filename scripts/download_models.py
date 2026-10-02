"""
Model downloader
================

What this script does
---------------------
Fetches everything the editing pipeline needs, once::

    uv run python scripts/download_models.py            # everything
    uv run python scripts/download_models.py --only acc # just the 4-step adapter

* ``Qwen/Qwen-Image-2.1``: full-precision weights, about 31 GB. Needed once
  so ``scripts/quantise_models.py`` can shrink them; the VAE, processor and
  scheduler are also loaded from here at run time.
* ``alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs``: the 4-step adapter
  (optimisation step 1), about 350 MB, plus the two helper Python files it
  needs (``qwenimage21_pdd.py`` and ``lora_utils_pdd.py``). These go into
  ``models/fun-acc`` so the editor can import them.
* ``facebook/sam3``: about 3.5 GB. Gated: accept Meta's terms on the model
  page and run ``uv run hf auth login`` first.
* ``IDEA-Research/grounding-dino-base`` + ``facebook/sam-vit-huge``: about
  3 GB, open (Apache 2.0). The stand-in segmenter used while SAM 3 access is
  pending (``segmentation.backend: grounded_sam``).

By default the script fetches whichever segmenter ``config.yaml`` selects.

Large files go into the Hugging Face cache. Set ``paths.hf_cache_dir`` in
``config.yaml`` to put it on a drive with room to spare.

Licences: Qwen-Image-2.1 and the adapter are under the Qwen Research Licence
(non-commercial use only); SAM 3 is under Meta's SAM licence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, repo_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["qwen", "acc", "sam3", "grounded"], action="append", help="download only these (repeatable)")
    args = parser.parse_args()

    cfg = load_config()  # Also applies paths.hf_cache_dir before huggingface_hub is imported.
    # By default fetch whichever segmentation backend config.yaml selects.
    seg_default = "grounded" if cfg["segmentation"].get("backend") == "grounded_sam" else "sam3"
    wanted = set(args.only or ["qwen", "acc", seg_default])
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    if "acc" in wanted:
        acc = cfg["editor"]["acceleration"]
        dest = repo_path(acc["local_dir"])
        print(f"4-step adapter -> {dest}")
        snapshot_download(acc["repo_id"], local_dir=dest, allow_patterns=["*.py", "*.json", "*.safetensors", "LICENSE*"])

    failed = []
    if "sam3" in wanted:
        model_id = cfg["segmentation"]["model_id"]
        print(f"SAM 3 ({model_id}) -> Hugging Face cache")
        try:
            snapshot_download(model_id)
        except GatedRepoError:
            failed.append(
                f"SAM 3: access denied. Request access at https://huggingface.co/{model_id} (Meta must approve it), "
                "then log in on this PC with `uv run hf auth login`."
            )

    if "grounded" in wanted:
        g = cfg["segmentation"].get("grounded_sam") or {}
        for model_id in (g.get("detector_id", "IDEA-Research/grounding-dino-base"), g.get("sam_id", "facebook/sam-vit-huge")):
            print(f"{model_id} -> Hugging Face cache")
            snapshot_download(model_id)

    if "qwen" in wanted:
        model_id = cfg["editor"]["base_model_id"]
        print(f"{model_id} (about 31 GB) -> Hugging Face cache")
        snapshot_download(model_id)

    if failed:
        print("\nINCOMPLETE. These downloads failed:")
        for msg in failed:
            print(f"  - {msg}")
        sys.exit(1)
    print("Done. Next: uv run python scripts/quantise_models.py")


if __name__ == "__main__":
    main()
