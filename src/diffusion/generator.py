from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from diffusers import DiffusionPipeline
import torch

def load_config() -> dict:
    with open(Path(__file__).parent.parent.parent / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_model() -> DiffusionPipeline:
    cfg = load_config()
    model = DiffusionPipeline.from_pretrained(cfg["model_id"], torch_dtype=cfg["dtype"])
    model = model.to(cfg["device"])
    return model

def generate_image(model):

    cfg = load_config()

    generator = None
    if cfg["seed"] is not None:
        generator = torch.Generator(device=cfg["device"]).manual_seed(cfg["seed"])

    image = model(
        prompt=cfg["prompt"],
        num_inference_steps=cfg["steps"],
        guidance_scale=0.0,
        generator=generator,
    ).images[0]

    image.save(cfg["output_dir"])

def main():
    generate_image(load_model())

if __name__ == "__main__":
    main()
