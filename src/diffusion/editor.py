"""
Qwen-Image-2.1 region editor
============================

What this file does
-------------------
Wraps the diffusers ``QwenImage21Pipeline`` so the rest of the app can call::

    editor.edit(crop, mask, "a green velvet sofa", quality="preview")

and get back an edited crop. Cropping, pasting back and segmentation live
elsewhere (``src/segmentation/``); this file only talks to the image model.

How it fits a 16 GB RTX 5080
----------------------------
At full precision Qwen-Image-2.1 is about 31 GB (7.1B generator, 8B Qwen3-VL
text encoder, 0.34B VAE), so this file applies four of the optimisation steps:

* Step 4, shrink once: it loads the 8-bit (FP8) generator and 4-bit (NF4)
  text encoder written by ``scripts/quantise_models.py``. If those are
  missing it shrinks them while loading, which is slower and needs more
  system RAM, and logs a warning.
* Step 5, text encoder offload: the text encoder lives in system RAM and is
  only moved to the GPU while it encodes the prompt and photo. It is moved
  back off as soon as the generator starts denoising, so the two never sit on
  the GPU together. The generator and VAE stay resident.
* Step 1, 4-step adapter: Alibaba PAI's "Fun-Acc" adapter (Parallel Decoding
  Distillation) gives 4-step previews. It is not a normal LoRA: it brings its
  own scheduler and per-step callback (``qwenimage21_pdd.py``, downloaded
  next to the weights by ``scripts/download_models.py``). Final renders
  switch the adapter off with ``pdd_teacher_mode`` and run the full model for
  ``editor.final_steps`` steps.
* Step 2 is honoured by the caller: ``edit`` expects an already-cropped
  image at the model's working size.

What testing on the RTX 5080 established
----------------------------------------
* The region to edit is painted translucent red on the crop ("annotated"
  mask mode). Sending a separate black-and-white mask image instead made the
  model output transparent areas with junk colours, so that mode is only
  kept as an option. ``paste_back`` still guarantees nothing outside the
  mask changes.
* The 4-step adapter works on top of the FP8 weights. Its timestep hook has
  to be paused for full-model renders (``_adapter_hooks_paused``).
* The generator must leave the GPU while the text encoder runs; together
  they overflow 16 GB (see ``_install_text_encoder_offload``).

Licence: Qwen-Image-2.1 and the Fun-Acc adapter are under the Qwen Research
Licence (non-commercial use only).
"""

from __future__ import annotations

import importlib
import logging
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image

from src.config import repo_path
from src.segmentation.masks import annotate_region

log = logging.getLogger(__name__)


@contextmanager
def _adapter_hooks_paused(transformer):
    """
    Temporarily remove the 4-step adapter's forward hooks from the generator.

    ``load_pdd_lora`` registers a hook (``native_time``) that overwrites the
    timestep on every call with the adapter's own 4-step schedule.
    ``pdd_teacher_mode`` switches off the adapter's layers but leaves that
    hook running, so a 20-30 step final render was told it was at the same
    noise level on every step and came out as noise. The hooks are re-added
    under their original keys and in their original order afterwards.
    """
    paused = []
    for store in (transformer._forward_pre_hooks, transformer._forward_hooks):
        for key, fn in list(store.items()):
            if getattr(fn, "__module__", "") == "qwenimage21_pdd":
                paused.append((store, key, fn))
                del store[key]
    try:
        yield
    finally:
        for store, key, fn in paused:
            store[key] = fn


def _flatten(image: Image.Image, background: Image.Image) -> Image.Image:
    """
    Turn the model's RGBA output into a plain RGB image.

    Qwen-Image-2.1 can output transparency. Measured: with a separate
    black-and-white mask image as input it made about 28% of a rug edit
    transparent, and the colour data behind those pixels is junk (it showed
    up as purple blotches). Any transparent pixels are therefore filled from
    ``background`` (the input crop for edits), never left to chance.
    """
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        base = background.convert("RGBA").resize(image.size, Image.LANCZOS)
        image = Image.alpha_composite(base, image)
    return image.convert("RGB")


def _find_one(root: Path, pattern: str) -> Path | None:
    """First file under ``root`` matching ``pattern`` (sorted, so the choice is stable), or None."""
    hits = sorted(root.rglob(pattern)) if root.exists() else []
    return hits[0] if hits else None


class RoomEditor:
    """Qwen-Image-2.1 wrapper: loads the quantised weights, manages offload and runs edits."""

    def __init__(self, cfg: dict):
        e = cfg["editor"]
        self.cfg = e
        self.device = torch.device(e.get("device", "cuda"))
        self.dtype = torch.bfloat16
        self.pipe = None
        self._base_scheduler = None
        self._pdd = None  # holds the adapter's scheduler, sigmas and helpers when loaded

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        """Build the pipeline from the pre-shrunk components, apply offload and attach the 4-step adapter if present."""
        from diffusers import QwenImage21Pipeline

        transformer, text_encoder = self._load_quantised_components()
        pipe = QwenImage21Pipeline.from_pretrained(
            self.cfg["base_model_id"],
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=self.dtype,
        )

        # The generator and VAE stay on the GPU for the life of the process.
        pipe.transformer.to(self.device)
        pipe.vae.to(self.device)
        if self.cfg.get("offload_text_encoder", True):
            self._install_text_encoder_offload(pipe)
        else:
            pipe.text_encoder.to(self.device)
        self._pin_execution_device(pipe)

        self.pipe = pipe
        self._base_scheduler = pipe.scheduler
        if self.cfg.get("acceleration", {}).get("enabled", True):
            self._load_acceleration_adapter()

    def _load_quantised_components(self):
        """Load the pre-shrunk generator and text encoder (step 4), or shrink them on the fly."""
        from diffusers import QwenImage21Transformer2DModel
        from transformers import Qwen3VLForConditionalGeneration

        qdir = repo_path(self.cfg["quantised_dir"])
        t_dir, te_dir = qdir / "transformer", qdir / "text_encoder"

        if t_dir.exists():
            # FP8 (torchao) weights are saved in PyTorch's own format and NF4 in
            # safetensors; diffusers picks whichever it finds.
            transformer = QwenImage21Transformer2DModel.from_pretrained(t_dir, torch_dtype=self.dtype)
        else:
            log.warning("No pre-shrunk generator in %s; shrinking while loading. Run scripts/quantise_models.py.", t_dir)
            transformer = QwenImage21Transformer2DModel.from_pretrained(
                self.cfg["base_model_id"],
                subfolder="transformer",
                torch_dtype=self.dtype,
                quantization_config=transformer_quant_config(self.cfg.get("transformer_quant", "fp8")),
                device_map="cuda",
            )

        if te_dir.exists():
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(te_dir, dtype=self.dtype)
        else:
            log.warning("No pre-shrunk text encoder in %s; shrinking while loading.", te_dir)
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                self.cfg["base_model_id"],
                subfolder="text_encoder",
                dtype=self.dtype,
                quantization_config=text_encoder_quant_config(self.cfg.get("text_encoder_quant", "nf4")),
                device_map="cuda",
            )
        return transformer, text_encoder

    def _install_text_encoder_offload(self, pipe) -> None:
        """
        Step 5: the text encoder only runs once per edit (the generator caches
        its output), so keep it in system RAM and swap it in and out with
        two small forward hooks.

        Measured on the RTX 5080: the encoder (about 6 GB) plus the resident
        FP8 generator (about 7.2 GB) plus the encoder's working memory goes
        just over 16 GB. Windows then spills into system RAM and encoding
        took about 40 s. With ``editor.swap_generator_during_encode`` (the
        default) the generator is parked in system RAM while the encoder
        runs, so the two are never on the GPU together.
        """
        te = pipe.text_encoder
        transformer = pipe.transformer
        te.to("cpu")
        device = self.device
        swap_generator = self.cfg.get("swap_generator_during_encode", True)
        # Track placement ourselves rather than asking the weights: an FP8
        # (torchao) weight can still report "cuda" after its data has moved
        # to the CPU, which once left the generator stranded in system RAM.
        on_gpu = {"te": False, "transformer": True}

        def _te_to_gpu(module, args, kwargs):
            if not on_gpu["te"]:
                if swap_generator and on_gpu["transformer"]:
                    transformer.to("cpu")
                    on_gpu["transformer"] = False
                    torch.cuda.empty_cache()
                module.to(device)
                on_gpu["te"] = True
            # Inputs may have been prepared on the CPU; move any tensors across.
            args = tuple(a.to(device) if torch.is_tensor(a) else a for a in args)
            kwargs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
            return args, kwargs

        def _te_off_gpu(module, args, kwargs):
            if on_gpu["te"]:
                te.to("cpu")
                on_gpu["te"] = False
                torch.cuda.empty_cache()
            if not on_gpu["transformer"]:
                transformer.to(device)
                on_gpu["transformer"] = True

        te.register_forward_pre_hook(_te_to_gpu, with_kwargs=True)
        pipe.transformer.register_forward_pre_hook(_te_off_gpu, with_kwargs=True)

    def _pin_execution_device(self, pipe) -> None:
        """
        diffusers works out its "execution device" from wherever the first
        component happens to sit. With the text encoder parked on the CPU
        that could be guessed wrong, so pin it to the GPU explicitly.
        """
        device = self.device
        base = type(pipe)
        pipe.__class__ = type(base.__name__, (base,), {"_execution_device": property(lambda self: device)})

    def _load_acceleration_adapter(self) -> None:
        """Step 1: attach the 4-step Fun-Acc adapter (PDD). Falls back to plain sampling if it is missing."""
        acc = self.cfg["acceleration"]
        root = repo_path(acc["local_dir"])
        code = _find_one(root, "qwenimage21_pdd.py")
        weights = _find_one(root, "*4Step*.safetensors") or _find_one(root, "*.safetensors")
        if code is None or weights is None:
            log.warning("4-step adapter not found in %s; previews will use %s plain steps.", root, self.cfg["preview_fallback_steps"])
            return

        # The adapter ships as loose Python files; import them from where they were downloaded.
        if str(code.parent) not in sys.path:
            sys.path.insert(0, str(code.parent))
        pdd = importlib.import_module("qwenimage21_pdd")
        lora_utils = importlib.import_module("lora_utils_pdd")

        config = pdd.load_pdd_lora(self.pipe.transformer, str(weights))
        self.pipe.transformer.to(self.device).eval()
        scheduler = pdd.QwenImage21PDDScheduler.from_config(self._base_scheduler.config)
        scheduler.register_to_config(**config)

        self._pdd = {
            "module": pdd,
            "teacher_mode": lora_utils.pdd_teacher_mode,
            "scheduler": scheduler,
            "sigmas": torch.tensor(config["pdd_sigmas"], dtype=torch.float32),
            "block_size": config["pdd_block_size"],
            "steps": config["pdd_num_steps"] // config["pdd_block_size"],
        }
        log.info("4-step adapter loaded (%s steps).", self._pdd["steps"])

    # ------------------------------------------------------------------ editing
    def build_inputs(self, crop: Image.Image, mask: Image.Image, instruction: str) -> tuple[list[Image.Image], str]:
        """Turn crop + mask + instruction into the images and prompt the model expects (see ``mask_mode`` in config.yaml)."""
        mode = self.cfg.get("mask_mode", "separate")
        template = self.cfg["prompt_templates"][mode]
        prompt = template.format(instruction=instruction.strip().rstrip("."))
        if mode == "annotated":
            return [annotate_region(crop, mask)], prompt
        return [crop.convert("RGB"), mask.convert("RGB")], prompt

    @torch.inference_mode()
    def edit(
        self,
        crop: Image.Image,
        mask: Image.Image,
        instruction: str,
        quality: str = "preview",
        seed: int = 42,
    ) -> Image.Image:
        """
        Edit a crop that is already at the model's working size.

        ``quality="preview"`` uses the 4-step adapter (or a few plain steps if
        it is unavailable); ``quality="final"`` runs the full model.

        Qwen-Image-2.1's autoencoder works in RGBA (it supports transparency),
        so outputs are flattened onto the input crop: see ``_flatten``.
        """
        if self.pipe is None:
            self.load()
        return _flatten(self._run_edit(crop, mask, instruction, quality, seed), crop)

    def _run_edit(self, crop: Image.Image, mask: Image.Image, instruction: str, quality: str, seed: int) -> Image.Image:
        """Run the diffusion model once. Previews use the 4-step adapter when present; finals use the full model."""
        images, prompt = self.build_inputs(crop, mask, instruction)
        width, height = crop.size
        common = dict(
            prompt=prompt,
            image=images,
            width=width,
            height=height,
            output_resolution=self.cfg.get("reference_resolution", 1024),
            true_cfg_scale=self.cfg.get("true_cfg_scale", 1.0),
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )

        if quality == "preview" and self._pdd is not None:
            p = self._pdd
            self.pipe.scheduler = p["scheduler"]
            try:
                return self.pipe(
                    **common,
                    num_inference_steps=p["steps"],
                    # Alibaba's reference script turns the prefix KV cache off for the adapter.
                    use_kv_cache=self.cfg.get("acceleration", {}).get("use_kv_cache", False),
                    callback_on_step_end=p["module"].pdd_step_callback(self.pipe.transformer, p["sigmas"], p["block_size"]),
                ).images[0]
            finally:
                self.pipe.scheduler = self._base_scheduler

        steps = self.cfg["final_steps"] if quality == "final" else self.cfg["preview_fallback_steps"]
        # The prefix KV cache makes each step much faster but costs about 2.3 GB
        # of VRAM (measured peak 15.2 GB instead of 12.9 GB on the RTX 5080).
        common["use_kv_cache"] = self.cfg.get("final_use_kv_cache", False)
        self.pipe.scheduler = self._base_scheduler
        if self._pdd is not None:
            # Switch the adapter off so the original model runs. teacher_mode
            # alone is not enough: see _adapter_hooks_paused.
            with self._pdd["teacher_mode"](self.pipe.transformer), _adapter_hooks_paused(self.pipe.transformer):
                return self.pipe(**common, num_inference_steps=steps).images[0]
        return self.pipe(**common, num_inference_steps=steps).images[0]

    @torch.inference_mode()
    def generate(self, prompt: str, width: int = 1344, height: int = 896, steps: int | None = None, seed: int = 42) -> Image.Image:
        """
        Text-to-image with the full model (no input photo). Used to make
        example rooms, e.g. for the README via scripts/make_docs_images.py.
        """
        if self.pipe is None:
            self.load()
        self.pipe.scheduler = self._base_scheduler
        kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps or self.cfg["final_steps"],
            true_cfg_scale=self.cfg.get("true_cfg_scale", 1.0),
            use_kv_cache=False,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        white = Image.new("RGB", (width, height), (255, 255, 255))
        if self._pdd is not None:
            with self._pdd["teacher_mode"](self.pipe.transformer), _adapter_hooks_paused(self.pipe.transformer):
                return _flatten(self.pipe(**kwargs).images[0], white)
        return _flatten(self.pipe(**kwargs).images[0], white)

    def warmup(self) -> None:
        """Run one tiny edit so CUDA kernels are compiled before the first real request."""
        crop = Image.new("RGB", (256, 256), (180, 170, 160))
        mask = Image.new("L", (256, 256), 0)
        mask.paste(255, (64, 64, 192, 192))
        self.edit(crop, mask, "a small wooden stool", quality="preview")


# ---------------------------------------------------------------------- quantisation configs
def transformer_quant_config(kind: str):
    """Quantisation settings for the 7.1B generator. Shared with scripts/quantise_models.py."""
    if kind == "none":
        return None
    if kind == "fp8":
        from diffusers import TorchAoConfig
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

        # FP8 weights and activations use the RTX 50 series' FP8 tensor cores.
        # proj_out is left alone because the 4-step adapter replaces it.
        return TorchAoConfig(Float8DynamicActivationFloat8WeightConfig(), modules_to_not_convert=["proj_out"])
    if kind == "nf4":
        from diffusers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["proj_out"],
        )
    raise ValueError(f"Unknown transformer_quant: {kind!r}")


def text_encoder_quant_config(kind: str):
    """Quantisation settings for the Qwen3-VL-8B text encoder."""
    if kind == "none":
        return None
    if kind == "nf4":
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # Keep the vision tower at full precision: it reads the room photo,
            # and it is small (about 0.6B parameters).
            llm_int8_skip_modules=["visual", "lm_head"],
        )
    raise ValueError(f"Unknown text_encoder_quant: {kind!r}")
