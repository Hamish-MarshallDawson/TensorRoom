"""
TensorRoom web UI (Streamlit)
=============================

Note: the main front end is now the mobile web app in ``web/``, served by
``server.py`` at "/". This Streamlit page is kept as a simple desktop
fallback for testing the worker.

What this file does
-------------------
The page the user opens in a browser:

1. Upload or take a photo of a room. On a phone, the upload button offers the
   camera directly and works over plain HTTP on the local network.
2. Say what to change ("sofa, coffee table") and what it should become.
3. "Find objects" asks the GPU worker to segment the photo, then shows each
   match numbered and colour-coded, with a tick box per object.
4. "Quick preview" runs the 4-step edit; "Final render" runs the full model.
5. "Keep editing this result" makes the result the new starting photo.

Why it does no AI work itself (optimisation step 6)
---------------------------------------------------
Streamlit re-runs this whole script on every click. All models live in the
long-running worker (``server.py``), so start that first::

    uv run uvicorn server:app --host 127.0.0.1 --port 8765
    uv run streamlit run app.py

To open the page from a phone on the same Wi-Fi, add
``--server.address 0.0.0.0`` to the Streamlit command.

The sidebar logo and title are kept from Jake's original UI. His SDXL
generator (``src/diffusion/generator.py``) is untouched and can still be run
on its own.
"""

from __future__ import annotations

import base64
import io

import requests
import streamlit as st
from PIL import Image

from src.config import load_config

CFG = load_config()
# "0.0.0.0" means "listen everywhere"; this page still connects locally.
_host = CFG["server"]["host"]
WORKER = f"http://{'127.0.0.1' if _host == '0.0.0.0' else _host}:{CFG['server']['port']}"

st.set_page_config(page_title="TensorRoom", layout="wide")


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _worker_status() -> dict | None:
    try:
        return requests.get(f"{WORKER}/health", timeout=2).json()
    except requests.RequestException:
        return None


for key, default in {"image_id": None, "instances": [], "overlay": None, "result": None, "source": None}.items():
    st.session_state.setdefault(key, default)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    left, right = st.columns([1, 6])
    with left:
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        st.image("public/logo.png", width=100)
    with right:
        st.title("TensorRoom")

    status = _worker_status()
    if status is None:
        st.error(f"GPU worker not reachable at {WORKER}. Start it with `uv run uvicorn server:app --port {CFG['server']['port']}`.")
    elif not status.get("ready"):
        st.info("GPU worker is loading models...")
    else:
        vram = status.get("vram_gb", {})
        st.success("GPU worker ready")
        if vram:
            st.caption(f"VRAM in use: {vram.get('device_used')} / {vram.get('device_total')} GB")

    seed = st.number_input("Seed", value=42, step=1)

# ------------------------------------------------------------------ step 1: photo
st.header("1. Your room")
upload = st.file_uploader("Upload or take a photo", type=["jpg", "jpeg", "png", "webp"])
if upload is not None and upload.file_id != st.session_state.source:
    # A new photo resets everything that belonged to the previous one.
    st.session_state.update(source=upload.file_id, image_id=None, instances=[], overlay=None, result=None)

# ------------------------------------------------------------------ step 2: what to change
st.header("2. What should change?")
c1, c2 = st.columns(2)
terms = c1.text_input("Objects to change", placeholder="sofa, coffee table")
instruction = c2.text_input("Change them to", placeholder="a green velvet mid-century sofa")

if st.button("Find objects", disabled=not terms or (upload is None and st.session_state.image_id is None)):
    data = {"terms": terms}
    files = None
    if st.session_state.image_id:
        data["image_id"] = st.session_state.image_id
    else:
        files = {"image": (upload.name, upload.getvalue(), upload.type)}
    with st.spinner("Finding objects..."):
        r = requests.post(f"{WORKER}/segment", data=data, files=files, timeout=300)
    if r.ok:
        body = r.json()
        st.session_state.update(image_id=body["image_id"], instances=body["instances"], overlay=body["overlay_png"])
        if not body["instances"]:
            st.warning("Nothing matched. Try another word, e.g. 'couch' instead of 'settee'.")
    else:
        st.error(r.text)

# ------------------------------------------------------------------ step 3: choose and edit
if st.session_state.overlay:
    st.header("3. Pick the objects and edit")
    left, right = st.columns([3, 2])
    left.image(_decode(st.session_state.overlay), use_container_width=True)
    chosen = [
        inst["id"]
        for inst in st.session_state.instances
        if right.checkbox(f"{inst['id']}: {inst['label']} ({inst['score']:.2f})", value=True, key=f"inst_{st.session_state.image_id}_{inst['id']}")
    ]

    b1, b2 = right.columns(2)
    quality = None
    if b1.button("Quick preview (4 steps)", disabled=not (chosen and instruction)):
        quality = "preview"
    if b2.button("Final render", disabled=not (chosen and instruction)):
        quality = "final"

    if quality:
        payload = {
            "image_id": st.session_state.image_id,
            "instance_ids": chosen,
            "instruction": instruction,
            "quality": quality,
            "seed": int(seed),
        }
        with st.spinner("Editing..." if quality == "preview" else "Rendering at full quality..."):
            r = requests.post(f"{WORKER}/edit", json=payload, timeout=900)
        if r.ok:
            st.session_state.result = r.json()
        else:
            st.error(r.text)

# ------------------------------------------------------------------ step 4: result
result = st.session_state.result
if result:
    st.header("4. Result")
    before, after = st.columns(2)
    before.image(_decode(st.session_state.overlay), caption="Before (objects highlighted)", use_container_width=True)
    after.image(_decode(result["image_png"]), caption="After", use_container_width=True)
    t = result["timings"]
    v = result.get("vram_gb", {})
    st.caption(
        f"Edit {t.get('edit', 0):.1f}s at {result['model_size'][0]}x{result['model_size'][1]} | "
        f"peak VRAM (this process) {v.get('peak_allocated', '?')} GB"
    )
    if st.button("Keep editing this result"):
        st.session_state.update(image_id=result["image_id"], instances=[], overlay=None, result=None)
        st.rerun()
    st.markdown(f"[Download full resolution]({WORKER}/image/{result['image_id']})")
