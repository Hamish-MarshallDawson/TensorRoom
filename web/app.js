/*
  TensorRoom web app logic
  ========================
  Plain JavaScript with no build step and no framework, so it loads fast on
  any phone and anyone on the team can edit it.

  Flow (each step is a panel in index.html):
    01 Capture  - camera or photo library. The photo is shrunk on the phone
                  to at most 3072 px before upload: a 12-48 MP original
                  would take a long time over Wi-Fi, and the server crops
                  to 1024 px for the model anyway.
    02 Target   - type objects or tap suggestion chips, then "Find objects".
    03 Select   - every match is tinted on the photo. Tap an object (or its
                  row in the list) to toggle it, then describe the change
                  and choose Preview (4 steps) or Final render.
    04 Result   - drag to compare before and after, save or share, or keep
                  editing the result.

  The main action always sits in the dock at the bottom of the screen,
  within reach of a thumb.
*/
(() => {
  "use strict";

  const UPLOAD_MAX_SIDE = 3072;   // Longest side sent to the server
  const UPLOAD_QUALITY = 0.92;
  const STEPS = { 1: "Capture", 2: "Target", 3: "Select", 4: "Result" };

  const $ = (id) => document.getElementById(id);
  const el = {
    stepIndex: $("stepIndex"), stepName: $("stepName"), stepCount: $("stepCount"),
    linkStatus: $("linkStatus"), linkText: $("linkText"),
    sessionId: $("sessionId"), vram: $("vram"), quant: $("quant"),
    fileCamera: $("fileCamera"), fileLibrary: $("fileLibrary"),
    captureButtons: $("captureButtons"), photoFrame: $("photoFrame"), photoPreview: $("photoPreview"),
    photoMeta: $("photoMeta"), retake: $("retake"),
    secCapture: $("secCapture"), secTarget: $("secTarget"), secSelect: $("secSelect"), secResult: $("secResult"),
    terms: $("terms"), suggestions: $("suggestions"),
    stage: $("stage"), stagePhoto: $("stagePhoto"), stageCanvas: $("stageCanvas"),
    selectCount: $("selectCount"), instanceList: $("instanceList"), instruction: $("instruction"),
    compare: $("compare"), afterImg: $("afterImg"), beforeImg: $("beforeImg"), compareRange: $("compareRange"),
    resultMeta: $("resultMeta"), startOver: $("startOver"),
    alert: $("alert"), busy: $("busy"), busyText: $("busyText"), busyClock: $("busyClock"),
    btnPrimary: $("btnPrimary"), btnSecondary: $("btnSecondary"),
  };

  const state = {
    step: 1,
    photoBlob: null,      // Shrunk JPEG to upload (null once the server holds the photo)
    photoUrl: null,       // What the stage and "before" show
    imageId: null,        // Server-side id of the current photo
    serverSize: null,     // [width, height] of the server's copy (box coordinates use this)
    instances: [],        // From /segment, plus decoded masks
    selected: new Set(),
    result: null,
    busy: false,
  };

  // ---------------------------------------------------------------- helpers
  async function api(path, options = {}) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
      throw new Error(detail || `Request failed (${res.status})`);
    }
    return res.json();
  }

  function showError(message) {
    el.alert.textContent = message;
    el.alert.hidden = false;
    el.alert.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  function clearError() { el.alert.hidden = true; }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("This image format is not supported. Try a JPEG or PNG."));
      img.src = src;
    });
  }

  // Browsers apply the photo's EXIF rotation when drawing an <img>, so the
  // shrunk upload comes out the right way up on every phone.
  async function shrinkPhoto(file) {
    const url = URL.createObjectURL(file);
    try {
      const img = await loadImage(url);
      const scale = Math.min(1, UPLOAD_MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
      const w = Math.round(img.naturalWidth * scale);
      const h = Math.round(img.naturalHeight * scale);
      const canvas = document.createElement("canvas");
      canvas.width = w; canvas.height = h;
      canvas.getContext("2d").drawImage(img, 0, 0, w, h);
      const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", UPLOAD_QUALITY));
      if (!blob) throw new Error("Could not read this photo.");
      return { blob, width: w, height: h, original: [img.naturalWidth, img.naturalHeight] };
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function setBusy(text) {
    state.busy = Boolean(text);
    el.busy.hidden = !text;
    el.btnPrimary.disabled = state.busy || el.btnPrimary.dataset.ready !== "1";
    el.btnSecondary.disabled = state.busy;
    if (!text) { clearInterval(setBusy.timer); return; }
    el.busyText.textContent = text;
    const start = performance.now();
    clearInterval(setBusy.timer);
    setBusy.timer = setInterval(() => {
      const s = (performance.now() - start) / 1000;
      el.busyClock.textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
    }, 100);
  }

  function pad(n) { return String(n).padStart(2, "0"); }

  // ---------------------------------------------------------------- step + dock
  function goTo(step) {
    state.step = step;
    document.body.dataset.step = step;
    el.stepIndex.textContent = pad(step);
    el.stepName.textContent = STEPS[step];
    el.stepCount.textContent = `${pad(step)}/${pad(4)}`;
    el.secTarget.hidden = step < 2;
    el.secSelect.hidden = step < 3;
    el.secResult.hidden = step < 4;
    for (const sec of [el.secCapture, el.secTarget, el.secSelect, el.secResult]) {
      sec.classList.toggle("is-current", Number(sec.dataset.step) === step);
    }
    updateDock();
    const target = [null, el.secCapture, el.secTarget, el.secSelect, el.secResult][step];
    // Wait a moment so newly shown panels have a size before scrolling to them.
    if (step > 1) setTimeout(() => scrollToPanel(target), 50);
  }

  // Smooth scroll where supported; jump instead if the browser didn't move
  // (some embedded and low-power-mode browsers skip smooth scrolling).
  function scrollToPanel(panel) {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    panel.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    const start = window.scrollY;
    setTimeout(() => {
      if (Math.abs(window.scrollY - start) < 2 && Math.abs(panel.getBoundingClientRect().top) > 120) {
        panel.scrollIntoView({ block: "start" });
      }
    }, 450);
  }

  function setPrimary(label, ready, handler) {
    el.btnPrimary.textContent = label;
    el.btnPrimary.dataset.ready = ready ? "1" : "0";
    el.btnPrimary.disabled = state.busy || !ready;
    el.btnPrimary.onclick = handler;
  }
  function setSecondary(label, handler) {
    el.btnSecondary.hidden = !label;
    if (label) { el.btnSecondary.textContent = label; el.btnSecondary.onclick = handler; }
  }

  function updateDock() {
    const termsReady = el.terms.value.trim().length > 0;
    const editReady = state.selected.size > 0 && el.instruction.value.trim().length > 0;
    switch (state.step) {
      case 1:
        setSecondary(null);
        setPrimary("Take or choose a photo", false, null);
        break;
      case 2:
        setSecondary(null);
        setPrimary("Find objects", termsReady, findObjects);
        break;
      case 3:
        setSecondary("Final render", () => runEdit("final"));
        el.btnSecondary.disabled = state.busy || !editReady;
        setPrimary("Preview · 4 steps", editReady, () => runEdit("preview"));
        // While still choosing, a changed object list means searching again.
        if (termsReady && state.instances.length === 0) setPrimary("Find objects", true, findObjects);
        break;
      case 4:
        setSecondary("Save", saveResult);
        setPrimary("Keep editing", true, keepEditing);
        break;
    }
  }

  // ---------------------------------------------------------------- 01 capture
  async function onPhotoChosen(event) {
    const file = event.target.files && event.target.files[0];
    event.target.value = ""; // Allow choosing the same file again
    if (!file) return;
    clearError();
    setBusy("Preparing photo");
    try {
      const shrunk = await shrinkPhoto(file);
      resetFrom(1);
      state.photoBlob = shrunk.blob;
      setPhotoUrl(URL.createObjectURL(shrunk.blob));
      el.photoPreview.src = state.photoUrl;
      el.photoMeta.textContent = `${shrunk.original[0]}×${shrunk.original[1]} → ${shrunk.width}×${shrunk.height}`;
      el.photoFrame.hidden = false;
      el.captureButtons.hidden = true;
      goTo(2);
    } catch (err) {
      showError(err.message);
    } finally {
      setBusy(null);
      updateDock();
    }
  }

  function setPhotoUrl(url) {
    if (state.photoUrl && state.photoUrl.startsWith("blob:")) URL.revokeObjectURL(state.photoUrl);
    state.photoUrl = url;
  }

  function resetFrom(step) {
    if (step <= 1) { state.imageId = null; state.serverSize = null; el.sessionId.textContent = "—"; }
    if (step <= 2) { state.instances = []; state.selected.clear(); el.instanceList.innerHTML = ""; }
    if (step <= 3) { state.result = null; }
  }

  // ---------------------------------------------------------------- 02 target
  function renderSuggestions(words) {
    el.suggestions.innerHTML = "";
    for (const word of words) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = word;
      chip.setAttribute("aria-pressed", "false");
      chip.addEventListener("click", () => toggleTerm(word));
      el.suggestions.appendChild(chip);
    }
    syncChips();
  }

  function currentTerms() {
    return el.terms.value.split(",").map((t) => t.trim().toLowerCase()).filter(Boolean);
  }
  function toggleTerm(word) {
    const terms = currentTerms();
    const i = terms.indexOf(word);
    if (i >= 0) terms.splice(i, 1); else terms.push(word);
    el.terms.value = terms.join(", ");
    onTermsChanged();
  }
  function syncChips() {
    const terms = new Set(currentTerms());
    for (const chip of el.suggestions.children) chip.setAttribute("aria-pressed", String(terms.has(chip.textContent)));
  }
  function onTermsChanged() {
    syncChips();
    if (state.step === 3 && state.instances.length) {
      // New words mean a new search; keep the instruction the user typed.
      resetFrom(2);
      drawStage();
      el.selectCount.textContent = "Search again to update the matches";
    }
    updateDock();
  }

  async function findObjects() {
    clearError();
    el.terms.blur();
    const form = new FormData();
    form.append("terms", currentTerms().join(","));
    if (state.imageId) form.append("image_id", state.imageId);
    else form.append("image", state.photoBlob, "room.jpg");

    setBusy("Finding objects");
    try {
      const body = await api("/segment", { method: "POST", body: form });
      state.imageId = body.image_id;
      state.serverSize = [body.width, body.height];
      state.photoBlob = null;
      el.sessionId.textContent = body.image_id;
      state.instances = await Promise.all(body.instances.map(decodeInstance));
      // Pre-select everything: usually the user wants all the sofas they named.
      state.selected = new Set(state.instances.map((i) => i.id));
      el.stagePhoto.src = state.photoUrl;
      // Let the photo decode before drawing, but never wait more than a second
      // (decode() can stall in background tabs).
      await Promise.race([el.stagePhoto.decode().catch(() => {}), new Promise((r) => setTimeout(r, 1000))]);
      renderInstanceList();
      drawStage();
      if (!state.instances.length) {
        showError(`Nothing matched "${currentTerms().join(", ")}". Try another word, e.g. "couch" instead of "settee".`);
        goTo(2);
      } else {
        goTo(3);
      }
    } catch (err) {
      showError(err.message);
    } finally {
      setBusy(null);
      updateDock();
    }
  }

  // ---------------------------------------------------------------- 03 select
  // Each mask arrives as a small transparent PNG. Keep its pixels for
  // tap hit-testing, and a tinted copy for drawing.
  async function decodeInstance(inst) {
    const img = await loadImage(`data:image/png;base64,${inst.mask_png}`);
    const c = document.createElement("canvas");
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    const alpha = ctx.getImageData(0, 0, c.width, c.height).data;
    return { ...inst, maskImg: img, maskW: c.width, maskH: c.height, alpha, tinted: {} };
  }

  function tinted(inst, colour) {
    if (!inst.tinted[colour]) {
      const c = document.createElement("canvas");
      c.width = inst.maskW; c.height = inst.maskH;
      const ctx = c.getContext("2d");
      ctx.drawImage(inst.maskImg, 0, 0);
      ctx.globalCompositeOperation = "source-in";
      ctx.fillStyle = colour;
      ctx.fillRect(0, 0, c.width, c.height);
      inst.tinted[colour] = c;
    }
    return inst.tinted[colour];
  }

  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function drawStage() {
    const canvas = el.stageCanvas;
    const rect = el.stagePhoto.getBoundingClientRect();
    if (!rect.width) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!state.instances.length || !state.serverSize) return;

    const signal = cssVar("--signal") || "#1f1fff";
    const sx = rect.width / state.serverSize[0];
    const sy = rect.height / state.serverSize[1];

    for (const inst of state.instances) {
      const on = state.selected.has(inst.id);
      ctx.globalAlpha = on ? 0.55 : 0.28;
      ctx.drawImage(tinted(inst, on ? signal : "#000"), 0, 0, rect.width, rect.height);
    }
    ctx.globalAlpha = 1;

    // Numbered tags in the site's style: mono caps in a solid block.
    ctx.font = "700 11px 'JetBrains Mono', ui-monospace, monospace";
    ctx.textBaseline = "middle";
    for (const inst of state.instances) {
      const on = state.selected.has(inst.id);
      const text = `${pad(inst.id + 1)} ${inst.label.toUpperCase()}`;
      const w = ctx.measureText(text).width + 12;
      const x = Math.min(Math.max(0, inst.box[0] * sx), rect.width - w);
      const y = Math.max(0, inst.box[1] * sy);
      ctx.fillStyle = on ? signal : "#000";
      ctx.fillRect(x, y, w, 20);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, x + 6, y + 10);
    }
  }

  function renderInstanceList() {
    el.instanceList.innerHTML = "";
    for (const inst of state.instances) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "instance";
      btn.dataset.id = inst.id;
      btn.innerHTML = `<span class="instance__box" aria-hidden="true"></span>
        <span class="instance__name"></span><span class="mono"></span>`;
      btn.children[1].textContent = `${pad(inst.id + 1)} ${inst.label}`;
      btn.children[2].textContent = `${Math.round(inst.score * 100)}%`;
      btn.addEventListener("click", () => toggleInstance(inst.id));
      li.appendChild(btn);
      el.instanceList.appendChild(li);
    }
    syncSelection();
  }

  function toggleInstance(id) {
    if (state.selected.has(id)) state.selected.delete(id); else state.selected.add(id);
    if (navigator.vibrate) navigator.vibrate(8);
    syncSelection();
    drawStage();
    updateDock();
  }

  function syncSelection() {
    for (const btn of el.instanceList.querySelectorAll(".instance")) {
      btn.setAttribute("aria-pressed", String(state.selected.has(Number(btn.dataset.id))));
    }
    const n = state.selected.size;
    el.selectCount.textContent = state.instances.length
      ? `${n} of ${state.instances.length} selected · tap the photo to toggle`
      : "0 selected";
  }

  // Tap on the photo: toggle the smallest object under the finger, so a
  // cushion on a sofa can be picked on its own.
  function onStageTap(event) {
    if (!state.instances.length) return;
    const rect = el.stagePhoto.getBoundingClientRect();
    const u = (event.clientX - rect.left) / rect.width;
    const v = (event.clientY - rect.top) / rect.height;
    if (u < 0 || u > 1 || v < 0 || v > 1) return;
    const hits = state.instances.filter((inst) => {
      const x = Math.min(inst.maskW - 1, Math.floor(u * inst.maskW));
      const y = Math.min(inst.maskH - 1, Math.floor(v * inst.maskH));
      return inst.alpha[(y * inst.maskW + x) * 4 + 3] > 127;
    });
    if (!hits.length) return;
    hits.sort((a, b) => a.area - b.area);
    toggleInstance(hits[0].id);
  }

  async function runEdit(quality) {
    clearError();
    el.instruction.blur();
    setBusy(quality === "final" ? "Final render · full model" : "Preview · 4 steps");
    try {
      const body = await api("/edit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: state.imageId,
          instance_ids: [...state.selected],
          instruction: el.instruction.value.trim(),
          quality,
          seed: Math.floor(Math.random() * 1e9),
        }),
      });
      state.result = body;
      el.beforeImg.src = state.photoUrl;
      el.afterImg.src = `data:image/png;base64,${body.image_png}`;
      setComparePos(50);
      const t = body.timings || {};
      const v = body.vram_gb || {};
      el.resultMeta.textContent =
        `${quality} · edit ${(t.edit || 0).toFixed(1)}s · ${body.model_size[0]}×${body.model_size[1]}` +
        (v.peak_allocated ? ` · peak ${v.peak_allocated} GB` : "");
      refreshHealth();
      goTo(4);
    } catch (err) {
      showError(err.message);
    } finally {
      setBusy(null);
      updateDock();
    }
  }

  // ---------------------------------------------------------------- 04 result
  function setComparePos(p) {
    el.compareRange.value = p;
    el.compare.style.setProperty("--pos", `${p}%`);
  }

  async function resultBlob() {
    const res = await fetch(`/image/${state.result.image_id}`);
    if (!res.ok) throw new Error("Could not fetch the full-resolution image.");
    return res.blob();
  }

  async function saveResult() {
    clearError();
    try {
      const blob = await resultBlob();
      const file = new File([blob], `tensorroom-${state.result.image_id}.png`, { type: "image/png" });
      // Phones: the share sheet offers "Save image" and messaging apps.
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        await navigator.share({ files: [file], title: "TensorRoom" });
        return;
      }
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = file.name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    } catch (err) {
      if (err.name !== "AbortError") showError(err.message);
    }
  }

  function keepEditing() {
    // The result becomes the new starting photo; the server already holds it.
    state.imageId = state.result.image_id;
    el.sessionId.textContent = state.imageId;
    setPhotoUrl(el.afterImg.src);
    el.photoPreview.src = state.photoUrl;
    el.photoMeta.textContent = "Edited photo";
    resetFrom(2);
    el.instruction.value = "";
    goTo(2);
    el.terms.focus({ preventScroll: true });
  }

  function startOver() {
    resetFrom(1);
    el.terms.value = "";
    el.instruction.value = "";
    syncChips();
    el.photoFrame.hidden = true;
    el.captureButtons.hidden = false;
    goTo(1);
    window.scrollTo({ top: 0 });
  }

  // ---------------------------------------------------------------- status bar
  async function refreshHealth() {
    try {
      const h = await api("/health");
      el.linkStatus.classList.toggle("is-online", h.ready);
      el.linkStatus.classList.toggle("is-offline", !h.ready);
      el.linkText.textContent = h.ready ? "GPU online" : "Loading";
      const v = h.vram_gb || {};
      el.vram.textContent = v.device_total ? `${v.device_used} / ${v.device_total} GB` : "—";
      if (h.quant) el.quant.textContent = h.quant.toUpperCase();
      if (!el.suggestions.children.length && h.vocabulary) renderSuggestions(h.vocabulary);
      return h.ready;
    } catch {
      el.linkStatus.classList.remove("is-online");
      el.linkStatus.classList.add("is-offline");
      el.linkText.textContent = "Offline";
      // Say once why nothing works, e.g. when the page is opened from a plain file server.
      if (!refreshHealth.warned) {
        refreshHealth.warned = true;
        showError(
          "No GPU worker found. Start it with: uv run uvicorn server:app --host 0.0.0.0 --port 8765 " +
          "and open this page from that address, or add ?demo to the address to try the app without a GPU."
        );
      }
      return false;
    }
  }

  // ---------------------------------------------------------------- wiring
  el.fileCamera.addEventListener("change", onPhotoChosen);
  el.fileLibrary.addEventListener("change", onPhotoChosen);
  el.retake.addEventListener("click", startOver);
  el.startOver.addEventListener("click", startOver);
  el.terms.addEventListener("input", onTermsChanged);
  el.terms.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && currentTerms().length && !state.busy) { e.preventDefault(); findObjects(); }
  });
  el.instruction.addEventListener("input", updateDock);
  el.stage.addEventListener("click", onStageTap);
  el.compareRange.addEventListener("input", (e) => setComparePos(e.target.value));
  window.addEventListener("resize", () => requestAnimationFrame(drawStage));
  el.stagePhoto.addEventListener("load", drawStage);

  goTo(1);
  // Poll until the worker has finished loading, then every 30 s for VRAM.
  (async function poll() {
    const ready = await refreshHealth();
    setTimeout(poll, ready ? 30000 : 3000);
  })();
})();
