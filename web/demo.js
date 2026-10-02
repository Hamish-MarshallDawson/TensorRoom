/*
  TensorRoom demo mode
  ====================
  Open the app with "?demo" on the end of the address (for example
  http://127.0.0.1:8765/?demo) to click through the whole flow with fake
  results and no GPU, models or server work. Useful for:

  * working on the UI on a laptop without the RTX 5080 stack installed;
  * testing layouts on real phones and in browser device emulators.

  Add "&guard" as well (e.g. ?demo&guard) to have every new photo refused
  by the people guardrail, to try the warning and return to the start.

  It quietly replaces fetch() for the four API calls. "Objects" are made-up
  ellipses, and the "edit" just recolours the chosen areas after a short
  pause. Without "?demo" this file does nothing.
*/
(() => {
  "use strict";
  if (!new URLSearchParams(location.search).has("demo")) return;

  const realFetch = window.fetch.bind(window);
  const store = new Map(); // image_id -> HTMLCanvasElement holding the photo
  let lastInstances = [];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const id = () => Math.random().toString(16).slice(2, 14);
  const json = (body) => new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });
  const b64 = (canvas) => canvas.toDataURL("image/png").split(",")[1];

  async function blobToCanvas(blob) {
    const img = await createImageBitmap(blob);
    const c = document.createElement("canvas");
    c.width = img.width; c.height = img.height;
    c.getContext("2d").drawImage(img, 0, 0);
    return c;
  }

  // Fake objects: a few ellipses in plausible furniture spots.
  function fakeInstances(w, h, terms) {
    const spots = [
      [0.30, 0.68, 0.22, 0.14], [0.70, 0.72, 0.16, 0.10], [0.52, 0.40, 0.10, 0.18], [0.15, 0.35, 0.08, 0.20],
    ];
    const labels = terms.length ? terms : ["object"];
    return spots.slice(0, Math.max(2, Math.min(4, labels.length + 1))).map(([cx, cy, rx, ry], n) => {
      const full = document.createElement("canvas");
      full.width = w; full.height = h;
      const ctx = full.getContext("2d");
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.ellipse(cx * w, cy * h, rx * w, ry * h, 0, 0, Math.PI * 2);
      ctx.fill();
      const scale = Math.min(1, 768 / Math.max(w, h));
      const small = document.createElement("canvas");
      small.width = Math.round(w * scale); small.height = Math.round(h * scale);
      small.getContext("2d").drawImage(full, 0, 0, small.width, small.height);
      return {
        id: n,
        label: labels[n % labels.length],
        phrase: labels[n % labels.length],
        score: 0.95 - n * 0.07,
        box: [Math.round((cx - rx) * w), Math.round((cy - ry) * h), Math.round((cx + rx) * w), Math.round((cy + ry) * h)],
        area: Math.round(Math.PI * rx * w * ry * h),
        mask_png: b64(small),
        _full: full,
      };
    });
  }

  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    switch (url.pathname) {
      case "/health":
        return json({
          ready: true,
          vram_gb: { device_used: 13.9, device_total: 15.9, peak_allocated: 12.4 },
          vocabulary: ["chair", "coffee table", "curtains", "lamp", "ottoman", "plant", "rug", "sofa", "table", "window"],
          quant: "fp8",
        });

      case "/segment": {
        await sleep(900);
        const form = init.body;
        let imageId = form.get("image_id");
        if (!imageId && new URLSearchParams(location.search).has("guard")) {
          return new Response(JSON.stringify({ detail: {
            code: "guardrail_person",
            message: "This photo looks like it shows a person or a character. TensorRoom only edits rooms, " +
              "so take or choose a photo of the room with nobody in it.",
            found: ["person"],
          } }), { status: 422, headers: { "Content-Type": "application/json" } });
        }
        if (!imageId) { imageId = id(); store.set(imageId, await blobToCanvas(form.get("image"))); }
        const photo = store.get(imageId);
        const terms = form.get("terms").split(",").map((t) => t.trim()).filter(Boolean);
        lastInstances = fakeInstances(photo.width, photo.height, terms);
        return json({
          image_id: imageId, width: photo.width, height: photo.height,
          instances: lastInstances.map(({ _full, ...rest }) => rest), timings: { segment: 0.4 },
        });
      }

      case "/edit": {
        const req = JSON.parse(init.body);
        await sleep(req.quality === "final" ? 4000 : 2000);
        const src = store.get(req.image_id);
        const out = document.createElement("canvas");
        out.width = src.width; out.height = src.height;
        const ctx = out.getContext("2d");
        ctx.drawImage(src, 0, 0);
        for (const n of req.instance_ids) {
          // Recolour inside the fake mask to stand in for a real edit.
          const tmp = document.createElement("canvas");
          tmp.width = src.width; tmp.height = src.height;
          const t = tmp.getContext("2d");
          t.filter = "hue-rotate(160deg) saturate(1.8)";
          t.drawImage(src, 0, 0);
          t.filter = "none";
          t.globalCompositeOperation = "destination-in";
          t.drawImage(lastInstances[n]._full, 0, 0);
          ctx.drawImage(tmp, 0, 0);
        }
        const newId = id();
        store.set(newId, out);
        const scale = Math.min(1, 1600 / Math.max(out.width, out.height));
        const display = document.createElement("canvas");
        display.width = Math.round(out.width * scale); display.height = Math.round(out.height * scale);
        display.getContext("2d").drawImage(out, 0, 0, display.width, display.height);
        return json({
          image_id: newId, image_png: b64(display), crop_box: [0, 0, out.width, out.height],
          model_size: [1024, 768], timings: { edit: req.quality === "final" ? 31.2 : 5.4 },
          vram_gb: { peak_allocated: 12.4 },
        });
      }

      default:
        if (url.pathname.startsWith("/image/")) {
          const c = store.get(url.pathname.split("/").pop());
          const blob = await new Promise((r) => c.toBlob(r, "image/png"));
          return new Response(blob, { headers: { "Content-Type": "image/png" } });
        }
        return realFetch(input, init);
    }
  };
  console.info("TensorRoom demo mode: API calls are simulated.");
})();
