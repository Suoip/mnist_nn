// How a computer reads handwriting - everything the page does.
//
// The network was trained in Python (network.py) and exported by export_web.py
// into model.js, which sets the global MODEL. This file only *runs* it: the same
// forward pass as network.forward(), plus the drawing pad and the pictures.
//
// Shapes follow network.py. Matrices are flat arrays stored row by row, so
// W1[j, i] (neuron j, pixel i) is W1[j * 784 + i].
"use strict";

const SIZE = 28;           // images are 28 x 28 pixels...
const PIXELS = SIZE * SIZE; // ...so 784 numbers

// ------------------------------------------------------------------ model --

// Turn a base64 string from model.js back into a typed array
function decode(base64, ArrayType) {
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  return new ArrayType(bytes.buffer);
}

const HIDDEN = MODEL.hidden_size;
const W1 = (() => {
  // export_web.py stored W1 as 8-bit integers plus one scale per row:
  // weight = integer * scale of its row
  const q = decode(MODEL.W1_q, Int8Array);
  const scale = decode(MODEL.W1_scale, Float32Array);
  const w = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) w[i] = q[i] * scale[Math.floor(i / PIXELS)];
  return w;
})();
const b1 = decode(MODEL.b1, Float32Array);
const W2 = decode(MODEL.W2, Float32Array);
const b2 = decode(MODEL.b2, Float32Array);
const EXAMPLES = decode(MODEL.examples, Uint8Array); // 100 real test images, 0-255

// Same steps as network.forward(): Linear -> ReLU -> Linear -> softmax.
// x is 784 numbers in [0, 1]. Returns the hidden activations and 10 probabilities.
function forward(x) {
  const hidden = new Float32Array(HIDDEN);
  for (let j = 0; j < HIDDEN; j++) {
    let z = b1[j];
    const row = j * PIXELS;
    for (let i = 0; i < PIXELS; i++) z += W1[row + i] * x[i];
    hidden[j] = Math.max(z, 0); // ReLU
  }
  const scores = new Float32Array(10);
  for (let d = 0; d < 10; d++) {
    let z = b2[d];
    const row = d * HIDDEN;
    for (let j = 0; j < HIDDEN; j++) z += W2[row + j] * hidden[j];
    scores[d] = z;
  }
  return { hidden, probs: softmax(scores) };
}

function softmax(scores) {
  const max = Math.max(...scores); // subtracting the max avoids overflow, see network.py
  const exps = scores.map((s) => Math.exp(s - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

// ----------------------------------------------------------------- colors --

// The colors live in style.css (light and dark); the canvases read them from there
let colors = {};
function readColors() {
  const css = getComputedStyle(document.documentElement);
  for (const name of ["paper", "ink", "grid", "accent", "seq-low", "seq-high",
                      "div-neg", "div-mid", "div-pos", "page"]) {
    colors[name] = hexToRgb(css.getPropertyValue("--" + name).trim());
  }
}
function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function mix(a, b, t) {
  return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * t));
}
const css = (rgb) => `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
const inkColor = (v) => mix(colors.paper, colors.ink, v);           // 0 = paper, 1 = ink
const activityColor = (v) => mix(colors["seq-low"], colors["seq-high"], v);
const weightColor = (v) => v >= 0                                   // -1 red ... 0 gray ... 1 blue
  ? mix(colors["div-mid"], colors["div-pos"], Math.min(v, 1))
  : mix(colors["div-mid"], colors["div-neg"], Math.min(-v, 1));

// Fill a canvas with a grid of colored cells, e.g. 28 x 28 pixels or 32 x 16 neurons.
// gap > 0 leaves thin lines of `gapColor` between the cells.
function drawCells(canvas, values, cols, colorOf, gap = 0, gapColor = colors.paper) {
  const rows = Math.ceil(values.length / cols);
  const ctx = canvas.getContext("2d");
  const cw = canvas.width / cols, ch = canvas.height / rows;
  if (gap) {
    ctx.fillStyle = css(gapColor);
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  for (let i = 0; i < values.length; i++) {
    const x = (i % cols) * cw, y = Math.floor(i / cols) * ch;
    ctx.fillStyle = css(colorOf(values[i]));
    ctx.fillRect(Math.round(x), Math.round(y),
                 Math.round(x + cw) - Math.round(x) - gap, Math.round(y + ch) - Math.round(y) - gap);
  }
}

function makeCanvas(width, height) {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  return c;
}

// ------------------------------------------------------------ drawing pad --

const pad = document.getElementById("pad");
const padCtx = pad.getContext("2d", { willReadFrequently: true });
const STROKE = 22; // pen width, in pad pixels (the pad is 280 x 280)
let penDown = false;
let lastPoint = null;

function padPoint(event) {
  const r = pad.getBoundingClientRect();
  return { x: (event.clientX - r.left) * pad.width / r.width,
           y: (event.clientY - r.top) * pad.height / r.height };
}

function drawLine(a, b) {
  padCtx.strokeStyle = padCtx.fillStyle = css(colors.ink);
  padCtx.lineWidth = STROKE;
  padCtx.lineCap = padCtx.lineJoin = "round";
  padCtx.beginPath();
  if (a === b) {
    padCtx.arc(a.x, a.y, STROKE / 2, 0, 2 * Math.PI); // a single tap draws a dot
    padCtx.fill();
  } else {
    padCtx.moveTo(a.x, a.y);
    padCtx.lineTo(b.x, b.y);
    padCtx.stroke();
  }
  scheduleUpdate();
}

pad.addEventListener("pointerdown", (event) => {
  penDown = true;
  pad.setPointerCapture(event.pointerId);
  lastPoint = padPoint(event);
  setNote("");
  drawLine(lastPoint, lastPoint);
});
pad.addEventListener("pointermove", (event) => {
  if (!penDown) return;
  const p = padPoint(event);
  drawLine(lastPoint, p);
  lastPoint = p;
});
for (const type of ["pointerup", "pointercancel"]) {
  pad.addEventListener(type, () => { penDown = false; });
}

function clearPad() {
  padCtx.clearRect(0, 0, pad.width, pad.height);
  setNote("");
  scheduleUpdate();
}

// Copy one of the real test images onto the pad, as if someone had drawn it
function showExample() {
  const k = Math.floor(Math.random() * MODEL.example_labels.length);
  const image = EXAMPLES.subarray(k * PIXELS, (k + 1) * PIXELS);
  const small = makeCanvas(SIZE, SIZE);
  const ctx = small.getContext("2d");
  const data = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < PIXELS; i++) {
    data.data.set([...colors.ink, image[i]], i * 4); // ink color, with the pixel value as opacity
  }
  ctx.putImageData(data, 0, 0);
  padCtx.clearRect(0, 0, pad.width, pad.height);
  padCtx.imageSmoothingEnabled = true;
  padCtx.drawImage(small, 0, 0, pad.width, pad.height);
  setNote(`A real handwritten ${MODEL.example_labels[k]} from the test set. ` +
          "The network never saw it while it was learning. Clear it and try your own!");
  scheduleUpdate();
}

// The ink is stored in the pad's transparency, so when the theme changes we can
// recolor everything already drawn without losing it
function recolorPad() {
  padCtx.save();
  padCtx.globalCompositeOperation = "source-in";
  padCtx.fillStyle = css(colors.ink);
  padCtx.fillRect(0, 0, pad.width, pad.height);
  padCtx.restore();
}

function setNote(text) {
  document.getElementById("example-note").textContent = text;
}

// ---------------------------------------------------------- preprocessing --

// Make the drawing look like an MNIST image. Every training image was made the
// same way, so a drawing has to be too:
//   1. crop to the ink,
//   2. shrink so its longer side is 20 pixels (keeping its shape),
//   3. put it in a 28 x 28 grid with its center of mass in the middle.
// Without this the network does badly: it never saw a digit that was huge,
// tiny, or off in a corner.
// Returns the 784 numbers plus the crop box, or null if nothing is drawn.
function preprocess() {
  const { width, height } = pad;
  const rgba = padCtx.getImageData(0, 0, width, height).data;

  // 1. The bounding box of the ink (every pixel more than ~10% covered)
  let left = width, top = height, right = -1, bottom = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (rgba[(y * width + x) * 4 + 3] > 25) {
        left = Math.min(left, x); right = Math.max(right, x);
        top = Math.min(top, y); bottom = Math.max(bottom, y);
      }
    }
  }
  if (right < 0) return null;
  const box = { x: left, y: top, w: right - left + 1, h: bottom - top + 1 };

  // 2. Shrink the box so its longer side is 20 pixels
  const scale = 20 / Math.max(box.w, box.h);
  const w = Math.max(1, Math.round(box.w * scale));
  const h = Math.max(1, Math.round(box.h * scale));
  const small = shrink(pad, box, w, h);

  // 3. Center of mass of the ink, then shift it to the middle of the 28 x 28 grid
  const ink = small.getContext("2d").getImageData(0, 0, w, h).data;
  let total = 0, cx = 0, cy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = ink[(y * w + x) * 4 + 3];
      total += v; cx += v * (x + 0.5); cy += v * (y + 0.5);
    }
  }
  if (total === 0) return null;
  const grid = makeCanvas(SIZE, SIZE);
  const gridCtx = grid.getContext("2d");
  gridCtx.drawImage(small, Math.round(SIZE / 2 - cx / total), Math.round(SIZE / 2 - cy / total));

  const out = gridCtx.getImageData(0, 0, SIZE, SIZE).data;
  const pixels = new Float32Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) pixels[i] = out[i * 4 + 3] / 255;
  return { pixels, box };
}

// Shrink part of a canvas to w x h. Going down at most 2x per step keeps thin
// strokes smooth; one big jump would skip over pixels and make them patchy.
function shrink(source, box, w, h) {
  let canvas = source, { x, y, w: sw, h: sh } = box;
  while (sw > 2 * w || sh > 2 * h) {
    const next = makeCanvas(Math.max(w, Math.ceil(sw / 2)), Math.max(h, Math.ceil(sh / 2)));
    next.getContext("2d").drawImage(canvas, x, y, sw, sh, 0, 0, next.width, next.height);
    canvas = next; x = 0; y = 0; sw = next.width; sh = next.height;
  }
  const out = makeCanvas(w, h);
  const ctx = out.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(canvas, x, y, sw, sh, 0, 0, w, h);
  return out;
}

// --------------------------------------------------------------- updating --

// Redraw at most once per screen refresh, however fast the pointer moves
let updateQueued = false;
function scheduleUpdate() {
  if (updateQueued) return;
  updateQueued = true;
  requestAnimationFrame(() => { updateQueued = false; update(); });
}

let current = null; // what is on screen now, for the tooltips

function update() {
  const seen = preprocess();
  document.getElementById("pad-hint").hidden = !!seen;
  if (!seen) {
    current = null;
    showResult(null);
    showSeen(null);
    showInside(null);
    return;
  }
  const { hidden, probs } = forward(seen.pixels);
  current = { ...seen, hidden, probs };
  showResult(probs);
  showSeen(seen);
  showInside(current);
}

// ------------------------------------------------- section 1: the result --

const pct = (p) => (p >= 0.995 ? "100%" : p < 0.01 ? "<1%" : Math.round(p * 100) + "%");

const barsEl = document.getElementById("bars");
const bars = [];
for (let d = 0; d < 10; d++) {
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.innerHTML = `<span class="bar-value"></span><div class="bar-track"><div class="bar-fill"></div></div>` +
                  `<span class="bar-digit">${d}</span>`;
  barsEl.append(bar);
  bars.push(bar);
  hover(bar, () => current && `${d}: ${pct(current.probs[d])} sure`);
}

function showResult(probs) {
  const guess = document.getElementById("guess");
  const sure = document.getElementById("sure");
  if (!probs) {
    guess.textContent = "?";
    sure.textContent = "Draw something to find out.";
  } else {
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
    const [first, second] = order;
    guess.textContent = first;
    const p = probs[first];
    sure.textContent =
      p >= 0.9 ? `It's ${pct(p)} sure.` :
      p >= 0.5 ? `Fairly sure (${pct(p)}), but it also thought about a ${second} (${pct(probs[second])}).` :
                 `Not sure at all: maybe a ${first} (${pct(p)}) or a ${second} (${pct(probs[second])}).`;
  }
  const top = probs ? probs.indexOf(Math.max(...probs)) : -1;
  bars.forEach((bar, d) => {
    const p = probs ? probs[d] : 0;
    bar.classList.toggle("top", d === top);
    bar.querySelector(".bar-fill").style.height = (p * 100) + "%";
    // Only label the bars that matter; hovering shows the rest
    bar.querySelector(".bar-value").textContent = p >= 0.05 ? pct(p) : "";
  });
}

// --------------------------------------------- section 2: what it sees --

const seeDrawing = document.getElementById("see-drawing");
const seeGrid = document.getElementById("see-grid");
const seeNumbers = document.getElementById("see-numbers");

function showSeen(seen) {
  // Your drawing, with the crop box around the ink
  const ctx = seeDrawing.getContext("2d");
  ctx.clearRect(0, 0, seeDrawing.width, seeDrawing.height);
  ctx.drawImage(pad, 0, 0);
  if (seen) {
    ctx.strokeStyle = css(colors.accent);
    ctx.lineWidth = 4;
    const { x, y, w, h } = seen.box;
    ctx.strokeRect(x - 2, y - 2, w + 4, h + 4);
  }

  // The 28 x 28 grid
  const pixels = seen ? seen.pixels : new Float32Array(PIXELS);
  drawCells(seeGrid, pixels, SIZE, inkColor, 1, colors.grid);

  // The same 784 numbers in one row: a thin bar per number, on bands that
  // alternate every 28 numbers to show where each row of the grid starts
  const n = seeNumbers.getContext("2d");
  const W = seeNumbers.width, H = seeNumbers.height, step = W / PIXELS;
  n.clearRect(0, 0, W, H);
  n.fillStyle = css(colors.grid);
  for (let row = 1; row < SIZE; row += 2) n.fillRect(row * SIZE * step, 0, SIZE * step, H);
  n.fillStyle = css(colors.accent);
  for (let i = 0; i < PIXELS; i++) {
    const barHeight = pixels[i] * (H - 8);
    n.fillRect(i * step, H - barHeight, step, barHeight);
  }
}

hover(seeGrid, (e) => {
  if (!current) return null;
  const { col, row } = cellAt(seeGrid, e, SIZE, SIZE);
  return `Row ${row + 1}, column ${col + 1}: ${current.pixels[row * SIZE + col].toFixed(2)}`;
});
hover(seeNumbers, (e) => {
  if (!current) return null;
  const { col: i } = cellAt(seeNumbers, e, PIXELS, 1);
  return `Number ${i + 1} of 784 (row ${Math.floor(i / SIZE) + 1}, column ${i % SIZE + 1}): ` +
         current.pixels[i].toFixed(2);
});

// ---------------------------------------------- section 3: inside the net --

const HIDDEN_COLS = 32; // the 512 neurons are drawn as a 32 x 16 grid
const TOP_NEURONS = 8;
const netInput = document.getElementById("net-input");
const netHidden = document.getElementById("net-hidden");
const netOutput = document.getElementById("net-output");
const neuronsEl = document.getElementById("neurons");

const outputRows = [];
for (let d = 0; d < 10; d++) {
  const digit = document.createElement("span");
  digit.className = "digit";
  digit.textContent = d;
  const track = document.createElement("div");
  track.className = "track";
  track.innerHTML = `<div class="fill" style="width:0"></div>`;
  const value = document.createElement("span");
  value.className = "pct";
  netOutput.append(digit, track, value);
  outputRows.push({ digit, fill: track.firstChild, value });
}

function showInside(state) {
  const pixels = state ? state.pixels : new Float32Array(PIXELS);
  const hidden = state ? state.hidden : new Float32Array(HIDDEN);
  drawCells(netInput, pixels, SIZE, inkColor);

  // Neurons: brightness relative to the most active one
  const maxActivity = Math.max(...hidden, 1e-6);
  drawCells(netHidden, hidden, HIDDEN_COLS, (a) => activityColor(a / maxActivity), 2,
            colors.page);

  const top = state ? state.probs.indexOf(Math.max(...state.probs)) : -1;
  outputRows.forEach((row, d) => {
    const p = state ? state.probs[d] : 0;
    row.fill.style.width = (p * 100) + "%";
    row.value.textContent = state ? pct(p) : "";
    row.digit.classList.toggle("top", d === top);
  });

  showTopNeurons(state);
}

hover(netHidden, (e) => {
  if (!current) return null;
  const { col, row } = cellAt(netHidden, e, HIDDEN_COLS, HIDDEN / HIDDEN_COLS);
  const j = row * HIDDEN_COLS + col;
  return `Neuron ${j + 1}: activity ${current.hidden[j].toFixed(2)}`;
});

// Cards for the neurons that did the most to pick the winning digit.
// A neuron's vote for digit d is its activity times its weight to d, W2[d, j]:
// the final score of each digit is just b2 plus all 512 votes for it added up.
function showTopNeurons(state) {
  const title = document.getElementById("neurons-title");
  neuronsEl.textContent = "";
  if (!state) {
    title.textContent = "The neurons that voted hardest";
    neuronsEl.innerHTML = `<p class="empty">Draw a digit to see which neurons light up.</p>`;
    return;
  }
  const winner = state.probs.indexOf(Math.max(...state.probs));
  title.textContent = `The neurons that voted hardest for ${winner}`;
  const vote = (d, j) => state.hidden[j] * W2[d * HIDDEN + j];
  const order = [...state.hidden.keys()].sort((a, b) => vote(winner, b) - vote(winner, a));
  const strongest = vote(winner, order[0]) || 1;

  for (const j of order.slice(0, TOP_NEURONS)) {
    const weights = W1.subarray(j * PIXELS, (j + 1) * PIXELS);
    const biggest = weights.reduce((m, w) => Math.max(m, Math.abs(w)), 1e-6);

    // Left: the pattern this neuron looks for. Right: pattern x drawing, i.e. how
    // much each pixel of the drawing adds to (blue) or takes from (red) its activity
    const pattern = makeCanvas(112, 112);
    drawCells(pattern, weights, SIZE, (w) => weightColor(w / biggest));
    const overlay = makeCanvas(112, 112);
    drawCells(overlay, weights.map((w, i) => w * state.pixels[i]), SIZE,
              (w) => weightColor(w / biggest));

    // The digit this neuron votes against the hardest right now
    const votes = Array.from({ length: 10 }, (_, d) => vote(d, j));
    const against = votes.indexOf(Math.min(...votes));

    const card = document.createElement("div");
    card.className = "neuron";
    const images = document.createElement("div");
    images.className = "neuron-images";
    images.append(pattern, overlay);
    card.append(images);
    card.insertAdjacentHTML("beforeend",
      `<p class="neuron-title">Neuron ${j + 1}</p>` +
      `<div class="neuron-meter"><div style="width:${Math.max(0, vote(winner, j) / strongest) * 100}%"></div></div>` +
      `<p class="neuron-votes">For <b>${winner}</b>, against <b>${against}</b></p>`);
    neuronsEl.append(card);
  }
}

// ---------------------------------------------------------------- tooltip --

const tooltip = document.getElementById("tooltip");
function hover(element, textAt) {
  element.addEventListener("pointermove", (e) => {
    const text = textAt(e);
    tooltip.hidden = !text;
    if (!text) return;
    tooltip.textContent = text;
    const x = Math.min(e.clientX + 14, window.innerWidth - tooltip.offsetWidth - 8);
    tooltip.style.left = x + "px";
    tooltip.style.top = (e.clientY + 16) + "px";
  });
  element.addEventListener("pointerleave", () => { tooltip.hidden = true; });
}

// Which cell of a cols x rows grid drawn on `canvas` the pointer is over
function cellAt(canvas, e, cols, rows) {
  const r = canvas.getBoundingClientRect();
  const clamp = (v, n) => Math.min(n - 1, Math.max(0, Math.floor(v * n)));
  return { col: clamp((e.clientX - r.left) / r.width, cols),
           row: clamp((e.clientY - r.top) / r.height, rows) };
}

// ------------------------------------------------------------------ start --

document.getElementById("clear").addEventListener("click", clearPad);
document.getElementById("example").addEventListener("click", showExample);
document.getElementById("accuracy").textContent =
  (MODEL.test_accuracy * 100).toFixed(1) + "%";

readColors();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  readColors();
  recolorPad();
  update();
});
showExample(); // start with a real digit on the pad, so nothing is empty
