// Sections 4-6 of the page: how the network learned, where it still fails, and
// what took it from 306 mistakes down to under 100.
//
// It reuses the helpers from app.js (decode, drawCells, makeCanvas, colors, hover,
// inkColor, weightColor, ...). Everything here is wrapped in a function so its
// names can't clash with app.js's.
"use strict";

(() => {
  const fmt = (n) => n.toLocaleString("en-US");
  const SVG = "http://www.w3.org/2000/svg";

  function svgEl(name, attrs, parent) {
    const el = document.createElementNS(SVG, name);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (parent) parent.append(el);
    return el;
  }

  // A small canvas showing a 28 x 28 image (bytes 0-255 or numbers -1..1)
  function imageCanvas(className) {
    const c = makeCanvas(112, 112);
    if (className) c.className = className;
    return c;
  }

  // ------------------------------------------------ section 4: the replay --

  // export_web.py saved ~40 snapshots taken while the network was training.
  // For each: how many images it had practiced on, its test accuracy, the
  // patterns of 8 neurons, and its guesses for 12 test digits.
  const R = MODEL.replay;
  const SNAPSHOTS = R.images_seen.length;
  const patterns = decode(R.patterns, Int8Array);       // (snapshots, neurons, 784)
  const digitImages = decode(R.digits, Uint8Array);     // (digits, 784)
  const guesses = decode(R.guesses, Uint8Array);        // (snapshots, digits)
  const confidence = decode(R.confidence, Uint8Array);  // (snapshots, digits), percent
  const PER_PASS = 55000;                               // images in one pass (epoch)

  const slider = document.getElementById("replay-slider");
  const playButton = document.getElementById("play");
  slider.max = SNAPSHOTS - 1;
  let snapshot = 0;
  let timer = null;

  // The 12 test digits: the image, and below it the guess at this moment
  const digitTiles = R.digit_labels.map((label, k) => {
    const tile = document.createElement("div");
    tile.className = "replay-digit";
    const canvas = imageCanvas();
    const text = document.createElement("p");
    tile.append(canvas, text);
    document.getElementById("replay-digits").append(tile);
    return { canvas, text, label, pixels: digitImages.subarray(k * PIXELS, (k + 1) * PIXELS) };
  });

  // The 8 neurons
  const neuronTiles = R.neurons.map((number, k) => {
    const canvas = imageCanvas();
    document.getElementById("replay-neurons").append(canvas);
    hover(canvas, () => `Neuron ${number}: the pattern it looks for at this moment`);
    return canvas;
  });

  function describe(images) {
    if (images === 0) return "Before any practice. All its numbers are random, so it is just guessing.";
    if (images < PER_PASS) return `After practicing on ${fmt(images)} images.`;
    const passes = Math.round(images / PER_PASS);
    const done = snapshot === SNAPSHOTS - 1 ? " Training is finished." : "";
    return `After ${passes} pass${passes > 1 ? "es" : ""} through all 55,000 practice images ` +
           `(${fmt(images)} images in total).${done}`;
  }

  function showSnapshot(s) {
    snapshot = s;
    slider.value = s;
    document.getElementById("replay-status").textContent = describe(R.images_seen[s]);
    document.getElementById("replay-accuracy").textContent = (R.accuracy[s] * 100).toFixed(1) + "%";

    digitTiles.forEach((tile, k) => {
      drawCells(tile.canvas, tile.pixels, SIZE, (v) => inkColor(v / 255));
      const guess = guesses[s * digitTiles.length + k];
      const sure = confidence[s * digitTiles.length + k];
      const right = guess === tile.label;
      tile.text.innerHTML =
        `<span class="verdict ${right ? "right" : "wrong"}">${right ? "✓" : "✗"} Says ${guess}</span>` +
        `<br>${sure}% sure`;
      tile.text.title = right ? "Right" : `Wrong: it is a ${tile.label}`;
    });

    neuronTiles.forEach((canvas, k) => {
      const start = (s * neuronTiles.length + k) * PIXELS;
      drawCells(canvas, patterns.subarray(start, start + PIXELS), SIZE, (w) => weightColor(w / 127));
    });

    moveChartMarker();
  }

  function stop() {
    clearInterval(timer);
    timer = null;
    playButton.textContent = "▶ Play";
  }
  function play() {
    if (snapshot === SNAPSHOTS - 1) showSnapshot(0);
    playButton.textContent = "❚❚ Pause";
    timer = setInterval(() => {
      if (snapshot === SNAPSHOTS - 1) stop();
      else showSnapshot(snapshot + 1);
    }, 450);
  }
  playButton.addEventListener("click", () => (timer ? stop() : play()));
  slider.addEventListener("input", () => { stop(); showSnapshot(Number(slider.value)); });

  // The accuracy chart. Practice is drawn on a log scale (each gridline is 10x
  // more images), because almost all the learning happens in the first few
  // thousand images and would be squashed against the left edge otherwise.
  const chartBox = document.getElementById("replay-chart");
  const LOG_START = Math.log10(30);  // "no practice yet" is drawn at 30 images
  const LOG_END = Math.log10(R.images_seen[SNAPSHOTS - 1]);
  let chart = null;

  function drawChart() {
    const width = chartBox.clientWidth, height = 220;
    const m = { left: 40, right: 12, top: 10, bottom: 28 };
    const x = (images) => m.left + (Math.log10(Math.max(images, 30)) - LOG_START) /
                          (LOG_END - LOG_START) * (width - m.left - m.right);
    const y = (acc) => m.top + (1 - acc) * (height - m.top - m.bottom);

    chartBox.textContent = "";
    const svg = svgEl("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
      "aria-label": "Line chart: test accuracy rises from 11% to 99% as the network practices" }, chartBox);
    for (const acc of [0, 0.25, 0.5, 0.75, 1]) {
      svgEl("line", { x1: m.left, x2: width - m.right, y1: y(acc), y2: y(acc),
                      class: acc === 0 ? "chart-axis" : "chart-grid" }, svg);
      svgEl("text", { x: m.left - 6, y: y(acc) + 4, "text-anchor": "end", class: "chart-tick" }, svg)
        .textContent = acc * 100 + "%";
    }
    const ticks = [[30, "none"], [100, "100"], [1000, "1,000"], [10000, "10,000"],
                   [100000, "100,000"], [1000000, "1 million"]];
    for (const [images, label] of ticks) {
      // Leave out labels that would collide on a narrow screen
      if (width < 420 && (images === 100 || images === 100000)) continue;
      svgEl("text", { x: x(images), y: height - 8, "text-anchor": images === 30 ? "start" : "middle",
                      class: "chart-tick" }, svg).textContent = label;
    }
    const points = R.images_seen.map((n, s) => [x(n), y(R.accuracy[s])]);
    const line = points.map((p) => p.join(",")).join(" L");
    svgEl("path", { d: `M${line} L${points[points.length - 1][0]},${y(0)} L${points[0][0]},${y(0)} Z`,
                    class: "chart-area" }, svg);
    svgEl("path", { d: "M" + line, class: "chart-line" }, svg);
    const now = svgEl("line", { y1: m.top, y2: y(0), class: "chart-now" }, svg);
    const dot = svgEl("circle", { r: 5, class: "chart-dot" }, svg);
    chart = { svg, points, now, dot };

    // Hovering shows the nearest snapshot; clicking jumps there
    const nearest = (e) => {
      const px = e.clientX - svg.getBoundingClientRect().left;
      let best = 0;
      points.forEach((p, s) => { if (Math.abs(p[0] - px) < Math.abs(points[best][0] - px)) best = s; });
      return best;
    };
    hover(svg, (e) => {
      const s = nearest(e);
      const n = R.images_seen[s];
      return `${n === 0 ? "No practice yet" : fmt(n) + " images practiced"}: ` +
             `${(R.accuracy[s] * 100).toFixed(1)}% right. Click to jump here.`;
    });
    svg.addEventListener("click", (e) => { stop(); showSnapshot(nearest(e)); });
    moveChartMarker();
  }

  function moveChartMarker() {
    if (!chart) return;
    const [px, py] = chart.points[snapshot];
    for (const attr of ["x1", "x2"]) chart.now.setAttribute(attr, px);
    chart.dot.setAttribute("cx", px);
    chart.dot.setAttribute("cy", py);
  }

  // --------------------------------------------- section 5: the mistakes --

  const M = MODEL.mistakes;
  const mistakeImages = decode(M.images, Uint8Array);
  const COUNT = M.labels.length;
  const FIRST_SHOWN = 24;
  let filter = null;      // { label, guess } when a cell of the grid is selected
  let showAll = false;

  // Sorted by real digit, then by guess, so similar mistakes sit together
  const mistakes = [...Array(COUNT).keys()]
    .sort((a, b) => M.labels[a] - M.labels[b] || M.guesses[a] - M.guesses[b]);

  document.getElementById("mistake-count").textContent = `${COUNT} (${(COUNT / 100).toFixed(2)}%)`;

  // How often each real digit (row) was taken for each guess (column)
  const counts = Array.from({ length: 10 }, () => new Array(10).fill(0));
  for (let k = 0; k < COUNT; k++) counts[M.labels[k]][M.guesses[k]]++;
  const maxCount = Math.max(...counts.flat());

  const confusion = document.getElementById("confusion");
  const cells = [];
  confusion.append(Object.assign(document.createElement("span"), { className: "head" }));
  for (let guess = 0; guess < 10; guess++) {
    confusion.append(Object.assign(document.createElement("span"), { className: "head", textContent: guess }));
  }
  for (let label = 0; label < 10; label++) {
    confusion.append(Object.assign(document.createElement("span"), { className: "head", textContent: label }));
    for (let guess = 0; guess < 10; guess++) {
      const n = counts[label][guess];
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "cell";
      if (label === guess) {
        // The diagonal is where it was right: show how many, quietly
        const right = M.test_counts[label] - counts[label].reduce((a, b) => a + b, 0);
        cell.classList.add("diagonal");
        cell.textContent = "✓";
        cell.disabled = true;
        hover(cell, () => `${label}: right ${fmt(right)} of ${fmt(M.test_counts[label])} times`);
      } else if (n > 0) {
        cell.classList.add("count");
        cell.textContent = n;
        cell.setAttribute("aria-label", `A ${label} taken for a ${guess}: ${n} times`);
        hover(cell, () => `A real ${label} taken for a ${guess}: ${n} time${n > 1 ? "s" : ""}. Click to see them.`);
        cell.addEventListener("click", () => {
          const same = filter && filter.label === label && filter.guess === guess;
          filter = same ? null : { label, guess };
          showMistakes();
        });
      } else {
        cell.disabled = true;
      }
      cells.push({ cell, label, guess, n });
      confusion.append(cell);
    }
  }

  function colorConfusion() {
    for (const { cell, n, label, guess } of cells) {
      if (n === 0 || label === guess) continue;
      const rgb = mix(colors["seq-low"], colors["seq-high"], 0.25 + 0.75 * n / maxCount);
      cell.style.background = css(rgb);
      // White or black text, whichever stands out more on this background
      const light = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) > 140;
      cell.style.color = light ? "#0b0b0b" : "#ffffff";
    }
  }

  const grid = document.getElementById("mistake-grid");
  const title = document.getElementById("mistakes-title");
  const moreButton = document.getElementById("mistakes-more");
  moreButton.addEventListener("click", () => {
    if (filter) filter = null;
    else showAll = !showAll;
    showMistakes();
  });

  function showMistakes() {
    for (const { cell, label, guess } of cells) {
      cell.classList.toggle("selected", !!filter && filter.label === label && filter.guess === guess);
    }
    let list = mistakes;
    if (filter) {
      list = mistakes.filter((k) => M.labels[k] === filter.label && M.guesses[k] === filter.guess);
      title.textContent = `Every ${filter.label} it took for a ${filter.guess}`;
      moreButton.textContent = "Show all mistakes";
    } else {
      title.textContent = showAll ? `All ${COUNT} mistakes` : `${FIRST_SHOWN} of its ${COUNT} mistakes`;
      moreButton.textContent = showAll ? "Show fewer" : `Show all ${COUNT}`;
      if (!showAll) list = list.slice(0, FIRST_SHOWN);
    }
    grid.textContent = "";
    for (const k of list) {
      const tile = document.createElement("div");
      tile.className = "mistake";
      const canvas = imageCanvas();
      drawCells(canvas, mistakeImages.subarray(k * PIXELS, (k + 1) * PIXELS), SIZE, (v) => inkColor(v / 255));
      const text = document.createElement("p");
      text.innerHTML = `Said <b>${M.guesses[k]}</b><br>Was <b>${M.labels[k]}</b>`;
      tile.append(canvas, text);
      hover(tile, () => `It said ${M.guesses[k]} (${Math.round(M.confidence[k] * 100)}% sure). ` +
                        `The right answer is ${M.labels[k]}.`);
      grid.append(tile);
    }
  }

  // ------------------------------------- section 6: from 306 mistakes ... --

  // Mistakes on the 10,000 test digits after each change, measured one change at
  // a time (see the git history and the README). The last one is this page's
  // own network, counted live from model.js.
  const STEPS = [
    ["The first version", 306,
     "128 neurons, 10 passes over the practice images, the simplest way of learning."],
    ["Less precise numbers", 273,
     "Calculating with fewer decimal places made practice twice as fast without hurting it. " +
     "(The small drop in mistakes is luck.)"],
    ["A better starting point", 249,
     "The random patterns it starts from were set to the right strength: not so faint that " +
     "learning crawls, not so strong that it goes haywire."],
    ["Momentum, then slowing down", 189,
     "Like a ball rolling downhill, each nudge keeps some speed from the ones before instead of " +
     "zig-zagging. The nudges also get smaller toward the end, so it settles in precisely."],
    ["More neurons, with random days off", 158,
     "4 times more neurons and twice the practice. While practicing, a random 1 in 5 neurons is " +
     "switched off each time, so it can't just memorize the examples."],
    ["Practicing with shifted images", COUNT,
     "Every practice image is moved up to 2 pixels in some direction, so it learns that a " +
     "digit is the same digit wherever it sits."],
  ];
  document.getElementById("final-mistakes").textContent = COUNT;
  const steps = document.getElementById("steps");
  STEPS.forEach(([name, mistakesAfter, detail]) => {
    steps.insertAdjacentHTML("beforeend",
      `<div class="step-text"><p class="step-title">${name}</p><p class="step-detail">${detail}</p></div>` +
      `<div class="step-bar"><div class="fill" style="width:${mistakesAfter / STEPS[0][1] * 80}%"></div>` +
      `<span class="value">${mistakesAfter}</span></div>`);
  });

  // ------------------------------------------------------------- start --

  function redrawAll() {
    showSnapshot(snapshot);
    colorConfusion();
    showMistakes();
  }
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(drawChart, 150);
  });
  // app.js re-reads the colors on a theme change; this listener runs after its one
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);

  drawChart();
  redrawAll();
})();
