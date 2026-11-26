/* The research assembly: a wireframe rig drawn with three.js. Scrolling the "Test run" sheet drops a
   topic through it: plan, your review, four section lanes, the verify sieve, the report tray. */
import * as THREE from "three";

const M = window.Motion || null;
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
const FINE_POINTER = matchMedia("(hover: hover) and (pointer: fine)").matches;

const LANE_X = [-3.75, -1.25, 1.25, 3.75];
const Y = { top: 9.2, plan: 6, review: 3, split: 1.7, search: 0.4, read: -1.1, write: -2.6, sieve: -4.4, tray: -7.2 };
const STATIONS = [Y.search, Y.read, Y.write];
const REJECTED_LANE = 1;  // the lane that loses a sentence at the sieve

const css = getComputedStyle(document.documentElement);
const INK = new THREE.Color(css.getPropertyValue("--sc-ink").trim() || "#EAF0F6");
const RED = new THREE.Color(css.getPropertyValue("--sc-accent").trim() || "#FF6B45");

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const smooth = (t) => t * t * (3 - 2 * t);
const lerp = (a, b, t) => a + (b - a) * t;
const deg = (d) => (d * Math.PI) / 180;
const between = (p, a, b) => clamp((p - a) / (b - a));

function sheetProgress(el) {
  const r = el.getBoundingClientRect();
  const travel = Math.max(el.offsetHeight - innerHeight, 1);
  return { p: clamp(-r.top / travel), visible: r.bottom > 0 && r.top < innerHeight, top: r.top };
}

// ---------------------------------------------------------------- where every packet is at progress p
// Each row is [progress, x, y]. Before the split all four lanes travel together as one packet.
function laneFrames(i) {
  const x = LANE_X[i];
  return [
    [0.00, 0, Y.top], [0.10, 0, Y.plan + 0.7], [0.16, 0, Y.plan + 0.7],
    [0.24, 0, Y.review + 0.8], [0.32, 0, Y.review + 0.8],
    [0.40, x, Y.split], [0.42 + i * 0.02, x, Y.split],
    [0.66 + i * 0.012, x, Y.sieve + 0.9], [0.72, x, Y.sieve + 0.9],
    [0.82, x, Y.sieve - 0.9], [0.92, x, Y.tray + 0.55], [1, x, Y.tray + 0.55],
  ];
}
const FRAMES = LANE_X.map((_, i) => laneFrames(i));

function along(frames, p) {
  for (let i = 1; i < frames.length; i++) {
    if (p <= frames[i][0]) {
      const [p0, x0, y0] = frames[i - 1];
      const [p1, x1, y1] = frames[i];
      const t = smooth(clamp((p - p0) / Math.max(p1 - p0, 1e-6)));
      return { x: lerp(x0, x1, t), y: lerp(y0, y1, t) };
    }
  }
  const last = frames[frames.length - 1];
  return { x: last[1], y: last[2] };
}

// ---------------------------------------------------------------- the scene
const canvas = $("#rig");
const flat = $("#rig-flat");

function buildRig() {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  } catch {
    return null;
  }
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 200);

  const mat = (opacity, color = INK) => new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const segments = (points, material) =>
    new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p))), material);
  const edges = (geometry, material) => new THREE.LineSegments(new THREE.EdgesGeometry(geometry), material);
  const ring = (radius, y, material, n = 64) => {
    const points = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      points.push(new THREE.Vector3(Math.cos(a) * radius, y, Math.sin(a) * radius));
    }
    return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), material);
  };
  const plate = (w, d, y, material) => {
    const mesh = edges(new THREE.BoxGeometry(w, 0.3, d), material);
    mesh.position.y = y;
    return mesh;
  };

  // centre line the topic falls along, then the four lane guides
  const axis = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, Y.top + 0.8, 0), new THREE.Vector3(0, Y.split, 0)]),
    new THREE.LineDashedMaterial({ color: INK, transparent: true, opacity: 0.35, dashSize: 0.3, gapSize: 0.25 }),
  );
  axis.computeLineDistances();
  scene.add(axis);
  const guides = [];
  LANE_X.forEach((x) => guides.push([0, Y.split + 0.6, 0], [x, Y.split, 0], [x, Y.split, 0], [x, Y.tray + 0.5, 0]));
  scene.add(segments(guides, mat(0.22)));

  // 1 plan: a plate with one slot per section
  const plan = { body: mat(0.85), slots: mat(0) };
  scene.add(plate(7, 5, Y.plan, plan.body));
  const slotLines = [];
  for (let i = 0; i < 5; i++) slotLines.push([-2.8, Y.plan + 0.16, -1.8 + i * 0.9], [2.8, Y.plan + 0.16, -1.8 + i * 0.9]);
  scene.add(segments(slotLines, plan.slots));

  // 2 your review: a ring with an iris that opens when you approve
  const review = { body: mat(0.85), iris: new THREE.Group() };
  scene.add(ring(3.2, Y.review, review.body));
  scene.add(ring(3.4, Y.review, mat(0.3)));
  const blades = [];
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    blades.push([Math.cos(a) * 3.2, 0, Math.sin(a) * 3.2], [Math.cos(a + 1.1) * 1.1, 0, Math.sin(a + 1.1) * 1.1]);
  }
  review.iris.add(segments(blades, review.body));
  review.iris.position.y = Y.review;
  scene.add(review.iris);

  // 3 lanes: search (ring), read (stack of pages), write (small plate) on each
  const stations = LANE_X.map((x) => {
    const mats = [mat(0.35), mat(0.35), mat(0.35)];
    const search = ring(0.75, Y.search, mats[0], 28);
    search.position.x = x;
    scene.add(search);
    for (let k = 0; k < 3; k++) {
      const page = edges(new THREE.BoxGeometry(1.3, 0.05, 1), mats[1]);
      page.position.set(x, Y.read - 0.22 + k * 0.22, 0);
      scene.add(page);
    }
    const write = edges(new THREE.BoxGeometry(1.5, 0.2, 1.5), mats[2]);
    write.position.set(x, Y.write, 0);
    scene.add(write);
    return mats;
  });

  // 4 verify sieve: a wide plate with a mesh
  const sieve = { body: mat(0.85), mesh: mat(0.3) };
  scene.add(plate(10.5, 3, Y.sieve, sieve.body));
  const mesh = [];
  for (let i = -5; i <= 5; i++) mesh.push([i, Y.sieve + 0.16, -1.5], [i, Y.sieve + 0.16, 1.5]);
  for (let j = -1; j <= 1; j++) mesh.push([-5.25, Y.sieve + 0.16, j], [5.25, Y.sieve + 0.16, j]);
  scene.add(segments(mesh, sieve.mesh));

  // 5 report tray
  const tray = LANE_X.map((x) => {
    const material = mat(0.65);
    const box = edges(new THREE.BoxGeometry(2, 1.1, 2), material);
    box.position.set(x, Y.tray, 0);
    scene.add(box);
    return material;
  });
  scene.add(segments([[-5.4, Y.tray - 0.6, 1.2], [5.4, Y.tray - 0.6, 1.2], [-5.4, Y.tray - 0.6, -1.2], [5.4, Y.tray - 0.6, -1.2]], mat(0.3)));

  // the packets: one per lane (they overlap until the split), and the rejected sentence
  const makePacket = (size) => {
    const group = new THREE.Group();
    group.add(edges(new THREE.OctahedronGeometry(size), mat(1, RED)));
    const fill = new THREE.MeshBasicMaterial({ color: RED, transparent: true, opacity: 0.28, depthWrite: false });
    group.add(new THREE.Mesh(new THREE.OctahedronGeometry(size * 0.98), fill));
    scene.add(group);
    return group;
  };
  const packets = LANE_X.map(() => makePacket(0.46));
  const shard = makePacket(0.2);
  const shardLine = mat(0, RED);
  const shardPath = new THREE.Line(new THREE.BufferGeometry(), shardLine);
  scene.add(shardPath);

  return { renderer, scene, camera, plan, review, stations, sieve, tray, packets, shard, shardPath, shardLine };
}

const gl = buildRig();
if (!gl) buildFlat();

// Without WebGL: the same stations as a flat drawing, with one dot that travels down it.
function buildFlat() {
  canvas.hidden = true;
  flat.hidden = false;
  const NS = "http://www.w3.org/2000/svg";
  const add = (tag, attrs) => {
    const node = document.createElementNS(NS, tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    flat.appendChild(node);
    return node;
  };
  add("rect", { x: -3.5, y: -Y.plan - 0.15, width: 7, height: 0.3, class: "fl" });
  add("circle", { cx: 0, cy: -Y.review, r: 1.6, class: "fl" });
  add("rect", { x: -5.25, y: -Y.sieve - 0.15, width: 10.5, height: 0.3, class: "fl" });
  LANE_X.forEach((x) => {
    add("path", { d: `M${x} ${-Y.split} V${-Y.tray}`, class: "fl-faint" });
    add("rect", { x: x - 1, y: -Y.tray - 0.55, width: 2, height: 1.1, class: "fl" });
  });
  LANE_X.forEach((_, i) => add("circle", { r: 0.32, class: "fl-red", "data-dot": i }));
}

// ---------------------------------------------------------------- camera
const cam = { az: deg(38), el: deg(24), dist: 40, ty: 0.5, ox: 0, oy: 0 };
const pointer = { x: 0, y: 0 };
if (FINE_POINTER && !REDUCE) {
  addEventListener("pointermove", (e) => {
    pointer.x = e.clientX / innerWidth - 0.5;
    pointer.y = e.clientY / innerHeight - 0.5;
  }, { passive: true });
}
const isMobile = () => innerWidth <= 860;

function fitDist(halfH, halfW) {
  const t = Math.tan(deg(15));
  return Math.max(halfH / t, halfW / (t * (innerWidth / innerHeight)));
}

function cameraTarget(hero, run, followY) {
  const mobile = isMobile();
  const hp = REDUCE ? 0 : hero.p;
  const wide = {
    az: deg(lerp(38, 24, hp)), el: deg(lerp(24, 13, hp)),
    dist: fitDist(10.4, 8.6) * (mobile ? 1.35 : 1.14), ty: mobile ? 1.4 : 1,
    ox: mobile ? 0 : -0.2, oy: mobile ? -0.3 : 0,
  };
  if (run.p <= 0 && run.top > 0) return wide;
  const p = REDUCE ? 1 : run.p;
  const landed = smooth(between(p, 0.9, 0.98));
  const close = {
    az: deg(lerp(24, 8, p)), el: deg(lerp(13, 7, p)),
    dist: lerp(fitDist(6.2, 8.4), fitDist(7.4, 8.8), landed),
    ty: lerp(clamp(followY, -5.4, 7.5), -5.4, landed),
    ox: mobile ? 0 : lerp(-0.19, -0.15, landed), oy: mobile ? 0.15 : 0,
  };
  const blend = smooth(clamp(p / 0.07));
  const out = {};
  for (const key in wide) out[key] = lerp(wide[key], close[key], blend);
  return out;
}

function applyCamera() {
  const c = gl.camera;
  const az = cam.az + pointer.x * deg(3), el = cam.el - pointer.y * deg(2);
  c.position.set(Math.sin(az) * Math.cos(el) * cam.dist, cam.ty + Math.sin(el) * cam.dist, Math.cos(az) * Math.cos(el) * cam.dist);
  c.lookAt(0, cam.ty, 0);
  c.setViewOffset(innerWidth, innerHeight, cam.ox * innerWidth, cam.oy * innerHeight, innerWidth, innerHeight);
  c.updateProjectionMatrix();
}

function resize() {
  if (!gl) return;
  gl.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, isMobile() ? 1.5 : 2));
  gl.renderer.setSize(innerWidth, innerHeight, false);
  gl.camera.aspect = innerWidth / innerHeight;
}
addEventListener("resize", resize);
resize();

// ---------------------------------------------------------------- labels pinned to points of the rig
const tagEl = Object.fromEntries($$("[data-tag]").map((node) => [node.dataset.tag, node]));
const v3 = new THREE.Vector3();

function project(x, y, z = 0) {
  if (gl) {
    v3.set(x, y, z).project(gl.camera);
    return { x: (v3.x * 0.5 + 0.5) * innerWidth, y: (-v3.y * 0.5 + 0.5) * innerHeight, ok: v3.z < 1 };
  }
  const matrix = flat.getScreenCTM();
  if (!matrix) return { x: 0, y: 0, ok: false };
  const point = new DOMPoint(x, -y).matrixTransform(matrix);
  return { x: point.x, y: point.y, ok: true };
}

function place(node, pos, opacity, dx = 0) {
  const fits = pos.x + dx > 8 && pos.x + dx + node.offsetWidth < innerWidth - 8;  // never show a label cut by the edge
  node.style.transform = `translate3d(${Math.round(pos.x + dx)}px, ${Math.round(pos.y)}px, 0) translateY(-50%)`;
  node.style.opacity = pos.ok && fits ? String(opacity) : "0";
}

// ---------------------------------------------------------------- the log beside the rig
const STEPS = [
  { key: "plan", from: 0.10, done: 0.16, doing: "splitting the topic into sections", after: "outline drafted: 4 sections" },
  { key: "review", from: 0.24, done: 0.32, doing: "waiting for you to approve or edit", after: "approved by you" },
  { key: "lanes", from: 0.40, done: 0.70, doing: "4 sections at once: searching, reading, writing", after: "4 drafts, every sentence cited" },
  { key: "sieve", from: 0.72, done: 0.84, doing: "checking each sentence against its source", after: "1 sentence removed: its source did not say it" },
  { key: "tray", from: 0.90, done: 0.92, doing: "assembling", after: "ready, with references and a trust score" },
];
const logLines = Object.fromEntries($$(".log__lines li").map((li) => [li.dataset.step, li]));
let lastLog = "";

function updateLog(p) {
  const states = STEPS.map((step) => (p >= step.done ? "done" : p >= step.from ? "doing" : "waiting"));
  const key = states.join();
  if (key === lastLog) return;
  lastLog = key;
  STEPS.forEach((step, i) => {
    const li = logLines[step.key];
    const value = li.querySelector(".log__v");
    const last = step.key === "tray";
    if (states[i] === "waiting") {
      li.className = last ? "log__out" : "";
      value.textContent = last ? "pending" : "waiting";
    } else if (states[i] === "doing") {
      li.className = last ? "log__out" : "is-caught";
      value.textContent = step.doing;
    } else {
      li.className = last ? "log__out is-done" : "is-passed";
      value.textContent = step.after;
    }
  });
}

// ---------------------------------------------------------------- the stamp
let stampText = "";
function setStamp(text, at) {
  const stamp = tagEl.stamp;
  if (!text) { stamp.hidden = true; stampText = ""; return; }
  stamp.hidden = false;
  if (text !== stampText) {
    stampText = text;
    stamp.textContent = text;
    if (M && !REDUCE) M.animate(stamp, { scale: [1.35, 1], opacity: [0, 1] }, { duration: 0.28 });
  }
  place(stamp, at, 1, -stamp.offsetWidth / 2);
}

// ---------------------------------------------------------------- frame loop
const heroEl = $("#assembly"), runEl = $("#run"), startEl = $("#start");
const paper = $(".paper"), dim = $(".dimline"), dimLabel = $(".dimline__label");
const titleblock = $(".titleblock");
const sheetLinks = $$(".tb-sheets a");
const sheets = sheetLinks.map((a) => $("#" + a.dataset.sheet));
let topic = $("#q").value;

function frame(time) {
  const hero = sheetProgress(heroEl);
  const run = sheetProgress(runEl);
  const show = hero.visible || run.visible;

  if (!REDUCE) paper.style.transform = `translate3d(0, ${-((scrollY * 0.12) % 140)}px, 0)`;
  if (!REDUCE && dim) {
    const y = -hero.p * 160;
    dim.style.transform = `translate3d(0, ${y}px, 0)`;
    dimLabel.style.transform = `translate3d(0, ${y}px, 0)`;
  }
  (gl ? canvas : flat).classList.toggle("is-off", !show);
  $(".tags").style.visibility = show ? "visible" : "hidden";

  // which sheet the title block highlights, and hide it over the white last sheet
  let current = 0;
  sheets.forEach((section, i) => { if (section.getBoundingClientRect().top <= innerHeight * 0.5) current = i; });
  sheetLinks.forEach((a, i) => a.classList.toggle("is-current", i === current));
  titleblock.classList.toggle("is-away", startEl.getBoundingClientRect().top < innerHeight * 0.6);

  if (show) {
    const p = run.p <= 0 && run.top > 0 ? 0 : (REDUCE ? 1 : run.p);
    const spots = FRAMES.map((frames) => along(frames, p));
    const split = p > 0.32;
    const followY = spots.reduce((sum, s) => sum + s.y, 0) / spots.length;
    const heroOn = run.p <= 0 && run.top > innerHeight * 0.35;

    if (gl) {
      const target = cameraTarget(hero, run, followY);
      for (const key in target) cam[key] = lerp(cam[key], target[key], REDUCE ? 1 : 0.1);
      applyCamera();

      const bob = REDUCE || p > 0.05 ? 0 : Math.sin(time / 600) * 0.08;
      gl.packets.forEach((packet, i) => {
        packet.visible = split || i === 0;
        packet.position.set(spots[i].x, spots[i].y + bob, 0);
        packet.scale.setScalar(split ? 0.62 : 1);
        if (!REDUCE) packet.rotation.y = time / 1400 + i;
      });

      // 1 plan
      const planning = p > 0.08 && p < 0.18;
      gl.plan.body.color.copy(planning ? RED : INK);
      gl.plan.slots.opacity = between(p, 0.10, 0.15) * 0.7;
      // 2 your review: red while it waits, then the iris opens
      const waiting = p > 0.22 && p < 0.30;
      gl.review.body.color.copy(waiting ? RED : INK);
      const open = smooth(between(p, 0.30, 0.34));
      gl.review.iris.rotation.y = open * 0.9;
      gl.review.iris.scale.setScalar(lerp(1, 0.02, open));
      // 3 lanes: a station is red while its packet is on it, bright once passed
      gl.stations.forEach((mats, i) => {
        STATIONS.forEach((y, k) => {
          const distance = spots[i].y - y;
          const on = split && Math.abs(distance) < 0.55;
          mats[k].color.copy(on ? RED : INK);
          mats[k].opacity = on ? 1 : split && distance < 0 ? 0.8 : 0.35;
        });
      });
      // 4 sieve, and the sentence it throws out
      const checking = p > 0.72 && p < 0.84;
      gl.sieve.body.color.copy(checking ? RED : INK);
      gl.sieve.mesh.opacity = checking ? 0.6 : 0.3;
      const out = between(p, 0.75, 0.86);
      // it is thrown forward, out of the lane, toward the viewer
      const sx = lerp(LANE_X[REJECTED_LANE], LANE_X[REJECTED_LANE] - 1.4, smooth(out));
      const sy = Y.sieve + 0.3 + Math.sin(out * Math.PI) * 1.2;
      const sz = 5 * smooth(out);
      gl.shard.visible = out > 0 && out < 1;
      gl.shard.position.set(sx, sy, sz);
      gl.shardLine.opacity = out > 0 && out < 1 ? 0.7 : 0;
      gl.shardPath.geometry.setFromPoints([new THREE.Vector3(LANE_X[REJECTED_LANE], Y.sieve + 0.3, 0), new THREE.Vector3(sx, sy, sz)]);
      // 5 tray
      gl.tray.forEach((material) => { material.color.copy(p >= 0.92 ? RED : INK); material.opacity = p >= 0.92 ? 1 : 0.65; });

      if (!document.hidden) gl.renderer.render(gl.scene, gl.camera);
    } else {
      $$("[data-dot]", flat).forEach((dot, i) => {
        dot.setAttribute("cx", spots[i].x);
        dot.setAttribute("cy", -spots[i].y);
      });
    }

    const tagOpacity = isMobile() ? 0 : heroOn ? 1 : 0.75;
    place(tagEl.plan, project(3.5, Y.plan, 2.5), tagOpacity, 8);
    place(tagEl.review, project(3.4, Y.review, 0), tagOpacity, 8);
    place(tagEl.lanes, project(4.6, Y.read, 0), tagOpacity, 8);
    place(tagEl.sieve, project(5.25, Y.sieve, 1.5), tagOpacity, 8);
    place(tagEl.tray, project(4.9, Y.tray, 1), tagOpacity, 8);
    tagEl.packet.textContent = split ? "" : topic;
    place(tagEl.packet, project(spots[0].x + 0.6, spots[0].y, 0), isMobile() || split ? 0 : 1, 6);

    if (heroOn) setStamp("");
    else if (p > 0.22 && p < 0.30) setStamp("Waiting for you", project(-3.2, Y.review - 1.4, 0));
    else if (p >= 0.30 && p < 0.37) setStamp("Outline approved", project(-3.2, Y.review - 1.4, 0));
    else if (p > 0.76 && p < 0.88) setStamp("1 sentence removed", project(LANE_X[REJECTED_LANE], Y.sieve + 2.3, 0));
    else if (p >= 0.92) setStamp("Report ready", project(0, Y.tray + 2.2, 0));
    else setStamp("");

    updateLog(p);
  }
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- the topic box and the two start forms
const signedIn = document.body.dataset.signedIn === "yes";

function setTopic(text) {
  topic = text.trim() || "your topic";
  $("#log-q").textContent = topic;
  $("#ask-q").value = text;
  $$(".chip").forEach((chip) => chip.classList.toggle("is-on", chip.dataset.q === text));
}

async function startReport(text, status, button) {
  text = text.trim();
  if (text.length < 4) { status.textContent = "Type a topic of at least 4 characters."; return; }
  if (!signedIn) {
    sessionStorage.setItem("pilot-topic", text);  // picked up on the new-report screen after sign-up
    location.href = "/auth?mode=signup";
    return;
  }
  button.disabled = true;
  status.textContent = "Planning the outline";
  const response = await fetch("/api/research", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ topic: text }),
  });
  if (response.ok) { location.href = "/research/" + (await response.json()).id; return; }
  const body = await response.json().catch(() => ({}));
  status.textContent = typeof body.detail === "string" ? body.detail : "That topic could not be started.";
  button.disabled = false;
}

$("#q").addEventListener("input", (e) => setTopic(e.target.value));
$$(".chip").forEach((chip) => chip.addEventListener("click", () => { $("#q").value = chip.dataset.q; setTopic(chip.dataset.q); }));
$("#qform").addEventListener("submit", (e) => {
  e.preventDefault();
  setTopic($("#q").value);
  $("#qstatus").textContent = "Scroll to follow it through the rig.";
  runEl.scrollIntoView({ behavior: REDUCE ? "auto" : "smooth" });
});
$("#askform").addEventListener("submit", (e) => {
  e.preventDefault();
  startReport($("#ask-q").value, $("#ask-status"), e.target.querySelector("button"));
});

// entrance of the hero copy, with Motion
if (M && !REDUCE) {
  document.body.classList.add("js-hero");
  M.animate("[data-hero]", { opacity: [0, 1], y: [14, 0] }, { duration: 0.6, delay: M.stagger(0.08, { startDelay: 0.1 }), easing: [0.22, 1, 0.36, 1] });
}

setTopic($("#q").value);
requestAnimationFrame(frame);
