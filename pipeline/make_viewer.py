"""Build interactive point-cloud pages (WebGL / three.js) so the clouds can be rotated by hand.

Each scene page shows one cloud at a time (GT lidar, one prediction or all predictions at once)
and can recolour the points either with their pixel colours or with the distance to the GT cloud
(green = on the surface, red = far / off).

Plotly's Scatter3d stops being usable above a few hundred thousand markers: the old pages showed
at most 100 000 points per cloud and the clouds therefore read as sparse voxels. The pages now
draw the points with three.js (one WebGL buffer per cloud): positions are quantised to int16 over
the cloud's own bounding box and the colour is a one-byte index into a 256-entry palette, so a
point costs 7 bytes. The payloads sit in data/ next to the page and are fetched the first time a
cloud is switched on, which keeps the HTML at a few kilobytes and lets one page carry millions of
points. Pages must be served over http (GitHub Pages does that); a file:// page is not allowed to
fetch the neighbouring binaries.
"""
import argparse
import base64
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import apply_sim3  # noqa: E402
from pose_align import centres, sim3 as pose_sim3  # noqa: E402
from common import (CLOUDS, GT, INFERENCE, OUTPUTS, crop_mask, diagonal,  # noqa: E402
                    load_crop, metric_text, percentile_box, read_json, read_ply,
                    stratified_subsample, view_basis)

VENDOR = Path(__file__).resolve().parent / "vendor"
# The prediction methods a page can show, in the order the buttons, the payload keys and the
# union view follow. Adding a method here (plus its label) is all it takes to widen the pages.
METHODS = ("da3-giant", "vggt-omega", "dust3r")
LABELS = {"da3-giant": "DA3-Giant", "vggt-omega": "VGGT-\u03a9", "dust3r": "DUSt3R"}
SHORTCUTS = {"da3-giant": "da3", "vggt-omega": "omega", "dust3r": "dust3r"}
# Page cloud keys of the predictions, in METHODS order; 'gt' is not a prediction.
PREDICTION_KEYS = tuple(SHORTCUTS.get(method, method) for method in METHODS)


def union_label(count):
    """Button text of the all-predictions view: 'оба' for two, 'все' from three on."""
    return "оба предсказания" if count == 2 else "все предсказания"


# Green to red, the same convention as the distance figures in the PDF.
DIST_SCALE = [
    [0.0, "#1a9850"], [0.25, "#91cf60"], [0.5, "#d9ef8b"],
    [0.75, "#fdae61"], [0.9, "#f46d43"], [1.0, "#d73027"],
]
RAMP_SIZE = 256
FOV = 45.0

CSS = """
html, body { margin: 0; background: #14171c; color: #e8e8e8; }
#gl { width: 100%; height: 86vh; overflow: hidden; position: relative; }
#gl canvas { display: block; }
.pick { background:#1e242c; color:#dfe5ec; border:1px solid #2e3742; border-radius:6px;
        padding:4px 11px; font:inherit; cursor:pointer; }
.pick:hover { background:#262e38; }
.pick.on { background:#2b6cb0; border-color:#3b82d0; color:#ffffff; font-weight:600; }
.pick:disabled { opacity:.35; cursor:not-allowed; }
.fail { padding:28px 24px; max-width:640px; color:#ffd08a; font-size:15px; line-height:1.5; }
#camshot { display:none; position:fixed; z-index:9; pointer-events:none; padding:6px 6px 4px;
           background:#0b0e12; border:1px solid #3a4450; border-radius:8px;
           box-shadow:0 12px 30px rgba(0,0,0,.6);
           font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif; }
#camshotimg { width:224px; height:168px; background-repeat:no-repeat; background-color:#0b0e12;
              border-radius:5px; }
#camshotcap { color:#c3cad3; font-size:11.5px; margin-top:5px; padding:0 2px;
              font-variant-numeric:tabular-nums; white-space:nowrap; }
#legend { display:none; position:fixed; left:14px; bottom:14px; z-index:6; pointer-events:none;
          background:rgba(20,23,28,.88); border:1px solid #2b313a; border-radius:8px;
          padding:8px 10px 7px; font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          box-shadow:0 8px 22px rgba(0,0,0,.45); max-width:360px; }
#legendtitle { color:#dfe5ec; font-size:12px; margin-bottom:6px; }
#legendbar { height:10px; border-radius:4px; border:1px solid #39424e; }
#legendticks { display:flex; justify-content:space-between; color:#a8b0ba; font-size:11px;
               margin-top:3px; font-variant-numeric:tabular-nums; }
#legendnote { color:#6b7480; font-size:10.5px; margin-top:3px; line-height:1.3; }
"""

HEADER = (
    "<div style='font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;"
    "padding:9px 16px 4px;background:#14171c;color:#e8e8e8;font-size:13.5px;line-height:1.35'>"
    "<div style='font-size:12px;margin-bottom:5px'><a href='index.html' "
    "style='color:#6cb6ff;text-decoration:none'>&larr; все сцены</a></div>"
    "<b>%s</b><br><span style='color:#a8b0ba'>%s</span></div>"
)

CONTROLS = """
<div id="bar" style="display:flex;flex-wrap:wrap;gap:14px;align-items:center;
     padding:6px 16px 10px;background:#14171c;color:#e8e8e8;
     font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;font-size:13px">
  <span style="color:#8f98a3">Облако:</span>
  __CLOUD_BUTTONS__
  <span style="width:1px;height:18px;background:#2b313a"></span>
  <span style="color:#8f98a3">Раскраска:</span>
  <button class="pick" data-colour="rgb">цвет пикселей</button>
  <button class="pick" data-colour="dist">__DIST_LABEL__</button>
  <span style="width:1px;height:18px;background:#2b313a"></span>
  <label style="cursor:pointer;color:#c3cad3"><input type="checkbox" id="cams"
     style="vertical-align:-2px;margin-right:4px">центры камер GT</label>
  <span style="color:#6b7480">наведите на пирамидку — покажется её кадр</span>
  <label style="cursor:pointer;color:#8f98a3">размер
     <input type="range" id="size" min="1" max="8" step="0.1"
            style="width:96px;vertical-align:-3px"></label>
  <span id="hint" style="color:#8f98a3"></span>
</div>
"""

PROGRESS = """
<div id="prog" style="display:none;position:absolute;left:50%;top:50%;
     transform:translate(-50%,-50%);z-index:5;flex-direction:column;align-items:center;gap:11px;
     pointer-events:none;background:rgba(20,23,28,.92);border:1px solid #2b313a;
     border-radius:10px;padding:15px 24px 17px;box-shadow:0 10px 30px rgba(0,0,0,.5);
     font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif">
  <span id="progtxt" style="color:#dfe5ec;font-size:13px;font-variant-numeric:tabular-nums;
        letter-spacing:.2px"></span>
  <span style="display:block;width:300px;height:8px;background:#262b33;
        border-radius:4px;overflow:hidden">
    <span id="progfill" style="display:block;width:0%;height:100%;background:#6cb6ff;
          transition:width .15s linear"></span>
  </span>
</div>
"""

# One tooltip for the whole page: it is positioned next to the cursor and shows the frame that
# belongs to the camera pyramid under it.
CAMSHOT = """
<div id="camshot"><div id="camshotimg"></div><div id="camshotcap"></div></div>
"""

# The distance colouring is meaningless without the scale it is drawn on: green means "on the
# surface", but how many metres that is depends on the scene, so the bar carries the actual tau.
LEGEND = """
<div id="legend">
  <div id="legendtitle"></div>
  <div id="legendbar"></div>
  <div id="legendticks"><span id="legendlo">0</span><span id="legendmid"></span>
    <span id="legendhi"></span></div>
  <div id="legendnote"></div>
</div>
"""

SCRIPT = r"""
<script>
(function () {
  var META = window.VIEWER;
  var host = document.getElementById('gl');
  var state = {cloud: META.default_cloud, colour: 'rgb', cams: META.cams !== false,
               size: META.point_size};
  var ready = false, dirty = true, benchMode = /[?&]bench/.test(location.search);

  // ---------------------------------------------------------------- URL hash
  // Everything the page shows is mirrored into the hash, so a view can be bookmarked
  // and deep-linked (and so headless renders can ask for a specific state):
  // #da3:dist:cams:2.5:@1.20,-0.30,4.10,9.80,0.60,1.25
  //   cloud : colour : camera centres : point size : @target xyz, radius, theta, phi
  function fromHash() {
    var parts = (location.hash || '').replace('#', '').split(':');
    if (parts[0] && (parts[0] === 'both' || META.traces[parts[0]])) { state.cloud = parts[0]; }
    if (parts[1] === 'rgb' || parts[1] === 'dist') { state.colour = parts[1]; }
    if (parts[2] === 'cams') { state.cams = true; }
    if (parts[2] === 'nocams') { state.cams = false; }
    var size = parseFloat(parts[3]);
    if (isFinite(size) && size > 0) { state.size = size; }
    for (var i = 4; i < parts.length; i++) {
      if (parts[i].charAt(0) !== '@' || !control) { continue; }
      var n = parts[i].slice(1).split(',').map(Number);
      if (n.length < 6 || !n.every(function (v) { return isFinite(v) && Math.abs(v) < 1e9; })) {
        continue;
      }
      control.target.set(n[0], n[1], n[2]);
      control.radius = n[3] > 0 ? n[3] : control.radius;
      control.theta = n[4]; control.phi = n[5];
    }
  }
  function toHash() {
    var tail = (state.cams ? ':cams' : ':nocams') + ':' + state.size.toFixed(1);
    if (control) {
      // the point we orbit around is part of the view: share or reload it, not just the rotation
      tail += ':@' + [control.target.x, control.target.y, control.target.z, control.radius,
                      control.theta, control.phi]
        .map(function (v) { return v.toFixed(3); }).join(',');
    }
    try { history.replaceState(null, '', '#' + state.cloud + ':' + state.colour + tail); } catch (e) {}
  }

  // ------------------------------------------------------------ base64 / bytes
  function bytes(b64) {
    var s = atob(b64), out = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) { out[i] = s.charCodeAt(i); }
    return out;
  }
  function floats(b64) {
    var raw = bytes(b64);
    return new Float32Array(raw.buffer, 0, raw.length >> 2);
  }
  function texture(palette, size) {
    var data = new Uint8Array(size * 4);
    for (var i = 0; i < size; i++) {
      data[4 * i] = palette[3 * i];
      data[4 * i + 1] = palette[3 * i + 1];
      data[4 * i + 2] = palette[3 * i + 2];
      data[4 * i + 3] = 255;
    }
    var made = new THREE.DataTexture(data, size, 1, THREE.RGBAFormat);
    made.magFilter = THREE.NearestFilter;
    made.minFilter = THREE.NearestFilter;
    made.wrapS = THREE.ClampToEdgeWrapping;
    made.wrapT = THREE.ClampToEdgeWrapping;
    made.generateMipmaps = false;
    made.needsUpdate = true;
    return made;
  }
  var RAMP = texture(bytes(META.ramp), META.ramp_size);

  // ---------------------------------------------------------------- renderer
  var renderer = null;
  try { renderer = new THREE.WebGLRenderer({antialias: false}); } catch (error) { renderer = null; }
  if (!renderer || !renderer.getContext()) {
    host.innerHTML = "<div class='fail'>Эти страницы рисуют облака через WebGL, а браузер его не дал." +
      " Обычно помогает включить аппаратное ускорение или открыть страницу в Chrome или Firefox.</div>";
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x14171c, 1);
  host.appendChild(renderer.domElement);
  var scene = new THREE.Scene();
  var root = new THREE.Group();
  root.applyMatrix4(new THREE.Matrix4().fromArray(META.basis));
  scene.add(root);
  var camera = new THREE.PerspectiveCamera(META.fov, 1, 0.001, 1e6);

  // Point size is a pixel size at REF_DIST, the distance of the opening view, not at whatever
  // distance the orbit happens to be at, and it grows as 1/distance from there. That law is the
  // one that keeps a surface closed: the spacing between neighbouring points on screen also grows
  // as 1/distance, so their ratio -- how much of the gap a dot covers -- stays exactly what it was
  // in the opening view. An exponent below 1 (the page used 0.7) grows the dots slower than the
  // gaps and a close-up dissolves into a lattice of dots with black between them; capping the size
  // at 26 px did the same beyond a few metres of zoom. The floor still keeps distant points
  // visible in a wide shot, and the ceiling only stops a single point from covering the screen
  // when the camera is inside the cloud.
  var REF_DIST = META.camera.radius;
  var MIN_POINT_PX = 1.2;   // floor in CSS pixels: distant points stay visible in a wide shot
  var MAX_POINT_PX = 160.0; // ceiling in CSS pixels: one voxel may fill a close-up, not the screen
  var VERTEX = [
    'uniform vec3 uCentre;', 'uniform vec3 uHalf;', 'uniform float uSize;',
    'uniform float uPixel;', 'uniform float uRefDist;',
    'uniform float uMinPx;', 'uniform float uMaxPx;',
    'attribute float aIndex;', 'attribute float aDist;',
    'varying float vIndex;', 'varying float vDist;',
    'void main() {',
    '  vIndex = aIndex; vDist = aDist / 255.0;',
    '  vec4 view = modelViewMatrix * vec4(position * uHalf + uCentre, 1.0);',
    '  gl_Position = projectionMatrix * view;',
    '  float atten = clamp(uRefDist / max(-view.z, 1e-9), 0.02, 4000.0);',
    '  float px = uSize * uPixel * atten;',
    '  gl_PointSize = clamp(px, uMinPx * uPixel, uMaxPx * uPixel);',
    '}'
  ].join('\n');

  var FRAGMENT = [
    'uniform sampler2D uPalette;', 'uniform sampler2D uRamp;',
    'uniform float uPalCount;', 'uniform float uMode;',
    'varying float vIndex;', 'varying float vDist;',
    'void main() {',
    '  vec2 off = gl_PointCoord - vec2(0.5);',
    '  if (dot(off, off) > 0.25) { discard; }',
    '  vec4 col = (uMode < 0.5)',
    '    ? texture2D(uPalette, vec2((vIndex + 0.5) / uPalCount, 0.5))',
    '    : texture2D(uRamp, vec2((vDist * 255.0 + 0.5) / 256.0, 0.5));',
    '  gl_FragColor = vec4(col.rgb, 1.0);',
    '}'
  ].join('\n');

  function materialFor(spec) {
    return new THREE.ShaderMaterial({
      uniforms: {
        uCentre: {value: new THREE.Vector3().fromArray(spec.centre)},
        uHalf: {value: new THREE.Vector3().fromArray(spec.half)},
        uSize: {value: state.size},
        uPixel: {value: renderer.getPixelRatio()},
        uRefDist: {value: REF_DIST},
        uMinPx: {value: MIN_POINT_PX},
        uMaxPx: {value: MAX_POINT_PX},
        uPalette: {value: texture(bytes(spec.pal), spec.pal_count)},
        uRamp: {value: RAMP},
        uPalCount: {value: spec.pal_count},
        uMode: {value: 0}
      },
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT
    });
  }

  // ------------------------------------------------------------- camera orbit
  var control = {
    target: new THREE.Vector3().fromArray(META.camera.target),
    radius: META.camera.radius, theta: META.camera.theta, phi: META.camera.phi
  };
  function applyCamera() {
    control.phi = Math.max(0.08, Math.min(Math.PI - 0.08, control.phi));
    var s = control, sp = Math.sin(s.phi);
    camera.position.set(
      s.target.x + s.radius * sp * Math.sin(s.theta),
      s.target.y + s.radius * Math.cos(s.phi),
      s.target.z + s.radius * sp * Math.cos(s.theta));
    camera.up.set(0, 1, 0);
    camera.lookAt(s.target);
    camera.near = Math.max(s.radius * 0.002, 1e-5);
    // The far plane follows the scene, not the orbit radius: zoomed into a table the old rule
    // (radius * 40) cut off the wall two metres behind it and the background simply disappeared.
    camera.far = Math.max(s.radius * 40.0, REF_DIST * 40.0);
    camera.updateProjectionMatrix();
    dirty = true;
  }
  function forEachObject(visit) {
    for (var key in META.traces) {
      if (META.traces[key].object) { visit(META.traces[key].object); }
    }
  }
  var drag = null;
  // Pan is asked for in several idioms: right or middle drag, or any modifier with the left button
  // (Shift as in three.js, Ctrl as in Blender / most desktop viewers, plus Alt and Cmd so the
  // gesture works whatever the user's muscle memory is).
  function wantsPan(event) {
    return event.button === 2 || event.button === 1 ||
           event.shiftKey || event.ctrlKey || event.metaKey || event.altKey;
  }
  host.addEventListener('contextmenu', function (event) { event.preventDefault(); });
  host.addEventListener('pointerdown', function (event) {
    drag = {x: event.clientX, y: event.clientY, pan: wantsPan(event)};
    try { host.setPointerCapture(event.pointerId); } catch (e) {}
  });
  host.addEventListener('pointerup', function (event) {
    drag = null;
    try { host.releasePointerCapture(event.pointerId); } catch (e) {}
    toHash();
  });
  host.addEventListener('pointermove', function (event) {
    if (!drag) {
      // idle hover: is the cursor on a camera pyramid?
      var over = cameraUnder(event);
      showCamShot(over, event.clientX, event.clientY);
      host.style.cursor = over === null ? 'grab' : 'help';
      return;
    }
    hideCamShot();
    var dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    drag.x = event.clientX; drag.y = event.clientY;
    if (drag.pan || event.shiftKey || event.ctrlKey || event.metaKey || event.altKey) {
      // pan: move the orbit target in the plane of the screen
      var scale = 2.0 * control.radius * Math.tan(META.fov * Math.PI / 360.0) / host.clientHeight;
      var right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      var up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      control.target.addScaledVector(right, -dx * scale).addScaledVector(up, dy * scale);
    } else {
      control.theta -= dx * 0.006;
      control.phi -= dy * 0.006;
    }
    applyCamera();
  });
  host.addEventListener('wheel', function (event) {
    event.preventDefault();
    control.radius = Math.max(control.radius * Math.exp(event.deltaY * 0.0012), 1e-4);
    applyCamera();
  }, {passive: false});

  // ------------------------------------------------------------------- progress
  // A determinate bar over the bytes still to come. The size of every payload is known from the
  // trace count (7 bytes per point, 1 byte per distance), so the bar never depends on
  // Content-Length surviving compression, and it measures what actually arrives rather than
  // what was promised.
  var prog = {
    total: 0, done: 0, phase: '', el: document.getElementById('prog'),
    fill: document.getElementById('progfill'), text: document.getElementById('progtxt')
  };
  function progPaint() {
    var share = prog.total > 0 ? Math.min(1, prog.done / prog.total) : 1;
    prog.fill.style.width = (100 * share).toFixed(1) + '%';
    prog.text.textContent = prog.phase + ' ' + Math.floor(100 * share) + ' %';
  }
  function progStart(total, phase) {
    if (!prog.el) { return false; }
    prog.total = Math.max(total, 1); prog.done = 0; prog.phase = phase;
    prog.el.style.display = 'flex';
    progPaint();
    return true;
  }
  function progAdd(delta) {
    if (!prog.el || prog.el.style.display === 'none') { return; }
    prog.done = Math.min(prog.done + delta, prog.total);
    progPaint();
  }
  function progFinish() {
    if (!prog.el || prog.el.style.display === 'none') { return; }
    prog.done = prog.total;
    progPaint();
    setTimeout(function () { prog.el.style.display = 'none'; }, 350);
  }

  // fetch() into an ArrayBuffer while reporting bytes as they arrive. Without a readable body the
  // download still happens, the bar just jumps to the end in one step.
  function noBytes() {}
  function fetchBuffer(url, expected, onBytes) {
    return fetch(url).then(function (response) {
      if (!response.ok) { throw new Error(response.status + ' на ' + url); }
      if (!response.body || !response.body.getReader) {
        return response.arrayBuffer().then(function (buffer) { onBytes(expected); return buffer; });
      }
      var reader = response.body.getReader(), chunks = [], got = 0;
      function pump() {
        return reader.read().then(function (step) {
          if (step.done) {
            var out = new Uint8Array(got), at = 0;
            for (var i = 0; i < chunks.length; i++) { out.set(chunks[i], at); at += chunks[i].length; }
            return out.buffer;
          }
          chunks.push(step.value);
          got += step.value.length;
          onBytes(step.value.length);
          return pump();
        });
      }
      return pump();
    });
  }

  // ------------------------------------------------------------- payload load
  // A cloud arrives in chunks, so the first one paints the scene and the rest grow the draw
  // range behind it; a 10.9 M point cloud is 76 MB and waiting for all of it before the first
  // frame would make the page feel broken. GitHub Pages hands one connection roughly 130 kB/s but
  // scales with the number of connections, so four files are in flight at any moment; every chunk
  // has its own fixed slot in the vertex arrays, so arrival order does not matter and the draw
  // range just follows the unbroken prefix.
  function bumpRange(attribute, itemSize, first, count) {
    if (attribute.updateRange) {   // three.js uploads only this slice when the driver supports it
      attribute.updateRange.offset = first * itemSize;
      attribute.updateRange.count = count * itemSize;
    }
    attribute.needsUpdate = true;
  }
  var FETCH_SLOTS = 4, urgentTasks = [], fetchTasks = [], fetchActive = 0, fetchSeq = 0;
  // Two lanes. Chunk zero of every cloud comes first, in the order the page asked for the clouds, so
  // the all-predictions view paints every cloud on parallel requests instead of one cloud whole. The
  // rest is ordered by depth: chunk 1 of every cloud, then chunk 2 of every cloud, and so on, so the
  // clouds on screen grow together instead of the last one sitting at 1 M points.
  function queued(task, rank) {
    return new Promise(function (resolve, reject) {
      var job = {task: task, rank: rank, resolve: resolve, reject: reject, seq: fetchSeq++};
      if (rank === 0) {
        urgentTasks.push(job);
      } else {
        var at = fetchTasks.length;
        while (at > 0 && fetchTasks[at - 1].rank > rank) { at--; }
        fetchTasks.splice(at, 0, job);
      }
      pumpQueue();
    });
  }
  function pumpQueue() {
    while (fetchActive < FETCH_SLOTS && (urgentTasks.length > 0 || fetchTasks.length > 0)) {
      var job = urgentTasks.length > 0 ? urgentTasks.shift() : fetchTasks.shift();
      fetchActive++;
      job.task().then(job.resolve, job.reject).then(function () {
        fetchActive--;
        pumpQueue();
      });
    }
  }
  function loadPositions(key) {
    var spec = META.traces[key];
    if (!spec) { return Promise.resolve(null); }
    if (spec.object) { return Promise.resolve(spec.object); }
    if (spec.pending) { return spec.pending; }
    var started = performance.now();
    var n = spec.count, chunks = spec.chunks || [{pos: spec.pos, count: n}];
    var positions = new Int16Array(3 * n), indices = new Uint8Array(n);
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3, true));
    geo.setAttribute('aIndex', new THREE.BufferAttribute(indices, 1, false));
    geo.setAttribute('aDist', new THREE.BufferAttribute(new Uint8Array(n), 1, false));
    geo.setDrawRange(0, 0);
    var radius = Math.sqrt(spec.half[0] * spec.half[0] + spec.half[1] * spec.half[1]
                           + spec.half[2] * spec.half[2]);
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3().fromArray(spec.centre), radius * 1.001);
    var object = new THREE.Points(geo, materialFor(spec));
    object.material.uniforms.uRefDist.value = REF_DIST;
    object.visible = false;
    root.add(object);
    spec.object = object;
    spec.loaded = 0;
    var resolveFirst = null;
    spec.pending = new Promise(function (resolve) { resolveFirst = resolve; });
    var offsets = [], at = 0;
    for (var s = 0; s < chunks.length; s++) { offsets.push(at); at += chunks[s].count; }
    var landed = new Array(chunks.length), prefix = 0, prefixPoints = 0;
    for (var d = 0; d < chunks.length; d++) { landed[d] = false; }
    spec.prefix = 0;
    spec.stream = Promise.all(chunks.map(function (chunk, index) {
      return queued(function () {
        return fetchBuffer(chunk.pos, 7 * chunk.count, progAdd).then(function (buffer) {
          if (buffer.byteLength !== 7 * chunk.count) { throw new Error('битый файл ' + chunk.pos); }
          var slot = offsets[index];
          positions.set(new Int16Array(buffer, 0, 3 * chunk.count), 3 * slot);
          indices.set(new Uint8Array(buffer, 6 * chunk.count, chunk.count), slot);
          bumpRange(geo.attributes.position, 3, slot, chunk.count);
          bumpRange(geo.attributes.aIndex, 1, slot, chunk.count);
          landed[index] = true;
          // Only the unbroken run from chunk zero may be drawn: growing the range over a hole would
          // put a block of uninitialised vertices on screen.
          while (prefix < landed.length && landed[prefix]) {
            prefixPoints += chunks[prefix].count;
            prefix++;
          }
          spec.prefix = prefix;
          spec.loaded = prefixPoints;
          geo.setDrawRange(0, prefixPoints);
          dirty = true;
          if (prefix === landed.length) { spec.decode_ms = Math.round(performance.now() - started); }
          if (resolveFirst && prefixPoints > 0) { resolveFirst(object); resolveFirst = null; }
        });
      }, index);
    }));
    spec.stream.catch(function () {
      if (resolveFirst) { resolveFirst(object); resolveFirst = null; }
    });
    return spec.pending;
  }
  // The distance buffers are a third of the payload and are only meaningful with the
  // "расстояние до GT" colouring on, so they are fetched on the first click that needs them.
  function loadDistances(key) {
    var spec = META.traces[key];
    if (!spec || !spec.dist || spec.dist_done || spec.dist_failed) { return Promise.resolve(); }
    if (spec.dist_pending) { return spec.dist_pending; }
    spec.dist_pending = queued(function () {
      return fetchBuffer(spec.dist, spec.count, noBytes);   // the bar is gone by the time these land
    }, 9999).then(function (buffer) {
      var values = new Uint8Array(buffer);
      if (values.length !== spec.count) { throw new Error('битый файл ' + spec.dist); }
      spec.object.geometry.setAttribute('aDist', new THREE.BufferAttribute(values, 1, false));
      spec.dist_done = true;
      dirty = true;
    }).catch(function (error) {
      spec.dist_failed = true;
      throw error;
    });
    return spec.dist_pending;
  }
  // One call per trace for "give me the geometry of this cloud".
  function loadAll(key) {
    var spec = META.traces[key];
    return loadPositions(key).then(function () {
      return spec.stream || null;
    });
  }

  // ----------------------------------------------------------------- drawing
  // One pyramid per GT camera rather than a single merged wireframe: a pyramid can then be picked
  // with the mouse, which is what makes "по наведению картинка из этой пирамидки" possible.
  var camGroup = null, raycaster = new THREE.Raycaster();
  if (META.frustums) {
    camGroup = new THREE.Group();
    var raw = floats(META.frustums.b64), per = META.frustums.per;
    var apexOf = function (slice) {
      var point = new THREE.Vector3();
      for (var v = 0; v < slice.length; v += 3) {
        point.x += slice[v]; point.y += slice[v + 1]; point.z += slice[v + 2];
      }
      return point.multiplyScalar(3 / slice.length);
    };
    // A pyramid seen head-on is a handful of thin lines, so an invisible sphere sits inside each
    // one: the cursor lands on it from any angle, the drawing stays a wireframe.
    var hitGeometry = new THREE.SphereGeometry(1, 10, 8);
    var hitMaterial = new THREE.MeshBasicMaterial({transparent: true, opacity: 0, depthWrite: false});
    // A hundred pyramids in one place turn into an orange tangle over the cloud, so the denser the
    // camera path, the fainter the wireframe: hovering still works, reading the cloud still works.
    var cameras = raw.length / per;
    var wireOpacity = cameras > 40 ? 0.35 : (cameras > 18 ? 0.55 : 0.85);
    for (var c = 0; c < raw.length / per; c++) {
      var slice = raw.subarray(c * per, (c + 1) * per);
      var geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(slice.slice(), 3));
      // The cameras sit inside the cloud they photographed, so the trail is drawn through it.
      var wire = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial(
        {color: 0xffb020, depthTest: false, transparent: true, opacity: wireOpacity}));
      wire.renderOrder = 1;
      wire.userData.cam = c;
      camGroup.add(wire);
      var hit = new THREE.Mesh(hitGeometry, hitMaterial);
      hit.position.copy(apexOf(slice));
      hit.scale.setScalar(META.frustums.edge * 0.9);
      hit.userData.cam = c;
      camGroup.add(hit);
    }
    camGroup.visible = state.cams;
    root.add(camGroup);
    raycaster.params.Line = {threshold: META.frustums.edge * 0.12};
  }

  // --------------------------------------------------------------- frame hover
  var camTip = document.getElementById('camshot'), camTipImg = document.getElementById('camshotimg'),
      camTipCap = document.getElementById('camshotcap'), camShown = null;
  function hideCamShot() {
    if (camTip && camShown !== null) { camTip.style.display = 'none'; camShown = null; }
  }
  function showCamShot(index, x, y) {
    var sheet = META.camshots;
    if (!camTip) { return; }
    if (!sheet || index === null || !sheet.tile[index]) { camShown = null; camTip.style.display = 'none'; return; }
    if (camShown !== index) {
      var cell = sheet.tile[index];
      camTipImg.style.backgroundImage = "url('" + sheet.url + "')";
      camTipImg.style.backgroundSize = (sheet.cols * 224) + 'px ' + (sheet.rows * 168) + 'px';
      camTipImg.style.backgroundPosition = (-cell[0] * 224) + 'px ' + (-cell[1] * 168) + 'px';
      camTipCap.textContent = 'кадр ' + (index + 1) + ' / ' + sheet.tile.length + ' · ' + sheet.names[index];
      camShown = index;
    }
    camTip.style.display = 'block';
    var width = camTip.offsetWidth, height = camTip.offsetHeight;
    camTip.style.left = Math.min(x + 18, window.innerWidth - width - 8) + 'px';
    camTip.style.top = Math.min(Math.max(y - height / 2, 8), window.innerHeight - height - 8) + 'px';
  }
  function cameraUnder(event) {
    if (!camGroup || !camGroup.visible || !META.camshots) { return null; }
    var box = host.getBoundingClientRect();
    if (box.width <= 0 || box.height <= 0) { return null; }
    var ndc = new THREE.Vector2(((event.clientX - box.left) / box.width) * 2 - 1,
                               -((event.clientY - box.top) / box.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    scene.updateMatrixWorld();
    var hits = raycaster.intersectObjects(camGroup.children, false);
    for (var i = 0; i < hits.length; i++) {
      var cam = hits[i].object.userData.cam;
      if (cam !== undefined) { return cam; }
    }
    return null;
  }

  function chosen() {
    if (state.cloud === 'both') { return META.predictions.slice(); }
    return [state.cloud];
  }
  function hint(text) { document.getElementById('hint').textContent = text; }

  // -------------------------------------------------------------- colour scale
  // Drawn only for the distance colouring: green/red only mean something together with the number
  // that separates them, and that number is per scene (1 % of the GT diagonal).
  var legend = document.getElementById('legend'), legendBuilt = false;
  // The clouds are metric, so the bar reads in metres and drops to centimetres where tau is small
  // enough that "0.07 м" would be a mouthful.
  function metric(value) {
    if (value >= 10) { return value.toFixed(1) + " м"; }
    if (value >= 0.1) { return value.toFixed(2) + " м"; }
    return (value * 100).toFixed(1) + " см";
  }
  function paintLegend() {
    if (!legend) { return; }
    var sheet = META.legend || {};
    var on = state.colour === 'dist';
    legend.style.display = on ? 'block' : 'none';
    if (!on) { return; }
    var title = state.cloud === 'gt' && sheet.gt ? sheet.gt : sheet.pred;
    document.getElementById('legendtitle').textContent = title || 'расстояние';
    if (!legendBuilt) {
      document.getElementById('legendbar').style.background = META.ramp_css;
      document.getElementById('legendmid').textContent = 'τ/2 = ' + metric(META.tau / 2);
      document.getElementById('legendhi').textContent = 'τ = ' + metric(META.tau);
      document.getElementById('legendnote').textContent = sheet.note || '';
      legendBuilt = true;
    }
  }
  function syncButtons() {
    var buttons = document.querySelectorAll('.pick');
    for (var i = 0; i < buttons.length; i++) {
      var button = buttons[i];
      button.classList.toggle('on', button.dataset.cloud
        ? state.cloud === button.dataset.cloud : state.colour === button.dataset.colour);
    }
    document.getElementById('cams').checked = state.cams;
    document.getElementById('size').value = state.size;
  }
  // Everything the page has is fetched once, up front, so switching GT / a prediction / all
  // predictions or turning the distance colour on is a visibility toggle: the whole payload is only a
  // few megabytes, which is a far better trade than a network round trip per click.
  function payloadKeys() {
    var keys = [];
    for (var key in META.traces) {
      if (META.traces[key]) { keys.push(key); }
    }
    return keys;
  }
  var preload = null;
  function preloadAll() {
    var keys = payloadKeys(), expected = 0;
    for (var i = 0; i < keys.length; i++) {
      var spec = META.traces[keys[i]];
      if (!spec.object) { expected += 7 * spec.count; }
    }
    var bar = expected > 0 && progStart(expected, 'загрузка облаков…');
    // Order of business: the clouds the page opens with, the GT the colours are measured against,
    // then the rest. Everything is started at once because the scheduler keeps four requests in
    // flight and serves the first chunk of each cloud in exactly this order, so a page with three
    // clouds paints after three parallel requests instead of three serial ones.
    var defaults = chosen();
    var gt = (defaults.indexOf('gt') < 0 && META.traces.gt) ? ['gt'] : [];
    var others = keys.filter(function (key) {
      return defaults.indexOf(key) < 0 && gt.indexOf(key) < 0;
    });
    var order = defaults.concat(gt, others), streams = [];
    for (var k = 0; k < order.length; k++) {
      var key = order[k], trace = META.traces[key];
      if (!trace) { continue; }
      loadPositions(key).then(sync).catch(function () {});
      // loadPositions() has created spec.stream synchronously, so the handles are all here.
      if (trace.stream) { streams.push(trace.stream); }
    }
    // Distance bytes ride at the very back of the queue: the colouring is one click away and nobody
    // should wait for it to click it, but a click should not start a fresh download either.
    for (var m = 0; m < order.length; m++) {
      if (META.traces[order[m]]) { loadDistances(order[m]).catch(function () {}); }
    }
    // The bar belongs to the selected clouds plus the GT: with those on screen the page is usable
    // and the remaining clouds are one click away, so they finish behind a hidden bar instead of
    // holding the user in front of a progress counter.
    var gate = streams.slice(0, defaults.length + gt.length);
    preload = Promise.all(gate).then(function () {
      if (bar) { progFinish(); }
      return sync();
    }).then(function () {
      return Promise.all(streams);
    }).then(function () {
      return sync();
    }).catch(function (error) {
      if (bar) { prog.el.style.display = 'none'; }
      hint('не загрузилось: ' + error.message);
    });
    return preload;
  }
  // apply() only waits for payloads that are still missing (an error retry, or a click while the
  // preload is in flight); normally it resolves immediately and just redraws.
  function apply() {
    var keys = chosen(), loading = [];
    for (var i = 0; i < keys.length; i++) {
      if (!META.traces[keys[i]]) { continue; }
      loading.push(loadAll(keys[i]));
    }
    Promise.all(loading).then(sync).catch(function (error) {
      hint('не загрузилось: ' + error.message);
    });
  }
  // sync() is the pure "show what state says" half: no network, no bar.
  function sync() {
    var keys = chosen(), shown = 0, wanted = 0, waiting = false;
    for (var key in META.traces) {
      var spec = META.traces[key];
      if (!spec.object) { continue; }
      var on = keys.indexOf(key) >= 0;
      var have = (spec.loaded === undefined) ? spec.count : spec.loaded;
      spec.object.visible = on && have > 0;
      spec.object.material.uniforms.uSize.value = state.size;
      spec.object.material.uniforms.uMode.value = (state.colour === 'dist' && spec.dist_done) ? 1 : 0;
      if (on) {
        shown += have; wanted += spec.count;
        if (state.colour === 'dist' && spec.dist && !spec.dist_done && !spec.dist_failed && have > 0) {
          waiting = true;
          loadDistances(key).then(function () { dirty = true; sync(); }).catch(function () {});
        }
      }
    }
    if (camGroup) { camGroup.visible = state.cams; }
    if (!state.cams) { hideCamShot(); }
    paintLegend();
    hint((state.colour === 'dist' ? META.hint_dist : META.hint_rgb)
         + ' · ' + shown.toLocaleString('ru-RU') + ' точек'
         + (shown < wanted
            ? ' — догружается ' + wanted.toLocaleString('ru-RU')
            : ' · всё облако в памяти')
         + (waiting ? ' · гружу расстояния' : ''));
    syncButtons();
    toHash();
    dirty = true;
    if (!ready) { ready = true; firstReady(); }
  }
  document.getElementById('cams').addEventListener('change', function () {
    state.cams = this.checked; apply();
  });
  host.addEventListener('pointerleave', hideCamShot);
  document.getElementById('size').addEventListener('input', function () {
    state.size = parseFloat(this.value);
    forEachObject(function (obj) { obj.material.uniforms.uSize.value = state.size; });
    dirty = true;
    toHash();
  });
  var picks = document.querySelectorAll('.pick');
  for (var p = 0; p < picks.length; p++) {
    picks[p].addEventListener('click', function () {
      if (this.disabled) { return; }
      if (this.dataset.cloud) { state.cloud = this.dataset.cloud; }
      else { state.colour = this.dataset.colour; }
      apply();
    });
  }

  // ------------------------------------------------------------------- frame
  function render() { renderer.render(scene, camera); }
  function loop() {
    requestAnimationFrame(loop);
    if (dirty && !benchMode) { dirty = false; render(); }
  }
  function resize() {
    var width = host.clientWidth || 900, height = host.clientHeight || 600;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    forEachObject(function (obj) { obj.material.uniforms.uPixel.value = renderer.getPixelRatio(); });
    dirty = true;
  }
  window.addEventListener('resize', resize);

  // ------------------------------------------------------------------- bench
  function rendererName() {
    var gl = renderer.getContext();
    var info = gl.getExtension('WEBGL_debug_renderer_info');
    return info ? String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL))
                : String(gl.getParameter(gl.RENDERER));
  }
  function firstReady() {
    if (!benchMode || window.__BENCH_DONE) { return; }
    window.__BENCH_DONE = true;
    var stats = {webgl: rendererName(), decode_ms: {}, points: {}, first_paint_ms: null};
    for (var key in META.traces) {
      var spec = META.traces[key];
      if (!spec.object) { continue; }
      stats.decode_ms[key] = spec.decode_ms;
      stats.points[key] = spec.count;
    }
    requestAnimationFrame(function () {
      render();
      stats.first_paint_ms = Math.round(performance.now());
      var frames = [], index = 0, previous = performance.now();
      function step() {
        var now = performance.now();
        if (index >= 4) { frames.push(now - previous); }
        previous = now;
        control.theta += 0.012;
        applyCamera();
        render();
        if (++index < 124) { requestAnimationFrame(step); return; }
        frames.sort(function (a, b) { return a - b; });
        stats.frame_median_ms = Math.round(frames[Math.floor(frames.length / 2)] * 10) / 10;
        stats.frame_p95_ms = Math.round(frames[Math.floor(frames.length * 0.95)] * 10) / 10;
        stats.frame_count = frames.length;
        window.__VIEWER_STATS = stats;
        try { fetch('/__report', {method: 'POST', body: JSON.stringify(stats)}); } catch (e) {}
      }
      requestAnimationFrame(step);
    });
  }

  // -------------------------------------------------------------------- init
  fromHash();
  resize();
  applyCamera();
  preloadAll();
  // The camera contact sheet is small next to the clouds, so fetch it up front: the first hover
  // then shows a picture instead of an empty frame while the JPEG is still on its way.
  if (META.camshots) { var warm = new Image(); warm.src = META.camshots.url; }
  loop();
})();
</script>
"""


# ----------------------------------------------------------------- payload bits
def ramp_table(stops, size=RAMP_SIZE):
    """The distance colourscale as a 256-entry lookup table, so the shader can sample it."""
    positions = np.array([stop[0] for stop in stops], dtype=np.float64)
    rgb = np.array([[int(stop[1][channel:channel + 2], 16) for channel in (1, 3, 5)]
                    for stop in stops], dtype=np.float64)
    xs = np.linspace(0.0, 1.0, size)
    return np.stack([np.interp(xs, positions, rgb[:, channel]) for channel in range(3)],
                    axis=1).round().astype(np.uint8)


def quantise(points):
    """int16 grid over the cloud's own bounding box: 6 bytes per point."""
    points = np.asarray(points, dtype=np.float64)
    low, high = points.min(axis=0), points.max(axis=0)
    centre = 0.5 * (low + high)
    half = np.maximum(0.5 * (high - low), 1e-9)
    scaled = np.clip((points - centre) / half, -1.0, 1.0) * 32767.0
    return np.rint(scaled).astype(np.int16), centre, half


def palette_index(colours, count):
    """One palette byte per point instead of a per-point colour string."""
    if colours is None:
        palette = np.array([[0x9a, 0xa4, 0xb0]], dtype=np.uint8)
        return np.zeros(count, dtype=np.uint8), palette, 0.0

    pixels = np.ascontiguousarray(np.asarray(colours, dtype=np.uint8)[:, None, :])
    image = Image.fromarray(pixels, "RGB")
    dither = getattr(getattr(Image, "Dither", Image), "NONE", 0)
    quantised = image.quantize(colors=256, method=Image.MEDIANCUT, dither=dither)
    index = np.asarray(quantised, dtype=np.uint8).reshape(-1)
    used = int(index.max()) + 1 if index.size else 1
    palette = np.asarray(quantised.getpalette()[:3 * used], dtype=np.uint8).reshape(-1, 3).copy()
    error = float(np.abs(palette[index].astype(np.int16)
                         - np.asarray(colours, dtype=np.uint8)).mean())
    return index, palette, error


# Points per payload file. A whole 100-frame cloud is 10.9 M points = 76 MB, which is one long
# download before anything appears on screen; split into chunks the first one paints the scene in
# a second and the rest streams in behind it (the page grows the draw range as chunks land).
# The first chunk is deliberately small: GitHub Pages hands one connection about 130 kB/s, so a
# 1 M point chunk is 7 MB of nothing on screen for a minute. A staircase that doubles up to
# CHUNK_POINTS puts the first frame about two seconds in and keeps the request count for a
# 10.9 M point cloud at a dozen or so.
CHUNK_POINTS = 1000000
FIRST_CHUNK_POINTS = 40000


def chunk_sizes(total):
    """Point counts per payload file: 40 k, 40 k, 80 k, 160 k … capped at CHUNK_POINTS."""
    sizes, done, step = [], 0, FIRST_CHUNK_POINTS
    while done < total:
        take = min(step, total - done)
        sizes.append(take)
        done += take
        step = min(step * 2, CHUNK_POINTS)
    return sizes


def write_payloads(out, stem, key, points, colours, distances, tau):
    """Write data/<page>-<key>-<n>.bin chunks (7 bytes/point) and the distance byte per point.

    One int16 grid over the whole cloud is shared by every chunk, so the chunks of a cloud cannot
    disagree about its bounding box, and the point order inside a chunk is the order of the
    arrays, i.e. positions and colours always stay paired.
    """
    quantised, centre, half = quantise(points)
    index, palette, error = palette_index(colours, len(points))
    folder = Path(out, "data")
    folder.mkdir(parents=True, exist_ok=True)
    chunks = []
    size = 0
    start = 0
    for take in chunk_sizes(len(points)):
        stop = start + take
        path = folder / ("%s-%s-%03d.bin" % (stem, key, len(chunks)))
        with path.open("wb") as handle:
            handle.write(np.ascontiguousarray(quantised[start:stop], dtype="<i2").tobytes())
            handle.write(np.ascontiguousarray(index[start:stop], dtype=np.uint8).tobytes())
        chunks.append(dict(pos="data/" + path.name, count=int(take)))
        size += path.stat().st_size
        start = stop
    distance_path = None
    if distances is not None:
        unit = np.clip(np.asarray(distances, dtype=np.float64) / tau, 0.0, 1.0)
        distance_path = folder / ("%s-%s-dist.bin" % (stem, key))
        distance_path.write_bytes(np.rint(unit * 255.0).astype(np.uint8).tobytes())
    spec = dict(
        count=int(len(points)),
        centre=[float(value) for value in centre],
        half=[float(value) for value in half],
        pal=base64.b64encode(palette.tobytes()).decode("ascii"),
        pal_count=int(len(palette)),
        chunks=chunks,
        dist=("data/" + distance_path.name) if distance_path is not None else None,
        colour_error=round(error, 2),
    )
    extra = distance_path.stat().st_size if distance_path is not None else 0
    return spec, size, extra


def budget(entries, limit, page_limit, gt_limit=0):
    """How many points every entry gets. A limit of zero means no cap at all.

    Entries are (key, size) pairs. Only the GT laser cloud takes the lower GT budget, because it
    only provides shape context. The identification is by key, not by position: on a scene with no
    laser scan (ignatius, meetingroom) the first entry is a prediction, and picking it by index
    silently cut that prediction to the GT budget — 300 k of 553 k points on tnt100/meetingroom.
    The page cap stays as an emergency brake for a machine that cannot hold every cloud of a
    scene; it is off by default, because with chunked payloads a big page is no longer a stall on
    open, only more memory.
    """
    takes = [(size if not gt_limit else min(size, gt_limit)) if key == "gt"
             else (size if not limit else min(size, limit))
             for key, size in entries]
    total = sum(takes)
    gt_index = next((index for index, (key, _) in enumerate(entries) if key == "gt"), None)
    if page_limit and total > page_limit:
        predictions = total - (takes[gt_index] if gt_index is not None else 0)
        if gt_index is not None and predictions < page_limit:
            takes[gt_index] = max(1, page_limit - predictions)
        else:
            factor = page_limit / float(total)
            takes = [max(1, int(round(take * factor))) for take in takes]
    return takes


def subsample(points, colours, limit, seed=0):
    """Stratified reduction: see common.stratified_subsample for why not a random cut.

    A zero limit keeps the cloud whole: that is the default now, and a reduction only happens
    when the caller asks for one.
    """
    if not limit or len(points) <= limit:
        return points, colours
    return stratified_subsample(points, colours, limit, seed=seed)


# ------------------------------------------------------------------- geometry
def alignment(dataset, scene, method):
    path = OUTPUTS / dataset / scene / method / "evaluation.json"
    if path.exists():
        evaluation = read_json(path)
        return (dict(s=evaluation["alignment"]["s"], R=np.asarray(evaluation["alignment_rotation"]),
                     t=np.asarray(evaluation["alignment_translation"])),
                evaluation["cloud"]["fscore@1"])
    return pose_sim3(dataset, scene, method), None


def cloud_path(dataset, scene, method, tag):
    """The display cloud, falling back to the metric cloud when no display copy was built."""
    if tag:
        path = CLOUDS / dataset / scene / (method + tag + ".ply")
        if path.exists():
            return path
        metric = CLOUDS / dataset / scene / (method + ".ply")
        if metric.exists():
            print("    (no %s%s.ply for %s, using the metric cloud)" % (method, tag, scene),
                  flush=True)
        return metric
    return CLOUDS / dataset / scene / (method + ".ply")


def load_cloud(dataset, scene, method, params, tag):
    path = cloud_path(dataset, scene, method, tag)
    if not path.exists():
        return None
    points, colours = read_ply(path)
    points = apply_sim3(points, params)
    # T&T predictions cover the whole surroundings while the laser GT covers only the official
    # polygon, so show exactly the region the metrics are computed on.
    region = load_crop(GT / dataset / scene)
    if region:
        mask = crop_mask(points, region)
        points, colours = points[mask], colours[mask]
    return points, colours



def _rodrigues(vector, axis, degrees):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(degrees)
    return (vector * np.cos(angle)
            + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1.0 - np.cos(angle)))


def view_direction(up, towards_camera, yaw=32.0, elevation=20.0):
    """Direction from the target to the eye for the opening shot.

    The azimuth comes from a camera (a view exactly along it reads as a flat picture) turned
    sideways by yaw; the elevation is counted from the scene's own horizon, so the page opens
    on the same kind of 3/4 aerial view the report figures use. Raising a camera's own axes
    instead does not work: there the eye and the up vector are perpendicular by construction,
    so the eye stays level with the target and only the head gets tilted.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    horizontal = np.asarray(towards_camera, dtype=np.float64)
    horizontal = horizontal - float(horizontal @ up) * up
    if np.linalg.norm(horizontal) < 1e-6:
        horizontal = np.array([1.0, 0.0, 0.0]) - up[0] * up
    horizontal = _rodrigues(horizontal / np.linalg.norm(horizontal), up, yaw)
    angle = np.radians(elevation)
    return horizontal * np.cos(angle) + up * np.sin(angle)


def scene_basis(up):
    """Rotation that maps the scene's up vector to +Y, so the orbit controls are Y-up."""
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    reference = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    xaxis = np.cross(reference, up)
    xaxis = xaxis / np.linalg.norm(xaxis)
    zaxis = np.cross(xaxis, up)
    rotation = np.stack([xaxis, up, zaxis])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    return rotation, [float(value) for value in matrix.T.ravel()]


def orbit_camera(rotation, centre, eye, radius):
    """Spherical camera parameters (in the rotated frame) that open the page on a 3/4 view."""
    direction = rotation @ np.asarray(eye, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    target = rotation @ np.asarray(centre, dtype=np.float64)
    return dict(target=[float(value) for value in target],
                radius=float(radius),
                theta=float(np.arctan2(direction[0], direction[2])),
                phi=float(np.arccos(np.clip(direction[1], -1.0, 1.0))))


def framing_radius(rotation, centre, eye, points, aspect=1.7, margin=1.12):
    """Camera distance that actually fills the frame with the points that are drawn.

    The bounding-box diagonal over-estimates the distance badly for flat, elongated scenes
    (a barn is long and low), which left the cloud small in the middle of the page, while a
    cloud framed by the box alone runs over the edges. Everything here is in the rotated frame
    the page uses, where up is +Y: the eye has to be rotated too, or the horizontal and the
    vertical half-angles swap and the picture is cropped on the wrong sides.
    """
    eye = np.asarray(eye, dtype=np.float64)
    eye = eye / np.linalg.norm(eye)
    local_eye = rotation @ eye
    right = np.cross(np.array([0.0, 1.0, 0.0]), local_eye)
    right = right / np.linalg.norm(right)
    up = np.cross(local_eye, right)
    local = (np.asarray(points, dtype=np.float64) - centre) @ rotation.T
    depth = local @ local_eye
    tan_v = np.tan(np.radians(FOV) / 2.0)
    tan_h = tan_v * aspect
    need = np.maximum(depth + np.abs(local @ right) / tan_h,
                      depth + np.abs(local @ up) / tan_v)
    return float(np.percentile(need, 99.9) * margin)


FRUSTUM_SEGMENTS = 8  # 4 sides from the apex + the 4 edges of the image plane


def frustum_wireframes(extrinsics, intrinsics, size):
    """One frustum pyramid per camera, flattened to a (cameras, 48) block of line vertices.

    Kept per camera rather than merged into one big buffer so a single pyramid can be picked with
    the mouse: hovering it shows the photo it stands for.
    """
    segments = []
    for rotation, translation, k in zip(extrinsics[:, :3, :3], extrinsics[:, :3, 3], intrinsics):
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        # images are resized so the principal point stays centred: width ~ 2*cx, height ~ 2*cy
        half_width = max(cx / fx, 1e-6)
        half_height = max(cy / fy, 1e-6)
        depth = size / (2.0 * max(half_width, half_height))
        corners = []
        for sign_x, sign_y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cam = np.array([sign_x * half_width * depth, sign_y * half_height * depth, depth])
            corners.append(rotation.T @ (cam - translation))
        apex = -rotation.T @ translation
        for index in range(4):
            segments.extend([apex, corners[index]])
            segments.append(corners[index])
            segments.append(corners[(index + 1) % 4])
    return np.asarray(segments, dtype=np.float32).reshape(len(extrinsics),
                                                          3 * 2 * FRUSTUM_SEGMENTS)


CAM_TILE = (256, 192)  # 4:3 tile: a whole photo at a glance, small enough for a tooltip
CAM_COLUMNS = 5


def camera_sheet(out, dataset, scene, count):
    """Contact sheet of the input frames, one tile per GT camera.

    The pages draw camera pyramids but no pictures, so a visitor sees a wireframe and cannot tell
    which wall it photographed. The frames go into one JPEG per scene and the tooltip shows the
    tile by CSS offset — one request for the whole set instead of one per camera.
    """
    manifest = INFERENCE / dataset / scene / "manifest.json"
    if not manifest.exists():
        return None
    frames = [frame["name"] for frame in read_json(manifest)["frames"]]
    folder = INFERENCE / dataset / scene / "images"
    names = [name for name in frames if (folder / name).exists()][:count]
    if not names:
        return None
    columns = min(CAM_COLUMNS, len(names))
    rows = int(np.ceil(len(names) / columns))
    tile_width, tile_height = CAM_TILE
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (11, 14, 18))
    for position, name in enumerate(names):
        with Image.open(folder / name) as opened:
            opened.draft("RGB", (3 * tile_width, 3 * tile_height))  # decode at a fraction of full size
            image = opened.convert("RGB")
        scale = max(tile_width / image.width, tile_height / image.height)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.LANCZOS)
        left = (image.width - tile_width) // 2
        top = (image.height - tile_height) // 2
        sheet.paste(image.crop((left, top, left + tile_width, top + tile_height)),
                    ((position % columns) * tile_width, (position // columns) * tile_height))
    out.mkdir(parents=True, exist_ok=True)
    name = "%s-%s-cams.jpg" % (dataset, scene)
    sheet.save(out / name, "JPEG", quality=80, optimize=True)
    print("    cameras: %d frames -> %s (%.0f kB)"
          % (len(names), name, (out / name).stat().st_size / 1e3), flush=True)
    return dict(url="data/" + name, cols=columns, rows=rows, w=tile_width, h=tile_height,
                tile=[[position % columns, position // columns] for position in range(len(names))],
                names=names)


def frame_transform(dataset, scene):
    """Transform that takes the reference cameras into the reference cloud's frame.

    Tanks and Temples keeps its COLMAP model and its laser scan in different coordinate systems, so
    gt_meta.json carries the official transform between them. Without it the camera pyramids hang
    twenty metres away from the cloud they photographed (and hovering them makes no sense).
    """
    path = GT / dataset / scene / "gt_meta.json"
    if not path.exists():
        return None
    value = read_json(path).get("frame_transform")
    return None if value is None else np.asarray(value, dtype=np.float64)


def nearest(query, reference):
    return cKDTree(np.asarray(reference, dtype=np.float64)).query(
        np.asarray(query, dtype=np.float64))[0]


# ----------------------------------------------------------------------- scene
def scene_page(out, dataset, scene, methods, tag, limit, page_limit, gt_limit=0):
    with np.load(GT / dataset / scene / "cameras.npz") as handle:
        extrinsics = np.asarray(handle["extrinsics"], dtype=np.float64)
        intrinsics = (np.asarray(handle["intrinsics"], dtype=np.float64)
                      if "intrinsics" in handle.files else None)
    clouds, scores = {}, {}
    for method in methods:
        params, score = alignment(dataset, scene, method)
        loaded = load_cloud(dataset, scene, method, params, tag)
        if loaded is None:
            continue
        clouds[method], scores[method] = loaded, score
    if not clouds:
        return None

    # Frame the view by what was actually reconstructed: a robust box around the
    # predictions, which also gives the GT lidar its crop (it covers the whole scan
    # area, so without it the view is framed by empty terrain).
    union = np.concatenate([points for points, _ in clouds.values()])
    low, high = percentile_box(union, 1.0, 99.0, 0.04)
    for method, (points, colours) in list(clouds.items()):
        keep = np.all((points > low) & (points < high), axis=1)
        clouds[method] = (points[keep], colours[keep])
    box = 0.06 * float(np.linalg.norm(high - low))
    gt_points = gt_colours = None
    gt_path = GT / dataset / scene / "gt_cloud.ply"
    if gt_path.exists():
        points, colours = read_ply(gt_path)
        inside = np.all((points > low - box) & (points < high + box), axis=1)
        # laser ground truth carries no colour: distance colouring is what it is for
        gt_points, gt_colours = points[inside], None if colours is None else colours[inside]

    # Green/red threshold: 1% of the GT cloud diagonal. The diagonal comes from gt_meta.json,
    # which is the full lidar scan the report measures F@1 against, so the plaque on the page
    # and the figure caption carry the very same number instead of two slightly different ones
    # (the GT cloud drawn here is cropped to the reconstructed area, and 1% of the crop was
    # up to 20% smaller than 1% of the scan).
    report_diagonal = None
    gt_meta_path = GT / dataset / scene / "gt_meta.json"
    if gt_meta_path.exists():
        gt_meta = read_json(gt_meta_path)
        if "diagonal" in gt_meta:
            report_diagonal = float(gt_meta["diagonal"])
    if report_diagonal:
        tau = 0.01 * report_diagonal
        tau_note = "1% диагонали GT-облака, та же τ, что у F@1 в отчёте"
    elif gt_points is not None and len(gt_points):
        tau = 0.01 * diagonal(gt_points)
        tau_note = "1% диагонали GT-облака"
    else:
        tau = 0.01 * diagonal(np.concatenate([points for points, _ in clouds.values()]))
        tau_note = "1% диагонали предсказаний: GT-облака для этой сцены нет"

    # Distances are computed on the points that are actually drawn, so their count drives cost.
    entries = []
    if gt_points is not None and len(gt_points):
        entries.append(("gt", gt_points, gt_colours, "GT: лидар", len(gt_points)))
    for method in methods:
        if method not in clouds:
            continue
        label = LABELS.get(method, method)
        if scores.get(method) is not None:
            label += " — F@1 = %.3f" % scores[method]
        entries.append((SHORTCUTS.get(method, method), clouds[method][0], clouds[method][1],
                        label, len(clouds[method][0])))

    takes = budget([(entry[0], entry[4]) for entry in entries], limit, page_limit, gt_limit)
    gt_tree = cKDTree(gt_points) if gt_points is not None and len(gt_points) else None

    traces, sizes = {}, {}
    framed = []  # a sample of what is actually drawn, used to frame the opening view
    payload = dist_bytes = 0
    for seed, (key, points, colours, label, available), take in zip(range(len(entries)),
                                                                   entries, takes):
        points, colours = subsample(points, colours, take, seed=seed)
        framed.append(points[::max(1, len(points) // 60000)])
        if key == "gt":
            distances = None
            if clouds:
                distances = np.min([nearest(points, full) for full, _ in clouds.values()], axis=0)
        elif gt_tree is not None:
            distances = gt_tree.query(points)[0]
        else:
            others = [other for name, (other, _) in clouds.items()
                      if SHORTCUTS.get(name, name) != key]
            distances = nearest(points, np.concatenate(others)) if others else np.zeros(len(points))
        if distances is not None:
            distances = np.minimum(distances, tau)
        spec, size, extra = write_payloads(out, "%s-%s" % (dataset, scene), key, points,
                                           colours, distances, tau)
        spec["label"] = label
        traces[key] = spec
        sizes[key] = len(points)
        payload += size
        dist_bytes += extra
        print("    %-5s %8d / %8d pts  payload %.1f MB  colour error %.1f"
              % (key, len(points), available, size / 1e6, spec["colour_error"]), flush=True)

    # Open the page from a three-quarter view built on camera 0 of the scene: the axis-aligned
    # default tends to look straight into a wall, and the exact camera-0 view is so frontal that
    # a cloud reads as a flat picture instead of a 3D model.
    # Tanks and Temples keeps its cameras and its laser scan in different frames, so the shot has
    # to be built in the frame the cloud is drawn in: taken raw, the view is rolled by whatever
    # the official transform is, and the barn opened upside down.
    frame = frame_transform(dataset, scene)
    to_cloud = np.eye(3) if frame is None else frame[:3, :3] / np.cbrt(abs(np.linalg.det(frame[:3, :3])))
    # A single camera carries the tilt of a hand-held shot; averaging the "up" row over all of
    # them cancels it and recovers the vertical of the reconstruction itself.
    averaged = -extrinsics[:, :3, :3][:, 1, :].mean(axis=0)
    _, _, forward = view_basis(extrinsics[0])
    up = to_cloud @ averaged
    rotation, basis = scene_basis(up)
    eye = view_direction(up, -(to_cloud @ np.asarray(forward, dtype=np.float64)),
                         yaw=32.0, elevation=20.0)
    span = float(np.linalg.norm(high - low))
    drawn = np.concatenate(framed) if framed else 0.5 * (low + high)[None, :]
    drawn_low, drawn_high = percentile_box(drawn, 1.0, 99.0, 0.0)
    centre = 0.5 * (drawn_low + drawn_high)
    camera = orbit_camera(rotation, centre, eye,
                          framing_radius(rotation, centre, eye, drawn))

    frustums = None
    if intrinsics is not None:
        # Pyramids have to be big enough to aim at with the mouse, and they are the only hint of
        # where the photos were taken from, so 3.5% of the scene span rather than a token 2.5%.
        # They are built in the camera frame and then moved into the cloud's frame, so the scale
        # the transform carries has to be divided out first: on the T&T scenes (scale ~4) they
        # otherwise came out four times the intended size and covered the building.
        wire_size = 0.035 * span
        if frame is not None:
            wire_size /= np.cbrt(abs(np.linalg.det(frame[:3, :3])))
        wire = frustum_wireframes(extrinsics, intrinsics, wire_size)
        if frame is not None:
            moved = wire.reshape(-1, 3) @ frame[:3, :3].T + frame[:3, 3]
            wire = moved.reshape(wire.shape).astype(np.float32)
        frustums = dict(b64=base64.b64encode(wire.tobytes()).decode("ascii"),
                        size=int(len(extrinsics)), per=int(wire.shape[1]), edge=0.035 * span)
    camshots = camera_sheet(out / "data", dataset, scene, len(extrinsics))
    predictions = [key for key in PREDICTION_KEYS if key in traces]
    has_gt = "gt" in traces
    meta = dict(
        traces=traces,
        predictions=predictions,
        # VGGT-Omega opens first: it is the strongest of the methods, and "all predictions"
        # is a comparison, not a starting point.
        default_cloud=(SHORTCUTS["vggt-omega"] if SHORTCUTS["vggt-omega"] in predictions
                       else (predictions[0] if predictions else "gt")),
        cams=True,
        camshots=camshots,
        basis=basis,
        camera=camera,
        fov=FOV,
        frustums=frustums,
        ramp=base64.b64encode(ramp_table(DIST_SCALE).tobytes()).decode("ascii"),
        ramp_size=RAMP_SIZE,
        point_size=2.2,
        tau=float(tau),
        tau_note=tau_note,
        ramp_css="linear-gradient(90deg, " + ", ".join(
            "%s %.0f%%" % (colour, 100.0 * share) for share, colour in DIST_SCALE) + ")",
        legend=dict(
            pred=("расстояние до GT, м" if has_gt else "расхождение методов, м"),
            gt=("GT → ближайшее предсказание, м" if has_gt else None),
            note=("дальше τ — одним красным; τ = 1 % диагонали GT" if has_gt
                  else "дальше τ — одним красным; τ = 1 % диагонали предсказаний"),
        ),
        hint_rgb="цвет пикселей, из которых получены точки",
        hint_dist=("зелёный — на поверхности GT, красный — дальше \u03c4" if has_gt
                   else "зелёный — совпадение, красный — расхождение"),
    )
    return meta, sizes, payload, dist_bytes


def write_page(path, meta, heading, note, methods=None):
    buttons = []
    for key, title in [("gt", "GT (лидар)")] + [(SHORTCUTS.get(method, method),
                                                LABELS.get(method, method))
                                               for method in (methods or PREDICTION_KEYS)]:
        disabled = "" if key in meta["traces"] else " disabled"
        buttons.append('<button class="pick" data-cloud="%s"%s>%s</button>' % (key, disabled, title))
    if len(meta["predictions"]) >= 2:
        # The hash key stays 'both' whatever the number of methods, so old links keep working.
        buttons.append('<button class="pick" data-cloud="both">%s</button>'
                       % union_label(len(meta["predictions"])))
    controls = CONTROLS.replace("__CLOUD_BUTTONS__", "\n  ".join(buttons))
    controls = controls.replace("__DIST_LABEL__",
                                "расстояние до GT" if "gt" in meta["traces"]
                                else "расхождение с другим методом")
    html = (
        "<!doctype html>\n<html lang='ru'>\n<head>\n<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        "<title>%s</title>\n<link rel='icon' href='data:,'>\n<style>%s</style>\n"
        "<script src='three.min.js'></script>\n</head>\n<body>\n"
        "%s%s\n<div id='gl'>%s</div>\n%s\n"
        "<script>window.VIEWER = %s;</script>\n%s\n"
        "</body>\n</html>\n"
    ) % (heading, CSS, HEADER % (heading, note), controls, PROGRESS + LEGEND, CAMSHOT,
         json.dumps(meta, ensure_ascii=False), SCRIPT)
    Path(path).write_text(html, encoding="utf-8")
    return len(html.encode("utf-8"))


def thousands(value):
    return "{:,}".format(int(value)).replace(",", "\u2009")


def pose_warnings(dataset, scene, methods):
    """A cloud whose poses did not converge is drawn misaligned: say it in the header."""
    warnings = []
    for method in methods:
        path = OUTPUTS / dataset / scene / method / "pose_evaluation.json"
        if not path.exists():
            continue
        rotation = read_json(path)["rotation"]["mean_deg"]
        if rotation > 10.0:
            warnings.append("%s: позы не сходятся (средняя ошибка поворота %.1f°), "
                            "облако смещено относительно GT" % (LABELS.get(method, method), rotation))
    return warnings


INDEX_HEAD = """<!doctype html>
<meta charset="utf-8">
<title>Облака точек: интерактивный просмотр</title>
<style>
  body {{ margin: 0; padding: 28px; background: #14171c; color: #e8e8e8;
         font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; line-height: 1.5; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; }}
  p  {{ color: #a8b0ba; max-width: 820px; }}
  ul {{ columns: 2; max-width: 980px; padding-left: 20px; }}
  li {{ margin: 6px 0; }}
  a  {{ color: #6cb6ff; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .hint {{ color: #6b7480; font-size: 12.5px; margin-left: 10px; }}
  code {{ background: #22262e; padding: 1px 5px; border-radius: 4px; }}
</style>
<h1>Облака точек: интерактивный просмотр</h1>
<p>По одному файлу на сцену. Облака приведены в систему координат GT через sim(3) по центрам
камер и раскрашены цветом пикселей. Кнопки сверху переключают, какое облако смотреть (GT,
{methods} или {union}), как его красить — цветом пикселей или расстоянием до
GT (зелёный — на поверхности, красный — дальше τ) — и задают размер точки. Мышь вращает, колесо
зумит, панорама — Ctrl, Shift, Alt или Cmd с левой кнопкой, а также правая и средняя кнопки.
По умолчанию открывается VGGT-Ω, а центры камер GT включены: наведите курсор на пирамидку
камеры — всплывёт кадр, снятый с этой точки.</p>
<p>Сцены идут от лучших к худшим по среднему F@1 {count} методов, а не по алфавиту: так первым делом
видно, что получается хорошо, а не самое слабое.</p>
<p>Рендер — three.js: точки рисуются буфером WebGL, координаты упакованы в int16, цвет — один
байт палитры, поэтому на странице сотни тысяч и миллионы точек, а не сто тысяч, как раньше в
Plotly. Полезная нагрузка каждой сцены лежит рядом в <code>data/*.bin</code> кусками-ступенькой
(40 тыс., 40 тыс., 80 тыс. точек, дальше по миллиону): мелкий первый кусок рисуется через секунду,
дальше в полёте держится до четырёх кусков сразу, индикатор относится к открытым облакам и GT, а к
концу загрузки в памяти лежат все облака целиком (10.9 млн точек на облако у 100-кадровых сцен). Страницы нужно
открывать по http (GitHub Pages):
из <code>file://</code> браузер не отдаёт соседние бинарные файлы. <code>three.min.js</code> лежит
рядом, интернет не нужен.</p>
<ul>
"""


# LABELS / METHODS live at the top of the file; they are shared by the pages and the index.
# The same building appears under two schemes, so the link text has to say which one it is.
SCHEMES = {"eth3d": "ETH3D-10", "eth3d20": "ETH3D-20", "tnt": "T&T-10", "tnt100": "T&T-100"}


def index_metrics(stem):
    """The hint under each link: how well the clouds match the GT lidar, plus a quality score.

    The score is the mean F@1 over the methods, and drives the order of the list: the scenes the
    reconstructions handle best come first, so a reader starts with the ones worth looking at.
    """
    dataset, _, scene = stem.replace("-", " ", 1).partition(" ")
    scene = scene.replace(" ", "-")
    labels, scores, present = [], [], []
    for method in METHODS:
        evaluation = OUTPUTS / dataset / scene / method / "evaluation.json"
        if evaluation.exists():
            score = float(read_json(evaluation)["cloud"]["fscore@1"])
            labels.append("%s %.3f" % (LABELS.get(method, method), score))
            scores.append(score)
            present.append(method)
    hint = " · F@1: " + ", ".join(labels) if labels else ""
    return (sum(scores) / len(scores) if scores else -1.0), hint, present


def index_title(stem):
    """'eth3d20-courtyard' -> 'courtyard · ETH3D-20'."""
    dataset, _, scene = stem.replace("-", " ", 1).partition(" ")
    return "%s · %s" % (scene.replace(" ", "-"), SCHEMES.get(dataset, dataset))


def write_index(out):
    pages = [path.name for path in Path(out).glob("*.html") if path.name != "index.html"]
    scored = [(index_metrics(name[:-5]), name) for name in pages]
    scored.sort(key=lambda item: (-item[0][0], item[1]))
    items = "\n".join('<li><a href="%s">%s</a><span class="hint">%s</span></li>'
                      % (name, index_title(name[:-5]), hint) for (_, hint, _), name in scored)
    # The preamble names the methods this build actually found, not every known method.
    found = [method for method in METHODS if any(method in item[0][2] for item in scored)]
    prose = dict(methods=", ".join(LABELS.get(method, method) for method in found),
                 union=union_label(len(found)),
                 count={2: "двух", 3: "трёх"}.get(len(found), "всех"))
    Path(out, "index.html").write_text(INDEX_HEAD.format(**prose) + items + "\n</ul>\n",
                                       encoding="utf-8")
    print("index.html written (%d scenes, best first)" % len(pages), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--points", type=int, default=0,
                        help="point budget per prediction cloud; 0 keeps the whole display cloud "
                             "and the page streams it in chunks")
    parser.add_argument("--gt-points", type=int, default=300000,
                        help="point budget for the GT laser cloud: it is drawn for shape context, "
                             "not for detail")
    parser.add_argument("--page-points", type=int, default=0,
                        help="emergency cap of points on one page; 0 means no cap")
    parser.add_argument("--cloud-tag", default="-display",
                        help="suffix of the display clouds built with build_clouds.py "
                             "--output-tag; the metric cloud is used when no display copy exists")
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(VENDOR / "three.min.js", out / "three.min.js")
    rows = []
    for scene in args.scenes:
        present = []
        for method in args.methods:
            if cloud_path(args.dataset, scene, method, args.cloud_tag).exists():
                present.append(method)
            else:
                # A missing method is skipped, not fatal: the benchmark pages are rebuilt scene
                # by scene as the methods land, and one absent cloud must not stop the rest.
                print("  warning: %s / %s: no %s cloud, skipping that method"
                      % (args.dataset, scene, method), file=sys.stderr, flush=True)
        if not present:
            print("  warning: %s / %s: none of the methods has a cloud, no page written"
                  % (args.dataset, scene), file=sys.stderr, flush=True)
            continue
        print("%s / %s: %s" % (args.dataset, scene, ", ".join(present)), flush=True)
        built = scene_page(out, args.dataset, scene, present, args.cloud_tag,
                           args.points, args.page_points, args.gt_points)
        if built is None:
            continue
        meta, sizes, payload, dist_bytes = built
        name = "%s-%s.html" % (args.dataset, scene)
        total = sum(sizes.values())
        budget_text = ("целиком, %s на облако" % thousands(args.points)
                       if args.points else "целиком, без прореживания")
        # The GT-lidar budget only means something on a scene that has the lidar: on ignatius and
        # meetingroom the plaque used to promise "GT-лидар — до 300 000" with no GT cloud in sight.
        gt_part = ("GT-лидар — до %s" % thousands(args.gt_points) if "gt" in meta["traces"]
                   else "лазерного GT у этой сцены нет")
        note = ("Облака после sim(3) по позам камер. Точек: %s (%s; %s). "
                "Раскраска «расстояние»: τ = %s (%s), всё, что дальше, — одним красным."
                % (thousands(total), budget_text, gt_part,
                   metric_text(meta["tau"]), meta["tau_note"]))
        note += (" <span style='color:#8f98a3'>· вращение — ЛКМ, панорама — ЛКМ с Shift/⌃"
                 " (или ПКМ, СКМ), зум — колесо.</span>")
        warnings = pose_warnings(args.dataset, scene, present)
        if warnings:
            note += " <span style='color:#ffb020'>Внимание: " + "; ".join(warnings) + ".</span>"
        page_bytes = write_page(out / name, meta, "%s / %s" % (args.dataset, scene), note,
                                present)
        rows.append((name, sizes, page_bytes, payload, dist_bytes))
        print("%s written: %d points, %.1f MB payload, %.0f kB html" % (
            name, total, payload / 1e6, page_bytes / 1e3), flush=True)
    if not args.no_index:
        write_index(out)
    if rows:
        print("\n%-26s %9s %9s %9s %8s %8s" % ("page", "points", "payload", "html", "total", "dist"))
        for name, sizes, page_bytes, payload, dist_bytes in rows:
            print("%-26s %9d %8.1fM %7.0fkB %7.1fM %7.1fM" % (
                name, sum(sizes.values()), payload / 1e6, page_bytes / 1e3,
                (payload + page_bytes) / 1e6, dist_bytes / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
