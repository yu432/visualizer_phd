"""Build interactive HTML viewers (Plotly / WebGL) so the clouds can be rotated by hand.

Each scene page shows one cloud at a time (GT lidar, DA3-Giant, VGGT-Omega or both
predictions) and can recolour the points either with their pixel colours or with the
distance to the GT cloud (green = on the surface, red = far / off). Everything is
inlined, so the pages work offline over file:// with the neighbouring plotly.min.js.
"""
import argparse
import base64
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import apply_sim3  # noqa: E402
from pose_align import centres, sim3 as pose_sim3  # noqa: E402
from common import (CLOUDS, GT, OUTPUTS, crop_mask, diagonal, load_crop,  # noqa: E402
                    percentile_box, read_json, read_ply, view_basis)

SHORT_HEX = "0123456789abcdef"
LABELS = {"da3-giant": "DA3-Giant", "vggt-omega": "VGGT-\u03a9"}
# Green to red, the same convention as the distance figures in the PDF.
DIST_SCALE = [
    [0.0, "#1a9850"], [0.25, "#91cf60"], [0.5, "#d9ef8b"],
    [0.75, "#fdae61"], [0.9, "#f46d43"], [1.0, "#d73027"],
]
# Kept compact and top-right so it does not eat the view; the title is set per mode.
COLORBAR = dict(colorbar=dict(
    len=0.62, thickness=12, x=0.995, xanchor="right", y=0.95, yanchor="top",
    bgcolor="rgba(20,23,28,0)", outlinewidth=0,
    title=dict(text="", side="top", font=dict(size=11.5)),
    tickfont=dict(size=10.5),
))

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
  <span id="hint" style="color:#8f98a3"></span>
</div>
<style>
  .pick { background:#1e242c; color:#dfe5ec; border:1px solid #2e3742; border-radius:6px;
          padding:4px 11px; font:inherit; cursor:pointer; }
  .pick:hover { background:#262e38; }
  .pick.on { background:#2b6cb0; border-color:#3b82d0; color:#ffffff; font-weight:600; }
  .pick:disabled { opacity:.35; cursor:not-allowed; }
</style>
"""

SCRIPT = """
<script>
(function () {
  var META = window.VIEWER;
  var gd = document.getElementById('cloud');
  var ORIG = null, ready = false, CACHE = {};
  var state = {cloud: META.default_cloud, colour: 'rgb', cams: false};

  // The page state is mirrored into the URL hash, so a view can be bookmarked or
  // deep-linked (and so the render can be checked headlessly): #da3:dist
  function fromHash() {
    var parts = (location.hash || '').replace('#', '').split(':');
    if (parts[0] && (META.index[parts[0]] !== null || parts[0] === 'both')) { state.cloud = parts[0]; }
    if (parts[1] === 'rgb' || parts[1] === 'dist') { state.colour = parts[1]; }
    if (parts[2] === 'cams') { state.cams = true; }
  }
  function toHash() {
    var tail = state.cams ? ':cams' : '';
    try { history.replaceState(null, '', '#' + state.cloud + ':' + state.colour + tail); } catch (e) {}
  }

  function decode(b64) {
    var s = atob(b64), n = s.length, buf = new ArrayBuffer(n), u = new Uint8Array(buf);
    for (var i = 0; i < n; i++) { u[i] = s.charCodeAt(i); }
    return Array.from(new Float32Array(buf));
  }
  function distOf(i) {
    if (!CACHE[i]) { CACHE[i] = decode(META.traces[i].dist); }
    return CACHE[i];
  }
  function chosen() {
    if (state.cloud === 'both') { return META.predictions.slice(); }
    return [state.cloud];
  }
  function active(key) {
    return chosen().indexOf(key) >= 0;
  }
  function syncButtons() {
    var buttons = document.querySelectorAll('.pick');
    for (var i = 0; i < buttons.length; i++) {
      var b = buttons[i], on;
      if (b.dataset.cloud) { on = state.cloud === b.dataset.cloud; }
      else { on = state.colour === b.dataset.colour; }
      b.classList.toggle('on', on);
    }
    document.getElementById('cams').checked = state.cams;
    var hint = document.getElementById('hint');
    if (state.colour === 'dist') {
      hint.textContent = META.hint_dist;
    } else {
      hint.textContent = META.hint_rgb;
    }
  }
  function apply() {
    if (!ready) { return; }
    var visible = gd.data.map(function () { return false; });
    var keys = chosen(), indices = [], colours = [], scales = [], showscales = [];
    var mins = [], maxs = [], titles = [], shown = false;
    for (var k = 0; k < keys.length; k++) {
      var key = keys[k], i = META.index[key];
      if (i === null || i === undefined) { continue; }
      visible[i] = true;
      indices.push(i);
      if (state.colour === 'dist') {
        colours.push(distOf(i));
        scales.push(META.scale);
        mins.push(0);
        maxs.push(META.traces[i].tau);
        titles.push(key === 'gt' ? META.title_gt : META.title_pred);
        // A single colour bar is enough, even when both predictions are shown.
        showscales.push(!shown);
        shown = true;
      } else {
        colours.push(ORIG[i]);
        scales.push(META.scale);
        mins.push(0);
        maxs.push(1);
        titles.push('');
        showscales.push(false);
      }
    }
    if (state.cams) { visible[META.cams] = true; }
    Plotly.restyle(gd, {'visible': visible});
    if (indices.length) {
      Plotly.restyle(gd, {
        'marker.color': colours,
        'marker.colorscale': scales,
        'marker.cmin': mins,
        'marker.cmax': maxs,
        'marker.showscale': showscales,
        'marker.colorbar.title.text': titles
      }, indices);
    }
    syncButtons();
    toHash();
  }
  function init() {
    if (ready) { return; }
    ready = true;
    ORIG = gd.data.map(function (trace) {
      return (trace.marker && Array.isArray(trace.marker.color)) ? trace.marker.color.slice() : null;
    });
    fromHash();
    var buttons = document.querySelectorAll('.pick');
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener('click', function () {
        if (this.dataset.cloud) { state.cloud = this.dataset.cloud; }
        else { state.colour = this.dataset.colour; }
        apply();
      });
    }
    document.getElementById('cams').addEventListener('change', function () {
      state.cams = this.checked;
      apply();
    });
    apply();
  }
  var timer = setInterval(function () {
    if (gd && gd.data && gd.data.length) { clearInterval(timer); init(); }
  }, 80);
})();
</script>
"""


def rounded(points):
    """Two decimals is well below the cloud resolution and keeps the HTML small."""
    return np.round(np.asarray(points, dtype=np.float64), 2).astype(np.float32)


def subsample(points, colours, limit, seed=0):
    if len(points) <= limit:
        return points, colours
    rng = np.random.default_rng(seed)
    index = rng.choice(len(points), size=limit, replace=False)
    index.sort()
    return points[index], None if colours is None else colours[index]


def short_colours(colours, count):
    """4 bits per channel as a CSS shorthand hex colour: much smaller than rgb() strings."""
    if colours is None:
        return ["#9aa4b0"] * count
    palette = np.asarray(colours, dtype=np.uint16)[:, :3] // 17
    codes = (palette[:, 0] << 8) | (palette[:, 1] << 4) | palette[:, 2]
    table, out = {}, []
    for code in codes:
        code = int(code)
        value = table.get(code)
        if value is None:
            value = "#" + SHORT_HEX[(code >> 8) & 15] + SHORT_HEX[(code >> 4) & 15] + SHORT_HEX[code & 15]
            table[code] = value
        out.append(value)
    return out


def encode(values):
    """float32 in base64: compact enough to inline, decoded lazily in the page."""
    return base64.b64encode(np.asarray(values, dtype=np.float32).tobytes()).decode("ascii")


def alignment(dataset, scene, method):
    path = OUTPUTS / dataset / scene / method / "evaluation.json"
    if path.exists():
        evaluation = read_json(path)
        return (dict(s=evaluation["alignment"]["s"], R=np.asarray(evaluation["alignment_rotation"]),
                     t=np.asarray(evaluation["alignment_translation"])),
                evaluation["cloud"]["fscore@1"])
    return pose_sim3(dataset, scene, method), None


def load_cloud(dataset, scene, method, params):
    path = CLOUDS / dataset / scene / (method + ".ply")
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


def three_quarter(towards_camera, up, yaw=32.0, pitch=20.0):
    """Turn a frontal camera direction into an oblique one: first sideways, then raise it."""
    towards_camera = towards_camera / np.linalg.norm(towards_camera)
    up = up - towards_camera * np.dot(up, towards_camera)
    up = up / np.linalg.norm(up)
    side = np.cross(up, towards_camera)
    direction = _rodrigues(towards_camera, up, yaw)
    side = _rodrigues(side, up, yaw)
    direction = _rodrigues(direction, side, pitch)
    return direction / np.linalg.norm(direction), _rodrigues(up, side, pitch)


def crop_box(clouds, low_percentile=1.0, high_percentile=99.0, pad=0.04):
    """Box around the predictions: it frames the view and crops the GT lidar to it."""
    union = np.concatenate([points for points, _ in clouds])
    return percentile_box(union, low_percentile, high_percentile, pad)


def nearest(query, reference):
    return cKDTree(np.asarray(reference, dtype=np.float64)).query(
        np.asarray(query, dtype=np.float64))[0]


def scene_figure(dataset, scene, methods, limit):
    with np.load(GT / dataset / scene / "cameras.npz") as handle:
        extrinsics = np.asarray(handle["extrinsics"], dtype=np.float64)
    cameras = centres(extrinsics)

    clouds, scores = {}, {}
    for method in methods:
        params, score = alignment(dataset, scene, method)
        loaded = load_cloud(dataset, scene, method, params)
        if loaded is None:
            continue
        clouds[method], scores[method] = loaded, score
    if not clouds:
        return None, None

    # Frame the view by what was actually reconstructed: a robust box around the
    # predictions, which also gives the GT lidar its crop (it covers the whole scan
    # area, so without it the view is framed by empty terrain).
    low, high = crop_box(list(clouds.values()))
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

    # Green/red threshold: 1% of the GT diagonal is exactly the tau of F@1 in the report.
    if gt_points is not None and len(gt_points):
        tau = 0.01 * diagonal(gt_points)
        tau_note = "1% диагонали GT-облака"
    else:
        tau = 0.01 * diagonal(np.concatenate([points for points, _ in clouds.values()]))
        tau_note = "1% диагонали предсказаний: GT-облака для этой сцены нет"

    traces, meta = [], {"index": {"gt": None, "da3": None, "omega": None},
                        "predictions": [], "cams": None, "default_cloud": "both",
                        "traces": {}, "scale": DIST_SCALE,
                        "title_pred": "d до GT", "title_gt": "d до предсказаний",
                        "hint_dist": "зелёный — на поверхности GT, красный — дальше \u03c4",
                        "hint_rgb": "цвет пикселей, из которых получены точки"}
    shortcuts = {"da3-giant": "da3", "vggt-omega": "omega"}

    gt_tree = cKDTree(gt_points) if gt_points is not None and len(gt_points) else None

    if gt_points is not None and len(gt_points):
        points, colours = subsample(gt_points, gt_colours, limit, seed=0)
        distances = None
        if clouds:
            distances = np.min([nearest(points, full) for full, _ in clouds.values()], axis=0)
        traces.append(go.Scatter3d(
            x=rounded(points)[:, 0], y=rounded(points)[:, 1], z=rounded(points)[:, 2],
            mode="markers", name="GT: лидар",
            marker=dict(size=1.3, color=short_colours(colours, len(points)), **COLORBAR),
            hoverinfo="skip"))
        index = len(traces) - 1
        meta["index"]["gt"] = index
        if distances is not None:
            meta["traces"][str(index)] = {"dist": encode(np.minimum(distances, tau)), "tau": float(tau)}

    for method in methods:
        if method not in clouds:
            continue
        full, colours = clouds[method]
        points, colours = subsample(full, colours, limit, seed=1)
        label = LABELS.get(method, method)
        if scores.get(method) is not None:
            label += " — F@1 = %.3f" % scores[method]
        traces.append(go.Scatter3d(
            x=rounded(points)[:, 0], y=rounded(points)[:, 1], z=rounded(points)[:, 2],
            mode="markers", name=label,
            marker=dict(size=1.3, color=short_colours(colours, len(points)), **COLORBAR),
            hoverinfo="skip"))
        index = len(traces) - 1
        key = shortcuts.get(method)
        if key:
            meta["index"][key] = index
            meta["predictions"].append(key)
        if gt_tree is not None:
            distances = gt_tree.query(points)[0]
        else:
            others = [other for name, (other, _) in clouds.items() if name != method]
            distances = nearest(points, np.concatenate(others)) if others else np.zeros(len(points))
        meta["traces"][str(index)] = {"dist": encode(np.minimum(distances, tau)), "tau": float(tau)}

    traces.append(go.Scatter3d(
        x=rounded(cameras)[:, 0], y=rounded(cameras)[:, 1], z=rounded(cameras)[:, 2],
        mode="markers+lines", name="центры камер GT",
        marker=dict(size=3.5, color="#ffb020"), line=dict(color="#ffb020", width=3),
        hoverinfo="skip", visible=False))
    meta["cams"] = len(traces) - 1

    if len(meta["predictions"]) == 1:
        meta["default_cloud"] = meta["predictions"][0]
    if gt_points is None:
        meta["default_cloud"] = meta["predictions"][0] if meta["predictions"] else "gt"
        meta["title_pred"] = "d до другого метода"
        meta["hint_dist"] = "зелёный — совпадение, красный — расхождение"

    # Open the page from a three-quarter view built on camera 0 of the scene: the plotly
    # default tends to look straight into a wall, and the exact camera-0 view is so
    # frontal that a cloud reads as a flat picture instead of a 3D model.
    _, down, forward = view_basis(extrinsics[0])
    eye, up = three_quarter(-np.asarray(forward, dtype=np.float64),
                            -np.asarray(down, dtype=np.float64), yaw=32.0, pitch=20.0)
    spans = np.maximum(high - low, 1e-6)
    eye = 1.8 * (eye * spans) / np.linalg.norm(eye * spans)
    up = up / np.linalg.norm(up)

    figure = go.Figure(data=traces)
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#14171c",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), bgcolor="#14171c",
                   camera=dict(eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
                               up=dict(x=float(up[0]), y=float(up[1]), z=float(up[2])),
                               center=dict(x=0.0, y=0.0, z=0.0))),
        uirevision="keep",
    )
    meta["tau"] = float(tau)
    meta["tau_note"] = tau_note
    return figure, meta


def write_viewer(path, figure, meta, heading, note):
    cloud_buttons = []
    for key, title in (("gt", "GT (лидар)"), ("da3", LABELS["da3-giant"]),
                       ("omega", LABELS["vggt-omega"])):
        disabled = "" if meta["index"][key] is not None else " disabled"
        cloud_buttons.append('<button class="pick" data-cloud="%s"%s>%s</button>' % (key, disabled, title))
    if len(meta["predictions"]) == 2:
        cloud_buttons.append('<button class="pick" data-cloud="both">оба предсказания</button>')
    controls = CONTROLS.replace("__CLOUD_BUTTONS__", "\n  ".join(cloud_buttons))
    controls = controls.replace("__DIST_LABEL__",
                            "расстояние до GT" if meta["index"]["gt"] is not None
                            else "расхождение с другим методом")

    html = figure.to_html(include_plotlyjs=False, full_html=True, div_id="cloud",
                          default_width="100%", default_height="86vh",
                          config=dict(displaylogo=False, responsive=True))
    html = html.replace("</head>", '<script src="plotly.min.js"></script>\n</head>')
    html = html.replace("<body>", "<body>\n" + HEADER % (heading, note) + controls, 1)
    payload = "<script>window.VIEWER = %s;</script>\n" % json.dumps(meta, ensure_ascii=False)
    html = html.replace("</body>", payload + SCRIPT + "\n</body>")
    Path(path).write_text(html, encoding="utf-8")


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
                            "облако смещено относительно GT" % (method, rotation))
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
<p>По одному файлу на сцену. Облака уже приведены в систему координат GT через sim(3) по центрам камер
и раскрашены цветом пикселей. Кнопки сверху переключают, какое облако смотреть (GT, DA3-Giant,
VGGT-Ω или оба предсказания) и как его красить — цветом пикселей или расстоянием до GT
(зелёный — на поверхности, красный — дальше τ). Мышь вращает, колесо зумит, правая кнопка сдвигает.
Файлы самодостаточны, кроме соседнего <code>plotly.min.js</code> — работает и по <code>file://</code>,
без интернета.</p>
<ul>
"""


def index_metrics(stem):
    """A one-line hint under each link: how well the two clouds match the GT lidar."""
    dataset, _, scene = stem.replace("-", " ", 1).partition(" ")
    scene = scene.replace(" ", "-")
    scores = []
    for method in ("da3-giant", "vggt-omega"):
        evaluation = OUTPUTS / dataset / scene / method / "evaluation.json"
        if evaluation.exists():
            scores.append("%s %.3f" % (method, read_json(evaluation)["cloud"]["fscore@1"]))
    return " · F@1: " + ", ".join(scores) if scores else ""


def write_index(out):
    pages = sorted(path.name for path in Path(out).glob("*.html") if path.name != "index.html")
    items = "\n".join('<li><a href="%s">%s</a><span class="hint">%s</span></li>'
                      % (name, name[:-5], index_metrics(name[:-5])) for name in pages)
    Path(out, "index.html").write_text(INDEX_HEAD.format() + items + "\n</ul>\n", encoding="utf-8")
    print("index.html written (%d scenes)" % len(pages), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=["da3-giant", "vggt-omega"])
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plotly_js = Path(__import__("plotly").__file__).parent / "package_data" / "plotly.min.js"
    shutil.copyfile(plotly_js, out / "plotly.min.js")
    for scene in args.scenes:
        present = [method for method in args.methods
                   if (CLOUDS / args.dataset / scene / (method + ".ply")).exists()]
        if not present:
            continue
        figure, meta = scene_figure(args.dataset, scene, present, args.points)
        if figure is None:
            continue
        name = "%s-%s.html" % (args.dataset, scene)
        note = ("Облака после sim(3) по позам камер, до %s точек на облако. "
                "Раскраска «расстояние»: τ = %.3f ед. GT (%s), всё, что дальше, — одним красным."
                % (thousands(args.points), meta["tau"], meta["tau_note"]))
        warnings = pose_warnings(args.dataset, scene, present)
        if warnings:
            note += " <span style='color:#ffb020'>Внимание: " + "; ".join(warnings) + ".</span>"
        write_viewer(out / name, figure, meta, "%s / %s" % (args.dataset, scene), note)
        print(name, "written", flush=True)
    if not args.no_index:
        write_index(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())