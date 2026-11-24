"""Render qualitative figures: input frames, aligned cloud comparisons, depth panels."""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import apply_sim3  # noqa: E402
from pose_align import centres  # noqa: E402
from common import (CLOUDS, GT, INFERENCE, OUTPUTS, crop_mask, load_crop,  # noqa: E402
                    metric_text, percentile_box, read_json, read_ply,
                    stratified_subsample, view_basis, voxel_downsample_attributed)

RENDER_POINTS = 320000
VIEW_FRAMES = 8
# The report places every figure at ~10.9 in wide on an A4 landscape page, so the
# figure canvas is resized to that width before saving: text keeps its nominal point
# size on paper and the 320 dpi raster lands at ~300 dpi after placement.
PAGE_WIDTH = 10.9
DPI = 240


def wrap_title(text, width=40):
    """Titles are wrapped, otherwise long labels overflow the canvas and inflate the
    tight bounding box, which in turn shrinks the text when the page scales it."""
    lines = []
    for part in str(text).split("\n"):
        lines.extend(textwrap.wrap(part, width) or [""])
    return "\n".join(lines)


def save(figure, out_path, tight=True, **adjust):
    fig_width, fig_height = figure.get_size_inches()
    figure.set_size_inches(PAGE_WIDTH, fig_height * PAGE_WIDTH / fig_width)
    if tight:
        figure.tight_layout()
    if adjust:
        figure.subplots_adjust(**adjust)
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(figure)


def subsample(points, colours, limit, seed=0):
    """Stratified cap: a random cut here re-thinned whatever region happened to be dense."""
    return stratified_subsample(points, colours, limit, seed=seed)


def render_sample(points, colours, extent, limit=RENDER_POINTS, columns=760, seed=0):
    """Sample a cloud at a constant density per pixel instead of a constant point count.

    A 100-frame cloud of the same scene holds three to twelve times more points than the
    16-frame one, and it covers more of the building. Drawing both from one fixed budget of
    320 000 points would therefore paint the same facade with fewer dots on the dense scheme
    simply because the panel shows more of the scene — the two schemes would look like different
    reconstructions even when they agree. Tying the sampling grid to the panel width (one voxel
    per this many columns) makes points per pixel the same everywhere, and the point budget only
    remains as a cap that a pathological cloud cannot exceed.
    """
    points = np.asarray(points)
    if len(points) <= limit:
        return points, colours
    voxel = float(extent) / float(columns) if extent else 0.0
    if voxel > 0:
        kept, averaged = voxel_downsample_attributed(points, voxel, colours)
        if len(kept) >= 64:
            points, colours = kept, averaged
    return subsample(points, colours, limit, seed)


def robust_limits(clouds, basis, pad=0.05):
    """Percentile-based framing so a few stray points do not shrink the building."""
    u = np.concatenate([cloud @ basis[0] for cloud in clouds])
    v = np.concatenate([-(cloud @ basis[1]) for cloud in clouds])
    u_low, u_high = np.percentile(u, [0.5, 99.5])
    v_low, v_high = np.percentile(v, [0.5, 99.5])
    span = max(u_high - u_low, v_high - v_low)
    pad = pad * span
    return (u_low - pad, u_high + pad), (v_low - pad, v_high + pad)


def cloud_subsample(points, limit=200000, seed=0):
    """Cheap random subsample, enough statistics for a PCA axis and a handful of quantiles."""
    if len(points) <= limit:
        return points
    rng = np.random.default_rng(seed)
    index = rng.choice(len(points), size=limit, replace=False)
    return points[index]


def world_up(extrinsics):
    """World direction that is `up` for the scene, read off the cameras themselves.

    The world frame of every scene comes from COLMAP, so nothing guarantees that +Z is up.
    Hand-held and rig-mounted captures are roughly upright, though, so the mean of the camera
    up-vectors (-y of each world-to-camera rotation) is a stable estimate: it is what the
    photographer would call up, and averaging cancels the individual tilts.
    """
    rotations = np.asarray(extrinsics, dtype=np.float64)[:, :3, :3]
    ups = -rotations[:, 1, :]
    ups = ups / np.maximum(np.linalg.norm(ups, axis=1, keepdims=True), 1e-12)
    mean = ups.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return mean / norm


def cloud_frame_rotation(dataset, scene):
    """Rotation that takes COLMAP-frame directions into the frame the clouds are drawn in.

    Tanks and Temples keeps its cameras and its laser scan in different coordinate systems, and
    pose_align composes the official transform from gt_meta.json into the alignment, so every
    cloud is drawn in the laser frame while cameras.npz still lives in the COLMAP frame. A view
    built from the raw cameras is then rolled by whatever that transform happens to be: the barn
    and courthouse comparison figures used to look down on the scene as if from straight above.
    """
    path = GT / dataset / scene / "gt_meta.json"
    value = read_json(path).get("frame_transform") if path.exists() else None
    if value is None:
        return np.eye(3)
    rotation = np.asarray(value, dtype=np.float64)[:3, :3]
    # A similarity carries a scale; only its rotation is of any use for a viewing direction.
    return rotation / np.cbrt(abs(float(np.linalg.det(rotation))))


def cloud_frame_extrinsics(dataset, scene, extrinsics):
    """The same cameras, turned into the cloud's frame. Directions only, translations unused.

    World directions are the rows of a world-to-camera rotation, so moving them through the frame
    rotation means right-multiplying by its transpose.
    """
    moved = np.asarray(extrinsics, dtype=np.float64).copy()
    moved[:, :3, :3] = moved[:, :3, :3] @ cloud_frame_rotation(dataset, scene).T
    return moved


def scene_basis(points, extrinsics=None, elevation=22.0, seed=0):
    """3/4 aerial orthographic basis chosen from the cloud, not from one camera.

    A camera pose is a bad viewport: on Tanks and Temples the first frame is often a
    ground-level shot of grass, and the resulting panel showed a green smear instead of the
    building. Here the union cloud is reduced with PCA; its dominant horizontal direction gives
    the building's long axis, the view looks across that axis (so the long side fills the panel
    width) from `elevation` degrees above the horizon, and the image `up` stays the scene's up,
    so the horizon is level and no panel is rolled. The side of the axis is picked on the side
    the cameras looked from, so the panels show the surfaces the reconstruction actually saw.
    """
    sample = np.asarray(cloud_subsample(points, 200000, seed), dtype=np.float64)
    up = world_up(extrinsics) if extrinsics is not None else np.array([0.0, 0.0, 1.0])
    centre = np.median(sample, axis=0)
    values, vectors = np.linalg.eigh(np.cov((sample - centre).T))
    order = np.argsort(values)[::-1]
    horizontal = None
    for index in order:
        direction = vectors[:, index]
        flat = direction - float(direction @ up) * up
        length = float(np.linalg.norm(flat))
        # A vertical principal axis says nothing about where to stand; skip it.
        if length > 0.35 * max(float(np.linalg.norm(direction)), 1e-12):
            horizontal = flat / length
            break
    if horizontal is None:
        side = np.cross(up, [0.0, 1.0, 0.0])
        horizontal = side / max(float(np.linalg.norm(side)), 1e-12)
    aside = np.cross(up, horizontal)
    aside /= max(float(np.linalg.norm(aside)), 1e-12)
    forward_flat = aside
    if extrinsics is not None:
        mean_forward = np.asarray(extrinsics, dtype=np.float64)[:, :3, :3][:, 2, :].mean(axis=0)
        mean_forward = mean_forward - float(mean_forward @ up) * up
        if float(mean_forward @ aside) < 0.0:
            forward_flat = -aside
    tilt = np.radians(float(np.clip(elevation, 0.0, 80.0)))
    forward = np.cos(tilt) * forward_flat - np.sin(tilt) * up
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    return right, -np.cross(right, forward), forward


def ground_level(points, up):
    """Height of the scene's floor: the densest height slab in the lower half of the cloud.

    Aerial captures put half their points on the ground, which is why a plain percentile frame
    showed a lawn with a church somewhere in it. The floor is the strongest peak of the height
    histogram below the median, and everything above it is scene content worth framing.
    """
    heights = np.asarray(points, dtype=np.float64) @ up
    low, high = np.percentile(heights, [1.0, 99.0])
    if not np.isfinite(high - low) or high - low < 1e-9:
        return float(low)
    bins = 120
    histogram, edges = np.histogram(np.clip(heights, low, high), bins=bins)
    peak = int(np.argmax(histogram[: bins // 2]))
    return float(0.5 * (edges[peak] + edges[peak + 1]))


def frame_limits(clouds, basis, up, aspect=1.4, trim=0.03, letterbox=0.06):
    """Framing box that shows the scene, not the lawn it stands on.

    The horizontal extent comes from the points above the ground, so a building fills the panel
    width even when the capture also reconstructed a field around it; the ground is kept, but
    only as the strip under the content, and it simply runs out of the frame. `aspect` widens
    the shorter side so equal-aspect panels are not letterboxed by matplotlib at draw time.
    """
    right, down, forward = basis
    points = np.asarray(cloud_subsample(np.concatenate(clouds), 300000, seed=5), dtype=np.float64)
    heights = points @ up
    floor = ground_level(points, up)
    span = float(np.percentile(heights, 99.0) - np.percentile(heights, 1.0))
    above = heights > floor + trim * max(span, 1e-9)
    content = points[above] if np.count_nonzero(above) > 0.05 * len(points) else points
    u = content @ right
    v = -(content @ down)
    u_low, u_high = np.percentile(u, [1.0, 99.0])
    v_low, v_high = np.percentile(v, [1.0, 99.0])
    # Keep a band of floor under the content: without it the building looks like it floats.
    v_low -= 0.18 * max(v_high - v_low, 1e-9)
    u_span = max(u_high - u_low, 1e-9)
    v_span = max(v_high - v_low, 1e-9)
    margin = letterbox * max(u_span, v_span)
    u_low, u_high = u_low - margin, u_high + margin
    v_low, v_high = v_low - margin, v_high + margin
    u_mid, v_mid = 0.5 * (u_low + u_high), 0.5 * (v_low + v_high)
    u_span, v_span = u_high - u_low, v_high - v_low
    if aspect is not None:
        if u_span / v_span < aspect:
            u_span = v_span * aspect
        else:
            v_span = u_span / aspect
    return ((u_mid - 0.5 * u_span, u_mid + 0.5 * u_span),
            (v_mid - 0.5 * v_span, v_mid + 0.5 * v_span))


def fit_aspect(box, aspect):
    """Widen a framing box to a required width/height ratio without cropping anything."""
    (u_low, u_high), (v_low, v_high) = box
    u_span, v_span = u_high - u_low, v_high - v_low
    if aspect is not None:
        if u_span / max(v_span, 1e-9) < aspect:
            extra = 0.5 * (v_span * aspect - u_span)
            u_low, u_high = u_low - extra, u_high + extra
        else:
            extra = 0.5 * (u_span / aspect - v_span)
            v_low, v_high = v_low - extra, v_high + extra
    return (u_low, u_high), (v_low, v_high)


def alignment_params(dataset, scene, method):
    """How a method's cloud is placed on the GT cameras: the stored metric fit, or a pose fit.

    A scene without a laser scan has no evaluation.json, and its clouds are still compared between
    frame schemes, so the placement has to come from the poses there — otherwise the shared
    viewpoint of the scene cannot be built at all and the 16- and 100-frame panels of that scene
    keep their own private framing.
    """
    path = OUTPUTS / dataset / scene / method / "evaluation.json"
    if path.exists():
        evaluation = read_json(path)
        return dict(s=evaluation["alignment"]["s"],
                    R=np.asarray(evaluation["alignment_rotation"]),
                    t=np.asarray(evaluation["alignment_translation"]))
    from pose_align import sim3 as pose_sim3
    return pose_sim3(dataset, scene, method)


def scene_schemes(scene):
    """Every frame scheme present on disk for a scene, shortest name first (the sparse one)."""
    found = []
    for path in CLOUDS.iterdir():
        if path.is_dir() and (path / scene).is_dir() and (GT / path.name / scene).exists():
            found.append(path.name)
    return sorted(found, key=lambda name: (len(name), name))


def scene_reference(scene, elevation=22.0):
    """One viewpoint, one sampling scale and one frame per scene, shared by every frame scheme.

    Each scheme used to pick its own viewpoint: the 16-frame figure looked along its own first
    camera and framed the prediction box of its own cloud, the 100-frame figure did the same with
    a different camera and a larger, noisier cloud. Two panels of the same barn therefore showed
    different sides at different zoom, and the reader could not tell a real difference between the
    16- and 100-frame reconstructions from a difference between two viewports. The reference is
    built from everything on disk for the scene — every scheme's prediction and the laser scan —
    and cached, so both schemes render the same view at the same points per pixel.

    Returns the scene up vector, the orthographic basis, the framing box before aspect fitting and
    the diagonal of the box (the sampling scale for render_sample), or None if nothing is on disk.
    """
    cache = OUTPUTS / "views" / ("%s-%g.json" % (scene, elevation))
    if cache.exists():
        stored = read_json(cache)
        return {"up": np.asarray(stored["up"], dtype=np.float64),
                "basis": [np.asarray(row, dtype=np.float64) for row in stored["basis"]],
                "box": ((float(stored["box"][0][0]), float(stored["box"][0][1])),
                        (float(stored["box"][1][0]), float(stored["box"][1][1]))),
                "extent": float(stored["extent"])}

    schemes = scene_schemes(scene)
    if not schemes:
        return None
    base = schemes[0]
    with np.load(GT / base / scene / "cameras.npz") as handle:
        extrinsics = cloud_frame_extrinsics(base, scene,
                                            np.asarray(handle["extrinsics"], dtype=np.float64))
    samples = []
    for scheme in schemes:
        for method in ("da3-giant", "vggt-omega"):
            cloud_path = display_cloud(scheme, scene, method)
            if not cloud_path.exists():
                continue
            points, _ = read_ply(cloud_path)
            points, _ = subsample(points, None, 200000, seed=7)
            params = alignment_params(scheme, scene, method)
            aligned, _ = crop_to_region(scheme, scene, apply_sim3(points, params), None)
            if len(aligned):
                samples.append(np.asarray(aligned, dtype=np.float64))
    if not samples:
        return None
    union = np.concatenate(samples)
    up = world_up(extrinsics)
    basis = scene_basis(union, extrinsics, elevation=elevation)
    low, high = percentile_box(union)
    extent = float(np.linalg.norm(high - low))
    gt_path = GT / base / scene / "gt_cloud.ply"
    content = [union]
    if gt_path.exists():
        gt_points, _ = read_ply(gt_path)
        gt_points, _ = subsample(gt_points, None, 300000, seed=11)
        box = 0.06 * extent
        inside = np.all((gt_points > low - box) & (gt_points < high + box), axis=1)
        if np.count_nonzero(inside):
            content.append(np.asarray(gt_points[inside], dtype=np.float64))
    box = frame_limits(content, basis, up, aspect=None)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"up": up.tolist(), "basis": [row.tolist() for row in basis],
                                 "box": [[box[0][0], box[0][1]], [box[1][0], box[1][1]]],
                                 "extent": extent,
                                 "schemes": schemes}, indent=2, sort_keys=True) + chr(10))
    return {"up": up, "basis": list(basis), "box": box, "extent": extent}


def draw_cloud(ax, points, colours, basis, title, limits=None, depth_tint=False, point_size=1.3,
               wrap=40, equal_aspect=False):
    """Painter's-algorithm scatter: far points first, image-plane orthographic projection."""
    right, down, forward = basis
    u = points @ right
    v = -(points @ down)
    depth = points @ forward
    order = np.argsort(-depth)
    u, v, depth = u[order], v[order], depth[order]
    colours = colours[order] if colours is not None else None
    if colours is None:
        colours = np.full((len(u), 3), 150, dtype=np.uint8)
    else:
        colours = np.asarray(colours)[:, :3]
    if depth_tint:
        low, high = np.percentile(depth, [2, 98])
        ramp = np.clip((depth - low) / max(high - low, 1e-9), 0.0, 1.0)
        colours = (0.35 * colours + 0.65 * (255.0 * plt.cm.viridis(1.0 - ramp)[:, :3])).astype(np.uint8)
    ax.scatter(u, v, c=colours / 255.0, s=point_size, linewidths=0, marker=".", rasterized=True)
    ax.set_facecolor("#101010")
    ax.set_title(wrap_title(title, wrap), fontsize=11.5)
    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
    else:
        ax.autoscale()
        ax.set_aspect("equal", adjustable="datalim")
    if equal_aspect:
        # Square units in both directions: the three panels of a comparison then differ only in
        # what was reconstructed, never in how the scene was stretched to fill the axes.
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        # A panel is a picture, not a plot: a white frame around the cloud only adds an edge that
        # the card thumbnail would inherit.
        side.set_visible(False)


def inputs_figure(dataset, scene, out_path):
    manifest = read_json(INFERENCE / dataset / scene / "manifest.json")
    frames = [frame["name"] for frame in manifest["frames"]][:VIEW_FRAMES]
    columns = min(4, len(frames))
    rows = int(np.ceil(len(frames) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.1 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes:
        axis.axis("off")
    for axis, name in zip(axes, frames):
        with Image.open(INFERENCE / dataset / scene / "images" / name) as opened:
            axis.imshow(opened.convert("RGB"))
        axis.set_title(name, fontsize=9.5, pad=4)
    figure.suptitle("%s / %s: входные кадры (%d из %d)" % (dataset, scene, len(frames), len(manifest["frames"])),
                    fontsize=13)
    save(figure, out_path, tight=False, hspace=0.22, wspace=0.05, top=0.86, bottom=0.02, left=0.01, right=0.99)


LABEL = {"da3-giant": "DA3-Giant", "vggt-omega": "VGGT-Ω"}


def gt_label(dataset):
    """"Lidar" is ETH3D's term; Tanks and Temples ships a terrestrial laser scan."""
    return "GT: лазерный скан T&T" if dataset.startswith("tnt") else "GT: лидар ETH3D"


def crop_to_region(dataset, scene, points, colours):
    """T&T predictions cover the whole surroundings; the laser GT covers the official polygon."""
    region = load_crop(GT / dataset / scene)
    if not region:
        return points, colours
    mask = crop_mask(points, region)
    return points[mask], None if colours is None else colours[mask]


def display_cloud(dataset, scene, method):
    """The cloud a picture is drawn from: the display copy when it exists, else the metric one.

    Numbers come from the metric cloud, which keeps every pixel above the protocol confidence
    quantile. The display copy is the same reconstruction with a stricter quantile, so the
    low-confidence tail that smears thin structures against the sky is gone (regen_clouds.py).
    A figure drawn from the metric cloud would show artefacts the metrics do not reward.
    """
    display = CLOUDS / dataset / scene / (method + "-display.ply")
    return display if display.exists() else CLOUDS / dataset / scene / (method + ".ply")


def clouds_figure(dataset, scene, method, out_path):
    evaluation = read_json(OUTPUTS / dataset / scene / method / "evaluation.json")
    gt_points, gt_colours = read_ply(GT / dataset / scene / "gt_cloud.ply")
    pred_points, pred_colours = read_ply(display_cloud(dataset, scene, method))
    params = dict(s=evaluation["alignment"]["s"], R=np.asarray(evaluation["alignment_rotation"]),
                  t=np.asarray(evaluation["alignment_translation"]))
    aligned = apply_sim3(pred_points, params)
    aligned, pred_colours = crop_to_region(dataset, scene, aligned, pred_colours)
    from scipy.spatial import cKDTree

    # Shared with the other frame scheme of the same scene: same viewpoint, same points per pixel.
    reference = scene_reference(scene)
    if reference is not None:
        basis = reference["basis"]
    else:
        with np.load(GT / dataset / scene / "cameras.npz") as handle:
            extrinsics = cloud_frame_extrinsics(dataset, scene, np.asarray(handle["extrinsics"]))
        basis = view_basis(extrinsics[0])

    # The lidar ground truth covers the whole scan area; crop it to the region the
    # ten frames actually reconstructed so both panels show the same buildings.
    low, high = aligned.min(axis=0), aligned.max(axis=0)
    box = 0.06 * float(np.linalg.norm(high - low))
    inside = np.all((gt_points > low - box) & (gt_points < high + box), axis=1)
    gt_points = gt_points[inside]
    gt_colours = None if gt_colours is None else gt_colours[inside]
    # One sampling grid for all three panels, tied to the box the panels are framed on: the
    # prediction, the lidar and the distance colouring then show the same density per pixel.
    extent = reference["extent"] if reference is not None else float(np.linalg.norm(high - low))
    gt_points, gt_colours = render_sample(gt_points, gt_colours, extent)
    aligned, pred_colours = render_sample(aligned, pred_colours, extent, seed=1)
    xlim, ylim = (reference["box"] if reference is not None
                  else robust_limits([gt_points, aligned], basis))
    cloud = evaluation["cloud"]
    tau = 0.01 * evaluation["ground_diagonal"]
    distance = cKDTree(gt_points).query(aligned, k=1)[0]
    error = np.clip(distance / tau, 0.0, 1.0)
    error_colours = (255.0 * plt.cm.RdYlGn_r(error)[:, :3]).astype(np.uint8)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    draw_cloud(axes[0], gt_points, gt_colours, basis,
               "%s, %d точек" % (gt_label(dataset), cloud["gt"]), limits=(xlim, ylim), point_size=1.1)
    draw_cloud(axes[1], aligned, pred_colours, basis,
               "%s: %d точек" % (method, cloud["pred"]), limits=(xlim, ylim), point_size=1.1)
    draw_cloud(axes[2], aligned, error_colours, basis,
               "расстояние до GT (τ = %s)" % metric_text(tau), limits=(xlim, ylim), point_size=1.1)
    figure.suptitle("%s / %s: облако против GT после sim(3) по позам, F@1%% = %.3f (диагональ GT %.2f)" % (
        dataset, scene, cloud["fscore@1"], evaluation["ground_diagonal"]), fontsize=13)
    save(figure, out_path, tight=False, top=0.80, bottom=0.02, left=0.01, right=0.99, wspace=0.04)


def pred_cloud_figure(dataset, scene, method, out_path):
    """Prediction-only cloud, placed in the GT camera frame with the pose sim(3)."""
    from pose_align import sim3 as pose_sim3

    pred_points, pred_colours = read_ply(display_cloud(dataset, scene, method))
    params = pose_sim3(dataset, scene, method)
    aligned = apply_sim3(pred_points, params)
    # Percentile box, not a ball around the camera centres: see common.percentile_box.
    low, high = percentile_box(aligned)
    reference = scene_reference(scene)
    extent = reference["extent"] if reference is not None else float(np.linalg.norm(high - low))
    aligned, pred_colours = render_sample(aligned, pred_colours, extent, seed=3)
    keep = np.all((aligned > low) & (aligned < high), axis=1)
    aligned, pred_colours = aligned[keep], pred_colours[keep]
    with np.load(GT / dataset / scene / "cameras.npz") as handle:
        extrinsics = cloud_frame_extrinsics(dataset, scene,
                                            np.asarray(handle["extrinsics"], dtype=np.float64))
    basis = view_basis(extrinsics[0])
    forwards = extrinsics[:, :3, :3][:, 2, :]
    angles = np.degrees(np.arccos(np.clip(forwards @ (forwards[0] / np.linalg.norm(forwards[0])), -1.0, 1.0)))
    other = int(np.argmax(angles))
    basis_other = view_basis(extrinsics[other])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if reference is not None:
        # Same viewpoint and frame as every other figure of the scene, so the 16- and 100-frame
        # versions of this panel show the same side of the scene instead of two private ones.
        basis = reference["basis"]
        first_limits = fit_aspect(reference["box"], 1.6)
        first_title = "%s: общий ракурс сцены, %d точек, x%.2f sim(3)" % (method, len(aligned), params["s"])
    else:
        first_limits = robust_limits([aligned], basis)
        first_title = "%s: камера GT 0, %d точек, x%.2f sim(3)" % (method, len(aligned), params["s"])
    draw_cloud(axes[0], aligned, pred_colours, basis, first_title,
               limits=first_limits, point_size=1.8, wrap=56)
    draw_cloud(axes[1], aligned, pred_colours, basis_other,
               "то же облако, камера GT %d: самый другой ракурс, %.0f°" % (other, angles[other]),
               limits=robust_limits([aligned], basis_other), point_size=1.8, wrap=56)
    figure.suptitle("%s / %s: облако предсказания (GT-облако для сцены недоступно)" % (dataset, scene), fontsize=13)
    save(figure, out_path, tight=False, top=0.80, bottom=0.02, left=0.01, right=0.99, wspace=0.04)


def depth_figure(dataset, scene, method, out_path):
    root = OUTPUTS / dataset / scene / method
    inference = read_json(root / "inference.json")
    names = [frame["name"] if isinstance(frame, dict) else frame for frame in inference["frames"]]
    depth = np.load(root / "depth.npy")
    evaluation = read_json(root / "evaluation.json")
    scale = evaluation["scale_correction"]
    picks = [0, len(names) // 2] if len(names) > 1 else [0]
    # One row of four panels (GT | prediction per frame) keeps the figure wide and
    # short, so it survives being placed at full page width in the report.
    figure, axes = plt.subplots(1, 2 * len(picks), figsize=(3.1 * 2 * len(picks), 2.9))
    axes = np.atleast_1d(axes)
    for column, index in enumerate(picks):
        gt = np.load(GT / dataset / scene / "depths" / (Path(names[index]).stem + ".npz"))["depth"]
        valid = np.isfinite(gt) & (gt > 0)
        low, high = np.percentile(gt[valid], [5, 95])
        # Sparse lidar: show only valid pixels, on a dark face so the ~8% of
        # measured pixels read as a survey of the scene instead of a white page.
        shown_gt = np.ma.masked_invalid(np.where(valid, gt, np.nan))
        left, right = axes[2 * column], axes[2 * column + 1]
        left.imshow(shown_gt, cmap="turbo", vmin=low, vmax=high)
        left.set_facecolor("#111111")
        left.set_title("%s, GT (валидно %.1f%%)" % (Path(names[index]).stem, 100.0 * valid.mean()), fontsize=10.5)
        right.imshow(depth[index] * scale, cmap="turbo", vmin=low, vmax=high)
        right.set_title("%s, масштаб x%.2f" % (method, scale), fontsize=10.5)
        for axis in (left, right):
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("%s / %s: карты глубины, общая шкала turbo на пару кадров (синий — ближе, красный — дальше)\n"
                    "AbsRel с масштабом и сдвигом %.4f, без подгонки %.3f" % (
                        dataset, scene, evaluation["depth"]["scale_shift"]["AbsRel"],
                        evaluation["depth"]["metric"]["AbsRel"]), fontsize=12.5)
    save(figure, out_path, tight=False, top=0.76, bottom=0.02, left=0.01, right=0.99, wspace=0.03)


def pose_figure(dataset, scene, method, out_path):
    """Top-down camera-centre plot: GT vs aligned prediction."""
    poses = read_json(OUTPUTS / dataset / scene / method / "pose_evaluation.json")
    with np.load(GT / dataset / scene / "cameras.npz") as handle:
        gt_extrinsics = np.asarray(handle["extrinsics"], dtype=np.float64)
    root = OUTPUTS / dataset / scene / method
    pred_extrinsics = np.load(root / "extrinsics.npy")
    gt_centres = centres(gt_extrinsics)
    pred_centres = centres(pred_extrinsics)
    aligned = apply_sim3(pred_centres, dict(s=poses["alignment"]["s"],
                                            R=np.asarray(poses["alignment"]["r"]),
                                            t=np.asarray(poses["alignment"]["t"])))
    # The trajectory is often close to a plane or a line, so plot it in its own
    # principal plane rather than an arbitrary x/z projection.
    centre = gt_centres.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((gt_centres - centre).T))
    axes = vectors[:, np.argsort(values)[::-1]][:, :2]
    gt_2d = (gt_centres - centre) @ axes
    aligned_2d = (aligned - centre) @ axes
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6),
                                         gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    left.plot(gt_2d[:, 0], gt_2d[:, 1], "o-", ms=6, lw=0.8, label="GT (COLMAP)", color="#1f77b4")
    left.plot(aligned_2d[:, 0], aligned_2d[:, 1], "x", ms=8, label=method, color="#d62728")
    left.set_aspect("equal", adjustable="datalim")
    left.set_title("центры камер в плоскости траектории (ед. GT)", fontsize=11.5)
    left.legend(fontsize=10)
    left.set_xlabel("главная ось 1", fontsize=11)
    left.set_ylabel("главная ось 2", fontsize=11)
    frames = np.arange(1, len(gt_centres) + 1)
    translation = np.array([frame["translation_fraction"] for frame in poses["per_frame"]]) * 100.0
    rotation = np.array([frame["rotation_deg"] for frame in poses["per_frame"]])
    right.bar(frames - 0.2, translation, width=0.4, color="#d62728", label="ошибка центра, % протяжённости")
    right.set_xlabel("кадр", fontsize=11)
    right.set_ylabel("ошибка центра, % протяжённости GT", color="#d62728", fontsize=11)
    right.tick_params(axis="y", labelcolor="#d62728", labelsize=10)
    twin = right.twinx()
    twin.bar(frames + 0.2, rotation, width=0.4, color="#1f77b4", label="ошибка вращения, °")
    twin.set_ylabel("ошибка вращения, °", color="#1f77b4", fontsize=11)
    twin.tick_params(axis="y", labelcolor="#1f77b4", labelsize=10)
    right.set_title("в среднем: %.2f°, %.2f%% протяжённости" % (
        poses["rotation"]["mean_deg"], 100.0 * poses["translation"]["mean_fraction"]), fontsize=11.5)
    figure.suptitle("%s / %s: позы предсказания против GT" % (dataset, scene), fontsize=13)
    save(figure, out_path)


def sky_figure(dataset, scene, method, out_path, picks=2):
    """Show the SegFormer sky mask on the frames with the largest sky fraction."""
    meta = read_json(OUTPUTS / dataset / scene / method / "sky_meta.json")
    masks = np.load(OUTPUTS / dataset / scene / method / "sky_masks.npy", mmap_mode="r")
    fractions = np.asarray(meta["sky_fraction_per_frame"])
    order = np.argsort(-fractions)[:picks]
    # One wide row of (frame, mask) pairs: the figure stays short enough for the page.
    figure, axes = plt.subplots(1, 2 * len(order), figsize=(2.9 * 2 * len(order), 2.6))
    axes = np.atleast_1d(axes)
    for column, index in enumerate(order):
        name = meta["frames"][index]
        with Image.open(INFERENCE / dataset / scene / "images" / name) as opened:
            image = np.asarray(opened.convert("RGB"))
        mask = np.asarray(masks[index])
        resized = np.asarray(Image.fromarray(image).resize((mask.shape[1], mask.shape[0])))
        overlay = resized.copy()
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
        left, right = axes[2 * column], axes[2 * column + 1]
        left.imshow(resized)
        left.set_title(Path(name).stem, fontsize=10.5)
        right.imshow(overlay)
        right.set_title("маска неба: %.1f%% кадра" % (100.0 * mask.mean()), fontsize=10.5)
        for axis in (left, right):
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("%s / %s: сегментация неба (SegFormer-B1 Cityscapes, в среднем %.1f%% кадра)" % (
        dataset, scene, 100.0 * float(meta["sky_fraction_mean"])), fontsize=12.5)
    save(figure, out_path, tight=False, top=0.76, bottom=0.02, left=0.01, right=0.99, wspace=0.02)


def compare_clouds_figure(dataset, scene, methods, out_path, elevation=22.0):
    """One 3/4 aerial view, GT next to every method, so the report can compare them directly.

    The viewpoint is chosen from the cloud itself (see scene_basis) instead of from a camera
    pose, every panel shares it, and the framing trims the stray points and the far edge of the
    ground plane so the building, not the ground, fills the panel.
    """
    gt_path = GT / dataset / scene / "gt_cloud.ply"
    gt_points, gt_colours = read_ply(gt_path) if gt_path.exists() else (None, None)
    with np.load(GT / dataset / scene / "cameras.npz") as handle:
        extrinsics = cloud_frame_extrinsics(dataset, scene,
                                            np.asarray(handle["extrinsics"], dtype=np.float64))
    aligned = {}
    for method in methods:
        cloud_path = display_cloud(dataset, scene, method)
        if not cloud_path.exists():
            continue
        points, colours = read_ply(cloud_path)
        params = alignment_params(dataset, scene, method)
        evaluation_path = OUTPUTS / dataset / scene / method / "evaluation.json"
        score = read_json(evaluation_path)["cloud"]["fscore@1"] if evaluation_path.exists() else None
        point_cloud = apply_sim3(points, params)
        point_cloud, colours = crop_to_region(dataset, scene, point_cloud, colours)
        aligned[method] = (point_cloud, colours, score)
    if not aligned:
        return
    union = np.concatenate([value[0] for value in aligned.values()])
    # One sampling grid for every panel, taken from the box the panels are framed on: without it
    # a cloud that covers more ground (a hundred frames instead of sixteen) would be thinned to
    # the same point count over a larger area and read as sparse, which is a sampling artefact
    # and not a property of the reconstruction.
    low, high = percentile_box(union)
    reference = scene_reference(scene, elevation=elevation)
    extent = reference["extent"] if reference is not None else float(np.linalg.norm(high - low))
    panels = []
    if gt_points is not None:
        box = 0.06 * extent
        inside = np.all((gt_points > low - box) & (gt_points < high + box), axis=1)
        gt_points = gt_points[inside]
        gt_colours = None if gt_colours is None else gt_colours[inside]
        gt_points, gt_colours = render_sample(gt_points, gt_colours, extent, seed=2)
        panels.append((gt_label(dataset), gt_points, gt_colours))
    for method, (points, colours, score) in aligned.items():
        label = LABEL.get(method, method)
        if score is not None:
            label = "%s (F@1%% = %.3f)" % (label, score)
        # Keep the pixel colours: a grey cloud next to a coloured GT panel reads as a bug.
        points, colours = render_sample(points, colours, extent, seed=1)
        panels.append((label, points, colours))
    basis = scene_basis(union, extrinsics, elevation=elevation)
    up = -basis[1]
    clouds = [panel[1] for panel in panels]
    # The panel is made roughly as wide as the framed content, so a wide building is not
    # letterboxed into a square with empty sky above and below it (that is what made the T&T
    # panels look like a strip in the middle of nothing at thumbnail size).
    if reference is not None:
        # Same frame as every other scheme of this scene, so two figures of one barn are directly
        # comparable instead of being two different viewports of it.
        basis, up = reference["basis"], reference["up"]
        raw = reference["box"]
    else:
        raw = frame_limits(clouds, basis, up, aspect=None)
    content_aspect = (raw[0][1] - raw[0][0]) / max(raw[1][1] - raw[1][0], 1e-9)
    panel_aspect = float(np.clip(content_aspect, 1.25, 2.1))
    limits = fit_aspect(raw, panel_aspect)
    width = 4.9 * len(panels)
    wspace = 0.035
    left, right, bottom = 0.01, 0.99, 0.02
    # Reserve the title strip in inches, not as a fraction of the canvas: a short wide figure
    # otherwise ends up with the panel titles on top of the suptitle.
    header = 0.95
    axes_width = width * (right - left) / (len(panels) + (len(panels) - 1) * wspace)
    height = (header + axes_width / panel_aspect) / (1.0 - bottom)
    height = float(np.clip(height, 3.1, 8.0))
    top = 1.0 - header / height
    figure, axes = plt.subplots(1, len(panels), figsize=(width, height))
    axes = np.atleast_1d(axes)
    for axis, (title, points, colours) in zip(axes, panels):
        draw_cloud(axis, points, colours, basis, title, limits=limits, point_size=1.4,
                   equal_aspect=True, wrap=34)
    figure.suptitle("%s / %s: облака с одного ракурса после sim(3) по позам камер" % (dataset, scene),
                    fontsize=13, y=1.0 - 0.30 / height)
    save(figure, out_path, tight=False, top=top, bottom=bottom, left=left, right=right, wspace=wspace)


def gt_overlay_figure(dataset, scene, methods, out_path):
    """Top and side orthographic views: prediction outlines against the laser ground truth.

    The 3D panels elsewhere show surfaces, which hides the thing that matters on Tanks and
    Temples — whether the reconstruction traces the scanned walls. In a top view the laser
    footprint and the predicted walls lie on the same lines, so the agreement is readable
    at a glance; the side view shows the same for height.
    """
    gt_path = GT / dataset / scene / "gt_cloud.ply"
    if not gt_path.exists():
        return
    gt_points, _ = read_ply(gt_path)
    gt_sample, _ = subsample(gt_points, None, 120000, seed=0)
    panels = [("вид сверху (XY)", 0, 1), ("вид сбоку (XZ)", 0, 2)]
    colours = {"da3-giant": "#1f6fd0", "vggt-omega": "#d97706"}
    drawn = []
    from pose_align import sim3 as pose_sim3

    for method in methods:
        cloud_path = display_cloud(dataset, scene, method)
        evaluation_path = OUTPUTS / dataset / scene / method / "evaluation.json"
        if not cloud_path.exists():
            continue
        points, _ = read_ply(cloud_path)
        if evaluation_path.exists():
            evaluation = read_json(evaluation_path)
            params = dict(s=evaluation["alignment"]["s"], R=np.asarray(evaluation["alignment_rotation"]),
                          t=np.asarray(evaluation["alignment_translation"]))
        else:
            params = pose_sim3(dataset, scene, method)
        aligned, _ = crop_to_region(dataset, scene, apply_sim3(points, params), None)
        drawn.append((method, subsample(aligned, None, 200000, seed=2)[0]))
    if not drawn:
        return
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 6.2))
    handles = None
    for axis, (title, first, second) in zip(axes, panels):
        # Predictions first, the laser outline on top: an outline buried under a dense cloud
        # is exactly the thing these panels exist to show.
        for method, points in drawn:
            axis.scatter(points[:, first], points[:, second], s=1.5, c=colours.get(method, "#7a7a7a"),
                         linewidths=0, alpha=0.45, label=LABEL.get(method, method))
        laser = axis.scatter(gt_sample[:, first], gt_sample[:, second], s=1.0, c="#1c1c1c",
                             linewidths=0, alpha=0.65, label="лазерный GT", zorder=3)
        axis.set_title("%s: %s" % (scene, title))
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=9)
        if handles is None:
            # Built from the first panel only, and the laser goes last: reading the list off
            # axes[0] after the loop picked up the second panel's laser and listed it twice.
            handles = [artist for artist in axis.collections if artist is not laser] + [laser]
    labels = [artist.get_label() for artist in handles]
    axes[0].legend(handles, labels, markerscale=6, fontsize=9, loc="upper right", framealpha=0.9)
    figure.suptitle("%s / %s: предсказания и лазерный GT в одной системе координат" % (dataset, scene),
                    fontsize=12.5)
    save(figure, out_path, tight=True)


def sky_cloud_figure(dataset, scene, method, out_path):
    """Same viewpoint, cloud with and without the sky filter, to show what the mask removes."""
    from pose_align import sim3 as pose_sim3

    params = pose_sim3(dataset, scene, method)
    with np.load(GT / dataset / scene / "cameras.npz") as handle:
        extrinsics = cloud_frame_extrinsics(dataset, scene,
                                            np.asarray(handle["extrinsics"], dtype=np.float64))
    basis = view_basis(extrinsics[0])
    # No radius crop: a ball around the camera centres cut the cloud in half whenever the
    # trajectory was small next to the reconstructed area (office keeps 12 % of the points),
    # and the frame then showed a sphere-shaped chunk instead of the scene. The panels are
    # framed by robust_limits below, which drops stray points without touching the shape.
    panels = []
    for tag, label in (("", "с маской неба"), ("-nosky", "без маски неба")):
        path = CLOUDS / dataset / scene / (method + tag + ".ply")
        if not path.exists():
            return
        points, colours = read_ply(path)
        aligned = apply_sim3(points, params)
        natural = len(aligned)
        aligned, colours = subsample(aligned, colours, RENDER_POINTS, seed=2)
        panels.append((label, aligned, colours, natural))
    limits = robust_limits([panel[1] for panel in panels], basis)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, (label, points, colours, natural) in zip(axes, panels):
        draw_cloud(axis, points, colours, basis,
                   "%s: %s\n%d точек, показано %d" % (method, label, natural, len(points)),
                   limits=limits, point_size=1.8, wrap=56)
    figure.suptitle("%s / %s: что убирает маска неба" % (dataset, scene), fontsize=13)
    save(figure, out_path, tight=False, top=0.80, bottom=0.02, left=0.01, right=0.99, wspace=0.04)


# One token per figure kind, so a re-render after a plotting tweak does not have to redo the
# expensive depth/pose/sky panels of every scene. --only takes these names; the default is all.
FIGURE_KINDS = ("inputs", "clouds", "depth", "pose", "sky", "skycloud", "compare", "check")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--method", default="da3-giant")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="+", choices=FIGURE_KINDS, default=None,
                        help="render only these figure kinds (default: every kind)")
    parser.add_argument("--elevation", type=float, default=22.0,
                        help="degrees above the horizon for the compare-figure viewpoint")
    args = parser.parse_args()
    methods = args.methods or [args.method]
    wanted = set(args.only) if args.only else set(FIGURE_KINDS)

    def want(kind):
        return kind in wanted

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.titlesize"] = 11.5
    plt.rcParams["axes.labelsize"] = 11
    for scene in args.scenes:
        prefix = "%s-%s" % (args.dataset, scene)
        if want("inputs"):
            inputs_figure(args.dataset, scene, out / (prefix + "-inputs.png"))
        for method in methods:
            root = OUTPUTS / args.dataset / scene / method
            if not (root / "inference.json").exists():
                continue
            stem = "%s-%s-" % (prefix, method)
            if (root / "evaluation.json").exists():
                if want("clouds"):
                    clouds_figure(args.dataset, scene, method, out / (stem + "clouds.png"))
                # Tanks and Temples ships a laser cloud only, no per-frame GT depth.
                if want("depth") and (GT / args.dataset / scene / "depths").exists():
                    depth_figure(args.dataset, scene, method, out / (stem + "depth.png"))
            elif want("clouds"):
                pred_cloud_figure(args.dataset, scene, method, out / (stem + "clouds.png"))
            if want("pose") and (root / "pose_evaluation.json").exists():
                pose_figure(args.dataset, scene, method, out / (stem + "poses.png"))
            if (root / "sky_meta.json").exists():
                if want("sky"):
                    sky_figure(args.dataset, scene, method, out / (stem + "sky.png"))
                if want("skycloud"):
                    sky_cloud_figure(args.dataset, scene, method, out / (stem + "skycloud.png"))
        if want("compare"):
            compare_clouds_figure(args.dataset, scene, methods, out / (prefix + "-compare.png"),
                                  elevation=args.elevation)
        if want("check"):
            gt_overlay_figure(args.dataset, scene, methods, out / (prefix + "-check.png"))
        print(args.dataset, scene, "figures written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())