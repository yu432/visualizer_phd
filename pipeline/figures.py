"""Render qualitative figures: input frames, aligned cloud comparisons, depth panels."""
import argparse
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
                    percentile_box, read_json, read_ply, view_basis)

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
    if len(points) <= limit:
        return points, colours
    rng = np.random.default_rng(seed)
    index = rng.choice(len(points), size=limit, replace=False)
    index.sort()
    return points[index], None if colours is None else colours[index]


def robust_limits(clouds, basis, pad=0.05):
    """Percentile-based framing so a few stray points do not shrink the building."""
    u = np.concatenate([cloud @ basis[0] for cloud in clouds])
    v = np.concatenate([-(cloud @ basis[1]) for cloud in clouds])
    u_low, u_high = np.percentile(u, [0.5, 99.5])
    v_low, v_high = np.percentile(v, [0.5, 99.5])
    span = max(u_high - u_low, v_high - v_low)
    pad = pad * span
    return (u_low - pad, u_high + pad), (v_low - pad, v_high + pad)


def draw_cloud(ax, points, colours, basis, title, limits=None, depth_tint=False, point_size=1.3,
               wrap=40):
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
    ax.set_xticks([])
    ax.set_yticks([])


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


def clouds_figure(dataset, scene, method, out_path):
    evaluation = read_json(OUTPUTS / dataset / scene / method / "evaluation.json")
    gt_points, gt_colours = read_ply(GT / dataset / scene / "gt_cloud.ply")
    pred_points, pred_colours = read_ply(CLOUDS / dataset / scene / (method + ".ply"))
    params = dict(s=evaluation["alignment"]["s"], R=np.asarray(evaluation["alignment_rotation"]),
                  t=np.asarray(evaluation["alignment_translation"]))
    aligned = apply_sim3(pred_points, params)
    aligned, pred_colours = crop_to_region(dataset, scene, aligned, pred_colours)
    with np.load(GT / dataset / scene / "cameras.npz") as handle:
        basis = view_basis(np.asarray(handle["extrinsics"])[0])
    from scipy.spatial import cKDTree

    # The lidar ground truth covers the whole scan area; crop it to the region the
    # ten frames actually reconstructed so both panels show the same buildings.
    low, high = aligned.min(axis=0), aligned.max(axis=0)
    box = 0.06 * float(np.linalg.norm(high - low))
    inside = np.all((gt_points > low - box) & (gt_points < high + box), axis=1)
    gt_points = gt_points[inside]
    gt_colours = None if gt_colours is None else gt_colours[inside]
    gt_points, gt_colours = subsample(gt_points, gt_colours, RENDER_POINTS)
    aligned, pred_colours = subsample(aligned, pred_colours, RENDER_POINTS, seed=1)
    xlim, ylim = robust_limits([gt_points, aligned], basis)
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
               "расстояние до GT (τ = %.3f ед. GT)" % tau, limits=(xlim, ylim), point_size=1.1)
    figure.suptitle("%s / %s: облако против GT после sim(3) по позам, F@1%% = %.3f (диагональ GT %.2f)" % (
        dataset, scene, cloud["fscore@1"], evaluation["ground_diagonal"]), fontsize=13)
    save(figure, out_path, tight=False, top=0.80, bottom=0.02, left=0.01, right=0.99, wspace=0.04)


def pred_cloud_figure(dataset, scene, method, out_path):
    """Prediction-only cloud, placed in the GT camera frame with the pose sim(3)."""
    from pose_align import sim3 as pose_sim3

    pred_points, pred_colours = read_ply(CLOUDS / dataset / scene / (method + ".ply"))
    params = pose_sim3(dataset, scene, method)
    aligned = apply_sim3(pred_points, params)
    aligned, pred_colours = subsample(aligned, pred_colours, RENDER_POINTS, seed=3)
    # Percentile box, not a ball around the camera centres: see common.percentile_box.
    low, high = percentile_box(aligned)
    keep = np.all((aligned > low) & (aligned < high), axis=1)
    aligned, pred_colours = aligned[keep], pred_colours[keep]
    with np.load(GT / dataset / scene / "cameras.npz") as handle:
        extrinsics = np.asarray(handle["extrinsics"], dtype=np.float64)
    basis = view_basis(extrinsics[0])
    forwards = extrinsics[:, :3, :3][:, 2, :]
    angles = np.degrees(np.arccos(np.clip(forwards @ (forwards[0] / np.linalg.norm(forwards[0])), -1.0, 1.0)))
    other = int(np.argmax(angles))
    basis_other = view_basis(extrinsics[other])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    draw_cloud(axes[0], aligned, pred_colours, basis,
               "%s: камера GT 0, %d точек, x%.2f sim(3)" % (method, len(aligned), params["s"]),
               limits=robust_limits([aligned], basis), point_size=1.8, wrap=56)
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


def compare_clouds_figure(dataset, scene, methods, out_path):
    """One viewpoint, GT next to every method, so the report can compare them directly."""
    gt_path = GT / dataset / scene / "gt_cloud.ply"
    gt_points, gt_colours = read_ply(gt_path) if gt_path.exists() else (None, None)
    with np.load(GT / dataset / scene / "cameras.npz") as handle:
        basis = view_basis(np.asarray(handle["extrinsics"])[0])
    from pose_align import sim3 as pose_sim3

    aligned = {}
    for method in methods:
        path = OUTPUTS / dataset / scene / method / "evaluation.json"
        cloud_path = CLOUDS / dataset / scene / (method + ".ply")
        if not cloud_path.exists():
            continue
        points, colours = read_ply(cloud_path)
        if path.exists():
            evaluation = read_json(path)
            params = dict(s=evaluation["alignment"]["s"], R=np.asarray(evaluation["alignment_rotation"]),
                          t=np.asarray(evaluation["alignment_translation"]))
            score = evaluation["cloud"]["fscore@1"]
        else:
            params = pose_sim3(dataset, scene, method)
            score = None
        point_cloud = apply_sim3(points, params)
        point_cloud, colours = crop_to_region(dataset, scene, point_cloud, colours)
        # Keep the pixel colours: a grey cloud next to a coloured GT panel reads as a bug.
        sampled, sampled_colours = subsample(point_cloud, colours, RENDER_POINTS, seed=1)
        aligned[method] = (sampled, sampled_colours, score)
    if not aligned:
        return
    union = np.concatenate([value[0] for value in aligned.values()])
    panels = []
    if gt_points is not None:
        low, high = union.min(axis=0), union.max(axis=0)
        box = 0.06 * float(np.linalg.norm(high - low))
        inside = np.all((gt_points > low - box) & (gt_points < high + box), axis=1)
        gt_points = gt_points[inside]
        gt_colours = None if gt_colours is None else gt_colours[inside]
        gt_points, gt_colours = subsample(gt_points, gt_colours, RENDER_POINTS)
        panels.append((gt_label(dataset), gt_points, gt_colours))
    for method, (points, colours, score) in aligned.items():
        label = method if score is None else "%s (F@1%% = %.3f)" % (method, score)
        panels.append((label, points, colours))
    limits = robust_limits([panel[1] for panel in panels], basis)
    figure, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.2))
    axes = np.atleast_1d(axes)
    for axis, (title, points, colours) in zip(axes, panels):
        draw_cloud(axis, points, colours, basis, title, limits=limits)
    figure.suptitle("%s / %s: облака с одного ракурса после sim(3) по позам камер" % (dataset, scene), fontsize=13)
    save(figure, out_path, tight=False, top=0.86, bottom=0.02, left=0.01, right=0.99, wspace=0.04)


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
        cloud_path = CLOUDS / dataset / scene / (method + ".ply")
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
        extrinsics = np.asarray(handle["extrinsics"], dtype=np.float64)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--method", default="da3-giant")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    methods = args.methods or [args.method]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.titlesize"] = 11.5
    plt.rcParams["axes.labelsize"] = 11
    for scene in args.scenes:
        prefix = "%s-%s" % (args.dataset, scene)
        inputs_figure(args.dataset, scene, out / (prefix + "-inputs.png"))
        for method in methods:
            root = OUTPUTS / args.dataset / scene / method
            if not (root / "inference.json").exists():
                continue
            stem = "%s-%s-" % (prefix, method)
            if (root / "evaluation.json").exists():
                clouds_figure(args.dataset, scene, method, out / (stem + "clouds.png"))
                # Tanks and Temples ships a laser cloud only, no per-frame GT depth.
                if (GT / args.dataset / scene / "depths").exists():
                    depth_figure(args.dataset, scene, method, out / (stem + "depth.png"))
            else:
                pred_cloud_figure(args.dataset, scene, method, out / (stem + "clouds.png"))
            if (root / "pose_evaluation.json").exists():
                pose_figure(args.dataset, scene, method, out / (stem + "poses.png"))
            if (root / "sky_meta.json").exists():
                sky_figure(args.dataset, scene, method, out / (stem + "sky.png"))
                sky_cloud_figure(args.dataset, scene, method, out / (stem + "skycloud.png"))
        compare_clouds_figure(args.dataset, scene, methods, out / (prefix + "-compare.png"))
        gt_overlay_figure(args.dataset, scene, methods, out / (prefix + "-check.png"))
        print(args.dataset, scene, "figures written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())