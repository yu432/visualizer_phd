"""Shared geometry helpers: layout, hashing, back-projection, voxel downsampling, PLY IO."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "data/inference"
GT = ROOT / "data/gt"
OUTPUTS = ROOT / "outputs"
CLOUDS = ROOT / "clouds"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + chr(10))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def percentile_box(points, low_percentile=1.0, high_percentile=99.0, pad=0.04):
    """Axis-aligned box around the bulk of a cloud, in the cloud's own coordinates.

    A ball around the GT camera centres used to serve as the crop for figures and viewer
    pages. That quietly cut the reconstruction in half whenever the camera trajectory was
    small next to the reconstructed area (ETH3D office keeps 12 % of the points), so the
    render showed a sphere-shaped chunk instead of the scene. Percentiles only drop a few
    stray points and never change the shape of what is kept.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    low = np.percentile(points, low_percentile, axis=0)
    high = np.percentile(points, high_percentile, axis=0)
    margin = pad * float(np.linalg.norm(high - low))
    return low - margin, high + margin


def metric_text(value):
    """A length in the units the clouds are stored in, which are metres.

    The distance colouring used to be labelled with the bare number and "ед. GT", which asks the
    reader to find the ground-truth diagonal before the number means anything. ETH3D lidar and the
    T&T laser scans are both metric, so the viewer legend and the report figures can just say
    metres (and centimetres once tau gets small).
    """
    value = float(value)
    if value >= 10.0:
        return "%.1f м" % value
    if value >= 0.1:
        return "%.2f м" % value
    return "%.1f см" % (value * 100.0)


def view_basis(extrinsics):
    """Orthographic view basis taken from a camera, so GT and prediction share one viewpoint."""
    rotation = np.asarray(extrinsics, dtype=np.float64)[:3, :3]
    right = rotation[0]
    down = rotation[1]
    forward = rotation[2]
    return right, down, forward


def backproject(depth, intrinsics, extrinsics):
    """Back-project depth maps into world coordinates; returns finite (M,3) points."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 3:
        raise ValueError("depth must be (N,H,W), got %s" % (depth.shape,))
    n, height, width = depth.shape
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if intrinsics.shape != (n, 3, 3):
        raise ValueError("intrinsics must be (N,3,3), got %s" % (intrinsics.shape,))
    if extrinsics.shape != (n, 4, 4):
        raise ValueError("extrinsics must be (N,4,4), got %s" % (extrinsics.shape,))
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    out = []
    for index in range(n):
        z = depth[index]
        valid = np.isfinite(z) & (z > 0)
        if not valid.any():
            continue
        k = intrinsics[index]
        ray = np.stack(
            [
                (xs[valid] - k[0, 2]) / k[0, 0] * z[valid],
                (ys[valid] - k[1, 2]) / k[1, 1] * z[valid],
                z[valid],
            ],
            axis=-1,
        )
        rotation, translation = extrinsics[index][:3, :3], extrinsics[index][:3, 3]
        out.append((ray - translation) @ rotation)
    if not out:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(out, axis=0)


def _voxel_order(points, voxel):
    keys = np.floor(points / voxel).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered = keys[order]
    boundaries = np.any(np.diff(ordered, axis=0) != 0, axis=1)
    group = np.concatenate([[0], np.cumsum(boundaries)])
    return order, group


def voxel_downsample_attributed(points, voxel, attrs=None):
    """Average points and optional per-point attributes inside each voxel cell."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0 or voxel <= 0:
        return points, None if attrs is None else np.asarray(attrs)
    order, group = _voxel_order(points, voxel)
    counts = np.bincount(group)
    sums = np.zeros((counts.size, 3))
    np.add.at(sums, group, points[order])
    means = sums / counts[:, None]
    if attrs is None:
        return means, None
    attrs = np.asarray(attrs)
    accumulator = np.zeros((counts.size,) + attrs.shape[1:], dtype=np.float64)
    np.add.at(accumulator, group, attrs[order].astype(np.float64))
    average = accumulator / counts.reshape((-1,) + (1,) * (attrs.ndim - 1))
    return means, average.astype(attrs.dtype)


def voxel_downsample_focused(points, voxel, attrs=None, focus_centre=None, focus_radius=0.0,
                            focus_voxel=0.0):
    """Voxel grid, refined inside a sphere around the object of interest.

    One grid for the whole scene forces a single tradeoff: fine enough for the hero object and the
    cloud is too heavy for a page, coarse enough for the page and the hero dissolves into dust.
    On Tanks and Temples the hero is a few percent of the diagonal and the rest is ground, so the
    fine grid is only paid for inside the sphere.
    """
    points = np.asarray(points, dtype=np.float64)
    if (focus_centre is None or focus_radius <= 0 or focus_voxel <= 0 or focus_voxel >= voxel
            or len(points) == 0):
        return voxel_downsample_attributed(points, voxel, attrs)
    distance = np.linalg.norm(points - np.asarray(focus_centre, dtype=np.float64), axis=1)
    inside = distance <= focus_radius
    if inside.all():
        return voxel_downsample_attributed(points, focus_voxel, attrs)
    if not inside.any():
        return voxel_downsample_attributed(points, voxel, attrs)
    attrs = None if attrs is None else np.asarray(attrs)
    inside_points, inside_attrs = voxel_downsample_attributed(
        points[inside], focus_voxel, None if attrs is None else attrs[inside])
    outside_points, outside_attrs = voxel_downsample_attributed(
        points[~inside], voxel, None if attrs is None else attrs[~inside])
    merged = np.concatenate([outside_points, inside_points])
    if attrs is None:
        return merged, None
    return merged, np.concatenate([outside_attrs, inside_attrs])


def voxel_downsample(points, voxel):
    """Average points inside a voxel grid, deterministically."""
    return voxel_downsample_attributed(points, voxel, None)[0]


def diagonal(points):
    points = np.asarray(points)
    if len(points) == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def write_ply(path, points, colors=None):
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if colors is None:
        vertex = np.core.records.fromarrays(points.T, names="x,y,z")
    else:
        colors = np.asarray(colors, dtype=np.uint8)
        vertex = np.core.records.fromarrays(
            [*points.T, *colors.T], names="x,y,z,red,green,blue"
        )
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def read_ply(path):
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float64)
    names = vertex.data.dtype.names
    if "red" in names:
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1)
        return points, colors.astype(np.uint8)
    return points, None

def load_crop(gt_root):
    """Official region-of-interest volume for a dataset, when the dataset defines one."""
    path = Path(gt_root) / "crop_volume.json"
    return json.loads(path.read_text()) if path.exists() else None


def crop_mask(points, config):
    """Boolean mask of the points inside a Tanks and Temples SelectionPolygonVolume.

    The official evaluation only compares the region the ground truth was cut to, and the
    reconstruction covers the whole surroundings, so without this the accuracy is dominated by
    trees and ground that no laser scan exists for. Colours travel alongside the points, so the
    mask is exposed separately from crop_volume.
    """
    from matplotlib.path import Path as Polygon

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    polygon = np.array([[p[0], p[1]] for p in config["bounding_polygon"]])
    inside = Polygon(polygon).contains_points(points[:, :2])
    axis = config.get("orthogonal_axis", "Z").upper()
    column = {"X": 0, "Y": 1, "Z": 2}[axis]
    low, high = float(config["axis_min"]), float(config["axis_max"])
    if low > high:
        low, high = high, low
    return inside & (points[:, column] >= low) & (points[:, column] <= high)


def crop_volume(points, config):
    """The points inside a Tanks and Temples SelectionPolygonVolume."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points[crop_mask(points, config)]


def image_colours(image_path, height, width, tolerance=0.075) -> np.ndarray:
    """Resample an RGB image onto the depth grid of one frame; returns (H*W, 3) uint8.

    Models that resize to patch multiples (VGGT-Omega rounds to 16 px) shift the aspect by up to
    a few percent, so the guard is relative and tolerant; the colours are resampled anyway.
    """
    from PIL import Image

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        aspect = image.width / image.height
        expected = width / height
        if abs(aspect - expected) > tolerance * expected:
            raise ValueError(
                "%s: aspect %.4f does not match depth aspect %.4f (tolerance %.1f%%)"
                % (Path(image_path).name, aspect, expected, 100.0 * tolerance)
            )
        resized = image.resize((width, height), Image.BILINEAR)
    return np.asarray(resized).reshape(-1, 3)


def backproject_coloured(depths, intrinsics, extrinsics, images, masks=None, tolerance=0.075):
    """Back-project depth maps and keep per-pixel colours aligned with the points.

    The optional masks argument is a per-frame or per-pixel boolean selection applied
    on top of the validity mask, so filtered pixels never reach the cloud.
    """
    points, colours = [], []
    for index in range(len(depths)):
        depth = np.asarray(depths[index], dtype=np.float64)
        height, width = depth.shape
        valid = np.isfinite(depth) & (depth > 0)
        if masks is not None:
            extra = np.asarray(masks)
            extra = extra if extra.shape == valid.shape else extra[index]
            valid &= extra.reshape(valid.shape).astype(bool)
        if not valid.any():
            continue
        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        k = np.asarray(intrinsics[index], dtype=np.float64)
        z = depth[valid]
        ray = np.stack(
            [
                (xs[valid] - k[0, 2]) / k[0, 0] * z,
                (ys[valid] - k[1, 2]) / k[1, 1] * z,
                z,
            ],
            axis=-1,
        )
        rotation, translation = extrinsics[index][:3, :3], extrinsics[index][:3, 3]
        points.append((ray - translation) @ rotation)
        colours.append(image_colours(images[index], height, width, tolerance=tolerance)[valid.reshape(-1)])
    if not points:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(points, axis=0), np.concatenate(colours, axis=0)

def _voxel_keys(points, origin, voxel):
    keys = np.floor((points - origin) / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    return (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]


def density_cap(counts, limit):
    """Tallest per-voxel cap k with sum(min(count, k)) <= limit, or 0 if even k = 1 is too much."""
    ordered = np.sort(counts)
    prefix = np.cumsum(ordered)
    total = len(ordered)

    def capped(k):
        below = int(np.searchsorted(ordered, k, side="left"))
        return (int(prefix[below - 1]) if below else 0) + (total - below) * k

    if capped(1) > limit:
        return 0
    low, high = 1, int(counts.max())
    while low < high:
        middle = (low + high + 1) // 2
        if capped(middle) <= limit:
            low = middle
        else:
            high = middle - 1
    return low


def stratified_subsample(points, colours, limit, seed=0):
    """Reduce a cloud to `limit` points without letting any patch of the scene go bare.

    A uniform random cut keeps the same *fraction* of every patch, so a patch the dense scheme
    covers thinly — the far side of a building that only a couple of its frames see — keeps a
    couple of dots per hand's width and reads as holes, while a patch every frame saw keeps
    hundreds of stacked points nobody can tell apart. On the 100-frame barn a random cut of the
    7.06 M point cloud to 1 M kept 65 % of the occupied 3 cm voxels, where the 16-frame cloud at
    its 750 k budget kept 89 %: the same barn read as a different reconstruction.

    So cap the *density* instead of the point count: voxelise, keep every point of a sparse voxel
    and only the first k of a dense one, with k as tall as the budget allows. Sparse surface keeps
    its detail, dense surface gives up points it did not need, and the budget lands where nobody
    else looks. Any leftover budget — sparse voxels rarely fill it alone — is drawn at random from
    what the cap removed.
    """
    points = np.asarray(points)
    if len(points) <= limit:
        return points, colours
    low, high = np.percentile(points, [0.5, 99.5], axis=0)
    scale = float(np.linalg.norm(high - low)) or 1.0

    # A voxel grid that leaves about half the budget in occupied voxels puts the cap near two
    # points per voxel: fine enough that a thin patch keeps all of it, coarse enough that a patch
    # seen by every frame is genuinely thinned. Six halvings are enough for any cloud here.
    target = max(64.0, 0.45 * limit)
    voxel = scale / 1000.0
    for _ in range(6):
        occupied = np.unique(_voxel_keys(points, low, voxel)).size
        if abs(occupied / target - 1.0) < 0.15:
            break
        voxel = float(np.clip(voxel * np.sqrt(occupied / target), scale / 2e5, scale / 8.0))

    rng = np.random.default_rng(seed)
    keys = _voxel_keys(points, low, voxel)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    cap = density_cap(counts, limit)
    if cap == 0:
        # Even one point per voxel busts the budget: take one point from a random share of them.
        picked = rng.choice(len(starts), size=limit, replace=False)
        index = np.sort(order[starts[picked]])
        return points[index], None if colours is None else colours[index]
    keep = np.arange(len(order)) - np.repeat(starts, counts) < cap
    index = order[keep]
    rest = limit - len(index)
    if rest > 0:
        extra = order[~keep]
        if len(extra) > rest:
            extra = rng.choice(extra, size=rest, replace=False)
        index = np.concatenate([index, extra])
    index.sort()
    return points[index], None if colours is None else colours[index]
