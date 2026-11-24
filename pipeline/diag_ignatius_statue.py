"""Diagnose why ignatius (Tanks and Temples) does not reconstruct its central statue.

Reproduces the numbers quoted in docs/experiments/2026-11-22-geometry-clouds/README.md
("Почему ignatius выглядит кашей, а не статуей"):

1. depth relief inside the statue silhouette is nearly flat per view;
2. in 3D the statue zone holds almost no point above the ground plane;
3. the voxel size only decides how coarse that flat patch looks.

Run from the repo root:  .venv/bin/python method/geometry/diag_ignatius_statue.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    INFERENCE,
    OUTPUTS,
    backproject_coloured,
    diagonal,
    read_json,
    voxel_downsample_attributed,
)

SCENE = "ignatius"
METHOD = "vggt-omega"


def load():
    out = OUTPUTS / "tnt" / SCENE / METHOD
    inference = INFERENCE / "tnt" / SCENE
    manifest = read_json(inference / "manifest.json")
    names = [frame["name"] for frame in manifest["frames"]]
    depth = np.load(out / "depth.npy")
    intrinsics = np.load(out / "intrinsics.npy")
    extrinsics = np.load(out / "extrinsics.npy")
    confidence = np.load(out / "depth_conf.npy")
    masks = confidence >= np.quantile(confidence, 0.2)
    valid = np.isfinite(depth) & (depth > 0)
    cap = np.quantile(depth[valid], 0.95)
    masks &= valid & (depth <= cap)
    sky = out / "sky_masks.npy"
    if sky.exists():
        masks &= ~np.load(sky).astype(bool)
    images = [inference / "images" / name for name in names]
    points, colours = backproject_coloured(depth, intrinsics, extrinsics, images, masks)
    return depth, intrinsics, extrinsics, points, colours


def ground_frame(points, cameras):
    """Rotation taking the dominant plane normal to +y, and that plane's level."""
    centre = (np.percentile(points, 1, axis=0) + np.percentile(points, 99, axis=0)) / 2
    centred = points[::7] - centre
    _, vectors = np.linalg.eigh(centred.T @ centred)
    normal = vectors[:, 0]
    if float(np.median((cameras - np.median(points, axis=0)) @ normal)) < 0:
        normal = -normal  # camera must end up on the +y side
    cross = np.cross(normal, [0.0, 1.0, 0.0])
    sine = np.linalg.norm(cross)
    cosine = float(normal @ [0.0, 1.0, 0.0])
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    rotation = np.eye(3) + skew + skew @ skew * ((1 - cosine) / (sine * sine))
    aligned = (points - centre) @ rotation.T
    near = np.hypot(aligned[:, 0], aligned[:, 2]) < 0.6
    counts, edges = np.histogram(aligned[near, 1], bins=80)
    return rotation, centre, edges[int(np.argmax(counts))]


def main():
    depth, intrinsics, extrinsics, points, colours = load()
    print("raw points %d, diagonal %.2f" % (len(points), diagonal(points)))
    for index in (0, 13):
        frame = depth[index]
        column = frame[:, int(0.45 * depth.shape[2])]
        statue = column[column < 0.5 * np.median(frame)]
        print("frame %2d depth along statue column: %.3f..%.3f (x%.2f)"
              % (index, statue.min(), statue.max(), statue.max() / statue.min()))
    cameras_world = np.array(
        [-extrinsics[i][:3, :3].T @ extrinsics[i][:3, 3] for i in range(len(depth))])
    rotation, centre, ground = ground_frame(points, cameras_world)
    # the statue stands in the middle of the camera ring, so the ring centroid is the axis
    axis = (cameras_world.mean(0) - centre) @ rotation.T
    cameras = (cameras_world - centre) @ rotation.T
    # local ground reference: the platform/lawn ring around the statue, not a global fit
    aligned_all = (points - centre) @ rotation.T
    radius_all = np.hypot(aligned_all[:, 0] - axis[0], aligned_all[:, 2] - axis[2])
    ground = float(np.median(aligned_all[(radius_all > 0.18) & (radius_all < 0.35), 1]))
    for fraction in (0.0015, 0.0007, 0.0003):
        voxel = fraction * diagonal(points)
        reduced, _ = voxel_downsample_attributed(points, voxel, colours)
        aligned = (reduced - centre) @ rotation.T
        radius = np.hypot(aligned[:, 0] - axis[0], aligned[:, 2] - axis[2])
        heights = aligned[:, 1] - ground
        inner = radius < 0.12
        print("voxel %.4f: %7d points; r<0.12: %6d points, height p50 %+.3f p90 %+.3f p99 %+.3f"
              % (voxel, len(reduced), inner.sum(), *np.percentile(heights[inner], [50, 90, 99])))
    for low, high in ((0.00, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 0.50)):
        band = (radius_all >= low) & (radius_all < high)
        heights = aligned_all[band, 1] - ground
        print("  r %.2f-%.2f: %6d points, height p50 %+.3f p90 %+.3f p99 %+.3f max %+.3f"
              % (low, high, band.sum(), *np.percentile(heights, [50, 90, 99]), heights.max()))
    print("camera ring: radius %.3f, height above the platform ring %+.3f..%+.3f"
          % (float(np.median(np.hypot(cameras[:, 0] - axis[0], cameras[:, 2] - axis[2]))),
             cameras[:, 1].min() - ground, cameras[:, 1].max() - ground))


if __name__ == "__main__":
    main()
