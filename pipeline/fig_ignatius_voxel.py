"""Render the ignatius statue zone at three voxel sizes.

Companion to diag_ignatius_statue.py: the same points, aligned to the lawn, cropped to a cylinder
around the statue and drawn at the voxel of the old page, of the current page and of the finest
setting measured in README.md ("Почему ignatius выглядит кашей, а не статуей").

Run from the repo root:
  MPLCONFIGDIR=$PWD/.mpl .venv/bin/python method/geometry/fig_ignatius_voxel.py \
      --out docs/experiments/2026-11-22-geometry-clouds/figures/tnt-ignatius-statue-voxel.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import diag_ignatius_statue as diag  # noqa: E402
from common import diagonal, voxel_downsample_attributed  # noqa: E402

ROWS = ((0.0015, "was on the page"), (0.0004, "on the page now"), (0.0003, "finest measured"))
ZONE = 0.16


def platform_level(aligned, axis, radius=0.12):
    """Height of the point cluster the statue stands on, so the profile starts at zero."""
    radial = np.hypot(aligned[:, 0] - axis[0], aligned[:, 2] - axis[2])
    core = aligned[radial < radius, 1]
    counts, edges = np.histogram(core, bins=60)
    return float(edges[int(np.argmax(counts))])


def crater(points, colours, rotation, centre, level, axis):
    """Points and colours inside the statue cylinder, in the ground frame."""
    aligned = (points - centre) @ rotation.T
    height = aligned[:, 1] - level
    radial = np.hypot(aligned[:, 0] - axis[0], aligned[:, 2] - axis[2])
    inside = (radial < ZONE) & (height > -0.12)
    return aligned[inside], height[inside], colours[inside], int(inside.sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", default="vggt-omega")
    args = parser.parse_args()

    _, _, extrinsics, points, colours = diag.load()
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    cameras = -np.einsum("nij,nj->ni", np.transpose(rotation, (0, 2, 1)), translation)
    rot, centre, level = diag.ground_frame(points, cameras)
    axis = (cameras.mean(0) - centre) @ rot.T
    aligned_full = (points - centre) @ rot.T
    level = platform_level(aligned_full, axis)
    del aligned_full
    diag_points = diagonal(points)
    cos, sin = np.cos(0.87), np.sin(0.87)

    figure, axes = plt.subplots(len(ROWS), 3, figsize=(11.5, 7.1), dpi=150)
    for row, (fraction, note) in enumerate(ROWS):
        voxel = fraction * diag_points
        reduced, averaged = voxel_downsample_attributed(points, voxel, colours)
        aligned, height, tint, count = crater(reduced, averaged, rot, centre, level, axis)
        if tint is None or tint.dtype == object:
            tint = np.full((len(aligned), 3), 0.25)
        tint = np.asarray(tint, dtype=float) / 255.0
        title = "%s: voxel %.4f (%.2f%% of diagonal), %d pts in the zone" % (
            note, voxel, 100.0 * fraction, count)
        left = aligned[:, 0]
        oblique = left * cos + aligned[:, 2] * sin
        oblique_axis = axis[0] * cos + axis[2] * sin
        for column, (dx, dy, name) in enumerate((
                (left, height, "front"),
                (left, aligned[:, 2], "top view x-z"),
                (oblique, height, "oblique 50deg"))):
            ax = axes[row, column]
            ax.scatter(dx, dy, s=1.0, c=tint, alpha=0.85, linewidths=0)
            if name != "top view x-z":
                ax.axhline(0.0, color="#d9a22b", lw=0.9)
            ax.set_title(title if column == 0 else name, fontsize=7.2)
            ax.set_xlabel("x", fontsize=7)
            if column == 0:
                ax.set_ylabel("height above the platform", fontsize=7)
            ax.tick_params(labelsize=6.5)
            if name == "top view x-z":
                ax.set_aspect("equal", adjustable="datalim")
            else:
                middle = axis[0] if name == "front" else oblique_axis
                ax.set_xlim(middle - 0.22, middle + 0.22)
                ax.set_ylim(-0.06, 0.46)
        print(row, fraction, "voxel %.4f zone %d total %d" % (voxel, count, len(reduced)), flush=True)

    figure.suptitle(
        "ignatius (%s): statue zone r < %.2f units, ground-aligned, same viewpoint, source colours"
        % (args.method, ZONE), fontsize=8.5)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
