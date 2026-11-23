"""Prepare Tanks and Temples scenes: RGB frames for inference, COLMAP poses on the GT side."""
import argparse
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GT, INFERENCE, sha256, write_json  # noqa: E402
from tnt_colmap import read_cameras, read_images, read_points  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "data/_staging/tnt"
TARBALL = STAGING / "zips/data/tanksandtemples.tar.gz"
SCENES = ["barn", "courthouse", "ignatius", "meetingroom"]


def extract(scene):
    target = STAGING / "raw"
    scene_root = target / "tanksandtemples" / scene
    if (scene_root / "sparse").is_dir():
        return scene_root
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TARBALL) as archive:
        members = [
            item
            for item in archive.getmembers()
            if item.name.startswith("tanksandtemples/" + scene + "/")
            and not (item.issym() or item.islnk())
        ]
        if not members:
            raise ValueError("scene missing from tarball: " + scene)
        archive.extractall(target, members=members, filter="data")
    return scene_root


def visibility(extrinsics, intrinsics, points, size, radius=0.05):
    """Boolean [frames, points] matrix of sparse points projected inside each image."""
    height, width = size
    rows = []
    for index in range(len(extrinsics)):
        rotation = extrinsics[index][:3, :3]
        translation = extrinsics[index][:3, 3]
        camera_points = (points - translation) @ rotation
        depth = camera_points[:, 2]
        k = intrinsics[index]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = k[0, 0] * camera_points[:, 0] / depth + k[0, 2]
            v = k[1, 1] * camera_points[:, 1] / depth + k[1, 2]
        tolerance = radius * width
        inside = (depth > 0) & (u > -tolerance) & (u < width + tolerance) & (v > -tolerance) & (v < height + tolerance)
        rows.append(inside)
    return np.stack(rows)


def select_frames(visible, count):
    """Greedy max-coverage frame selection; deterministic and order-agnostic."""
    visible = np.asarray(visible, dtype=bool)
    if visible.ndim != 2:
        raise ValueError("visibility matrix must be 2-D")
    count = min(count, len(visible))
    covered = np.zeros(visible.shape[1], dtype=bool)
    chosen = []
    available = np.ones(len(visible), dtype=bool)
    for _ in range(count):
        gains = (visible & ~covered).sum(axis=1).astype(np.float64)
        gains[~available] = -1.0
        best = int(np.argmax(gains))
        if gains[best] <= 0 and chosen:
            remaining = np.flatnonzero(available)
            if remaining.size == 0:
                break
            best = int(remaining[len(remaining) // 2])
        chosen.append(best)
        available[best] = False
        covered |= visible[best]
    return sorted(chosen)
