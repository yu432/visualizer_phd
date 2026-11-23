"""Prepare Tanks and Temples scenes: RGB frames for inference, COLMAP poses kept on the GT side."""
import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GT, INFERENCE, sha256, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TARBALL = ROOT / "data/_staging/tnt/zips/data/tanksandtemples.tar.gz"
SCENES = ["barn", "courthouse", "ignatius", "meetingroom"]


def quaternion_to_rotation(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_cameras(path):
    cameras = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        camera_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = [float(value) for value in parts[4:]]
        if model in ("PINHOLE", "OPENCV"):
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            raise ValueError("unsupported camera model " + model)
        cameras[camera_id] = dict(
            model=model, width=width, height=height,
            intrinsics=[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        )
    return cameras


def read_images(path):
    lines = [line for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]
    records = []
    for line in lines[::2]:  # COLMAP text format alternates image lines and POINTS2D lines
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        quaternion = [float(value) for value in parts[1:5]]
        translation = np.array([float(value) for value in parts[5:8]])
        camera_id = int(parts[8])
        name = parts[9]
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = quaternion_to_rotation(quaternion)
        extrinsics[:3, 3] = translation
        records.append(dict(image_id=image_id, name=name, camera_id=camera_id, extrinsics=extrinsics))
    return records


def read_points(path, limit=200000):
    points = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        points.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(points) >= limit:
            break
    return np.asarray(points, dtype=np.float64)
