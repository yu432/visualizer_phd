"""Prepare ETH3D-10 for geometry runs: inference inputs and GT kept in separate roots.

Upstream eval/prepare.py is reused so that GT depth is undistorted and intrinsics
match the undistorted images; method/baselines/prepare_eth3d.py is pose-only and
deliberately not touched.
"""
import argparse
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method/geometry"))
from common import GT, INFERENCE, sha256, write_json  # noqa: E402

SCENES = "courtyard delivery_area facade terrace office".split()
HF_REPO = "Livioni/eth3d_omnivggt"
HF_REV = "64154bdca815ef7712f5f2621e231bac7dc99d59"
UPSTREAM = ROOT / "third_party/vggt-omega/eval"


def load_upstream():
    spec = importlib.util.spec_from_file_location("omega_prepare", UPSTREAM / "prepare.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def to_homogeneous(extrinsic) -> np.ndarray:
    """Upstream stores world-to-camera 3x4 matrices; promote to 4x4."""
    matrix = np.asarray(extrinsic, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape != (3, 4):
        raise ValueError("unexpected extrinsic shape %s" % (matrix.shape,))
    return np.concatenate([matrix, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)


def split_scene(staging: Path, scene: str, indices: list[int], force: bool) -> dict:
    source = staging / scene
    inference = INFERENCE / "eth3d" / scene
    gt_root = GT / "eth3d" / scene
    manifest_path = inference / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

    names = (source / "frames.txt").read_text().split()
    cameras = np.load(source / "cameras.npz")
    extrinsics, intrinsics = cameras["extrinsics"], cameras["intrinsics"]
    if not names or len(extrinsics) != len(names):
        raise ValueError("unexpected upstream layout for " + scene)

    (inference / "images").mkdir(parents=True, exist_ok=True)
    (gt_root / "depths").mkdir(parents=True, exist_ok=True)
    frames, kept_extrinsics, kept_intrinsics = [], [], []
    for index in indices:
        name = names[index]
        stem = Path(name).stem
        image = source / "images" / name
        depth_file = source / "depths" / (stem + ".npz")
        if not image.is_file() or not depth_file.is_file():
            raise FileNotFoundError("missing upstream artifact for %s/%s" % (scene, name))
        target = inference / "images" / name
        shutil.copy2(image, target)
        shutil.copy2(depth_file, gt_root / "depths" / (stem + ".npz"))
        with Image.open(target) as opened:
            size = list(opened.size)
        frames.append(
            dict(
                index=index,
                name=name,
                size=size,
                input_sha256=sha256(target),
                source_sha256=sha256(image),
            )
        )
        kept_extrinsics.append(to_homogeneous(extrinsics[index]))
        kept_intrinsics.append(intrinsics[index])

    np.savez_compressed(
        gt_root / "cameras.npz",
        extrinsics=np.asarray(kept_extrinsics),
        intrinsics=np.asarray(kept_intrinsics),
    )
    manifest = dict(
        dataset="eth3d",
        scene=scene,
        frame_selection="omega_eth3d10",
        frame_indices=list(indices),
        source=dict(kind="upstream_prepare", repo=HF_REPO, revision=HF_REV),
        frames=frames,
    )
    write_json(manifest_path, manifest)
    gt_meta_path = gt_root / "gt_meta.json"
    previous = {}
    if gt_meta_path.exists():
        previous = json.loads(gt_meta_path.read_text())
    merged = dict(
        previous,
        dataset="eth3d",
        scene=scene,
        frames=[f["name"] for f in frames],
        has_depth=True,
    )
    write_json(gt_meta_path, merged)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    frame_map = json.loads((UPSTREAM / "frames/eth3d.json").read_text())
    scenes = [scene for scene in args.scenes if scene in frame_map]
    staging = ROOT / "data/_staging/eth3d"
    upstream = load_upstream()
    upstream_args = types.SimpleNamespace(
        dataset="eth3d",
        output=str(staging),
        source="hf",
        raw_root=None,
        archive_dir=None,
        work_dir=str(ROOT / "data/_staging/eth3d_work"),
        hf_repo=HF_REPO,
        hf_revision=HF_REV,
        scenes=scenes,
    )
    upstream._prepare_eth3d(upstream_args)
    for scene in scenes:
        manifest = split_scene(staging, scene, frame_map[scene], args.force)
        print(scene, "prepared", len(manifest["frames"]), "frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
