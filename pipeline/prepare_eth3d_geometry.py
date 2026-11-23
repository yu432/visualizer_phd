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


def restage(staging: Path, scene: str, indices: list[int], upstream, raw_root: Path) -> int:
    """Copy images and undistorted GT depth for frames the upstream stager skipped.

    Upstream keeps every camera in frames.txt and cameras.npz but only writes the ten
    selected images and depths, so a denser frame set has to be materialised from the raw
    download. Depth goes through the same undistortion as the ten frame protocol, otherwise
    the dense selection would be compared against geometry in a different camera model.
    """
    scene_output = staging / scene
    names = (scene_output / "frames.txt").read_text().split()
    missing = [
        (index, names[index])
        for index in indices
        if not (scene_output / "images" / names[index]).is_file()
        or not (scene_output / "depths" / (Path(names[index]).stem + ".npz")).is_file()
    ]
    if not missing:
        return 0
    scene_root = upstream._find_scene(raw_root, scene)
    undistorted_root = scene_root / "dslr_calibration_undistorted"
    distorted_root = scene_root / "dslr_calibration_jpg"
    undistorted_cameras = upstream._read_colmap_cameras(undistorted_root / "cameras.txt")
    undistorted_images = upstream._read_colmap_images(undistorted_root / "images.txt")
    distorted_cameras = upstream._read_colmap_cameras(distorted_root / "cameras.txt")
    distorted_images = upstream._read_colmap_images(distorted_root / "images.txt")
    (scene_output / "images").mkdir(parents=True, exist_ok=True)
    (scene_output / "depths").mkdir(parents=True, exist_ok=True)
    for _, name in missing:
        stem = Path(name).stem
        item = undistorted_images["dslr_images_undistorted/" + name]
        camera = undistorted_cameras[item["camera_id"]]
        distorted_item = distorted_images["dslr_images/" + stem + ".JPG"]
        distorted_camera = distorted_cameras[distorted_item["camera_id"]]
        image = scene_output / "images" / name
        if not image.is_file():
            shutil.copy2(scene_root / "images" / "dslr_images_undistorted" / name, image)
        depth_path = scene_output / "depths" / (stem + ".npz")
        if depth_path.is_file():
            continue
        raw_path = scene_root / "ground_truth_depth" / "dslr_images" / name
        raw_depth = np.fromfile(raw_path, dtype=np.float32)
        expected = distorted_camera.width * distorted_camera.height
        if raw_depth.size != expected:
            raise ValueError("invalid ETH3D depth file: %s" % raw_path)
        raw_depth = np.nan_to_num(
            raw_depth.reshape(distorted_camera.height, distorted_camera.width),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        np.savez_compressed(
            depth_path, depth=upstream._undistort_depth(raw_depth, distorted_camera, camera)
        )
    return len(missing)


def split_scene(staging: Path, scene: str, indices: list[int], force: bool, dataset: str = "eth3d",
               selection: str = "omega_eth3d10") -> dict:
    source = staging / scene
    inference = INFERENCE / dataset / scene
    gt_root = GT / dataset / scene
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
        dataset=dataset,
        scene=scene,
        frame_selection=selection,
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
        dataset=dataset,
        scene=scene,
        frames=[f["name"] for f in frames],
        has_depth=True,
    )
    write_json(gt_meta_path, merged)
    return manifest


def share_gt_cloud(scene: str, source: str, dataset: str) -> None:
    """Reuse the ground-truth cloud of another selection of the same scene.

    On ETH3D the ground truth cloud is a back-projection of the ground truth depth of the
    selected frames, so a denser frame set would also change the reference and the comparison
    would no longer isolate the prediction. Copying the cloud keeps the reference fixed.
    """
    if source == dataset:
        return
    from_source = GT / source / scene
    to_target = GT / dataset / scene
    cloud = to_target / "gt_cloud.ply"
    if not cloud.exists():
        shutil.copy2(from_source / "gt_cloud.ply", cloud)
    meta_path = to_target / "gt_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    reference = json.loads((from_source / "gt_meta.json").read_text())
    for key in ("diagonal", "kind", "points", "raw_points", "stride", "cloud_sha256", "voxel"):
        if key in reference:
            meta[key] = reference[key]
    meta.update(dataset=dataset, scene=scene, gt_source=source,
                gt_source_manifest_sha256=reference.get("source_manifest_sha256"))
    write_json(meta_path, meta)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--dataset", default="eth3d",
                        help="name of the inference and ground truth roots to write")
    parser.add_argument("--budget", type=int, default=0,
                        help="number of frames per scene; 0 keeps the upstream ten frame list")
    parser.add_argument("--gt-source", default=None,
                        help="dataset whose ground truth cloud is reused as the reference")
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
    raw_root, _ = upstream._eth3d_raw(upstream_args, tuple(scenes))
    upstream._prepare_eth3d(upstream_args)
    for scene in scenes:
        if args.budget:
            count = len((staging / scene / "frames.txt").read_text().split())
            if args.budget < count:
                # Evenly spaced rather than the first N: the staged order is capture order, so a
                # prefix would cover only one side of the scene.
                indices = sorted({round(i * (count - 1) / (args.budget - 1)) for i in range(args.budget)})
            else:
                indices = list(range(count))
            selection = "evenly_spaced_%d" % len(indices)
        else:
            indices = frame_map[scene]
            selection = "omega_eth3d10"
        restaged = restage(staging, scene, indices, upstream, raw_root)
        manifest = split_scene(staging, scene, indices, args.force, dataset=args.dataset,
                               selection=selection)
        if args.gt_source:
            share_gt_cloud(scene, args.gt_source, args.dataset)
        print(scene, "prepared", len(manifest["frames"]), "frames into", args.dataset,
              "(restaged %d)" % restaged, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
