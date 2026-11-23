"""Build the Tanks and Temples preparation entry point."""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GT, INFERENCE, sha256, write_json  # noqa: E402
from tnt_colmap import read_cameras, read_images, read_points  # noqa: E402
from tnt_select import extract, select_frames, visibility  # noqa: E402

SCENES = ["barn", "courthouse", "ignatius", "meetingroom"]
FRAME_BUDGET = 16


IMAGE_DIRS = ("images", "images_2", "images_4", "images_8")


def image_roots(scene_root: Path):
    """Candidate image directories, best resolution first."""
    roots = [scene_root / name for name in IMAGE_DIRS]
    roots.extend(sorted(p for p in scene_root.glob("images*") if p.is_dir() and p not in roots))
    return [root for root in roots if root.is_dir()]


def scene_frames(scene_root: Path, records, cameras):
    """Resolve COLMAP entries to on-disk images, in COLMAP order.

    Tanks and Temples tarballs ship only downscaled image folders for some
    scenes, so intrinsics are rescaled to whatever resolution is available.
    """
    roots = image_roots(scene_root)
    if not roots:
        raise ValueError("no image directory found for " + str(scene_root))
    kept = []
    for record in records:
        basename = Path(record["name"]).name
        for root in roots:
            candidate = root / basename
            if candidate.is_file():
                kept.append((record, candidate))
                break
    if not kept:
        raise ValueError("no COLMAP images found on disk for " + str(scene_root))
    return kept


def scaled_intrinsics(camera: dict, path: Path) -> tuple:
    """Scale a COLMAP camera to the resolution of the image actually on disk."""
    with Image.open(path) as opened:
        width, height = opened.size
    scale_x = width / camera["width"]
    scale_y = height / camera["height"]
    if abs(scale_x - scale_y) > 0.02 * max(scale_x, scale_y):
        raise ValueError("anisotropic resize for " + str(path) + ": %.4f vs %.4f" % (scale_x, scale_y))
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).copy()
    intrinsics[0, :] *= scale_x
    intrinsics[1, :] *= scale_y
    intrinsics[2, :] = [0.0, 0.0, 1.0]
    return intrinsics, (height, width)


def prepare(scene: str, budget: int, force: bool) -> dict:
    scene_root = extract(scene)
    inference = INFERENCE / "tnt" / scene
    gt_root = GT / "tnt" / scene
    manifest_path = inference / "manifest.json"
    if manifest_path.exists() and not force:
        import json

        return json.loads(manifest_path.read_text())
    sparse = scene_root / "sparse" / "0"
    cameras = read_cameras(sparse / "cameras.txt")
    records = read_images(sparse / "images.txt")
    points = read_points(sparse / "points3D.txt", limit=50000)
    kept = scene_frames(scene_root, records, cameras)
    scaled = [scaled_intrinsics(cameras[record["camera_id"]], path) for record, path in kept]
    sizes = [shape for _, shape in scaled]
    if len(set(sizes)) != 1:
        raise ValueError("mixed image sizes in " + scene)
    shape = sizes[0]
    extrinsics = np.stack([record["extrinsics"] for record, _ in kept])
    intrinsics = np.stack([value for value, _ in scaled])
    visible = visibility(extrinsics, intrinsics, points, shape)
    indices = select_frames(visible, budget)
    coverage = float(visible[indices].any(axis=0).mean())

    (inference / "images").mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    frames, kept_extrinsics, kept_intrinsics = [], [], []
    for index in indices:
        record, source = kept[index]
        target = inference / "images" / Path(record["name"]).name
        shutil.copy2(source, target)
        with Image.open(target) as opened:
            size = list(opened.size)
        frames.append(dict(index=index, name=target.name, size=size, input_sha256=sha256(target)))
        kept_extrinsics.append(extrinsics[index])
        kept_intrinsics.append(intrinsics[index])
    np.savez_compressed(
        gt_root / "cameras.npz",
        extrinsics=np.asarray(kept_extrinsics),
        intrinsics=np.asarray(kept_intrinsics),
    )
    write_json(
        gt_root / "gt_meta.json",
        dict(
            dataset="tnt",
            scene=scene,
            frames=[frame["name"] for frame in frames],
            has_depth=False,
            has_cloud=False,
            frame_selection=dict(kind="sparse_point_coverage", budget=budget, coverage=coverage),
            source=dict(kind="colmap_sparse", scene_root=str(scene_root)),
            note="Tanks and Temples GT point clouds require the official form download; only COLMAP poses are local",
        ),
    )
    manifest = dict(
        dataset="tnt",
        scene=scene,
        frame_selection=dict(kind="sparse_point_coverage", budget=budget, coverage=coverage),
        source=dict(kind="alexxxcr/tanks_temples tarball", scene_root=str(scene_root)),
        frames=frames,
    )
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--budget", type=int, default=FRAME_BUDGET)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for scene in args.scenes:
        manifest = prepare(scene, args.budget, args.force)
        print(scene, "prepared", len(manifest["frames"]), "frames",
              "coverage %.3f" % manifest["frame_selection"]["coverage"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())