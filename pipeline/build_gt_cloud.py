"""Build the GT point cloud for one scene from undistorted GT depth maps."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    GT,
    INFERENCE,
    backproject_coloured,
    diagonal,
    read_json,
    sha256,
    voxel_downsample_attributed,
    write_json,
    write_ply,
)


def build(dataset: str, scene: str, stride: int, voxel: float, force: bool) -> dict:
    gt_root = GT / dataset / scene
    inference = INFERENCE / dataset / scene
    meta_path = gt_root / "gt_meta.json"
    if meta_path.exists() and not force:
        meta = read_json(meta_path)
        if meta.get("cloud_sha256"):
            return meta
    manifest = read_json(inference / "manifest.json")
    names = [frame["name"] for frame in manifest["frames"]]
    cameras = np.load(gt_root / "cameras.npz")
    extrinsics, intrinsics = cameras["extrinsics"], cameras["intrinsics"]
    if extrinsics.shape[1:] == (3, 4):
        bottom = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (len(extrinsics), 1, 1))
        extrinsics = np.concatenate([extrinsics, bottom], axis=1)
    if len(extrinsics) != len(names):
        raise ValueError("camera/frame count mismatch for " + scene)
    depths = []
    for name in names:
        payload = np.load(gt_root / "depths" / (Path(name).stem + ".npz"))
        depths.append(np.asarray(payload["depth"], dtype=np.float64)[::stride, ::stride])
    scaled = intrinsics.copy()
    scaled[:, :2, :] /= stride
    points, colours = backproject_coloured(
        depths, scaled, extrinsics, [inference / "images" / name for name in names]
    )
    if len(points) < 1000:
        raise ValueError("GT cloud too small for " + scene)
    reduced, kept_colours = voxel_downsample_attributed(points, voxel, colours)
    cloud_path = gt_root / "gt_cloud.ply"
    write_ply(cloud_path, reduced, kept_colours)
    meta = dict(
        dataset=dataset,
        scene=scene,
        frames=names,
        has_depth=True,
        stride=stride,
        voxel=float(voxel),
        raw_points=int(len(points)),
        points=int(len(reduced)),
        diagonal=diagonal(reduced),
        cloud_sha256=sha256(cloud_path),
        source_manifest_sha256=sha256(inference / "manifest.json"),
        kind="ground_truth_depth_backprojection",
    )
    write_json(meta_path, meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--voxel", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for scene in args.scenes:
        meta = build(args.dataset, scene, args.stride, args.voxel, args.force)
        print(scene, "gt points", meta["points"], "diagonal %.4f" % meta["diagonal"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())