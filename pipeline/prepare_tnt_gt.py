
"""Build a laser ground-truth cloud for a Tanks and Temples scene.

T&T ships, per training scene, a laser scan (<Scene>.ply), the crop volume the official
evaluation uses (<Scene>.json) and the similarity that maps the official COLMAP frame to the
laser frame (<Scene>_trans.txt). The scans are mirrored publicly (johnwlambert/gtsfm-datasets-mirror
GitHub releases), which is what makes a cloud-to-GT comparison possible here at all.

The script crops the scan with the official polygon, downsamples it to a comparable point count,
and records the official frame transform in gt_meta.json so evaluate.py can place predictions in
the laser frame.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GT, crop_volume, diagonal, read_json, read_ply, sha256, voxel_downsample, write_json, write_ply  # noqa: E402

TARGET_POINTS = 4_000_000
VOXELS = (0.02, 0.03, 0.05, 0.08, 0.12, 0.2)


def crop_and_describe(points, config):
    cropped = crop_volume(points, config)
    return cropped, dict(
        polygon=[[p[0], p[1]] for p in config["bounding_polygon"]],
        axis=config.get("orthogonal_axis", "Z").upper(),
        axis_min=float(config["axis_min"]), axis_max=float(config["axis_max"]),
        kept=int(len(cropped)))


def downsample(points, np_random):
    for voxel in VOXELS:
        candidate = voxel_downsample(points, voxel)
        if len(candidate) <= TARGET_POINTS:
            return candidate, voxel
    index = np_random.choice(len(candidate), TARGET_POINTS, replace=False)
    return candidate[np.sort(index)], VOXELS[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--cloud", required=True, help="laser scan .ply (<Scene>.ply)")
    parser.add_argument("--crop-json", required=True, help="crop volume (<Scene>.json)")
    parser.add_argument("--transform", required=True, help="official COLMAP -> laser similarity (<Scene>_trans.txt)")
    parser.add_argument("--source", default="johnwlambert/gtsfm-datasets-mirror release")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    scene_root = GT / "tnt" / args.scene
    meta_path = scene_root / "gt_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else dict(dataset="tnt", scene=args.scene)

    points, _ = read_ply(Path(args.cloud))
    config = json.loads(Path(args.crop_json).read_text())
    cropped, crop_info = crop_and_describe(points, config)
    cloud, voxel = downsample(cropped, np.random.default_rng(0))
    cloud = cloud.astype(np.float32)

    write_ply(scene_root / "gt_cloud.ply", cloud.astype(np.float64))
    transform = np.loadtxt(args.transform)
    np.savetxt(scene_root / "colmap_to_gt.txt", transform, fmt="%.18e")
    (scene_root / "crop_volume.json").write_text(json.dumps(config, indent=2, sort_keys=True) + chr(10))

    meta.update(
        has_cloud=True,
        kind="laser_scan_crop",
        points=int(len(cloud)),
        raw_points=int(len(points)),
        cropped_points=int(len(cropped)),
        voxel=float(voxel),
        diagonal=float(diagonal(cloud.astype(np.float64))),
        cloud_sha256=sha256(scene_root / "gt_cloud.ply"),
        raw_cloud_sha256=sha256(Path(args.cloud)),
        crop=crop_info,
        raw_diagonal=float(diagonal(points)),
        frame_transform=transform.tolist(),
        frame_transform_sha256=sha256(Path(args.transform)),
        frame_transform_note="official COLMAP frame -> laser frame, as shipped in <Scene>_trans.txt",
        source_cloud=args.source,
        note=args.note or "laser scan cropped with the official polygon; poses come from the official COLMAP model",
    )
    write_json(meta_path, meta)
    print("%s: %d raw -> %d cropped -> %d points, voxel %.3f m, diagonal %.2f m" % (
        args.scene, len(points), len(cropped), len(cloud), voxel, meta["diagonal"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())