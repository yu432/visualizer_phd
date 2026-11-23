"""Score predicted camera poses against COLMAP ground truth after a global sim(3) alignment."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import apply_sim3, umeyama  # noqa: E402
from common import GT, OUTPUTS, read_json, sha256, write_json  # noqa: E402
from pose_align import centres  # noqa: E402


def rotation_angle(left, right):
    """Angle in degrees between two rotation matrices."""
    relative = np.einsum("nji,njk->nik", left, right)
    trace = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def evaluate(dataset, scene, method):
    root = OUTPUTS / dataset / scene / method
    inference = read_json(root / "inference.json")
    names = [frame["name"] if isinstance(frame, dict) else frame for frame in inference["frames"]]
    pred_extrinsics = np.load(root / "extrinsics.npy")
    pred_intrinsics = np.load(root / "intrinsics.npy")
    if len(names) != len(pred_extrinsics):
        raise ValueError("%s/%s: %d names vs %d extrinsics" % (dataset, scene, len(names), len(pred_extrinsics)))

    gt_root = GT / dataset / scene
    gt_meta = read_json(gt_root / "gt_meta.json")
    with np.load(gt_root / "cameras.npz") as handle:
        gt_extrinsics = np.asarray(handle["extrinsics"], dtype=np.float64)
        gt_intrinsics = np.asarray(handle["intrinsics"], dtype=np.float64)
    order = {name: index for index, name in enumerate(gt_meta["frames"])}
    missing = [name for name in names if name not in order]
    if missing:
        raise ValueError("%s/%s: frames absent from GT cameras: %s" % (dataset, scene, missing[:3]))
    index = np.array([order[name] for name in names])
    gt_extrinsics, gt_intrinsics = gt_extrinsics[index], gt_intrinsics[index]

    pred_centres, gt_centres = centres(pred_extrinsics), centres(gt_extrinsics)
    scale, rotation, translation = umeyama(pred_centres, gt_centres)
    aligned_centres = apply_sim3(pred_centres, dict(s=scale, R=rotation, t=translation))
    extent = np.linalg.norm(gt_centres.max(axis=0) - gt_centres.min(axis=0))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("%s/%s: degenerate GT camera cloud" % (dataset, scene))
    offset = np.linalg.norm(aligned_centres - gt_centres, axis=1) / extent

    aligned_rotations = np.einsum("nij,kj->nik", pred_extrinsics[:, :3, :3], rotation)
    angles = rotation_angle(aligned_rotations, gt_extrinsics[:, :3, :3])

    focal = 0.5 * (pred_intrinsics[:, 0, 0] + pred_intrinsics[:, 1, 1]) / (0.5 * (gt_intrinsics[:, 0, 0] + gt_intrinsics[:, 1, 1]))
    report = dict(
        dataset=dataset,
        scene=scene,
        method=method,
        frames=len(names),
        alignment=dict(s=float(scale), r=rotation.tolist(), t=translation.tolist(),
                       rms_residual=float(np.sqrt(np.mean((aligned_centres - gt_centres) ** 2)))),
        gt_camera_extent=float(extent),
        translation=dict(mean_fraction=float(offset.mean()), median_fraction=float(np.median(offset)),
                         max_fraction=float(offset.max()), at_3cm=float((offset <= 0.03).mean()),
                         at_1cm=float((offset <= 0.01).mean())),
        rotation=dict(mean_deg=float(angles.mean()), median_deg=float(np.median(angles)),
                      max_deg=float(angles.max()), at_3deg=float((angles <= 3.0).mean()),
                      at_30deg=float((angles <= 30.0).mean())),
        focal_ratio=dict(mean=float(focal.mean()), std=float(focal.std())),
        per_frame=[dict(name=name, translation_fraction=float(offset[i]), rotation_deg=float(angles[i]))
                   for i, name in enumerate(names)],
        note="translation error is the camera-centre distance to GT, divided by the GT camera-centre bounding-box diagonal; poses are compared after a global sim(3) fit on camera centres",
        gt_source=dict(kind="colmap_sparse", cameras_sha256=sha256(gt_root / "cameras.npz")),
    )
    target = root / "pose_evaluation.json"
    write_json(target, report)
    print("%s/%s %s poses %d  rot %.2f deg  trans %.4f diag" % (dataset, scene, method, len(names),
                                                               report["rotation"]["mean_deg"],
                                                               report["translation"]["mean_fraction"]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--method", default="da3-giant")
    args = parser.parse_args()
    for scene in args.scenes:
        evaluate(args.dataset, scene, args.method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
