"""Align a predicted cloud to GT and compute depth + cloud metrics."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import apply_sim3, refine_from  # noqa: E402
from common import CLOUDS, GT, INFERENCE, OUTPUTS, crop_volume, diagonal, load_crop, read_json, read_ply, sha256, write_json  # noqa: E402
from metrics import cloud_metrics, depth_metrics  # noqa: E402
from pose_align import sim3 as pose_sim3  # noqa: E402


def subsample(points, limit, seed=0):
    if len(points) <= limit:
        return points
    generator = np.random.default_rng(seed)
    index = generator.choice(len(points), size=limit, replace=False)
    return points[np.sort(index)]


def resize_nearest(array, height, width):
    rows = (np.arange(height) * (array.shape[0] / height)).astype(np.int64)
    cols = (np.arange(width) * (array.shape[1] / width)).astype(np.int64)
    rows = np.clip(rows, 0, array.shape[0] - 1)
    cols = np.clip(cols, 0, array.shape[1] - 1)
    return array[rows][:, cols]


def scaled_intrinsics(intrinsics, from_shape, to_shape):
    scale_x = to_shape[1] / from_shape[1]
    scale_y = to_shape[0] / from_shape[0]
    out = intrinsics.copy()
    out[:, 0, 0] *= scale_x
    out[:, 1, 1] *= scale_y
    out[:, 0, 2] *= scale_x
    out[:, 1, 2] *= scale_y
    return out


def depth_block(dataset, scene, method):
    gt_root = GT / dataset / scene
    meta = read_json(gt_root / "gt_meta.json")
    if not meta.get("has_depth"):
        return dict(available=False, reason="dataset has no dense GT depth")
    inference = INFERENCE / dataset / scene
    names = [frame["name"] for frame in read_json(inference / "manifest.json")["frames"]]
    prediction = OUTPUTS / dataset / scene / method
    pred_depth = np.load(prediction / "depth.npy")
    pred_intrinsics = np.load(prediction / "intrinsics.npy")
    if len(pred_depth) != len(names):
        raise ValueError("prediction/frame count mismatch")
    grid = pred_depth.shape[-2:]
    gt_stack = []
    for name in names:
        payload = np.load(gt_root / "depths" / (Path(name).stem + ".npz"))
        dense = np.asarray(payload["depth"], dtype=np.float64)
        gt_stack.append(resize_nearest(dense, grid[0], grid[1]))
    gt_depth = np.stack(gt_stack)
    scale_shift = depth_metrics(pred_depth, gt_depth, per_frame_align=True)
    metric = depth_metrics(pred_depth, gt_depth, per_frame_align=False)
    result = dict(
        scale_shift=scale_shift,
        metric=metric,
        AbsRel=scale_shift["AbsRel"],
        delta125=scale_shift["delta125"],
        AbsRel_metric=metric["AbsRel"],
        delta125_metric=metric["delta125"],
    )
    result["available"] = True
    result["grid"] = list(grid)
    result["note"] = (
        "GT depth resampled to the prediction grid with nearest neighbour; "
        "scale_shift follows the published per-frame scale+shift protocol, metric keeps the raw prediction scale"
    )
    result["pred_intrinsics_sha256"] = sha256(prediction / "intrinsics.npy")
    return result


def evaluate(dataset: str, scene: str, method: str, limit: int, force: bool,
             icp: bool = True) -> dict:
    target = OUTPUTS / dataset / scene / method / "evaluation.json"
    if target.exists() and not force:
        return read_json(target)
    gt_meta = read_json(GT / dataset / scene / "gt_meta.json")
    ground, _ = read_ply(GT / dataset / scene / "gt_cloud.ply")
    cloud_meta = read_json(CLOUDS / dataset / scene / (method + "_cloud.json"))
    predicted, _ = read_ply(CLOUDS / dataset / scene / (method + ".ply"))
    ground_diag = float(gt_meta["diagonal"])
    start = pose_sim3(dataset, scene, method)
    if icp:
        refined = refine_from(predicted, ground, start, tau=0.01 * ground_diag)
        params, variant, fscore = refined["params"], refined["variant"], refined["fscore"]
    else:
        # Tanks and Temples: ICP is not used, because on courthouse it partially repairs a gross
        # failure of VGGT-Omega and the reported accuracy would hide it.
        params, variant, fscore = start, "pose", None
    alignment = dict(params)
    alignment.update(status="ok", variant=variant, fscore=fscore,
                     centre_rms=start["centre_rms"], source=start["source"],
                     gt_camera_extent=start["gt_camera_extent"])
    aligned = apply_sim3(predicted, alignment)
    region = load_crop(GT / dataset / scene)
    kept = aligned
    crop_note = None
    if region is not None:
        kept = crop_volume(aligned, region)
        crop_note = dict(region="official polygon volume", before=int(len(aligned)), after=int(len(kept)))
    cloud = cloud_metrics(kept, ground, ground_diag)
    if alignment["fscore"] is None:
        alignment["fscore"] = cloud["fscore@1"]
    result = dict(
        dataset=dataset,
        scene=scene,
        method=method,
        alignment=alignment,
        alignment_rotation=np.asarray(alignment["R"]).tolist(),
        alignment_translation=np.asarray(alignment["t"]).tolist(),
        scale_correction=float(alignment["s"]),
        predicted_scale_hint=float(cloud_meta["diagonal"] / ground_diag),
        cloud=cloud,
        cloud_crop=crop_note,
        ground_diagonal=ground_diag,
        depth=depth_block(dataset, scene, method),
        cloud_metrics_note=("accuracy/completeness are mean NN distances in GT units after the camera-pose "
                            "sim(3) alignment" + ("" if icp else " with no ICP refinement")),
        cloud_sha256=sha256(CLOUDS / dataset / scene / (method + ".ply")),
        ground_sha256=sha256(GT / dataset / scene / "gt_cloud.ply"),
    )
    write_json(target, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--limit", type=int, default=80000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-icp", action="store_true",
                        help="keep the pose-only sim(3); used for Tanks and Temples")
    args = parser.parse_args()
    for scene in args.scenes:
        result = evaluate(args.dataset, scene, args.method, args.limit, args.force,
                          icp=not args.no_icp)
        cloud = result["cloud"]
        depth = result["depth"]
        print(
            scene, args.method,
            "F@0.5 %.3f F@1 %.3f" % (cloud["fscore@0.5"], cloud["fscore@1"]),
            "AbsRel %s" % (None if not depth.get("available") else round(depth["AbsRel"], 4)),
            "scale %.4f" % result["scale_correction"],
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())