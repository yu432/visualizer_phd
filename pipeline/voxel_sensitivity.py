"""How much of the reported cloud error is just the voxel filter?

The prediction cloud is voxel-averaged at frac * diagonal(model cloud). Because the
model cloud lives in arbitrary units, that voxel becomes frac * diagonal(GT) in metric
terms -- 0.15 % of the scene diagonal, i.e. 6.5 cm on courtyard and 14 cm on facade, while
the GT lidar spacing there is 2.5-3.2 cm. This script re-measures the same alignment at
several voxel sizes (up to "no voxel at all") so the density floor can be separated from
the model's own error.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import apply_sim3  # noqa: E402
from build_clouds import confidence_mask, far_mask, load_prediction, sky_mask  # noqa: E402
from common import CLOUDS, GT, INFERENCE, OUTPUTS, backproject_coloured, diagonal  # noqa: E402
from common import read_json, read_ply, voxel_downsample_attributed, write_json  # noqa: E402
from tolerance_metrics import bootstrap_median, lidar_spacing  # noqa: E402

FRACTIONS = (0.0, 0.0004, 0.0015)
SPACING_SAMPLE = 150_000
QUERY_SAMPLE = 400_000


def nn(query, tree):
    return tree.query(query, k=1, workers=-1)[0]


def spacing(points, seed=0):
    """Median distance from a point to its nearest other point: the cloud sampling step."""
    if len(points) < 3:
        return float("nan")
    generator = np.random.default_rng(seed)
    if len(points) > SPACING_SAMPLE:
        index = generator.choice(len(points), size=SPACING_SAMPLE, replace=False)
        sample = points[np.sort(index)]
    else:
        sample = points
    tree = cKDTree(points)
    distances = tree.query(sample, k=2, workers=-1)[0][:, 1]
    return float(np.median(distances))


def block(predicted, ground, ground_tree, query_limit=QUERY_SAMPLE, seed=0):
    generator = np.random.default_rng(seed)
    if len(predicted) > query_limit:
        index = generator.choice(len(predicted), size=query_limit, replace=False)
        to_gt = nn(predicted[np.sort(index)], ground_tree)
    else:
        to_gt = nn(predicted, ground_tree)
    if len(ground) > query_limit:
        index = generator.choice(len(ground), size=query_limit, replace=False)
        ground_query = ground[np.sort(index)]
    else:
        ground_query = ground
    predicted_tree = cKDTree(predicted)
    to_pred = nn(ground_query, predicted_tree)
    del predicted_tree
    to_gt_cm = to_gt * 100.0
    to_pred_cm = to_pred * 100.0
    row = dict(
        predicted_points=int(len(predicted)),
        spacing_cm=spacing(predicted) * 100.0,
        accuracy_median_cm=float(np.median(to_gt_cm)),
        accuracy_p90_cm=float(np.quantile(to_gt_cm, 0.90)),
        accuracy_ci_cm=list(bootstrap_median(to_gt_cm)),
        cover90_cm=float(np.quantile(to_pred_cm, 0.90)),
        completeness_median_cm=float(np.median(to_pred_cm)),
    )
    for tol in (2.0, 5.0, 10.0):
        precision = float(np.mean(to_gt_cm <= tol))
        recall = float(np.mean(to_pred_cm <= tol))
        row["precision_at_%g" % tol] = precision
        row["recall_at_%g" % tol] = recall
        row["fscore_at_%g" % tol] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return row


def run_scene(dataset, scene, method):
    inference = INFERENCE / dataset / scene
    out = OUTPUTS / dataset / scene / method
    evaluation = read_json(out / "evaluation.json")
    meta = read_json(CLOUDS / dataset / scene / (method + "_cloud.json"))
    names = [frame["name"] for frame in read_json(inference / "manifest.json")["frames"]]
    prediction = load_prediction(out)
    # Same mask as the cloud this row is compared with, so the voxel ladder changes only the grid.
    masks = confidence_mask(prediction["confidence"], float(meta["confidence_quantile"]))
    segmentation = None
    if prediction["sky_masks"] is not None:
        segmentation = ~prediction["sky_masks"].astype(bool)
    for extra in (sky_mask(prediction["sky"], 0.5), segmentation,
                  far_mask(prediction["depth"], 95.0)):
        if extra is not None:
            masks = extra if masks is None else (masks & extra)
    points, _ = backproject_coloured(prediction["depth"], prediction["intrinsics"],
                                     prediction["extrinsics"],
                                     [inference / "images" / name for name in names], masks)
    ground, _ = read_ply(GT / dataset / scene / "gt_cloud.ply")
    ground_diag = float(evaluation["ground_diagonal"])
    ground_tree = cKDTree(ground)
    spacing_cm = lidar_spacing(ground) * 100.0
    model_diag = diagonal(points)
    rows = []
    for fraction in FRACTIONS:
        voxel = fraction * model_diag
        if voxel <= 0:
            reduced = points
        else:
            reduced = voxel_downsample_attributed(points, voxel, None)[0]
        aligned = apply_sim3(reduced, evaluation["alignment"])
        row = block(aligned, ground, ground_tree)
        row.update(voxel_fraction=float(fraction),
                   voxel_metric_cm=float(fraction * ground_diag * 100.0),
                   voxel_over_lidar=float(fraction * ground_diag * 100.0 / spacing_cm))
        rows.append(row)
        print("%-14s %-11s frac %.4f voxel %6.2f cm (%4.1f lidar steps) pts %8d spacing %5.2f cm | "
              "med %6.2f cm p90 %6.2f cover90 %7.2f | F@5 %.3f rec@5 %.3f"
              % (scene, method, fraction, row["voxel_metric_cm"], row["voxel_over_lidar"],
                 row["predicted_points"], row["spacing_cm"], row["accuracy_median_cm"],
                 row["accuracy_p90_cm"], row["cover90_cm"], row["fscore_at_5"],
                 row["recall_at_5"]), flush=True)
    return dict(dataset=dataset, scene=scene, method=method, gt_points=int(len(ground)),
                gt_diagonal_m=ground_diag, lidar_spacing_cm=spacing_cm,
                stored_voxel_metric_cm=float(meta.get("voxel", 0.0)),
                current_fraction=float(meta.get("voxel_fraction", 0.0)), rows=rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eth3d")
    parser.add_argument("--scenes", nargs="+",
                        default=["courtyard", "delivery_area", "facade", "office", "terrace"])
    parser.add_argument("--methods", nargs="+", default=["da3-giant", "vggt-omega"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results = [run_scene(args.dataset, scene, method)
               for scene in args.scenes for method in args.methods]
    if args.out:
        write_json(Path(args.out), {"fractions": list(FRACTIONS), "scenes": results})
        print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
