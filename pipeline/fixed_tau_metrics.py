"""F-score at a threshold expressed in metres instead of in percent of the GT diagonal.

Why this exists: cloud_metrics uses tau = 1% of the GT cloud diagonal, so a reference built from
more frames is also a *larger* reference box, its 1% is a bigger distance, and F@1 rises for free
(facade: 0.94 m -> 1.09 m). Here tau is pinned to 1% of the *published 10-frame* diagonal, which
makes the 10-frame and 20-frame references comparable, and reports exactly which geometric test
the numbers below the PDF table answer.

Precision = share of predicted points within tau of the reference; recall = share of reference
points within tau of the prediction; both after the sim(3) stored in evaluation.json.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import apply_sim3  # noqa: E402
from common import CLOUDS, GT, OUTPUTS, load_crop, crop_volume, read_json, read_ply, write_json  # noqa: E402

METHODS = ("da3-giant", "vggt-omega")


def pair(prediction, reference, tau):
    pred_tree = cKDTree(prediction)
    ref_tree = cKDTree(reference)
    to_ref, _ = ref_tree.query(prediction, k=1, workers=-1)
    to_pred, _ = pred_tree.query(reference, k=1, workers=-1)
    precision = float((to_ref <= tau).mean())
    recall = float((to_pred <= tau).mean())
    fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return dict(precision=precision, recall=recall, fscore=fscore,
                accuracy=float(np.median(to_ref)), completeness=float(np.median(to_pred)))


def evaluate(dataset, reference_dataset, scene, method, tau):
    evaluation = read_json(OUTPUTS / dataset / scene / method / "evaluation.json")
    alignment = dict(evaluation["alignment"])
    predicted, _ = read_ply(CLOUDS / dataset / scene / (method + ".ply"))
    aligned = apply_sim3(predicted, alignment)
    ground, _ = read_ply(GT / reference_dataset / scene / "gt_cloud.ply")
    region = load_crop(GT / reference_dataset / scene)
    kept = aligned if region is None else crop_volume(aligned, region)
    return pair(kept, ground, tau)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eth3d20")
    parser.add_argument("--reference", default="eth3d20dense",
                        help="dataset whose GT cloud is the reference")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = dict(dataset=args.dataset, reference=args.reference, rows=[])
    for scene in args.scenes:
        published = read_json(GT / "eth3d" / scene / "gt_meta.json")
        tau = 0.01 * float(published["diagonal"])
        # Same clouds against both references: only the reference coverage and size differ.
        sparse = {method: evaluate(args.dataset, "eth3d", scene, method, tau) for method in args.methods}
        dense = {method: evaluate(args.dataset, args.reference, scene, method, tau)
                 for method in args.methods}
        row = dict(scene=scene, tau_metres=tau,
                   gt10_diagonal=float(published["diagonal"]), by_method={})
        for method in args.methods:
            reference_meta = read_json(GT / args.reference / scene / "gt_meta.json")
            row["gt_dense_diagonal"] = float(reference_meta["diagonal"])
            row["by_method"][method] = dict(gt10=sparse[method], gt_dense=dense[method])
        result["rows"].append(row)
        for method in args.methods:
            a, b = row["by_method"][method]["gt10"], row["by_method"][method]["gt_dense"]
            print("%-14s %-11s tau %.3f m  F %.3f -> %.3f  P %.3f -> %.3f  R %.3f -> %.3f"
                  % (scene, method, tau, a["fscore"], b["fscore"], a["precision"], b["precision"],
                     a["recall"], b["recall"]))
    write_json(Path(args.out), result)
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
