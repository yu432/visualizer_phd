"""Sky-mask ablation: same pose sim(3), clouds built with and without the sky filter."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import apply_sim3  # noqa: E402
from common import CLOUDS, GT, OUTPUTS, read_json, read_ply, write_json  # noqa: E402
from metrics import cloud_metrics  # noqa: E402

EDGE = 400000


def subsample(points, limit=EDGE, seed=0):
    if len(points) <= limit:
        return points
    generator = np.random.default_rng(seed)
    index = generator.choice(len(points), size=limit, replace=False)
    return points[np.sort(index)]


def variant_metrics(dataset, scene, method, tag, params, ground, ground_diag):
    cloud_path = CLOUDS / dataset / scene / (method + tag + ".ply")
    if not cloud_path.exists():
        return None
    points, _ = read_ply(cloud_path)
    meta = read_json(CLOUDS / dataset / scene / (method + tag + "_cloud.json"))
    aligned = apply_sim3(points, params)
    metrics = cloud_metrics(subsample(aligned), ground, ground_diag)
    return dict(
        points=int(len(points)),
        raw_points=meta.get("raw_points"),
        sky_source=meta.get("sky_source"),
        precision_one=metrics["precision@1"],
        recall_one=metrics["recall@1"],
        fscore_half=metrics["fscore@0.5"],
        fscore_one=metrics["fscore@1"],
        accuracy=metrics["accuracy"],
        completeness=metrics["completeness"],
    )


def run(dataset, scene, method):
    evaluation_path = OUTPUTS / dataset / scene / method / "evaluation.json"
    if not evaluation_path.exists():
        return None
    evaluation = read_json(evaluation_path)
    params = dict(evaluation["alignment"])
    ground_diag = float(evaluation["ground_diagonal"])
    ground, _ = read_ply(GT / dataset / scene / "gt_cloud.ply")
    ground = subsample(ground)
    with_sky = variant_metrics(dataset, scene, method, "", params, ground, ground_diag)
    without = variant_metrics(dataset, scene, method, "-nosky", params, ground, ground_diag)
    if with_sky is None or without is None:
        return None
    result = dict(
        dataset=dataset,
        scene=scene,
        method=method,
        alignment_source="pose sim(3) reused from evaluation.json (identical for both variants)",
        ground_diagonal=ground_diag,
        with_sky=with_sky,
        without_sky=without,
        delta_fscore_one=with_sky["fscore_one"] - without["fscore_one"],
        delta_precision_one=with_sky["precision_one"] - without["precision_one"],
        delta_recall_one=with_sky["recall_one"] - without["recall_one"],
    )
    write_json(OUTPUTS / dataset / scene / method / "sky_ablation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eth3d")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=["da3-giant", "vggt-omega"])
    args = parser.parse_args()
    for method in args.methods:
        for scene in args.scenes:
            result = run(args.dataset, scene, method)
            if result is None:
                print(scene, method, "skipped", flush=True)
                continue
            print(scene, method,
                  "F@1 with sky %.3f without %.3f (delta %+.3f)" % (
                      result["with_sky"]["fscore_one"], result["without_sky"]["fscore_one"],
                      result["delta_fscore_one"]),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
