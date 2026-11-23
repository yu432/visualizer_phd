"""Collect per-scene metrics into one summary table for the report."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUTS, read_json, write_json  # noqa: E402

# tnt100 — the dense variation of the Tanks and Temples scenes: same scenes and the same laser
# GT, 100 greedily selected frames instead of 16.
# eth3d20 — the dense variation of the ETH3D scenes: same scenes, same camera poses and the same
# lidar depth maps, but 20 frames instead of the official 10. eth3d20dense is the same 20-frame run
# scored against a reference rebuilt from those 20 frames' own depth maps (see make_pdf).
DATASETS = ("eth3d", "tnt", "tnt100", "eth3d20", "eth3d20dense")


def scene_row(dataset, scene, method):
    root = OUTPUTS / dataset / scene / method
    if not (root / "inference.json").exists():
        return None
    row = dict(dataset=dataset, scene=scene, method=method)
    inference = read_json(root / "inference.json")
    row["frames"] = len(inference["frames"])
    row["device"] = inference.get("device")
    row["seconds"] = inference.get("elapsed_seconds")
    row["input_shape"] = inference.get("input_shape")
    if (root / "evaluation.json").exists():
        evaluation = read_json(root / "evaluation.json")
        cloud = evaluation["cloud"]
        row["cloud"] = dict(
            fscore_half=cloud["fscore@0.5"],
            fscore_one=cloud["fscore@1"],
            precision_one=cloud["precision@1"],
            recall_one=cloud["recall@1"],
            accuracy=cloud["accuracy"],
            completeness=cloud["completeness"],
            gt_points=cloud["gt"],
            pred_points=cloud["pred"],
            gt_diagonal=evaluation["ground_diagonal"],
        )
        row["alignment"] = dict(
            variant=evaluation["alignment"].get("variant"),
            scale=evaluation["alignment"]["s"],
            centre_rms=evaluation["alignment"].get("centre_rms"),
        )
        depth = evaluation["depth"]
        if depth.get("available"):
            row["depth"] = dict(
                absrel_scale_shift=depth["scale_shift"]["AbsRel"],
                delta125_scale_shift=depth["scale_shift"]["delta125"],
                absrel_metric=depth["metric"]["AbsRel"],
                delta125_metric=depth["metric"]["delta125"],
                grid=depth["grid"],
            )
    if (root / "tolerance.json").exists():
        tolerance = read_json(root / "tolerance.json")
        keys = ("accuracy_median_cm", "accuracy_p90_cm", "cover90_cm", "completeness_median_cm",
                "auc_recall", "auc_precision", "precision@5", "recall@5", "fscore_at_5")
        row["tolerance"] = dict(
            lidar_spacing_cm=tolerance.get("lidar_spacing_cm"),
            # T&T has no dense GT of the same frames, so only the pose variant is unbiased there.
            primary="pose" if dataset.startswith("tnt") else "icp",
            variants={tag: {key: block.get(key) for key in keys}
                      for tag, block in tolerance["variants"].items()},
        )
    if (root / "pose_evaluation.json").exists():
        poses = read_json(root / "pose_evaluation.json")
        row["pose"] = dict(
            rotation_mean_deg=poses["rotation"]["mean_deg"],
            rotation_median_deg=poses["rotation"]["median_deg"],
            translation_mean_fraction=poses["translation"]["mean_fraction"],
            translation_median_fraction=poses["translation"]["median_fraction"],
            gt_camera_extent=poses["gt_camera_extent"],
        )
    if (root / "sky_ablation.json").exists():
        row["sky_ablation"] = read_json(root / "sky_ablation.json")
    if (root / "sky_meta.json").exists():
        row["sky"] = read_json(root / "sky_meta.json")
    return row


def scene_rows(dataset, methods):
    rows = []
    scene_root = OUTPUTS / dataset
    if not scene_root.exists():
        return rows
    scenes = sorted({path.name for path in scene_root.iterdir() if path.is_dir()})
    for scene in scenes:
        for method in methods:
            row = scene_row(dataset, scene, method)
            if row is not None:
                rows.append(row)
    return rows


def fmt(value, spec="%.3f"):
    return "-" if value is None else spec % value


def markdown(rows):
    header = ("| датасет | сцена | метод | кадров | F@0.5% | F@1% | prec@1% | rec@1% | AbsRel (s+t) | AbsRel raw "
              "| поворот, ° | центр, %диаг |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [header]
    for row in rows:
        cloud = row.get("cloud") or {}
        depth = row.get("depth") or {}
        pose = row.get("pose") or {}
        translation = pose.get("translation_mean_fraction")
        lines.append("| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["dataset"], row["scene"], row["method"], row["frames"],
            fmt(cloud.get("fscore_half")), fmt(cloud.get("fscore_one")),
            fmt(cloud.get("precision_one")), fmt(cloud.get("recall_one")),
            fmt(depth.get("absrel_scale_shift"), "%.4f"), fmt(depth.get("absrel_metric")),
            fmt(pose.get("rotation_mean_deg"), "%.2f"),
            fmt(None if translation is None else 100.0 * translation, "%.2f"),
        ))
    return "\n".join(lines) + "\n"


def ablation_markdown(rows):
    header = ("| сцена | метод | небо, % кадра | F@1% с фильтром | F@1% без фильтра | ΔF@1% "
              "| prec@1% с | prec@1% без | rec@1% с | rec@1% без |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
    lines = [header]
    for row in rows:
        ablation = row.get("sky_ablation")
        if not ablation:
            continue
        with_sky, without = ablation["with_sky"], ablation["without_sky"]
        sky = row.get("sky") or {}
        lines.append("| %s | %s | %s | %.3f | %.3f | %+.3f | %.3f | %.3f | %.3f | %.3f |" % (
            row["scene"], row["method"],
            "%.1f" % (100.0 * sky["sky_fraction_mean"]) if sky.get("sky_fraction_mean") is not None else "-",
            with_sky["fscore_one"], without["fscore_one"], ablation["delta_fscore_one"],
            with_sky["precision_one"], without["precision_one"],
            with_sky["recall_one"], without["recall_one"],
        ))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="da3-giant")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    methods = args.methods or [args.method]
    rows = []
    for dataset in DATASETS:
        rows.extend(scene_rows(dataset, methods))
    if not rows:
        raise SystemExit("no evaluations found")
    payload = dict(methods=methods, rows=rows,
                   markdown=markdown(rows),
                   ablation_markdown=ablation_markdown(rows),
                   provenance=dict(
                       scenes={row["dataset"] + "/" + row["scene"] + "/" + row["method"]:
                               str(OUTPUTS / row["dataset"] / row["scene"] / row["method"])
                               for row in rows},
                   ))
    target = Path(args.out)
    write_json(target / "summary.json", payload)
    (target / "summary.md").write_text(markdown(rows) + "\n" + ablation_markdown(rows))
    print(markdown(rows))
    print(ablation_markdown(rows))
    print("wrote", target / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())