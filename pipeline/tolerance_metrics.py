#!/usr/bin/env python3
"""Scale-anchored cloud metrics: tolerances in centimetres instead of fractions of a diagonal.

"F@1% = 0.966" is hard to read without knowing what 1 % of a diagonal is, and
"accuracy 0.084 m" says nothing about how much of the scene that is. This script
reports the same comparison in the units a reader can picture:

* the error distribution in cm (median, p90) and per 10 m of scene size;
* the tolerance needed to cover 50/90/95/99 % of the GT lidar  ->  "Omega needs
  1.7 cm to explain 90 % of the scan, DA3 needs 4.3 cm";
* precision/recall at a few absolute tolerances (1, 2, 5, 10 cm);
* the area under the recall/ precision-vs-tolerance curve over 0-30 cm, which is
  threshold-free;
* the GT lidar's own point spacing, so every error can be read in "lidar steps";
* the same numbers under the pose-only alignment, to show that the ranking does not
  depend on the extra ICP refinement.

ETH3D ground truth is metric, so cm are real centimetres.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import apply_sim3  # noqa: E402
from common import CLOUDS, GT, OUTPUTS, crop_volume, load_crop, read_json, read_ply, write_json  # noqa: E402
from pose_align import sim3 as pose_sim3  # noqa: E402

TOLS_CM = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0)
AUC_CM = 30.0
GT_SPACING_SAMPLE = 200_000
BOOTSTRAP = 400


def nn(query, tree, workers=-1):
    return tree.query(np.asarray(query, dtype=np.float64), k=1, workers=workers)[0]


def lidar_spacing(points, sample=GT_SPACING_SAMPLE, seed=0):
    """Median distance to the nearest *other* GT point: the resolution floor of the scan."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) > sample:
        rng = np.random.default_rng(seed)
        points = points[np.sort(rng.choice(len(points), size=sample, replace=False))]
    tree = cKDTree(points)
    distances = tree.query(points, k=2, workers=-1)[0][:, 1]
    return float(np.median(distances))


def tolerance_block(to_gt_cm, to_pred_cm, diag_cm, spacing_cm):
    to_gt_cm = np.asarray(to_gt_cm, dtype=np.float64)
    to_pred_cm = np.asarray(to_pred_cm, dtype=np.float64)
    block = {
        "n_pred": int(len(to_gt_cm)),
        "n_gt": int(len(to_pred_cm)),
        "accuracy_median_cm": float(np.median(to_gt_cm)),
        "accuracy_p90_cm": float(np.quantile(to_gt_cm, 0.90)),
        "accuracy_mean_cm": float(to_gt_cm.mean()),
        "completeness_median_cm": float(np.median(to_pred_cm)),
        "completeness_p90_cm": float(np.quantile(to_pred_cm, 0.90)),
        # Tolerance at which a given share of the GT scan is explained.
        "cover50_cm": float(np.quantile(to_pred_cm, 0.50)),
        "cover90_cm": float(np.quantile(to_pred_cm, 0.90)),
        "cover95_cm": float(np.quantile(to_pred_cm, 0.95)),
        "cover99_cm": float(np.quantile(to_pred_cm, 0.99)),
        "auc_recall": float(np.clip(AUC_CM - to_pred_cm, 0.0, AUC_CM).mean() / AUC_CM),
        "auc_precision": float(np.clip(AUC_CM - to_gt_cm, 0.0, AUC_CM).mean() / AUC_CM),
        "lidar_spacing_cm": spacing_cm,
        "accuracy_lidar_steps": float(np.median(to_gt_cm) / spacing_cm) if spacing_cm > 0 else None,
        # "cm of error per 10 m of scene size": the scale-free way to read a cm value.
        "error_cm_per_10m": float(1000.0 * np.median(to_gt_cm) / diag_cm),
        "scene_diagonal_m": float(diag_cm / 100.0),
    }
    for tol in TOLS_CM:
        key = ("%g" % tol).replace(".", "p")
        block["precision@" + key] = float((to_gt_cm < tol).mean())
        block["recall@" + key] = float((to_pred_cm < tol).mean())
        precision = block["precision@" + key]
        recall = block["recall@" + key]
        block["fscore@" + key] = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    return block


def bootstrap_median(values, rounds=BOOTSTRAP, seed=0, size=50_000):
    """CI of the median over points: shows how much of the gap is point sampling noise."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) > size:
        rng = np.random.default_rng(seed)
        values = values[np.sort(rng.choice(len(values), size=size, replace=False))]
    rng = np.random.default_rng(seed + 1)
    medians = [float(np.median(rng.choice(values, size=len(values), replace=True)))
               for _ in range(rounds)]
    low, high = np.percentile(medians, [2.5, 97.5])
    return [float(low), float(high)]


def density_floor(ground, count, ground_tree=None, seed=0):
    """Accuracy that a *perfect* cloud of the prediction's own density would score.

    The GT lidar is a sparse point sample, so even an ideal reconstruction keeps a
    nearest-neighbour distance of roughly half the lidar spacing. Measuring that floor
    turns "5.7 cm" into "5.7 cm against a 2.4 cm floor", which is what makes the numbers
    comparable between scenes with different scan densities.
    """
    tree = ground_tree if ground_tree is not None else cKDTree(ground)
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(ground), size=min(count, len(ground)), replace=False))
    distances = tree.query(ground[index], k=2, workers=-1)[0][:, 1]
    return float(np.median(100.0 * distances))


def evaluate_scene(dataset, scene, method):
    ground, _ = read_ply(GT / dataset / scene / "gt_cloud.ply")
    predicted, _ = read_ply(CLOUDS / dataset / scene / (method + ".ply"))
    diag_cm = 100.0 * float(read_json(GT / dataset / scene / "gt_meta.json")["diagonal"])
    spacing_cm = 100.0 * lidar_spacing(ground)
    ground_tree = cKDTree(ground)
    floor_cm = density_floor(ground, len(predicted), ground_tree)

    variants = {}
    evaluation = OUTPUTS / dataset / scene / method / "evaluation.json"
    if evaluation.exists():
        payload = read_json(evaluation)
        variants["icp"] = dict(s=payload["alignment"]["s"],
                               R=np.asarray(payload["alignment_rotation"]),
                               t=np.asarray(payload["alignment_translation"]))
    variants["pose"] = pose_sim3(dataset, scene, method)

    result = {"dataset": dataset, "scene": scene, "method": method,
              "lidar_spacing_cm": spacing_cm, "floor_cm": floor_cm,
              "scene_diagonal_m": diag_cm / 100.0, "variants": {}}
    region = load_crop(GT / dataset / scene)
    if region is not None:
        result["cloud_crop"] = "official polygon volume"
    # GT lives in the GT frame already; only the prediction moves.
    for tag, params in variants.items():
        aligned = apply_sim3(predicted, params)
        if region is not None:
            aligned = crop_volume(aligned, region)
        pred_tree = cKDTree(aligned)
        to_gt = 100.0 * nn(aligned, ground_tree)      # accuracy: prediction -> GT
        to_pred = 100.0 * nn(ground, pred_tree)       # completeness: GT -> prediction
        block = tolerance_block(to_gt, to_pred, diag_cm, spacing_cm)
        block["accuracy_median_ci_cm"] = bootstrap_median(to_gt)
        block["accuracy_over_floor"] = float(block["accuracy_median_cm"] / floor_cm) if floor_cm > 0 else None
        result["variants"][tag] = block
    return result


COLOURS = {"da3-giant": "#1f6fd0", "vggt-omega": "#d97706"}
LABELS = {"da3-giant": "DA3-Giant", "vggt-omega": "VGGT-Ω"}


def plot(summary_path, out_path):
    """Recall/precision vs absolute tolerance: where the two methods actually separate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figures import PAGE_WIDTH, save

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.titlesize"] = 11.5

    summary = read_json(Path(summary_path))
    tols = np.asarray(summary["tolerances_cm"], dtype=np.float64)
    rows = summary["rows"]
    scenes = []
    for row in rows:
        if row["scene"] not in scenes:
            scenes.append(row["scene"])
    lookup = {(row["scene"], row["method"]): row for row in rows}

    figure, axes = plt.subplots(1, len(scenes), figsize=(2.7 * len(scenes), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, scene in zip(axes, scenes):
        for method in ("da3-giant", "vggt-omega"):
            row = lookup.get((scene, method))
            if row is None:
                continue
            block = row["variants"]["icp"]
            colour = COLOURS[method]
            recall = [block["recall@" + ("%g" % tol).replace(".", "p")] for tol in tols]
            precision = [block["precision@" + ("%g" % tol).replace(".", "p")] for tol in tols]
            axis.plot(tols, recall, color=colour, lw=1.9, label=LABELS[method])
            axis.plot(tols, precision, color=colour, lw=1.1, ls=(0, (4, 2)), alpha=0.85)
            cover = block["cover90_cm"]
            if cover <= tols[-1]:
                axis.plot([cover], [0.90], marker="o", ms=5, color=colour, zorder=5)
            else:
                axis.annotate("", xy=(tols[-1], 0.90), xytext=(tols[-1] * 0.82, 0.90),
                              arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.6})
                axis.annotate("%.0f см →" % cover, xy=(tols[-1], 0.90), fontsize=8.5,
                              color=colour, ha="right", va="bottom")
        axis.axhline(0.90, color="#9aa4b0", lw=0.8, ls=":", zorder=0)
        axis.set_xscale("log")
        axis.set_xlim(tols[0], tols[-1])
        axis.set_ylim(0.0, 1.02)
        axis.set_title(scene, fontsize=11)
        axis.grid(alpha=0.25, lw=0.6)
        axis.tick_params(labelsize=9.5)
        axis.set_xlabel("допуск, см", fontsize=10)
    axes[0].set_ylabel("доля поверхности", fontsize=10)
    figure.suptitle(
        "ETH3D: сколько поверхности объяснено при данном допуске.\n"
        "сплошная — доля GT-лидара в пределах допуска (полнота), штрих — доля предсказания (точность); "
        "точка — допуск, где покрыто 90 % лидара",
        fontsize=11.5)
    handles = [plt.Line2D([], [], color=COLOURS[m], lw=1.9, label=LABELS[m])
               for m in ("da3-giant", "vggt-omega")]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10.5,
                  bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0, 0.105, 1, 0.855))
    save(figure, Path(out_path), tight=False)
    print("figure written:", out_path, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eth3d")
    parser.add_argument("--scenes", nargs="+",
                        default=["courtyard", "delivery_area", "facade", "office", "terrace"])
    parser.add_argument("--methods", nargs="+", default=["da3-giant", "vggt-omega"])
    parser.add_argument("--out", default=None, help="JSON summary path")
    parser.add_argument("--figure", default=None, help="PNG with the tolerance curves")
    parser.add_argument("--only-figure", action="store_true", help="reuse --out JSON, just redraw")
    args = parser.parse_args()
    if args.only_figure:
        plot(args.out, args.figure)
        return 0

    rows = []
    for scene in args.scenes:
        for method in args.methods:
            result = evaluate_scene(args.dataset, scene, method)
            write_json(OUTPUTS / args.dataset / scene / method / "tolerance.json", result)
            rows.append(result)
            main_variant = result["variants"]["icp"]
            print("%-14s %-11s med %.2f cm (95%% CI %.2f-%.2f) p90 %.2f  cover90 %.2f cm  "
                  "F@2cm %.3f  lidar step %.2f cm"
                  % (scene, method, main_variant["accuracy_median_cm"],
                     main_variant["accuracy_median_ci_cm"][0], main_variant["accuracy_median_ci_cm"][1],
                     main_variant["accuracy_p90_cm"], main_variant["cover90_cm"],
                     main_variant["fscore@2"], main_variant["lidar_spacing_cm"]),
                  flush=True)
    if args.out:
        write_json(Path(args.out), {"rows": rows,
                                    "tolerances_cm": list(TOLS_CM),
                                    "auc_range_cm": AUC_CM})
        print("summary written:", args.out)
    if args.figure and args.out:
        plot(args.out, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())