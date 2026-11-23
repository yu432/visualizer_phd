"""GT-only metrics: depth (per-frame scale+shift) and cloud accuracy/completeness/F-score."""
from __future__ import annotations

import numpy as np


def _distance(source, target):
    from scipy.spatial import cKDTree

    return cKDTree(target).query(source, k=1)[0]


def depth_metrics(pred, gt, per_frame_align=True):
    """AbsRel and delta1.25 under the published protocol (per-frame scale+shift)."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError("shape mismatch: %s vs %s" % (pred.shape, gt.shape))
    frames = pred.reshape((-1,) + pred.shape[-2:]) if pred.ndim > 2 else pred[None]
    targets = gt.reshape((-1,) + gt.shape[-2:]) if gt.ndim > 2 else gt[None]
    ratios, delta = [], []
    for predicted, target in zip(frames, targets):
        valid = np.isfinite(predicted) & np.isfinite(target) & (target > 0) & (predicted > 0)
        if not valid.any():
            continue
        p, t = predicted[valid], target[valid]
        if per_frame_align:
            design = np.stack([p, np.ones_like(p)], axis=1)
            solution, *_ = np.linalg.lstsq(design, t, rcond=None)
            p = design @ solution
        keep = p > 0
        if not keep.any():
            continue
        ratio = np.maximum(t[keep] / p[keep], p[keep] / t[keep])
        ratios.append(float(np.abs(p[keep] - t[keep]).mean() / t[keep].mean()))
        delta.append(float((ratio < 1.25).mean()))
    if not ratios:
        return {"AbsRel": None, "delta125": None, "frames": 0, "per_frame_align": per_frame_align}
    return {
        "AbsRel": float(np.mean(ratios)),
        "delta125": float(np.mean(delta)),
        "frames": len(ratios),
        "per_frame_align": bool(per_frame_align),
    }


def cloud_metrics(pred, gt, diag, taus=(0.005, 0.01)):
    """Accuracy/completeness/F-score with thresholds expressed as a fraction of diag."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    empty = {"accuracy": None, "completeness": None, "taus": list(taus), "pred": len(pred), "gt": len(gt)}
    if len(pred) == 0 or len(gt) == 0 or diag <= 0:
        return empty
    to_gt = _distance(pred, gt)
    to_pred = _distance(gt, pred)
    out = {
        "accuracy": float(to_gt.mean()),
        "completeness": float(to_pred.mean()),
        "pred": int(len(pred)),
        "gt": int(len(gt)),
        "taus": list(taus),
    }
    for tau in taus:
        threshold = tau * diag
        precision = float((to_gt < threshold).mean())
        recall = float((to_pred < threshold).mean())
        key = str(round(tau * 100, 3)).rstrip("0").rstrip(".")
        out["precision@" + key] = precision
        out["recall@" + key] = recall
        out["fscore@" + key] = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    return out
