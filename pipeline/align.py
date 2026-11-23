"""Scale-free cloud alignment: seeded sim(3) Umeyama over nearest neighbours, then ICP."""
from __future__ import annotations

import numpy as np

from common import diagonal


def apply_sim3(points, params):
    return params["s"] * np.asarray(points) @ np.asarray(params["R"]).T + np.asarray(params["t"])


def _tree(points, cached=None):
    from scipy.spatial import cKDTree

    return cached if cached is not None else cKDTree(points)


def _nearest_index(source, target, tree=None):
    return _tree(target, tree).query(source, k=1)[1]


def _nearest_distance(source, target, tree=None):
    return _tree(target, tree).query(source, k=1)[0]


def umeyama(source, target):
    """Least-squares similarity transform mapping source onto target, no reflection."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    s0, t0 = source - mu_s, target - mu_t
    covariance = t0.T @ s0 / len(source)
    u, d, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    variance = (s0 ** 2).sum() / len(source)
    scale = float(np.trace(np.diag(d) @ correction) / variance) if variance > 0 else 1.0
    translation = mu_t - scale * rotation @ mu_s
    return scale, rotation, translation


def compose(outer, inner):
    """Return the sim(3) equivalent to applying inner first, then outer."""
    scale = outer["s"] * inner["s"]
    rotation = np.asarray(outer["R"]) @ np.asarray(inner["R"])
    translation = outer["s"] * np.asarray(outer["R"]) @ np.asarray(inner["t"]) + np.asarray(outer["t"])
    return {"s": float(scale), "R": rotation.tolist(), "t": translation.tolist()}


def _rmse(source, target, tree=None):
    return float(np.sqrt((_nearest_distance(source, target, tree) ** 2).mean()))


def seed_scale(pred, gt):
    """Ratio of median radii around each centroid: robust rough scale guess."""
    pred_radius = float(np.median(np.linalg.norm(pred - pred.mean(axis=0), axis=1)))
    gt_radius = float(np.median(np.linalg.norm(gt - gt.mean(axis=0), axis=1)))
    if pred_radius <= 0 or gt_radius <= 0:
        return 1.0
    return gt_radius / pred_radius




def _median_distance(source, target, tree=None):
    return float(np.median(_nearest_distance(source, target, tree)))


def _principal_axes(points):
    centre = points.mean(axis=0)
    centred = points - centre
    covariance = centred.T @ centred / len(points)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    return centre, values[order], vectors[:, order]


def pca_seeds(pred, gt):
    """Candidate sim(3) seeds from principal axes, one per proper sign combination."""
    centre_s, values_s, axes_s = _principal_axes(pred)
    centre_t, values_t, axes_t = _principal_axes(gt)
    spread_s, spread_t = float(values_s.sum()), float(values_t.sum())
    if spread_s <= 0 or spread_t <= 0:
        return []
    scale = float(np.sqrt(spread_t / spread_s))
    seeds = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                rotation = axes_t @ np.diag([sx, sy, sz]) @ axes_s.T
                if np.linalg.det(rotation) < 0:
                    continue
                translation = centre_t - scale * rotation @ centre_s
                seeds.append({"s": scale, "R": rotation.tolist(), "t": translation.tolist()})
    return seeds


def _mutual_pairs(source, target, source_tree, target_tree, trim):
    d_st, i_st = target_tree.query(source, k=1)
    d_ts, i_ts = source_tree.query(target, k=1)
    keep = d_st <= np.quantile(d_st, trim)
    forward = keep & (i_ts[i_st] == np.arange(len(source)))
    back_keep = d_ts <= np.quantile(d_ts, trim)
    backward_source = i_ts[back_keep]
    backward = backward_source[(i_st[backward_source] == np.flatnonzero(back_keep))]
    matched_source = np.concatenate([np.flatnonzero(forward), backward])
    if len(matched_source) == 0:
        return None, None
    matched_source = np.unique(matched_source)
    matched_target = i_st[matched_source]
    return source[matched_source], target[matched_target]


def fscore_at(pred, gt, tau, pred_tree=None, gt_tree=None):
    """Symmetric F-score: collapse-safe quality measure for partial reconstructions."""
    d_pg = _nearest_distance(pred, gt, gt_tree)
    d_gp = _nearest_distance(gt, pred, pred_tree)
    precision = float((d_pg < tau).mean())
    recall = float((d_gp < tau).mean())
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _icp(pred, gt, params, iterations, trim, gt_tree=None, tau=None):
    from scipy.spatial import cKDTree

    current = apply_sim3(pred, params)
    for _ in range(iterations):
        source_tree = cKDTree(current)
        target_tree = gt_tree if gt_tree is not None else cKDTree(gt)
        matched_source, matched_target = _mutual_pairs(current, gt, source_tree, target_tree, trim)
        if matched_source is None or len(matched_source) < max(50, 0.01 * len(pred)):
            return None, None
        step_scale, step_rotation, step_translation = umeyama(matched_source, matched_target)
        previous = params
        params = compose(
            {"s": step_scale, "R": step_rotation.tolist(), "t": step_translation.tolist()}, previous
        )
        current = apply_sim3(pred, params)
        if abs(params["s"] - previous["s"]) / max(params["s"], 1e-12) < 1e-9 and np.allclose(
            np.asarray(params["t"]), np.asarray(previous["t"]), atol=1e-8
        ):
            break
    return params, current


def _icp_plain(pred, gt, params, iterations, trim, gt_tree=None):
    """Classic trimmed ICP on prediction->GT nearest neighbours (final polish only)."""
    current = apply_sim3(pred, params)
    minimum = max(50, 0.01 * len(pred))
    for _ in range(iterations):
        index = _nearest_index(current, gt, gt_tree)
        distance = np.linalg.norm(current - gt[index], axis=1)
        keep = distance <= np.quantile(distance, trim)
        if keep.sum() < minimum:
            return None, None
        step_scale, step_rotation, step_translation = umeyama(current[keep], gt[index[keep]])
        previous = params
        params = compose(
            {"s": step_scale, "R": step_rotation.tolist(), "t": step_translation.tolist()}, previous
        )
        current = apply_sim3(pred, params)
        if abs(params["s"] - previous["s"]) / max(params["s"], 1e-12) < 1e-9 and np.allclose(
            np.asarray(params["t"]), np.asarray(previous["t"]), atol=1e-8
        ):
            break
    return params, current


def align_clouds(pred, gt, iterations=60, trim=0.8, refine=20, sample=20000, seed=0, tau_fraction=0.01):
    """Align prediction onto the GT cloud without using GT camera poses.

    Candidates (radius-ratio seed plus principal-axis seeds) are refined with a
    symmetric mutual-nearest-neighbour ICP and ranked by the symmetric F-score,
    which cannot be won by collapsing the prediction into a dense blob. The
    returned s, R, t map the input prediction cloud onto the GT frame, so
    apply_sim3(pred, result) reproduces the aligned cloud.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    result = {
        "s": 1.0,
        "R": np.eye(3).tolist(),
        "t": np.zeros(3).tolist(),
        "status": "failed",
        "rmse_before": None,
        "rmse_after": None,
        "inliers": 0,
        "seed_s": None,
        "fscore_seed": None,
        "tau": None,
    }
    if len(pred) < 100 or len(gt) < 100:
        return result
    generator = np.random.default_rng(seed)
    pred_small = pred[np.sort(generator.choice(len(pred), size=min(len(pred), sample), replace=False))]
    gt_small = gt[np.sort(generator.choice(len(gt), size=min(len(gt), sample), replace=False))]
    diagonal_gt = diagonal(gt)
    tau = float(tau_fraction * diagonal_gt)
    result["tau"] = tau
    small_tree = _tree(gt_small)
    radius_seed = seed_scale(pred_small, gt_small)
    result["seed_s"] = float(radius_seed)
    candidates = [{"s": float(radius_seed), "R": np.eye(3).tolist(), "t": np.zeros(3).tolist()}]
    candidates.extend(pca_seeds(pred_small, gt_small))
    scored = []
    for candidate in candidates:
        params, current = _icp(pred_small, gt_small, candidate, refine, trim, small_tree, tau)
        if params is None:
            continue
        score = fscore_at(current, gt_small, tau, _tree(current), small_tree)
        scored.append((score, params))
    if scored:
        scored.sort(key=lambda item: item[0])
        chosen = scored[0][1]
        result["fscore_seed"] = float(scored[0][0])
    else:
        chosen = candidates[0]
    gt_tree = _tree(gt)
    result["rmse_before"] = _rmse(pred, gt, gt_tree)
    params, current = _icp(pred, gt, chosen, iterations, trim, gt_tree, tau)
    if params is None:
        return result
    polish, polished_cloud = _icp_plain(pred, gt, chosen, iterations, 0.9, gt_tree)
    if polish is not None:
        score_polish = fscore_at(
            apply_sim3(pred_small, polish), gt_small, tau, _tree(apply_sim3(pred_small, polish)), small_tree
        )
        score_current = fscore_at(
            apply_sim3(pred_small, params), gt_small, tau, _tree(apply_sim3(pred_small, params)), small_tree
        )
        if score_polish >= score_current:
            params, current = polish, polished_cloud
    index = _nearest_index(current, gt, gt_tree)
    distance = np.linalg.norm(current - gt[index], axis=1)
    result["inliers"] = int((distance <= np.median(distance)).sum())
    result.update(params)
    result["rmse_after"] = _rmse(current, gt, gt_tree)
    result["status"] = "ok"
    return result


def refine_from(pred, gt, params, iterations=25, trim=0.8, tau=None, sample=20000, seed=0):
    """Refine a supplied sim(3) start, keeping a candidate only if its F-score improves.

    Used after camera-pose initialisation: the pose fit already places the cloud
    correctly, so ICP variants are accepted only when they measurably help.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    generator = np.random.default_rng(seed)
    pred_small = pred[np.sort(generator.choice(len(pred), size=min(len(pred), sample), replace=False))]
    gt_small = gt[np.sort(generator.choice(len(gt), size=min(len(gt), sample), replace=False))]
    if tau is None:
        tau = 0.01 * diagonal(gt)
    small_tree = _tree(gt_small)
    start = {
        "s": float(params["s"]),
        "R": np.asarray(params["R"], dtype=np.float64).tolist(),
        "t": np.asarray(params["t"], dtype=np.float64).tolist(),
    }
    cloud = apply_sim3(pred_small, start)
    best = dict(params=start, variant="camera_poses",
                fscore=float(fscore_at(cloud, gt_small, tau, _tree(cloud), small_tree)))
    for variant, runner, runner_trim in (("mutual_icp", _icp, trim), ("plain_icp", _icp_plain, 0.9)):
        candidate, _ = runner(pred_small, gt_small, start, iterations, runner_trim, small_tree)
        if candidate is None:
            continue
        moved = apply_sim3(pred_small, candidate)
        score = float(fscore_at(moved, gt_small, tau, _tree(moved), small_tree))
        if score > best["fscore"]:
            best = dict(params=candidate, variant=variant, fscore=score)
    return best
