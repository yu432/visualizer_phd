"""Build a predicted point cloud for one scene from a saved depth prediction."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLOUDS,
    INFERENCE,
    OUTPUTS,
    backproject_coloured,
    diagonal,
    read_json,
    sha256,
    voxel_downsample_attributed,
    voxel_downsample_focused,
    write_json,
    write_ply,
)


def camera_centre(extrinsics):
    """Mean camera position of a prediction, i.e. the point the rig was pointed at."""
    rotation = np.asarray(extrinsics)[:, :3, :3]
    translation = np.asarray(extrinsics)[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(rotation, (0, 2, 1)), translation).mean(axis=0)


def load_prediction(out: Path) -> dict:
    depth = np.load(out / "depth.npy")
    intrinsics = np.load(out / "intrinsics.npy")
    extrinsics = np.load(out / "extrinsics.npy")
    confidence_path = out / "depth_conf.npy"
    confidence = np.load(confidence_path) if confidence_path.exists() else None
    sky_path = out / "sky.npy"
    sky = np.load(sky_path) if sky_path.exists() else None
    mask_path = out / "sky_masks.npy"
    sky_masks = np.load(mask_path) if mask_path.exists() else None
    if extrinsics.shape[1:] == (3, 4):
        bottom = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (len(extrinsics), 1, 1))
        extrinsics = np.concatenate([extrinsics, bottom], axis=1)
    return dict(depth=depth, intrinsics=intrinsics, extrinsics=extrinsics, confidence=confidence,
                sky=sky, sky_masks=sky_masks)


def confidence_mask(confidence, quantile):
    """Pixels whose depth confidence is below the pooled quantile are dropped.

    The protocol value is 0.2: that is what the published numbers are computed on. The display
    clouds (figures, viewer) are built at 0.4 — the level at which the thin, sky-bordered
    structures stop smearing. On courthouse 100 frames the rooftop statue and the flags around
    the parapet kept their low-confidence pixels at 0.2 and the merged cloud grew a multi-metre
    plume of them above the roof; at 0.4 the plume disappears on both methods, at the price of
    completeness (barn 100 frames recall@1 0.881 -> 0.450), which is why the tables stay at 0.2.
    Raising it further starts thinning real surface.
    """
    if confidence is None or quantile <= 0:
        return None
    threshold = np.quantile(confidence, quantile)
    return confidence >= threshold


def sky_mask(sky, threshold):
    """DA3 pushes sky pixels to the maximum depth, so they must be dropped explicitly."""
    if sky is None:
        return None
    return sky < threshold


def far_mask(depth, percentile):
    """Drop the far tail of each frame: without a sky head the sky becomes huge depths."""
    if percentile <= 0:
        return None
    valid = np.isfinite(depth) & (depth > 0)
    cap = np.quantile(depth[valid], percentile / 100.0)
    return valid & (depth <= cap)


def build(dataset: str, scene: str, method: str, voxel_fraction: float, quantile: float,
          sky_threshold: float, far_percentile: float, force: bool, sky_source: str = "auto",
          output_tag: str = "", voxel_meters: float = 0.0, focus_radius: float = 0.0,
          focus_voxel_fraction: float = 0.0, voxel_units: float = 0.0,
          focus_voxel_units: float = 0.0) -> dict:
    inference = INFERENCE / dataset / scene
    out = OUTPUTS / dataset / scene / method
    target_dir = CLOUDS / dataset / scene
    name_tag = method + output_tag
    meta_path = target_dir / (name_tag + "_cloud.json")
    if meta_path.exists() and not force:
        return read_json(meta_path)
    manifest = read_json(inference / "manifest.json")
    names = [frame["name"] for frame in manifest["frames"]]
    prediction = load_prediction(out)
    depth = prediction["depth"]
    if len(depth) != len(names):
        raise ValueError("prediction/frame count mismatch")
    masks = confidence_mask(prediction["confidence"], quantile)
    segmentation = None
    used_sky = "none"
    if sky_source != "none":
        if prediction["sky_masks"] is not None:
            segmentation = ~prediction["sky_masks"].astype(bool)
            used_sky = "segformer"
        elif sky_source == "segformer":
            raise FileNotFoundError("sky_masks.npy is missing for %s/%s" % (dataset, scene))
        if prediction["sky"] is not None:
            used_sky = "da3" if used_sky == "none" else used_sky
    for extra in (sky_mask(prediction["sky"], sky_threshold), segmentation,
                  far_mask(depth, far_percentile)):
        if extra is not None:
            masks = extra if masks is None else (masks & extra)
    points, colours = backproject_coloured(
        depth, prediction["intrinsics"], prediction["extrinsics"],
        [inference / "images" / name for name in names], masks,
    )
    if len(points) < 1000:
        raise ValueError("predicted cloud too small for %s/%s" % (dataset, scene))
    if voxel_units:
        # Absolute voxel in cloud units, taken from an existing meta. A fraction-of-diagonal voxel
        # is relative to the cloud the confidence quantile happens to leave: a stricter quantile
        # drops the far outliers, the diagonal shrinks and the grid silently gets finer, so a
        # "thinned copy" comes out denser than the cloud it copies. Variants of an existing cloud
        # therefore reuse its absolute voxel (regen_clouds.py).
        voxel = float(voxel_units)
    elif voxel_meters:
        # Metric voxel: on Tanks and Temples the prediction covers the whole surroundings, so a
        # fixed fraction of its diagonal would be coarser than the 2-6 cm lidar spacing and the
        # recall would measure the voxel grid instead of the reconstruction.
        from pose_align import sim3 as pose_sim3
        voxel = float(voxel_meters) / float(pose_sim3(dataset, scene, method)["s"])
    else:
        voxel = voxel_fraction * diagonal(points)
    focus_centre = None
    focus_voxel = 0.0
    focus_points = 0
    if focus_radius > 0 and (focus_voxel_fraction > 0 or focus_voxel_units > 0):
        # Refine the grid around the point the rig was aimed at: on Tanks and Temples that is the
        # object the scene is about, and a fraction-of-diagonal voxel spends the whole budget on
        # the surrounding ground instead.
        focus_centre = camera_centre(prediction["extrinsics"])
        focus_voxel = (float(focus_voxel_units) if focus_voxel_units
                       else float(focus_voxel_fraction) * diagonal(points))
        focus_points = int((np.linalg.norm(points - focus_centre, axis=1)
                            <= focus_radius).sum())
    reduced, kept = voxel_downsample_focused(
        points, voxel, colours, focus_centre, focus_radius, focus_voxel)
    cloud_path = target_dir / (name_tag + ".ply")
    write_ply(cloud_path, reduced, kept)
    meta = dict(
        dataset=dataset,
        scene=scene,
        method=method,
        frames=names,
        confidence_quantile=float(quantile),
        sky_threshold=float(sky_threshold),
        far_percentile=float(far_percentile),
        sky_available=prediction["sky"] is not None,
        sky_source=used_sky,
        output_tag=output_tag,
        sky_fraction_mean=(None if prediction["sky_masks"] is None
                           else float(prediction["sky_masks"].mean())),
        voxel=float(voxel),
        voxel_fraction=float(voxel_fraction),
        voxel_meters=float(voxel_meters) or None,
        raw_points=int(len(points)),
        points=int(len(reduced)),
        focus_radius=float(focus_radius) or None,
        focus_voxel=float(focus_voxel) or None,
        focus_voxel_fraction=float(focus_voxel_fraction) or None,
        focus_centre=None if focus_centre is None else [float(value) for value in focus_centre],
        focus_raw_points=focus_points,
        diagonal=diagonal(reduced),
        cloud_sha256=sha256(cloud_path),
        inference_sha256=sha256(out / "inference.json"),
        source_manifest_sha256=sha256(inference / "manifest.json"),
    )
    write_json(meta_path, meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--voxel-fraction", type=float, default=0.0015)
    parser.add_argument("--voxel-meters", type=float, default=0.0,
                        help="voxel size in GT units (overrides --voxel-fraction)")
    parser.add_argument("--confidence-quantile", type=float, default=0.2,
                        help="protocol value; the display clouds (figures, viewer) are built "
                             "with --output-tag=-display at 0.4, see regen_clouds.py")
    parser.add_argument("--sky-threshold", type=float, default=0.3)
    parser.add_argument("--far-percentile", type=float, default=95.0)
    parser.add_argument("--sky-source", choices=["auto", "segformer", "none"], default="auto")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--focus-radius", type=float, default=0.0,
                        help="refine the voxel grid inside this radius around the mean camera "
                             "centre, in cloud units")
    parser.add_argument("--focus-voxel-fraction", type=float, default=0.0,
                        help="voxel fraction used inside the focus radius")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for scene in args.scenes:
        meta = build(args.dataset, scene, args.method, args.voxel_fraction,
                     args.confidence_quantile, args.sky_threshold, args.far_percentile, args.force,
                     args.sky_source, args.output_tag, args.voxel_meters, args.focus_radius,
                     args.focus_voxel_fraction)
        print(scene, args.method, "points", meta["points"], "diagonal %.4f" % meta["diagonal"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())