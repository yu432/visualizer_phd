"""Camera-pose sim(3) used to place predictions into the GT world frame.

Both benchmark families ship authoritative camera poses, so the predicted cloud is
registered with the similarity transform that best maps the predicted camera
centres onto the GT ones. This is the usual way multi-view reconstructions are
compared to a reference (camera-pose alignment), and unlike a blind ICP it cannot
slide a facade onto the ground plane.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import umeyama  # noqa: E402
from common import GT, OUTPUTS, read_json  # noqa: E402


def centres(extrinsics):
    """Camera centres from world-to-camera matrices."""
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations)


def load(dataset, scene, method):
    """Predicted and GT extrinsics for one scene, matched by frame name."""
    root = OUTPUTS / dataset / scene / method
    inference = read_json(root / "inference.json")
    names = [frame["name"] if isinstance(frame, dict) else frame for frame in inference["frames"]]
    pred_extrinsics = np.load(root / "extrinsics.npy")
    pred_intrinsics = np.load(root / "intrinsics.npy")
    if len(names) != len(pred_extrinsics):
        raise ValueError("%s/%s: %d names vs %d extrinsics" % (dataset, scene, len(names), len(pred_extrinsics)))
    if pred_extrinsics.shape[1:] == (3, 4):
        bottom = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(pred_extrinsics), 1, 1))
        pred_extrinsics = np.concatenate([pred_extrinsics, bottom], axis=1)
    gt_root = GT / dataset / scene
    gt_meta = read_json(gt_root / "gt_meta.json")
    with np.load(gt_root / "cameras.npz") as handle:
        gt_extrinsics = np.asarray(handle["extrinsics"], dtype=np.float64)
        gt_intrinsics = np.asarray(handle["intrinsics"], dtype=np.float64)
    order = {name: index for index, name in enumerate(gt_meta["frames"])}
    missing = [name for name in names if name not in order]
    if missing:
        raise ValueError("%s/%s: frames without GT cameras: %s" % (dataset, scene, missing[:3]))
    index = np.array([order[name] for name in names])
    return dict(
        names=names,
        pred_extrinsics=pred_extrinsics,
        pred_intrinsics=pred_intrinsics,
        gt_extrinsics=gt_extrinsics[index],
        gt_intrinsics=gt_intrinsics[index],
    )


def sim3(dataset, scene, method):
    """Similarity transform mapping the predicted world frame onto the GT frame.

    When the reference cameras live in a frame of their own (Tanks and Temples keeps its COLMAP
    model and its laser scan in different coordinate systems) gt_meta.json carries the official
    frame_transform; the pose sim(3) is then composed with it, so the result still maps the
    predicted cloud straight onto the GT cloud.
    """
    bundle = load(dataset, scene, method)
    pred_centres = centres(bundle["pred_extrinsics"])
    gt_centres = centres(bundle["gt_extrinsics"])
    scale, rotation, translation = umeyama(pred_centres, gt_centres)
    residual = np.linalg.norm(scale * pred_centres @ rotation.T + translation - gt_centres, axis=1)
    extent = float(np.linalg.norm(gt_centres.max(axis=0) - gt_centres.min(axis=0)))
    frame = read_json(GT / dataset / scene / "gt_meta.json").get("frame_transform")
    source = "gt_camera_poses"
    if frame is not None:
        frame = np.asarray(frame, dtype=np.float64)
        scale = scale * float(np.cbrt(np.linalg.det(frame[:3, :3])))
        translation = frame[:3, :3] @ translation + frame[:3, 3]
        rotation = frame[:3, :3] @ rotation
        rotation = rotation / float(np.cbrt(np.linalg.det(rotation)))
        source = "gt_camera_poses+official_frame_transform"
    return dict(
        s=float(scale),
        R=rotation.tolist(),
        t=translation.tolist(),
        status="ok",
        source=source,
        centre_rms=float(np.sqrt((residual ** 2).mean())),
        gt_camera_extent=extent,
        frames=len(bundle["names"]),
    )
