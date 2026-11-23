"""VGGT-Omega geometry worker: RGB frames in, depth/matrices out; no GT is ever opened."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INFERENCE, OUTPUTS, sha256, write_json  # noqa: E402

METHOD = "vggt-omega"
REPO = ROOT / "repo/vggt-omega"


def load_frames(inference: Path):
    manifest = json.loads((inference / "manifest.json").read_text())
    frames = manifest["frames"]
    if len(frames) < 2:
        raise ValueError("at least two frames are required for multi-view depth")
    for frame in frames:
        digest = sha256(inference / "images" / frame["name"])
        if digest != frame["input_sha256"]:
            raise ValueError("input hash mismatch: " + frame["name"])
    return [frame["name"] for frame in frames], manifest


def build_model(checkpoint: Path, device: str):
    sys.path.insert(0, str(REPO))
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega(autocast=False).eval()
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device)


def run(model, images, device: str, image_resolution: int):
    sys.path.insert(0, str(REPO))
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    batch = load_and_preprocess_images([str(path) for path in images],
                                       image_resolution=image_resolution).to(device)
    with torch.inference_mode():
        predictions = model(batch)
    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    depth = predictions["depth"].float().cpu().numpy()
    depth = depth[0] if depth.ndim == 5 else depth
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    confidence = predictions["depth_conf"].float().cpu().numpy()
    confidence = confidence[0] if confidence.ndim == 4 else confidence
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    extrinsics = extrinsic.float().cpu().numpy()
    if extrinsics.ndim == 4:
        extrinsics = extrinsics[0]
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (extrinsics.shape[0], 1, 1))
        extrinsics = np.concatenate([extrinsics, bottom], axis=1)
    intrinsics = intrinsic.float().cpu().numpy()
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]
    return depth, confidence, extrinsics, intrinsics, list(batch.shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/vggt-omega/vggt_omega_1b_512.pt")
    parser.add_argument("--out-root", type=Path, default=OUTPUTS)
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    inference = INFERENCE / args.dataset / args.scene
    out = args.out_root / args.dataset / args.scene / args.method
    if (out / "inference.json").exists() and not args.force:
        print("skip (cached):", out, flush=True)
        return 0
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    names, manifest = load_frames(inference)
    model = build_model(args.checkpoint, args.device)
    depth, confidence, extrinsics, intrinsics, shape = run(
        model, [inference / "images" / name for name in names], args.device, args.image_resolution
    )
    if depth.ndim != 3 or depth.shape[0] != len(names):
        raise ValueError("unexpected depth shape %s" % (depth.shape,))
    if extrinsics.shape != (len(names), 4, 4) or intrinsics.shape != (len(names), 3, 3):
        raise ValueError("unexpected camera shapes %s %s" % (extrinsics.shape, intrinsics.shape))
    if confidence.shape != depth.shape:
        raise ValueError("depth/confidence shape mismatch %s %s" % (depth.shape, confidence.shape))
    if not (np.isfinite(extrinsics).all() and np.isfinite(intrinsics).all()):
        raise ValueError("non-finite camera parameters")
    if not (np.isfinite(depth).all() and np.isfinite(confidence).all()):
        raise ValueError("non-finite depth output")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "depth.npy", depth.astype(np.float32))
    np.save(out / "depth_conf.npy", confidence.astype(np.float32))
    np.save(out / "extrinsics.npy", extrinsics.astype(np.float64))
    np.save(out / "intrinsics.npy", intrinsics.astype(np.float64))
    provenance = dict(
        method=args.method, dataset=args.dataset, scene=args.scene, frames=names,
        device=args.device, precision="float32", image_resolution=args.image_resolution,
        load_mode="balanced", input_shape=shape, gt_used=False,
        depth_shape=list(depth.shape),
        source_revision=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        checkpoint=dict(
            name=args.checkpoint.name,
            model_sha256=sha256(args.checkpoint),
            provenance="mirror 1kaiser/vggt-omega-jax (byte-identical to facebook/VGGT-Omega)",
        ),
        torch_version=torch.__version__,
        elapsed_seconds=round(time.monotonic() - started, 3),
        manifest_sha256=sha256(inference / "manifest.json"),
        adapter="vggt_omega.models.VGGTOmega (autocast off) + load_and_preprocess_images + encoding_to_camera",
    )
    write_json(out / "inference.json", provenance)
    print(args.dataset, args.scene, "depth", depth.shape, "elapsed %.1fs" % provenance["elapsed_seconds"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
