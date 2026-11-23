"""DA3-Giant geometry worker: RGB frames in, depth/matrices out; no GT is ever opened."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INFERENCE, OUTPUTS, sha256, write_json  # noqa: E402

METHOD = "da3-giant"


def load_frames(inference: Path) -> tuple[list[str], dict]:
    manifest = json.loads((inference / "manifest.json").read_text())
    frames = manifest["frames"]
    if len(frames) < 2:
        raise ValueError("at least two frames are required for multi-view depth")
    for frame in frames:
        digest = sha256(inference / "images" / frame["name"])
        if digest != frame["input_sha256"]:
            raise ValueError("input hash mismatch: " + frame["name"])
    return [frame["name"] for frame in frames], manifest


def build_model(weights: Path, device: str):
    sys.path.insert(0, str(ROOT / "third_party/Depth-Anything-3/src"))
    from depth_anything_3.cfg import create_object

    config = json.loads((weights / "config.json").read_text())
    model = create_object(OmegaConf.create(config["config"]))
    wrapper = torch.nn.Module()
    wrapper.add_module("model", model)
    load_model(wrapper, str(weights / "model.safetensors"), strict=True)
    del wrapper
    return model.eval().to(device)


def run(model, images, device: str, process_res: int):
    sys.path.insert(0, str(ROOT / "third_party/Depth-Anything-3/src"))
    from depth_anything_3.utils.io.input_processor import InputProcessor

    batch, _, _ = InputProcessor()(
        [str(path) for path in images],
        process_res=process_res,
        process_res_method="upper_bound_resize",
        sequential=True,
    )
    tensor = batch[None].to(device).float()
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
        output = model(tensor, ref_view_strategy="saddle_balanced", use_ray_pose=False)
    depth = output["depth"][0].float().cpu().numpy()
    confidence = output["depth_conf"][0].float().cpu().numpy()
    extrinsics = output["extrinsics"][0].float().cpu().numpy()
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (extrinsics.shape[0], 1, 1))
        extrinsics = np.concatenate([extrinsics, bottom], axis=1)
    intrinsics = output["intrinsics"][0].float().cpu().numpy()
    sky = output.get("sky")
    sky = None if sky is None else sky[0].float().cpu().numpy().reshape(depth.shape)
    return depth, confidence, extrinsics, intrinsics, list(tensor.shape), sky


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/da3-giant")
    parser.add_argument("--out-root", type=Path, default=OUTPUTS)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    inference = INFERENCE / args.dataset / args.scene
    out = args.out_root / args.dataset / args.scene / METHOD
    if (out / "inference.json").exists() and not args.force:
        print("skip (cached):", out, flush=True)
        return 0
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    names, manifest = load_frames(inference)
    model = build_model(args.weights, args.device)
    depth, confidence, extrinsics, intrinsics, shape, sky = run(
        model, [inference / "images" / name for name in names], args.device, args.process_res
    )
    if depth.ndim != 3 or depth.shape[0] != len(names) or depth.shape != confidence.shape:
        raise ValueError("unexpected depth shape %s" % (depth.shape,))
    if extrinsics.shape != (len(names), 4, 4) or intrinsics.shape != (len(names), 3, 3):  # noqa: E501
        raise ValueError("unexpected camera shapes %s %s" % (extrinsics.shape, intrinsics.shape))
    if not (np.isfinite(extrinsics).all() and np.isfinite(intrinsics).all()):
        raise ValueError("non-finite camera parameters")
    if not (np.isfinite(depth).all() and np.isfinite(confidence).all()):
        raise ValueError("non-finite depth output")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "depth.npy", depth.astype(np.float32))
    np.save(out / "depth_conf.npy", confidence.astype(np.float32))
    np.save(out / "extrinsics.npy", extrinsics.astype(np.float64))
    np.save(out / "intrinsics.npy", intrinsics.astype(np.float64))
    if sky is not None:
        np.save(out / "sky.npy", sky.astype(np.float32))
    provenance = dict(
        method=METHOD, dataset=args.dataset, scene=args.scene, frames=names,
        device=args.device, precision="autocast-float16", process_res=args.process_res,
        process_res_method="upper_bound_resize", ref_view_strategy="saddle_balanced",
        use_ray_pose=False, input_shape=shape, gt_used=False,
        depth_shape=list(depth.shape),
        sky_saved=sky is not None,
        source_revision=subprocess.check_output(
            ["git", "-C", str(ROOT / "third_party/Depth-Anything-3"), "rev-parse", "HEAD"], text=True
        ).strip(),
        checkpoint=dict(
            name=args.weights.name,
            model_sha256=sha256(args.weights / "model.safetensors"),
            config_sha256=sha256(args.weights / "config.json"),
        ),
        torch_version=torch.__version__,
        elapsed_seconds=round(time.monotonic() - started, 3),
        manifest_sha256=sha256(inference / "manifest.json"),
        adapter="depth_anything_3.cfg.create_object + InputProcessor; direct author forward",
    )
    write_json(out / "inference.json", provenance)
    print(args.dataset, args.scene, "depth", depth.shape, "elapsed %.1fs" % provenance["elapsed_seconds"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())