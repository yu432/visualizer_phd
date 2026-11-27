"""DUSt3R geometry worker: RGB frames in, depth/matrices out; no GT is ever opened.

DUSt3R is not a many-view model: the network only ever sees two images at a time and predicts a
pointmap in the first view frame plus a confidence map. Geometry across the whole scene therefore
needs a step the other two methods do not have: a global aligner that optimises per-pair camera
poses and per-image depthmaps so that every pair pointmap agrees with every other one through one
common set of cameras, initialised from a minimum spanning tree of the pair graph. The cameras and
depths written here are the output of that optimisation.

The pair graph is complete for the small frame sets and a cyclic sliding window for the 100-frame
ones: a complete graph on 100 frames is 4950 edges, i.e. roughly 24 GiB of pair pointmaps held in
memory at once, which this machine does not have. Named deviation, recorded in inference.json.
"""
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

METHOD = "dust3r"
DUST3R = ROOT / "third_party/DUSt3R"


def load_frames(inference: Path) -> tuple[list[str], dict]:
    manifest = json.loads((inference / "manifest.json").read_text())
    frames = manifest["frames"]
    if len(frames) < 2:
        raise ValueError("at least two frames are required for pairwise prediction")
    for frame in frames:
        digest = sha256(inference / "images" / frame["name"])
        if digest != frame["input_sha256"]:
            raise ValueError("input hash mismatch: " + frame["name"])
    return [frame["name"] for frame in frames], manifest


def build_model(weights: Path, device: str):
    sys.path.insert(0, str(DUST3R))
    from dust3r.model import AsymmetricCroCo3DStereo

    model = AsymmetricCroCo3DStereo.from_pretrained(str(weights))
    return model.to(device).eval()


def scene_graph_for(frames: int, complete_limit: int, window: int) -> str:
    """Complete pair graph while it still fits in memory, cyclic sliding window above it."""
    if frames <= complete_limit:
        return "complete"
    return "swin-%d" % window


def run(model, images, device: str, image_size: int, graph: str, niter: int, batch_size: int):
    sys.path.insert(0, str(DUST3R))
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images

    imgs = load_images([str(path) for path in images], size=image_size, verbose=False,
                       patch_size=model.patch_size,
                       square_ok=getattr(model, "square_ok", False))
    pairs = make_pairs(imgs, scene_graph=graph, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=True)
    # Pairwise forward passes are done; the aligner needs the accelerator to itself.
    del model
    if device == "mps":
        torch.mps.empty_cache()
    mode = (GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2
            else GlobalAlignerMode.PairViewer)
    scene = global_aligner(output, device=device, mode=mode, verbose=True)
    loss = None
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule="linear", lr=0.01)
    depth = [np.asarray(d, dtype=np.float32) for d in to_numpy(scene.get_depthmaps())]
    confidence = [np.asarray(c, dtype=np.float32) for c in to_numpy(scene.get_conf())]
    cam_to_world = np.asarray(to_numpy(scene.get_im_poses()), dtype=np.float64)
    world_to_cam = np.linalg.inv(cam_to_world)
    intrinsics = np.asarray(to_numpy(scene.get_intrinsics()), dtype=np.float64)
    note = dict(pairs=len(pairs), edges=int(scene.n_edges),
                aligner_mode=str(mode).split(".")[-1],
                final_loss=None if loss is None else float(loss),
                depth_grid=list(depth[0].shape))
    return depth, confidence, world_to_cam, intrinsics, note


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/dust3r")
    parser.add_argument("--out-root", type=Path, default=OUTPUTS)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--scene-graph", default="auto",
                        help="complete, swin-N, logwin-N, oneref-N, or auto")
    parser.add_argument("--complete-limit", type=int, default=24,
                        help="frame count up to which the pair graph stays complete")
    parser.add_argument("--window", type=int, default=3,
                        help="sliding window size used above --complete-limit")
    parser.add_argument("--niter", type=int, default=300,
                        help="global alignment iterations (the published protocol)")
    parser.add_argument("--batch-size", type=int, default=1)
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
    names, _manifest = load_frames(inference)
    graph = (scene_graph_for(len(names), args.complete_limit, args.window)
             if args.scene_graph == "auto" else args.scene_graph)
    print("%s %s: %d frames, graph %s" % (args.dataset, args.scene, len(names), graph), flush=True)
    model = build_model(args.weights, args.device)
    depth, confidence, extrinsics, intrinsics, note = run(
        model, [inference / "images" / name for name in names], args.device,
        args.image_size, graph, args.niter, args.batch_size
    )
    depth = np.stack(depth)
    confidence = np.stack(confidence)
    if depth.ndim != 3 or depth.shape[0] != len(names) or depth.shape != confidence.shape:
        raise ValueError("unexpected depth shape %s" % (depth.shape,))
    if extrinsics.shape != (len(names), 4, 4) or intrinsics.shape != (len(names), 3, 3):
        raise ValueError("unexpected camera shapes %s %s" % (extrinsics.shape, intrinsics.shape))
    if not (np.isfinite(extrinsics).all() and np.isfinite(intrinsics).all()):
        raise ValueError("non-finite camera parameters")
    if not (np.isfinite(depth).all() and np.isfinite(confidence).all()):
        raise ValueError("non-finite depth output")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "depth.npy", depth)
    np.save(out / "depth_conf.npy", confidence)
    np.save(out / "extrinsics.npy", extrinsics)
    np.save(out / "intrinsics.npy", intrinsics)
    aligner = ("dust3r.cloud_opt.global_aligner(PointCloudOptimizer), init=mst, "
               "schedule=linear, lr=0.01")
    provenance = dict(
        method=METHOD, dataset=args.dataset, scene=args.scene, frames=names,
        device=args.device, precision="float32", image_size=args.image_size,
        scene_graph=graph, gt_used=False, depth_shape=list(depth.shape),
        aligner=aligner, aligner_niter=args.niter, aligner_note=note,
        source_revision=subprocess.check_output(
            ["git", "-C", str(DUST3R), "rev-parse", "HEAD"], text=True
        ).strip(),
        checkpoint=dict(
            name="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
            model_sha256=sha256(args.weights / "model.safetensors"),
            config_sha256=sha256(args.weights / "config.json"),
        ),
        torch_version=torch.__version__,
        elapsed_seconds=round(time.monotonic() - started, 3),
        manifest_sha256=sha256(inference / "manifest.json"),
        adapter="dust3r.model.AsymmetricCroCo3DStereo.from_pretrained + load_images + make_pairs",
    )
    write_json(out / "inference.json", provenance)
    print(args.dataset, args.scene, "depth", depth.shape, "edges", note["edges"],
          "elapsed %.1fs" % provenance["elapsed_seconds"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
