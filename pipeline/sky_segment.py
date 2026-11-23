"""Sky segmentation for frames of a scene: DA3-Giant ships no sky head, so mask the sky explicitly."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INFERENCE, OUTPUTS, read_json, write_json  # noqa: E402

MODEL_ID = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
SKY_INDEX = 10
SHORT_SIDE = 512


def load_model(device):
    import torch
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.eval().to(device)
    return model


def frame_masks(model, frame_paths, device, threshold=0.5, dilate=2, size=None):
    """Return a bool array (frames, H, W) that is True on sky pixels; size is the prediction grid."""
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from scipy.ndimage import binary_dilation

    masks = []
    for path in frame_paths:
        opened = Image.open(path).convert("RGB")
        source_width, source_height = opened.size
        height, width = (source_height, source_width) if size is None else (int(size[0]), int(size[1]))
        scale = SHORT_SIDE / min(source_width, source_height)
        model_size = (max(32, int(round(source_height * scale / 32)) * 32),
                      max(32, int(round(source_width * scale / 32)) * 32))
        resized = opened.resize((model_size[1], model_size[0]), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalised = (array - mean) / std
        tensor = torch.from_numpy(np.ascontiguousarray(normalised.transpose(2, 0, 1))).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(pixel_values=tensor).logits
        logits = functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        probability = torch.softmax(logits, dim=1)[0, SKY_INDEX].cpu().numpy()
        mask = probability > threshold
        if dilate > 0:
            mask = binary_dilation(mask, iterations=int(dilate))
        masks.append(mask)
    return np.stack(masks)


def segment(dataset, scene, method, threshold=0.5, dilate=2, device=None, force=False):
    import torch

    inference = INFERENCE / dataset / scene
    out = OUTPUTS / dataset / scene / method
    target = out / "sky_masks.npy"
    meta_path = out / "sky_meta.json"
    if target.exists() and meta_path.exists() and not force:
        return read_json(meta_path)
    manifest = read_json(inference / "manifest.json")
    names = [frame["name"] for frame in manifest["frames"]]
    paths = [inference / "images" / name for name in names]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    depth = np.load(out / "depth.npy", mmap_mode="r")
    if len(depth) != len(names):
        raise ValueError("prediction/frame count mismatch")
    model = load_model(device)
    masks = frame_masks(model, paths, device, threshold, dilate, size=depth.shape[1:])
    np.save(target, masks)
    fractions = [float(mask.mean()) for mask in masks]
    meta = dict(
        dataset=dataset,
        scene=scene,
        method=method,
        model=MODEL_ID,
        label="sky",
        threshold=float(threshold),
        dilation_px=int(dilate),
        device=device,
        frames=names,
        sky_fraction_per_frame=fractions,
        sky_fraction_mean=float(np.mean(fractions)),
        shape=list(masks.shape),
        masks_sha256=None,
    )
    write_json(meta_path, meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dilate", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for scene in args.scenes:
        meta = segment(args.dataset, scene, args.method, args.threshold, args.dilate, force=args.force)
        print(scene, "sky %.3f" % meta["sky_fraction_mean"], meta["device"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
