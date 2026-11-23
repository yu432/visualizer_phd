#!/usr/bin/env python3
"""Verify that DA3-Giant and VGGT-Omega processed byte-identical input frames.

For every dataset/scene the check compares

1. frame names + order recorded by each method (inference.json) against the
   shared frame manifest (data/inference/<ds>/<scene>/manifest.json);
2. manifest_sha256 recorded by each method against a fresh hash of that file;
3. sha256 of every image on disk against manifest.input_sha256;
4. tensor shapes (depth, conf, intrinsics, extrinsics, sky masks) against the
   number of frames.

Native inference grids (input_shape) are allowed to differ between the
methods: they are a property of each network, not of the input frame set.
Use --json to emit machine-readable output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "inference"
OUT = ROOT / "outputs"
METHODS = ("da3-giant", "vggt-omega")
GT_DEPTH_DIR = {"eth3d": ("data", "gt", "eth3d")}
SCENES = {
    "eth3d": ["courtyard", "delivery_area", "facade", "office", "terrace"],
    "tnt": ["barn", "courthouse", "ignatius", "meetingroom"],
    "tnt100": ["barn", "courthouse", "ignatius", "meetingroom"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def leading_dims(path: Path):
    arr = np.load(path, mmap_mode="r")
    return tuple(arr.shape)


def check_scene(dataset: str, scene: str) -> dict:
    scene_dir = DATA / dataset / scene
    manifest_path = scene_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = [f["name"] for f in manifest["frames"]]
    recomputed = sha256_file(manifest_path)

    row = {
        "dataset": dataset,
        "scene": scene,
        "n_frames_manifest": len(expected),
        "manifest_sha256_recomputed": recomputed,
        "problems": [],
        "methods": {},
    }

    gt_dir = GT_DEPTH_DIR.get(dataset)
    gt_found = 0
    if gt_dir is not None:
        gt_scene = ROOT.joinpath(*gt_dir) / scene
        ddir = gt_scene / "depths"
        for name in expected:
            if any((ddir / (Path(name).stem + ext)).exists()
                   for ext in (".npz", ".npy", ".png")):
                gt_found += 1
        row["gt_depths"] = f"{gt_found}/{len(expected)}"
        if gt_found != len(expected):
            row["problems"].append(
                f"GT depths missing: {gt_found}/{len(expected)}")

        # the GT cloud must have been fused from the same frame set
        meta_path = gt_scene / "gt_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            row["gt_cloud_manifest_sha256_matches"] = (
                meta.get("source_manifest_sha256") == recomputed)
            row["gt_cloud_frames_equal_manifest"] = (
                list(meta.get("frames") or []) == expected)
            if not row["gt_cloud_manifest_sha256_matches"]:
                row["problems"].append("GT cloud built from another manifest")
            if not row["gt_cloud_frames_equal_manifest"]:
                row["problems"].append("GT cloud frame list != manifest")

    for method in METHODS:
        mdir = OUT / dataset / scene / method
        info = json.loads((mdir / "inference.json").read_text())
        frames = list(info.get("frames") or [])
        entry = {
            "frames_equal_manifest": frames == expected,
            "manifest_sha256_recorded": info.get("manifest_sha256"),
            "manifest_sha256_matches": info.get("manifest_sha256") == recomputed,
            "input_shape": info.get("input_shape"),
            "depth_shape": info.get("depth_shape"),
            "precision": info.get("precision"),
        }
        if frames != expected:
            entry["frames_recorded"] = frames
            row["problems"].append(f"{method}: frame list != manifest")
        if not entry["manifest_sha256_matches"]:
            row["problems"].append(f"{method}: manifest_sha256 mismatch")

        n = len(expected)
        for fname, expect_n in (
            ("depth.npy", True),
            ("depth_conf.npy", True),
            ("sky_masks.npy", True),
            ("intrinsics.npy", True),
            ("extrinsics.npy", True),
        ):
            p = mdir / fname
            if not p.exists():
                row["problems"].append(f"{method}: missing {fname}")
                continue
            shape = leading_dims(p)
            if shape[0] != n and not (fname == "extrinsics.npy" and shape[0] in (n, n - 1)):
                row["problems"].append(
                    f"{method}: {fname} has {shape[0]} entries, expected {n}")
            entry[fname.split(".")[0]] = list(shape)

        # pose evaluation must reference the same frames, in the same order
        pose_path = mdir / "pose_evaluation.json"
        if pose_path.exists():
            pose = json.loads(pose_path.read_text())
            pose_names = [r["name"] for r in pose.get("per_frame") or []]
            entry["pose_frames_equal_manifest"] = pose_names == expected
            entry["pose_frames"] = len(pose_names)
            if pose_names != expected:
                row["problems"].append(
                    f"{method}: pose_evaluation frames != manifest")
        row["methods"][method] = entry

    # cross-method: identical frame list?
    a = row["methods"].get(METHODS[0], {}).get("frames_equal_manifest")
    b = row["methods"].get(METHODS[1], {}).get("frames_equal_manifest")
    row["identical_frames"] = bool(a and b)

    # image bytes on disk
    img_dir = scene_dir / "images"
    bad = [f["name"] for f in manifest["frames"]
           if sha256_file(img_dir / f["name"]) != f["input_sha256"]]
    row["images_sha256_ok"] = not bad
    if bad:
        row["problems"].append(f"image sha256 mismatch: {bad}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = [check_scene(ds, sc) for ds, scenes in SCENES.items() for sc in scenes]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        head = (f"{'scene':<16}{'frames':>7}{'same':>6}{'mhash':>7}"
                f"{'imgs':>6}{'pose':>6}{'GT d':>7}{'GTcl':>6}  "
                f"da3 grid / omega grid")
        print(head)
        print("-" * len(head))
        for r in rows:
            da3 = r["methods"]["da3-giant"].get("input_shape")
            om = r["methods"]["vggt-omega"].get("input_shape")
            mhash = all(m["manifest_sha256_matches"] for m in r["methods"].values())
            pose_ok = all(m.get("pose_frames_equal_manifest")
                          for m in r["methods"].values())
            gtcl = r.get("gt_cloud_manifest_sha256_matches")
            gtcl_txt = "-" if gtcl is None else ("ok" if gtcl else "BAD")
            print(f"{r['dataset'] + '/' + r['scene']:<16}"
                  f"{r['n_frames_manifest']:>7}"
                  f"{'yes' if r['identical_frames'] else 'NO':>6}"
                  f"{'ok' if mhash else 'BAD':>7}"
                  f"{'ok' if r['images_sha256_ok'] else 'BAD':>6}"
                  f"{'ok' if pose_ok else 'BAD':>6}"
                  f"{r.get('gt_depths', '-'):>7}"
                  f"{gtcl_txt:>6}  "
                  f"{'x'.join(map(str, da3)) if da3 else '?'} / "
                  f"{'x'.join(map(str, om)) if om else '?'}")
        bad_rows = [r for r in rows if r["problems"]]
        print()
        if bad_rows:
            for r in bad_rows:
                print(f"PROBLEM {r['dataset']}/{r['scene']}: {r['problems']}")
            return 1
        print(f"all {len(rows)} scenes: identical frame sets, "
              f"identical manifest hash, image bytes verified")
    return 0 if not any(r["problems"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
