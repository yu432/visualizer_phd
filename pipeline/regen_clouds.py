"""Rebuild cloud sets: metric clouds at the protocol quantile, display clouds at the strict one.

The metric cloud is what the numbers are computed on (protocol: pooled confidence quantile 0.2,
SegFormer sky mask, far tail at the 95th percentile). The display cloud is the same reconstruction
with a stricter confidence cut (0.4), which is what the figures and the viewer show: the extra
low-confidence pixels are exactly the ones that smear thin structures against the sky, and they
are invisible in the numbers but very visible in a picture.
"""
import argparse
import sys

sys.path.insert(0, "method/geometry")
from build_clouds import build
from common import CLOUDS, read_json

parser = argparse.ArgumentParser()
parser.add_argument("--quantile", type=float, required=True)
parser.add_argument("--tag", default="", help="output tag; empty rebuilds the metric clouds")
parser.add_argument("--only-site", action="store_true",
                    help="only the eighteen clouds that carry site pages")
args = parser.parse_args()

SITE = [("eth3d", ("courtyard", "delivery_area", "facade", "office", "terrace")),
        ("eth3d20", ("courtyard", "delivery_area", "facade", "office", "terrace")),
        ("tnt", ("barn", "courthouse", "ignatius", "meetingroom")),
        ("tnt100", ("barn", "courthouse", "ignatius", "meetingroom"))]

targets = []
for meta_path in sorted(CLOUDS.glob("*/*/*_cloud.json")):
    meta = read_json(meta_path)
    if meta_path.parent.parent.name != meta["dataset"]:
        # A copy under another dataset root (clouds/eth3d20dense holds copies of clouds/eth3d20
        # with the old dataset field). build() would write into the *original* directory, which is
        # how a copy used to overwrite the clouds it was copied from.
        continue
    tag = meta.get("output_tag", "")
    if tag.startswith("-d") or tag == "-display":
        continue
    if args.tag and tag:
        continue  # display copies exist for the metric clouds only
    if args.only_site and meta["scene"] not in dict(SITE).get(meta["dataset"], ()):
        continue
    targets.append(meta)

print("облаков к пересборке:", len(targets), "квантиль", args.quantile, "тег", args.tag or "(metric)", flush=True)
for meta in targets:
    result = build(
        meta["dataset"], meta["scene"], meta["method"],
        voxel_fraction=float(meta["voxel_fraction"]),
        quantile=args.quantile,
        sky_threshold=float(meta["sky_threshold"]),
        far_percentile=float(meta["far_percentile"]),
        force=True,
        sky_source=meta.get("sky_source", "auto"),
        output_tag=args.tag,
        voxel_meters=float(meta.get("voxel_meters") or 0.0),
        focus_radius=float(meta.get("focus_radius") or 0.0),
        focus_voxel_fraction=float(meta.get("focus_voxel_fraction") or 0.0),
        # Absolute grid, not a fraction of the new cloud's own diagonal: the stricter quantile of a
        # display copy drops the far outliers, the diagonal shrinks and the same fraction would
        # give a finer voxel, i.e. more points than the metric cloud it is supposed to thin.
        voxel_units=float(meta["voxel"]),
        focus_voxel_units=float(meta.get("focus_voxel") or 0.0),
    )
    print("%-10s %-12s %-11s%-8s frames %3d points %9d (metric %9d) diagonal %.4f voxel %.5f" % (
        meta["dataset"], meta["scene"], meta["method"], args.tag,
        len(result["frames"]), result["points"], meta["points"], result["diagonal"],
        result["voxel"]), flush=True)
    if args.tag and result["points"] > meta["points"]:
        print("  !! variant is denser than the cloud it copies: %s/%s/%s" % (
            meta["dataset"], meta["scene"], meta["method"]), flush=True)
print("REBUILD-DONE", flush=True)
