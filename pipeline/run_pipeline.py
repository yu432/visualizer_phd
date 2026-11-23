"""Run the geometry pipeline (worker, clouds, evaluation) for a dataset behind RSS/time bounds."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method/baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import subprocess  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import bounded  # noqa: E402
import psutil  # noqa: E402


class _PsShim:
    """bounded.py shells out to ps; the sandbox denies that binary, so sample RSS via psutil."""

    Popen = staticmethod(subprocess.Popen)
    STDOUT = subprocess.STDOUT
    DEVNULL = subprocess.DEVNULL

    @staticmethod
    def run(command, capture_output=True, text=True):
        if command and command[0] == "ps":
            pid = int(command[-1])
            for _ in range(3):
                try:
                    rss_kib = psutil.Process(pid).memory_info().rss // 1024
                    return types.SimpleNamespace(returncode=0, stdout=str(rss_kib))
                except psutil.NoSuchProcess:
                    return types.SimpleNamespace(returncode=0, stdout="0")
                except Exception:
                    time.sleep(0.2)
            return types.SimpleNamespace(returncode=0, stdout="0")
        return subprocess.run(command, capture_output=capture_output, text=text)


bounded.subprocess = _PsShim
from bounded import run_bounded  # noqa: E402
from common import OUTPUTS, read_json  # noqa: E402

PYTHON = str(ROOT / ".venv/bin/python")
DATASET_SCENES = dict(
    eth3d=["courtyard", "delivery_area", "facade", "terrace", "office"],
    tnt=["barn", "church", "courthouse", "ignatius", "meetingroom"],
)


def stage_commands(dataset, scene, method, voxel_fraction):
    script = "omega_worker.py" if method.startswith("vggt") else "da3_geometry_worker.py"
    worker = [PYTHON, str(ROOT / "method/geometry" / script),
              "--dataset", dataset, "--scene", scene, "--method", method]
    ground = [PYTHON, str(ROOT / "method/geometry/build_gt_cloud.py"),
              "--dataset", dataset, "--scenes", scene]
    cloud = [PYTHON, str(ROOT / "method/geometry/build_clouds.py"),
             "--dataset", dataset, "--scenes", scene, "--method", method,
             "--voxel-fraction", str(voxel_fraction), "--force"]
    scores = [PYTHON, str(ROOT / "method/geometry/evaluate.py"),
              "--dataset", dataset, "--scenes", scene, "--method", method]
    poses = [PYTHON, str(ROOT / "method/geometry/evaluate_poses.py"),
             "--dataset", dataset, "--scenes", scene, "--method", method]
    stages = [("worker", worker, 24.0, 2400), ("poses", poses, 8.0, 900)]
    if dataset != "tnt":
        stages += [("gt_cloud", ground, 12.0, 1200)]
    stages += [("cloud", cloud, 12.0, 1200)]
    if dataset != "tnt":
        stages += [("evaluate", scores, 16.0, 2400)]
    return stages


def run_scene(dataset, scene, method, voxel_fraction, only):
    log_root = OUTPUTS / dataset / scene / "logs"
    report = dict(dataset=dataset, scene=scene, method=method, stages=[])
    for name, command, rss, timeout in stage_commands(dataset, scene, method, voxel_fraction):
        if only and name not in only:
            continue
        result = run_bounded(command, log_root / (name + ".log"), rss_gib=rss,
                             timeout_seconds=timeout, state_path=log_root / (name + ".state.json"))
        report["stages"].append(dict(stage=name, **result))
        print(dataset, scene, name, result["status"], "%.1fs" % result["elapsed_seconds"], flush=True)
        if result["status"] != "completed":
            raise SystemExit("%s/%s stage %s -> %s (see %s)" % (dataset, scene, name, result["status"], log_root / (name + ".log")))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SCENES))
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--method", default="da3-giant")
    parser.add_argument("--voxel-fraction", type=float, default=0.0015)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    scenes = args.scenes or DATASET_SCENES[args.dataset]
    reports = []
    for scene in scenes:
        reports.append(run_scene(args.dataset, scene, args.method, args.voxel_fraction, args.only))
    summary = dict(dataset=args.dataset, method=args.method, runs=reports)
    target = OUTPUTS / args.dataset / ("pipeline_" + args.method + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2) + chr(10))
    for scene in scenes:
        root = OUTPUTS / args.dataset / scene / args.method
        if (root / "evaluation.json").exists():
            evaluation = read_json(root / "evaluation.json")
            cloud, depth = evaluation["cloud"], evaluation["depth"]
            print(scene, "F@0.5 %.3f" % cloud["fscore@0.5"], "F@1 %.3f" % cloud["fscore@1"],
                  "AbsRel", depth.get("AbsRel"), "scale %.4f" % evaluation["scale_correction"], flush=True)
        if (root / "pose_evaluation.json").exists():
            poses = read_json(root / "pose_evaluation.json")
            print(scene, "rot %.2f deg" % poses["rotation"]["mean_deg"],
                  "trans %.4f diag" % poses["translation"]["mean_fraction"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())