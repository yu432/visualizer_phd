#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
echo "=== venv ==="
python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
echo "=== base deps ==="
.venv/bin/pip install -q -r method/baselines/requirements-lock.txt
echo "=== torch ==="
.venv/bin/pip install -q torch torchvision
echo "=== open3d/matplotlib ==="
.venv/bin/pip install -q open3d matplotlib plyfile omegaconf addict imageio 2>&1 | tail -5 || echo "OPEN3D_FAILED"
echo "=== clone DA3 ==="
mkdir -p third_party
if [ ! -d third_party/Depth-Anything-3 ]; then
  git clone -q https://github.com/ByteDance-Seed/Depth-Anything-3.git third_party/Depth-Anything-3 ||   git clone -q https://github.com/depth-anything/Depth-Anything-3.git third_party/Depth-Anything-3
fi
git -C third_party/Depth-Anything-3 rev-parse HEAD
.venv/bin/pip install -q -e third_party/Depth-Anything-3 2>&1 | tail -3 || echo "DA3_INSTALL_FAILED"
echo "=== DA3 weights ==="
.venv/bin/python - <<'PY'
import urllib.request, os, hashlib
from pathlib import Path
out = Path("weights/da3-giant"); out.mkdir(parents=True, exist_ok=True)
for name in ("config.json", "model.safetensors"):
    dest = out / name
    if dest.exists() and dest.stat().st_size > 0:
        print("exists", name); continue
    url = f"https://huggingface.co/depth-anything/DA3-GIANT-1.1/resolve/main/{name}"
    print("downloading", name, flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, dest.with_suffix(dest.suffix + ".part").open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    dest.with_suffix(dest.suffix + ".part").replace(dest)
    print("done", name, dest.stat().st_size, flush=True)
PY
echo "=== BOOTSTRAP_OK ==="
