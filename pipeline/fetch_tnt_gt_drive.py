"""Download the official Tanks and Temples laser ground truth from Google Drive.

The dataset itself lives on Google Drive and the download page now links every training scene
with a resource key (https://www.tanksandtemples.org/download/). The resource key is what makes
the files readable without a Google account: the bare file id answers with a sign-in page, the
keyed link answers with Drive's "virus scan warning", and following that form starts the
transfer. The only remaining obstacle is Drive's per-file download quota, which replies
"Quota exceeded" for a while after the file has been downloaded often. That block is transient,
so this script resolves one file at a time, resumes partial transfers and can be run in a retry
loop:

    python fetch_tnt_gt_drive.py --check
    python fetch_tnt_gt_drive.py --scene ignatius --scene meetingroom --retry 8 --wait 600

Files land in data/_staging/tnt/gt next to the Barn.ply that came from the gtsfm mirror, named as
they are shipped (<Scene>.ply, <Scene>_trans.txt, <Scene>.json). prepare_tnt_gt.py consumes them
afterwards:

    python prepare_tnt_gt.py --scene ignatius \
        --cloud data/_staging/tnt/gt/Ignatius.ply \
        --crop-json data/_staging/tnt/gt/Ignatius.json \
        --transform data/_staging/tnt/gt/Ignatius_trans.txt \
        --source "official tanksandtemples.org Drive link"
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data/_staging/tnt/gt"  # same place the gtsfm mirror put Barn.ply
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

# scene -> role -> (drive file id, resource key). Read off the download page: the "ground truth"
# column of the training table holds the laser scan, the "Results on Training Data" table holds
# the alignment matrix and the crop volume. resource keys only exist for the 2017-era folders.
ASSETS = {
    "ignatius": {
        "cloud": ("0B-ePgl6HF260aFJKYm9VdWNVbU0", "0-SQeDh2MSoHxFkSCNijG79Q"),
        "transform": ("1wSCbCrOT7GsGVLDq0aXs4RHhs4FUzUeM", ""),
        "crop": ("1_0fESbNxfNI5NWzQ4RBhh460a9_0J54q", ""),
    },
    "meetingroom": {
        "cloud": ("0B-ePgl6HF260Ty1QRGt3bmRMcVE", "0-3eY6JXVmtYvS2CFF6klcIA"),
        "transform": ("1ZQ9g0g0VWrdmihj6PnU-KS3rYZxIf7zH", ""),
        "crop": ("1Rsc5BPURtVvWhQIfZGQWh6M6QDfl76GW", ""),
    },
}
FILENAMES = {
    ("ignatius", "cloud"): "Ignatius.ply",
    ("ignatius", "transform"): "Ignatius_trans.txt",
    ("ignatius", "crop"): "Ignatius.json",
    ("meetingroom", "cloud"): "Meetingroom.ply",
    ("meetingroom", "transform"): "Meetingroom_trans.txt",
    ("meetingroom", "crop"): "Meetingroom.json",
}


class QuotaExceeded(RuntimeError):
    """Drive answered "Quota exceeded": the file exists and is readable, retry later."""


class LoginRequired(RuntimeError):
    """Drive answered with a sign-in page: the resource key is missing or no longer valid."""


def drive_url(file_id, resource_key=""):
    url = "https://drive.google.com/uc?export=download&id=" + file_id
    if resource_key:
        url += "&resourcekey=" + resource_key
    return url


def form_params(html):
    """Pull the confirmation form out of Drive's "virus scan warning" page."""
    match = re.search(r'<form[^>]*action="([^"]+)"[^>]*>([\s\S]*?)</form>', html)
    if not match:
        return None
    action, inner = match.group(1), match.group(2)
    pairs = re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', inner)
    return action, dict(pairs)


def bandwidth_note(html):
    """Drive names the file and its size in the warning page: "Ignatius.ply (129M) is too large"."""
    text = re.sub(r"<[^>]+>", " ", html)
    match = re.search(r"([\w\-]+\.\w{2,5})\s*\((\d+[KMG]?)\) is too large", text)
    return match.group(1).strip() + " (" + match.group(2) + ")" if match else None


def open_url(url, byte_range=None, timeout=120):
    headers = {"User-Agent": UA}
    if byte_range:
        headers["Range"] = "bytes=%d-%d" % byte_range
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def resolve(file_id, resource_key=""):
    """Return (direct_url, shipped_name) or raise QuotaExceeded / LoginRequired."""
    with open_url(drive_url(file_id, resource_key)) as response:
        html = response.read().decode("utf-8", "replace")
    if "Quota exceeded" in html:
        raise QuotaExceeded("Drive quota exceeded")
    if "accounts.google.com" in html or "ServiceLogin" in html:
        raise LoginRequired("Drive requires a Google sign-in")
    params = form_params(html)
    if params is None:
        # No warning page at all: the response body is the file itself, so it is small enough
        # for Drive to serve directly.
        return drive_url(file_id, resource_key), None
    action, fields = params
    return action + "?" + urllib.parse.urlencode(fields), bandwidth_note(html)


def probe(url):
    """Read the first bytes of a resolved link to tell a real transfer from the quota page."""
    with open_url(url, (0, 4096)) as response:
        head = response.read(4096)
    if b"Quota exceeded" in head:
        raise QuotaExceeded("Drive quota exceeded")
    return head


def download(url, destination, resume=True):
    """Stream url into destination, continuing from the bytes already on disk when possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = destination.stat().st_size if (resume and destination.exists()) else 0
    with open_url(url, (offset, 1 << 40) if offset else None) as response:
        head = response.read(1 << 16)
        if head[:1] == b"<" and b"Quota exceeded" in head:
            raise QuotaExceeded("Drive quota exceeded")
        mode = "ab" if offset and response.status == 206 else "wb"
        with open(destination, mode) as sink:
            sink.write(head)
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                sink.write(chunk)
    return destination.stat().st_size


def verify(kind, path):
    """Cheap sanity check per asset kind; returns a short description or raises ValueError."""
    if kind == "cloud":
        with open(path, "rb") as handle:
            head = handle.read(512)
        if not head.startswith(b"ply"):
            raise ValueError("not a PLY file")
        count = re.search(rb"element vertex (\d+)", head)
        return "PLY, %s vertices claimed" % (count.group(1).decode() if count else "?")
    if kind == "crop":
        config = json.loads(path.read_text())
        return "crop json, polygon with %d points, axis %s" % (
            len(config["bounding_polygon"]), config.get("orthogonal_axis", "?"))
    matrix = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("expected a 4x4 matrix, found %d rows" % len(matrix))
    values = [[float(cell) for cell in row] for row in matrix]
    determinant = (values[0][0] * (values[1][1] * values[2][2] - values[1][2] * values[2][1])
                   - values[0][1] * (values[1][0] * values[2][2] - values[1][2] * values[2][0])
                   + values[0][2] * (values[1][0] * values[2][1] - values[1][1] * values[2][0]))
    return "4x4 similarity, scale %.4f" % abs(determinant) ** (1.0 / 3.0)


def fetch(scene, role, waits):
    """Fetch one asset, sleeping between attempts while Drive answers "Quota exceeded"."""
    file_id, resource_key = ASSETS[scene][role]
    destination = DEST / FILENAMES[(scene, role)]
    for attempt in range(1, len(waits) + 2):
        try:
            url, shipped = resolve(file_id, resource_key)
            size = download(url, destination)
            note = verify(role, destination)
            print("%-11s %-9s ok   %8d B  %s  (%s)" % (
                scene, role, size, destination.name, note))
            return True
        except QuotaExceeded:
            if attempt > len(waits):
                print("%-11s %-9s quota exceeded, giving up after %d attempts" % (
                    scene, role, attempt))
                return False
            wait = waits[attempt - 1]
            print("%-11s %-9s quota exceeded, retry %d in %d s" % (scene, role, attempt, wait))
            time.sleep(wait)
        except (LoginRequired, urllib.error.HTTPError, ValueError, OSError) as error:
            print("%-11s %-9s failed: %s: %s" % (scene, role, type(error).__name__, error))
            return False
    return False


def check(scene, role):
    file_id, resource_key = ASSETS[scene][role]
    try:
        url, shipped = resolve(file_id, resource_key)
        head = probe(url)
    except (QuotaExceeded, LoginRequired, urllib.error.HTTPError) as error:
        print("%-11s %-9s not readable now: %s: %s" % (scene, role, type(error).__name__, error))
        return False
    print("%-11s %-9s readable, shipped as %s, first bytes %r" % (
        scene, role, shipped or "unknown", head[:4]))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", choices=sorted(ASSETS),
                        help="scene to fetch (default: all)")
    parser.add_argument("--role", action="append", choices=("cloud", "transform", "crop"),
                        help="asset to fetch (default: all three)")
    parser.add_argument("--check", action="store_true", help="only probe readability, download nothing")
    parser.add_argument("--retry", type=int, default=0, help="extra attempts per asset on quota")
    parser.add_argument("--wait", type=int, default=600, help="seconds between quota retries")
    args = parser.parse_args()

    scenes = args.scene or sorted(ASSETS)
    roles = args.role or ["cloud", "transform", "crop"]
    waits = [args.wait * (index + 1) for index in range(args.retry)]
    ok = True
    for scene in scenes:
        for role in roles:
            ok = (check(scene, role) if args.check else fetch(scene, role, waits)) and ok
    print("all readable" if ok and args.check else ("all downloaded" if ok else "incomplete"),
          "->", DEST)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
