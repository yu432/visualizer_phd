"""Analytic checks for back-projection, sim(3) alignment and GT metrics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from align import align_clouds, apply_sim3, umeyama
from common import backproject, diagonal, voxel_downsample, voxel_downsample_focused
from metrics import cloud_metrics, depth_metrics


def _euler_zyx(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def test_backproject_roundtrip():
    depth = np.full((1, 4, 4), 2.0, np.float32)
    k = np.eye(3)
    k[0, 0] = k[1, 1] = 2.0
    k[0, 2] = k[1, 2] = 2.0
    points = backproject(depth, k[None], np.eye(4)[None])
    assert points.shape == (16, 3)
    xs = np.sort(np.unique(np.round(points[:, 0], 6)))
    assert np.allclose(xs, [-2.0, -1.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(points[:, 2], 2.0)


def test_backproject_drops_invalid_depth():
    depth = np.full((1, 2, 2), np.nan, np.float32)
    depth[0, 0, 0] = 1.0
    k = np.eye(3)
    k[0, 0] = k[1, 1] = 1.0
    assert len(backproject(depth, k[None], np.eye(4)[None])) == 1


def test_recovers_known_sim3():
    """A helix has a shape-determined scale, so NN correspondences stay meaningful."""
    rng = np.random.default_rng(0)
    angle = np.linspace(0, 6 * np.pi, 3000)
    gt = np.stack([np.cos(angle), np.sin(angle), 0.12 * angle], axis=-1)
    gt = gt + rng.normal(scale=0.01, size=gt.shape)
    scale, rotation, translation = 1.6, _euler_zyx(0.3, -0.2, 0.1), np.array([1.0, -0.5, 0.2])
    pred = scale * gt @ rotation.T + translation
    result = align_clouds(pred, gt)
    assert result["status"] == "ok"
    assert abs(result["s"] - 1.0 / scale) / (1.0 / scale) < 0.05
    assert result["rmse_after"] < 0.05 * diagonal(gt)
    assert result["rmse_after"] < result["rmse_before"]


def test_align_converges_on_self_similar_cloud():
    """A filled box is scale-ambiguous by nature; the alignment still has to converge."""
    rng = np.random.default_rng(5)
    grid = np.stack(np.meshgrid(np.linspace(0, 2, 25), np.linspace(0, 1, 12), np.linspace(0, 1, 10)), -1)
    gt = grid.reshape(-1, 3) + rng.normal(scale=0.004, size=(grid.size // 3, 3))
    pred = 1.6 * gt @ _euler_zyx(0.3, -0.2, 0.1).T + np.array([1.0, -0.5, 0.2])
    result = align_clouds(pred, gt)
    assert result["status"] == "ok"
    assert result["rmse_after"] < 0.03 * diagonal(gt)


def test_umeyama_identity():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(100, 3))
    scale, rotation, translation = umeyama(points, points)
    assert abs(scale - 1.0) < 1e-9
    assert np.allclose(rotation, np.eye(3), atol=1e-9)
    assert np.allclose(translation, 0.0, atol=1e-9)


def test_identical_clouds_perfect():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(500, 3))
    result = cloud_metrics(points, points, diag=diagonal(points), taus=(0.005, 0.01))
    assert result["fscore@0.5"] == 1.0
    assert result["fscore@1"] == 1.0
    assert result["accuracy"] == 0.0


def test_depth_metrics_need_per_frame_scale_shift():
    rng = np.random.default_rng(3)
    gt = rng.uniform(1.0, 5.0, (1, 8, 8)).astype(np.float32)
    shifted = gt * 3.0 + 1.0
    assert depth_metrics(shifted, gt, per_frame_align=True)["AbsRel"] < 1e-6
    assert depth_metrics(shifted, gt, per_frame_align=False)["AbsRel"] > 0.5


def test_voxel_downsample_is_deterministic_and_reduces():
    rng = np.random.default_rng(4)
    points = rng.uniform(0, 1, (5000, 3))
    first = voxel_downsample(points, 0.1)
    second = voxel_downsample(points, 0.1)
    assert np.allclose(first, second)
    assert 100 < len(first) < len(points)


def test_focused_voxel_keeps_the_hero_and_the_rest():
    """A fine grid inside the focus sphere, the coarse one outside, and nothing duplicated."""
    rng = np.random.default_rng(5)
    hero = rng.uniform(-0.05, 0.05, (6000, 3))
    ground = rng.uniform(-1.0, 1.0, (6000, 3))
    ground[:, 1] = 0.0
    points = np.concatenate([hero, ground])
    plain = voxel_downsample(points, 0.05)
    focused, _ = voxel_downsample_focused(points, 0.05, None, np.zeros(3), 0.2, 0.01)
    near = lambda cloud: int((np.linalg.norm(cloud, axis=1) <= 0.2).sum())  # noqa: E731
    far = lambda cloud: int((np.linalg.norm(cloud, axis=1) > 0.2).sum())  # noqa: E731
    assert near(focused) > 4 * near(plain)
    # Cells straddling the sphere average to slightly different positions, so a handful of points
    # move across the radius; the coarse part stays the same to within that boundary effect.
    assert abs(far(focused) - far(plain)) < 0.05 * far(plain)
    assert len(focused) < len(points)


def test_focused_voxel_without_a_focus_is_plain_voxel():
    rng = np.random.default_rng(6)
    points = rng.uniform(0, 1, (2000, 3))
    plain = voxel_downsample(points, 0.1)
    focused = voxel_downsample_focused(points, 0.1, None, None, 0.0, 0.0)[0]
    assert np.allclose(plain, focused)


def test_align_rejects_degenerate_input():
    assert align_clouds(np.zeros((10, 3)), np.zeros((10, 3)))["status"] == "failed"

def test_render_sample_keeps_density_not_the_point_count():
    """Tiles of the 16- and 100-frame schemes must be drawn at the same density.

    Sampling to a fixed 320 000 points paints the same facade with fewer dots on the denser cloud,
    so a four times denser cloud of the same object has to come out with about the same number of
    drawn points, not four times more.
    """
    from figures import render_sample
    fine = np.stack([np.arange(4000) * 0.00025, np.zeros(4000), np.zeros(4000)], axis=1)
    coarse = np.stack([np.arange(1000) * 0.001, np.zeros(1000), np.zeros(1000)], axis=1)
    # Below the cap the cloud is drawn as it is: a sparse cloud must not be thinned further.
    untouched, _ = render_sample(coarse, None, 1.0, limit=5000, columns=100)
    assert len(untouched) == 1000
    # Above it the grid decides, and both clouds land on the same density: 100 cells of 0.01.
    drawn_fine, _ = render_sample(fine, None, 1.0, limit=300, columns=100)
    drawn_coarse, _ = render_sample(coarse, None, 1.0, limit=300, columns=100)
    assert 90 <= len(drawn_fine) <= 110
    assert 90 <= len(drawn_coarse) <= 110


def occupied(points, voxel=0.05):
    keys = np.floor(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    return np.unique((keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]).size


def test_stratified_subsample_keeps_every_patch():
    """A budget cut must thin the crowded patch, not delete the thin one.

    The dense scheme stacks a hundred frames on the same wall and a couple of them on the far side
    of the barn. A random cut keeps the same fraction of both, so the far side loses most of its
    dots and reads as holes; capping the density keeps the thin patch whole and takes the budget
    out of the stacked points instead.
    """
    from common import stratified_subsample
    rng = np.random.default_rng(0)
    stacked = rng.normal(size=(180000, 3)) * 0.02          # the wall every frame saw
    thin = rng.normal(size=(3000, 3)) * 0.02 + [2.0, 0, 0]  # the far side, few frames
    points = np.concatenate([stacked, thin])
    random_cut = points[rng.choice(len(points), size=30000, replace=False)]
    kept, _ = stratified_subsample(points, None, 30000, seed=0)
    assert len(kept) == 30000
    # The far side survives whole under the density cap, and mostly disappears under a random cut.
    assert occupied(kept[kept[:, 0] > 1.0], 0.05) >= 0.95 * occupied(thin, 0.05)
    assert occupied(random_cut[random_cut[:, 0] > 1.0], 0.05) < 0.5 * occupied(thin, 0.05)
    # And the budget really went into the thin patch rather than being spent at random.
    assert np.count_nonzero(kept[:, 0] > 1.0) > 5 * np.count_nonzero(random_cut[:, 0] > 1.0)


def test_stratified_subsample_is_a_no_op_below_the_budget():
    from common import stratified_subsample
    points = np.zeros((100, 3))
    kept, _ = stratified_subsample(points, None, 1000)
    assert len(kept) == 100


def test_translation_error_in_centimetres():
    """The pose error is stored as a fraction of the GT camera diagonal; the site prints it in cm.

    A scene whose GT cameras span 10 m diagonally and a 2 % mean centre offset is 20 cm, whatever
    the scene's own units look like in the report tables (2 % of the diagonal).
    """
    from summarize import translation_cm
    assert translation_cm(0.02, 10.0) == pytest.approx(20.0)
    assert translation_cm(0.0032, 9.744) == pytest.approx(3.118, abs=1e-3)
    assert translation_cm(None, 10.0) is None
    assert translation_cm(0.02, None) is None


def test_drive_url_carries_resource_key():
    """Without the resource key Drive answers with a sign-in page, so the key has to be in the URL."""
    from fetch_tnt_gt_drive import drive_url
    assert drive_url("abc", "0-key") == "https://drive.google.com/uc?export=download&id=abc&resourcekey=0-key"
    assert "resourcekey" not in drive_url("abc")


def test_drive_confirm_form_is_parsed():
    """Drive's "virus scan warning" page hides the real transfer in a GET form."""
    from fetch_tnt_gt_drive import bandwidth_note, form_params
    page = (
        '<html><body><p>Google Drive can not scan this file for viruses.</p>'
        '<p><b>Ignatius.ply</b> (129M) is too large for Google to scan for viruses.</p>'
        '<form id="download-form" action="https://drive.usercontent.google.com/download" method="get">'
        '<input type="submit" value="Download anyway"/>'
        '<input type="hidden" name="id" value="0B-file"/>'
        '<input type="hidden" name="export" value="download"/>'
        '<input type="hidden" name="confirm" value="t"/>'
        "</form></body></html>"
    )
    action, fields = form_params(page)
    assert action == "https://drive.usercontent.google.com/download"
    assert fields == {"id": "0B-file", "export": "download", "confirm": "t"}
    assert bandwidth_note(page) == "Ignatius.ply (129M)"


def test_fetcher_verify_rejects_wrong_payload(tmp_path):
    """A quota page saved as .ply must not pass as a laser scan."""
    from fetch_tnt_gt_drive import verify
    good = tmp_path / "Scene.ply"
    good.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 42\n")
    assert "42" in verify("cloud", good)
    bad = tmp_path / "Quota.ply"
    bad.write_bytes(b"<!DOCTYPE html><title>Google Drive - Quota exceeded</title>")
    with pytest.raises(ValueError):
        verify("cloud", bad)
    crop = tmp_path / "Scene.json"
    crop.write_text('{"bounding_polygon": [[0, 0], [1, 0], [1, 1]], "orthogonal_axis": "Z", '
                    '"axis_min": -1.0, "axis_max": 1.0}')
    assert "3 points" in verify("crop", crop)
    transform = tmp_path / "Scene_trans.txt"
    transform.write_text("0.5 0 0 1\n0 0.5 0 2\n0 0 0.5 3\n0 0 0 1\n")
    assert "scale 0.5000" in verify("transform", transform)
