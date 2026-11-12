"""Exercise the immutable original-reference P4 audit, without physical stepping."""

from pathlib import Path

import torch

from supermimic.hoi.rewards import reference_contact_at


def test_accepted_reference_geometry_and_exact_contact_indices(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/hoi"))
    from audit_proximity_reward import inspect_geometry

    report, rows, clip, _, _ = inspect_geometry()
    assert len(rows) == 840 and report["on_count"] == 290 and report["off_count"] == 550
    assert report["on_frames"] == list(range(260, 550))
    assert report["torch_vs_mujoco_eef_max_m"] < 1e-10
    assert [c["box_geom"] for c in report["target_checks"]] == ["side3", "side4"]
    assert all(c["eef_inside_yaw_capsule_signed_distance_m"] < 0 for c in report["target_checks"])
    assert len(report["native_hand_geoms"]) == 4
    assert report["training_ready"] is False
    assert report["asset_gates"]["physical_hand_contact"]["status"] == "FAIL"
    labels = torch.tensor(clip.arrays["object_contact"].copy())
    assert torch.equal(reference_contact_at(labels, torch.arange(840)), labels)
