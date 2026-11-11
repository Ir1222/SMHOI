import pytest
import torch

from supermimic.hoi.contact import aggregate_interval, torque_at_point


def samples():
    raw = torch.zeros(1, 4, 2, 2, 3, dtype=torch.float64)
    raw[0, :, 0, 0, 0] = torch.tensor([1, 2, 3, 4])
    raw[0, :, 0, 1, 0] = torch.tensor([4, 3, 2, 1])
    raw[0, 0, 1, 0, 1] = 8
    q = torch.zeros(1, 4, 4, dtype=torch.float64)
    angles = torch.tensor([0, 0.5, 1, 1.5], dtype=torch.float64)
    q[0, :, 0], q[0, :, 3] = torch.cos(angles / 2), torch.sin(angles / 2)
    found = torch.ones(1, 4, 2, 2, dtype=torch.float64)
    return raw, q, found, angles


def test_units_both_bodies_contemporaneous_rotation_and_copy():
    raw, q, found, a = samples()
    result = aggregate_interval(raw, q, found, force_on=1e-3)
    expected = torch.tensor([5 * torch.cos(a).mean(), -5 * torch.sin(a).mean(), 0])
    torch.testing.assert_close(result["force_mean_box"][0, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        result["impulse_world"][0, 0], torch.tensor([0.1, 0, 0], dtype=torch.float64)
    )
    assert result["force_peak"].tolist() == [[5, 8]]
    assert result["occupancy"].tolist() == [[1, 0.25]]
    assert result["found_final"].tolist() == [[2, 2]]
    assert not torch.allclose(
        expected, torch.tensor([5 * torch.cos(a[-1]), -5 * torch.sin(a[-1]), 0])
    )
    raw.zero_()
    q.zero_()
    found.zero_()
    assert result["force_raw_world"].abs().sum() > 0
    assert result["found_substeps"].sum() == 16


@pytest.mark.parametrize("active", [[1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0]])
def test_transients_and_cancellation(active):
    raw, q, found, _ = samples()
    q.zero_()
    q[..., 0] = 1
    raw.zero_()
    raw[0, :, 0, 0, 0] = torch.tensor(active) * 4
    out = aggregate_interval(raw, q, found, force_on=1e-3)
    assert out["force_mean_box"][0, 0, 0] == sum(active)
    assert out["occupancy"][0, 0] == sum(active) / 4
    raw[0, :, 0, 1] = -raw[0, :, 0, 0]
    out = aggregate_interval(raw, q, found, force_on=1e-3)
    assert out["force_mean_box"].count_nonzero() == 0
    assert out["occupancy"].count_nonzero() == 0
    assert out["found_final"].min() > 0


def test_common_point_torque():
    force = torch.tensor([[[1.0, 0, 0], [0, 2, 0]]], dtype=torch.float64)
    centroid = torch.tensor([[[0.0, 1, 0], [1, 0, 0]]], dtype=torch.float64)
    torque = torch.zeros_like(force)
    out = torque_at_point(force, torque, centroid, torch.zeros(1, 3, dtype=torch.float64))
    torch.testing.assert_close(out, torch.tensor([[0.0, 0, 1]], dtype=torch.float64))


@pytest.mark.parametrize("bad", ["threshold", "quaternion", "nonfinite", "shape", "found"])
def test_reject_invalid_measurement(bad):
    raw, q, found, _ = samples()
    threshold = 1e-3
    if bad == "threshold":
        threshold = 0
    if bad == "quaternion":
        q.zero_()
    if bad == "nonfinite":
        raw[0, 0, 0, 0, 0] = float("nan")
    if bad == "shape":
        raw = raw[:, :3]
    if bad == "found":
        found[0, 0, 0, 0] = 0.5
    with pytest.raises(ValueError):
        aggregate_interval(raw, q, found, force_on=threshold)
