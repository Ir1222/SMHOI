"""Force unit/loss tests, with measured P2 distributions distinguished from fixtures."""

from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.hoi.force import ForceStatistics

ROOT = Path(__file__).resolve().parents[2]


def measured_stats():
    with np.load(
        ROOT / "outputs/hoi/contact_supervision/replay_diagnostics/run_03/transitions.npz"
    ) as d:
        force = torch.tensor(d["force_mean_box"])
        occupancy = torch.tensor(d["occupancy"])
    # Normalization test only. No model fit on incomplete old archive (missing xi).
    return (
        ForceStatistics.fit(force, occupancy, force_on_N=0.001, active_mass=0.5),
        force,
        occupancy,
    )


def test_reversible_physical_units_and_measured_class_balance():
    stats, force, occupancy = measured_stats()
    torch.testing.assert_close(stats.decode(stats.encode(force)), force, atol=2e-5, rtol=2e-6)
    assert stats.counts_inactive_active.tolist() == [[86, 10], [88, 8]]
    assert torch.equal(stats.decode(torch.zeros_like(force)), torch.zeros_like(force))
    for hand in range(2):
        count = stats.counts_inactive_active[hand]
        torch.testing.assert_close(
            stats.class_weights[hand] * count / count.sum(), torch.full((2,), 0.5)
        )
    # Duplicate all fully inactive examples; refit distribution and preserve the
    # balanced group objective rather than allowing zero samples to dominate.
    zeros = (occupancy == 0).all(-1)
    expanded_f = torch.cat((force, force[zeros].repeat(4, 1, 1)))
    expanded_o = torch.cat((occupancy, occupancy[zeros].repeat(4, 1)))
    expanded = ForceStatistics.fit(expanded_f, expanded_o, force_on_N=0.001, active_mass=0.5)
    for stat, values, occ in [(stats, force, occupancy), (expanded, expanded_f, expanded_o)]:
        assert torch.isfinite(
            stat.loss_components(torch.ones_like(values), values, occ, kind="huber", delta=1)
        ).all()
    # Constant residual in each class gives exactly the same balanced loss.
    losses = []
    for stat, values, occ in [(stats, force, occupancy), (expanded, expanded_f, expanded_o)]:
        predicted = stat.encode(values) + torch.where(occ[..., None] > 0, 2.0, 0.2)
        losses.append(
            stat.loss_components(predicted, values, occ, kind="huber", delta=1).sum((-1, -2)).mean()
        )
    torch.testing.assert_close(*losses)
    with pytest.raises(ValueError):
        ForceStatistics.fit(
            torch.zeros(4, 2, 3), torch.zeros(4, 2), force_on_N=0.001, active_mass=0.5
        )


def test_huber_rejects_invalid_and_has_bounded_outlier_gradient():
    stats, force, occupancy = measured_stats()
    prediction = torch.full_like(force, 100.0, requires_grad=True)
    loss = stats.loss_components(prediction, force, occupancy, kind="huber", delta=1.0)
    loss.sum().backward()
    assert prediction.grad.abs().max() <= stats.class_weights.max() / 6 + 1e-6
    with pytest.raises(ValueError):
        stats.loss_components(prediction, force, occupancy, kind="huber", delta=0.0)
