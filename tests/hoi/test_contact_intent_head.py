"""Deterministic contract tests, not physical or learned-performance evidence."""

import pytest
import torch
from torch.nn import functional as F

from supermimic.hoi.wm_trainer import JointWorldModelTrainer
from supermimic.hoi.world_model import (
    JointWorldModel,
    JointWorldModelCfg,
    LossWeights,
    augment_states,
)
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.wm_trainer import WorldModelTrainer


def fixture():
    torch.manual_seed(401)
    wm = JointWorldModel(
        JointWorldModelCfg(hidden=(16, 16), weights=LossWeights(1, 1, 1)), AnalyticHeads, "cpu"
    )
    state = torch.zeros(3, 85)
    state[:, 3] = state[:, 74] = 1
    state[:, 2] = 1
    state[:, 71:74] = torch.tensor([0.2, 0.1, 1.1])
    xi = torch.zeros(3, 33)
    xi[:, 0] = 1
    return wm, state, torch.randn(3, 29), xi


def test_dimensions_and_contact_output_excluded_from_physical_integration():
    wm, state, action, xi = fixture()
    predicted, logit = wm.predict(state, action, xi)
    assert predicted.shape == (3, 85) and logit.shape == (3, 1)
    assert wm.dynamics.mlp.net[0].in_features == 167
    assert wm.dynamics.mlp.net[-1].out_features == 83
    torch.testing.assert_close(
        predicted[:, :84], wm.spec.integrate(state[:, :84], torch.zeros(3, 82))
    )
    assert torch.equal(predicted[:, 84], torch.full((3,), 0.5))
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].bias[82] = 1.25
    changed, _ = wm.predict(state, action, xi)
    torch.testing.assert_close(changed[:, :84], predicted[:, :84], atol=0, rtol=0)
    assert (changed[:, 84] > 0.5).all()
    with pytest.raises(ValueError, match="physical84"):
        wm.predict(state[:, :84], action, xi)
    assert not any("force" in k for k in wm.state_dict())
    assert JointWorldModelTrainer.update is WorldModelTrainer.update


@pytest.mark.parametrize("logit_value", [-100.0, -0.73, 0.0, 100.0])
def test_contact_bce_matches_pytorch_with_extreme_logits(logit_value):
    wm, state, action, xi = fixture()
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].bias[82] = logit_value
    target = torch.tensor([[0.0], [1.0], [0.0]])
    _, components, _, _ = wm._loss_step(
        state, action, xi, state, torch.zeros(3), state[:, :3], state[:, 3:7], target
    )
    expected = F.binary_cross_entropy_with_logits(
        torch.full_like(target, logit_value), target, reduction="none"
    )
    torch.testing.assert_close(components[:, 82:83], expected, atol=0, rtol=0)
    components.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None)


def test_eight_step_fractional_feedback_and_future_labels_are_targets_only():
    wm, state, action, xi = fixture()
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(0, 0.001)
        wm.dynamics.mlp.net[-1].bias[82] = -0.8
    logged = state.expand(9, -1, -1).clone()
    logged[1:, :, 84] = torch.arange(8)[:, None] % 2
    logged[1:, :, 2] += 0.01
    logged[1:, :, 71] += 0.01
    ref = ReferenceFrame(None, None, None, None, state[:, :3], state[:, 3:7])
    inputs, outputs = [], []
    before = wm.dynamics.register_forward_pre_hook(lambda _, args: inputs.append(args[0][:, 86]))
    after = wm.dynamics.register_forward_hook(lambda _, args, out: outputs.append(out[:, 82]))
    loss, metrics = wm.window_loss(
        logged,
        action.expand(8, -1, -1),
        xi,
        torch.zeros(8, 3),
        [ref] * 8,
        auxiliary_targets={"contact_intent_next": logged[1:, :, 84:85]},
    )
    before.remove()
    after.remove()
    assert len(inputs) == 8
    for j in range(1, 8):
        torch.testing.assert_close(inputs[j], outputs[j - 1].sigmoid(), atol=0, rtol=0)
        assert inputs[j].requires_grad and ((inputs[j] > 0) & (inputs[j] < 1)).all()
        assert not torch.equal(inputs[j], logged[j, :, 84])
    expected = F.binary_cross_entropy_with_logits(torch.stack(outputs), logged[1:, :, 84])
    torch.testing.assert_close(metrics["contact_bce"], expected)
    loss.backward()
    last = wm.dynamics.mlp.net[-1]
    for rows in (slice(0, 70), slice(70, 82), slice(82, 83)):
        assert last.weight.grad[rows].abs().sum() > 0
    assert torch.isfinite(wm.rollout_error(logged, action.expand(8, -1, -1), xi)).all()


def test_pair_alignment_and_last_endpoint_not_replay_padding():
    _, state, _, _ = fixture()
    physical = state[:, :84].expand(9, -1, -1).clone()
    bits = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 1.0])[:, None].expand(-1, 3)
    pairs = torch.stack((bits[:-1], bits[1:]), -1)
    z = augment_states(physical, pairs)
    assert z.shape == (9, 3, 85) and torch.equal(z[..., 84], bits)
    invalid = pairs.clone()
    invalid[0, 0, 0] = 99
    with pytest.raises(ValueError, match="binary"):
        augment_states(physical, invalid)
    invalid = pairs.clone()
    invalid[1, 0, 0] = 1
    with pytest.raises(ValueError, match="Discontinuous"):
        augment_states(physical, invalid)
