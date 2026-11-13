"""Final P3 software contracts. Synthetic backward only; no optimizer updates."""

from dataclasses import asdict

import pytest
import torch

from supermimic.hoi.wm_trainer import JointWorldModelTrainer
from supermimic.hoi.world_model import JointWorldModel, JointWorldModelCfg, LossWeights
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.wm_trainer import WorldModelTrainer


def fixture():
    torch.manual_seed(903)
    model = JointWorldModel(
        JointWorldModelCfg(hidden=(16, 16), weights=LossWeights(0.7, 1.3)), AnalyticHeads, "cpu"
    )
    state = torch.randn(9, 3, 84) * 0.05
    for offset in (3, 74):
        state[..., offset] = 1
        state[..., offset : offset + 4] /= state[..., offset : offset + 4].norm(
            dim=-1, keepdim=True
        )
    actions = torch.randn(8, 3, 29) * 0.1
    xi = torch.zeros(3, 33)
    xi[:, 0] = 1
    refs = [ReferenceFrame(None, None, None, None, s[:, :3], s[:, 3:7]) for s in state[1:]]
    return model, state, actions, xi, refs


def test_final_dimensions_prior_and_no_auxiliary_modules():
    wm, states, action, xi, _ = fixture()
    state = states[0]
    predicted = wm.predict(state, action[0], xi)
    assert predicted.shape == (3, 84)
    assert wm.spec.canonical(state).shape == (3, 86)
    assert wm.support(state).shape == (3, 18)
    assert wm.dynamics.mlp.net[0].in_features == 166
    assert wm.dynamics.mlp.net[-1].out_features == 82
    torch.testing.assert_close(
        predicted, wm.spec.integrate(state, torch.zeros(3, 82)), atol=0, rtol=0
    )
    assert set(asdict(wm.cfg.weights)) == {"robot", "object"}
    assert not any("contact" in name or "force" in name for name in wm.state_dict())
    assert JointWorldModelTrainer.update is WorldModelTrainer.update
    with pytest.raises(ValueError, match="state84"):
        wm.step(torch.cat((state, torch.zeros(3, 1)), -1), action[0], xi)
    with pytest.raises(ValueError, match="aligned"):
        wm.step(state, action[0, :, :28], xi)
    with pytest.raises(ValueError, match="Nonfinite"):
        wm.step(state, action[0] * float("nan"), xi)


def test_eight_step_physical_loss_backward_and_termination_separation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Optimizer updates are outside P3 cleanup")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    wm, states, actions, xi, refs = fixture()
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(0, 0.01)
    actions.requires_grad_()
    initial = states[0].clone().requires_grad_()
    logged = torch.cat((initial[None], states[1:]), 0)
    done = torch.zeros(8, 3)
    loss, metrics = wm.window_loss(logged, actions, xi, done, refs)
    assert not any("contact" in key or "force" in key for key in metrics)
    assert {"robot_weighted_loss", "object_weighted_loss", "term_bce"} <= metrics.keys()
    torch.testing.assert_close(
        loss.detach(),
        metrics["weighted_physical_loss"] + wm.cfg.term_loss_weight * metrics["term_bce"],
    )
    loss.backward()
    for grad in (initial.grad, actions.grad, wm.dynamics.mlp.net[-1].weight.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert wm.dynamics.mlp.net[-1].weight.grad[:70].abs().sum() > 0
    assert wm.dynamics.mlp.net[-1].weight.grad[70:].abs().sum() > 0
    # Same physical prediction, independent termination target.
    _, components0, term0, _ = wm._loss_step(
        states[0], actions[0], xi, states[1], done[0], refs[0].root_pos_w, refs[0].root_quat_w
    )
    _, components1, term1, _ = wm._loss_step(
        states[0], actions[0], xi, states[1], torch.ones(3), refs[0].root_pos_w, refs[0].root_quat_w
    )
    assert components0.shape == (3, 82)
    torch.testing.assert_close(components0, components1, atol=0, rtol=0)
    assert not torch.equal(term0, term1)
    with pytest.raises(ValueError, match="physical targets only"):
        wm.window_loss(
            states, actions, xi, done, refs, auxiliary_targets={"contact_intent_next": done}
        )
    with pytest.raises(ValueError, match="Misaligned"):
        wm.window_loss(torch.cat((states, torch.zeros(9, 3, 1)), -1), actions, xi, done, refs)


@pytest.mark.parametrize("values", [(-1, 1), (0, 0), (float("nan"), 1), (1, float("inf"))])
def test_invalid_physical_loss_weights(values):
    with pytest.raises(ValueError):
        LossWeights(*values)


def test_support_rejects_modified_manifest_and_missing_or_changed_runtime_source(
    monkeypatch, tmp_path
):
    import json

    import supermimic.hoi.support_context as module
    from supermimic.hoi import provenance
    from supermimic.hoi.reference import DEFAULT_ASSET

    bad = tmp_path / "modified_manifest.json"
    bad.write_bytes(DEFAULT_ASSET.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="exact accepted"):
        module.SupportContext(bad)
    sources = json.loads(DEFAULT_ASSET.read_text())["sources"]
    chosen = next(x["path"] for x in sources if "IMPLEMENTATION_PLAN_V3" not in x["path"])
    original = provenance.sha256

    def wrong_source(path):
        return "wrong" if str(path) == chosen else original(path)

    monkeypatch.setattr(provenance, "sha256", wrong_source)
    with pytest.raises(ValueError, match="source changed"):
        module.SupportContext()

    def missing_source(path):
        if str(path) == chosen:
            raise FileNotFoundError(chosen)
        return original(path)

    monkeypatch.setattr(provenance, "sha256", missing_source)
    with pytest.raises(FileNotFoundError):
        module.SupportContext()
