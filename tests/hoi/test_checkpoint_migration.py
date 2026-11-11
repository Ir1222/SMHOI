"""Named-column migration tests; no Actor/optimizer or integrated-resume claims."""

import copy
import json
from pathlib import Path

import pytest
import torch

from supermimic.hoi.checkpoint import (
    input_columns,
    joint_checkpoint,
    load_joint_weights,
    migrate_robot_state_dict,
)
from supermimic.hoi.force import ForceStatistics
from supermimic.hoi.world_model import JointWorldModel, JointWorldModelCfg, LossWeights
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def models():
    torch.manual_seed(30)
    stats = ForceStatistics(
        [2.0, 3.0], [[80, 20], [90, 10]], [1.0, 2.0], force_on_N=0.001, active_mass=0.5
    )
    cfg = WorldModelCfg(hidden=(32, 32, 32))
    old = WorldModel(cfg, AnalyticHeads, "cpu")
    with torch.no_grad():
        old.dynamics.mlp.net[-1].weight.normal_(0, 0.01)
        old.dynamics.mlp.net[-1].bias.normal_(0, 0.01)
    new = JointWorldModel(
        JointWorldModelCfg(hidden=cfg.hidden, weights=LossWeights(1.0, 1.0, 1.0)),
        AnalyticHeads,
        stats,
        "cpu",
    )
    return old, new


def test_named_action_context_columns_and_nonzero_robot_parity():
    old, new = models()
    mapping = migrate_robot_state_dict(old.state_dict(), old.cfg, new)
    assert mapping["input_mapping"]["action.executed"] == {
        "source": [71, 100],
        "destination": [104, 133],
    }
    assert mapping["input_mapping"]["xi.friction"] == {
        "source": [100, 101],
        "destination": [133, 134],
    }
    state = torch.randn(12, 84) * 0.1
    state[:, 3] = 1
    state[:, 74] = 1
    state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
    state[:, 74:78] /= state[:, 74:78].norm(dim=-1, keepdim=True)
    action = torch.randn(12, 29)
    xi = torch.randn(12, 33) * 0.001
    xi[:, 0] = 1
    old_pred = old.step(state[:, :71], action, xi)
    new_pred, force = new.predict(state, action, xi)
    torch.testing.assert_close(old_pred, new_pred[:, :71], atol=2e-6, rtol=2e-6)
    assert (old_pred - state[:, :71]).abs().max() > 0.001
    assert force.count_nonzero() == 0
    torch.testing.assert_close(
        new_pred[:, 71:], new.spec.integrate(state, torch.zeros(12, 82))[:, 71:]
    )
    columns = input_columns(joint=True, support=True)
    for name, sl in columns.items():
        if name.startswith(("object.", "support")):
            assert new.dynamics.mlp.net[0].weight[:, sl].count_nonzero() == 0
    bad = copy.deepcopy(old.cfg)
    bad.state_spec.in_scale = (2.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    with pytest.raises(ValueError, match="normalization"):
        migrate_robot_state_dict(old.state_dict(), bad, new)
    extra = dict(old.state_dict())
    extra["unexpected"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        migrate_robot_state_dict(extra, old.cfg, new)


def test_strict_checkpoint_normalization_metadata_and_roundtrip(tmp_path):
    old, new = models()
    migrate_robot_state_dict(old.state_dict(), old.cfg, new)
    root = Path(__file__).resolve().parents[2]
    meta = json.loads(
        (
            root
            / "outputs/hoi/contact_supervision/force_model_validation/collection_02/metadata.json"
        ).read_text()
    )["replay"]
    payload = joint_checkpoint(
        new, replay_metadata=meta, readiness={"ready": False}, provenance={"kind": "unit_test"}
    )
    path = tmp_path / "new_weights.pt"
    torch.save(payload, path)
    loaded = torch.load(path, weights_only=True)
    other = models()[1]
    load_joint_weights(other, loaded, expected_replay_metadata=meta)
    for key, value in new.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], atol=0, rtol=0)
    changed = copy.deepcopy(loaded)
    changed["metadata"]["replay"]["force_sign"] = "box_on_hand"
    with pytest.raises(ValueError, match="replay"):
        load_joint_weights(other, changed, expected_replay_metadata=meta)
    changed = copy.deepcopy(loaded)
    changed["wm"]["force_statistics.scale_N"] *= 2
    with pytest.raises(ValueError, match="normalization"):
        load_joint_weights(other, changed, expected_replay_metadata=meta)
    changed = copy.deepcopy(loaded)
    del changed["wm"]["dynamics.mlp.net.2.bias"]
    with pytest.raises(ValueError, match="keys"):
        load_joint_weights(other, changed, expected_replay_metadata=meta)
