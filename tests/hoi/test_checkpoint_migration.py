"""Strict named migration; deterministic probes never claim new learned performance."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.hoi.checkpoint import (
    input_columns,
    joint_checkpoint,
    joint_config,
    legacy_joint_step,
    load_joint_weights,
    migrate_joint_weights,
    migrate_robot_state_dict,
)
from supermimic.hoi.world_model import JointWorldModel, JointWorldModelCfg, LossWeights
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/hoi/contact_supervision/force_model_validation"


def metadata():
    return json.loads((OLD / "collection_02/metadata.json").read_text())["replay"]


def legacy_models():
    payload = torch.load(
        OLD / "migration_01/diagnostic_joint_weights.pt", map_location="cpu", weights_only=True
    )
    new = JointWorldModel(
        joint_config(payload["metadata"]["config"], legacy=True), AnalyticHeads, "cpu"
    )
    return payload, new


def robot_models():
    torch.manual_seed(30)
    cfg = WorldModelCfg(hidden=(32, 32, 32))
    old = WorldModel(cfg, AnalyticHeads, "cpu")
    with torch.no_grad():
        old.dynamics.mlp.net[-1].weight.normal_(0, 0.01)
        old.dynamics.mlp.net[-1].bias.normal_(0, 0.01)
    new = JointWorldModel(
        JointWorldModelCfg(hidden=cfg.hidden, weights=LossWeights(1, 1)), AnalyticHeads, "cpu"
    )
    return old, new


def test_robot_named_columns_and_eight_step_prediction_preservation():
    old, new = robot_models()
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
    state[:, 3] = state[:, 74] = 1
    action = torch.randn(12, 29)
    xi = torch.randn(12, 33) * 0.001
    xi[:, 0] = 1
    robot = state[:, :71].clone()
    with torch.no_grad():
        for _ in range(8):
            prior = new.spec.integrate(state[:, :84], torch.zeros(12, 82))
            robot = old.step(robot, action, xi)
            state = new.predict(state, action, xi)
            torch.testing.assert_close(robot, state[:, :71], atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(state[:, 71:84], prior[:, 71:84], atol=0, rtol=0)
    for name, sl in input_columns(joint=True, support=True).items():
        if name.startswith(("object.", "support", "contact")):
            assert new.dynamics.mlp.net[0].weight[:, sl].count_nonzero() == 0
    bad = copy.deepcopy(old.cfg)
    bad.state_spec.in_scale = (2.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    with pytest.raises(ValueError, match="normalization"):
        migrate_robot_state_dict(old.state_dict(), bad, new)
    extra = dict(old.state_dict(), unexpected=torch.zeros(1))
    with pytest.raises(ValueError):
        migrate_robot_state_dict(extra, old.cfg, new)


def test_old_joint_preserves_all_named_columns_and_physical_eight_steps():
    old, new = legacy_models()
    mapping = migrate_joint_weights(old, new, expected_replay_metadata=metadata())
    assert mapping["discarded_output_rows"] == [82, 88]
    first = "dynamics.mlp.net.0.weight"
    for name, span in input_columns(joint=True, support=True).items():
        dst = input_columns(joint=True, support=True)[name]
        torch.testing.assert_close(
            new.state_dict()[first][:, dst], old["wm"][first][:, span], atol=0, rtol=0
        )
    for suffix in ("weight", "bias"):
        key = f"dynamics.mlp.net.6.{suffix}"
        torch.testing.assert_close(new.state_dict()[key][:82], old["wm"][key][:82], atol=0, rtol=0)
        assert new.state_dict()[key].shape[0] == 82
    with np.load(OLD / "collection_02/transitions.npz") as data:
        physical = torch.tensor(data["state"][:4])
        actions = torch.tensor(data["action_exec"][:32]).reshape(8, 4, 29)
        xi = torch.tensor(data["xi"][:4])
    state = physical.clone()
    with torch.no_grad():
        for action in actions:
            physical = legacy_joint_step(old, new, physical, action, xi)
            state = new.predict(state, action, xi)
            torch.testing.assert_close(physical, state[:, :84], atol=2e-6, rtol=2e-6)


def test_strict_new_checkpoint_roundtrip_and_metadata(tmp_path):
    old, new = legacy_models()
    migrate_joint_weights(old, new, expected_replay_metadata=metadata())
    payload = joint_checkpoint(
        new,
        replay_metadata=metadata(),
        provenance={"kind": "unit_test"},
    )
    assert payload["metadata"]["schema"] == "hoi-joint-physical-wm-v3"
    assert "force_normalization" not in payload["metadata"]
    assert not any(k.startswith("force_statistics.") for k in payload["wm"])
    path = tmp_path / "new.pt"
    torch.save(payload, path)
    restored = JointWorldModel(joint_config(payload["metadata"]["config"]), AnalyticHeads, "cpu")
    load_joint_weights(
        restored, torch.load(path, weights_only=True), expected_replay_metadata=metadata()
    )
    for key, value in new.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    for section, key, value in [
        ("metadata", "schema", "hoi-joint-wm-v1"),
        ("metadata", "dimensions", {"input": 167, "output": 83}),
        ("wm", "unexpected", torch.zeros(1)),
        ("wm", "dynamics.mlp.net.0.weight", torch.zeros(64, 167)),
        ("wm", "dynamics.mlp.net.0.bias", torch.full((64,), float("nan"))),
        ("wm", "support.positions", payload["wm"]["support.positions"] + 1),
    ]:
        bad = copy.deepcopy(payload)
        bad[section][key] = value
        with pytest.raises(ValueError):
            load_joint_weights(restored, bad, expected_replay_metadata=metadata())
    bad = copy.deepcopy(payload)
    del bad["wm"]["dynamics.mlp.net.2.bias"]
    with pytest.raises(ValueError, match="keys"):
        load_joint_weights(restored, bad, expected_replay_metadata=metadata())
    wrong_meta = copy.deepcopy(metadata())
    wrong_meta["references"][0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="replay"):
        load_joint_weights(restored, payload, expected_replay_metadata=wrong_meta)


@pytest.mark.parametrize("damage", ["missing", "extra", "schema", "config", "stats", "nonfinite"])
def test_legacy_joint_rejects_incompatible_or_incomplete_source(damage):
    old, new = legacy_models()
    if damage == "missing":
        del old["wm"]["dynamics.mlp.net.4.bias"]
    elif damage == "extra":
        old["wm"]["unknown"] = torch.zeros(1)
    elif damage == "schema":
        old["metadata"]["schema"] = "unknown"
    elif damage == "config":
        old["metadata"]["config"]["state_spec"]["dt"] = 0.01
    elif damage == "stats":
        old["wm"]["force_statistics.class_weights"] *= 2
    else:
        old["wm"]["dynamics.mlp.net.6.weight"][70, 0] = float("nan")
    with pytest.raises(ValueError):
        migrate_joint_weights(old, new, expected_replay_metadata=metadata())


def test_transitional_and_raw_fit_checkpoints_are_rejected():
    _, wm = legacy_models()
    transitional = torch.load(
        ROOT
        / "outputs/hoi/contact_supervision/contact_intent_model_validation/migration_02/migrated_contact_weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    with pytest.raises(ValueError, match="transitional"):
        load_joint_weights(wm, transitional, expected_replay_metadata=metadata())
    with pytest.raises(ValueError, match="legacy schema"):
        migrate_joint_weights(transitional, wm, expected_replay_metadata=metadata())
    raw = torch.load(OLD / "fit_01/diagnostic_weights.pt", map_location="cpu", weights_only=True)
    with pytest.raises(ValueError, match="payload"):
        migrate_joint_weights(raw, wm, expected_replay_metadata=metadata())


def test_named_robot_sentinel_columns_and_final_metadata():
    old, new = robot_models()
    with torch.no_grad():
        w = old.dynamics.mlp.net[0].weight
        w.copy_(torch.arange(w.shape[1]).expand_as(w))
    migrate_robot_state_dict(old.state_dict(), old.cfg, new)
    source, destination = input_columns(joint=False), input_columns(joint=True, support=True)
    for name, span in source.items():
        torch.testing.assert_close(
            new.dynamics.mlp.net[0].weight[:, destination[name]],
            old.dynamics.mlp.net[0].weight[:, span],
            atol=0,
            rtol=0,
        )
    payload = joint_checkpoint(new, replay_metadata=metadata(), provenance={"kind": "synthetic"})
    assert payload["metadata"]["dimensions"] == {
        "physical_state": 84,
        "canonical": 86,
        "support": 18,
        "action": 29,
        "xi": 33,
        "input": 166,
        "physical_residual": 82,
        "output": 82,
    }
    assert payload["metadata"]["predictive_validation"] == {"status": "DEFERRED", "stage": "P7-A"}
    for key in ("input", "output", "physical_state", "canonical", "support", "action", "xi"):
        bad = copy.deepcopy(payload)
        bad["metadata"]["dimensions"][key] += 1
        with pytest.raises(ValueError, match="dimensions"):
            load_joint_weights(new, bad, expected_replay_metadata=metadata())
