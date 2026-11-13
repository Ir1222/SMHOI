import copy
from dataclasses import replace

import pytest
import torch
from p7_policy_fixtures import controlled_fusion, save_evidence

from supermimic.hoi.policy import HOIActor, HOICritic
from supermimic.hoi.policy_checkpoint import (
    load_policy_weights,
    migrate_robot_actor,
    migrate_robot_critic,
    policy_checkpoint,
)
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg

SOURCE = {"checkpoint_sha256": "1" * 64, "provenance_status": "synthetic-test"}
IDENTITY = {
    "references": [
        {
            "source_sha256": "f38071e5347403f9eff993dd37884bf374e0c003400156dc4cb87d3d2b856cb0",
            "manifest_sha256": "2" * 64,
            "training_ready": False,
            "role": "experimental",
        }
    ],
    "object_asset_sha256": "3" * 64,
    "support_asset_sha256": "4" * 64,
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_named_migration_exact_preservation_zero_sensitivity_and_roundtrip(dtype, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(49)
    old_actor, old_critic = Policy(PolicyCfg()).to(dtype), Critic(CriticCfg()).to(dtype)
    actor, critic = HOIActor().to(dtype), HOICritic().to(dtype)
    am = migrate_robot_actor(old_actor.state_dict(), old_actor.cfg, actor, provenance=SOURCE)
    cm = migrate_robot_critic(
        old_critic.state_dict(), CriticCfg(), old_actor.cfg, critic, provenance=SOURCE
    )
    obs = torch.randn(2, 1080, dtype=dtype, requires_grad=True)
    goal = torch.randn(2, 830, dtype=dtype, requires_grad=True)
    expected = old_actor(obs[:, :930], goal[:, :680])
    actual = actor(obs, goal)
    atol, rtol = (5e-7, 5e-6) if dtype == torch.float32 else (1e-12, 1e-10)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    grad_o, grad_g = torch.autograd.grad(actual.sum(), (obs, goal), retain_graph=True)
    assert not grad_o[:, 930:].any() and not grad_g[:, 680:].any()
    actual.sum().backward()
    for branch in (actor.object_history_encoder, actor.object_goal_encoder):
        assert all(p.grad is not None and not p.grad.any() for p in branch.parameters())
    feat, xi, root = (
        torch.randn(2, 148, dtype=dtype),
        torch.randn(2, 33, dtype=dtype),
        torch.randn(2, 4, dtype=dtype),
    )
    cv = critic(feat, goal, xi, root)
    ov = old_critic(feat[:, :71], goal[:, :680], xi, root)
    torch.testing.assert_close(cv, ov, atol=atol, rtol=rtol)
    # Exact named slices; a contiguous first-788 copy would fail these checks.
    w, old = critic.mlp.net[0].weight, old_critic.mlp.net[0].weight
    for dst, src in (
        (slice(0, 71), slice(0, 71)),
        (slice(148, 828), slice(71, 751)),
        (slice(978, 1011), slice(751, 784)),
        (slice(1011, 1015), slice(784, 788)),
    ):
        assert torch.equal(w[:, dst], old[:, src])
    assert not w[:, 71:148].any() and not w[:, 828:978].any()
    checkpoint = policy_checkpoint(
        actor, critic, identity=IDENTITY, provenance={"actor": am, "critic": cm}
    )
    path = tmp_path / "policy_weights.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    # Mutate then restore, so equality isn't a no-op load test.
    with torch.no_grad():
        actor.decoder.net[-1].bias.add_(1)
        critic.mlp.net[-1].bias.add_(2)
    meta = load_policy_weights(actor, critic, loaded, expected_identity=IDENTITY)
    torch.testing.assert_close(actor(obs, goal), actual, atol=0, rtol=0)
    torch.testing.assert_close(critic(feat, goal, xi, root), cv, atol=0, rtol=0)
    assert meta["training_ready"] is False and meta["hoi_value_calibrated"] is False
    assert "weights-only" in meta["lifecycle"]
    controlled_fusion(actor)
    go, gg = torch.autograd.grad(actor(obs, goal).sum(), (obs, goal))
    assert go[:, 930:].norm() > 0 and gg[:, 680:].norm() > 0
    save_evidence(
        f"migration_{str(dtype).split('.')[-1]}",
        {
            "actor_max_abs_error": float((actual - expected).detach().abs().max()),
            "critic_max_abs_error": float((cv - ov).detach().abs().max()),
            "atol": atol,
            "rtol": rtol,
            "strict_roundtrip": True,
            "zero_fusion_object_gradient": "exact zero",
            "nonzero_fixture_object_history_gradient": float(go[:, 930:].norm()),
            "nonzero_fixture_object_goal_gradient": float(gg[:, 680:].norm()),
            "source": SOURCE,
            "optimizer_migration": False,
        },
    )


def test_strict_schema_scale_identity_dtype_keys_and_atomic_rejection():
    torch.set_num_threads(1)
    actor, critic = HOIActor(), HOICritic()
    payload = policy_checkpoint(actor, critic, identity=IDENTITY, provenance={"scope": "synthetic"})
    baseline = actor.decoder.net[0].weight.detach().clone()
    variants = []
    for mutate in (
        lambda m: m.update(schema="old-robot-policy"),
        lambda m: m.pop("research_authority"),
        lambda m: m.update(research_authority={"sha256": "wrong"}),
        lambda m: m["actor"]["features"].update(anchor_obs=True),
        lambda m: m["actor"]["features"].update(object_history="all in current heading"),
        lambda m: m["critic"]["features"]["multipliers"]["object_error"].__setitem__(14, 1.0),
        lambda m: m["identity"].update(object_asset_sha256="5" * 64),
        lambda m: m.update(training_ready=True),
    ):
        v = dict(payload)
        v["metadata"] = copy.deepcopy(payload["metadata"])
        mutate(v["metadata"])
        variants.append(v)
    v = dict(payload, optimizer={})
    variants.append(v)
    for key, change in (
        ("object_history_scale", lambda x: x * 2),
        ("decoder.net.0.weight", lambda x: x[:, :962]),
        ("decoder.net.0.bias", lambda x: x.double()),
        ("decoder.net.0.bias", lambda x: x * float("nan")),
    ):
        v = dict(payload, actor=dict(payload["actor"]))
        v["actor"][key] = change(v["actor"][key])
        variants.append(v)
    v = dict(payload, critic=dict(payload["critic"]))
    v["critic"].pop("object_error_scale")
    variants.append(v)
    for bad in variants:
        with pytest.raises(ValueError):
            load_policy_weights(actor, critic, bad, expected_identity=IDENTITY)
        assert torch.equal(actor.decoder.net[0].weight, baseline)


def test_robot_migration_rejects_anchor_changed_scales_hidden_widths_and_provenance():
    torch.set_num_threads(1)
    actor, critic = HOIActor(), HOICritic()
    old_actor, old_critic = Policy(PolicyCfg()), Critic(CriticCfg())
    for cfg in (
        replace(PolicyCfg(), goal_dim=689),
        replace(PolicyCfg(), history=5),
        replace(PolicyCfg(), obs_term_scale=(1, 1, 1, 1, 1)),
        replace(PolicyCfg(), encoder_hidden=(16, 16)),
    ):
        with pytest.raises(ValueError):
            migrate_robot_actor(old_actor.state_dict(), cfg, actor, provenance=SOURCE)
        with pytest.raises(ValueError):
            migrate_robot_critic(
                old_critic.state_dict(), CriticCfg(), cfg, critic, provenance=SOURCE
            )
    with pytest.raises(ValueError):
        migrate_robot_actor(old_actor.state_dict(), old_actor.cfg, actor, provenance={})
    bad = dict(old_actor.state_dict())
    bad["decoder.scale"] = bad["decoder.scale"] * 2
    with pytest.raises(ValueError, match="normalization"):
        migrate_robot_actor(bad, old_actor.cfg, actor, provenance=SOURCE)
