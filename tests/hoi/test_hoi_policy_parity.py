"""Completed Inductor forward/backward comparisons of the approved HOI path."""

from dataclasses import replace

import pytest
import torch
from p7_policy_fixtures import controlled_fusion, fixture, save_evidence, start_batch

from supermimic.hoi.imagination import HOIPolicyTrainer
from supermimic.imagination.policy_trainer import frozen_parameters


def run_parity(device, dtype):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    assert not torch._dynamo.config.suppress_errors
    fx = fixture(dtype, device, horizon=2)
    controlled_fusion(fx.policy)
    st = start_batch(fx, (300, 838))
    eager = fx.trainer
    compiled = HOIPolicyTrainer(
        fx.wm,
        fx.policy,
        fx.critic,
        fx.registry,
        fx.reward,
        replace(eager.cfg, compile="default"),
        backend="inductor",
    )
    # The production adapters compile step/observation/history/live Critic features.
    # Also compile both complete fixed-width networks for this acceptance test.
    compiled.policy = torch.compile(fx.policy, backend="inductor", fullgraph=True)
    compiled.critic = torch.compile(fx.critic, backend="inductor", fullgraph=True)
    runs = []
    for trainer in (eager, compiled):
        fx.policy.zero_grad(set_to_none=True)
        with frozen_parameters(fx.wm, fx.critic):
            loss, objective, ro, features, xi = trainer.actor_loss(st)
            loss.backward()
            gradients = [p.grad.detach().clone() for p in fx.policy.parameters()]
        batch = trainer.critic_batch(ro, xi, features)
        runs.append((loss.detach(), objective.detach(), ro, features, gradients, batch))
    atol, rtol = (2e-5, 2e-5) if dtype == torch.float32 else (1e-9, 1e-7)
    errors = {}

    def check(name, a, b):
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)
        errors[name] = float((a - b).detach().abs().max())

    a, b = runs
    check("loss", a[0], b[0])
    check("objective", a[1], b[1])
    for key in (
        "states",
        "actions",
        "rewards",
        "dhat",
        "survival",
        "goals",
        "object_goals",
        "previous_actions",
        "e_root",
    ):
        check(key, getattr(a[2], key), getattr(b[2], key))
    for key in ("valid", "lengths", "bootstrap", "continuation", "frames", "object_goal_valid"):
        assert torch.equal(getattr(a[2], key), getattr(b[2], key))
    for i, (x, y) in enumerate(zip(a[3], b[3])):
        check(f"critic_input_{i}", x, y)
    for i, (x, y) in enumerate(zip(a[4], b[4])):
        check(f"actor_gradient_{i}", x, y)
    for i, (x, y) in enumerate(zip(a[5][0], b[5][0])):
        check(f"critic_batch_{i}", x, y)
    for i in range(1, 4):
        check(f"td_value_target_{i}", a[5][i], b[5][i])
    # Critic regression gradients as well as its input/state derivatives must match.
    critic_grads = []
    for trainer, result in zip((eager, compiled), runs):
        fx.critic.zero_grad(set_to_none=True)
        X, Y = result[5][:2]
        (trainer.critic(*X) - Y).square().mean().backward()
        critic_grads.append([p.grad.detach().clone() for p in fx.critic.parameters()])
    for i, (x, y) in enumerate(zip(*critic_grads)):
        check(f"critic_gradient_{i}", x, y)
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    unimplemented = dict(torch._dynamo.utils.counters["unimplemented"])
    assert graphs > 0
    assert not any("recompile" in k or "cache_size_limit" in k for k in unimplemented)
    return {
        "status": "PASS",
        "device": device,
        "dtype": str(dtype),
        "backend": "inductor",
        "horizon": 2,
        "valid_lengths": a[2].lengths.tolist(),
        "atol": atol,
        "rtol": rtol,
        "max_absolute_errors": errors,
        "compiled_unique_graphs": graphs,
        "graph_breaks": dict(torch._dynamo.utils.counters["graph_break"]),
        "compiler_unimplemented": unimplemented,
        "single_full_graph_claim": False,
        "recompile_limit": torch._dynamo.config.recompile_limit,
        "suppress_errors": False,
        "actor_gradient_norm": sum(float(g.square().sum()) for g in a[4]) ** 0.5,
        "optimizer_steps": 0,
        "dynamics": "synthetic coupling; no physical fidelity claim",
    }


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
def test_inductor_actor_critic_features_and_rollout(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; GPU parity NOT TESTED")
    # Same validation-only specialization allowance as P5; no global/default edit.
    with torch._dynamo.config.patch(recompile_limit=64):
        result = run_parity(device, dtype)
    save_evidence(f"parity_{device}_{str(dtype).split('.')[-1]}", result)
