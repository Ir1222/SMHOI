from dataclasses import replace

import pytest
import torch
from p5_fixtures import fixture, start_batch

from supermimic.hoi.imagination import HOIRolloutContext
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.policy_trainer import frozen_parameters


def rollout(fx, start):
    with frozen_parameters(fx.wm, fx.critic):
        loss, J, ro, inputs, xi = fx.trainer.actor_loss(start)
    return loss, J, ro, inputs, xi


def test_exact_next_frames_and_no_repeated_terminal_rewards():
    fx = fixture(horizon=3)
    start = start_batch(fx)
    loss, J, ro, _, _ = rollout(fx, start)
    assert ro.lengths.tolist() == [3, 3, 3, 3, 1]
    assert ro.bootstrap.tolist() == [True, True, True, True, False]
    d = ro.diagnostics
    assert d["evaluated_local_frame"][0].tolist() == [259, 260, 549, 550, 839]
    assert d["contact"][0].tolist() == [False, True, True, False, False]
    assert d["evaluated_local_frame"][:, 4].tolist() == [839, -1, -1]
    assert torch.equal(ro.states[1, 4], ro.states[3, 4])
    assert torch.equal(ro.goals[1, 4], ro.goals[3, 4])
    assert torch.equal(ro.e_root[1, 4], ro.e_root[3, 4])
    assert not ro.rewards[1:, 4].any() and not ro.actions[1:, 4].any()
    assert ro.frames[:, 4].tolist() == [838, 839, 839, 839]
    assert torch.isfinite(loss) and torch.equal(loss, -J / 3)
    for t in range(3):
        rows = ro.valid[t].nonzero().squeeze(-1)
        r = fx.registry.gather(start.clip_id[rows], start.local_frame[rows] + t + 1)
        s = ro.states[t + 1, rows]
        torch.testing.assert_close(
            ro.e_root[t + 1, rows], fx.heads.root_error_features(s[:, :71], r.robot)
        )
        expected = fx.reward(
            s,
            ro.actions[t, rows],
            start.prev_action[rows] if t == 0 else ro.actions[t - 1, rows],
            r.robot,
            r.box,
        )
        torch.testing.assert_close(ro.rewards[t, rows], expected["total"], atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(
            ro.dhat[t + 1, rows], torch.sigmoid(fx.wm.termination_logits(s[:, :71], r.robot))
        )
        torch.testing.assert_close(ro.goals[t + 1, rows], fx.lib.goal_window(r.global_frame, 10))
    with pytest.raises(ValueError, match="outgoing"):
        HOIRolloutContext(fx.registry, start_batch(fx, (839,)), 3)


def test_timeout_clip_hard_failure_and_coincident_metadata():
    fx = fixture(horizon=4)
    start = start_batch(
        fx,
        (100, 838, 100, 100, 100),
        remaining_steps=torch.tensor([2, 1, 4, 4, 4]),
        hard_failure_step=torch.tensor([0, 1, 2, 3, 0]),
        robot_failure=torch.tensor([False, True, True, False, False]),
        object_failure=torch.tensor([False, True, False, True, False]),
    )
    _, _, ro, _, _ = rollout(fx, start)
    assert ro.lengths.tolist() == [2, 1, 2, 3, 4]
    assert ro.bootstrap.tolist() == [True, False, False, False, True]
    assert ro.boundaries["timeout"].tolist() == [True, True, False, False, True]
    assert ro.boundaries["clip_end"].tolist() == [False, True, False, False, False]
    assert ro.boundaries["physical_failure"].tolist() == [False, True, True, True, False]
    assert ro.dhat[1, 1] == ro.dhat[2, 2] == ro.dhat[3, 3] == 1
    # Natural task completion alone never becomes a physical failure label.
    _, _, natural, _, _ = rollout(fx, start_batch(fx, (838,)))
    assert natural.dhat[1, 0] < 1 and not natural.boundaries["physical_failure"].any()
    assert not natural.continuation[0, 0]


def test_hand_computed_returns_td_padding_and_original_all_valid():
    fx = fixture(horizon=3)
    _, _, base, _, _ = rollout(fx, start_batch(fx, (100, 838, 100)))
    valid = torch.tensor([[True, True, True], [True, False, True], [True, False, False]])
    lengths = torch.tensor([3, 1, 2])
    boot = torch.tensor([True, False, True])
    B = valid.clone()
    B[0, 1] = False
    rewards = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, float("nan"), 5.0], [6.0, float("nan"), float("nan")]],
        dtype=torch.float64,
        requires_grad=True,
    )
    d = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.0, 0.0]], dtype=torch.float64)
    c = torch.cat((torch.ones(1, 3, dtype=d.dtype), (1 - d).cumprod(0)))
    ro = replace(
        base,
        rewards=rewards,
        dhat=torch.cat((torch.zeros(1, 3, dtype=d.dtype), d)),
        survival=c,
        valid=valid,
        lengths=lengths,
        bootstrap=boot,
        continuation=B,
    )
    V = torch.arange(12, dtype=d.dtype).reshape(4, 3) + 10
    gamma, lam = 0.9, 0.7
    tails = V[lengths, torch.arange(3)]
    manual = []
    targets = torch.zeros_like(rewards)
    for n, L in enumerate(lengths.tolist()):
        value = sum(gamma**k * c[k, n] * rewards[k, n] for k in range(L))
        value = value + gamma**L * c[L, n] * tails[n] * boot[n]
        manual.append(value)
        G = tails[n] * boot[n]
        for k in range(L - 1, -1, -1):
            G = rewards[k, n] + gamma * (1 - d[k, n]) * B[k, n] * (
                (1 - lam) * V[k + 1, n] + lam * G
            )
            targets[k, n] = G
    actual = policy_objective(ro, tails, gamma)
    torch.testing.assert_close(actual, torch.stack(manual).mean(), atol=1e-10, rtol=1e-8)
    expected_grad = torch.zeros_like(rewards)
    for n, L in enumerate(lengths.tolist()):
        for k in range(L):
            expected_grad[k, n] = gamma**k * c[k, n] / 3
    torch.testing.assert_close(
        torch.autograd.grad(actual, rewards)[0], expected_grad, atol=1e-10, rtol=1e-8
    )
    out = gae_targets(
        rewards, d, V, gamma, lam, valid=valid, lengths=lengths, bootstrap=boot, continuation=B
    )
    torch.testing.assert_close(out, targets, atol=1e-10, rtol=1e-8)
    allvalid = torch.ones_like(valid)
    allL = torch.full_like(lengths, 3)
    allboot = torch.ones_like(boot)
    r = torch.nan_to_num(rewards.detach())
    rr = replace(ro, rewards=r, valid=None, lengths=None, bootstrap=None, continuation=None)
    rm = replace(rr, valid=allvalid, lengths=allL, bootstrap=allboot, continuation=allvalid)
    assert torch.equal(policy_objective(rr, V[-1], gamma), policy_objective(rm, V[-1], gamma))
    assert torch.equal(
        gae_targets(r, d, V, gamma, lam),
        gae_targets(
            r,
            d,
            V,
            gamma,
            lam,
            valid=allvalid,
            lengths=allL,
            bootstrap=allboot,
            continuation=allvalid,
        ),
    )


def test_padded_critic_samples_excluded_and_small_minibatches():
    fx = fixture(horizon=4)
    _, _, ro, inputs, xi = rollout(fx, start_batch(fx, (838,)))
    X, Y, _, _ = fx.trainer.critic_batch(ro, xi, inputs)
    assert Y.numel() == 1 and all(len(x) == 1 and not x.requires_grad for x in X)
    polluted = replace(
        ro,
        states=ro.states.detach().clone(),
        goals=ro.goals.detach().clone(),
        e_root=ro.e_root.detach().clone(),
        rewards=ro.rewards.detach().clone(),
        dhat=ro.dhat.detach().clone(),
    )
    polluted.states[2:] = float("nan")
    polluted.goals[2:] = float("nan")
    polluted.e_root[2:] = float("nan")
    polluted.rewards[1:] = float("nan")
    polluted.dhat[2:] = float("nan")
    X2, Y2, _, _ = fx.trainer.critic_batch(polluted, xi)
    assert torch.equal(Y, Y2)
    for a, b in zip(X, X2):
        assert torch.equal(a, b)
    pred = fx.critic(*X)
    pred2 = fx.critic(*X2)
    assert torch.equal((pred - Y).square().mean(), (pred2 - Y2).square().mean())
    batches = fx.trainer.critic_minibatch_indices(1, "cpu", masked=True)
    assert len(batches) == 1 and batches[0].tolist() == [0]


def test_padded_reward_and_survival_cannot_change_actor_gradients():
    fx = fixture(horizon=4)
    with frozen_parameters(fx.wm, fx.critic):
        _, _, ro, _, _ = fx.trainer.actor_loss(start_batch(fx, (838, 259)))
        original = policy_objective(ro, None, fx.trainer.cfg.gamma)
        params = tuple(fx.policy.parameters())
        expected = torch.autograd.grad(original, params, retain_graph=True)
        # Make invalid rewards explicitly depend on Actor weights, with a large
        # coefficient, and invalid pre-step survival nonfinite. Neither is data.
        poison = sum(p.sum() for p in params) * 1e6
        rewards = torch.where(ro.valid, ro.rewards, poison)
        survival = torch.cat(
            (
                torch.where(ro.valid, ro.survival[:-1], float("nan")),
                ro.survival[-1:],
            )
        )
        polluted = replace(ro, rewards=rewards, survival=survival)
        actual = policy_objective(polluted, None, fx.trainer.cfg.gamma)
        gradients = torch.autograd.grad(actual, params)
    assert torch.equal(original, actual)
    for a, b in zip(expected, gradients):
        assert torch.isfinite(b).all() and torch.equal(a, b)
