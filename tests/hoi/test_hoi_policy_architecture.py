from dataclasses import replace
from itertools import pairwise

import pytest
import torch
from p7_policy_fixtures import controlled_fusion, fixture, save_evidence, start_batch

from supermimic.hoi.imagination import HOIPolicyTrainer, HOIRolloutContext
from supermimic.hoi.policy_features import (
    critic_features,
    object_future_goal,
    object_reference_error,
    rollout_critic_inputs,
)
from supermimic.imagination.policy_trainer import PolicyTrainer, frozen_parameters
from supermimic.kinematics.rotations import quat_from_rotvec, quat_mul


def inputs(fx, st):
    obs = fx.trainer.observation_adapter.history(st.hist_states, st.hist_actions, st.xi)
    obj = object_future_goal(fx.registry, st.clip_id, st.local_frame, fx.wm.spec)
    goal = torch.cat((fx.lib.goal_window(st.frame, 10), obj.values), -1)
    return obs, goal


def test_frozen_architecture_widths_and_unchanged_wm():
    fx = fixture(coupled=False)
    st = start_batch(fx, (300,))
    obs, goal = inputs(fx, st)
    assert obs.shape == (1, 1080) and goal.shape == (1, 830)
    assert fx.policy.decoder_input(obs, goal).shape == (1, 1058)
    assert fx.policy(obs, goal).shape == (1, 29)
    assert fx.critic.mlp.net[0].in_features == 1015
    for mlp, dims in (
        (fx.policy.robot_encoder.mlp, [680, 2048, 1024, 512, 256, 32]),
        (fx.policy.decoder, [1058, 2048, 1024, 512, 256, 29]),
        (fx.policy.object_history_encoder, [150, 128, 64]),
        (fx.policy.object_goal_encoder, [150, 128, 64, 32]),
        (fx.critic.mlp, [1015, 256, 128, 1]),
    ):
        linears = [m for m in mlp.net if isinstance(m, torch.nn.Linear)]
        assert [(m.in_features, m.out_features) for m in linears] == list(pairwise(dims))
    assert fx.wm.dynamics.mlp.net[0].in_features == 166
    assert fx.wm.dynamics.mlp.net[-1].out_features == 82
    assert fx.wm.spec.canonical(st.state).shape == (1, 86)
    assert not ({id(p) for p in fx.policy.parameters()} & {id(p) for p in fx.critic.parameters()})
    assert HOIPolicyTrainer.update is PolicyTrainer.update
    assert HOIPolicyTrainer.critic_batch is PolicyTrainer.critic_batch
    assert fx.trainer.initialization_metadata()["training_ready"] is False
    with pytest.raises(ValueError):
        HOIPolicyTrainer(
            fx.wm,
            fx.policy,
            fx.critic,
            fx.registry,
            fx.reward,
            replace(fx.trainer.cfg, anchor_obs=True),
        )


@pytest.mark.parametrize("kind", ["position", "rotation", "linear", "angular", "goal"])
def test_independent_object_feedback_and_nonzero_fusion_jacobians(kind):
    fx = fixture()
    controlled_fusion(fx.policy)
    st = start_batch(fx, (300,))
    obs, goal = inputs(fx, st)
    changed = st.hist_states.clone()
    if kind == "goal":
        obs2, goal2 = obs, goal.clone()
        goal2[:, 680] += 0.03
    else:
        if kind == "rotation":
            changed[-1, :, 74:78] = quat_mul(
                quat_from_rotvec(changed.new_tensor([[0.1, -0.05, 0.02]])), changed[-1, :, 74:78]
            )
        else:
            index = {"position": 71, "linear": 78, "angular": 81}[kind]
            changed[-1, :, index] += 0.03
        obs2 = fx.trainer.observation_adapter.history(changed, st.hist_actions, st.xi)
        goal2 = goal
        assert torch.equal(obs2[:, :930], obs[:, :930])
        assert not torch.equal(obs2[:, 930:], obs[:, 930:])
    a, b = fx.policy(obs, goal), fx.policy(obs2, goal2)
    assert not torch.equal(a, b)
    o, g = obs2.detach().requires_grad_(), goal2.detach().requires_grad_()
    grad_o, grad_g = torch.autograd.grad(fx.policy(o, g).sum(), (o, g))
    assert torch.isfinite(grad_o).all() and torch.isfinite(grad_g).all()
    assert grad_o[:, 930:].norm() > 0 and grad_g[:, 680:].norm() > 0
    fx.policy(o, g).square().sum().backward()
    for encoder in (fx.policy.object_history_encoder, fx.policy.object_goal_encoder):
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters()
        )
        assert sum(p.grad.square().sum() for p in encoder.parameters()) > 0


def test_critic_named_input_scaling_and_controlled_sensitivity():
    fx = fixture()
    st = start_batch(fx, (300,))
    _obs, goal = inputs(fx, st)
    ref = fx.registry.gather(st.clip_id, st.local_frame)
    feat = critic_features(st.state, st.prev_action, ref, fx.wm.spec, fx.wm.support)
    xi = fx.wm.spec.normalize_xi(st.xi)
    root = fx.heads.root_error_features(st.state[:, :71], ref.robot)
    x = fx.critic.input_tensor(feat, goal, xi, root)
    assert x.shape == (1, 1015)
    assert torch.equal(x[:, :104], feat[:, :104])
    assert torch.equal(x[:, 104:119], feat[:, 104:119] * fx.critic.object_error_scale)
    assert torch.equal(x[:, 119:148], st.prev_action)
    assert torch.equal(x[:, 148:828], goal[:, :680] * fx.critic.robot_goal_scale)
    assert torch.equal(x[:, 828:978], goal[:, 680:] * fx.critic.object_goal_scale)
    assert torch.equal(x[:, 978:1011], xi) and torch.equal(x[:, 1011:], root)
    with torch.no_grad():
        for p in fx.critic.parameters():
            p.fill_(0.002)
    f, g = feat.detach().requires_grad_(), goal.detach().requires_grad_()
    gf, gg = torch.autograd.grad(fx.critic(f, g, xi, root).sum(), (f, g))
    for sl in (slice(71, 86), slice(86, 104), slice(104, 119), slice(119, 148)):
        assert torch.isfinite(gf[:, sl]).all() and gf[:, sl].norm() > 0
    assert torch.isfinite(gg[:, 680:]).all() and gg[:, 680:].norm() > 0
    changed = st.state.clone()
    changed[:, 71] += 0.1
    f2 = critic_features(changed, st.prev_action, ref, fx.wm.spec, fx.wm.support)
    assert not torch.equal(f2[:, 104:119], feat[:, 104:119])
    assert not torch.equal(fx.critic(feat, goal, xi, root), fx.critic(f2, goal, xi, root))
    g2 = goal.clone()
    g2[:, 680] += 1
    x2 = fx.critic.input_tensor(feat, g2, xi, root)
    assert torch.equal(x2[:, :828], x[:, :828])
    assert torch.equal(x2[:, 978:], x[:, 978:])
    assert not torch.equal(x2[:, 828:978], x[:, 828:978])


def test_temporal_alignment_recomputed_object_history_and_masked_critic():
    fx = fixture(horizon=3)
    controlled_fusion(fx.policy)
    st = start_batch(fx, (300, 838), remaining_steps=torch.tensor([2, 9]))
    observations, actions = [], []

    def capture(module, args, output):
        observations.append(args[0])
        actions.append(output)

    hook = fx.policy.register_forward_hook(capture)
    with frozen_parameters(fx.wm, fx.critic):
        loss, _, ro, (feat, goal, root), xi = fx.trainer.actor_loss(st)
        hook.remove()
        assert ro.states.shape == (4, 2, 84)
        assert ro.lengths.tolist() == [2, 1] and ro.bootstrap.tolist() == [True, False]
        assert ro.object_goals.shape == (4, 2, 150)
        assert ro.object_goal_valid.shape == (4, 2, 10)
        assert torch.equal(ro.previous_actions[0], st.prev_action)
        for t in range(3):
            valid = ro.valid[t]
            assert torch.equal(ro.previous_actions[t + 1, valid], ro.actions[t, valid])
        for t in range(4):
            valid = t <= ro.lengths
            ref = fx.registry.gather(st.clip_id[valid], ro.context.local_frames[t, valid])
            torch.testing.assert_close(
                feat[t, valid, 104:119], object_reference_error(ro.states[t, valid], ref)
            )
            assert not feat[t, ~valid].any()
        assert ro.diagnostics["evaluated_local_frame"].tolist() == [[301, 839], [302, -1], [-1, -1]]
        # After WM prediction the newly observed Object changes and keeps action gradients.
        expected = fx.trainer.observation_adapter.observation(
            ro.states[1, :1], ro.actions[0, :1], st.xi[:1]
        )
        assert torch.allclose(
            observations[1][:, 957:960], expected[:, 93:96]
        )  # latest Object position
        g = torch.autograd.grad(observations[1][:, 930:].sum(), actions[0], retain_graph=True)[0]
        assert torch.isfinite(g).all() and g[0].norm() > 0
        assert g[1].norm() == 0  # completed clip excluded from later policy calls
        loss.backward()
    X, Y, _, _ = fx.trainer.critic_batch(ro, xi, (feat, goal, root))
    assert Y.numel() == 3 and X[0].shape == (3, 148)
    assert not any(x.requires_grad for x in X) and not Y.requires_grad
    bad = replace(st, reference_sha256=("0" * 64, "0" * 64))
    with pytest.raises(ValueError, match="identity"):
        HOIRolloutContext(fx.registry, bad, 3)


def test_terminal_object_value_gradient_through_frozen_wm_and_detach_control():
    fx = fixture(horizon=2)
    controlled_fusion(fx.policy)
    st = start_batch(fx, (300,))
    # The terminal value depends only on Object canonical position x, no reward/prev-action path.
    with torch.no_grad():
        for p in fx.critic.parameters():
            p.zero_()
        fx.critic.mlp.net[0].weight[0, 71] = 1
        fx.critic.mlp.net[2].weight[0, 0] = 1
        fx.critic.mlp.net[4].weight[0, 0] = 1
    with frozen_parameters(fx.wm, fx.critic):
        _, _, ro, (feat, goals, roots), xi = fx.trainer.actor_loss(st)
        tail = fx.critic(feat[-1], goals[-1], xi, roots[-1]).sum()
        gs = torch.autograd.grad(tail, ro.states, retain_graph=True)[0]
        assert gs[-1, :, 71:].norm() > 0 and torch.isfinite(gs).all()
        gp = torch.autograd.grad(tail, tuple(fx.policy.parameters()), retain_graph=True)
        assert sum(g.square().sum() for g in gp) > 0
        detached = replace(ro, states=ro.states.detach())
        dfeat, dg, dr = rollout_critic_inputs(detached, ro.context, fx.wm.spec, fx.wm.support)
        negative = fx.critic(dfeat[-1], dg[-1], xi, dr[-1]).sum()
        ng = torch.autograd.grad(negative, tuple(fx.policy.parameters()), allow_unused=True)
        assert all(g is None or torch.count_nonzero(g) == 0 for g in ng)
        # Finite difference on terminal physical Object position isolates live feature/value path.
        terminal = ro.states[-1].detach().requires_grad_()
        ref = fx.registry.gather(st.clip_id, st.local_frame + 2)

        def value(s):
            f = critic_features(s, ro.previous_actions[-1].detach(), ref, fx.wm.spec, fx.wm.support)
            return fx.critic(f, goals[-1].detach(), xi, roots[-1].detach()).sum()

        g = torch.autograd.grad(value(terminal), terminal)[0]
        delta = torch.zeros_like(terminal)
        delta[:, 71] = 1e-6
        fd = (value(terminal + delta) - value(terminal - delta)) / 2e-6
        torch.testing.assert_close(fd, g[:, 71].sum(), atol=1e-8, rtol=1e-6)
    save_evidence(
        "terminal_gradient",
        {
            "object_state_gradient_norm": float(gs[-1, :, 71:].norm()),
            "actor_gradient_norm": sum(float(g.square().sum()) for g in gp) ** 0.5,
            "detach_control": "zero",
            "finite_difference_error": float((fd - g[:, 71].sum()).detach().abs()),
            "dynamics": "synthetic coupled residual; not learned physical fidelity",
        },
    )


def test_update_isolation_and_exception_restore_with_synthetic_unit_steps():
    fx = fixture(horizon=1)
    controlled_fusion(fx.policy)
    st = start_batch(fx, (300,))
    protected = list(fx.wm.parameters()) + list(fx.critic.parameters())
    protected[0].requires_grad_(False)
    flags = [p.requires_grad for p in protected]
    before = [p.detach().clone() for p in protected]
    actor_before = [p.detach().clone() for p in fx.policy.parameters()]
    with frozen_parameters(fx.wm, fx.critic):
        assert all(not p.requires_grad for p in protected)
        loss, _, ro, data, xi = fx.trainer.actor_loss(st)
        loss.backward()
        assert all(p.grad is None for p in protected)
        for branch in (fx.policy.object_history_encoder, fx.policy.object_goal_encoder):
            assert sum(p.grad.square().sum() for p in branch.parameters()) > 0
        fx.trainer.opt.step()  # one synthetic unit-test update; no training loop
    assert flags == [p.requires_grad for p in protected]
    assert all(torch.equal(a, b) for a, b in zip(before, protected))
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, fx.policy.parameters()))
    fx.policy.zero_grad(set_to_none=True)
    actor_after = [p.detach().clone() for p in fx.policy.parameters()]
    X, Y, _, _ = fx.trainer.critic_batch(ro, xi, data)
    closs = (fx.critic(*X) - Y).square().mean()
    closs.backward()
    assert all(p.grad is None for p in fx.policy.parameters())
    assert all(p.grad is None for p in fx.wm.parameters())
    fx.trainer.critic_opt.step()
    assert all(torch.equal(a, b) for a, b in zip(actor_after, fx.policy.parameters()))
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before[len(list(fx.wm.parameters())) :], fx.critic.parameters())
    )
    with pytest.raises(RuntimeError), frozen_parameters(fx.wm, fx.critic):
        raise RuntimeError("controlled exception")
    assert flags == [p.requires_grad for p in protected]
    save_evidence(
        "isolation",
        {
            "actor_loss": float(loss.detach()),
            "critic_loss": float(closs.detach()),
            "actor_only_step": True,
            "critic_only_step": True,
            "wm_steps": 0,
            "scope": "one synthetic unit step each, no collection or training loop",
            "freeze_restore_on_exception": True,
        },
    )


@pytest.mark.parametrize(
    "frame,timeout,failure,length,bootstrap",
    [
        (838, None, None, 1, False),
        (300, 1, None, 1, True),
        (300, None, 1, 1, False),
        (838, 1, 1, 1, False),
        (300, None, None, 3, True),
        (300, 1, 2, 1, True),
    ],
)
def test_new_interface_preserves_boundary_and_td_masks(frame, timeout, failure, length, bootstrap):
    fx = fixture(horizon=3)
    st = start_batch(
        fx,
        (frame,),
        remaining_steps=None if timeout is None else torch.tensor([timeout]),
        hard_failure_step=None if failure is None else torch.tensor([failure]),
    )
    with frozen_parameters(fx.wm, fx.critic):
        loss, _, ro, data, xi = fx.trainer.actor_loss(st)
        assert torch.isfinite(loss)
        assert ro.lengths.tolist() == [length] and ro.bootstrap.tolist() == [bootstrap]
        assert int(ro.valid.sum()) == length
        assert not ro.rewards[length:].any()
        X, Y, values, targets = fx.trainer.critic_batch(ro, xi, data)
    assert len(Y) == length and len(X[0]) == length
    assert not targets[length:].any()
    # Independently verify final valid TD target: hard/natural end eliminates tail.
    expected = ro.rewards[length - 1, 0].detach()
    if bootstrap:
        expected = (
            expected + fx.trainer.cfg.gamma * (1 - ro.dhat[length, 0].detach()) * values[length, 0]
        )
    torch.testing.assert_close(targets[length - 1, 0], expected)


def test_terminal_previous_clean_action_retains_actor_gradient():
    fx = fixture(horizon=2)
    st = start_batch(fx, (300,))
    with torch.no_grad():
        for p in fx.critic.parameters():
            p.zero_()
        fx.critic.mlp.net[0].weight[0, 119] = 1
        fx.critic.mlp.net[2].weight[0, 0] = 1
        fx.critic.mlp.net[4].weight[0, 0] = 1
    with frozen_parameters(fx.wm, fx.critic):
        _, _, ro, (feat, goals, root), xi = fx.trainer.actor_loss(st)
        tail = fx.critic(feat[-1], goals[-1], xi, root[-1]).sum()
        grad = torch.autograd.grad(tail, ro.previous_actions, retain_graph=True)[0]
        assert grad[-1, :, 0].norm() > 0
        actor = torch.autograd.grad(tail, tuple(fx.policy.parameters()))
        assert sum(g.square().sum() for g in actor) > 0


def test_real_force_contact_sidecars_cannot_change_policy_value_targets_or_gradients():
    from test_policy_adapter import fill_replay

    from supermimic.hoi.imagination import HOIPolicyReplayBuffer
    from supermimic.hoi.replay import FORCE_SHAPES

    fx = fixture(torch.float32, horizon=2)
    replay = HOIPolicyReplayBuffer(2, 32, "cpu", force_on=0.001, registry=fx.registry)
    fill_replay(replay, fx)
    outcomes = []
    for changed in (False, True):
        if changed:
            for name in (*FORCE_SHAPES, "intent_ref"):
                replay.fields[name].fill_(999.0)  # adversarial sidecars, never features/rewards
        st = replay._gather_starts(torch.tensor([0, 1]), torch.tensor([2, 2]), 10)
        fx.policy.zero_grad(set_to_none=True)
        with frozen_parameters(fx.wm, fx.critic):
            loss, _, ro, data, xi = fx.trainer.actor_loss(st)
            loss.backward()
            grads = [p.grad.clone() for p in fx.policy.parameters()]
        _X, Y, values, targets = fx.trainer.critic_batch(ro, xi, data)
        outcomes.append((ro, data, grads, Y, values, targets))
    a, b = outcomes
    for key in ("actions", "states", "rewards", "object_goals"):
        assert torch.equal(getattr(a[0], key), getattr(b[0], key))
    for group in (1, 2):
        for x, y in zip(a[group], b[group]):
            assert torch.equal(x, y)
    for i in (3, 4, 5):
        assert torch.equal(a[i], b[i])
