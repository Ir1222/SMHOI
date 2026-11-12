import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from p5_fixtures import TEST_CFG, fixture, save_evidence, start_batch

from supermimic.data.replay_buffer import ReplayBuffer, StartBatch
from supermimic.hoi.imagination import HOIPolicyReplayBuffer, HOIPolicyTrainer, HOIRolloutContext
from supermimic.hoi.replay import FORCE_SHAPES
from supermimic.hoi.tracking_reward import HOI_TERM_NAMES, robot_state
from supermimic.imagination.policy_trainer import PolicyTrainer, frozen_parameters
from supermimic.imagination.rollout import imagine
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major


def load_original(relative, name):
    path = Path("/home/han/pyx/SuperMimic-Reproduced/src/supermimic") / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_component_isolated_action_actor_paths_and_finite_differences():
    fx = fixture()
    st = start_batch(fx, (300,))
    ref = fx.registry.gather(st.clip_id, st.local_frame + 1)
    obs = flatten_history_term_major(
        torch.stack(
            [
                analytic_observation(robot_state(s), a, st.xi, fx.q0)
                for s, a in zip(st.hist_states, st.hist_actions)
            ]
        )
    )
    goal = fx.lib.goal_window(st.frame, 10)
    evidence = {}
    with frozen_parameters(fx.wm, fx.critic):
        a = fx.policy(obs, goal)
        a.retain_grad()
        sn, total, terms, *_ = fx.trainer.step_fn(st.state, a, st.prev_action, st.xi, ref)
        values = {
            k: terms[HOI_TERM_NAMES.index(k)].sum()
            for k in (*HOI_TERM_NAMES[:6], *HOI_TERM_NAMES[8:])
        }
        for name, value in values.items():
            action_grad = torch.autograd.grad(value, a, retain_graph=True)[0]
            grads = torch.autograd.grad(value, tuple(fx.policy.parameters()), retain_graph=True)
            assert torch.isfinite(action_grad).all() and action_grad.norm() > 1e-12, name
            assert all(torch.isfinite(g).all() for g in grads)
            assert sum(float(g.square().sum()) for g in grads) > 1e-24, name
            evidence[name] = {
                "action_gradient_norm": float(action_grad.norm()),
                "actor_gradient_norm": sum(float(g.square().sum()) for g in grads) ** 0.5,
            }
        # Independent robot -> EEF and Box -> target Actor paths, no regularizers.
        for state in (
            torch.cat((sn[:, :71], sn[:, 71:].detach()), -1),
            torch.cat((sn[:, :71].detach(), sn[:, 71:]), -1),
        ):
            prox = fx.reward.proximity(state, ref.box.contact)["proximity"].sum()
            grad = torch.autograd.grad(prox, a, retain_graph=True)[0]
            assert torch.isfinite(grad).all() and grad.norm() > 1e-12
        off = fx.reward.proximity(sn, torch.zeros_like(ref.box.contact))["proximity"].sum()
        assert torch.count_nonzero(torch.autograd.grad(off, a, retain_graph=True)[0]) == 0
        # Deliberate detach control eliminates geometry->action reachability.
        detached = fx.reward.proximity(sn.detach().requires_grad_(), ref.box.contact)[
            "proximity"
        ].sum()
        assert torch.autograd.grad(detached, a, allow_unused=True)[0] is None
        # FD on the actual action-to-reward graph, isolating every Box term + proximity.
        base = a.detach()
        eps = 1e-6
        for name in HOI_TERM_NAMES[8:]:
            i = HOI_TERM_NAMES.index(name)
            grad = torch.autograd.grad(terms[i].sum(), a, retain_graph=True)[0]
            j = int(grad.abs().argmax())
            delta = torch.zeros_like(base)
            delta.view(-1)[j] = eps

            def f(action, i=i):
                return fx.trainer.step_fn(st.state, action, st.prev_action, st.xi, ref)[2][i].sum()

            fd = (f(base + delta) - f(base - delta)) / (2 * eps)
            torch.testing.assert_close(fd, grad.view(-1)[j], atol=1e-7, rtol=1e-5)
        total.sum().backward()
        assert a.grad.norm() > 0
        assert all(p.grad is None for p in fx.wm.parameters())
    save_evidence(
        "component_gradients",
        {
            "fixture": "DeterministicCoupledResidual; synthetic nonzero coupling only",
            "test_only_coefficients": TEST_CFG.metadata(),
            "components": evidence,
            "proximity_off_exact_zero": True,
            "detached_control_disconnected": True,
            "box_and_proximity_action_finite_differences": "PASS: atol=1e-7 rtol=1e-5 eps=1e-6",
        },
    )


def test_full_bptt_freezing_restore_exception_and_critic_isolation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("P5 forbids optimizer steps")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    fx = fixture()
    st = start_batch(fx)
    assert HOIPolicyTrainer.update is PolicyTrainer.update
    params = list(fx.wm.parameters()) + list(fx.critic.parameters())
    params[0].requires_grad_(False)
    flags = [p.requires_grad for p in params]
    before = [p.detach().clone() for p in params]
    with frozen_parameters(fx.wm, fx.critic):
        loss, J, ro, inputs, xi = fx.trainer.actor_loss(st)
        loss.backward()
        assert all(p.grad is None for p in params)
    assert flags == [p.requires_grad for p in params]
    assert all(torch.equal(a, b) for a, b in zip(before, params))
    actor_grads = [p.grad.clone() for p in fx.policy.parameters()]
    assert all(torch.isfinite(g).all() for g in actor_grads)
    assert sum(g.square().sum() for g in actor_grads) > 0
    for p in fx.policy.parameters():
        p.grad = None
    X, Y, _, _ = fx.trainer.critic_batch(ro, xi, inputs)
    closs = (fx.critic(*X) - Y).square().mean()
    closs.backward()
    assert all(p.grad is None for p in fx.policy.parameters())
    assert all(p.grad is None for p in fx.wm.parameters())
    assert any(p.grad is not None and p.grad.norm() > 0 for p in fx.critic.parameters())
    with pytest.raises(RuntimeError, match="intentional"), frozen_parameters(fx.wm, fx.critic):
        raise RuntimeError("intentional")
    assert flags == [p.requires_grad for p in params]
    save_evidence(
        "bptt_isolation",
        {
            "actor_loss": float(loss.detach()),
            "objective": float(J.detach()),
            "actor_gradient_norm": sum(float(g.square().sum()) for g in actor_grads) ** 0.5,
            "critic_loss": float(closs.detach()),
            "valid_lengths": ro.lengths.tolist(),
            "wm_weights_unchanged": True,
            "freeze_flags_restored": True,
            "exception_restoration": True,
            "critic_loss_actor_wm_gradients_absent": True,
            "optimizer_steps": 0,
        },
    )


def test_initialized_model_zero_box_jacobian_is_reported_separately():
    fx = fixture(coupled=False)
    st = start_batch(fx, (300,))
    ref = fx.registry.gather(st.clip_id, st.local_frame + 1)
    a = torch.zeros(1, 29, dtype=st.state.dtype, requires_grad=True)
    with frozen_parameters(fx.wm):
        sn = fx.wm.step(st.state, a, st.xi)
        assert torch.equal(
            torch.autograd.grad(sn[:, 71:].sum(), a, retain_graph=True)[0], torch.zeros_like(a)
        )
        for value in fx.reward.box_terms(sn, ref.box).values():
            assert (
                torch.count_nonzero(torch.autograd.grad(value.sum(), a, retain_graph=True)[0]) == 0
            )
    save_evidence(
        "initialized_model",
        {
            "model": "actual JointWorldModel, zero initialized output layer",
            "action_to_box_jacobian": "zero",
            "four_box_action_gradients": "zero",
            "learned_physical_fidelity": "NOT TESTED",
            "optimizer_steps": 0,
        },
    )


def test_projection_history_aliases_and_strict_network_state_dicts():
    fx = fixture(anchor=True)
    st = start_batch(fx, (300, 301))
    states = st.state[None].repeat(3, 1, 1).requires_grad_()
    feat = fx.wm.spec.robot.canonical(robot_state(states))
    assert feat.shape == (3, 2, 71)
    changed = states.detach().clone()
    changed[..., 71] += 2
    assert torch.equal(feat, fx.wm.spec.robot.canonical(robot_state(changed)))
    a = st.prev_action
    obs = analytic_observation(robot_state(st.state), a, st.xi, fx.q0)
    assert obs.shape == (2, 93)
    assert torch.equal(obs, analytic_observation(robot_state(changed[0]), a, st.xi, fx.q0))
    for width in (71, 83, 85):
        with pytest.raises(ValueError):
            robot_state(torch.zeros(3, 2, width))
    pc = type(fx.policy)(fx.pcfg).double()
    pc.load_state_dict(fx.policy.state_dict(), strict=True)
    from supermimic.policy.nets import Critic, CriticCfg

    cc = Critic(CriticCfg(hidden=(16, 8)), pcfg=fx.pcfg).double()
    cc.load_state_dict(fx.critic.state_dict(), strict=True)
    assert fx.policy.decoder.scale.numel() == 930 and fx.critic.mlp.net[0].in_features == 797
    goal = fx.lib.goal_window(st.frame, 10)
    goal = torch.cat((goal, torch.zeros(2, 9, dtype=goal.dtype)), -1)
    hist = flatten_history_term_major(obs[None].repeat(10, 1, 1))
    assert torch.equal(pc(hist, goal), fx.policy(hist, goal))
    assert fx.trainer.initialization_metadata()["optimizer_resume"] is False
    assert not fx.trainer.initialization_metadata()["training_ready"]
    # A self-consistent wider goal/critic pair must still fail the fixed HOI
    # dimension contract; merely comparing the two widths would accept it.
    wide_cfg = replace(fx.pcfg, goal_dim=1360)
    wide_policy = type(fx.policy)(wide_cfg)
    wide_critic = Critic(CriticCfg(hidden=(16, 8)), pcfg=wide_cfg)
    with pytest.raises(ValueError, match="dimensions"):
        HOIPolicyTrainer(
            fx.wm,
            wide_policy,
            wide_critic,
            fx.registry,
            fx.reward,
            replace(fx.trainer.cfg, future_frames=20, anchor_obs=False),
        )


def fill_replay(buffer, fx):
    st = start_batch(fx, (258, 548))
    s = st.state.float()
    for k in range(4):
        nxt = s.clone()
        nxt[:, 7] += 0.001
        frame = torch.tensor([258, 548]) + k
        clip = torch.full_like(frame, 7)
        labels = {name: torch.zeros(2, *shape) for name, shape in FORCE_SHAPES.items()}
        flags = torch.tensor([k == 3, k == 3])
        buffer.append(
            torch.arange(2),
            state=s,
            next_state=nxt,
            action_clean=torch.full((2, 29), float(k)),
            action_exec=torch.zeros(2, 29),
            xi=st.xi.float(),
            frame=frame,
            clip=clip,
            labels=labels,
            intent_ref=torch.zeros(2, 2),
            physical_failure=flags,
            timeout=torch.zeros_like(flags),
            clip_end=torch.zeros_like(flags),
        )
        s = nxt


def test_replay_metadata_and_original_row_selection_history():
    fx = fixture(torch.float32)
    hoi = HOIPolicyReplayBuffer(2, 32, "cpu", force_on=0.001, registry=fx.registry)
    fill_replay(hoi, fx)
    sample = hoi.sample_starts(64, 10)
    assert sample.clip_id.unique().tolist() == [7]
    assert torch.equal(sample.frame, sample.global_frame)
    assert sample.local_frame.max() <= 551
    assert sample.reference_sha256 == fx.registry.hashes(sample.clip_id)
    # Logged physical_failure of another action must NOT terminate counterfactual imagination.
    rows = torch.tensor([0, 1])
    t = torch.tensor([3, 3])
    last = hoi._gather_starts(rows, t, 10)
    assert last.hard_failure_step is None and HOIRolloutContext(
        fx.registry, last, 3
    ).lengths.tolist() == [3, 3]
    assert not hoi.start_mask(10)[:, 4].any()  # terminal state rows
    assert torch.equal(last.prev_action, torch.full((2, 29), 2.0))
    assert not last.hist_actions[0].any()
    # Explicit stale clip/history row cannot leak an action or observation across a clip.
    hoi.clip[0, 2] = 42
    history = hoi._gather_starts(torch.tensor([0]), torch.tensor([3]), 4)
    assert torch.equal(history.hist_states[0], history.hist_states[-1])
    assert not history.prev_action.any()
    # The original sampler retains exact RNG, selected rows and backfill output.
    oldclass = load_original("data/replay_buffer.py", "p5_original_replay").ReplayBuffer
    old = oldclass(2, 32, 71, 29, 33, "cpu")
    new = ReplayBuffer(2, 32, 71, 29, 33, "cpu")
    for key, value in vars(new).items():
        if torch.is_tensor(value) and hasattr(hoi, key) and getattr(hoi, key).shape == value.shape:
            value.copy_(getattr(hoi, key))
    new.s.copy_(hoi.s[:, :, :71])
    for key, value in vars(new).items():
        if torch.is_tensor(value):
            getattr(old, key).copy_(value)
    torch.manual_seed(202)
    a = old.sample_starts(64, 10)
    torch.manual_seed(202)
    b = new.sample_starts(64, 10)
    for key in vars(a):
        assert torch.equal(getattr(a, key), getattr(b, key)), key


def test_original_robot_imagination_values_noise_and_gradients_unchanged():
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    fx = fixture(torch.float32)
    st = start_batch(fx, (258, 548, 838))
    start = StartBatch(
        robot_state(st.state),
        st.xi,
        st.frame,
        st.prev_action,
        robot_state(st.hist_states),
        st.hist_actions,
    )
    wm = WorldModel(WorldModelCfg(hidden=(16, 16), term_hidden=(8, 8)), fx.heads, "cpu")
    # Nondegenerate orientation references avoid the separately documented old NaN.
    from supermimic.kinematics.rotations import quat_from_rotvec, quat_mul

    rotation = quat_from_rotvec(torch.tensor([0.08, -0.12, 0.1])).expand_as(fx.lib.body_quat_w)
    fx.lib.body_quat_w = quat_mul(rotation, fx.lib.body_quat_w)
    old = load_original("imagination/rollout.py", "p5_original_rollout")
    for noise in (False, True):
        fx.policy.zero_grad(set_to_none=True)
        torch.manual_seed(734)
        a = old.imagine(wm, fx.policy, fx.lib, start, 3, obs_noise=noise)
        a.rewards.sum().backward()
        ga = [p.grad.clone() for p in fx.policy.parameters()]
        fx.policy.zero_grad(set_to_none=True)
        torch.manual_seed(734)
        b = imagine(wm, fx.policy, fx.lib, start, 3, obs_noise=noise)
        b.rewards.sum().backward()
        gb = [p.grad.clone() for p in fx.policy.parameters()]
        for key in (
            "states",
            "actions",
            "rewards",
            "dhat",
            "survival",
            "goals",
            "e_root",
            "frames",
        ):
            assert torch.equal(getattr(a, key), getattr(b, key)), key
        assert a.reward_terms == b.reward_terms
        for x, y in zip(ga, gb):
            assert torch.isfinite(x).all() and torch.equal(x, y)
