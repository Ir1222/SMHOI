"""Additional acceptance evidence, read-only checkpoints and scale diagnostics."""

from pathlib import Path

import pytest
import torch
from p5_fixtures import TEST_CFG, fixture, save_evidence, start_batch

from supermimic.hoi.reference import sha256
from supermimic.hoi.tracking_reward import HOI_TERM_NAMES, box_errors, robot_state
from supermimic.imagination.policy_trainer import frozen_parameters
from supermimic.kinematics.rotations import quat_from_rotvec, quat_mul
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.heads import ReferenceFrame


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_both_robot_orientation_terms_at_exact_match(dtype):
    fx = fixture(dtype)
    st = start_batch(fx, (300,))
    s = st.state.clone().requires_grad_()
    body = fx.heads.robot_bodies(robot_state(s.detach()))
    reference = ReferenceFrame(
        body["pos"],
        body["quat"],
        body["lin_vel"],
        body["ang_vel"],
        body["root_pos"],
        body["root_quat"],
    )
    ref = fx.registry.gather(st.clip_id, st.local_frame).box
    for sign in (1, -1):
        current = s.clone()
        current[:, 3:7] = current[:, 3:7] * sign
        terms = fx.reward(current, st.prev_action, st.prev_action, reference, ref)
        for key, beta in (("motion_global_root_ori", 0.5), ("motion_body_ori", 1.0)):
            gradient = torch.autograd.grad(terms[key].sum(), s, retain_graph=True)[0]
            assert torch.isfinite(gradient).all()
            tolerance = 5e-6 if dtype == torch.float32 else 1e-10
            torch.testing.assert_close(
                terms[key], s.new_tensor([0.02 * beta]), atol=tolerance, rtol=tolerance
            )
            # FK/reanchoring arithmetic can leave roundoff; no reward epsilon floor.
            assert gradient.abs().max() < tolerance


def test_replay_initial_action_constant_but_adjacent_imagined_actions_keep_graph():
    fx = fixture()
    st = start_batch(fx, (300,))
    captured = []
    original = fx.trainer.step_fn

    class Capture:
        def __call__(self, state, action, previous, xi, reference):
            output = original(state, action, previous, xi, reference)
            captured.append((state, action, previous, output))
            return output

        diagnostics = original.diagnostics

    fx.trainer.step_fn = Capture()
    with frozen_parameters(fx.wm, fx.critic):
        loss, _, ro, inputs, xi = fx.trainer.actor_loss(st)
        assert captured[0][2].requires_grad is False
        assert captured[1][2] is captured[0][1] or captured[1][2].grad_fn is not None
        first_action = captured[0][1]
        # Future reward reaches an earlier action through carried state and history.
        g = torch.autograd.grad(ro.rewards[2].sum(), first_action, retain_graph=True)[0]
        assert torch.isfinite(g).all() and g.norm() > 0
        survival_grad = torch.autograd.grad(
            ro.survival[-1].sum(), first_action, retain_graph=True
        )[0]
        features, goals, root_errors = inputs
        tail = fx.critic(features[-1], goals[-1], xi, root_errors[-1])
        tail_grad = torch.autograd.grad(tail.sum(), first_action, retain_graph=True)[0]
        assert torch.isfinite(survival_grad).all() and survival_grad.norm() > 0
        assert torch.isfinite(tail_grad).all() and tail_grad.norm() > 0
        # Previous-action regularization path is independently live.
        previous = captured[1][2]
        penalty = captured[1][3][2][6].sum()
        gp = torch.autograd.grad(penalty, previous, retain_graph=True)[0]
        torch.testing.assert_close(gp, 0.004 * (captured[1][1] - previous))
        loss.backward()
    save_evidence(
        "recurrence_paths",
        {
            "future_reward_to_first_action_norm": float(g.norm()),
            "survival_to_first_action_norm": float(survival_grad.norm()),
            "frozen_critic_tail_to_first_action_norm": float(tail_grad.norm()),
            "previous_action_regularizer_derivative": "exact original equation",
            "optimizer_steps": 0,
        },
    )


def test_external_checkpoint_strict_policy_critic_loading_and_alias_gradient():
    path = Path("/home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt")
    if not path.exists():
        pytest.skip("External original checkpoint unavailable; compatibility NOT TESTED")
    expected = "16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7"
    assert sha256(path) == expected
    ck = torch.load(path, map_location="cpu", weights_only=False)
    pcfg = PolicyCfg(**ck["config"]["policy"])
    ccfg = CriticCfg(**ck["config"]["critic"])
    actor, critic = Policy(pcfg), Critic(ccfg, pcfg=pcfg)
    actor.load_state_dict(ck["policy"], strict=True)
    critic.load_state_dict(ck["critic"], strict=True)
    assert actor.decoder.scale.numel() == 930
    assert actor.encoders["motion"].scale.numel() == 680
    assert actor.decoder.mlp.net[-1].out_features == 29
    assert critic.mlp.net[0].in_features == 788
    fx = fixture(torch.float32)
    st = start_batch(fx, (300,))
    state = st.state.clone().requires_grad_()
    ref = fx.registry.gather(st.clip_id, st.local_frame).robot
    with frozen_parameters(critic):
        value = critic(
            fx.wm.spec.robot.canonical(robot_state(state)),
            fx.lib.goal_window(st.frame, 10),
            fx.wm.spec.normalize_xi(st.xi),
            fx.heads.root_error_features(robot_state(state), ref),
        )
        gradient = torch.autograd.grad(value.sum(), state)[0]
    assert torch.count_nonzero(gradient[:, 71:]) == 0
    # Tensor keys/shapes and normalization buffers survive strict loading exactly.
    assert set(actor.state_dict()) == set(ck["policy"])
    assert all(torch.equal(v, ck["policy"][k]) for k, v in actor.state_dict().items())
    assert all(torch.equal(v, ck["critic"][k]) for k, v in critic.state_dict().items())
    assert sha256(path) == expected
    save_evidence(
        "checkpoint",
        {
            "path": str(path),
            "sha256": expected,
            "policy_strict": True,
            "critic_strict": True,
            "dimensions": [930, 680, 29, 788],
            "critic_direct_box_gradient": 0,
            "optimizer_or_sampler_state_loaded": False,
            "training_provenance": "source has no embedded training Git/dataset SHA",
        },
    )


def test_all_frame_test_configuration_scale_audit():
    fx = fixture()
    errors, contributions, state_gradients = [], [], []
    for lo in range(0, 840, 84):
        st = start_batch(fx, tuple(range(lo, lo + 84)))
        s = st.state.clone()
        # Explicit controlled deviations; these are neither learned rollouts nor final weights.
        delta = quat_from_rotvec(s.new_tensor([[0.02, -0.01, 0.015]]).expand(84, 3))
        s[:, 74:78] = quat_mul(s[:, 74:78], delta)
        s[:, 78:81] += s.new_tensor([0.03, -0.02, 0.01])
        s[:, 81:84] += s.new_tensor([0.02, 0.03, -0.01])
        s.requires_grad_()
        ref = fx.registry.gather(st.clip_id, st.local_frame)
        terms = fx.reward(s, st.prev_action, st.prev_action, ref.robot, ref.box)
        errors.append(box_errors(s, ref.box).detach())
        contributions.append(torch.stack([terms[k] for k in HOI_TERM_NAMES], -1).detach())
        gradients = []
        for name in ("robot_positive", "box_total", "proximity_weighted", "total"):
            grad = torch.autograd.grad(terms[name].sum(), s, retain_graph=True)[0]
            assert torch.isfinite(grad).all()
            gradients.append(grad.norm(dim=-1))
        state_gradients.append(torch.stack(gradients, -1))

    def stats(x):
        return {
            "mean": x.mean(0).tolist(),
            "min": x.min(0).values.tolist(),
            "p95": x.quantile(0.95, dim=0).tolist(),
            "max": x.max(0).values.tolist(),
        }

    save_evidence(
        "scale_audit_840",
        {
            "configuration_status": "TEST ONLY; no final training coefficients selected",
            "configuration": TEST_CFG.metadata(),
            "frames": 840,
            "fixture": "reference poses plus explicit small root/Box pose/twist perturbations; no dynamics rollout",
            "box_squared_physical_errors": stats(torch.cat(errors)),
            "contribution_order": list(HOI_TERM_NAMES),
            "weighted_contributions": stats(torch.cat(contributions)),
            "state_gradient_order": ["robot_positive", "box_total", "proximity_weighted", "total"],
            "state_gradient_norms": stats(torch.cat(state_gradients)),
        },
    )
