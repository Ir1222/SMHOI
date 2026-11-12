import math
from dataclasses import replace

import pytest
import torch
from p5_fixtures import TEST_CFG, fixture, start_batch

from supermimic.hoi.tracking_reward import (
    BOX_TERM_NAMES,
    HOI_TERM_NAMES,
    BoxReference,
    HOIRewardCfg,
    box_errors,
)
from supermimic.kinematics.rotations import quat_apply_inverse, quat_from_rotvec


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_four_equations_maxima_sigma_monotonicity_world_frames(dtype):
    fx = fixture(dtype)
    s = start_batch(fx, (300,)).state
    q = quat_from_rotvec(s.new_tensor([[0.2, -0.3, 0.4]]))
    s[:, 74:78] = q
    world = s.new_tensor([[0.3, -0.5, 0.2]])
    s[:, 81:84] = quat_apply_inverse(q, world)
    ref = BoxReference(
        s[:, 71:74].clone(), q.clone(), s[:, 78:81].clone(), world, torch.zeros(1, dtype=torch.bool)
    )
    atol, rtol = (5e-6, 5e-6) if dtype == torch.float32 else (1e-10, 1e-8)
    for key, beta in zip(BOX_TERM_NAMES, TEST_CFG.box_weights):
        torch.testing.assert_close(
            fx.reward.box_terms(s, ref)[key], s.new_tensor([0.02 * beta]), atol=atol, rtol=rtol
        )
    for j, key in enumerate(BOX_TERM_NAMES):
        scores = []
        for multiplier in (0.0, 1.0, 2.0):
            p = s.clone()
            error = TEST_CFG.box_sigmas[j] * multiplier
            if j == 0:
                p[:, 71] += error
            elif j == 1:
                p[:, 74:78] = quat_from_rotvec(s.new_tensor([[error, 0, 0]]))
            elif j == 2:
                p[:, 78] += error
            else:
                p[:, 81:84] = quat_apply_inverse(q, world + s.new_tensor([[error, 0, 0]]))
            r = ref if j != 1 else replace(ref, quaternion=s.new_tensor([[1.0, 0, 0, 0]]))
            val = fx.reward.box_terms(p, r)[key]
            expected = 0.02 * TEST_CFG.box_weights[j] * math.exp(-(multiplier**2))
            torch.testing.assert_close(val, s.new_tensor([expected]), atol=atol, rtol=rtol)
            scores.append(val)
        assert scores[0] > scores[1] > scores[2]
    # Different orientations/body vectors, identical world omega -> zero error.
    ref2 = replace(ref, quaternion=s.new_tensor([[1.0, 0, 0, 0]]))
    torch.testing.assert_close(box_errors(s, ref2)[:, 3], s.new_zeros(1), atol=atol, rtol=rtol)
    wrong = s.clone()
    wrong[:, 81:84] = world
    assert box_errors(wrong, ref2)[0, 3] > 0.001
    s.requires_grad_()
    err = box_errors(s, replace(ref, angular_world=world + 0.1))[:, 3].sum()
    grad = torch.autograd.grad(err, s)[0]
    assert grad[:, 74:78].norm() > 0 and grad[:, 81:84].norm() > 0


def test_total_regularization_once_and_tracking_off():
    fx = fixture()
    st = start_batch(fx, (259, 549))
    s = st.state.clone().requires_grad_()
    ref = fx.registry.gather(st.clip_id, st.local_frame + 1)
    a = torch.full((2, 29), 0.2, dtype=s.dtype, requires_grad=True)
    prev = torch.full_like(a, -0.1, requires_grad=True)
    terms = fx.reward(s, a, prev, ref.robot, ref.box)
    torch.testing.assert_close(
        sum(terms[k] for k in HOI_TERM_NAMES), terms["total"], atol=1e-10, rtol=1e-8
    )
    torch.testing.assert_close(
        terms["total"],
        terms["robot_positive"]
        + terms["box_total"]
        + terms["proximity_weighted"]
        + terms["regularization"],
        atol=1e-10,
        rtol=1e-8,
    )
    assert terms["proximity"][1] == 0 and terms["box_total"][1] > 0
    ga, gp = torch.autograd.grad(terms["action_rate_l2"].sum(), (a, prev))
    torch.testing.assert_close(ga, 2 * 0.02 * (-0.1) * (a - prev))
    torch.testing.assert_close(gp, -ga)
    # Original limit penalty is linear (not squared), preserving configured weights.
    s2 = s.detach().clone()
    s2[:, 7] = fx.heads.soft_limits[0, 1] + 0.2
    changed = fx.reward(s2, a, prev, ref.robot, ref.box)
    torch.testing.assert_close(
        changed["joint_limit"] - terms["joint_limit"],
        s.new_full((2,), -0.008),
        atol=1e-10,
        rtol=1e-8,
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("box_sigmas", (1, 2, 3)),
        ("box_weights", (1, 0, 1, 1)),
        ("box_sigmas", (1, float("nan"), 1, 1)),
        ("sigma_pos", 0),
        ("beta_c", 0),
        ("beta_c", float("inf")),
        ("dt", -1),
    ],
)
def test_required_configuration(name, value):
    with pytest.raises(ValueError):
        replace(TEST_CFG, **{name: value})
    with pytest.raises(TypeError):
        HOIRewardCfg()


def test_named_ablation_and_invalid_quaternion():
    cfg = replace(TEST_CFG, beta_c=0, proximity_ablation="proximity-off")
    assert cfg.metadata()["beta_c"] == 0
    fx = fixture()
    st = start_batch(fx, (260,))
    r = fx.registry.gather(st.clip_id, st.local_frame).box
    for value in (0.0, float("nan")):
        bad = st.state.clone()
        bad[:, 74:78] = value
        with pytest.raises(ValueError):
            box_errors(bad, r)
