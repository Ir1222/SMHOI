"""P4 geometry/gating and state gradients; no imagination or optimizer steps."""

import math

import mujoco
import numpy as np
import pytest
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.mdp.rewards import TrackingRewardCfg, tracking_reward_terms
from supermimic.hoi.rewards import (
    ProximityReward,
    combine_tracking_and_proximity,
    proximity_from_points,
    reference_contact_at,
)
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_from_rotvec


@pytest.fixture
def geometry():
    model = load_g1_model()
    fk = G1Kinematics(model, dtype=torch.float64)
    reward = ProximityReward(
        fk,
        ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        [[0, 0.17, 0.14], [0, -0.17, 0.14]],
        [[0.05, 0, 0], [0.05, 0, 0]],
        sigma_pos=0.0423,
    )
    state = torch.zeros(2, 84, dtype=torch.float64)
    state[:, :36] = torch.tensor(model.qpos0)
    state[:, 2] = 0.8
    state[:, 71:74] = torch.tensor([0.3, 0.04, 0.7])
    state[:, 74] = 1
    state[1, 7:36] = torch.linspace(-0.1, 0.1, 29)
    return model, reward, state


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_gate_monotonicity_independent_targets_and_zero_distance(dtype):
    distances = torch.tensor([0.0, 0.01, 0.03, 0.1, 1.0], dtype=dtype)
    eef = torch.zeros(5, 2, 3, dtype=dtype)
    eef[:, 0, 0] = distances
    eef[:, 1, 1] = distances * 2
    eef.requires_grad_()
    target = torch.zeros_like(eef, requires_grad=True)
    on = torch.ones(5, dtype=torch.bool)
    result = proximity_from_points(eef, target, on, sigma_pos=0.05)
    expected = torch.tensor(
        [(math.exp(-d / 0.05) + math.exp(-2 * d / 0.05)) / 2 for d in distances.tolist()],
        dtype=dtype,
    )
    torch.testing.assert_close(result["proximity"], expected)
    assert torch.all(result["proximity"][1:] < result["proximity"][:-1])
    assert torch.all(result["proximity"] > 0)
    result["proximity"].sum().backward()
    assert torch.isfinite(eef.grad).all() and torch.isfinite(target.grad).all()
    assert torch.equal(eef.grad[0], torch.zeros_like(eef.grad[0]))
    eef.grad = target.grad = None
    off = proximity_from_points(eef, target, ~on, sigma_pos=0.05)
    off["proximity"].sum().backward()
    assert torch.count_nonzero(off["proximity"]) == 0
    assert torch.count_nonzero(eef.grad) == torch.count_nonzero(target.grad) == 0


def test_exact_reference_boundaries_and_shuffled_batch():
    labels = torch.zeros(840, dtype=torch.bool)
    labels[260:550] = True
    original = labels.clone()
    indices = torch.tensor([[550, 260, 839], [259, 549, 0]])
    assert torch.equal(
        reference_contact_at(labels, indices),
        torch.tensor([[False, True, False], [False, True, False]]),
    )
    assert torch.equal(labels, original)
    # Next-state reward for the transition 259 -> 260 activates exactly at 260.
    assert not reference_contact_at(labels, torch.tensor(259))
    assert reference_contact_at(labels, torch.tensor(259) + 1)
    for bad in (-1, 840):
        with pytest.raises(IndexError):
            reference_contact_at(labels, torch.tensor([bad]))
    with pytest.raises(ValueError):
        reference_contact_at(labels.float(), indices)
    with pytest.raises(ValueError):
        reference_contact_at(labels, indices.float())


def test_rotated_targets_named_eef_and_current_box_pose(geometry):
    model, reward, state = geometry
    state[:, 74:78] = quat_from_rotvec(
        torch.tensor([[0.0, 0.0, math.pi / 2], [0.2, -0.3, 0.4]], dtype=state.dtype)
    )
    result = reward(state, torch.tensor([True, False]))
    data = mujoco.MjData(model)
    for b in range(2):
        data.qpos[:] = state[b, :36].numpy()
        mujoco.mj_kinematics(model, data)
        for i, name in enumerate(reward.wrist_body_names):
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, data.xquat[model.body(name).id])
            eef = data.xpos[model.body(name).id] + rotation.reshape(3, 3) @ np.array([0.05, 0, 0])
            np.testing.assert_allclose(result["eef_pos"][b, i], eef, atol=1e-12, rtol=0)
            mujoco.mju_quat2Mat(rotation, state[b, 74:78].numpy())
            target = (
                state[b, 71:74].numpy() + rotation.reshape(3, 3) @ reward.target_root[i].numpy()
            )
            np.testing.assert_allclose(result["target_pos"][b, i], target, atol=1e-12, rtol=0)
    shifted = state.clone()
    shifted[:, 71:74] += torch.tensor([0.12, -0.08, 0.03])
    assert not torch.allclose(
        reward(shifted, torch.ones(2, dtype=torch.bool))["distance"], result["distance"]
    )


def test_batch_independence_common_frame_and_state_untouched(geometry):
    _, reward, state = geometry
    before = state.clone()
    labels = torch.tensor([True, False])
    batch = reward(state, labels)
    for i in range(2):
        single = reward(state[i], labels[i])
        for key in batch:
            torch.testing.assert_close(batch[key][i], single[key])
    shifted = state.clone()
    shifted[:, :3] += 7
    shifted[:, 71:74] += 7
    torch.testing.assert_close(
        batch["distance"], reward(shifted, labels)["distance"], atol=1e-12, rtol=0
    )
    assert torch.equal(before, state)
    multi = state.expand(3, -1, -1).clone()
    assert reward(multi, labels.expand(3, -1))["proximity"].shape == (3, 2)


def test_joint_translation_and_rotation_gradients_finite_difference(geometry):
    model, reward, state = geometry
    joints = [
        int(model.jnt_qposadr[model.joint(f"{s}_{j}_joint").id])
        for s in ("left", "right")
        for j in ("shoulder_pitch", "elbow", "wrist_yaw")
    ]

    def evaluate(x):
        s = state[:1].clone()
        s[:, joints] = x[:6]
        s[:, 71:74] = x[6:9]
        s[:, 74:78] = quat_from_rotvec(x[9:12])
        return reward(s, torch.ones(1, dtype=torch.bool))["proximity"].sum()

    x = torch.tensor(
        [0.1, -0.4, 0.2, -0.1, -0.3, -0.2, 0.2, 0.03, 0.65, 0.2, -0.1, 0.3],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(evaluate, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)
    grad = torch.autograd.grad(evaluate(x), x)[0]
    for part in (grad[:3], grad[3:6], grad[6:9], grad[9:]):
        assert torch.isfinite(part).all() and part.abs().sum() > 1e-8
    fd = []
    for i in range(len(x)):
        delta = torch.zeros_like(x)
        delta[i] = 1e-6
        fd.append((evaluate(x + delta) - evaluate(x - delta)) / 2e-6)
    torch.testing.assert_close(grad, torch.stack(fd), atol=1e-5, rtol=1e-3)


def test_sensor_independence_and_no_velocity_reward_input(geometry, monkeypatch):
    from supermimic.hoi import contact

    def forbidden(*args, **kwargs):
        raise AssertionError("Physical measurement called from reward")

    for name in ("mj_step", "mj_forward", "mj_contactForce", "mj_geomDistance"):
        monkeypatch.setattr(mujoco, name, forbidden)
    for name, value in vars(contact).copy().items():
        if callable(value) and getattr(value, "__module__", None) == contact.__name__:
            monkeypatch.setattr(contact, name, forbidden)
    _, reward, state = geometry
    state.requires_grad_()
    values = reward(state, torch.tensor([True, False]))["proximity"]
    values.sum().backward()
    assert torch.isfinite(state.grad).all()
    assert state.grad[0, 7:36].abs().sum() > 0
    assert state.grad[0, 71:78].abs().sum() > 0
    assert torch.count_nonzero(state.grad[1]) == 0
    assert torch.count_nonzero(state.grad[:, 36:71]) == 0
    assert torch.count_nonzero(state.grad[:, 78:84]) == 0


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scale(sigma):
    with pytest.raises(ValueError, match="sigma"):
        proximity_from_points(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 3),
            torch.ones(1, dtype=torch.bool),
            sigma_pos=sigma,
        )


def test_invalid_geometry_labels_and_state(geometry):
    _, reward, state = geometry
    with pytest.raises(ValueError, match="bool"):
        reward(state, torch.ones(2))
    with pytest.raises(ValueError, match="bool"):
        reward(state, torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="state84"):
        reward(state[:, :71], torch.ones(2, dtype=torch.bool))
    bad = state.clone()
    bad[:, 74:78] = 0
    with pytest.raises(ValueError, match="unit"):
        reward(bad, torch.ones(2, dtype=torch.bool))
    bad[:, 74] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        reward(bad, torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="nonempty"):
        proximity_from_points(
            torch.zeros(2, 0, 3),
            torch.zeros(2, 0, 3),
            torch.ones(2, dtype=torch.bool),
            sigma_pos=0.1,
        )


def test_original_robot_tracking_and_additive_object_term():
    dtype = torch.float64
    pos = torch.zeros(2, 3, 3, dtype=dtype)
    quat = torch.zeros(2, 3, 4, dtype=dtype)
    quat[..., 0] = 1
    zeros = torch.zeros_like(pos)
    action = torch.zeros(2, 29, dtype=dtype)
    cfg = TrackingRewardCfg()
    kwargs = {
        "robot_body_pos_w": pos,
        "robot_body_quat_w": quat,
        "robot_body_lin_vel_w": zeros,
        "robot_body_ang_vel_w": zeros,
        "ref_body_pos_w": pos,
        "ref_body_quat_w": quat,
        "ref_body_lin_vel_w": zeros,
        "ref_body_ang_vel_w": zeros,
        "anchor_index": 0,
        "action": action,
        "prev_action": action,
        "joint_pos": action,
        "soft_limits": torch.tensor([[-1.0, 1.0]] * 29, dtype=dtype),
        "cfg": cfg,
    }
    original = tracking_reward_terms(**kwargs)
    torch.testing.assert_close(
        original["total"], torch.full((2,), 0.1, dtype=dtype), atol=1e-12, rtol=0
    )
    moved = pos.clone()
    moved[:, :, 2] = 0.3
    perturbed = tracking_reward_terms(**{**kwargs, "robot_body_pos_w": moved})
    expected = 0.02 * (1.5 * math.exp(-1) + 3.5)
    torch.testing.assert_close(
        perturbed["total"], torch.full((2,), expected, dtype=dtype), atol=1e-12, rtol=0
    )
    obj = torch.tensor([0.03, 0.06], dtype=dtype, requires_grad=True)
    proximity = torch.tensor([0.7, 0.0], dtype=dtype, requires_grad=True)
    total = combine_tracking_and_proximity(
        robot=original["total"],
        object=obj,
        proximity=proximity,
        regularization=torch.zeros(2, dtype=dtype),
        robot_weight=1.0,
        object_weight=2.0,
        contact_weight=0.02,
    )
    torch.testing.assert_close(total, original["total"] + 2 * obj + 0.02 * proximity)
    gradients = torch.autograd.grad(total.sum(), (obj, proximity))
    torch.testing.assert_close(gradients[0], torch.full_like(obj, 2.0))
    torch.testing.assert_close(gradients[1], torch.full_like(proximity, 0.02))
    for key in original:
        assert torch.equal(original[key], tracking_reward_terms(**kwargs)[key])


def test_single_target_is_explicit_not_best_hand(geometry):
    _, reward, state = geometry
    single = ProximityReward(
        reward.fk,
        (reward.wrist_body_names[0],),
        reward.target_root[:1],
        reward.eef_offset[:1],
        sigma_pos=reward.sigma_pos,
    )
    on = torch.ones(2, dtype=torch.bool)
    torch.testing.assert_close(
        single(state, on)["proximity"], reward(state, on)["per_target"][:, 0]
    )
