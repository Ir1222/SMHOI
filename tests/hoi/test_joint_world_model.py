"""P3 numerical state tests; these are not measured physical training evidence."""

import mujoco
import numpy as np
import pytest
import torch

from supermimic.hoi.state_spec import JointStateSpec
from supermimic.kinematics.rotations import (
    matrix_from_quat,
    quat_apply,
    quat_from_rotvec,
    quat_from_yaw,
    quat_mul,
)


def states(n=12, dtype=torch.float64):
    generator = torch.Generator().manual_seed(23)
    s = torch.randn(n, 84, generator=generator, dtype=dtype)
    for start in (3, 74):
        s[:, start : start + 4] /= s[:, start : start + 4].norm(dim=-1, keepdim=True)
    return s


def test_canonical_robot_prefix_invariance_and_roundtrip():
    spec = JointStateSpec(dtype=torch.float64)
    s = states()
    torch.testing.assert_close(
        spec.canonical(s)[..., :71], spec.robot.canonical(s[..., :71]), atol=0, rtol=0
    )
    transformed = s.clone()
    yaw = quat_from_yaw(torch.linspace(-2.0, 2.0, len(s), dtype=s.dtype))
    shift = torch.tensor([2.0, -3.0, 0.0], dtype=s.dtype)
    for p, q, v in ((0, 3, 36), (71, 74, 78)):
        transformed[:, p : p + 3] = quat_apply(yaw, s[:, p : p + 3]) + shift
        transformed[:, q : q + 4] = quat_mul(yaw, s[:, q : q + 4])
        transformed[:, v : v + 3] = quat_apply(yaw, s[:, v : v + 3])
    torch.testing.assert_close(spec.canonical(transformed), spec.canonical(s), atol=1e-9, rtol=0)
    restored = spec.object_from_canonical(s[:, :71], spec.object_canonical(s))
    torch.testing.assert_close(restored[:, :3], s[:, 71:74], atol=1e-9, rtol=0)
    torch.testing.assert_close(restored[:, 7:], s[:, 78:], atol=1e-9, rtol=0)
    torch.testing.assert_close(
        matrix_from_quat(restored[:, 3:7]), matrix_from_quat(s[:, 74:78]), atol=1e-9, rtol=0
    )
    transformed = s.clone()
    transformed[:, 3:7] *= -1
    transformed[:, 74:78] *= -1
    torch.testing.assert_close(spec.canonical(transformed), spec.canonical(s), atol=1e-9, rtol=0)


def test_object_prior_against_mujoco_and_com():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><inertial pos="0 0 -.2" mass="2" diaginertia=".0334 .02167 .04167"/></body></worldbody></mujoco>'
    )
    spec = JointStateSpec(dtype=torch.float64)
    s = states(1)
    expected = s[0, 71:78].numpy().copy()
    velocity = s[0, 78:].numpy().copy()
    for _ in range(8):
        mujoco.mj_integratePos(model, expected, velocity, 0.02)
        robot = spec.robot.integrate(s[:, :71], torch.zeros(1, 70, dtype=s.dtype))
        s = spec.integrate(s, torch.zeros(1, 82, dtype=s.dtype))
        np.testing.assert_allclose(s[0, 71:78], expected, atol=1e-9)
        torch.testing.assert_close(s[:, :71], robot, atol=0, rtol=0)
    data = mujoco.MjData(model)
    data.qpos[:] = s[0, 71:78]
    data.qvel[:] = s[0, 78:]
    mujoco.mj_forward(model, data)
    measured = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, measured, 0)
    omega = quat_apply(s[:, 74:78], s[:, 81:84])
    offset = quat_apply(s[:, 74:78], torch.tensor([[0.0, 0.0, -0.2]], dtype=s.dtype))
    com_v = s[:, 78:81] + torch.cross(omega, offset, dim=-1)
    np.testing.assert_allclose(com_v[0], measured[3:], atol=1e-9)
    assert np.linalg.norm(measured[3:] - velocity[:3]) > 0.01


def test_residual_frames_order_and_gradients():
    spec = JointStateSpec(dtype=torch.float64)
    s = states(2)
    d = torch.zeros(2, 82, dtype=s.dtype, requires_grad=True)
    with torch.no_grad():
        d[:, 70:] = torch.tensor([0.1, 0.2, -0.3, -0.2, 0.1, 0.3, 0.5, -0.4, 0.1, 0.4, -0.3, 0.2])
    predicted = spec.integrate(s, d)
    # Right body increment must differ from the deliberately incorrect left product.
    correction = 0.02 * predicted[:, 81:84] + d[:, 79:82] * 0.02
    right = quat_mul(s[:, 74:78], quat_from_rotvec(correction))
    wrong = quat_mul(quat_from_rotvec(correction), s[:, 74:78])
    torch.testing.assert_close(predicted[:, 74:78], right, atol=1e-9, rtol=0)
    assert (right - wrong).abs().max() > 0.001
    loss = spec.error_vector(predicted, s).square().sum()
    loss.backward()
    assert torch.isfinite(d.grad).all() and d.grad[:, 70:].abs().sum() > 0
    with pytest.raises(ValueError):
        spec.integrate(s, torch.zeros(2, 88))
    with pytest.raises(ValueError):
        spec.canonical(s[:, :71])
