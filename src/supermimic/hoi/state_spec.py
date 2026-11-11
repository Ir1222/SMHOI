"""Joint physical contracts. Compose the original robot spec without reordering it."""

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_from_matrix

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

ROBOT = slice(0, 71)
OBJECT = slice(71, 84)
OBJECT_FIELDS = {
    "position": slice(0, 3),
    "quaternion": slice(3, 7),
    "linear_world": slice(7, 10),
    "angular_body": slice(10, 13),
}
ROBOT_FEATURES = {
    "height": slice(0, 1),
    "rotation6": slice(1, 7),
    "linear_heading": slice(7, 10),
    "angular_body": slice(10, 13),
    "joint_position": slice(13, 42),
    "joint_velocity": slice(42, 71),
}
ROBOT_RESIDUALS = {
    "dv_heading": slice(0, 3),
    "dw_body": slice(3, 6),
    "djoint_velocity": slice(6, 35),
    "dp_heading": slice(35, 38),
    "dtheta_body": slice(38, 41),
    "djoint_position": slice(41, 70),
}
OBJECT_RESIDUALS = {
    "dv_heading": slice(0, 3),
    "dw_body": slice(3, 6),
    "dp_heading": slice(6, 9),
    "dtheta_body": slice(9, 12),
}
STATE_DIM, FEAT_DIM, DELTA_DIM, FORCE_DIM = 84, 86, 82, 6


@dataclass(frozen=True)
class ObjectSpecCfg:
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0)
    out_scale: tuple[float, ...] = (0.3, 0.5, 0.01, 0.02)
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0)


def _scales(values, sizes, device, dtype):
    if len(values) != len(sizes) or any(not 0 < v < float("inf") for v in values):
        raise ValueError("Scales must be finite, positive and match named field groups")
    return torch.cat(
        [torch.full((n,), v, device=device, dtype=dtype) for v, n in zip(values, sizes)]
    )


def _width(tensor, size):
    if tensor.shape[-1] != size:
        raise ValueError(f"Expected last dimension {size}, got {tensor.shape}")


class JointStateSpec:
    """Positions environment-local; linear velocities at roots/world, angular body.

    Object features: root-relative heading position3, heading-relative rotation6,
    absolute object linear velocity in heading3, object body angular3. This is NOT
    a moving-frame velocity derivative; no robot velocity is subtracted.
    """

    def __init__(self, robot_cfg=None, object_cfg=None, device="cpu", dtype=torch.float32):
        self.robot = StateSpec(robot_cfg or StateSpecCfg(), device, dtype)
        self.object_cfg = object_cfg or ObjectSpecCfg()
        self.dt = self.robot.dt
        self.object_in_scale = _scales(self.object_cfg.in_scale, (3, 6, 3, 3), device, dtype)
        self.object_out_scale = _scales(self.object_cfg.out_scale, (3, 3, 3, 3), device, dtype)
        self.object_err_scale = _scales(self.object_cfg.err_scale, (3, 3, 3, 3), device, dtype)

    def normalize_xi(self, xi):
        _width(xi, 33)
        return self.robot.normalize_xi(xi)

    def object_canonical(self, state):
        _width(state, STATE_DIM)
        p, q, _, _, _, _ = self.robot.split(state[..., ROBOT])
        obj = state[..., OBJECT]
        heading = heading_quat(quat_normalize(q))
        relative = quat_mul(quat_conjugate(heading), quat_normalize(obj[..., 3:7]))
        return (
            torch.cat(
                (
                    quat_apply_inverse(heading, obj[..., :3] - p),
                    rot6d_from_quat(relative),
                    quat_apply_inverse(heading, obj[..., 7:10]),
                    obj[..., 10:13],
                ),
                -1,
            )
            / self.object_in_scale
        )

    def canonical(self, state):
        _width(state, STATE_DIM)
        return torch.cat(
            (self.robot.canonical(state[..., ROBOT]), self.object_canonical(state)), -1
        )

    def object_from_canonical(self, robot_state, features):
        """Inverse using the supplied robot root/heading; for valid rotation6 features."""
        _width(robot_state, 71)
        _width(features, 15)
        x = features * self.object_in_scale
        heading = heading_quat(quat_normalize(robot_state[..., 3:7]))
        cols = x[..., 3:9].reshape(*x.shape[:-1], 3, 2)
        first = torch.nn.functional.normalize(cols[..., 0], dim=-1)
        second = cols[..., 1] - (first * cols[..., 1]).sum(-1, keepdim=True) * first
        second = torch.nn.functional.normalize(second, dim=-1)
        matrix = torch.stack((first, second, torch.cross(first, second, dim=-1)), -1)
        quaternion = quat_mul(heading, quat_from_matrix(matrix))
        return torch.cat(
            (
                robot_state[..., :3] + quat_apply(heading, x[..., :3]),
                quat_normalize(quaternion),
                quat_apply(heading, x[..., 9:12]),
                x[..., 12:15],
            ),
            -1,
        )

    def integrate(self, state, delta):
        _width(state, STATE_DIM)
        _width(delta, DELTA_DIM)  # Auxiliary forces must never enter integration.
        robot = self.robot.integrate(state[..., ROBOT], delta[..., :70])
        obj = state[..., OBJECT]
        q = quat_normalize(obj[..., 3:7])
        heading = heading_quat(quat_normalize(state[..., 3:7]))
        d = delta[..., 70:] * self.object_out_scale
        v = obj[..., 7:10] + quat_apply(heading, d[..., :3])
        w = obj[..., 10:13] + d[..., 3:6]
        # Object prior always follows physical dt; robot's legacy option stays local.
        p = obj[..., :3] + self.dt * v + quat_apply(heading, d[..., 6:9])
        q = quat_normalize(quat_mul(q, quat_from_rotvec(self.dt * w + d[..., 9:12])))
        return torch.cat((robot, p, q, v, w), -1)

    def object_error_vector(self, prediction, target):
        _width(prediction, STATE_DIM)
        _width(target, STATE_DIM)
        a, b = prediction[..., OBJECT], target[..., OBJECT]
        rotation = rotvec_from_quat(
            quat_mul(quat_conjugate(quat_normalize(b[..., 3:7])), quat_normalize(a[..., 3:7]))
        )
        return (
            torch.cat(
                (
                    a[..., :3] - b[..., :3],
                    rotation,
                    a[..., 7:10] - b[..., 7:10],
                    a[..., 10:13] - b[..., 10:13],
                ),
                -1,
            )
            / self.object_err_scale
        )

    def error_vector(self, prediction, target):
        return torch.cat(
            (
                self.robot.error_vector(prediction[..., ROBOT], target[..., ROBOT]),
                self.object_error_vector(prediction, target),
            ),
            -1,
        )

    def normalized_sq_error(self, prediction, target):
        return self.error_vector(prediction, target).square().mean(-1)
