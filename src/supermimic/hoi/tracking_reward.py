"""P5 Scheme A reward. Explicit coefficients; no sensors or training defaults."""

import math
from dataclasses import asdict, dataclass

import torch

from supermimic.hoi.rewards import ProximityReward, combine_tracking_and_proximity
from supermimic.hoi.state_spec import ROBOT
from supermimic.imagination.compiled import REWARD_TERM_NAMES
from supermimic.kinematics.rotations import quat_apply, quat_conjugate, quat_mul

BOX_TERM_NAMES = ("box_pos", "box_ori", "box_lin_vel", "box_ang_vel")
HOI_TERM_NAMES = REWARD_TERM_NAMES + BOX_TERM_NAMES + ("proximity_weighted",)
REWARD_VERSION = "scheme-a-squared-log-next-frame-v1"


def robot_state(state):
    """Named physical projection, preserving every leading time/batch axis."""
    if state.shape[-1] != 84:
        raise ValueError("Expected physical84 for robot71 projection")
    return state[..., ROBOT]


def safe_quat_box_minus(q1, q2):
    """Original selected formula; guard only its unused zero denominator.

    Finite first derivative at agreement. The shortest-log pi cut is not smooth.
    Kept HOI-local: neither the original robot default nor dependencies change.
    """
    q = quat_mul(q1, quat_conjugate(q2))
    q = q * (1.0 - 2.0 * (q[..., :1] < 0.0))
    mag = torch.linalg.vector_norm(q[..., 1:], dim=-1)
    half = torch.atan2(mag, q[..., 0])
    angle = 2.0 * half
    large = angle.abs() > 1e-6
    safe_angle = torch.where(large, angle, torch.ones_like(angle))
    ratio = torch.where(large, torch.sin(half) / safe_angle, 0.5 - angle * angle / 48)
    return q[..., 1:] / ratio[..., None]


def safe_squared_orientation_error(q1, q2):
    return safe_quat_box_minus(q1, q2).square().sum(-1)


def unit_quaternion_copy(q):
    if not q.is_floating_point() or q.shape[-1] != 4 or not torch.isfinite(q).all():
        raise ValueError("Expected finite wxyz quaternion")
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    if torch.any((norm - 1).abs() > 1e-5):
        raise ValueError("Expected unit quaternion within 1e-5")
    return q / norm


@dataclass(frozen=True)
class HOIRewardCfg:
    box_sigmas: tuple[float, float, float, float]  # m, rad, m/s, rad/s
    box_weights: tuple[float, float, float, float]
    sigma_pos: float  # m, unchanged P4 distance scale
    beta_c: float
    dt: float
    proximity_ablation: str | None = None

    def __post_init__(self):
        for name in ("box_sigmas", "box_weights"):
            values = tuple(getattr(self, name))
            if len(values) != 4 or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"{name} requires four finite positive values")
            object.__setattr__(self, name, values)
        if any(not math.isfinite(v) or v <= 0 for v in (self.sigma_pos, self.dt)):
            raise ValueError("sigma_pos and dt must be finite and positive")
        if self.proximity_ablation not in (None, "proximity-off"):
            raise ValueError("Unknown proximity ablation")
        if not math.isfinite(self.beta_c) or self.beta_c < 0:
            raise ValueError("beta_c must be finite and nonnegative")
        if (self.beta_c == 0) != (self.proximity_ablation == "proximity-off"):
            raise ValueError("Zero beta_c requires the named proximity-off ablation")

    def metadata(self):
        return {"version": REWARD_VERSION, **asdict(self)}


@dataclass(frozen=True)
class BoxReference:
    position: torch.Tensor
    quaternion: torch.Tensor
    linear_world: torch.Tensor
    angular_world: torch.Tensor
    contact: torch.Tensor


def box_errors(state, reference: BoxReference):
    robot_state(state)  # width guard
    if not state.is_floating_point() or not torch.isfinite(state).all():
        raise ValueError("Expected finite physical state")
    batch = state.shape[:-1]
    for value, width in (
        (reference.position, 3),
        (reference.quaternion, 4),
        (reference.linear_world, 3),
        (reference.angular_world, 3),
    ):
        if (
            value.shape != (*batch, width)
            or value.dtype != state.dtype
            or value.device != state.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Misaligned Box reference shape/dtype/device or nonfinite values")
    q = unit_quaternion_copy(state[..., 74:78])
    q_ref = unit_quaternion_copy(reference.quaternion)
    return torch.stack(
        (
            (state[..., 71:74] - reference.position).square().sum(-1),
            safe_squared_orientation_error(q_ref, q),
            (state[..., 78:81] - reference.linear_world).square().sum(-1),
            (quat_apply(q, state[..., 81:84]) - reference.angular_world).square().sum(-1),
        ),
        -1,
    )


class HOITrackingReward:
    """Reuse robot equations and P4 byte-for-byte; add four physical Box scores."""

    def __init__(self, heads, proximity: ProximityReward, cfg: HOIRewardCfg):
        if heads.reward_cfg.dt != cfg.dt or proximity.sigma_pos != cfg.sigma_pos:
            raise ValueError("Robot/Box/P4 configuration timestep or sigma mismatch")
        self.heads, self.proximity, self.cfg = heads, proximity, cfg

    def box_terms(self, state, reference):
        errors = box_errors(state, reference)
        scores = [
            self.cfg.dt * beta * torch.exp(-errors[..., j] / sigma**2)
            for j, (sigma, beta) in enumerate(zip(self.cfg.box_sigmas, self.cfg.box_weights))
        ]
        return dict(zip(BOX_TERM_NAMES, scores))

    def __call__(self, state, action, previous_action, robot_reference, box_reference, bodies=None):
        robot = self.heads.reward(
            robot_state(state),
            action,
            previous_action,
            robot_reference,
            bodies,
            orientation_error=safe_squared_orientation_error,
        )
        box = self.box_terms(state, box_reference)
        proximity = self.proximity(state, box_reference.contact)
        box_total = sum(box.values())
        weighted = self.cfg.dt * self.cfg.beta_c * proximity["proximity"]
        total = combine_tracking_and_proximity(
            robot=robot["total"],
            object=box_total,
            proximity=proximity["proximity"],
            regularization=torch.zeros_like(robot["total"]),
            robot_weight=1,
            object_weight=1,
            contact_weight=self.cfg.dt * self.cfg.beta_c,
        )
        regularization = robot["action_rate_l2"] + robot["joint_limit"]
        return {
            **robot,
            **box,
            **proximity,
            "proximity_weighted": weighted,
            "total": total,
            "box_total": box_total,
            "regularization": regularization,
            "robot_positive": sum(robot[k] for k in REWARD_TERM_NAMES[:6]),
        }

    @property
    def max_step_reward(self):
        return self.heads.reward_cfg.max_step_reward + self.cfg.dt * (
            sum(self.cfg.box_weights) + self.cfg.beta_c
        )
