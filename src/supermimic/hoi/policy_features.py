"""Versioned Actor B / Critic B+ features; no force/contact observation inputs.

Physical state is always 84D. Public history functions accept (N, 10, D),
oldest first. The rollout adapter accepts the existing Replay (10, N, 84)
layout, so Robot and Object necessarily use identical rows and reset backfill.
Raw observations/goals are scaled by the networks exactly once. Joint canonical
features are already normalized by JointStateSpec and remain untouched.
"""

from dataclasses import dataclass

import torch

from supermimic.imagination.rollout import OBS_NOISE_HALF
from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
)
from supermimic.policy.nets import PolicyCfg, _goal_scale_vector, _obs_scale_vector
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

SCHEMA = "hoi-actor-b-critic-bplus-v1"
HISTORY = FUTURE = 10
OBJECT_GROUPS = {"position": 3, "rotation6": 6, "linear_heading": 3, "angular_body": 3}
OBJECT_DIVISORS = (1.0, 1.0, 1.0, 2.0)
DIMENSIONS = {
    "robot_history": 930,
    "object_history": 150,
    "observation": 1080,
    "robot_goal": 680,
    "object_goal": 150,
    "goal": 830,
    "robot_latent": 32,
    "object_history_embedding": 64,
    "object_goal_latent": 32,
    "decoder": 1058,
    "action": 29,
    "critic_features": 148,
    "critic": 1015,
    "physical": 84,
    "canonical": 86,
    "support": 18,
    "xi": 33,
    "wm_input": 166,
    "wm_residual": 82,
}


def _slices(groups):
    result, offset = {}, 0
    for name, width in groups.items():
        result[name] = slice(offset, offset + width)
        offset += width
    return result


OBJECT_SLICES = _slices(OBJECT_GROUPS)
OBJECT_HISTORY_SLICES = _slices({k: HISTORY * d for k, d in OBJECT_GROUPS.items()})
OBSERVATION_SLICES = _slices({"robot_history": 930, "object_history": 150})
GOAL_SLICES = _slices({"robot_goal": 680, "object_goal": 150})
DECODER_SLICES = _slices(
    {
        "robot_latent": 32,
        "robot_history": 930,
        "object_history": 64,
        "object_goal": 32,
    }
)
CRITIC_SLICES = _slices(
    {
        "joint_canonical": 86,
        "support": 18,
        "object_error": 15,
        "previous_clean_action": 29,
        "robot_goal": 680,
        "object_goal": 150,
        "xi": 33,
        "root_error": 4,
    }
)


def width(value, size):
    if value.shape[-1] != size:
        raise ValueError(f"Expected feature width {size}, got {value.shape}")


def object_scale(*, history=False, future=False):
    """Multipliers (inverse divisors), in the explicitly requested flatten order."""
    if history and future:
        raise ValueError("Choose history or future layout")
    parts = [
        torch.full((n * (HISTORY if history else 1),), 1 / d)
        for n, d in zip(OBJECT_GROUPS.values(), OBJECT_DIVISORS)
    ]
    scale = torch.cat(parts)
    return scale.repeat(FUTURE) if future else scale


def validate_feature_spec(spec):
    """Reject silent canonical/xi scale changes at the versioned adapter boundary."""
    original = StateSpec(StateSpecCfg(), spec.robot.in_scale.device, spec.robot.in_scale.dtype)
    for name in ("in_scale", "xi_offset", "xi_scale"):
        if not torch.equal(getattr(spec.robot, name), getattr(original, name)):
            raise ValueError(f"HOI policy canonical/context scale mismatch: {name}")
    expected = 1 / object_scale().to(spec.object_in_scale)
    if not torch.equal(spec.object_in_scale, expected):
        raise ValueError("HOI policy Object canonical scale mismatch")


def feature_schema():
    """Fresh JSON-compatible metadata, including scales and frame/timing semantics."""
    return {
        "schema": SCHEMA,
        "dimensions": dict(DIMENSIONS),
        "history": HISTORY,
        "future": FUTURE,
        "slices": {
            name: {k: [v.start, v.stop] for k, v in mapping.items()}
            for name, mapping in (
                ("object", OBJECT_SLICES),
                ("object_history", OBJECT_HISTORY_SLICES),
                ("observation", OBSERVATION_SLICES),
                ("goal", GOAL_SLICES),
                ("decoder", DECODER_SLICES),
                ("critic", CRITIC_SLICES),
            )
        },
        "robot_history": "oldest-first; term-major; biased-joints,jvel,body-omega,gravity,previous-clean-action",
        "object_history": "oldest-first; group-then-time; own actual Robot heading at each frame",
        "object_goal": "frame-major; reference Robot heading at each f..f+9; goal-only edge padding",
        "object_velocity": "absolute root-origin world linear expressed in Robot heading; Object-body angular",
        "rotation": "wxyz; r6=[R00,R01,R10,R11,R20,R21]",
        "object_error": "review-6.2; actual-minus-reference at f_t; reference Robot heading; r6(delta)-r6(I)",
        "critic_time": "s_t/ref_f_t; prior clean action; H+1 states; invalid rows excluded",
        "reward_time": "s_next/ref_f_next; unchanged P5 reward",
        "noise": {
            "robot_uniform_half_widths": [0.01, 0.5, 0.2, 0.05, 0.0],
            "object": "zero; simulator state observation; no deployment estimator",
        },
        "multipliers": {
            "robot_history": _obs_scale_vector(PolicyCfg()).tolist(),
            "robot_goal": _goal_scale_vector(PolicyCfg(), 680).tolist(),
            "object_history": object_scale(history=True).tolist(),
            "object_goal": object_scale(future=True).tolist(),
            "object_error": object_scale().tolist(),
        },
        "joint_canonical": "JointStateSpec normalized once; no policy rescaling",
        "support": "support-root-heading-p3-r6-v1; unscaled; separate from xi",
        "xi": "StateSpec.normalize_xi; offsets .75,0x32; divisors .45,.01x29,.05x3",
        "root_error": "original reference-heading position3/wrapped-yaw1; unscaled",
        "action": "native29; clean policy command; original affine actuation",
        "anchor_obs": False,
        "excluded": ["measured_force", "actual_contact", "reference_intent", "contact_geometry"],
    }


def robot_observation(state, previous_clean_action, xi, default_joint_pos):
    width(state, 84)
    width(previous_clean_action, 29)
    width(xi, 33)
    return analytic_observation(state[..., :71], previous_clean_action, xi, default_joint_pos)


def object_observation(state, spec):
    """Raw Object15, undoing ONLY the existing spec's normalization.

    This reuses the exact WM algebra without changing its contract. Network
    buffers subsequently apply the approved policy scale, never twice.
    """
    return spec.object_canonical(state) * spec.object_in_scale


def flatten_robot_history(history):
    if history.ndim != 3 or history.shape[1:] != (HISTORY, 93):
        raise ValueError("Robot history must be (N,10,93)")
    return flatten_history_term_major(history.transpose(0, 1))


def flatten_object_history(history):
    if history.ndim != 3 or history.shape[1:] != (HISTORY, 15):
        raise ValueError("Object history must be (N,10,15)")
    return torch.cat([history[..., s].flatten(1) for s in OBJECT_SLICES.values()], -1)


class HOIObservationAdapter:
    """Shared causal transform/noise law for collection and imagination.

    No storage: callers supply the existing aligned states and previous actions.
    Object noise is explicitly zero. Robot noise draws exactly 93 values/frame,
    preserving the original noise stream; no noise is added to previous actions.
    """

    def __init__(self, spec, default_joint_pos):
        validate_feature_spec(spec)
        self.spec, self.q0 = spec, default_joint_pos

    def observation(self, state, previous_clean_action, xi):
        return torch.cat(
            (
                robot_observation(state, previous_clean_action, xi, self.q0),
                object_observation(state, self.spec),
            ),
            -1,
        )

    @staticmethod
    def add_noise(observation, enabled):
        width(observation, 108)
        if not enabled:
            return observation
        robot = observation[..., :93]
        noisy = robot + (torch.rand_like(robot) * 2 - 1) * OBS_NOISE_HALF.to(robot)
        return torch.cat((noisy, observation[..., 93:]), -1)

    @staticmethod
    def flatten_history(history):
        """Replay/rollout time-major (10,N,108) -> raw Actor1080."""
        h = history.transpose(0, 1)
        return torch.cat(
            (flatten_robot_history(h[..., :93]), flatten_object_history(h[..., 93:])), -1
        )

    def history(self, states, previous_actions, xi, *, noise=False):
        if states.shape[0] != HISTORY or states.shape[-1] != 84:
            raise ValueError("Expected Replay histories (10,N,84)")
        if previous_actions.shape != (*states.shape[:2], 29):
            raise ValueError("History actions must use the same Replay indices")
        rows = [
            self.add_noise(self.observation(s, a, xi), noise)
            for s, a in zip(states, previous_actions)
        ]
        return self.flatten_history(torch.stack(rows))


@dataclass(frozen=True)
class ObjectGoalWindow:
    values: torch.Tensor  # (*batch,150), raw units; frame-major
    valid: torch.Tensor  # (*batch,10), goal-only padding mask
    local_frames: torch.Tensor
    global_frames: torch.Tensor


def object_future_goal(registry, clip_id, local_frame, spec):
    """Gather f..f+9 from checked immutable caches; never clamp an invalid start."""
    registry.global_frames(clip_id, local_frame)  # validates identity/start first
    length = registry.lengths(clip_id)
    frames = local_frame[..., None] + torch.arange(FUTURE, device=local_frame.device)
    valid = frames < length[..., None]
    local = torch.minimum(frames, length[..., None] - 1)
    clips = clip_id[..., None].expand_as(local)
    ref = registry.gather(clips.reshape(-1), local.reshape(-1))
    qpos, qvel = registry.lib.ref_qpos_qvel(ref.global_frame)
    b = ref.box
    state = torch.cat(
        (
            qpos,
            qvel,
            b.position,
            b.quaternion,
            b.linear_world,
            quat_apply_inverse(quat_normalize(b.quaternion), b.angular_world),
        ),
        -1,
    )
    values = object_observation(state, spec).reshape(*clip_id.shape, FUTURE * 15)
    return ObjectGoalWindow(values, valid, local, ref.global_frame.reshape_as(local))


def object_reference_error(state, reference):
    """Raw error15 from approved §6.2, live graph; reference is at s_t's frame."""
    width(state, 84)
    h = heading_quat(quat_normalize(reference.robot.root_quat_w))
    q = quat_normalize(state[..., 74:78])
    qr = quat_normalize(reference.box.quaternion)
    relative_world = quat_mul(q, quat_conjugate(qr))
    delta_h = quat_mul(quat_mul(quat_conjugate(h), relative_world), h)
    rotation_error = rot6d_from_quat(delta_h) - state.new_tensor([1, 0, 0, 1, 0, 0])
    return torch.cat(
        (
            quat_apply_inverse(h, state[..., 71:74] - reference.box.position),
            rotation_error,
            quat_apply_inverse(h, state[..., 78:81] - reference.box.linear_world),
            quat_apply_inverse(h, quat_apply(q, state[..., 81:84]) - reference.box.angular_world),
        ),
        -1,
    )


def critic_features(state, previous_clean_action, reference, spec, support):
    """Feature148: canonical86 already normalized; support/error/action raw."""
    width(previous_clean_action, 29)
    return torch.cat(
        (
            spec.canonical(state),
            support(state),
            object_reference_error(state, reference),
            previous_clean_action,
        ),
        -1,
    )


def assemble_critic_input(
    features, goals, xi_norm, root_error, *, error_scale, robot_goal_scale, object_goal_scale
):
    """The sole approved 1015D order; only raw error/goals need scaling here."""
    for tensor, size in ((features, 148), (goals, 830), (xi_norm, 33), (root_error, 4)):
        width(tensor, size)
        if tensor.shape[:-1] != features.shape[:-1]:
            raise ValueError("Critic inputs must have identical leading dimensions")
    return torch.cat(
        (
            features[..., :104],
            features[..., 104:119] * error_scale,
            features[..., 119:148],
            goals[..., :680] * robot_goal_scale,
            goals[..., 680:] * object_goal_scale,
            xi_norm,
            root_error,
        ),
        -1,
    )


def rollout_critic_inputs(rollout, context, spec, support, *, feature_builder=None):
    """Build H+1 live rows; padding stays zero and is never a regression sample."""
    if rollout.object_goals is None or rollout.previous_actions is None:
        raise ValueError("Actor B rollout fields required for Critic B+")
    state_valid = (
        torch.arange(len(rollout.states), device=rollout.states.device)[:, None]
        <= rollout.lengths[None]
    )
    clips = context.start.clip_id[None].expand_as(context.local_frames)
    reference = context.registry.gather(clips[state_valid], context.local_frames[state_valid])
    builder = feature_builder or (lambda s, a, r: critic_features(s, a, r, spec, support))
    live = builder(rollout.states[state_valid], rollout.previous_actions[state_valid], reference)
    features = live.new_zeros(*state_valid.shape, 148)
    features[state_valid] = live  # differentiable index put; no detached diagnostics
    return features, torch.cat((rollout.goals, rollout.object_goals), -1), rollout.e_root
