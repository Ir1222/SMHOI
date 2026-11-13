"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagStep
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import (
    ReferenceFrame,
    analytic_observation,
    anchor_relative_obs,
    flatten_history_term_major,
)
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel

# Paper Table A.1 sensor noise (uniform half-widths) in the 93-d observation order.
OBS_NOISE_HALF = torch.tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29)
ANCHOR_NOISE_HALF = torch.tensor([0.25] * 3 + [0.05] * 6)


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, state_dim); Robot71 or HOI84
    actions: torch.Tensor  # (H, n, 29)
    rewards: torch.Tensor  # (H, n)      r_{t+1}
    dhat: torch.Tensor  # (H+1, n)    predicted termination prob of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n)    c_t
    goals: torch.Tensor  # (H+1, n, 680)
    e_root: torch.Tensor  # (H+1, n, 4)
    frames: torch.Tensor  # (H+1, n)
    reward_terms: dict  # mean of each reward term over the rollout (diagnostics)
    valid: torch.Tensor | None = None  # (H,n), None preserves original robot path
    lengths: torch.Tensor | None = None
    bootstrap: torch.Tensor | None = None
    continuation: torch.Tensor | None = None
    boundaries: dict | None = None
    diagnostics: dict | None = None
    object_goals: torch.Tensor | None = None  # (H+1,n,150), raw, reference-only
    object_goal_valid: torch.Tensor | None = None  # (H+1,n,10), goal padding only
    previous_actions: torch.Tensor | None = None  # (H+1,n,29), clean incoming action
    context: object | None = None  # optional reference identity for live Critic builders


def imagine(
    wm: WorldModel,
    policy: Policy,
    lib: MotionLibrary,
    start: StartBatch,
    horizon: int,
    future_frames: int = 10,
    obs_noise: bool = True,
    step_fn: ImagStep | None = None,
    anchor_obs: bool = False,
    context=None,
    observation_adapter=None,
    object_goal_builder=None,
) -> ImagRollout:
    device = start.state.device
    n = start.state.shape[0]
    q0 = lib.default_joint_pos
    step_fn = step_fn or ImagStep(wm, q0)
    noise_half = OBS_NOISE_HALF.to(device)
    anchor_noise_half = ANCHOR_NOISE_HALF.to(device)
    ai = wm.heads.anchor_index
    project = (lambda s: s) if context is None else context.project_state
    term_names = REWARD_TERM_NAMES if context is None else context.term_names

    def add_noise(o):
        if observation_adapter is not None:
            return observation_adapter.add_noise(o, obs_noise)
        if obs_noise:
            return o + (torch.rand_like(o) * 2 - 1) * noise_half
        return o

    def make_goal(frame_t, ref_t, anchor_pose):
        g = lib.goal_window(frame_t, future_frames)
        if not anchor_obs:
            return g
        rel = anchor_relative_obs(anchor_pose[..., 0:3], anchor_pose[..., 3:7], ref_t.body_pos_w[..., ai, :], ref_t.body_quat_w[..., ai, :])
        if obs_noise:
            rel = rel + (torch.rand_like(rel) * 2 - 1) * anchor_noise_half
        return torch.cat([g, rel], dim=-1)

    # Initial observation history from the logged states/actions (noise re-sampled per frame).
    observe = (lambda s, a, xi: analytic_observation(project(s), a, xi, q0)) if observation_adapter is None else observation_adapter.observation
    flatten_history = flatten_history_term_major if observation_adapter is None else observation_adapter.flatten_history
    hist = [add_noise(observe(start.hist_states[i], start.hist_actions[i], start.xi)) for i in range(start.hist_states.shape[0])]
    s = start.state
    frame = start.frame.clone()
    a_prev = start.prev_action
    previous_actions = [a_prev] if observation_adapter is not None else None
    if object_goal_builder is not None and context is None:
        raise ValueError("Object goal builder requires explicit rollout reference context")
    object_window = None if object_goal_builder is None else object_goal_builder(context)
    c = torch.ones(n, device=device)

    states = [s]
    actions, rewards = [], []
    dhats, survivals, frames = [torch.zeros(n, device=device)], [c], [frame]
    # All target frames of the rollout are known up front (frame + t, clamped at the clip end):
    # gather / FK the references for every step in one batched call.
    if context is None:
        last = lib.last_frame_of(frame)
        frames_all = torch.minimum(frame[None, :] + torch.arange(horizon + 1, device=device)[:, None], last[None, :])  # (H+1, n)
        ref_all = reference_at(lib, frames_all.reshape(-1))
    else:
        frames_all = context.frames

    def ref_at(t):
        sl = slice(t * n, (t + 1) * n)
        return ReferenceFrame(ref_all.body_pos_w[sl], ref_all.body_quat_w[sl], ref_all.body_lin_vel_w[sl], ref_all.body_ang_vel_w[sl], ref_all.root_pos_w[sl], ref_all.root_quat_w[sl])

    ref = ref_at(0) if context is None else context.initial_reference
    if anchor_obs:
        b0 = wm.heads.robot_bodies(project(s))
        anchor_pose = torch.cat([b0["pos"][..., ai, :], b0["quat"][..., ai, :]], dim=-1)
    else:
        anchor_pose = None
    goals = [make_goal(frame, ref, anchor_pose)]
    e_roots = [wm.heads.root_error_features(project(s), ref)]
    term_sum = torch.zeros(len(term_names), device=device, dtype=s.dtype if context is not None else None)

    for t in range(horizon):
        obs_flat = flatten_history(torch.stack(hist, 0))
        rows = None if context is None else context.valid[t].nonzero().squeeze(-1)
        select = (lambda x: x) if context is None else (lambda x, rows=rows: x[rows])
        if rows is None or rows.numel():
            policy_goal = goals[-1] if object_window is None else torch.cat((goals[-1], object_window.values[t]), -1)
            a = policy(select(obs_flat), select(policy_goal))
            scoring_ref = ref if context is None else context.step_reference(t, rows)
            s_next, r, term_vec, logits, obs_clean, anchor_pose = step_fn(select(s), a, select(a_prev), select(start.xi), scoring_ref)
            if observation_adapter is not None:
                # Recompute from the reached physical state, retaining the WM graph.
                obs_clean = observe(s_next, a, select(start.xi))
            d = torch.sigmoid(logits)
            if context is None:
                term_sum += term_vec.detach().mean(-1)
                ref = ref_at(t + 1)
                hist = hist[1:] + [add_noise(obs_clean)]
                a_prev = a
                s = s_next
                goal_next = make_goal(frames_all[t + 1], ref, anchor_pose)
                root_next = wm.heads.root_error_features(s, ref)
            else:
                d = torch.where(context.hard_failure[t, rows], torch.ones_like(d), d)
                term_sum += term_vec.detach().sum(-1)
                context.record(t, rows, s_next, term_vec, scoring_ref, step_fn)
                next_hist = [select(h) for h in hist[1:]] + [add_noise(obs_clean)]
                hist = [old.index_copy(0, rows, new) for old, new in zip(hist, next_hist)]
                a_prev = a_prev.index_copy(0, rows, a)
                s = s.index_copy(0, rows, s_next)
                goal_next = goals[-1].index_copy(0, rows, make_goal(frames_all[t + 1, rows], scoring_ref.robot, anchor_pose))
                root_next = e_roots[-1].index_copy(0, rows, wm.heads.root_error_features(project(s_next), scoring_ref.robot))
                a = torch.zeros_like(a_prev).index_copy(0, rows, a)
                r = s.new_zeros(n).index_copy(0, rows, r)
                d = s.new_zeros(n).index_copy(0, rows, d)
        else:
            # No calls to Actor/WM/FK/reference reward on empty/padded batches.
            a, r, d = torch.zeros_like(a_prev), s.new_zeros(n), s.new_zeros(n)
            goal_next, root_next = goals[-1], e_roots[-1]
        c = c * (1.0 - d)

        frame = frames_all[t + 1]
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        goals.append(goal_next)
        e_roots.append(root_next)
        frames.append(frame)
        if previous_actions is not None:
            previous_actions.append(a_prev)

    return ImagRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        goals=torch.stack(goals),
        e_root=torch.stack(e_roots),
        frames=torch.stack(frames),
        reward_terms={k: float(v / (horizon if context is None else int(context.valid.sum()))) for k, v in zip(term_names, term_sum.tolist())} | ({} if context is None else context.extra_metrics()),
        valid=None if context is None else context.valid,
        lengths=None if context is None else context.lengths,
        bootstrap=None if context is None else context.bootstrap,
        continuation=None if context is None else context.continuation,
        boundaries=None if context is None else context.boundaries,
        diagnostics=None if context is None else context.diagnostics,
        object_goals=None if object_window is None else object_window.values,
        object_goal_valid=None if object_window is None else object_window.valid,
        previous_actions=None if previous_actions is None else torch.stack(previous_actions),
        context=context,
    )
