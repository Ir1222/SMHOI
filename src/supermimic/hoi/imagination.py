"""HOI components for the original Replay/imagine/PolicyTrainer algorithms.

No independent recurrence, optimizer, Critic or TD(lambda) implementation lives
here. P7 must explicitly install these components; sm-train defaults are unchanged.
"""

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.hoi.replay import HOIReplayBuffer
from supermimic.hoi.reward_reference import HOIReferenceRegistry
from supermimic.hoi.tracking_reward import (
    HOI_TERM_NAMES,
    BoxReference,
    HOITrackingReward,
    box_errors,
    robot_state,
)
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.world_model.heads import ReferenceFrame, analytic_observation


@dataclass
class HOIStartBatch(StartBatch):
    clip_id: torch.Tensor
    local_frame: torch.Tensor
    global_frame: torch.Tensor
    reference_sha256: tuple[str, ...]
    env_id: torch.Tensor
    episode_id: torch.Tensor
    # Optional authoritative counterfactual boundaries, NOT the logged action's outcome.
    remaining_steps: torch.Tensor | None = None
    hard_failure_step: torch.Tensor | None = None  # 1-based reached step; 0 = absent
    robot_failure: torch.Tensor | None = None
    object_failure: torch.Tensor | None = None


class HOIPolicyReplayBuffer(HOIReplayBuffer):
    def __init__(self, *args, registry: HOIReferenceRegistry, **kwargs):
        super().__init__(*args, **kwargs)
        if self.s.device != registry.device:
            raise ValueError("Replay/reference device mismatch")
        self.registry = registry
        self.metadata["policy_reference_registry"] = registry.metadata()

    def start_mask(self, hist):
        if not 1 <= hist < self.T - 1:
            raise ValueError("History must fit the replay capacity")
        mask = super().start_mask(hist) & self.fields["transition_valid"]
        e, t = mask.nonzero(as_tuple=True)
        if e.numel():
            clip, frame = self.clip[e, t], self.frame[e, t]
            self.registry.global_frames(clip, frame)  # fail closed on corrupt identities
            mask[e, t] &= frame < self.registry.lengths(clip) - 1
        return mask

    def _history_valid(self, e, t, ee, tt):
        ok = super()._history_valid(e, t, ee, tt)
        offsets = torch.arange(tt.shape[0], device=self.device) - (tt.shape[0] - 1)
        ok &= self.clip[ee, tt] == self.clip[e, t][None]
        ok &= self.frame[ee, tt] == self.frame[e, t][None] + offsets[:, None]
        ok &= ~self.terminal[ee, tt]
        ok &= self.fields["transition_valid"][ee, tt]
        # A gap severs the entire earlier history, even if an older row matches again.
        return ok.flip(0).long().cumprod(0).flip(0).bool()

    def _previous_action_valid(self, e, t, ee, tt_fixed, tprev):
        ok = super()._previous_action_valid(e, t, ee, tt_fixed, tprev)
        return (
            ok
            & (self.clip[ee, tprev] == self.clip[ee, tt_fixed])
            & (self.frame[ee, tprev] + 1 == self.frame[ee, tt_fixed])
            & self.fields["transition_valid"][ee, tprev]
            & ~self.fields["transition_terminal"][ee, tprev]
        )

    def _make_start(self, e, t, **fields):
        clip, local = self.clip[e, t].clone(), self.frame[e, t].clone()
        global_frame = self.registry.global_frames(clip, local)
        fields["frame"] = global_frame.clone()
        return HOIStartBatch(
            **fields,
            clip_id=clip,
            local_frame=local,
            global_frame=global_frame,
            reference_sha256=self.registry.hashes(clip),
            env_id=e.clone(),
            episode_id=self.ep_id[e, t].clone(),
        )


class HOIImagStep:
    term_names = HOI_TERM_NAMES

    def __init__(
        self,
        wm,
        default_joint_pos,
        reward: HOITrackingReward,
        compile_mode=None,
        *,
        backend="inductor",
    ):
        if wm.heads is not reward.heads or wm.spec.robot.cfg.dt != reward.cfg.dt:
            raise ValueError("WM/head/reward configuration mismatch")
        self.wm, self.q0, self.reward = wm, default_joint_pos, reward
        self._fn = self._step
        self._diag_fn = self._diagnostics
        if compile_mode:
            options = {} if compile_mode == "default" else {"mode": compile_mode}
            self._fn = torch.compile(self._step, backend=backend, dynamic=False, **options)
            self._diag_fn = torch.compile(
                self._diagnostics, backend=backend, dynamic=False, **options
            )

    def _step(self, s, a, a_prev, xi, bp, bq, bv, bw, rp, rq, op, oq, ov, ow, contact):
        ref = ReferenceFrame(bp, bq, bv, bw, rp, rq)
        obj = BoxReference(op, oq, ov, ow, contact)
        s_next = self.wm.step(s, a, xi)
        robot = robot_state(s_next)
        bodies = self.wm.heads.robot_bodies(robot)
        terms = self.reward(s_next, a, a_prev, ref, obj, bodies)
        term_vec = torch.stack([terms[k] for k in self.term_names])
        logits = self.wm.termination_logits(robot, ref)
        obs = analytic_observation(robot, a, xi, self.q0)
        ai = self.wm.heads.anchor_index
        anchor = torch.cat((bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]), -1)
        return s_next, terms["total"], term_vec, logits, obs, anchor

    def __call__(self, s, a, a_prev, xi, reference):
        r, b = reference.robot, reference.box
        return self._fn(
            s,
            a,
            a_prev,
            xi,
            r.body_pos_w,
            r.body_quat_w,
            r.body_lin_vel_w,
            r.body_ang_vel_w,
            r.root_pos_w,
            r.root_quat_w,
            b.position,
            b.quaternion,
            b.linear_world,
            b.angular_world,
            b.contact,
        )

    def _diagnostics(self, state, op, oq, ov, ow, contact):
        reference = BoxReference(op, oq, ov, ow, contact)
        return {
            **self.reward.proximity(state, contact),
            "box_errors": box_errors(state, reference),
            "contact": contact,
        }

    def diagnostics(self, state, reference):
        b = reference.box
        return self._diag_fn(
            state, b.position, b.quaternion, b.linear_world, b.angular_world, b.contact
        )


class HOIRolloutContext:
    project_state = staticmethod(robot_state)
    term_names = HOI_TERM_NAMES

    def __init__(self, registry, start: HOIStartBatch, horizon):
        if not isinstance(start, HOIStartBatch) or horizon < 1:
            raise ValueError("HOI start metadata and positive horizon required")
        self.registry, self.start, self.horizon = registry, start, horizon
        global_frame = registry.global_frames(start.clip_id, start.local_frame)
        if (
            not torch.equal(global_frame, start.global_frame)
            or not torch.equal(global_frame, start.frame)
            or start.reference_sha256 != registry.hashes(start.clip_id)
        ):
            raise ValueError("Replay/global reference identity mismatch")
        n, device = len(start.frame), start.frame.device
        if start.frame.shape != (n,) or start.state.shape != (n, 84):
            raise ValueError("Expected aligned HOI start batch")
        remaining_clip = registry.lengths(start.clip_id) - 1 - start.local_frame
        lengths = remaining_clip.clamp_max(horizon)
        known_timeout = start.remaining_steps
        if known_timeout is not None:
            self._check_steps(known_timeout, n, device, positive=True)
            lengths = torch.minimum(lengths, known_timeout)
        failure = start.hard_failure_step
        if failure is not None:
            self._check_steps(failure, n, device, positive=False)
            lengths = torch.minimum(lengths, torch.where(failure > 0, failure, lengths))
        if torch.any(lengths < 1):
            raise ValueError("Terminal frame has no outgoing transition")
        self.lengths = lengths
        self.valid = torch.arange(horizon, device=device)[:, None] < lengths[None]
        natural = lengths == remaining_clip
        timeout = (
            torch.zeros(n, device=device, dtype=torch.bool)
            if known_timeout is None
            else lengths == known_timeout
        )
        hard = (
            torch.zeros_like(natural) if failure is None else (failure > 0) & (failure == lengths)
        )
        self.bootstrap = ~(natural | hard)
        last = torch.arange(horizon, device=device)[:, None] == lengths[None] - 1
        self.hard_failure = last & hard[None]
        self.continuation = self.valid & ~(last & (~self.bootstrap)[None])
        self.boundaries = {
            "clip_end": natural,
            "timeout": timeout,
            "physical_failure": hard,
            "horizon": lengths == horizon,
        }
        for name in ("robot_failure", "object_failure"):
            reason = getattr(start, name)
            if reason is None:
                reason = torch.zeros_like(hard)
            if reason.shape != (n,) or reason.dtype != torch.bool or reason.device != device:
                raise ValueError("Failure reasons must be aligned bool tensors")
            if torch.any(
                reason & ((failure == 0) if failure is not None else torch.ones_like(hard))
            ):
                raise ValueError("Failure reason requires an authoritative failure step")
            self.boundaries[name] = reason & hard
        self.boundaries["physical_unspecified"] = hard & ~(
            self.boundaries["robot_failure"] | self.boundaries["object_failure"]
        )
        elapsed = torch.minimum(torch.arange(horizon + 1, device=device)[:, None], lengths[None])
        self.local_frames = start.local_frame[None] + elapsed
        self.frames = start.global_frame[None] + elapsed
        self.initial_reference = registry.gather(start.clip_id, start.local_frame).robot
        self.diagnostics = {}

    @staticmethod
    def _check_steps(value, n, device, *, positive):
        if (
            value.shape != (n,)
            or value.dtype != torch.long
            or value.device != device
            or torch.any(value < (1 if positive else 0))
        ):
            raise ValueError("Invalid remaining/failure step metadata")

    def step_reference(self, t, rows):
        if not self.valid[t, rows].all():
            raise ValueError("Cannot gather padded reward references")
        return self.registry.gather(self.start.clip_id[rows], self.local_frames[t + 1, rows])

    @torch.no_grad()
    def record(self, t, rows, state, terms, reference, step_fn):
        data = step_fn.diagnostics(state, reference)
        data["terms"] = terms.transpose(0, 1)
        # A single checked reference supplies reward, termination and Critic root features.
        data["evaluated_local_frame"] = reference.local_frame
        data["evaluated_global_frame"] = reference.global_frame
        for key, value in data.items():
            if key not in self.diagnostics:
                fill = -1 if "frame" in key else 0
                self.diagnostics[key] = torch.full(
                    (self.horizon, len(self.start.frame), *value.shape[1:]),
                    fill,
                    device=value.device,
                    dtype=value.dtype,
                )
            self.diagnostics[key][t, rows] = value.detach()

    def extra_metrics(self):
        d, valid = self.diagnostics, self.valid
        terms, raw = d["terms"][valid], d["proximity"][valid]
        on = d["contact"][valid]
        result = {
            "total": float(terms.sum(-1).mean()),
            "robot_positive": float(terms[:, :6].sum(-1).mean()),
            "regularization": float(terms[:, 6:8].sum(-1).mean()),
            "box_total": float(terms[:, 8:12].sum(-1).mean()),
            "proximity_raw": float(raw.mean()),
            "on_count": int(on.sum()),
            "off_count": int((~on).sum()),
            "valid_length_mean": float(self.lengths.float().mean()),
        }
        # Undefined conditional means are omitted (counts remain explicit).
        for name, mask in (("on", on), ("off", ~on)):
            if mask.any():
                result[f"proximity_raw_{name}_mean"] = float(raw[mask].mean())
        for j, name in enumerate(
            ("position_m2", "orientation_rad2", "linear_m2_s2", "angular_rad2_s2")
        ):
            result[f"box_error/{name}"] = float(d["box_errors"][valid, j].mean())
        result.update({f"boundary/{k}": int(v.sum()) for k, v in self.boundaries.items()})
        return result


class HOIPolicyTrainer(PolicyTrainer):
    """Construction/projection only; inherits every optimizer and learning algorithm."""

    def __init__(self, wm, policy, critic, registry, reward, cfg, *, backend="inductor"):
        from supermimic.hoi.policy import HOIActor, HOICritic
        from supermimic.hoi.policy_features import (
            HOIObservationAdapter,
            critic_features,
            object_future_goal,
            rollout_critic_inputs,
        )

        if registry.dt != reward.cfg.dt:
            raise ValueError("Reference/reward dt mismatch")
        tracked_names = tuple(wm.heads.fk.body_names[i] for i in wm.heads.tracked_idx)
        if tracked_names != registry.lib.body_names:
            raise ValueError("Library/FK tracked body ordering mismatch")
        goal_dim = cfg.future_frames * 68 + 9 * int(cfg.anchor_obs)
        self.joint_policy = isinstance(policy, HOIActor)
        if self.joint_policy != isinstance(critic, HOICritic):
            raise ValueError("Actor B and Critic B+ must be selected together")
        if self.joint_policy:
            if cfg.history != 10 or cfg.future_frames != 10 or cfg.anchor_obs:
                raise ValueError("Actor B dimensions require history10/future10/no anchor")
            observation = HOIObservationAdapter(wm.spec, registry.lib.default_joint_pos)
            def build_critic_features(state, previous_action, reference):
                return critic_features(state, previous_action, reference, wm.spec, wm.support)

            if cfg.compile:
                options = {} if cfg.compile == "default" else {"mode": cfg.compile}
                observation.observation = torch.compile(
                    observation.observation, backend=backend, dynamic=False, **options,
                )
                observation.flatten_history = torch.compile(
                    observation.flatten_history, backend=backend, dynamic=False, **options,
                )
                build_critic_features = torch.compile(
                    build_critic_features, backend=backend, dynamic=False, **options,
                )
            adapters = {
                "observation_adapter": observation,
                "object_goal_builder": lambda context: object_future_goal(
                    registry, context.start.clip_id[None].expand_as(context.local_frames),
                    context.local_frames, wm.spec,
                ),
                "rollout_critic_inputs": lambda ro, context: rollout_critic_inputs(
                    ro, context, wm.spec, wm.support, feature_builder=build_critic_features,
                ),
            }
        elif (
            cfg.history != 10
            or cfg.future_frames != 10
            or policy.cfg.history != 10
            or policy.cfg.latent_dim != 32
            or policy.cfg.goal_dim != goal_dim
            or critic.mlp.net[0].in_features != 71 + goal_dim + 33 + 4
        ):
            raise ValueError("Preserve original Actor/Critic history/goal/state dimensions")
        else:
            # Historical P5 interface remains an explicit compatibility path.
            adapters = {"critic_features": lambda states: wm.spec.robot.canonical(robot_state(states))}
        self.registry, self.reward = registry, reward
        super().__init__(
            wm,
            policy,
            critic,
            registry.lib,
            cfg,
            step_fn=HOIImagStep(
                wm, registry.lib.default_joint_pos, reward, cfg.compile, backend=backend
            ),
            rollout_context_factory=lambda start, horizon: HOIRolloutContext(
                registry, start, horizon
            ),
            **adapters,
        )

    def initialization_metadata(self):
        if self.joint_policy:
            return {
                "schema": "hoi-policy-initialization-v2",
                "actor": self.policy.metadata(), "critic": self.critic.metadata(),
                "reward": self.reward.cfg.metadata(), "references": self.registry.metadata(),
                "reward_frame": "reached-local-frame+1", "critic_frame": "current-local-frame",
                "natural_end_bootstrap": False, "actor_obs": 1080, "goal": 830,
                "critic_input": 1015, "action": 29,
                "optimizer_resume": False, "sampler_resume": False, "training_ready": False,
            }
        return {
            "schema": "hoi-policy-initialization-v1",
            "reward": self.reward.cfg.metadata(),
            "references": self.registry.metadata(),
            "reward_frame": "reached-local-frame+1",
            "natural_end_bootstrap": False,
            "actor_obs": 930,
            "goal": self.cfg.future_frames * 68 + 9 * int(self.cfg.anchor_obs),
            "critic_canonical": 71,
            "action": 29,
            "optimizer_resume": False,
            "sampler_resume": False,
            "training_ready": False,
        }
