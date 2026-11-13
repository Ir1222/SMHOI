"""Physical lifecycle adapters around native stepping and the accepted P2 recorder."""

import math

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.hoi.imagination import HOIPolicyReplayBuffer
from supermimic.hoi.policy_features import critic_features, object_future_goal
from supermimic.hoi.recorder import NAME, HOIReplayRecorder
from supermimic.hoi.state import ReferenceResetCache
from supermimic.hoi.training_safety import finite


class TrainingReplay(HOIPolicyReplayBuffer):
    """Same ring/history/window algorithms with authoritative timeout metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("remaining_steps", "episode_step"):
            self.fields[name] = torch.zeros_like(self.frame)
        for name in ("robot_failure", "object_failure", "bootstrap", "continuation"):
            self.fields[name] = torch.zeros_like(self.valid)
        self.fields["terminal_value"] = torch.zeros_like(self.r)

    def _make_start(self, e, t, **fields):
        start = super()._make_start(e, t, **fields)
        start.remaining_steps = self.fields["remaining_steps"][e, t].clone()
        # Logged failures belong to the recorded action, never counterfactual actions.
        return start


def make_sampler(env, cfg):
    # Not registered as a command: its Robot reset/state-writing methods never run.
    # Reuse the original sampling and discounted-return accumulation implementation.
    return MultiClipMotionCommand(
        MultiClipMotionCommandCfg(
            npz_files=[str(cfg.reference)],
            anchor_body_name=ANCHOR_BODY,
            root_body_name=ROOT_BODY,
            body_names=TRACKED_BODIES,
            sampling_mode=cfg.env.sampling_mode,
            g_max=cfg.sampler_g_max,
            gamma=cfg.policy_trainer.gamma,
            max_logged_steps=max(1024, math.ceil(cfg.env.episode_length_s / 0.02) + 2),
            library_fp16=False,
        ),
        env,
    )


def robot_failure(env):
    recorder = env.recorder_manager.get_term(NAME)
    state = recorder.io.read()
    ref = recorder.registry.gather(recorder.clip, recorder.frame + 1)
    # Same original root height/tilt definitions and thresholds, fresh reached state.
    features = recorder.wm.heads.termination_features(state[:, :3], state[:, 3:7], ref.robot)
    return (features[:, 0].abs() > 0.25) | (features[:, 1].abs() > 0.8)


def object_failure(env):
    recorder = env.recorder_manager.get_term(NAME)
    threshold = recorder.config.box_min_height
    if threshold is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return recorder.io.read()[:, 73] < threshold


def time_limit(env):
    recorder = env.recorder_manager.get_term(NAME)
    return env.episode_length_buf >= recorder.max_steps


class TrainingRecorder(HOIReplayRecorder):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = {**cfg.params, **env._hoi_components}
        self.config, self.registry, self.wm = p["config"], p["registry"], p["wm"]
        self.reward, self.critic = p["reward"], p["critic"]
        self.observation, self.sampler = p["observation"], p["sampler"]
        self.max_steps = math.ceil(self.config.env.episode_length_s / env.step_dt)
        self.cache = ReferenceResetCache(self.references[0], self.io.joint_names, env.device)
        if (
            max(self.config.reset_frames)
            >= self.registry.lengths(torch.zeros(1, dtype=torch.long, device=env.device)).item() - 1
        ):
            raise ValueError("Reset frame must have an outgoing transition")
        original_metadata = self.buffer.metadata
        self.buffer = TrainingReplay(
            env.num_envs, p["capacity"], env.device, force_on=p["force_on"], registry=self.registry
        )
        self.buffer.metadata.update(original_metadata)
        self.buffer.metadata["training_identity"] = p["identity"]
        self.age = torch.zeros_like(self.frame)
        self.reset_count = torch.zeros_like(self.frame)
        self.previous_action = torch.zeros(env.num_envs, 29, device=env.device)
        self.history_states = torch.zeros(10, env.num_envs, 84, device=env.device)
        self.history_actions = torch.zeros(10, env.num_envs, 29, device=env.device)
        self.history_observations = torch.zeros(10, env.num_envs, 108, device=env.device)
        self.last = {
            "state": torch.zeros_like(self.state),
            "next_state": torch.zeros_like(self.state),
            "action": torch.zeros_like(self.previous_action),
            "executed_action": torch.zeros_like(self.previous_action),
            "previous_action": torch.zeros_like(self.previous_action),
            "frame": torch.zeros_like(self.frame),
            "episode": torch.zeros_like(self.frame),
            "row": torch.zeros_like(self.frame),
            "value": torch.zeros(env.num_envs, device=env.device),
            "reward": torch.zeros(env.num_envs, device=env.device),
        }
        self.last_terms = {}
        self.sampler.bootstrap_fn = lambda ids: self.last["value"][ids]

    def goals(self, clip, frame):
        global_frame = self.registry.global_frames(clip, frame)
        return torch.cat(
            (
                self.registry.lib.goal_window(global_frame, 10),
                object_future_goal(self.registry, clip, frame, self.wm.spec).values,
            ),
            -1,
        )

    @torch.no_grad()
    def value(self, state, action, clip, frame, xi):
        ref = self.registry.gather(clip, frame)
        return self.critic(
            critic_features(state, action, ref, self.wm.spec, self.wm.support),
            self.goals(clip, frame),
            self.wm.spec.normalize_xi(xi),
            self.wm.heads.root_error_features(state[:, :71], ref.robot),
        )

    @torch.no_grad()
    def record_post_reset(self, env_ids):
        super().record_post_reset(env_ids)
        if not len(env_ids):
            return
        if self.config.env.sampling_mode == "start":
            starts = torch.tensor(self.config.reset_frames, device=self._env.device)
            frames = starts[(env_ids + self.reset_count[env_ids]) % len(starts)]
        else:
            frames = self.sampler._sample_frames(len(env_ids))
        clips = torch.zeros_like(frames)  # This version explicitly binds one full source clip.
        self.io.write(env_ids, self.cache.frames(frames))
        self.begin_episode(env_ids, frames, clips)
        self.age[env_ids] = 0
        self.previous_action[env_ids] = 0
        self.sampler.frame[env_ids] = frames
        self.sampler.clip_end[env_ids] = False
        self.sampler.ep_len[env_ids] = 0
        self.reset_count[env_ids] += 1
        self.history_states[:, env_ids] = self.state[env_ids]
        self.history_actions[:, env_ids] = 0
        obs = self.observation.observation(
            self.state[env_ids], self.previous_action[env_ids], self.xi[env_ids]
        )
        for t in range(10):
            self.history_observations[t, env_ids] = self.observation.add_noise(
                obs, self.config.env.obs_noise
            )

    @torch.no_grad()
    def _write(self, ids):
        if not len(ids):
            return
        env = self._env
        reached = self.capture.next_state[ids].clone()
        action = env.action_manager.action[ids].clone()
        executed = env.action_manager.get_term("joint_pos").executed_raw[ids].clone()
        frame, clip, row = (
            self.frame[ids].clone(),
            self.clip[ids].clone(),
            self.buffer.ptr[ids].clone(),
        )
        ref = self.registry.gather(clip, frame + 1)
        terms = self.reward(reached, action, self.previous_action[ids], ref.robot, ref.box)
        value = self.value(reached, action, clip, frame + 1, self.xi[ids])
        finite("reached reward/value", terms, value)
        for key, tensor in (
            ("state", self.state[ids]),
            ("next_state", reached),
            ("action", action),
            ("executed_action", executed),
            ("previous_action", self.previous_action[ids]),
            ("frame", frame),
            ("row", row),
            ("episode", self.buffer.ep_counter[ids]),
            ("value", value),
            ("reward", terms["total"]),
        ):
            self.last[key][ids] = tensor
        # P2 validates and consumes all four samples, records the terminal state,
        # and advances metadata before native reset can clear anything.
        super()._write(ids)
        b = self.buffer
        b.r[ids, row] = terms["total"]
        b.fields["episode_step"][ids, row] = self.age[ids]
        b.fields["remaining_steps"][ids, row] = self.max_steps - self.age[ids]
        for key in ("robot_failure", "object_failure"):
            b.fields[key][ids, row] = env.termination_manager.get_term(key)[ids]
        tail = ~(b.fields["physical_failure"][ids, row] | b.fields["clip_end"][ids, row])
        b.fields["bootstrap"][ids, row] = tail
        b.fields["continuation"][ids, row] = tail
        b.fields["terminal_value"][ids, row] = value * tail
        env.reward_buf[ids] = terms["total"]  # Already dt-scaled; never through reward manager.
        for key, tensor in terms.items():
            if key not in self.last_terms:
                self.last_terms[key] = tensor.new_zeros(env.num_envs, *tensor.shape[1:])
            self.last_terms[key][ids] = tensor
        self.history_states[:, ids] = torch.cat((self.history_states[1:, ids], reached[None]), 0)
        self.history_actions[:, ids] = torch.cat((self.history_actions[1:, ids], action[None]), 0)
        obs = self.observation.observation(reached, action, self.xi[ids])
        obs = self.observation.add_noise(obs, self.config.env.obs_noise)
        self.history_observations[:, ids] = torch.cat(
            (self.history_observations[1:, ids], obs[None]), 0
        )
        self.previous_action[ids] = action
        self.age[ids] += 1
        self.sampler.frame[ids] = ref.global_frame
        self.sampler.clip_end[ids] = b.fields["clip_end"][ids, row]
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        mask[ids] = True
        self.sampler._log_step_rewards(mask)
        terminal = b.fields["transition_terminal"][ids, row]
        if terminal.any():
            self.sampler._finalize_episodes(ids[terminal])


class TrainingEnvironment:
    """Observation/reset bridge. Native mjlab owns the sole physical step loop."""

    def __init__(self, env, recorder, guard):
        self.raw, self.recorder, self.guard = env, recorder, guard
        self.num_envs, self.device = env.num_envs, env.device
        self.resume_pending = False

    def observations(self):
        r = self.recorder
        result = {
            "policy": r.observation.flatten_history(r.history_observations),
            "goal": r.goals(r.clip, r.frame),
        }
        finite("observations", result)
        return result

    def reset(self, *, env_ids=None):
        if self.resume_pending and env_ids is None:
            self.resume_pending = False
        else:
            self.raw.reset(env_ids=env_ids)
        return self.observations(), {}

    def step(self, action):
        finite("Actor action", action)
        if action.shape != (self.num_envs, 29):
            raise ValueError("Expected action29 for every world")
        if not self.recorder.ready.all():
            raise RuntimeError("Unbound HOI episode")
        _, reward, terminated, timeout, extras = self.raw.step(action)
        finite("physical reward", reward)
        extras["hoi"] = {k: v.clone() for k, v in self.recorder.last.items()}
        return self.observations(), reward, terminated, timeout, extras

    def close(self):
        self.raw.close()
