"""P2 collector using native mjlab substep metrics and recorder lifecycle hooks.

Call install_recording on a fresh P1 physical environment. After each physical
reset, call begin_episode with its reference start frames/clip IDs. P7 owns future
command/curriculum integration; this collector never overwrites physical states.
"""

from __future__ import annotations

import torch
from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg
from mjlab.managers.recorder_manager import RecorderManager, RecorderTerm, RecorderTermCfg
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg

from supermimic.envs.sim_context import SimContext
from supermimic.hoi.contact import SENSOR_NAMES, IntervalCapture
from supermimic.hoi.reference import DEFAULT_ASSET, DEFAULT_OUTPUT, ReferenceClip, sha256
from supermimic.hoi.replay import HOIReplayBuffer
from supermimic.hoi.state import JointStateIO

NAME = "hoi_replay"
CLIP_END = "hoi_clip_end"


def capture_substep(env):
    recorder = env.recorder_manager.get_term(NAME)
    if not recorder.ready.all():
        raise RuntimeError("Bind reference metadata with begin_episode after every reset")
    recorder.capture.sample()
    return torch.zeros(env.num_envs, device=env.device)


def natural_clip_end(env):
    recorder = env.recorder_manager.get_term(NAME)
    return recorder.ready & (recorder.frame + 1 == recorder.last_frame)


class HOIReplayRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.io = JointStateIO(env)
        self.ctx = SimContext(env, r"^(left|right)_foot[1-7]_collision$")
        self.buffer = HOIReplayBuffer(
            env.num_envs, cfg.params["capacity"], env.device, force_on=cfg.params["force_on"]
        )
        self.capture = IntervalCapture(env, self.io, cfg.params["force_on"])
        self.references = [ReferenceClip(path) for path in cfg.params["references"]]
        self.buffer.metadata.update(
            references=[
                {"path": str(r.path), "sha256": r.manifest["sha256"]} for r in self.references
            ],
            asset_manifest_sha256=sha256(DEFAULT_ASSET),
            sensors={
                name: {
                    "reduce": env.scene[name].cfg.reduce,
                    "num_slots": env.scene[name].cfg.num_slots,
                    "history_length": env.scene[name].cfg.history_length,
                    "secondary_policy": env.scene[name].cfg.secondary_policy,
                }
                for name in SENSOR_NAMES
            },
            normalization="none; raw physical units",
        )
        self.intents = [
            torch.tensor(r.arrays["object_contact"].copy(), device=env.device)
            for r in self.references
        ]
        n = env.num_envs
        self.ready = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.frame = torch.zeros(n, dtype=torch.long, device=env.device)
        self.clip = torch.zeros_like(self.frame)
        self.last_frame = torch.zeros_like(self.frame)
        self.state = torch.zeros(n, 84, device=env.device)
        self.xi = torch.zeros(n, 33, device=env.device)

    @torch.no_grad()
    def begin_episode(self, ids, frames, clips):
        """Bind metadata AFTER caller's synchronized robot/box reset, never reset state here."""
        if (
            frames.shape != ids.shape
            or clips.shape != ids.shape
            or frames.dtype != torch.long
            or clips.dtype != torch.long
        ):
            raise ValueError("One integer frame/clip per reset world required")
        if torch.any(clips < 0) or torch.any(clips >= len(self.references)):
            raise ValueError("Unknown reference clip")
        ends = torch.tensor([len(x) - 1 for x in self.intents], device=self._env.device)[clips]
        if torch.any((frames < 0) | (frames >= ends)):
            raise ValueError("Reset frame must have a valid outgoing transition")
        if self.ready[ids].any() or self.capture.count[ids].any():
            raise RuntimeError("Physical reset required before rebinding episode metadata")
        self.frame[ids], self.clip[ids], self.last_frame[ids] = frames, clips, ends
        self.state[ids], self.xi[ids] = self.io.read()[ids], self.ctx.read()[ids]
        self.ready[ids] = True

    def record_post_reset(self, env_ids):
        # Native explicit reset has no pre-reset callback. Close any open episode
        # without fabricating a physical-failure or terminal action label.
        live = env_ids[self.ready[env_ids]]
        if len(live):
            self.buffer.end_episode(live)
        self.ready[env_ids] = False
        self.capture.reset(env_ids)

    def _write(self, ids):
        if not len(ids):
            return
        env = self._env
        if not self.ready[ids].all():
            raise RuntimeError("No reference metadata for measured interval")
        next_state = self.capture.next_state[ids].clone()
        if not torch.equal(next_state, self.io.read()[ids]):
            raise RuntimeError("State changed after physics; interval target is ambiguous")
        labels = self.capture.consume(ids)
        clip_end = self.frame[ids] + 1 == self.last_frame[ids]
        failure = env.reset_terminated[ids].clone()
        # Read individual native timeout terms so a simultaneous timeout and
        # natural clip-end keep BOTH flags (the aggregate cannot distinguish them).
        timeout = torch.zeros_like(failure)
        for name, cfg in env.cfg.terminations.items():
            if name != CLIP_END and cfg is not None and cfg.time_out:
                timeout |= env.termination_manager.get_term(name)[ids]
        intent = torch.empty(len(ids), 2, device=env.device)
        for c, values in enumerate(self.intents):
            rows = self.clip[ids] == c
            f = self.frame[ids][rows]
            intent[rows] = torch.stack((values[f], values[f + 1]), dim=-1).float()
        self.buffer.append(
            ids,
            state=self.state[ids],
            next_state=next_state,
            action_clean=env.action_manager.action[ids],
            action_exec=env.action_manager.get_term("joint_pos").executed_raw[ids],
            xi=self.xi[ids],
            frame=self.frame[ids],
            clip=self.clip[ids],
            labels=labels,
            intent_ref=intent,
            physical_failure=failure,
            timeout=timeout,
            clip_end=clip_end,
        )
        self.state[ids] = next_state
        self.frame[ids] += 1
        self.ready[ids[failure | timeout | clip_end]] = False

    def record_pre_reset(self, env_ids):
        self._write(env_ids)

    def record_post_step(self):
        self._write((~self._env.reset_buf).nonzero().squeeze(-1))


def install_recording(env, *, force_on, capacity=1000, references=None):
    """Attach P2 hooks to a fresh P1 environment, before reset/collection.

    No shared source, step implementation, reward or training entry is changed.
    Existing metrics/termination definitions are preserved. State-mutating event
    and command paths need a separate timing contract before later integration.
    """
    if not env.cfg.auto_reset or env.cfg.commands or env.cfg.events or env.cfg.recorders:
        raise ValueError(
            "P2 collector requires auto-reset physical-only scene without commands/events/recorders"
        )
    if CLIP_END in env.cfg.terminations or NAME in env.cfg.metrics:
        raise ValueError("P2 hook name already registered")
    env.cfg.recorders = {
        NAME: RecorderTermCfg(
            func=HOIReplayRecorder,
            params={
                "capacity": capacity,
                "force_on": force_on,
                "references": references or [str(DEFAULT_OUTPUT / "reference.npz")],
            },
        )
    }
    env.recorder_manager = RecorderManager(env.cfg.recorders, env)
    env.cfg.metrics = {
        **env.cfg.metrics,
        NAME: MetricsTermCfg(func=capture_substep, per_substep=True),
    }
    env.metrics_manager = MetricsManager(env.cfg.metrics, env)
    env.cfg.terminations = {
        **env.cfg.terminations,
        CLIP_END: TerminationTermCfg(func=natural_clip_end, time_out=True),
    }
    env.termination_manager = TerminationManager(env.cfg.terminations, env)
    return env.recorder_manager.get_term(NAME)
