"""HOI component construction and checkpoint hooks for the original Trainer.run."""

import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.hoi import training_checkpoint
from supermimic.hoi.imagination import HOIPolicyTrainer
from supermimic.hoi.policy import HOIActor, HOICritic
from supermimic.hoi.policy_features import HOIObservationAdapter, feature_schema
from supermimic.hoi.recorder import install_recording
from supermimic.hoi.reference import ReferenceClip
from supermimic.hoi.reward_reference import ClipBinding, HOIReferenceRegistry
from supermimic.hoi.rewards import ProximityReward
from supermimic.hoi.scene import PhysicalSceneCfg, make_physical_env
from supermimic.hoi.startup import StartupProgress
from supermimic.hoi.state import JointStateIO
from supermimic.hoi.tracking_reward import HOITrackingReward
from supermimic.hoi.training_env import (
    TrainingEnvironment,
    TrainingRecorder,
    make_sampler,
    object_failure,
    robot_failure,
    time_limit,
)
from supermimic.hoi.training_safety import LearningSafety
from supermimic.hoi.training_scene import (
    compiled_model_identity,
    physical_identity,
    profile_manifest,
    validate_reference_binding,
)
from supermimic.hoi.wm_trainer import JointWorldModelTrainer
from supermimic.hoi.world_model import JointWorldModel
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.training.distributed import broadcast_params
from supermimic.training.loop import Trainer
from supermimic.world_model.heads import AnalyticHeads


class HOITrainer(Trainer):
    """No run/optimizer/rollout override: the original alternating lifecycle is used."""

    def __init__(self, cfg, guard):
        self.guard = guard
        self.saved_iterations = set()
        self.progress = StartupProgress()
        self.replay_readiness = {}
        try:
            super().__init__(cfg)
        except BaseException:
            if hasattr(self, "log_dir"):
                (self.log_dir / "construction_failure.txt").write_text(traceback.format_exc())
                (self.log_dir / "guard.json").write_text(
                    json.dumps(guard.evidence(), indent=2) + "\n"
                )
            self.close()
            raise

    def configure_task(self):
        self.cfg.validate()
        if not self.guard.active:
            raise RuntimeError("Construct HOITrainer inside the lifetime SolverGuard")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Exact HOI physical resume currently supports one process only")
        target = Path(self.cfg.log_root) / self.cfg.run_name
        if target.exists():
            raise FileExistsError(f"Refusing to reuse an existing run directory: {target}")
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def construct_components(self):
        cfg = self.cfg
        manifest = profile_manifest(cfg.profile)
        reference = ReferenceClip(cfg.reference)
        validate_reference_binding(cfg.profile, reference)
        self.identity = physical_identity(cfg.profile, manifest, reference, self.guard)
        self.identity["policy_schema"] = feature_schema()

        def configure(env_cfg):
            env_cfg.seed = cfg.seed
            env_cfg.episode_length_s = cfg.env.episode_length_s
            env_cfg.metrics["hoi_safety"] = MetricsTermCfg(
                func=lambda env: self.guard.monitor(env), per_substep=True
            )

        raw = make_physical_env(
            PhysicalSceneCfg(
                num_envs=cfg.env.num_envs,
                action_noise_std=cfg.env.action_noise_std,
                solver_iterations=cfg.solver_iterations,
            ),
            device=self.device,
            manifest=manifest,
            configure=configure,
        )
        self._raw = raw
        self.guard.attest(raw, "construction_after_capture")
        self.identity["iterations"] = int(raw.sim.mj_model.opt.iterations)
        if self.identity["iterations"] != cfg.solver_iterations:
            raise RuntimeError("Native solver iteration budget differs from config")
        self.identity["native_model"] = compiled_model_identity(raw)
        self.command = make_sampler(raw, cfg)
        self.lib = self.command.library
        robot = raw.scene["robot"]
        io = JointStateIO(raw)
        registry = HOIReferenceRegistry(
            self.lib,
            [ClipBinding(0, 0, reference)],
            library_sources=[reference.path],
            dt=0.02,
            joint_names=io.joint_names,
        )
        heads = AnalyticHeads(
            G1Kinematics(load_g1_model(), device=self.device),
            TRACKED_BODIES,
            ANCHOR_BODY,
            ROOT_BODY,
            robot.data.soft_joint_pos_limits[0].clone(),
            robot.data.default_joint_pos[0].clone(),
        )
        self.wm = JointWorldModel(cfg.world_model, heads, self.device)
        self.policy, self.critic = HOIActor().to(self.device), HOICritic().to(self.device)
        proximity = ProximityReward(
            heads.fk,
            ("left_wrist_yaw_link", "right_wrist_yaw_link"),
            manifest["contact"]["target_root_m"],
            manifest["contact"]["eef_offset_wrist_m"],
            sigma_pos=cfg.reward.sigma_pos,
        )
        reward = HOITrackingReward(heads, proximity, cfg.reward)
        self.wm_trainer = JointWorldModelTrainer(self.wm, [reference], cfg.wm_trainer)
        self.policy_trainer = HOIPolicyTrainer(
            self.wm, self.policy, self.critic, registry, reward, cfg.policy_trainer
        )
        self.learning_safety = LearningSafety(self.wm_trainer, self.policy_trainer)
        raw._hoi_components = {
            "config": cfg,
            "registry": registry,
            "wm": self.wm,
            "reward": reward,
            "critic": self.critic,
            "sampler": self.command,
            "identity": self.identity,
            "observation": HOIObservationAdapter(self.wm.spec, self.lib.default_joint_pos),
        }
        self.recorder = install_recording(
            raw,
            force_on=0.001,
            capacity=cfg.env.replay_capacity,
            references=[str(reference.path)],
            recorder_type=TrainingRecorder,
        )
        raw.cfg.terminations.update(
            {
                "robot_failure": TerminationTermCfg(func=robot_failure),
                "object_failure": TerminationTermCfg(func=object_failure),
                "hoi_timeout": TerminationTermCfg(func=time_limit, time_out=True),
            }
        )
        raw.termination_manager = TerminationManager(raw.cfg.terminations, raw)
        # mjlab normally creates these on its first step. Allocate the same
        # public buffers before loading a between-step checkpoint.
        raw.reset_buf = raw.termination_manager.dones
        raw.reset_terminated = raw.termination_manager.terminated
        raw.reset_time_outs = raw.termination_manager.time_outs
        raw.reward_buf = torch.zeros(raw.num_envs, device=self.device)
        self.buffer = self.recorder.buffer
        self.env = TrainingEnvironment(raw, self.recorder, self.guard)
        from supermimic.hoi.reset_diagnostics import inspect_reference_resets

        frames = (
            cfg.reset_frames if cfg.env.sampling_mode == "start" else range(reference.n_frames - 1)
        )
        self.reset_geometry = inspect_reference_resets(
            raw.sim.mj_model, self.recorder.cache, io, frames
        )
        (self.log_dir / "reset_geometry.json").write_text(
            json.dumps(self.reset_geometry, indent=2) + "\n"
        )
        if self.reset_geometry["frames_above_band"]:
            print(
                f"[HOI] UNRESOLVED reset overlap: {self.reset_geometry['frames_above_band']} frames "
                f"above 1 mm, maximum {self.reset_geometry['max_penetration_m'] * 1000:.3f} mm "
                f"at frame {self.reset_geometry['worst_frame']}; see reset_geometry.json. "
                "Numerical checks do not establish physical validity.",
                flush=True,
            )
        for module in (self.wm, self.policy, self.critic):
            broadcast_params(module)
        (self.log_dir / "identity.json").write_text(json.dumps(self.identity, indent=2) + "\n")
        return reward.max_step_reward

    def updates_ready(self):
        windows = int(self.buffer.window_mask(self.cfg.wm_trainer.window).sum())
        starts = int(self.buffer.start_mask(self.cfg.policy_trainer.history).sum())
        valid = self.buffer.valid & self.buffer.fields["transition_valid"]
        self.replay_readiness = {
            "windows8": windows,
            "policy_starts": starts,
            "retained_transitions": int(valid.sum()),
            **{
                k: int((valid & self.buffer.fields[k]).sum())
                for k in ("robot_failure", "object_failure", "timeout", "clip_end")
            },
        }
        try:
            return self.progress.ready(self.cfg, self.iteration, windows, starts)
        except RuntimeError:
            self.write_startup_status()
            raise
        finally:
            if self.iteration % self.cfg.schedule.log_interval == 0:
                self.write_startup_status()

    def policy_updates_enabled(self):
        return self.progress.phase == "alternating"

    def update_completed(self, wm_metrics, policy_metrics):
        from supermimic.hoi.training_safety import finite

        finite("update metrics", wm_metrics, policy_metrics)
        previous = self.progress.phase
        self.progress.completed(self.cfg)
        from supermimic.hoi.training_metrics import append_json

        append_json(self.log_dir / "learning.jsonl", {
            "iteration": self.iteration, "update": self.num_updates,
            "phase_executed": previous, "phase_next": self.progress.phase,
            "wm": wm_metrics, "policy": policy_metrics,
            "optimizer_steps": dict(self.learning_safety.steps),
            "gradient_norm_after_clipping": dict(self.learning_safety.gradient_norms),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        })
        if self.progress.phase != previous:
            print(
                f"[HOI] {previous} -> {self.progress.phase} at iteration {self.iteration}",
                flush=True,
            )
            self.write_startup_status()

    def write_startup_status(self):
        (self.log_dir / "startup.json").write_text(
            json.dumps(
                {
                    **self.progress.state_dict(),
                    "iteration": self.iteration,
                    "replay": self.replay_readiness,
                    "optimizer_steps": self.learning_safety.steps,
                    "meaning": "online startup scheduling; not a learned WM/task-success certificate",
                },
                indent=2,
            )
            + "\n"
        )

    def log_task_metrics(self):
        self.write_startup_status()
        from supermimic.hoi.training_metrics import (
            append_json,
            prediction_report,
            reached_geometry_report,
        )

        append_json(self.log_dir / "reached_geometry.jsonl", reached_geometry_report(self))
        if self.replay_readiness.get("windows8", 0):
            report = prediction_report(self)
            append_json(self.log_dir / "prediction.jsonl", report)
            for model, metrics in report["errors"].items():
                for key, value in metrics.items():
                    self.writer.add_scalar(f"prediction/{model}/{key}", value, self.iteration)
        for key, value in self.replay_readiness.items():
            self.writer.add_scalar("replay/" + key, value, self.iteration)
        for key, value in self.learning_safety.steps.items():
            self.writer.add_scalar("updates/" + key, value, self.iteration)
        self.writer.add_scalar(
            "startup/phase",
            ("collect", "wm_bootstrap", "alternating").index(self.progress.phase),
            self.iteration,
        )

    def online_error_ready(self):
        return self.buffer.size > 2048 and bool(self.buffer.window_mask(8).any())

    def save(self, name="ckpt_latest.pt"):
        if self.iteration in self.saved_iterations:
            return  # Original loop requests latest + numbered at the same boundary.
        path = self.log_dir / f"hoi_{self.iteration:07d}.pt"
        started = time.monotonic()
        training_checkpoint.save_exclusive(self, path)
        self.saved_iterations.add(self.iteration)
        from supermimic.hoi.training_metrics import append_json

        append_json(self.log_dir / "checkpoints.jsonl", {
            "iteration": self.iteration, "path": str(path), "bytes": path.stat().st_size,
            "save_seconds": time.monotonic() - started,
        })

    def load(self, path):
        training_checkpoint.load(self, path)

    def close(self):
        if hasattr(self, "_raw"):
            self._raw.close()
            del self._raw
        if getattr(self, "writer", None) is not None:
            self.writer.close()
