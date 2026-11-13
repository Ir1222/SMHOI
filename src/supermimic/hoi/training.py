"""HOI component construction and checkpoint hooks for the original Trainer.run."""

import json
import os
import random
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
from supermimic.hoi.reference import ReferenceClip, sha256
from supermimic.hoi.reward_reference import ClipBinding, HOIReferenceRegistry
from supermimic.hoi.rewards import ProximityReward
from supermimic.hoi.scene import PhysicalSceneCfg, make_physical_env
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
            ),
            device=self.device,
            manifest=manifest,
            configure=configure,
        )
        self._raw = raw
        self.guard.attest(raw, "construction_after_capture")
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
        self._validate_authorization()
        if cfg.initial_wm is not None and cfg.resume is None:
            from supermimic.hoi.checkpoint import load_joint_weights

            load_joint_weights(
                self.wm,
                torch.load(cfg.initial_wm, map_location=self.device, weights_only=True),
                expected_replay_metadata=self.buffer.metadata,
            )
        for module in (self.wm, self.policy, self.critic):
            broadcast_params(module)
        (self.log_dir / "identity.json").write_text(json.dumps(self.identity, indent=2) + "\n")
        return reward.max_step_reward

    def _validate_authorization(self):
        if self.cfg.execution == "integration":
            return
        approval = json.loads(self.cfg.approval_record.read_text())
        if (
            approval.get("stage") != "P7-B"
            or approval.get("authorized") is not True
            or approval.get("wm_readiness") != "PASS"
            or approval.get("identity") != self.identity
            or not approval.get("physical_limitations_accepted")
            or not approval.get("reward_and_sampler_calibration_accepted")
        ):
            raise ValueError("Missing matching P7-B authorization/readiness/disposition")
        source = self.cfg.resume or self.cfg.initial_wm
        if approval.get("checkpoint_sha256") != sha256(source):
            raise ValueError("Approved checkpoint identity mismatch")

    def save(self, name="ckpt_latest.pt"):
        if self.iteration in self.saved_iterations:
            return  # Original loop requests latest + numbered at the same boundary.
        path = self.log_dir / f"hoi_{self.iteration:07d}.pt"
        training_checkpoint.save_exclusive(self, path)
        self.saved_iterations.add(self.iteration)

    def load(self, path):
        training_checkpoint.load(self, path)

    def close(self):
        if hasattr(self, "_raw"):
            self._raw.close()
            del self._raw
        if getattr(self, "writer", None) is not None:
            self.writer.close()
