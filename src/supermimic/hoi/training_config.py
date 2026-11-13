"""Separate HOI integration configuration; Robot training defaults are untouched."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from supermimic.config import EnvCfg, ScheduleCfg, TrainConfig
from supermimic.hoi.reference import DEFAULT_OUTPUT
from supermimic.hoi.tracking_reward import HOIRewardCfg
from supermimic.hoi.world_model import JointWorldModelCfg, LossWeights
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg


@dataclass
class HOITrainConfig(TrainConfig):
    run_name: str = "hoi_integration"
    reference: Path = DEFAULT_OUTPUT / "reference.npz"
    profile: Literal["production", "p6-original", "p6-single-body"] = "production"
    execution: Literal["integration", "alternating"] = "integration"
    # Explicit experiment coefficients, never selected silently as training defaults.
    reward: HOIRewardCfg | None = None
    sampler_g_max: float | None = None
    approval_record: Path | None = None
    initial_wm: Path | None = None
    reset_frames: tuple[int, ...] = (0,)
    # An optional, explicitly chosen geometric loss-of-object boundary. No head.
    box_min_height: float | None = None
    env: EnvCfg = field(
        default_factory=lambda: EnvCfg(
            num_envs=2,
            episode_length_s=0.32,
            action_noise_std=0.0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="start",
            replay_capacity=96,
        )
    )
    world_model: JointWorldModelCfg = field(
        default_factory=lambda: JointWorldModelCfg(
            weights=LossWeights(1.0, 1.0),
        )
    )
    wm_trainer: WMTrainerCfg = field(default_factory=lambda: WMTrainerCfg(compile=None))
    policy_trainer: PolicyTrainerCfg = field(
        default_factory=lambda: PolicyTrainerCfg(
            obs_noise=False,
            compile=None,
        )
    )
    schedule: ScheduleCfg = field(
        default_factory=lambda: ScheduleCfg(
            total_iterations=16,
            updates_per_env_step=0.0,
            warmup_iterations=40,
            log_interval=16,
            checkpoint_interval=16,
        )
    )

    def validate(self):
        import math

        if self.reward is None or self.sampler_g_max is None:
            raise ValueError("Explicit reward coefficients and sampler_g_max are required")
        if not math.isfinite(self.sampler_g_max) or self.sampler_g_max <= 0:
            raise ValueError("sampler_g_max must be finite and positive")
        if self.reward.dt != 0.02:
            raise ValueError("Approved HOI control dt is .020 s")
        if self.env.reset_noise or self.env.domain_randomization or self.env.anchor_obs:
            raise ValueError("This pinned HOI profile requires synchronized unperturbed resets")
        if self.env.future_frames != 10 or self.policy_trainer.history != 10:
            raise ValueError("Approved history/future widths are fixed at ten")
        if self.policy_trainer.future_frames != 10 or self.policy_trainer.anchor_obs:
            raise ValueError("Approved Actor B goals require ten frames without anchor")
        if self.policy_trainer.obs_noise != self.env.obs_noise:
            raise ValueError("Physical and imagined observation noise must agree")
        if (
            not math.isfinite(self.env.episode_length_s)
            or self.env.episode_length_s < 0.02
            or self.env.replay_capacity < 12
        ):
            raise ValueError("Invalid timeout or Replay capacity")
        if self.env.sampling_mode not in ("start", "uniform", "return"):
            raise ValueError("Unknown original sampling mode")
        if not math.isfinite(self.env.action_noise_std) or self.env.action_noise_std < 0:
            raise ValueError("Invalid executed-action noise")
        if self.env.compact_library or self.data != TrainConfig().data:
            raise ValueError(
                "HOI uses its explicit full reference, not Robot data/library overrides"
            )
        if self.policy != TrainConfig().policy or self.critic != TrainConfig().critic:
            raise ValueError(
                "Robot policy configuration does not configure approved Actor B/Critic B+"
            )
        if (
            tuple(self.world_model.hidden) != (2048, 2048, 2048)
            or self.world_model.height_threshold != 0.25
            or self.world_model.tilt_threshold != 0.8
        ):
            raise ValueError("Approved Joint WM widths and Robot termination thresholds are fixed")
        if (
            not math.isfinite(self.schedule.updates_per_env_step)
            or self.schedule.updates_per_env_step < 0
            or self.schedule.warmup_iterations < 0
        ):
            raise ValueError("Invalid update schedule")
        if self.wm_trainer.window != 8:
            raise ValueError("Approved WM windows contain eight transitions")
        if self.env.num_envs < 1 or not self.device.startswith("cuda"):
            raise ValueError("Physical HOI integration requires CUDA worlds")
        if not self.reset_frames or min(self.reset_frames) < 0:
            raise ValueError("Explicit valid reset frames required")
        if self.box_min_height is not None and not math.isfinite(self.box_min_height):
            raise ValueError("Box failure height must be finite")
        if (
            self.schedule.total_iterations < 1
            or min(
                self.schedule.log_interval,
                self.schedule.checkpoint_interval,
                self.schedule.policy_updates_per_wm_update,
            )
            < 1
        ):
            raise ValueError("Invalid lifecycle schedule")
        if self.execution == "integration":
            if self.env.num_envs > 4 or self.schedule.total_iterations > 64:
                raise ValueError("P7.0.4 integration is bounded to four worlds / 64 steps")
            if self.schedule.updates_per_env_step != 0:
                raise ValueError("P7.0.4 physical integration permits zero optimizer updates")
        elif self.approval_record is None or (self.initial_wm is None and self.resume is None):
            raise ValueError("Alternating execution needs approved WM readiness and checkpoint")
        elif self.box_min_height is not None:
            raise ValueError(
                "Box-only failure/imagination semantics remain an unresolved research gate"
            )
