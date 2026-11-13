"""Separate HOI integration configuration; Robot training defaults are untouched."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from supermimic.config import EnvCfg, ScheduleCfg, TrainConfig
from supermimic.hoi.reference import REPO
from supermimic.hoi.tracking_reward import HOIRewardCfg
from supermimic.hoi.world_model import JointWorldModelCfg, LossWeights
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg

REFINED_REFERENCE = REPO / "outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz"
REFINED_SHA256 = "f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7"


def initial_reward():
    """Initial experiment scales, NOT fitted/calibrated learning results.

    Robot terms stay unchanged. Box tolerances are m/rad/m/s/rad/s;
    proximity's 5 cm is a rounded P4 distance-scale candidate.
    """
    return HOIRewardCfg((0.2, 0.4, 1.0, 2.0), (1.0, 1.0, 0.5, 0.5), 0.05, 1.0, 0.02)


@dataclass
class StartupCfg:
    # Counts are complete original WMTrainer.update calls, each with its own batches.
    wm_only_updates: int = 50
    min_windows: int = 32
    min_policy_starts: int = 32
    # Fail with counts/reasons instead of waiting forever for usable online data.
    max_collect_iterations: int = 2000


@dataclass
class HOITrainConfig(TrainConfig):
    run_name: str = "hoi_refined_scratch"
    reference: Path = REFINED_REFERENCE
    profile: Literal["refined-single-body", "production", "p6-original", "p6-single-body"] = (
        "refined-single-body"
    )
    execution: Literal["integration", "alternating"] = "alternating"
    reward: HOIRewardCfg = field(default_factory=initial_reward)
    # None derives the theoretical discounted upper bound from reward and gamma.
    sampler_g_max: float | None = None
    # Retained only to give an explicit error for the retired entry-point route.
    initial_wm: Path | None = None
    startup: StartupCfg = field(default_factory=StartupCfg)
    check_only: bool = False  # construct/reset/check inputs and geometry, no run or save
    reset_frames: tuple[int, ...] = (0,)
    # An optional, explicitly chosen geometric loss-of-object boundary. No head.
    box_min_height: float | None = None
    env: EnvCfg = field(
        default_factory=lambda: EnvCfg(
            num_envs=64,
            episode_length_s=16.8,
            action_noise_std=0.1,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="return",
            replay_capacity=256,
        )
    )
    world_model: JointWorldModelCfg = field(
        default_factory=lambda: JointWorldModelCfg(
            weights=LossWeights(1.0, 1.0),
        )
    )
    wm_trainer: WMTrainerCfg = field(
        default_factory=lambda: WMTrainerCfg(batch_size=256, compile=None)
    )
    policy_trainer: PolicyTrainerCfg = field(
        default_factory=lambda: PolicyTrainerCfg(
            batch_size=256,
            obs_noise=False,
            compile=None,
        )
    )
    schedule: ScheduleCfg = field(
        default_factory=lambda: ScheduleCfg(
            total_iterations=320_000,
            updates_per_env_step=0.4,
            warmup_iterations=40,
            log_interval=50,
            checkpoint_interval=4000,
        )
    )

    def validate(self):
        import math

        if self.reward is None:
            raise ValueError("HOI reward configuration is required")
        if self.sampler_g_max is not None and (
            not math.isfinite(self.sampler_g_max) or self.sampler_g_max <= 0
        ):
            raise ValueError("sampler_g_max must be finite and positive")
        if self.initial_wm is not None:
            raise ValueError(
                "HOI training starts from scratch; use --resume only for a complete HOI training state. Weight migration remains an offline tool."
            )
        if self.resume is not None and not self.resume.is_file():
            raise FileNotFoundError(f"Explicit resume checkpoint does not exist: {self.resume}")
        if self.execution not in ("integration", "alternating"):
            raise ValueError("Unknown HOI execution mode")
        if (
            min(
                self.startup.wm_only_updates,
                self.startup.min_windows,
                self.startup.min_policy_starts,
                self.startup.max_collect_iterations,
            )
            < 1
        ):
            raise ValueError("Startup counts must be positive")
        if (
            min(
                self.wm_trainer.batch_size,
                self.wm_trainer.batches_per_update,
                self.policy_trainer.batch_size,
                self.policy_trainer.horizon,
                self.policy_trainer.critic_epochs,
                self.policy_trainer.critic_minibatches,
            )
            < 1
        ):
            raise ValueError("Batch sizes, horizon and update counts must be positive")
        if self.policy_trainer.batch_size < self.policy_trainer.critic_minibatches:
            raise ValueError("Policy batch must cover all critic minibatches")
        if not 0 < self.policy_trainer.gamma < 1 or not 0 <= self.policy_trainer.gae_lambda <= 1:
            raise ValueError("Invalid discount or TD(lambda)")
        if not self.policy_trainer.use_critic:
            raise ValueError("HOI alternating training requires the approved Critic")
        for value in (self.wm_trainer.lr, self.policy_trainer.lr, self.policy_trainer.critic_lr):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Learning rates must be finite and positive")
        if self.reward.dt != 0.02 or self.world_model.state_spec.dt != 0.02:
            raise ValueError("Approved HOI control dt is .020 s")
        if self.world_model.weights is None or not self.world_model.support_context:
            raise ValueError("Joint WM requires explicit loss weights and support18")
        if (
            not math.isfinite(self.schedule.lr_final_ratio)
            or not 0 < self.schedule.lr_final_ratio <= 1
            or not math.isfinite(self.schedule.lr_decay_fraction)
            or not 0 < self.schedule.lr_decay_fraction <= 1
            or (
                self.schedule.max_hours is not None
                and (not math.isfinite(self.schedule.max_hours) or self.schedule.max_hours <= 0)
            )
        ):
            raise ValueError("Invalid cosine schedule or time budget")
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
            if self.env.num_envs > 2 or self.schedule.total_iterations > 16:
                raise ValueError("Zero-update integration is bounded to two worlds / 16 steps")
            if self.schedule.updates_per_env_step != 0:
                raise ValueError("Physical integration permits zero optimizer updates")
        elif self.schedule.updates_per_env_step <= 0:
            raise ValueError(
                "Alternating training requires positive UTD; use integration for zero updates"
            )
        elif self.box_min_height is not None:
            raise ValueError(
                "Box-only failure/imagination semantics remain an unresolved research gate"
            )

    @property
    def effective_sampler_g_max(self):
        from supermimic.envs.mdp.rewards import TrackingRewardCfg

        bound = TrackingRewardCfg().max_step_reward + self.reward.dt * (
            sum(self.reward.box_weights) + self.reward.beta_c
        )
        return (
            self.sampler_g_max
            if self.sampler_g_max is not None
            else bound / (1 - self.policy_trainer.gamma)
        )
