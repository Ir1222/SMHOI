"""One original DynamicsMLP trunk, robot70/object12/auxiliary force6 heads."""

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from supermimic.hoi.force import ForceStatistics
from supermimic.hoi.state_spec import JointStateSpec, ObjectSpecCfg
from supermimic.hoi.support_context import SupportContext
from supermimic.world_model.dynamics import DynamicsMLP
from supermimic.world_model.heads import ReferenceFrame
from supermimic.world_model.world_model import TerminationHead, WorldModel, WorldModelCfg


@dataclass(frozen=True)
class LossWeights:
    robot: float
    object: float
    force: float

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in (self.robot, self.object, self.force)):
            raise ValueError("Loss weights must be finite and nonnegative")
        if self.robot + self.object + self.force == 0:
            raise ValueError("At least one regression component must be active")


@dataclass
class JointWorldModelCfg(WorldModelCfg):
    object_spec: ObjectSpecCfg = field(default_factory=ObjectSpecCfg)
    weights: LossWeights | None = None  # Choose after measuring component scales.
    force_loss: str = "huber"
    force_delta: float = 1.0  # Dimensionless normalized Huber transition.
    support_context: bool = True  # False only for explicit ablation.


class JointWorldModel(WorldModel):
    def __init__(self, cfg: JointWorldModelCfg, heads, force_statistics: ForceStatistics, device):
        nn.Module.__init__(self)
        if cfg.weights is None:
            raise ValueError("Explicit measured-scale loss weights are required")
        if cfg.force_loss not in ("huber", "l1") or not 0 < cfg.force_delta < float("inf"):
            raise ValueError("Invalid force regression loss")
        if not 0 <= cfg.term_loss_weight < float("inf"):
            raise ValueError("Invalid termination loss weight")
        self.cfg = cfg
        self.spec = JointStateSpec(cfg.state_spec, cfg.object_spec, device)
        self.support = SupportContext() if cfg.support_context else None
        self.force_statistics = force_statistics
        # Distinct output rows are three linear heads on exactly the same hidden trunk.
        self.dynamics = DynamicsMLP(
            feat_dim=86 + (18 if self.support else 0), hidden=cfg.hidden, out_dim=88
        )
        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
        self.heads = heads
        self.to(device)

    def predict(self, state, action_exec, xi):
        if action_exec.shape != (*state.shape[:-1], 29) or xi.shape != (*state.shape[:-1], 33):
            raise ValueError("Joint prediction requires aligned action29 and episode xi33")
        features = self.spec.canonical(state)
        if self.support is not None:
            features = torch.cat((features, self.support(state)), -1)
        output = self.dynamics(features, action_exec, self.spec.normalize_xi(xi))
        next_state = self.spec.integrate(state, output[..., :82])
        force_normalized = output[..., 82:88].reshape(*state.shape[:-1], 2, 3)
        return next_state, self.force_statistics.decode(force_normalized)

    def step(self, s, action, xi):
        return self.predict(s, action, xi)[0]

    def _loss_step(
        self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat, force_mean_box, occupancy
    ):
        s_next, force = self.predict(s, a, xi)
        physical = self.spec.error_vector(s_next, s_true).square()
        robust = self.force_statistics.loss_components(
            self.force_statistics.encode(force),
            force_mean_box,
            occupancy,
            kind=self.cfg.force_loss,
            delta=self.cfg.force_delta,
        ).flatten(-2)
        weights = self.cfg.weights
        components = torch.cat(
            (
                weights.robot * physical[..., :70],
                weights.object * physical[..., 70:],
                weights.force * robust,
            ),
            -1,
        )
        features = self.heads.termination_features(
            s_next[..., :3],
            s_next[..., 3:7],
            ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat),
        )
        logits = self.term_head(features)
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        return s_next, components, bce, logits

    @staticmethod
    def auxiliary_loss_metrics(components):
        return {
            "robot_weighted_loss": components[..., :70].sum(-1).mean(),
            "object_weighted_loss": components[..., 70:82].sum(-1).mean(),
            "force_weighted_loss": components[..., 82:].sum(-1).mean(),
        }

    def window_loss(self, states, a_exec, xi, done, refs, *, auxiliary_targets=None):
        if auxiliary_targets is None or set(auxiliary_targets) != {"force_mean_box", "occupancy"}:
            raise ValueError(
                "Measured force and occupancy targets required; intent is not an input"
            )
        k, n = a_exec.shape[:2]
        if (
            states.shape != (k + 1, n, 84)
            or done.shape != (k, n)
            or auxiliary_targets["force_mean_box"].shape != (k, n, 2, 3)
            or auxiliary_targets["occupancy"].shape != (k, n, 2)
        ):
            raise ValueError("Misaligned joint autoregressive targets")
        loss, metrics = super().window_loss(
            states, a_exec, xi, done, refs, auxiliary_targets=auxiliary_targets
        )
        # In the base loop sq now includes robust force loss: never label it NSE.
        metrics["weighted_regression_loss"] = metrics.pop("state_loss")
        metrics["weighted_component_mean_1step"] = metrics.pop("nse_1step")
        metrics["weighted_component_mean_kstep"] = metrics.pop("nse_kstep")
        return loss, metrics
