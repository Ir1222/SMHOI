"""Final shared dynamics trunk: canonical86/support18/action29/xi33 -> residual82."""

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from supermimic.hoi.state_spec import JointStateSpec, ObjectSpecCfg
from supermimic.hoi.support_context import SupportContext
from supermimic.world_model.dynamics import DynamicsMLP
from supermimic.world_model.heads import ReferenceFrame
from supermimic.world_model.world_model import TerminationHead, WorldModel, WorldModelCfg


@dataclass(frozen=True)
class LossWeights:
    robot: float
    object: float

    def __post_init__(self):
        values = (self.robot, self.object)
        if any(not math.isfinite(x) or x < 0 for x in values) or sum(values) == 0:
            raise ValueError("Loss weights must be finite, nonnegative and not all zero")


@dataclass
class JointWorldModelCfg(WorldModelCfg):
    object_spec: ObjectSpecCfg = field(default_factory=ObjectSpecCfg)
    weights: LossWeights | None = None
    support_context: bool = True


class JointWorldModel(WorldModel):
    def __init__(self, cfg: JointWorldModelCfg, heads, device):
        nn.Module.__init__(self)
        if cfg.weights is None:
            raise ValueError("Explicit robot/object loss weights required")
        if cfg.support_context is not True:
            raise ValueError("Final joint WM requires the approved support18 context")
        if not math.isfinite(cfg.term_loss_weight) or cfg.term_loss_weight < 0:
            raise ValueError("Invalid termination loss weight")
        self.cfg = cfg
        self.spec = JointStateSpec(cfg.state_spec, cfg.object_spec, device)
        self.support = SupportContext()
        self.dynamics = DynamicsMLP(feat_dim=104, hidden=cfg.hidden, out_dim=82)
        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
        self.heads = heads
        self.to(device)

    def predict(self, state, action_exec, xi):
        """Return next physical84; reference labels are outside the dynamics interface."""
        if state.shape[-1] != 84:
            raise ValueError("Prediction requires joint physical state84")
        if action_exec.shape != (*state.shape[:-1], 29) or xi.shape != (*state.shape[:-1], 33):
            raise ValueError("Joint prediction requires aligned action29 and episode xi33")
        if any(not torch.isfinite(x).all() for x in (state, action_exec, xi)):
            raise ValueError("Nonfinite joint dynamics input")
        features = torch.cat((self.spec.canonical(state), self.support(state)), -1)
        residual = self.dynamics(features, action_exec, self.spec.normalize_xi(xi))
        return self.spec.integrate(state, residual)

    def step(self, s, action, xi):
        return self.predict(s, action, xi)

    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat):
        s_next = self.predict(s, a, xi)
        physical = self.spec.error_vector(s_next, s_true).square()
        weights = self.cfg.weights
        components = torch.cat(
            (weights.robot * physical[..., :70], weights.object * physical[..., 70:]), -1
        )
        features = self.heads.termination_features(
            s_next[..., :3],
            s_next[..., 3:7],
            ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat),
        )
        logits = self.term_head(features)
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        return s_next, components, bce, logits

    def auxiliary_loss_metrics(self, components):
        return {
            "robot_weighted_loss": components[..., :70].sum(-1).mean(),
            "object_weighted_loss": components[..., 70:82].sum(-1).mean(),
        }

    def window_loss(self, states, a_exec, xi, done, refs, *, auxiliary_targets=None):
        if auxiliary_targets is not None:
            raise ValueError("Joint dynamics accepts physical targets only")
        k, n = a_exec.shape[:2]
        if (
            k < 1
            or states.shape != (k + 1, n, 84)
            or a_exec.shape != (k, n, 29)
            or xi.shape != (n, 33)
            or done.shape != (k, n)
            or len(refs) != k
        ):
            raise ValueError("Misaligned joint autoregressive targets")
        if not torch.isfinite(states).all() or not ((done == 0) | (done == 1)).all():
            raise ValueError("Invalid physical states or termination targets")
        # Empty mapping enables the existing named-metric hook, without supplying
        # auxiliary supervision or changing the original autoregressive loop.
        loss, metrics = super().window_loss(states, a_exec, xi, done, refs, auxiliary_targets={})
        metrics["weighted_physical_loss"] = metrics.pop("state_loss")
        metrics["weighted_component_mean_1step"] = metrics.pop("nse_1step")
        metrics["weighted_component_mean_kstep"] = metrics.pop("nse_kstep")
        return loss, metrics
