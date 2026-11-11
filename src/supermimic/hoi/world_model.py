"""One shared dynamics trunk: robot70/object12/next-reference-intent logit1."""

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


def augment_states(states, intent_ref):
    """Adapt physical Replay windows without changing their storage or terminal rows."""
    if states.ndim != 3 or states.shape[-1] != 84:
        raise ValueError("Expected physical states [K+1,N,84]")
    k, n = states.shape[0] - 1, states.shape[1]
    if k < 1 or intent_ref.shape != (k, n, 2):
        raise ValueError("Expected aligned intent_ref [K,N,2]")
    if not ((intent_ref == 0) | (intent_ref == 1)).all():
        raise ValueError("Reference contact intent must be finite and binary")
    if not torch.equal(intent_ref[:-1, :, 1], intent_ref[1:, :, 0]):
        raise ValueError("Discontinuous reference contact pairs")
    contact = torch.cat((intent_ref[0, :, :1][None], intent_ref[..., 1:2]), 0)
    return torch.cat((states, contact.to(states)), -1)


@dataclass(frozen=True)
class LossWeights:
    robot: float
    object: float
    contact: float

    def __post_init__(self):
        values = (self.robot, self.object, self.contact)
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
            raise ValueError("Explicit robot/object/contact loss weights required")
        if cfg.support_context is not True:
            raise ValueError("Revised P3 requires the approved support18 context")
        if not math.isfinite(cfg.term_loss_weight) or cfg.term_loss_weight < 0:
            raise ValueError("Invalid termination loss weight")
        self.cfg = cfg
        self.spec = JointStateSpec(cfg.state_spec, cfg.object_spec, device)
        self.support = SupportContext()
        self.dynamics = DynamicsMLP(feat_dim=105, hidden=cfg.hidden, out_dim=83)
        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
        self.heads = heads
        self.to(device)

    def predict(self, state, action_exec, xi):
        if state.shape[-1] != 85:
            raise ValueError("Prediction requires physical84 plus contact probability1")
        if action_exec.shape != (*state.shape[:-1], 29) or xi.shape != (*state.shape[:-1], 33):
            raise ValueError("Joint prediction requires aligned action29 and episode xi33")
        contact = state[..., 84:85]
        if not torch.isfinite(state).all() or ((contact < 0) | (contact > 1)).any():
            raise ValueError("Invalid predictive state/contact probability")
        physical = state[..., :84]
        features = torch.cat((self.spec.canonical(physical), contact, self.support(physical)), -1)
        output = self.dynamics(features, action_exec, self.spec.normalize_xi(xi))
        next_physical = self.spec.integrate(physical, output[..., :82])
        logit = output[..., 82:83]
        return torch.cat((next_physical, logit.sigmoid()), -1), logit

    def step(self, s, action, xi):
        return self.predict(s, action, xi)[0]

    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat, contact_intent_next):
        s_next, contact_logit = self.predict(s, a, xi)
        physical = self.spec.error_vector(s_next[..., :84], s_true[..., :84]).square()
        contact_bce = F.binary_cross_entropy_with_logits(
            contact_logit, contact_intent_next, reduction="none"
        )
        weights = self.cfg.weights
        components = torch.cat(
            (
                weights.robot * physical[..., :70],
                weights.object * physical[..., 70:],
                weights.contact * contact_bce,
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

    def auxiliary_loss_metrics(self, components):
        result = {
            "robot_weighted_loss": components[..., :70].sum(-1).mean(),
            "object_weighted_loss": components[..., 70:82].sum(-1).mean(),
            "contact_weighted_loss": components[..., 82].mean(),
        }
        if self.cfg.weights.contact > 0:
            result["contact_bce"] = result["contact_weighted_loss"] / self.cfg.weights.contact
        return result

    def window_loss(self, states, a_exec, xi, done, refs, *, auxiliary_targets=None):
        if auxiliary_targets is None or set(auxiliary_targets) != {"contact_intent_next"}:
            raise ValueError("Next reference contact intent supervision required")
        k, n = a_exec.shape[:2]
        target = auxiliary_targets["contact_intent_next"]
        if (
            k < 1
            or states.shape != (k + 1, n, 85)
            or a_exec.shape != (k, n, 29)
            or xi.shape != (n, 33)
            or done.shape != (k, n)
            or target.shape != (k, n, 1)
            or len(refs) != k
        ):
            raise ValueError("Misaligned joint autoregressive targets")
        if not ((states[..., 84] == 0) | (states[..., 84] == 1)).all():
            raise ValueError("Logged contact intent must be binary")
        if not torch.equal(target, states[1:, :, 84:85]):
            raise ValueError("Next contact target/state alignment failed")
        loss, metrics = super().window_loss(
            states, a_exec, xi, done, refs, auxiliary_targets=auxiliary_targets
        )
        metrics["combined_prediction_loss"] = metrics.pop("state_loss")
        metrics["weighted_component_mean_1step"] = metrics.pop("nse_1step")
        metrics["weighted_component_mean_kstep"] = metrics.pop("nse_kstep")
        return loss, metrics

    @torch.no_grad()
    def rollout_error(self, states, a_exec, xi):
        """Physical NSE only; the recurrent 85th value remains a probability."""
        state = states[0]
        errors = []
        for j, action in enumerate(a_exec):
            state = self.step(state, action, xi)
            errors.append(
                self.spec.normalized_sq_error(state[..., :84], states[j + 1, ..., :84]).mean()
            )
        return torch.stack(errors)
