"""P4 reference-conditioned proximity only; no simulator sensing or rollout.

Geometry is a semantic proximity cue, not collision/contact probability. Explicit
targets and scales must be justified by a separate audit before task integration.
"""

from __future__ import annotations

import math

import torch

from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply, quat_normalize


def reference_contact_at(labels: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Exact evaluated-frame lookup; no lookahead, terminal clamping or padding.

    For a next-state reward the caller supplies t+1, not the action's start frame.
    ``labels`` is the unchanged global bool reference, never a measured label.
    """
    if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.bool:
        raise ValueError("Reference contact must be a nonempty 1-D bool tensor")
    if frames.dtype != torch.long or frames.device != labels.device:
        raise ValueError("Frame indices must be long and on the reference device")
    if torch.any((frames < 0) | (frames >= len(labels))):
        raise IndexError("Reward frame outside reference; padding is not a valid reward")
    return labels[frames]


def proximity_from_points(
    eef_pos: torch.Tensor,
    target_pos: torch.Tensor,
    contact_ref: torch.Tensor,
    *,
    sigma_pos: float,
) -> dict[str, torch.Tensor]:
    """Raw, unweighted c_ref * mean(exp(-||EEF-target|| / sigma)), in [0,1].

    Points are [...,Nc,3] in common axes/meters, gate is bool [...]. Sigma is
    explicit in meters. There is no implicit dt, force factor or OFF repulsion.
    At exact coincidence Torch's norm uses a finite zero subgradient.
    """
    if not math.isfinite(sigma_pos) or sigma_pos <= 0:
        raise ValueError("sigma_pos must be finite and positive")
    if eef_pos.ndim < 2 or eef_pos.shape[-1] != 3 or eef_pos.shape[-2] == 0:
        raise ValueError("Expected nonempty target set [...,Nc,3]")
    if target_pos.shape != eef_pos.shape or target_pos.device != eef_pos.device:
        raise ValueError("EEF and targets must have identical shape/device")
    if not eef_pos.is_floating_point() or target_pos.dtype != eef_pos.dtype:
        raise ValueError("EEF and targets must share a floating dtype")
    if contact_ref.dtype != torch.bool or contact_ref.shape != eef_pos.shape[:-2]:
        raise ValueError("Expected one global bool reference gate per evaluated state")
    if contact_ref.device != eef_pos.device:
        raise ValueError("Gate and geometry must share a device")
    if not torch.isfinite(eef_pos).all() or not torch.isfinite(target_pos).all():
        raise ValueError("Nonfinite contact geometry")
    distance = torch.linalg.vector_norm(eef_pos - target_pos, dim=-1)
    per_target = contact_ref[..., None] * torch.exp(-distance / sigma_pos)
    return {"distance": distance, "per_target": per_target, "proximity": per_target.mean(-1)}


class ProximityReward:
    """Differentiable native G1 FK + current Box geometry from physical84.

    No assets or defaults are loaded implicitly. Geometry constants are copied at
    construction in the FK's dtype/device. Construct another instance to change
    device/dtype; neither physical state nor Actor/Critic contracts are extended.
    """

    def __init__(
        self,
        fk: G1Kinematics,
        wrist_body_names: tuple[str, ...],
        target_root_m,
        eef_offset_wrist_m,
        *,
        sigma_pos: float,
    ):
        if fk.nq != 36 or fk.nv != 35:
            raise ValueError("Expected the unchanged native robot nq36/nv35 FK")
        if not wrist_body_names or len(set(wrist_body_names)) != len(wrist_body_names):
            raise ValueError("Expected unique, nonempty audited EEF body names")
        if not math.isfinite(sigma_pos) or sigma_pos <= 0:
            raise ValueError("sigma_pos must be finite and positive")
        self.fk, self.sigma_pos = fk, sigma_pos
        self.wrist_body_names = tuple(wrist_body_names)
        self.wrist_indices = fk.body_indices(wrist_body_names)
        self.target_root = (
            torch.as_tensor(target_root_m, device=fk.device, dtype=fk.dtype).detach().clone()
        )
        self.eef_offset = (
            torch.as_tensor(eef_offset_wrist_m, device=fk.device, dtype=fk.dtype).detach().clone()
        )
        for value in (self.target_root, self.eef_offset):
            if value.shape != (len(wrist_body_names), 3) or not torch.isfinite(value).all():
                raise ValueError("One finite local 3-vector required per audited EEF")

    def geometry(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape[-1] != 84 or not state.is_floating_point():
            raise ValueError("Expected joint physical state84")
        if state.device != self.fk.device or state.dtype != self.fk.dtype:
            raise ValueError("State dtype/device must match FK construction")
        if not torch.isfinite(state).all():
            raise ValueError("Nonfinite joint state")
        for q in (state[..., 3:7], state[..., 74:78]):
            if torch.any((q.norm(dim=-1) - 1).abs() > 1e-5):
                raise ValueError("Expected unit wxyz root quaternions")
        bodies = self.fk(state[..., :36])
        batch = state.shape[:-1]
        n = len(self.wrist_body_names)
        offsets = self.eef_offset.expand(*batch, n, 3)
        eef = bodies["pos"][..., self.wrist_indices, :] + quat_apply(
            bodies["quat"][..., self.wrist_indices, :], offsets
        )
        box_quat = quat_normalize(state[..., 74:78])[..., None, :].expand(*batch, n, 4)
        target = state[..., None, 71:74] + quat_apply(
            box_quat, self.target_root.expand(*batch, n, 3)
        )
        return eef, target

    def __call__(self, state: torch.Tensor, contact_ref: torch.Tensor) -> dict[str, torch.Tensor]:
        eef, target = self.geometry(state)
        return {
            "eef_pos": eef,
            "target_pos": target,
            **proximity_from_points(eef, target, contact_ref, sigma_pos=self.sigma_pos),
        }


def combine_tracking_and_proximity(
    *,
    robot: torch.Tensor,
    object: torch.Tensor,
    proximity: torch.Tensor,
    regularization: torch.Tensor,
    robot_weight: float,
    object_weight: float,
    contact_weight: float,
) -> torch.Tensor:
    """P4.5 additive composition of caller-supplied, equally scaled reward terms.

    Does not define or replace tracking equations. Robot/object weights stay
    positive regardless of the contact gate. Original robot totals already include
    regularization: supply zero here if using such a total, avoiding double counting.
    Original robot rewards include dt; raw proximity does not, so contact_weight
    must include any desired dt scaling. No final coefficient is prescribed.
    """
    weights = (robot_weight, object_weight, contact_weight)
    if any(not math.isfinite(w) for w in weights) or min(weights[:2]) <= 0 or contact_weight < 0:
        raise ValueError("Tracking weights must be positive; contact weight nonnegative")
    if any(
        t.shape != robot.shape or t.device != robot.device or t.dtype != robot.dtype
        for t in (object, proximity, regularization)
    ):
        raise ValueError("Reward terms must share shape, dtype and device")
    return (
        robot_weight * robot + object_weight * object + contact_weight * proximity + regularization
    )
