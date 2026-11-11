"""Physical-N Replay labels; train-only reversible normalization and robust loss."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ForceStatistics(nn.Module):
    """Per-hand isotropic scaling preserves force direction and physical zero.

    active = measured occupancy > 0, not reference intent or mean-force threshold.
    Class weights use training distribution, independently for the two hands.
    """

    def __init__(self, scale, counts, median_active_N, *, force_on_N, active_mass):
        super().__init__()
        scale = torch.as_tensor(scale).detach().clone().reshape(2, 1)
        counts = torch.as_tensor(counts, device=scale.device).detach().clone().reshape(2, 2)
        median = torch.as_tensor(median_active_N, device=scale.device).detach().clone().reshape(2)
        if (
            not torch.isfinite(scale).all()
            or (scale <= 0).any()
            or not torch.isfinite(counts).all()
            or (counts <= 0).any()
            or not torch.isfinite(median).all()
            or (median < 0).any()
            or not 0 < active_mass < 1
            or not 0 < force_on_N < float("inf")
        ):
            raise ValueError(
                "Finite positive scales/counts/threshold and both measured classes required"
            )
        self.register_buffer("scale_N", scale)
        self.register_buffer("counts_inactive_active", counts)
        self.register_buffer("median_active_N", median)
        masses = torch.tensor(
            [1 - active_mass, active_mass], device=scale.device, dtype=scale.dtype
        )
        self.register_buffer("class_weights", masses * counts.sum(-1, keepdim=True) / counts)
        self.register_buffer("force_on_N", torch.tensor(force_on_N, device=scale.device))
        self.active_mass = active_mass

    @classmethod
    def fit(cls, force_N, occupancy, *, force_on_N, active_mass, scale_floor_N=1e-6):
        if force_N.shape[-2:] != (2, 3) or occupancy.shape != force_N.shape[:-1]:
            raise ValueError("Expected force [...,2,3] and measured occupancy [...,2]")
        if (
            not torch.isfinite(force_N).all()
            or not torch.isfinite(occupancy).all()
            or ((occupancy < 0) | (occupancy > 1)).any()
            or not 0 < scale_floor_N < float("inf")
        ):
            raise ValueError("Invalid measured force/occupancy/scale floor")
        values, active = force_N.reshape(-1, 2, 3), occupancy.reshape(-1, 2) > 0
        counts = torch.stack(((~active).sum(0), active.sum(0)), -1)
        if (counts == 0).any():
            raise ValueError("Training data must contain active and inactive labels for each hand")
        scale = torch.stack(
            [
                values[:, h][active[:, h]].square().mean().sqrt().clamp_min(scale_floor_N)
                for h in range(2)
            ]
        )
        median = torch.stack([values[:, h][active[:, h]].norm(dim=-1).median() for h in range(2)])
        return cls(scale, counts, median, force_on_N=force_on_N, active_mass=active_mass)

    def encode(self, force_N):
        if force_N.shape[-2:] != (2, 3):
            raise ValueError("Force hand order must be [...,left/right,xyz]")
        return force_N / self.scale_N

    def decode(self, normalized):
        if normalized.shape[-2:] != (2, 3):
            raise ValueError("Normalized force must be [...,2,3]")
        return normalized * self.scale_N

    def loss_components(self, normalized_prediction, target_N, occupancy, *, kind, delta):
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("Positive finite robust loss delta required")
        target = self.encode(target_N)
        if normalized_prediction.shape != target.shape or occupancy.shape != target.shape[:-1]:
            raise ValueError("Force prediction/label/occupancy shapes differ")
        if kind == "huber":
            losses = F.huber_loss(normalized_prediction, target, reduction="none", delta=delta)
        elif kind == "l1":
            losses = (normalized_prediction - target).abs()
        else:
            raise ValueError("Supported force losses: huber, l1")
        weights = torch.where(occupancy > 0, self.class_weights[:, 1], self.class_weights[:, 0])
        # Sum of six components is the class-balanced mean over hands and axes.
        return losses * weights[..., None] / 6
