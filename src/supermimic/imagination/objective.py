"""Policy objective (Eq. 2) and critic GAE targets (Eq. 3)."""

from __future__ import annotations

import torch

from supermimic.imagination.rollout import ImagRollout


def policy_objective(ro: ImagRollout, terminal_values: torch.Tensor | None, gamma: float) -> torch.Tensor:
    """J = mean_n [ sum_t gamma^t c_t r_{t+1} + gamma^H c_H V_H ] (SHAC-style; V_H keeps the gradient w.r.t. s_H).

    terminal_values: (n,) critic estimate at s_H, or None for the critic-free variant.
    """
    H = ro.rewards.shape[0]
    disc = gamma ** torch.arange(H, device=ro.rewards.device, dtype=ro.rewards.dtype)
    if ro.valid is None:
        J = (disc[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
    else:
        # Select before arithmetic: even arbitrary nonfinite padding is not a sample.
        rewards = torch.where(ro.valid, ro.rewards, 0)
        survival = torch.where(ro.valid, ro.survival[:-1], 0)
        J = (disc[:, None] * survival * rewards).sum(0)
    if terminal_values is not None:
        if ro.lengths is None:
            J = J + (gamma**H) * ro.survival[-1] * terminal_values
        else:
            rows = torch.arange(J.shape[0], device=J.device)
            tail = torch.where(ro.bootstrap, terminal_values, 0)
            J = J + gamma**ro.lengths.to(ro.rewards.dtype) * ro.survival[ro.lengths, rows] * tail
    return J.mean()


@torch.no_grad()
def gae_targets(rewards: torch.Tensor, dhat_next: torch.Tensor, values: torch.Tensor, gamma: float, lam: float, *, valid=None, lengths=None, bootstrap=None, continuation=None) -> torch.Tensor:
    """R_t = r_{t+1} + gamma (1 - d_{t+1}) [ (1-lam) V_{t+1} + lam R_{t+1} ],  R_H = V_H.

    rewards (H, n), dhat_next (H, n) = d_{t+1}, values (H+1, n) = V_0..V_H. Returns (H, n) targets.
    """
    H = rewards.shape[0]
    if valid is not None:
        if lengths is None or bootstrap is None or continuation is None:
            raise ValueError("Masked TD requires lengths, bootstrap and continuation")
        rows = torch.arange(rewards.shape[1], device=rewards.device)
        boundary = torch.where(bootstrap, values[lengths, rows], 0)
        R = boundary.clone()
    else:
        R = values[H].clone()
    out = torch.empty_like(rewards)
    for t in range(H - 1, -1, -1):
        r, d, v, recursive = rewards[t], dhat_next[t], values[t + 1], R
        if valid is not None:
            active = valid[t]
            r = torch.where(active, r, 0)
            d = torch.where(active, d, 0)
            v = torch.where(continuation[t], v, 0)
            recursive = torch.where(continuation[t], recursive, 0)
        R = r + gamma * (1.0 - d) * ((1.0 - lam) * v + lam * recursive)
        if valid is not None:
            R = torch.where(active, R, boundary)
            out[t] = torch.where(active, R, 0)
        else:
            out[t] = R
    return out
