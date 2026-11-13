"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.rollout import ImagRollout, imagine
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.world_model import WorldModel


@contextmanager
def frozen_parameters(*modules):
    """Freeze weights, retain input Jacobians, and restore even after exceptions."""
    parameters = list(dict.fromkeys(p for module in modules for p in module.parameters()))
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"
    anchor_obs: bool = False  # None | "default" | "reduce-overhead" | "max-autotune-no-cudagraphs"


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, lib: MotionLibrary, cfg: PolicyTrainerCfg, *, step_fn=None, rollout_context_factory=None, critic_features=None, observation_adapter=None, object_goal_builder=None, rollout_critic_inputs=None):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        self.numerics_check = None  # Optional task guard; no change to Robot defaults.
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        from supermimic.imagination.compiled import ImagStep

        self.step_fn = step_fn if step_fn is not None else ImagStep(wm, lib.default_joint_pos, compile_mode=cfg.compile)
        self.rollout_context_factory = rollout_context_factory
        self.critic_features = critic_features
        self.observation_adapter = observation_adapter
        self.object_goal_builder = object_goal_builder
        self.rollout_critic_inputs = rollout_critic_inputs

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = lr

    def _critic_inputs(self, ro: ImagRollout):
        if self.rollout_critic_inputs is not None:
            return self.rollout_critic_inputs(ro, ro.context)
        H1, n, _ = ro.states.shape
        feat = (self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
                if self.critic_features is None else self.critic_features(ro.states))
        return feat, ro.goals, ro.e_root

    def actor_loss(self, start):
        """Build the original BPTT objective without backward or optimizer updates.

        Caller holds frozen_parameters(wm, critic) through backward. This same
        method is used by update and by bounded integration/gradient validation.
        """
        cfg = self.cfg
        xi_norm = self.wm.spec.normalize_xi(start.xi)
        context = None if self.rollout_context_factory is None else self.rollout_context_factory(start, cfg.horizon)
        ro = imagine(self.wm, self.policy, self.lib, start, cfg.horizon, cfg.future_frames, obs_noise=cfg.obs_noise, step_fn=self.step_fn, anchor_obs=cfg.anchor_obs, context=context, observation_adapter=self.observation_adapter, object_goal_builder=self.object_goal_builder)
        feat, goals, e_root = self._critic_inputs(ro)
        if cfg.use_critic:
            if ro.lengths is None:
                v_H = self.critic(feat[-1], goals[-1], xi_norm, e_root[-1])
            else:
                rows = ro.bootstrap.nonzero().squeeze(-1)
                v_H = feat.new_zeros(feat.shape[1])
                if rows.numel():
                    tail = self.critic(feat[ro.lengths[rows], rows], goals[ro.lengths[rows], rows], xi_norm[rows], e_root[ro.lengths[rows], rows])
                    v_H = v_H.index_copy(0, rows, tail)
        else:
            v_H = None
        J = policy_objective(ro, v_H, cfg.gamma)
        return -J / cfg.horizon, J, ro, (feat, goals, e_root), xi_norm

    @torch.no_grad()
    def critic_batch(self, ro, xi_norm, inputs=None):
        """Fixed TD targets and detached valid pre-transition samples only."""
        feat, goals, e_root = self._critic_inputs(ro) if inputs is None else inputs
        feat_d, goals_d, e_root_d = feat.detach(), goals.detach(), e_root.detach()
        H1, n, _ = feat_d.shape
        xi_rep = xi_norm.detach()[None].expand(H1, n, -1)
        if ro.valid is None:
            values = self.critic(feat_d.reshape(H1 * n, -1), goals_d.reshape(H1 * n, -1), xi_rep.reshape(H1 * n, -1), e_root_d.reshape(H1 * n, -1)).reshape(H1, n)
        else:
            state_valid = torch.arange(H1, device=feat.device)[:, None] <= ro.lengths[None]
            values = feat.new_zeros(H1, n)
            values[state_valid] = self.critic(feat_d[state_valid], goals_d[state_valid], xi_rep[state_valid], e_root_d[state_valid])
        targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values, self.cfg.gamma, self.cfg.gae_lambda,
            valid=ro.valid, lengths=ro.lengths, bootstrap=ro.bootstrap, continuation=ro.continuation)
        X = tuple(x[:-1].reshape(-1, x.shape[-1]) for x in (feat_d, goals_d, xi_rep, e_root_d))
        Y = targets.reshape(-1)
        if ro.valid is not None:
            mask = ro.valid.reshape(-1)
            X = tuple(x[mask] for x in X)
            Y = Y[mask]
        return X, Y, values, targets

    def critic_minibatch_indices(self, count, device, *, masked=False):
        """Keep original default partition; finite masked batches include all rows."""
        perm = torch.randperm(count, device=device)
        if masked:
            if count < 1 or self.cfg.critic_minibatches < 1:
                raise ValueError("Critic needs a nonempty valid batch and positive minibatch count")
            return perm.tensor_split(min(count, self.cfg.critic_minibatches))
        mb = count // self.cfg.critic_minibatches
        return tuple(perm[i * mb : (i + 1) * mb] for i in range(self.cfg.critic_minibatches))

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size, cfg.history)

        # ---- actor: BPTT through the frozen world model ------------------------------------
        with frozen_parameters(self.wm, self.critic):
            loss, J, ro, inputs, xi_norm = self.actor_loss(start)
            if self.numerics_check is not None:
                self.numerics_check("Actor loss", loss)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            allreduce_grads(self.policy)
            gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
            if self.numerics_check is not None:
                self.numerics_check("Actor gradient norm", gn)
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
            self.opt.step()

        metrics = {
            "imag_objective": float(J.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(gn),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
        }
        for k, v in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = v
        if ro.valid is not None:
            metrics.update(imag_reward_mean=float(ro.rewards[ro.valid].mean().detach()),
                imag_dhat_mean=float(ro.dhat[1:][ro.valid].mean().detach()),
                action_abs_mean=float(ro.actions[ro.valid].abs().mean().detach()),
                imag_survival_mean=float(ro.survival[:-1][ro.valid].mean().detach()),
                imag_terminal_survival_mean=float(ro.survival[ro.lengths, torch.arange(ro.lengths.numel(), device=ro.lengths.device)].mean().detach()))

        # ---- critic: fit to GAE targets of this rollout (targets fixed) ---------------------
        if cfg.use_critic:
            X, Y, values, targets = self.critic_batch(ro, xi_norm, inputs)
            N = Y.shape[0]
            last = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                for idx in self.critic_minibatch_indices(N, Y.device, masked=ro.valid is not None):
                    pred = self.critic(*(x[idx] for x in X))
                    closs = torch.nn.functional.mse_loss(pred, Y[idx])
                    if self.numerics_check is not None:
                        self.numerics_check("Critic loss", closs)
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    critic_gn = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                    if self.numerics_check is not None:
                        self.numerics_check("Critic gradient norm", critic_gn)
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = float(last)
            metrics["critic_value_mean"] = float(values.mean())
            metrics["critic_target_mean"] = float(targets.mean())
            if ro.valid is not None:
                metrics["critic_target_mean"] = float(Y.mean())
                state_valid = torch.arange(values.shape[0], device=values.device)[:, None] <= ro.lengths[None]
                metrics["critic_value_mean"] = float(values[state_valid].mean())
        return metrics
