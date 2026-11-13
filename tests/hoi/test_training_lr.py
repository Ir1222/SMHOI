"""Synthetic regression: configured HOI Critic LR must survive scheduling."""
from types import SimpleNamespace

import pytest
import torch

from supermimic.hoi.imagination import HOIPolicyTrainer
from supermimic.imagination.policy_trainer import PolicyTrainer


@pytest.mark.parametrize('fraction', [1.0, 0.5, 0.1])
def test_hoi_preserves_configured_ratio_and_robot_behavior(fraction):
    t = object.__new__(HOIPolicyTrainer)
    t.cfg = SimpleNamespace(lr=0.001, critic_lr=0.0003)
    t.opt = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=t.cfg.lr)
    t.critic_opt = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=t.cfg.critic_lr)
    t.set_lr(t.cfg.lr * fraction)
    assert t.opt.param_groups[0]['lr'] == pytest.approx(t.cfg.lr * fraction)
    assert t.critic_opt.param_groups[0]['lr'] == pytest.approx(t.cfg.critic_lr * fraction)
    PolicyTrainer.set_lr(t, 0.001)
    assert t.opt.param_groups[0]['lr'] == t.critic_opt.param_groups[0]['lr'] == 0.001
