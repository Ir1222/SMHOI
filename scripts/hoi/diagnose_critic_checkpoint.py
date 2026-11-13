"""Frozen-snapshot TD(lambda) and feature-scale diagnosis; zero optimizer steps."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from supermimic.hoi.play import _restore_config
from supermimic.hoi.training import HOITrainer
from supermimic.hoi.training_config import HOITrainConfig
from supermimic.hoi.training_safety import SolverGuard


def stats(value):
    value = value.detach().float().cpu().flatten()
    assert torch.isfinite(value).all()
    return dict(
        zip(
            ("min", "median", "p99", "max", "mean"),
            [
                float(value.min()),
                float(value.median()),
                float(value.quantile(0.99)),
                float(value.max()),
                float(value.mean()),
            ],
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = _restore_config(HOITrainConfig(), saved["config"])
    cfg = replace(
        cfg, log_root=args.output.parent, run_name=args.output.name, resume=args.checkpoint
    )
    with SolverGuard() as guard:
        t = HOITrainer(cfg, guard)
        try:
            before = dict(t.learning_safety.steps)
            with torch.no_grad():
                start = t.buffer.sample_starts(cfg.policy_trainer.batch_size, 10)
                loss, objective, ro, inputs, xi = t.policy_trainer.actor_loss(start)
                X, Y, values, targets = t.policy_trainer.critic_batch(ro, xi, inputs)
            reward, d, v = [x.cpu().double() for x in (ro.rewards, ro.dhat[1:], values)]
            lengths, bootstrap = ro.lengths.cpu(), ro.bootstrap.cpu()
            expected = torch.zeros_like(reward)
            for col, length in enumerate(lengths.tolist()):
                tail = float(v[length, col]) if bootstrap[col] else 0.0
                for i in range(length - 1, -1, -1):
                    nxt = float(v[i + 1, col]) if (i + 1 < length or bootstrap[col]) else 0.0
                    tail = float(reward[i, col]) + cfg.policy_trainer.gamma * (
                        1 - float(d[i, col])
                    ) * (
                        (1 - cfg.policy_trainer.gae_lambda) * nxt
                        + cfg.policy_trainer.gae_lambda * tail
                    )
                    expected[i, col] = tail
            torch.testing.assert_close(targets.cpu().double(), expected, rtol=1e-5, atol=1e-5)
            assert len(Y) == int(ro.valid.sum())
            assert t.learning_safety.steps == before
            state_valid = (
                torch.arange(values.shape[0], device=values.device)[:, None] <= ro.lengths[None]
            )
            result = {
                "checkpoint": str(args.checkpoint),
                "iteration": t.iteration,
                "optimizer_updates_during_diagnosis": 0,
                "independent_td_lambda_max_abs_error": float(
                    (targets.cpu().double() - expected).abs().max()
                ),
                "valid_transition_count": len(Y),
                "lengths": stats(lengths),
                "rewards": stats(ro.rewards[ro.valid]),
                "values": stats(values[state_valid]),
                "targets": stats(Y),
                "actor_loss": float(loss),
                "actor_objective": float(objective),
                "critic_feature_abs": stats(X[0].abs()),
                "joint_limit_reward": ro.reward_terms["joint_limit"],
                "actor_lr": t.policy_trainer.opt.param_groups[0]["lr"],
                "critic_lr": t.policy_trainer.critic_opt.param_groups[0]["lr"],
                "conclusion": "Existing TD(lambda) matches independent per-trajectory recurrence and masks; finite scale drift is not a numerical-stability PASS.",
            }
            (args.output / "critic_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result))
        finally:
            t.close()


if __name__ == "__main__":
    main()
