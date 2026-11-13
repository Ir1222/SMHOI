"""Check exact load and repeated native continuation separately from training."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from compare_training_checkpoints import compare

from supermimic.hoi import training_checkpoint as ck
from supermimic.hoi.play import _restore_config
from supermimic.hoi.training import HOITrainer
from supermimic.hoi.training_config import HOITrainConfig
from supermimic.hoi.training_safety import SolverGuard


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
            loaded = ck.snapshot(t)
            differences, counts = compare(
                {k: v for k, v in saved.items() if k != "guard_evidence"},
                {k: v for k, v in loaded.items() if k != "guard_evidence"},
            )
            branches = []
            for repeat in range(2):
                if repeat:
                    ck.restore(t, saved)
                obs, _ = t.env.reset()
                rows = []
                for step in range(8):
                    with torch.no_grad():
                        action = t.act(obs)
                        obs, *_ = t.env.step(action)
                    rows.append(
                        {
                            "action": action.cpu().clone(),
                            "executed": t.env.raw.action_manager.get_term("joint_pos")
                            .executed_raw.cpu()
                            .clone(),
                            "qpos": t.env.raw.sim.data.qpos[:].cpu().clone(),
                            "qvel": t.env.raw.sim.data.qvel[:].cpu().clone(),
                        }
                    )
                branches.append(rows)
            result = {
                "checkpoint": str(args.checkpoint),
                "load_bitwise_equal": not differences,
                "load_compared": counts,
                "load_differences": differences,
                "repeat_same_state_no_updates": [
                    {"step": i + 1, **{k: float((x[k] - y[k]).abs().max()) for k in x}}
                    for i, (x, y) in enumerate(zip(*branches))
                ],
                "meaning": "Repeated continuation of identical restored state, no optimizer updates; separates exact restoration from native GPU dynamics reproducibility.",
            }
            (args.output / "resume_state.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result))
            assert not differences
        finally:
            t.close()


if __name__ == "__main__":
    main()
