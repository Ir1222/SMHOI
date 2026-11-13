"""CPU-only exhaustive Replay and optimizer audit of a complete training snapshot."""

import argparse
import json
from pathlib import Path

import torch

from supermimic.hoi.replay import HOIReplayBuffer
from supermimic.hoi.wm_trainer import JointWorldModelTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    runtime = saved["runtime"]
    e, t = runtime["replay"]["valid"].shape
    buffer = HOIReplayBuffer(e, t, "cpu", force_on=saved["replay_metadata"]["force_on_N"])
    for name, value in runtime["replay"].items():
        setattr(buffer, name, value)
    buffer.fields = runtime["replay_fields"]
    envs, rows = buffer.window_mask(8).nonzero(as_tuple=True)
    # Reuse the actual gather and validation contracts, exhaust every eligible start.
    for ids in torch.arange(len(rows)).split(256):
        batch = buffer._gather_windows(envs[ids], rows[ids], 8)
        JointWorldModelTrainer.physical_states(None, batch)
    f = buffer.fields
    mask = buffer.valid & f["transition_valid"]
    reasons = f["physical_failure"] | f["timeout"] | f["clip_end"]
    assert torch.equal(f["transition_terminal"][mask], reasons[mask])
    assert torch.equal(f["bootstrap"][mask], (~(f["physical_failure"] | f["clip_end"]))[mask])
    assert not (buffer.valid & buffer.terminal & f["transition_valid"]).any()
    assert torch.isfinite(buffer.s[buffer.valid]).all()
    assert torch.isfinite(buffer.a_exec[mask]).all()
    assert torch.isfinite(f["next_state"][mask]).all()
    count = saved["optimizer_steps"]
    config = saved["config"]
    assert (
        count["wm"] == saved["counters"]["num_updates"] * config["wm_trainer"]["batches_per_update"]
    )
    assert (
        count["actor"]
        == saved["startup"]["alternating_updates"]
        * config["schedule"]["policy_updates_per_wm_update"]
    )
    adam = {}
    for name, optimizer in saved["optimizers"].items():
        steps = [float(s["step"]) for s in optimizer["state"].values()]
        assert steps and min(steps) == max(steps) == count[name]
        adam[name] = {
            "parameter_states": len(steps),
            "step_min": min(steps),
            "step_max": max(steps),
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "iteration": saved["counters"]["iteration"],
        "envs": e,
        "capacity": t,
        "all_windows8_checked": len(rows),
        "retained_transitions": int(mask.sum()),
        "terminal_state_rows": int((buffer.valid & buffer.terminal).sum()),
        "reasons": {
            k: int(f[k][mask].sum())
            for k in ("robot_failure", "object_failure", "timeout", "clip_end")
        },
        "optimizer_steps": count,
        "adam": adam,
        "status": "PASS",
        "scope": "temporal/finite-state Replay and optimizer integrity; does not establish geometric validity",
    }
    assert not torch.cuda.is_initialized(), "Snapshot audit must remain CPU-only"
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
