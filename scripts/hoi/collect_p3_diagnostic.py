"""Bounded supervised implementation diagnostic; no policy or natural-data claim."""

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
import warp as wp
from calibrate_contacts import Placements
from p3_common import reference, save_rows, snapshot

from supermimic.hoi.recorder import install_recording
from supermimic.hoi.scene import PhysicalSceneCfg, make_physical_env
from supermimic.hoi.state import ReferenceResetCache


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    clip = reference()
    wp.config.kernel_cache_dir = "/tmp/smhoi_p3_warp_cache"
    torch.manual_seed(303)
    env = make_physical_env(PhysicalSceneCfg(num_envs=4, action_noise_std=0.03))
    rows = []
    metadata = {
        "kind": "controlled_P3_diagnostic",
        "acceptance": "FAIL",
        "seed": 303,
        "budget": "8 episodes x 4 worlds x 10 transitions = 320",
        "natural_HOI_training_data": False,
    }
    try:
        rec = install_recording(env, force_on=0.001, capacity=256, references=[str(clip.path)])
        metadata["replay"] = rec.buffer.metadata
        placements = Placements(env, rec.io)
        cache = ReferenceResetCache(clip, rec.io.joint_names, env.device)
        ids = torch.arange(4, device=env.device)
        for episode in range(8):
            bilateral = cache.states[260].cpu().numpy().copy()
            bilateral[36:71] = 0
            bilateral[78:84] = 0
            bilateral[[2, 73]] += 2
            initial = np.stack(
                [
                    placements.base,
                    placements.contact("left", roll=0.1 * episode),
                    placements.contact("right", roll=-0.1 * episode),
                    bilateral,
                ]
            )
            # Paired physical roots remain together; independent small state/action
            # perturbations create distinct episodes but not motion diversity.
            shift = (episode - 3.5) * 0.003
            initial[:, [0, 71]] += shift
            env.reset()
            rec.io.write(ids, torch.tensor(initial, dtype=torch.float32, device=env.device))
            rec.begin_episode(
                ids,
                torch.full((4,), 260, device=env.device),
                torch.zeros(4, dtype=torch.long, device=env.device),
            )
            for step in range(10):
                pointers = rec.buffer.ptr.clone()
                action = 0.05 * torch.sin(
                    torch.arange(29, device=env.device)[None] * 0.2 + step * 0.31 + episode * 0.13
                ).expand(4, -1)
                env.step(action)
                rows.extend(snapshot(rec.buffer, ids, pointers))
                flags = env.sim.data.overflow[:].cpu().tolist()
                metadata["last_interval"] = {
                    "episode": episode,
                    "step": step,
                    "post_forward_overflow": flags,
                }
                if any(flags):
                    raise RuntimeError(f"Post-forward overflow: {flags}")
        metadata["acceptance"] = "PASS"
        metadata["transitions"] = len(rows)
        metadata["all_substep_capacity_guards_passed"] = True
    except Exception:
        metadata["error"] = traceback.format_exc()
        raise
    finally:
        save_rows(output, rows, metadata)
        env.close()
    print(f"Collected {len(rows)} measured transitions with xi33; controlled diagnostic only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
