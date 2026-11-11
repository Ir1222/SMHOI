"""Bounded supervised implementation diagnostic; no policy or natural-data claim."""

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
import warp as wp
from calibrate_contacts import Placements
from p3_common import reference, save_rows, snapshot

from supermimic.hoi.contact import assert_contact_capacity
from supermimic.hoi.recorder import install_recording
from supermimic.hoi.scene import PhysicalSceneCfg, make_physical_env
from supermimic.hoi.state import ReferenceResetCache

CONTACT_SCHEDULE = (
    ("train", "onset", 256, 411),
    ("train", "release", 546, 412),
    ("validation", "onset", 255, 413),
    ("validation", "release", 545, 414),
)


def run_contact(output):
    """Exactly four reference resets and eight transitions each; fail without retry."""
    output.mkdir(parents=True, exist_ok=False)
    rows, env, rec = [], None, None
    metadata = {
        "kind": "controlled_P3_contact_boundary_diagnostic",
        "acceptance": "FAIL",
        "budget": "4 episodes x 8 transitions = 32",
        "natural_HOI_training_data": False,
        "cross_clip_generalization": False,
        "episodes": [],
        "label_semantics": "global desired reference intent",
    }
    try:
        clip = reference()
        wp.config.kernel_cache_dir = "/tmp/smhoi_p3_warp_cache"
        env = make_physical_env(PhysicalSceneCfg(num_envs=1, action_noise_std=0.03))
        rec = install_recording(env, force_on=0.001, capacity=64, references=[str(clip.path)])
        metadata["replay"] = rec.buffer.metadata
        cache = ReferenceResetCache(clip, rec.io.joint_names, env.device)
        ids = torch.zeros(1, dtype=torch.long, device=env.device)
        for episode, (split, regime, frame, seed) in enumerate(CONTACT_SCHEDULE):
            torch.manual_seed(seed)
            np.random.seed(seed)
            entry = {
                "split": split,
                "regime": regime,
                "initial_frame": frame,
                "seed": seed,
                "env": 0,
                "row_start": len(rows),
                "rows": 0,
            }
            metadata["episodes"].append(entry)
            frames = torch.tensor([frame], dtype=torch.long, device=env.device)
            rec.io.reset_to_reference(ids, frames, cache)
            assert_contact_capacity(env)
            torch.testing.assert_close(
                rec.io.read()[ids], cache.frames(frames), atol=1e-6, rtol=1e-6
            )
            rec.begin_episode(ids, frames, torch.zeros_like(ids))
            for step in range(8):
                metadata["last_interval"] = {
                    "schedule_episode": episode,
                    "step": step,
                    "start_frame": frame + step,
                }
                pointers = rec.buffer.ptr.clone()
                action = 0.05 * torch.sin(
                    torch.arange(29, device=env.device)[None] * 0.2 + step * 0.31 + episode * 0.13
                )
                env.step(action)
                captured = snapshot(rec.buffer, ids, pointers)
                rows.extend(captured)
                entry.update(rows=step + 1, episode=int(captured[0]["episode"]))
                flags = env.sim.data.overflow[:].cpu().tolist()
                metadata["last_interval"]["post_forward_overflow"] = flags
                assert_contact_capacity(env)
                if any(flags):
                    raise RuntimeError(f"Post-forward overflow: {flags}")
                row = captured[0]
                expected = clip.arrays["object_contact"][frame + step : frame + step + 2]
                if int(row["frame"]) != frame + step or not np.array_equal(
                    row["intent_ref"], expected
                ):
                    raise RuntimeError("Recorded contact/frame differs from canonical reference")
                if not row["transition_valid"] or row["transition_terminal"]:
                    raise RuntimeError("Invalid or terminal transition in bounded contact capture")
            print(f"Captured {split} {regime}: frames {frame}..{frame + 8}", flush=True)
        if len(rows) != 32:
            raise RuntimeError("Incomplete contact diagnostic")
        metadata.update(acceptance="PASS", all_substep_capacity_guards_passed=True)
    except Exception:
        metadata["error"] = traceback.format_exc()
        if rec is not None:
            # Partial interval snapshots are failure diagnostics, never Replay labels.
            try:
                np.savez_compressed(
                    output / "failure_snapshot.npz",
                    state=rec.io.read().cpu().numpy(),
                    overflow=env.sim.data.overflow[:].cpu().numpy(),
                    captured_substeps=rec.capture.count.cpu().numpy(),
                    partial_force_raw_world=rec.capture.raw.cpu().numpy(),
                    partial_box_quat_solve=rec.capture.quat.cpu().numpy(),
                    partial_found_substeps=rec.capture.found.cpu().numpy(),
                )
                metadata["failure_snapshot"] = "failure_snapshot.npz"
            except (RuntimeError, ValueError, AttributeError, TypeError, OSError):
                metadata["failure_snapshot_error"] = traceback.format_exc()
        raise
    finally:
        metadata["transitions"] = len(rows)
        save_rows(output, rows, metadata)
        if env is not None:
            env.close()


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
    parser.add_argument(
        "--contact-boundaries",
        action="store_true",
        help="Approved 32-transition schedule; no retries",
    )
    args = parser.parse_args()
    (run_contact if args.contact_boundaries else run)(args.output)
