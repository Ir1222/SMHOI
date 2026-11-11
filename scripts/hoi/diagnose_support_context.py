"""P3.4 controlled paired support identifiability, not a natural HOI dataset.

No learned WM/optimizer is needed to test observational aliasing. A small ridge
predictor compares candidate encoding on whole held-out pairs; this diagnostic
regression is not a second production World Model training framework.
"""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
import warp as wp
from calibrate_contacts import Placements
from p3_common import reference, save_rows, snapshot

from supermimic.hoi.asset_manifest import load_manifest
from supermimic.hoi.recorder import install_recording
from supermimic.hoi.reference import DEFAULT_ASSET
from supermimic.hoi.scene import PhysicalSceneCfg, make_physical_env
from supermimic.hoi.state_spec import JointStateSpec
from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    rot6d_from_quat,
)


def candidate_context(states, manifest):
    p = torch.tensor([s["root_pos_world"] for s in manifest["supports"]], dtype=states.dtype)
    q = torch.tensor([s["root_quat_wxyz"] for s in manifest["supports"]], dtype=states.dtype)
    h = heading_quat(states[:, 3:7])[:, None].expand(-1, 2, -1)
    relative = quat_mul(quat_conjugate(h), q.expand(len(states), -1, -1))
    return torch.cat(
        (quat_apply_inverse(h, p[None] - states[:, None, :3]), rot6d_from_quat(relative)), -1
    ).reshape(len(states), 18)


def compare(rows, manifest):
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    state, target = [torch.tensor(arrays[k], dtype=torch.float64) for k in ("state", "next_state")]
    spec = JointStateSpec(dtype=torch.float64)
    features = spec.canonical(state)
    pairs = np.arange(len(state)).reshape(-1, 2)
    difference = float((features[::2] - features[1::2]).abs().max())
    assert difference < 2e-6, difference
    np.testing.assert_allclose(arrays["action_exec"][::2], arrays["action_exec"][1::2], atol=0)
    np.testing.assert_allclose(arrays["xi"][::2], arrays["xi"][1::2], atol=0)
    prior = spec.integrate(state, torch.zeros(len(state), 82, dtype=state.dtype))
    object_target = spec.object_error_vector(target, prior).numpy() * spec.object_err_scale.numpy()
    y = np.concatenate((object_target, arrays["force_mean_box"].reshape(-1, 6)), -1)
    # Pair number, never individual transition, is the partition unit.
    held_pairs = np.arange(len(pairs)) % 8 >= 6
    train, valid = pairs[~held_pairs].reshape(-1), pairs[held_pairs].reshape(-1)
    context = candidate_context(state, manifest).numpy()
    x = np.concatenate((features.numpy(), arrays["action_exec"], arrays["xi"]), -1)
    results = {}
    for name, inputs in [("without_context", x), ("with_candidate_context18", np.c_[x, context])]:
        # All standardization fitted ONLY on training pairs. Ridge regularization
        # is diagnostic numerical conditioning, fixed before either comparison.
        mean, std = inputs[train].mean(0), inputs[train].std(0)
        keep = std > 1e-5
        inputs = np.c_[np.ones(len(inputs)), (inputs[:, keep] - mean[keep]) / std[keep]]
        regularizer = np.eye(inputs.shape[1]) * 1e-4
        regularizer[0, 0] = 0
        weight = np.linalg.solve(
            inputs[train].T @ inputs[train] + regularizer, inputs[train].T @ y[train]
        )
        prediction = inputs[valid] @ weight
        mse = ((prediction - y[valid]) ** 2).mean(0)
        results[name] = {
            "object_RMSE_pos_rot_linear_angular": [
                float(np.sqrt(mse[i : i + 3].mean())) for i in (0, 3, 6, 9)
            ],
            "force_MAE_LR_N": np.abs(prediction[:, 12:] - y[valid, 12:])
            .reshape(-1, 2, 3)
            .mean((0, 2))
            .tolist(),
            "object_normalized_MSE": float((mse[:12] / spec.object_err_scale.numpy() ** 2).mean()),
        }
    delta = y[::2] - y[1::2]
    return {
        "canonical_pair_max_abs_error": difference,
        "pair_object_max_abs_pos_rot_linear_angular": [
            float(np.abs(delta[:, i : i + 3]).max()) for i in (0, 3, 6, 9)
        ],
        "pair_force_max_abs_LR_N": np.abs(delta[:, 12:]).reshape(-1, 2, 3).max((0, 2)).tolist(),
        "train_pair_ids": np.flatnonzero(~held_pairs).tolist(),
        "validation_pair_ids": np.flatnonzero(held_pairs).tolist(),
        "comparisons": results,
        "context_reduces_object_error": results["with_candidate_context18"]["object_normalized_MSE"]
        < results["without_context"]["object_normalized_MSE"],
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    clip = reference()
    manifest = load_manifest(DEFAULT_ASSET)
    wp.config.kernel_cache_dir = "/tmp/smhoi_p3_warp_cache"
    torch.manual_seed(301)
    env = make_physical_env(PhysicalSceneCfg(num_envs=4, action_noise_std=0.0))
    rows, metadata = [], {"kind": "controlled_support_pairs", "acceptance": "FAIL"}
    try:
        rec = install_recording(env, force_on=0.001, capacity=128, references=[str(clip.path)])
        metadata.update(replay=rec.buffer.metadata)
        placements = Placements(env, rec.io)
        ids = torch.arange(4, device=env.device)
        support = manifest["supports"][0]
        for iteration in range(16):
            initial = []
            for j in range(2):
                pair = iteration * 2 + j
                if pair < 16:
                    state = placements.box_support()
                else:
                    state = placements.contact("left" if pair % 2 == 0 else "right")
                    # Translate BOTH roots to place the bottom geom on support.
                    shift = np.array(support["root_pos_world"]) + [0, 0, 0.029] - state[71:74]
                    state[:3] += shift
                    state[71:74] += shift
                dz = (pair % 8) * 0.0002
                state[73] += dz
                state[80] = -0.01 * (1 + pair % 4)
                air = state.copy()
                air[[0, 71]] += 1.0
                initial.extend([state, air])
            env.reset()
            rec.io.write(
                ids, torch.tensor(np.asarray(initial), device=env.device, dtype=torch.float32)
            )
            rec.begin_episode(
                ids,
                torch.full((4,), 260, device=env.device),
                torch.zeros(4, dtype=torch.long, device=env.device),
            )
            pointers = rec.buffer.ptr.clone()
            # Same executed action within each pair, natural native actuation.
            env.step(torch.zeros(4, 29, device=env.device))
            rows.extend(snapshot(rec.buffer, ids, pointers))
            assert not env.sim.data.overflow[:].any()
        metadata.update(compare(rows, manifest))
        metadata["acceptance"] = "PASS" if metadata["context_reduces_object_error"] else "PARTIAL"
    except Exception:
        metadata["error"] = traceback.format_exc()
        raise
    finally:
        save_rows(output, rows, metadata)
        env.close()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
