"""Short P2 data-pipeline diagnostic. Controlled fixtures; no policy/training.

Reuses P1 native collider placements and independent raw-contact force sums.
Each output directory must be new. Two clip IDs intentionally reference the same
immutable Part B motion to exercise identity boundaries without inventing data.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
import warp as wp
from calibrate_contacts import Placements, raw_sample
from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from supermimic.hoi.contact import hand_force_history_world
from supermimic.hoi.recorder import install_recording
from supermimic.hoi.reference import DEFAULT_ASSET, DEFAULT_OUTPUT, ReferenceClip, sha256
from supermimic.hoi.scene import PhysicalSceneCfg, make_physical_env
from supermimic.hoi.state import ReferenceResetCache


def fixture_failure(env):
    e = torch.arange(env.num_envs, device=env.device)
    return ((e == 0) | (e == 3)) & (env.episode_length_buf >= getattr(env, "_p2_fixture_limit", 12))


def fixture_timeout(env):
    e = torch.arange(env.num_envs, device=env.device)
    return ((e == 1) | (e == 3)) & (env.episode_length_buf >= getattr(env, "_p2_fixture_limit", 12))


def distribution(x):
    return {
        "min": float(x.min()),
        "p50": float(np.quantile(x, 0.5)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
        "mean": float(x.mean()),
    }


def stats(data, select):
    occ = data["occupancy"][select]
    active = occ > 0
    return {
        "transitions": int(select.sum()),
        "active_ratio_LR": active.mean(0).tolist(),
        "bilateral_ratio": float(active.all(-1).mean()),
        "mean_force_norm_N": [
            distribution(x) for x in np.linalg.norm(data["force_mean_box"][select], axis=-1).T
        ],
        "peak_N": [distribution(x) for x in data["force_peak"][select].T],
        "impulse_norm_Ns": [
            distribution(x) for x in np.linalg.norm(data["impulse_world"][select], axis=-1).T
        ],
        "occupancy": [distribution(x) for x in occ.T],
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17)
    wp.config.kernel_cache_dir = "/tmp/smhoi_p2_warp_cache"
    env = make_physical_env(PhysicalSceneCfg(num_envs=4, action_noise_std=0.03))
    results = {
        "acceptance": "FAIL",
        "fixture": "P1 placements; 24 native steps x 4 worlds; no policy",
        "reference_sha256": sha256(DEFAULT_OUTPUT / "reference.npz"),
        "asset_sha256": sha256(DEFAULT_ASSET),
    }
    try:
        env.cfg.terminations = {
            "fixture_failure": TerminationTermCfg(func=fixture_failure),
            "fixture_timeout": TerminationTermCfg(func=fixture_timeout, time_out=True),
        }
        reference = str(DEFAULT_OUTPUT / "reference.npz")
        rec = install_recording(env, force_on=1e-3, capacity=128, references=[reference, reference])
        io = rec.io
        placements = Placements(env, io)
        cache = ReferenceResetCache(ReferenceClip(reference), io.joint_names, env.device)
        bilateral = cache.states[260].clone()
        bilateral[36:71] = 0
        bilateral[78:84] = 0
        # Translate both free roots together: retain hand-box geometry while
        # isolating this sensor/Replay fixture from subsequent floor impacts.
        bilateral[2] += 2.0
        bilateral[73] += 2.0
        states = torch.stack(
            [
                torch.tensor(placements.base, device=env.device, dtype=torch.float32),
                torch.tensor(
                    placements.contact("left", roll=0.63), device=env.device, dtype=torch.float32
                ),
                torch.tensor(placements.contact("right"), device=env.device, dtype=torch.float32),
                bilateral,
            ]
        )
        starts = torch.tensor([200, 260, 831, 827], device=env.device)
        clips = torch.tensor([0, 1, 0, 1], device=env.device)
        ids = torch.arange(4, device=env.device)
        env.reset()
        io.write(ids, states)
        rec.begin_episode(ids, starts, clips)
        observations = []
        expected_quats = []
        executed = []
        overflow = []
        native_step = env.sim.step

        def observed_step():
            expected_quats.append(io.read()[:, 74:78].clone())
            executed.append(env.action_manager.get_term("joint_pos").executed_raw.clone())
            native_step()

        env.sim.step = observed_step  # Diagnostic wrapper only; original simulation call unchanged.

        def observe(env):
            observations.append(raw_sample(env))
            overflow.append(env.sim.data.overflow[:].clone())
            return torch.zeros(4, device=env.device)

        env.cfg.metrics = {
            **env.cfg.metrics,
            "independent_probe": MetricsTermCfg(func=observe, per_substep=True),
        }
        env.metrics_manager = MetricsManager(env.cfg.metrics, env)
        errors = {
            "quaternion": 0.0,
            "raw_force_N": 0.0,
            "action": 0.0,
            "next_state": 0.0,
            "history_N": 0.0,
        }
        collected = []
        reset_count = 0
        for step in range(24):
            p = rec.buffer.ptr.clone()
            before = io.read().clone()
            action = (
                torch.sin(torch.arange(29, device=env.device)[None, :] * 0.2 + step * 0.31).expand(
                    4, -1
                )
                * 0.05
            )
            env.step(action)
            b = rec.buffer
            assert torch.equal(b.s[ids, p], before)
            assert torch.equal(b.a[ids, p], action)
            q = torch.stack(expected_quats[-4:], dim=1)
            err = (b.fields["box_quat_solve"][ids, p] - q).abs().max().item()
            errors["quaternion"] = max(errors["quaternion"], err)
            assert err < 2e-6
            raw = torch.tensor(
                np.array([x["force_columns_world_N"] for x in observations[-4:]]),
                device=env.device,
                dtype=torch.float32,
            ).permute(1, 0, 2, 3, 4)
            err = (b.fields["force_raw_world"][ids, p] - raw).abs().max().item()
            errors["raw_force_N"] = max(errors["raw_force_N"], err)
            assert err == 0
            expected_found = torch.tensor(
                np.array([x["found_columns"] for x in observations[-4:]]), device=env.device
            ).permute(1, 0, 2, 3)
            assert torch.equal(b.fields["found_substeps"][ids, p], expected_found)
            for a in executed[-4:]:
                err = (b.a_exec[ids, p] - a).abs().max().item()
                errors["action"] = max(errors["action"], err)
                assert err == 0
            # Independent matrix rotation with NumPy, not production quaternion math.
            from scipy.spatial.transform import Rotation

            rot = (
                Rotation.from_quat(q.cpu().numpy().reshape(-1, 4), scalar_first=True)
                .as_matrix()
                .reshape(4, 4, 3, 3)
            )
            force = raw.sum(3).cpu().numpy()
            expected_mean = np.einsum("ekji,ekhj->ekhi", rot, force).mean(1)
            np.testing.assert_allclose(
                b.fields["force_mean_box"][ids, p].cpu(), expected_mean, atol=2e-5, rtol=2e-6
            )
            errors["mean_box_N"] = max(
                errors.get("mean_box_N", 0),
                float(
                    np.abs(b.fields["force_mean_box"][ids, p].cpu().numpy() - expected_mean).max()
                ),
            )
            terminal = b.fields["transition_terminal"][ids, p]
            assert torch.equal(terminal, env.reset_buf)
            for e in range(4):
                row = {key: value[e, p[e]].cpu().numpy().copy() for key, value in b.fields.items()}
                row.update(
                    state=b.s[e, p[e]].cpu().numpy().copy(),
                    action_clean=b.a[e, p[e]].cpu().numpy().copy(),
                    action_exec=b.a_exec[e, p[e]].cpu().numpy().copy(),
                    frame=b.frame[e, p[e]].item(),
                    clip=b.clip[e, p[e]].item(),
                    episode=b.ep_id[e, p[e]].item(),
                    env=e,
                )
                collected.append(row)
            ongoing = ids[~terminal]
            if len(ongoing):
                assert torch.equal(io.read()[ongoing], b.fields["next_state"][ongoing, p[ongoing]])
                history = hand_force_history_world(env.scene)[ongoing].permute(0, 2, 1, 3).flip(1)
                assert torch.equal(history, raw[ongoing].sum(3))
            done = ids[terminal]
            if len(done):
                reset_count += len(done)
                assert hand_force_history_world(env.scene)[done].count_nonzero() == 0
                tail = (p[done] + 1) % b.T
                assert torch.equal(b.s[done, tail], b.fields["next_state"][done, p[done]])
                assert not b.fields["transition_valid"][done, tail].any()
                assert not rec.ready[done].any()
                io.write(done, states[done])
                rec.begin_episode(done, starts[done], clips[done])
        window = rec.buffer.sample_windows(512)
        assert torch.equal(window["states"][1:], window["next_state"])
        for key in ["env_id", "episode_id", "clip"]:
            assert (window[key] == window[key][0]).all()
        assert (window["frame"][1:] - window["frame"][:-1] == 1).all()
        assert not window["transition_terminal"][:-1].any()
        assert window["transition_valid"].all()
        # Explicit partial reset closes its old episode, changes no other rows/history.
        other_history = hand_force_history_world(env.scene)[1:].clone()
        ep = rec.buffer.ep_counter.clone()
        size = rec.buffer.size
        env.reset(env_ids=ids[:1])
        assert torch.equal(hand_force_history_world(env.scene)[1:], other_history)
        assert rec.buffer.size == size and rec.buffer.ep_counter[0] == ep[0] + 1
        assert torch.equal(rec.buffer.ep_counter[1:], ep[1:])
        try:
            rec.capture.consume(ids[:1])
            raise AssertionError("Stale interval accepted")
        except RuntimeError:
            pass
        try:
            rec.begin_episode(ids[:1], torch.tensor([839], device=env.device), clips[:1])
            raise AssertionError("Outgoing final-frame transition accepted")
        except ValueError:
            pass
        # Nonzero terminal labels must survive auto-reset in both execution modes.
        env._p2_fixture_limit = 1
        terminal_runs = []
        for use_graph in (True, False):
            env.sim.use_cuda_graph = use_graph
            env.reset()
            io.write(ids, states)
            probe_frames = torch.tensor([200, 260, 838, 838], device=env.device)
            rec.begin_episode(ids, probe_frames, clips)
            p = rec.buffer.ptr.clone()
            torch.manual_seed(911)
            env.step(torch.zeros(4, 29, device=env.device))
            assert env.reset_buf.all()
            assert hand_force_history_world(env.scene).count_nonzero() == 0
            labels = {k: v[ids, p].clone() for k, v in rec.buffer.fields.items()}
            assert labels["force_peak"][1, 0] > 1e-3
            assert labels["force_peak"][2, 1] > 1e-3
            assert (labels["force_peak"][3] > 1e-3).all()
            assert labels["physical_failure"].tolist() == [True, False, False, True]
            assert labels["timeout"].tolist() == [False, True, False, True]
            assert labels["clip_end"].tolist() == [False, False, True, True]
            assert not rec.capture.count.any()
            saved = labels["force_raw_world"].clone()
            env.reset()
            assert torch.equal(saved, rec.buffer.fields["force_raw_world"][ids, p])
            terminal_runs.append(labels)
        parity = {}
        for key in [
            "next_state",
            "force_mean_box",
            "impulse_world",
            "force_raw_world",
            "box_quat_solve",
        ]:
            x, y = [r[key] for r in terminal_runs]
            parity[key] = (x - y).abs().max().item()
            torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-6)
        results["active_terminal_probe"] = {
            "modes": ["cuda_graph", "eager"],
            "transitions": 8,
            "peak_force_N": terminal_runs[0]["force_peak"].cpu().tolist(),
            "parity_max_abs": parity,
            "retained_copy_unchanged_after_reset": True,
        }
        np.savez_compressed(
            output / "active_terminal.npz",
            **{k: v.cpu().numpy() for k, v in terminal_runs[0].items()},
        )
        data = {key: np.stack([r[key] for r in collected]) for key in collected[0]}
        np.savez_compressed(output / "transitions.npz", **data)
        active = data["occupancy"] > 0
        onset = np.zeros(2, dtype=int)
        loss = onset.copy()
        pairs = 0
        for e in range(4):
            for episode in np.unique(data["episode"][data["env"] == e]):
                a = active[(data["env"] == e) & (data["episode"] == episode)]
                onset += (~a[:-1] & a[1:]).sum(0)
                loss += (a[:-1] & ~a[1:]).sum(0)
                pairs += len(a) - 1
        selected = np.ones(len(collected), dtype=bool)
        results.update(
            acceptance="PASS",
            errors=errors,
            metadata=rec.buffer.metadata,
            transitions=len(collected),
            substeps=len(observations),
            terminal_resets=reset_count,
            windows_checked=512,
            all_overflow_zero=bool(torch.stack(overflow).count_nonzero() == 0),
            independent_raw_sum_max_error_N=max(x["raw_sum_max_error_N"] for x in observations),
            zero_force_tolerance_N=1e-6,
            no_contact_max_N=float(np.abs(data["force_raw_world"][data["env"] == 0]).max()),
            action_noise_std_observed=float((data["action_exec"] - data["action_clean"]).std()),
            all=stats(data, selected),
            per_env={str(e): stats(data, data["env"] == e) for e in range(4)},
            per_clip={str(c): stats(data, data["clip"] == c) for c in range(2)},
            onset_count_LR=onset.tolist(),
            loss_count_LR=loss.tolist(),
            adjacent_pairs=pairs,
            onset_frequency_Hz=(onset / (pairs * 0.02)).tolist(),
            loss_frequency_Hz=(loss / (pairs * 0.02)).tolist(),
            reason_counts={
                k: int(data[k].sum())
                for k in ["physical_failure", "timeout", "clip_end", "transition_terminal"]
            },
            examples=[
                {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in collected[i].items()
                }
                for i in [0, 1, 2, 3, 30, 47]
            ],
            threshold_sensitivity={
                str(t): (np.linalg.norm(data["force_raw_world"].sum(3), axis=-1) > t)
                .mean((0, 1))
                .tolist()
                for t in [1e-6, 1e-3, 0.1, 1.0]
            },
        )
        assert results["no_contact_max_N"] <= 1e-6
        assert results["all_overflow_zero"]
        assert results["reason_counts"] == {
            "physical_failure": 4,
            "timeout": 4,
            "clip_end": 5,
            "transition_terminal": 9,
        }
        (output / "raw_samples.json").write_text(json.dumps(observations, indent=2) + "\n")
    except Exception:
        if "observations" in locals():
            (output / "partial_raw_samples.json").write_text(
                json.dumps(observations, indent=2) + "\n"
            )
            results["completed_vector_substeps"] = len(observations)
            results["overflow_at_failure"] = env.sim.data.overflow[:].cpu().tolist()
        results["acceptance"] = "FAIL"
        results["traceback"] = traceback.format_exc()
        raise
    finally:
        (output / "diagnostics.json").write_text(json.dumps(results, indent=2) + "\n")
        env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["acceptance", "transitions", "substeps", "errors", "reason_counts"]
            }
        )
    )
