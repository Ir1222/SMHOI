"""Bounded P7.0 component preflight, not a training entrypoint or PD replay.

Uses the actual Policy class with random weights, physical HOI stepping and
backward-only existing trainers. No optimizer step is called. All outputs are
exclusive, experimental and unsuitable as learned-performance evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import traceback
from collections import Counter
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "outputs/hoi/contact_supervision/p6_closure"
sys.path.insert(0, str(P6 / "scripts"))
from closure_common import gpu_env, inputs
from mjlab.managers.metrics_manager import MetricsTermCfg
from mujoco_warp._src import solver

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.hoi.checkpoint import joint_checkpoint, load_joint_weights
from supermimic.hoi.contact import assert_contact_capacity
from supermimic.hoi.contact_validation import actual_contact
from supermimic.hoi.imagination import HOIPolicyReplayBuffer, HOIPolicyTrainer
from supermimic.hoi.recorder import install_recording
from supermimic.hoi.reference import ReferenceClip, sha256
from supermimic.hoi.reward_reference import ClipBinding, HOIReferenceRegistry
from supermimic.hoi.rewards import ProximityReward
from supermimic.hoi.solver_precision import SOURCE_SHA256, bracket_precision_guard
from supermimic.hoi.state import JointStateIO, ReferenceResetCache
from supermimic.hoi.tracking_reward import HOIRewardCfg, HOITrackingReward
from supermimic.hoi.wm_trainer import JointWorldModelTrainer
from supermimic.hoi.world_model import JointWorldModel, JointWorldModelCfg, LossWeights
from supermimic.imagination.policy_trainer import PolicyTrainerCfg, frozen_parameters
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.heads import (
    AnalyticHeads,
    analytic_observation,
    flatten_history_term_major,
)
from supermimic.world_model.wm_trainer import WMTrainerCfg

PINS = {
    "outputs/hoi/asset_audit/part_b/reference/reference.npz": "bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b",
    "outputs/hoi/contact_supervision/p6_closure/reference_candidate_v1/reference.npz": "f38071e5347403f9eff993dd37884bf374e0c003400156dc4cb87d3d2b856cb0",
    "outputs/hoi/contact_supervision/p6_closure/candidates/single_body.json": "6c3f1a9c601336cb264cd0f979ec4d039ed380e76df6ee1a0a33358d5a82e331",
    "outputs/hoi/contact_supervision/contact_validation/calibration.json": "34a47e06454c64f71b456bb428110fceff65302f6c6517dc85a95e151d400266",
}


def save(path, value):
    def encode(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))

    path.write_text(json.dumps(value, default=encode, allow_nan=False, indent=2) + "\n")


def finite(name, *values):
    if not all(torch.isfinite(x).all() for x in values):
        raise RuntimeError(f"Nonfinite {name}; preflight stopped")


def grad_norm(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        raise RuntimeError("No gradients")
    finite("gradient", *grads)
    return float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())


def reset_geometry(env, io):
    """CPU forward-only inspection of the actual reset pose; no GPU state writes."""
    model = env.sim.mj_model
    result = []
    for e in range(env.num_envs):
        data = mujoco.MjData(model)
        data.qpos[:] = env.sim.data.qpos[e].cpu().numpy()
        data.qvel[:] = env.sim.data.qvel[e].cpu().numpy()
        origin = env.scene.env_origins[e].cpu().numpy()
        data.qpos[io.robot_q[:3].cpu().numpy()] -= origin
        data.qpos[io.box_q[:3].cpu().numpy()] -= origin
        mujoco.mj_forward(model, data)
        distances = [float(c.dist) for c in data.contact]
        result.append(
            {
                "ncon": int(data.ncon),
                "min_signed_distance_m": min(distances, default=None),
                "finite_qacc": bool(np.isfinite(data.qacc).all()),
            }
        )
    return result


def run(out, result):
    torch.set_num_threads(1)
    torch.manual_seed(700)
    wp.config.kernel_cache_dir = "/tmp/smhoi_p7_warp_cache"
    for path, digest in PINS.items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Pinned input changed: {path}")
    result["pins"] = PINS
    result["versions"] = {
        n: metadata.version(n)
        for n in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "numpy")
    }
    result["python"] = sys.executable
    result["solver_path"] = solver.__file__
    result["installed_solver_sha256"] = sha256(solver.__file__)
    assert result["installed_solver_sha256"] == SOURCE_SHA256
    _original_manifest, original, sources = inputs()
    candidate = ReferenceClip(P6 / "reference_candidate_v1/reference.npz")
    assert candidate.manifest["training_ready"] is False
    assert np.array_equal(original.arrays["object_contact"], candidate.arrays["object_contact"])
    result["references"] = {
        "original_loaded": True,
        "candidate_loaded": True,
        "intent_byte_identical": True,
        "intent_on_frames": np.flatnonzero(candidate.arrays["object_contact"]),
        "candidate_training_ready": False,
        "sources": sources,
    }
    calibration = json.loads(
        (ROOT / next(p for p in PINS if p.endswith("calibration.json"))).read_text()
    )
    result["calibration"] = {
        k: calibration[k]
        for k in ("thresholds_N_LR", "occupancy_cutoffs_LR", "strict_comparison", "scope")
    }
    launches = Counter()
    original_launch, original_graph_launch = wp.launch, wp.capture_launch
    original_tiled_launch = wp.launch_tiled
    baseline_factory = solver._linesearch_iterative_kernel

    def audited_launch(kernel, *args, **kwargs):
        name = str(kernel.key)
        if "linesearch" in name:
            launches[name] += 1
        return original_launch(kernel, *args, **kwargs)

    def audited_tiled_launch(kernel, *args, **kwargs):
        name = str(kernel.key)
        if "linesearch" in name:
            launches[name] += 1
        return original_tiled_launch(kernel, *args, **kwargs)

    graph_launches = []

    def audited_graph_launch(*args, **kwargs):
        graph_launches.append(1)
        return original_graph_launch(*args, **kwargs)

    substeps = []

    def monitor(env):
        assert_contact_capacity(env)
        finite(
            "simulation",
            env.sim.data.qpos,
            env.sim.data.qvel,
            env.sim.data.qacc,
            env.sim.data.ctrl,
            env.sim.data.xfrc_applied,
            env.sim.data.qfrc_applied,
        )
        if torch.count_nonzero(env.sim.data.xfrc_applied) or torch.count_nonzero(
            env.sim.data.qfrc_applied
        ):
            raise RuntimeError("External generalized/body wrench detected")
        substeps.append(
            {
                "overflow": env.sim.data.overflow.cpu().tolist(),
                "nefc": env.sim.data.nefc.cpu().tolist(),
                "nacon": env.sim.data.nacon.cpu().tolist(),
            }
        )
        return torch.zeros(env.num_envs, device=env.device)

    def configure(cfg):
        cfg.metrics["p7_finite"] = MetricsTermCfg(func=monitor, per_substep=True)

    with (
        bracket_precision_guard() as guard,
        patch.object(wp, "launch", audited_launch),
        patch.object(wp, "launch_tiled", audited_tiled_launch),
        patch.object(wp, "capture_launch", audited_graph_launch),
    ):
        result["guard"] = guard
        assert solver._linesearch_iterative_kernel is not baseline_factory
        env, manifest = gpu_env("single_body", num_envs=2, change=configure)
        try:
            io = JointStateIO(env)
            ids = torch.arange(2, device=env.device)
            term = env.action_manager.get_term("joint_pos")
            result["model"] = {
                "nq": env.sim.mj_model.nq,
                "nv": env.sim.mj_model.nv,
                "nu": env.sim.mj_model.nu,
                "num_envs": 2,
                "warn_overflow": int(env.sim.wp_model.opt.warn_overflow),
                "iterations": env.sim.mj_model.opt.iterations,
                "ls_iterations": env.sim.mj_model.opt.ls_iterations,
                "mass_kg": float(env.sim.mj_model.body_mass[io.box_body_id]),
                "object": manifest["object"],
                "dt": env.physics_dt,
                "decimation": env.cfg.decimation,
                "environment_type": type(env).__qualname__,
                "native_observation_groups": list(env.cfg.observations),
                "native_reward_terms": list(env.cfg.rewards),
            }
            assert result["model"]["warn_overflow"] != 0
            cache = ReferenceResetCache(candidate, io.joint_names, env.device)
            original_cache = ReferenceResetCache(original, io.joint_names, env.device)
            reset_frames = torch.tensor([0, 258], device=env.device)
            io.reset_to_reference(ids, reset_frames, original_cache)
            torch.testing.assert_close(
                io.read(), original_cache.frames(reset_frames), atol=1e-6, rtol=0
            )
            result["original_reset_roundtrip"] = "PASS"
            result["original_reset_geometry"] = reset_geometry(env, io)

            # Reuse production network and HOI components with small test-only trunks.
            model = load_g1_model()
            q0 = env.scene["robot"].data.default_joint_pos[0].clone()
            fk = G1Kinematics(model, device=q0.device)
            heads = AnalyticHeads(
                fk,
                TRACKED_BODIES,
                ANCHOR_BODY,
                ROOT_BODY,
                env.scene["robot"].data.soft_joint_pos_limits[0].clone(),
                q0,
            )
            lib = MotionLibrary([candidate.path], TRACKED_BODIES, q0, device=env.device)
            registry = HOIReferenceRegistry(
                lib,
                [ClipBinding(0, 0, candidate)],
                library_sources=[candidate.path],
                dt=0.02,
                joint_names=io.joint_names,
            )
            pcfg = PolicyCfg(encoder_hidden=(64, 64), decoder_hidden=(64, 64))
            policy = Policy(pcfg).to(env.device)
            critic = Critic(CriticCfg(hidden=(64, 32)), pcfg=pcfg).to(env.device)
            wm = JointWorldModel(
                JointWorldModelCfg(
                    hidden=(64, 64), term_hidden=(16, 16), weights=LossWeights(1.0, 1.0)
                ),
                heads,
                env.device,
            )
            # P5 numerical fixture coefficients, explicitly NOT a calibrated pilot choice.
            reward_cfg = HOIRewardCfg((0.2, 0.4, 1.0, 2.0), (0.7, 0.8, 0.9, 1.1), 0.0423, 0.6, 0.02)
            proximity = ProximityReward(
                fk,
                ("left_wrist_yaw_link", "right_wrist_yaw_link"),
                manifest["contact"]["target_root_m"],
                manifest["contact"]["eef_offset_wrist_m"],
                sigma_pos=0.0423,
            )
            reward = HOITrackingReward(heads, proximity, reward_cfg)
            trainer = HOIPolicyTrainer(
                wm,
                policy,
                critic,
                registry,
                reward,
                PolicyTrainerCfg(horizon=3, batch_size=2, obs_noise=False, compile=None),
            )
            wmt = JointWorldModelTrainer(wm, [candidate], WMTrainerCfg(compile=None, batch_size=2))
            recorder = install_recording(
                env, force_on=0.001, capacity=96, references=[str(candidate.path)]
            )
            buffer = HOIPolicyReplayBuffer(2, 96, env.device, force_on=0.001, registry=registry)
            buffer.metadata.update(recorder.buffer.metadata)
            buffer.metadata["experimental_object"] = {
                "path": str(P6 / "candidates/single_body.json"),
                "sha256": PINS[
                    "outputs/hoi/contact_supervision/p6_closure/candidates/single_body.json"
                ],
                "production_selected": False,
            }
            recorder.buffer = buffer
            before = [
                {k: v.clone() for k, v in m.state_dict().items()} for m in (wm, policy, critic)
            ]
            rows, resets = [], []
            with (out / "transitions.jsonl").open("x") as log:
                for starts, count in (([0, 258], 12), ([548, 827], 12), ([838, 838], 1)):
                    frames = torch.tensor(starts, device=env.device)
                    io.reset_to_reference(ids, frames, cache)
                    torch.testing.assert_close(io.read(), cache.frames(frames), atol=1e-6, rtol=0)
                    assert torch.count_nonzero(term.executed_raw) == 0
                    recorder.begin_episode(ids, frames, torch.zeros_like(frames))
                    resets.append({"frames": starts, "geometry": reset_geometry(env, io)})
                    prev = torch.zeros(2, 29, device=env.device)
                    xi = recorder.xi.clone()
                    hist = analytic_observation(io.read()[:, :71], prev, xi, q0)[None].repeat(
                        10, 1, 1
                    )
                    initial = io.read().clone()
                    if not rows:
                        obs = flatten_history_term_major(hist)
                        goal = lib.goal_window(frames, 10)
                        with torch.no_grad():
                            mean = policy(obs, goal)
                            changed = initial.clone()
                            changed[:, 7] += 0.05
                            robot_obs = analytic_observation(changed[:, :71], prev, xi, q0)
                            robot_action = policy(
                                flatten_history_term_major(robot_obs[None].repeat(10, 1, 1)), goal
                            )
                            changed = initial.clone()
                            changed[:, 71] += 0.05
                            box_obs = analytic_observation(changed[:, :71], prev, xi, q0)
                            box_action = policy(
                                flatten_history_term_major(box_obs[None].repeat(10, 1, 1)), goal
                            )
                        result["feedback_probe"] = {
                            "robot_joint_0p05rad_action_max_change": float(
                                (mean - robot_action).abs().max()
                            ),
                            "box_x_0p05m_action_max_change": float((mean - box_action).abs().max()),
                            "actor_history_width": obs.shape[-1],
                            "goal_width": goal.shape[-1],
                            "critic_input_width": critic.mlp.net[0].in_features,
                            "joint_state_width": initial.shape[-1],
                            "box_feedback_available": False,
                        }
                        assert not torch.equal(mean, robot_action)
                        assert torch.equal(mean, box_action)
                    for step in range(count):
                        frame = recorder.frame.clone()
                        state = io.read().clone()
                        p = buffer.ptr.clone()
                        with torch.no_grad():
                            action = policy(
                                flatten_history_term_major(hist), lib.goal_window(frame, 10)
                            )
                        finite("Actor action", action)
                        env.step(action)
                        # Read transition-owned data even if native auto-reset has already run.
                        sn = buffer.fields["next_state"][ids, p]
                        torch.testing.assert_close(buffer.a[ids, p], action, atol=0, rtol=0)
                        torch.testing.assert_close(buffer.a_exec[ids, p], action, atol=0, rtol=0)
                        if not env.reset_buf.any():
                            torch.testing.assert_close(
                                term._processed_actions,
                                action * term._scale + term._offset,
                                atol=0,
                                rtol=0,
                            )
                        ref = registry.gather(torch.zeros_like(frame), frame + 1)
                        with torch.no_grad():
                            terms = reward(sn, action, prev, ref.robot, ref.box)
                        finite("reward/state", sn, *terms.values())
                        raw = buffer.fields["force_raw_world"][ids, p].cpu().numpy()
                        contact = actual_contact(
                            raw, calibration["thresholds_N_LR"], calibration["occupancy_cutoffs_LR"]
                        )
                        expected_intent = torch.stack(
                            (
                                registry.gather(torch.zeros_like(frame), frame).box.contact,
                                ref.box.contact,
                            ),
                            -1,
                        ).float()
                        assert torch.equal(buffer.fields["intent_ref"][ids, p], expected_intent)
                        row = {
                            "frame": frame.cpu().tolist(),
                            "state": state.cpu().tolist(),
                            "next_state": sn.cpu().tolist(),
                            "action": action.cpu().tolist(),
                            "intent": expected_intent.cpu().tolist(),
                            "hands": contact["hands"].tolist(),
                            "occupancy": contact["occupancy"].tolist(),
                            "force_peak_N": buffer.fields["force_peak"][ids, p].cpu().tolist(),
                            "reward": {k: v.cpu().tolist() for k, v in terms.items()},
                            "clip_end": buffer.fields["clip_end"][ids, p].cpu().tolist(),
                        }
                        log.write(json.dumps(row, allow_nan=False) + "\n")
                        log.flush()
                        rows.append(row)
                        hist = torch.cat(
                            (hist[1:], analytic_observation(sn[:, :71], action, xi, q0)[None])
                        )
                        prev = action
                        if env.reset_buf.any():
                            assert step == count - 1, "Never continue after an unbound auto-reset"
            batch = buffer.sample_windows(2, 8)
            loss, metrics = wmt.loss_on(batch)
            finite("WM loss", loss)
            loss.backward()
            result["wm_backward"] = {
                "loss": float(loss.detach()),
                "gradient_norm": grad_norm(wm),
                "metrics": {k: float(v) for k, v in metrics.items()},
                "window_states_shape": list(batch["states"].shape),
                "input": 166,
                "output": 82,
            }
            wm.zero_grad(set_to_none=True)
            start = buffer.sample_starts(2, 10)
            with frozen_parameters(wm, critic):
                actor_loss, _objective, rollout, critic_inputs, xi_norm = trainer.actor_loss(start)
                finite("Actor loss", actor_loss)
                actor_loss.backward()
            result["actor_backward"] = {
                "loss": float(actor_loss.detach()),
                "gradient_norm": grad_norm(policy),
                "wm_has_grad": any(p.grad is not None for p in wm.parameters()),
            }
            X, Y, _, _ = trainer.critic_batch(rollout, xi_norm, critic_inputs)
            critic_loss = (critic(*X) - Y).square().mean()
            finite("Critic loss", critic_loss)
            critic_loss.backward()
            result["critic_backward"] = {
                "loss": float(critic_loss.detach()),
                "gradient_norm": grad_norm(critic),
            }
            with frozen_parameters(wm):
                a = start.prev_action.clone().requires_grad_()
                box_grad = torch.autograd.grad(wm.step(start.state, a, start.xi)[:, 71:].sum(), a)[
                    0
                ]
            result["initialized_wm_box_action_gradient_norm"] = float(box_grad.norm())
            ck = joint_checkpoint(
                wm,
                replay_metadata=buffer.metadata,
                provenance={
                    "role": "P7.0 backward-only smoke",
                    "pins": PINS,
                    "optimizer_steps": 0,
                    "training_ready": False,
                },
            )
            torch.save(
                {"joint": ck, "policy": policy.state_dict(), "critic": critic.state_dict()},
                out / "smoke_weights.pt",
            )
            loaded = torch.load(
                out / "smoke_weights.pt", map_location=env.device, weights_only=False
            )
            clones = [copy.deepcopy(m) for m in (wm, policy, critic)]
            load_joint_weights(clones[0], loaded["joint"], expected_replay_metadata=buffer.metadata)
            clones[1].load_state_dict(loaded["policy"], strict=True)
            clones[2].load_state_dict(loaded["critic"], strict=True)
            for m, clone, old in zip((wm, policy, critic), clones, before):
                assert all(
                    torch.equal(v, old[k]) and torch.equal(v, clone.state_dict()[k])
                    for k, v in m.state_dict().items()
                )
            result["checkpoint"] = {
                "strict_weights_roundtrip": True,
                "parameters_unchanged": True,
                "optimizer_updates": 0,
                "resume_tested": False,
                "sha256": sha256(out / "smoke_weights.pt"),
            }
            result["collection"] = {
                "resets": resets,
                "control_steps": len(rows),
                "world_transitions": len(rows) * 2,
                "substep_checks": len(substeps),
                "reset_only_state_writes": True,
                "external_wrench": False,
                "source": "untrained actual Policy network; state-dependent native actions",
                "reward_cfg_test_only": reward_cfg.metadata(),
            }
            result["guard_kernel_launches"] = dict(launches)
            result["cuda_graph_launches"] = len(graph_launches)
            assert any("smhoi_p6_linesearch_precision" in name for name in launches), launches
            assert graph_launches
            result["preflight_status"] = "PASS"
            result["training_process_guard_status"] = (
                "NOT VERIFIED: integrated HOI training entrypoint absent"
            )
            result["p7_1_ready"] = False
        except BaseException:
            torch.save(
                {
                    name: getattr(env.sim.data, name).detach().cpu().clone()
                    for name in (
                        "qpos",
                        "qvel",
                        "qacc",
                        "ctrl",
                        "overflow",
                        "xfrc_applied",
                        "qfrc_applied",
                    )
                },
                out / "failure_state.pt",
            )
            raise
        finally:
            save(out / "substeps.json", substeps)
            save(out / "guard_launches.json", dict(launches))
            env.close()
    assert solver._linesearch_iterative_kernel is baseline_factory
    assert sha256(solver.__file__) == SOURCE_SHA256
    result["guard_restored_dependency_unchanged"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "source.py")
    result = {"role": "P7.0 preflight, not training", "optimizer_updates": 0}
    try:
        run(args.output, result)
    except BaseException:
        result["preflight_status"] = "FAIL"
        result["exception"] = traceback.format_exc()
        raise
    finally:
        save(args.output / "result.json", result)


if __name__ == "__main__":
    main()
