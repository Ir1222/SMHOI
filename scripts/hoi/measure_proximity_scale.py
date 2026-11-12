"""Bounded P4 reference-state reward-scale measurements; no optimization."""

import argparse
import json
from pathlib import Path

import torch
from audit_proximity_reward import ASSET_SHA, REFERENCE_SHA, distribution

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.hoi.reference import DEFAULT_ASSET, DEFAULT_OUTPUT, ReferenceClip, sha256
from supermimic.hoi.rewards import ProximityReward, reference_contact_at
from supermimic.hoi.state import ReferenceResetCache
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_from_rotvec, quat_mul
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.geometry.read_text())
    clip = ReferenceClip(DEFAULT_OUTPUT / "reference.npz")
    if audit["asset_sha256"] != ASSET_SHA or sha256(DEFAULT_ASSET) != ASSET_SHA:
        raise ValueError("Asset identity mismatch")
    if audit["reference_sha256"] != REFERENCE_SHA or sha256(clip.path) != REFERENCE_SHA:
        raise ValueError("Reference identity mismatch")
    model = load_g1_model()
    fk = G1Kinematics(model, dtype=torch.float64)
    # Reuse the actual state contract/reset adapter; it derives twist in memory.
    states = ReferenceResetCache(clip, tuple(fk.joint_names[1:]), "cpu").states.double()
    labels = torch.tensor(clip.arrays["object_contact"].copy())
    gates = reference_contact_at(labels, torch.arange(840))
    a = clip.arrays
    idx = [a["body_names"].tolist().index(n) for n in TRACKED_BODIES]
    root = a["body_names"].tolist().index(ROOT_BODY)
    ref = ReferenceFrame(
        *[
            torch.tensor(a[k][:, idx].copy(), dtype=torch.float64)
            for k in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        ],
        torch.tensor(a["body_pos_w"][:, root].copy(), dtype=torch.float64),
        torch.tensor(a["body_quat_w"][:, root].copy(), dtype=torch.float64),
    )
    heads = AnalyticHeads(
        fk,
        TRACKED_BODIES,
        ANCHOR_BODY,
        ROOT_BODY,
        torch.tensor(model.jnt_range[1:]),
        torch.tensor(model.qpos0[7:]),
    )
    actions = torch.zeros(840, 29, dtype=torch.float64)
    terms = heads.reward(states[:, :71], actions, actions, ref)
    # Six positive tracking terms only: actions are placeholders, not a rollout.
    positive_robot = sum(v for k, v in terms.items() if k.startswith("motion_"))
    robot_scale = float(positive_robot[gates].mean())
    manifest = json.loads(DEFAULT_ASSET.read_text())
    contact = manifest["contact"]
    candidates = []
    for sigma in audit["sigma_candidates_m"]:
        reward = ProximityReward(
            fk,
            tuple(contact["wrist_bodies"]),
            contact["target_root_m"],
            contact["eef_offset_wrist_m"],
            sigma_pos=sigma,
        )
        state = states.clone().requires_grad_()
        result = reward(state, gates)
        grad = torch.autograd.grad(result["proximity"].sum(), state)[0]
        values = result["proximity"].detach()
        assert torch.count_nonzero(values[~gates]) == 0
        assert torch.count_nonzero(grad[~gates]) == 0
        assert torch.isfinite(grad).all()
        response = []
        for axis in range(3):
            for sign in (-1, 1):
                shifted = states.clone()
                shifted[:, 71 + axis] += sign * 0.005
                rotated = states.clone()
                angle = torch.zeros(840, 3, dtype=states.dtype)
                angle[:, axis] = sign * 0.02
                rotated[:, 74:78] = quat_mul(rotated[:, 74:78], quat_from_rotvec(angle))
                response.append(
                    {
                        "axis": axis,
                        "translation_m": sign * 0.005,
                        "body_rotation_rad": sign * 0.02,
                        "translation_reward_delta_ON": distribution(
                            (reward(shifted, gates)["proximity"] - values)[gates].numpy()
                        ),
                        "rotation_reward_delta_ON": distribution(
                            (reward(rotated, gates)["proximity"] - values)[gates].numpy()
                        ),
                    }
                )
        mean = float(values[gates].mean())
        candidates.append(
            {
                "sigma_m": sigma,
                "raw_proximity_ON": distribution(values[gates].numpy()),
                "OFF_max_abs_reward": float(values[~gates].abs().max()),
                "OFF_max_abs_state_gradient": float(grad[~gates].abs().max()),
                "joint_gradient_norm_ON": distribution(grad[gates, 7:36].norm(dim=-1).numpy()),
                "box_position_gradient_norm_ON": distribution(
                    grad[gates, 71:74].norm(dim=-1).numpy()
                ),
                "box_quaternion_gradient_norm_ON": distribution(
                    grad[gates, 74:78].norm(dim=-1).numpy()
                ),
                "illustrative_contact_coefficients": [
                    {
                        "fraction_of_robot_ON_mean": fraction,
                        "coefficient_including_dt": fraction * robot_scale / mean,
                        "weighted_proximity_ON_mean": fraction * robot_scale,
                    }
                    for fraction in (0.1, 0.25, 0.5)
                ],
                "controlled_geometry_responses": response,
            }
        )
    report = {
        "mode": "P4 reference-state algebraic probes; no action rollout or training",
        "geometry_evidence_sha256": sha256(args.geometry),
        "reference_sha256": REFERENCE_SHA,
        "asset_sha256": ASSET_SHA,
        "robot_positive_tracking_scale": distribution(positive_robot.detach().numpy()),
        "robot_positive_tracking_ON_mean": robot_scale,
        "robot_cfg_dt": heads.reward_cfg.dt,
        "object_tracking_scale": None,
        "object_tracking_limitation": "No target-repository object-tracking reward exists; no substitute equation or weight invented",
        "candidates": candidates,
        "final_sigma_or_weights_selected": False,
        "optimizer_steps": 0,
    }
    args.output.mkdir(parents=True)
    (args.output / "scales.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "robot_ON_mean": robot_scale,
                "candidates": [
                    {
                        "sigma_m": c["sigma_m"],
                        "proximity_ON_mean": c["raw_proximity_ON"]["mean"],
                        "joint_gradient_ON_mean": c["joint_gradient_norm_ON"]["mean"],
                        "illustrative_coefficients": c["illustrative_contact_coefficients"],
                    }
                    for c in candidates
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
