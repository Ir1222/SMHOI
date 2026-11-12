"""P4 offline geometry audit; no physics stepping, contact labels or training.

Reuses the existing asset-audit geometry helpers without invoking their mains.
Output directories must be new; accepted reference/assets are read-only.
"""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from asset_common import geom, rot
from audit_contact import compound_point, point_primitive
from audit_robot import target_spec

from supermimic.hoi.asset_manifest import load_manifest
from supermimic.hoi.reference import DEFAULT_ASSET, DEFAULT_OUTPUT, REPO, ReferenceClip, sha256
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply

ASSET_SHA = "1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1"
REFERENCE_SHA = "bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b"


def distribution(values):
    x = np.asarray(values)
    return {
        **dict(
            zip(
                ("min", "p25", "p50", "p75", "max"), np.percentile(x, [0, 25, 50, 75, 100]).tolist()
            )
        ),
        "mean": float(x.mean()),
    }


def inspect_geometry():
    """Read exact accepted inputs and return reproducible, geometric-only evidence."""
    if sha256(DEFAULT_ASSET) != ASSET_SHA:
        raise ValueError("Unexpected M6.5 manifest")
    manifest = load_manifest(DEFAULT_ASSET, verify_sources=False)
    provenance = []
    for source in manifest["sources"]:
        p = Path(source["path"])
        actual = sha256(p) if p.is_file() else None
        provenance.append({**source, "actual_sha256": actual, "exists": p.is_file()})
        if actual is not None and actual != source["sha256"]:
            raise ValueError(f"Changed source: {p}")
    clip = ReferenceClip(DEFAULT_OUTPUT / "reference.npz")
    if sha256(clip.path) != REFERENCE_SHA:
        raise ValueError("Unexpected original Part B reference")
    a, contact = clip.arrays, manifest["contact"]
    model = target_spec().compile()
    fk = G1Kinematics(model, dtype=torch.float64)
    root = a["body_names"].tolist().index("pelvis")
    order = [a["joint_names"].tolist().index(n) for n in fk.joint_names[1:]]
    qpos = np.concatenate(
        (a["body_pos_w"][:, root], a["body_quat_w"][:, root], a["joint_pos"][:, order]), -1
    )
    bodies = fk(torch.tensor(qpos, dtype=torch.float64))
    names = contact["wrist_bodies"]
    indices = fk.body_indices(names)
    offsets = torch.tensor(contact["eef_offset_wrist_m"], dtype=torch.float64)
    eef = bodies["pos"][:, indices] + quat_apply(
        bodies["quat"][:, indices], offsets.expand(840, -1, -1)
    )
    targets = np.stack(
        [
            a["object_pos_w"] + rot(a["object_quat_w"]).apply(np.broadcast_to(p, (840, 3)))
            for p in contact["target_root_m"]
        ],
        1,
    )
    distances = np.linalg.norm(eef.numpy() - targets, axis=-1)
    data = mujoco.MjData(model)
    cpu_error, recorded_error, rotation_error = [], [], []
    for frame in range(840):
        data.qpos[:] = qpos[frame]
        mujoco.mj_kinematics(model, data)
        for i, name in enumerate(names):
            body = model.body(name).id
            cpu_eef = data.xpos[body] + rot(data.xquat[body]).apply(offsets[i].numpy())
            cpu_error.append(np.linalg.norm(cpu_eef - eef[frame, i].numpy()))
            ref_index = a["body_names"].tolist().index(name)
            recorded_error.append(
                np.linalg.norm(data.xpos[body] - a["body_pos_w"][frame, ref_index])
            )
            rotation_error.append(
                float(
                    (
                        rot(data.xquat[body]).inv() * rot(a["body_quat_w"][frame, ref_index])
                    ).magnitude()
                )
            )
    hand_geoms = [
        geom(model, i)
        for i in range(model.ngeom)
        if model.geom_contype[i]
        and model.body(model.geom_bodyid[i]).name
        in [f"{s}_wrist_{j}_link" for s in ("left", "right") for j in ("pitch", "yaw")]
    ]
    target_checks = []
    for i, side in enumerate(("left", "right")):
        k, (distance, closest, normal) = compound_point(
            contact["target_root_m"][i], manifest["object"]["geoms"]
        )
        capsule = next(g for g in hand_geoms if g["name"] == f"{side}_hand_collision")
        target_checks.append(
            {
                "side": side,
                "box_geom": manifest["object"]["geoms"][k]["name"],
                "target_surface_distance_m": distance,
                "closest_root_m": closest.tolist(),
                "normal_root": normal.tolist(),
                "eef_inside_yaw_capsule_signed_distance_m": point_primitive(
                    offsets[i].numpy(), capsule
                ),
            }
        )
    historical = list(csv.DictReader((REPO / "docs/hoi/asset_audit/contact_points.csv").open()))
    csv_error = max(
        abs(
            float(r["eef_error_m"]) - distances[int(r["frame"]), ("left", "right").index(r["side"])]
        )
        for r in historical
    )
    mask = a["object_contact"]
    assert len(historical) == 1680
    assert all(bool(int(r["desired_contact"])) == bool(mask[int(r["frame"])]) for r in historical)
    result = {
        "mode": "P4 kinematic geometry only; no physical rollout or measured contact",
        "reference_path": str(clip.path),
        "reference_sha256": REFERENCE_SHA,
        "asset_sha256": ASSET_SHA,
        "manifest_sources": provenance,
        "contact_contract": contact,
        "asset_status": manifest["status"],
        "training_ready": manifest["training_ready"],
        "asset_gates": manifest["gates"],
        "frames": 840,
        "fps": clip.fps,
        "on_frames": np.flatnonzero(mask).tolist(),
        "on_count": int(mask.sum()),
        "off_count": int((~mask).sum()),
        "native_hand_geoms": hand_geoms,
        "target_checks": target_checks,
        "torch_vs_mujoco_eef_max_m": float(max(cpu_error)),
        "fk_vs_recorded_wrist_max_m": float(max(recorded_error)),
        "fk_vs_recorded_wrist_rotation_max_rad": float(max(rotation_error)),
        "historical_csv_distance_max_difference_m": float(csv_error),
        "distance_m": {
            s: {
                label: distribution(distances[m, i])
                for label, m in (("ON", mask), ("OFF", ~mask), ("all", np.ones(840, dtype=bool)))
            }
            for i, s in enumerate(("left", "right"))
        },
        "sigma_candidates_m": np.percentile(distances[mask], [25, 50, 75]).tolist(),
        "bilateral_task_requirement": {
            "established_by": "M6.5 contact.physical_validation.bilateral_force_required=true",
            "scope": "reference ON frames 260–549; no independent per-hand reference annotation",
            "achieved_physical_contact": "NOT ESTABLISHED; preserve any-hand/bilateral diagnostics separately",
        },
    }
    assert max(cpu_error) <= 1e-10
    assert max(recorded_error) <= 2e-6
    assert csv_error <= 2e-6
    assert all(abs(c["target_surface_distance_m"]) <= 1e-7 for c in target_checks)
    rows = [
        {
            "frame": f,
            "contact_ref": int(mask[f]),
            "left_distance_m": distances[f, 0],
            "right_distance_m": distances[f, 1],
        }
        for f in range(840)
    ]
    return result, rows, clip, fk, qpos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result, rows, _, _, _ = inspect_geometry()
    args.output.mkdir(parents=True)
    (args.output / "geometry.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.output / "distances.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "on_count",
                    "off_count",
                    "torch_vs_mujoco_eef_max_m",
                    "fk_vs_recorded_wrist_max_m",
                    "distance_m",
                    "sigma_candidates_m",
                    "target_checks",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
