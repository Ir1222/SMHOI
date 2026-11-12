"""Audit saved solver samples, physical trajectories, slip and preservation reuse."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from common import ROOT, rot, save

from supermimic.hoi.contact_validation import actual_contact, penetration_summary
from supermimic.hoi.reference import sha256


def com_twist(state, com):
    r = rot(state[74:78])
    omega = r.apply(state[81:84])
    return state[78:81] + np.cross(omega, r.apply(com)), omega


def analyze_case(folder, tau, rho, model, origins):
    samples = json.loads((folder / "samples.json").read_text())
    rows = json.loads((folder / "intervals.json").read_text())
    with gzip.open(folder / "contacts.json.gz", "rt") as f:
        all_contacts = json.load(f)
    contacts = [c for c in all_contacts if c["type"] & 1]
    body = model.body("box/bread_box").id
    com = model.body_ipos[body]
    flags = np.array([x["overflow"] for x in samples])
    solver = np.array([x["solver_niter"] for x in samples])
    result = {
        "result": json.loads((folder / "result.json").read_text()),
        "contact_type_counts": dict(Counter(str(x["type"]) for x in all_contacts)),
        "max_nacon_shared": max(x["nacon"] for x in samples),
        "max_nefc": int(np.array([x["nefc"] for x in samples]).max()),
        "max_solver_niter": int(solver.max()),
        "flagged_samples": [
            {"sample": int(i), "env": int(e), "flags": int(flags[i, e]), "time_s": (i + 1) * 0.005}
            for i, e in np.argwhere(flags != 0)
        ],
        "constraint_penetration_by_env": {},
        "all_records_penetration_by_env": {},
        "worlds": {},
    }
    rq = list(range(36))
    rv = list(range(35))
    bq = list(range(36, 43))
    bv = list(range(35, 41))
    # Resolve both free roots and hinges by names; compiled global ordering can differ.
    robot_free = model.joint("robot/floating_base_joint").id
    box_free = model.joint("box/bread_box_root").id
    rq = list(range(model.jnt_qposadr[robot_free], model.jnt_qposadr[robot_free] + 7))
    rv = list(range(model.jnt_dofadr[robot_free], model.jnt_dofadr[robot_free] + 6))
    # Native qpos order is also the named JointStateIO order, verified against reference.
    robot_joints = sorted(
        [
            i
            for i in range(model.njnt)
            if model.joint(i).name.startswith("robot/")
            and model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
        ],
        key=lambda i: model.jnt_qposadr[i],
    )
    rq += [int(model.jnt_qposadr[i]) for i in robot_joints]
    rv += [int(model.jnt_dofadr[i]) for i in robot_joints]
    bq = list(range(model.jnt_qposadr[box_free], model.jnt_qposadr[box_free] + 7))
    bv = list(range(model.jnt_dofadr[box_free], model.jnt_dofadr[box_free] + 6))
    initial = np.array(json.loads((folder / "initial.json").read_text())["state84"])
    for e in range(4):
        cs = [c for c in contacts if c["env"] == e]
        result["constraint_penetration_by_env"][str(e)] = penetration_summary(cs, len(samples))
        result["all_records_penetration_by_env"][str(e)] = penetration_summary(
            [c for c in all_contacts if c["env"] == e], len(samples)
        )
        states = np.array([x["state84"][e] for x in samples])
        twists = np.array([com_twist(x, com) for x in states])
        force_support = np.zeros((len(samples), 3))
        slip = np.zeros(2)
        data = mujoco.MjData(model)
        grouped = {}
        for c in cs:
            grouped.setdefault(c["sample"], []).append(c)
            if c["category"] == "support":
                sign = 1 if c["names"][1].startswith("box/") else -1
                force_support[c["sample"]] += sign * np.array(c["world_force_on_geom2_N"])
        for k, points in grouped.items():
            hands = [c for c in points if c["category"] in ("left", "right")]
            if not hands:
                continue
            state = np.array(samples[k].get("solve_state84", samples[k]["state84"])[e])
            data.qpos[rq] = state[:36]
            data.qvel[rv] = state[36:71]
            data.qpos[bq] = state[71:78]
            data.qvel[bv] = state[78:84]
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_fwdVelocity(model, data)
            for h, side in enumerate(("left", "right")):
                weights = []
                speeds = []
                for c in hands:
                    if c["category"] != side:
                        continue
                    normal = np.array(c["frame"])[0]
                    weight = abs(np.dot(c["world_force_on_geom2_N"], normal))
                    if weight <= tau[h]:
                        continue
                    other = c["geom2"] if c["names"][0].startswith("box/") else c["geom1"]
                    point = np.array(c["pos"]) - origins[e]
                    velocities = []
                    for b in (int(model.geom_bodyid[other]), body):
                        v = np.zeros(6)
                        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, v, 0)
                        velocities.append(v[3:] + np.cross(v[:3], point - data.xpos[b]))
                    rel = velocities[0] - velocities[1]
                    tangent = rel - np.dot(rel, normal) * normal
                    weights.append(weight)
                    speeds.append(np.linalg.norm(tangent))
                if weights:
                    slip[h] += np.average(speeds, weights=weights) * 0.005
        raw = (
            np.array([x["force_raw_world"][e] for x in rows]) if rows else np.empty((0, 4, 2, 2, 3))
        )
        actual = actual_contact(raw, tau, rho)
        result["worlds"][str(e)] = {
            "initial_box_position": initial[e, 71:74],
            "final_box_position": states[-1, 71:74],
            "box_displacement_m": float(np.linalg.norm(states[-1, 71:74] - initial[e, 71:74])),
            "max_box_height_change_m": float((states[:, 73] - initial[e, 73]).max()),
            "final_root_z": states[-1, 2],
            "tail_max_box_COM_speed_m_s": float(np.linalg.norm(twists[-100:, 0], axis=-1).max()),
            "tail_max_box_angular_speed_rad_s": float(
                np.linalg.norm(twists[-100:, 1], axis=-1).max()
            ),
            "tail_mean_support_world_N": force_support[-100:].mean(0),
            "measured_hand_contact_duration_LR_s": actual["hands"].sum(0) * 0.02,
            "force_bearing_duration_LR_s": actual["occupancy"].sum(0) * 0.02,
            "last_contact_interval_LR": [
                int(np.flatnonzero(actual["hands"][:, h])[-1])
                if actual["hands"][:, h].any()
                else None
                for h in range(2)
            ],
            "peak_N_LR": actual["magnitude_N"].max((0, 1)) if len(rows) else None,
            "slip_path_m_LR": slip,
            "slip_convention": "normal-load-weighted relative tangential point speed integrated at 5ms; threshold calibrated hand force value; solve-state kinematics",
            "slip_solve_state_available": all("solve_state84" in x for x in samples),
        }
    other = folder / "non_box_contacts.json.gz"
    if other.exists():
        with gzip.open(other, "rt") as f:
            ncs = json.load(f)
        result["non_box_type_counts"] = dict(Counter(str(c["type"]) for c in ncs))
        result["non_box_at_failure"] = [
            c for c in ncs if c["type"] & 1 and flags[c["sample"], c["env"]] != 0
        ]
    return result


def main(root, output):
    output.mkdir(exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(root / "geometry_02/model.mjb"))
    inventory = json.loads((root / "geometry_02/inventory.json").read_text())
    origins = np.array(inventory["origins"])
    cal = json.loads((root / "calibration.json").read_text())
    tau = cal["thresholds_N_LR"]
    rho = cal["occupancy_cutoffs_LR"]
    results = {}
    for name in ("dynamics_01", "isolation_01", "filters_02", "replay_01"):
        for folder in sorted((root / name).iterdir()):
            if folder.is_dir() and (folder / "samples.json").exists():
                results[name + "/" + folder.name] = analyze_case(folder, tau, rho, model, origins)
    save(output / "dynamics_analysis.json", results)
    # Preserve checksums of reused P1/P2 evidence; current implementation can contain
    # documented P2 extensions, so compare to its final hashes, not old P1 whole file.
    reuse = {}
    for stage in ("p1", "p2"):
        old = json.loads(
            (ROOT / f"docs/hoi/force_contact/{stage}/validation_summary.json").read_text()
        )
        reused = {}
        for group in ("implementation_sha256", "evidence_sha256"):
            for path, expected in old[group].items():
                p = Path(path)
                p = p if p.is_absolute() else ROOT / p
                actual = sha256(p) if p.is_file() else None
                reused[str(p)] = {
                    "expected": expected,
                    "actual": actual,
                    "match": actual == expected,
                }
        reuse[stage] = reused
    save(output / "evidence_reuse.json", reuse)
    print("Analyzed", len(results), "GPU cases")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.output)
