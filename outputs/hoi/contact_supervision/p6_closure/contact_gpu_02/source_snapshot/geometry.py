"""All-frame native collision feasibility for immutable HDMI and object candidates."""

import argparse
import itertools
import shutil
import numpy as np
import mujoco
from closure_common import (
    E,
    active_pairs,
    candidate,
    cpu_model,
    dump_gz,
    inputs,
    reference_qpos,
    save,
)


def evaluate(name, out):
    m = cpu_model(name)
    d = mujoco.MjData(m)
    q = reference_qpos(m)
    obj, robot, hands, non = active_pairs(m)
    a = inputs()[1].arrays
    rows = []
    contact_records = []
    target = np.array(inputs()[0]["contact"]["target_root_m"])
    box = m.body("box/bread_box").id
    wrists = [m.body("robot/" + s + "_wrist_yaw_link").id for s in ("left", "right")]
    leg = [
        g for g in non if any(x in m.geom(g).name for x in ("thigh", "shin", "hip", "knee", "foot"))
    ]
    points = np.concatenate(
        [
            np.array(g["pos"])
            + np.array(g["size"]) * np.array(list(itertools.product((-1, 1), repeat=3)))
            for g in candidate(name)["geoms"]
        ]
    )

    def nearest(gs):
        values = [
            (float(mujoco.mj_geomDistance(m, d, g, b, 10.0, None)), g, b) for g in gs for b in obj
        ]
        dist, g, b = min(values)
        return {"distance": dist, "pair": [m.geom(g).name, m.geom(b).name]}

    for k in range(840):
        d.qpos[:] = q[k]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        R = d.xmat[box].reshape(3, 3)
        p = d.xpos[box]
        eef = np.array(
            [d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0.05, 0, 0]) for b in wrists]
        )
        centre = np.array([d.geom_xpos[h[1]] for h in hands])
        local = (eef - p) @ R
        self_contacts = []
        for c in d.contact[: d.ncon]:
            if c.geom1 in robot and c.geom2 in robot:
                self_contacts.append(float(c.dist))
            if c.geom1 in obj or c.geom2 in obj:
                contact_records.append(
                    {
                        "frame": k,
                        "dist": c.dist,
                        "pos": c.pos.copy(),
                        "frame_matrix": c.frame.copy(),
                        "geom1": int(c.geom1),
                        "geom2": int(c.geom2),
                        "names": [m.geom(c.geom1).name, m.geom(c.geom2).name],
                    }
                )
        vertices = p + points @ R.T
        rows.append(
            {
                "frame": k,
                "intent": bool(a["object_contact"][k]),
                "object_root_world": p.copy(),
                "object_quat_wxyz": d.xquat[box].copy(),
                "world_aabb_min": vertices.min(0),
                "world_aabb_max": vertices.max(0),
                "eef_world": eef,
                "eef_local": local,
                "hand_centre_local": (centre - p) @ R,
                "eef_separation_m": float(np.linalg.norm(eef[0] - eef[1])),
                "hand_axis_box_local": [R.T @ d.geom_xmat[h[1]].reshape(3, 3)[:, 2] for h in hands],
                "eef_target_distance_m": np.linalg.norm(local - target, axis=1),
                "hand_nearest": [nearest(h) for h in hands],
                "nonhand_nearest": nearest(non),
                "leg_nearest": nearest(leg),
                "self_min_dist_m": min(self_contacts, default=0.0),
            }
        )
    dist = np.array([[h["distance"] for h in r["hand_nearest"]] for r in rows])
    non_d = np.array([r["nonhand_nearest"]["distance"] for r in rows])
    leg_d = np.array([r["leg_nearest"]["distance"] for r in rows])
    eef = np.array([r["eef_local"] for r in rows])
    approach = np.gradient(eef, 0.02, axis=0)
    for k in range(840):
        rows[k]["eef_relative_velocity_box_m_s"] = approach[k]
    on = a["object_contact"]
    ok = np.all((dist >= -0.001) & (dist <= 0.002), axis=1)
    summary = {
        "candidate": name,
        "frames": 840,
        "ON_frames": 290,
        "bilateral_band_ON": int((ok & on).sum()),
        "ON_min_hand_distance_LR": dist[on].min(0),
        "ON_max_hand_gap_LR": dist[on].max(0),
        "ON_nonhand_penetration_frames": int(((non_d < -0.001) & on).sum()),
        "all_nonhand_min": float(non_d.min()),
        "all_leg_min": float(leg_d.min()),
        "ON_leg_min": float(leg_d[on].min()),
        "ON_eef_separation_min_median_max": np.quantile(
            [r["eef_separation_m"] for r in rows if r["intent"]], [0, 0.5, 1]
        ),
        "ON_hand_centre_box_coordinate_min": np.array([r["hand_centre_local"] for r in rows])[
            on
        ].min(0),
        "ON_hand_centre_box_coordinate_max": np.array([r["hand_centre_local"] for r in rows])[
            on
        ].max(0),
        "original_qpos_unchanged": True,
        "joint_limits_violating_frames": int(
            sum(
                any(
                    d < lo - 1e-7 or d > hi + 1e-7
                    for d, lo, hi in zip(
                        qq[m.jnt_qposadr[m.jnt_limited.astype(bool)]],
                        m.jnt_range[m.jnt_limited.astype(bool), 0],
                        m.jnt_range[m.jnt_limited.astype(bool), 1],
                    )
                )
                for qq in q
            )
        ),
    }
    folder = out / name
    folder.mkdir()
    dump_gz(folder / "frames.json.gz", rows)
    dump_gz(folder / "contacts.json.gz", contact_records)
    save(folder / "summary.json", summary)
    print(
        name,
        "bilateral",
        summary["bilateral_band_ON"],
        "nonhand",
        summary["all_nonhand_min"],
        flush=True,
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    from pathlib import Path

    out = Path(a.output)
    out.mkdir(exist_ok=False)
    shutil.copytree(
        Path(__file__).parent, out / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__")
    )
    save(
        out / "summary.json",
        [evaluate(p.stem, out) for p in sorted((E / "candidates").glob("*.json"))],
    )
