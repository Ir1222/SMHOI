"""Static signed-distance evidence for the exact reference reset states.

No simulation steps, pose corrections, collision edits or training approval.
The 1 mm reporting band comes from the existing P6 penetration diagnostics.
"""

import mujoco
import numpy as np


def inspect_reference_resets(model, cache, io, frames):
    data = mujoco.MjData(model)
    rq, bq = io.robot_q.cpu().numpy(), io.box_q.cpu().numpy()
    rv, bv = io.robot_v.cpu().numpy(), io.box_v.cpu().numpy()
    rows = []
    for frame in frames:
        state = cache.states[frame].cpu().numpy()
        data.qpos[rq], data.qpos[bq] = state[:36], state[71:78]
        data.qvel[rv], data.qvel[bv] = state[36:71], state[78:84]
        mujoco.mj_forward(model, data)
        if not all(np.isfinite(a).all() for a in (data.qpos, data.qvel, data.qacc)):
            raise FloatingPointError(f"Nonfinite reference reset forward at frame {frame}")
        contacts = [
            {
                "distance_m": float(c.dist),
                "geom1": model.geom(c.geom1).name,
                "geom2": model.geom(c.geom2).name,
            }
            for c in data.contact
            if c.dist < 0
        ]
        contacts.sort(key=lambda c: c["distance_m"])
        rows.append(
            {
                "frame": int(frame),
                "min_distance_m": contacts[0]["distance_m"] if contacts else 0.0,
                "penetrating_contacts": contacts,
            }
        )
    worst = min(rows, key=lambda row: row["min_distance_m"])
    count = sum(row["min_distance_m"] < -0.001 for row in rows)
    return {
        "scope": "CPU mj_forward on native scene, exact reference resets; zero control steps",
        "status": "UNRESOLVED_RESET_OVERLAP" if count else "NO_OVERLAP_ABOVE_REPORTING_BAND",
        "physical_validity": "NOT_ESTABLISHED",
        "reporting_band_m": 0.001,
        "frames_above_band": count,
        "worst_frame": worst["frame"],
        "max_penetration_m": -worst["min_distance_m"],
        "frames": rows,
    }
