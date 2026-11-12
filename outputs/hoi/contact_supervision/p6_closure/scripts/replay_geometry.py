"""Post-process actual integrated states; no replacement of measured dynamics."""

import gzip
import json
import mujoco
import numpy as np
from closure_common import E, P6, active_pairs, cpu_model, dump_gz, indices, inputs, save
from scipy.spatial.transform import Rotation


def main():
    out = E / "replay_geometry"
    out.mkdir(exist_ok=False)
    m = cpu_model("single_body")
    d = mujoco.MjData(m)
    bq, _bv, names, jq, _jv = indices(m)
    order = json.loads((P6 / "geometry_02/inventory.json").read_text())["runtime_joint_names"]
    source = [order.index(n) for n in names]
    obj, _robot, hands, non = active_pairs(m)
    wrists = [m.body("robot/" + s + "_wrist_yaw_link").id for s in ("left", "right")]
    box = m.body("box/bread_box").id
    targets = np.array(inputs()[0]["contact"]["target_root_m"])
    reference = inputs()[1].arrays
    summaries = {}
    for profile in ("original", "candidate"):
        with gzip.open(E / f"analysis_02/{profile}_rows.json.gz", "rt") as f:
            rows = json.load(f)
        results = []
        for row in rows:
            s = np.array(row["state84"])
            d.qpos[:7] = s[:7]
            d.qpos[jq] = s[7:36][source]
            d.qpos[bq : bq + 7] = s[71:78]
            mujoco.mj_kinematics(m, d)
            eef = np.array(
                [d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0.05, 0, 0]) for b in wrists]
            )
            actual_targets = d.xpos[box] + targets @ d.xmat[box].reshape(3, 3).T
            frame = row["frame_end"]
            reference_targets = reference["object_pos_w"][frame] + Rotation.from_quat(
                reference["object_quat_w"][frame], scalar_first=True
            ).apply(targets)

            def nearest(gs):
                return min(
                    float(mujoco.mj_geomDistance(m, d, g, b, 10, None)) for g in gs for b in obj
                )

            results.append(
                {
                    "episode": row["episode"],
                    "frame": frame,
                    "reference_intent": row["intent"],
                    "actual_EEF": eef,
                    "actual_Box_targets": actual_targets,
                    "reference_Box_targets": reference_targets,
                    "distance_to_actual_target_m": np.linalg.norm(eef - actual_targets, axis=1),
                    "distance_to_reference_target_m": np.linalg.norm(
                        eef - reference_targets, axis=1
                    ),
                    "native_surface_distance_LR_m": [nearest(h) for h in hands],
                    "nonhand_clearance_m": nearest(non),
                    "measured_contact_LR": row["hands"],
                    "measured_occupancy_LR": row["occupancy"],
                    "Box_root_error_m": float(
                        np.linalg.norm(s[71:74] - reference["object_pos_w"][frame])
                    ),
                }
            )
        dump_gz(out / (profile + ".json.gz"), results)
        on = np.array([r["reference_intent"] for r in results])
        summary = {"rows": len(results), "end_state_geometry_not_interval_force": True}
        for field in (
            "distance_to_actual_target_m",
            "distance_to_reference_target_m",
            "native_surface_distance_LR_m",
            "Box_root_error_m",
        ):
            v = np.array([r[field] for r in results])[on]
            summary[field] = {
                "ON_min": v.min(0),
                "ON_median": np.median(v, axis=0),
                "ON_max": v.max(0),
            }
        summaries[profile] = summary
    save(out / "summary.json", summaries)


if __name__ == "__main__":
    main()
