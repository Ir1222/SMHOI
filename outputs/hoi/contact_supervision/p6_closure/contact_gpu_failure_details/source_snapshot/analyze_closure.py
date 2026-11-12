"""Recompute C1-C4 numerical decisions from preserved physical evidence."""

import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from closure_common import E, P6, candidate, inputs, rot, save, dump_gz
from supermimic.hoi.contact_validation import actual_contact, agreement, penetration_summary


def read(path):
    return json.loads(path.read_text())


def gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def replay(folder):
    clip = inputs()[1]
    cal = read(P6 / "calibration.json")
    tau = cal["thresholds_N_LR"]
    rows = []
    raw = []
    starts = []
    ends = []
    eps = []
    windows = {}
    for wi, case in enumerate(sorted(folder.glob("reference_f*"))):
        start = int(case.name.split("_f")[1])
        intervals = read(case / "intervals.json")
        samples = read(case / "samples.json")
        contacts = gz(case / "contacts.json.gz")
        non = gz(case / "non_box_contacts.json.gz")
        cstats = {
            str(w): penetration_summary([c for c in contacts if c["env"] == w], len(samples))
            for w in range(4)
        }
        forces = np.array([s["raw_world_N"] for s in samples]).sum(3)
        result = read(case / "result.json")
        windows[case.name] = {
            "result": result,
            "penetration": cstats,
            "peak_force_LR_N": np.linalg.norm(forces, axis=-1).max((0, 1)),
            "max_nonbox_penetration_m": max((-x["dist"] for x in non), default=0),
            "last_Box_position_error_m": np.linalg.norm(
                np.array(intervals[-1]["next_state"])[:, 71:74]
                - clip.arrays["object_pos_w"][start + len(intervals)],
                axis=1,
            )
            if intervals
            else [],
        }
        for k, row in enumerate(intervals):
            rr = np.array(row["force_raw_world"])
            act = actual_contact(rr, tau, [0.25, 0.25])
            for w in range(4):
                raw.append(rr[w])
                starts.append(start + k)
                ends.append(start + k + 1)
                eps.append(4 * wi + w)
                state = np.array(row["next_state"])[w]
                rows.append(
                    {
                        "episode": 4 * wi + w,
                        "frame_start": start + k,
                        "frame_end": start + k + 1,
                        "intent": bool(clip.arrays["object_contact"][start + k + 1]),
                        "hands": act["hands"][w],
                        "occupancy": act["occupancy"][w],
                        "state84": state,
                        "peak_force_N": act["magnitude_N"][w].max(0),
                        "mean_force_world_N": rr[w].sum(2).mean(0),
                    }
                )
    summary = agreement(
        np.array(raw),
        clip.arrays["object_contact"],
        np.array(starts),
        np.array(ends),
        np.array(eps),
        np.ones(len(raw), bool),
        tau,
        [0.25, 0.25],
    )
    summary["windows"] = windows
    summary["distinct_reached_reference_frames"] = len(set(ends))
    return rows, summary


def independent(folder):
    case = folder / "native_PD_contact_lift_carry_release"
    rows = read(case / "intervals.json")
    samples = read(case / "samples.json")
    cs = gz(case / "contacts.json.gz")
    support_samples = {(c["env"], c["sample"] // 4) for c in cs if c["category"] == "support"}
    cal = read(P6 / "calibration.json")
    force = np.array([r["force_raw_world"] for r in rows])
    state = np.array([r["next_state"] for r in rows])
    initial = np.array(read(case / "initial.json")["state84"])
    allworld = []
    for w in range(4):
        act = actual_contact(force[:, w], cal["thresholds_N_LR"], [0.25, 0.25])
        bil = act["bilateral"]
        n = len(rows)
        rise = state[:, w, 73] - initial[w, 73]
        horizontal = np.linalg.norm(state[:, w, 71:73] - initial[w, 71:73], axis=1)
        support = np.array([(w, k) in support_samples for k in range(n)])
        flying = (rise >= 0.03) & ~support
        carrying = flying & (horizontal >= 0.03) & bil

        def longest(x):
            y = np.diff(np.r_[False, x, False].astype(int))
            return float(max(np.flatnonzero(y == -1) - np.flatnonzero(y == 1), default=0) * 0.02)

        tail = state[-25:, w]
        omega = rot(tail[:, 74:78]).apply(tail[:, 81:84])
        cv = tail[:, 78:81] + np.cross(
            omega, rot(tail[:, 74:78]).apply(np.tile([0, 0, 0.125], (len(tail), 1)))
        )
        phases = {}
        for name, a, b in [
            ("contact", 0, 25),
            ("lift", 25, 75),
            ("carry", 75, 125),
            ("lower", 125, 175),
            ("release", 175, 249),
        ]:
            x = act["hands"][a : min(n, b)]
            occ = act["occupancy"][a : min(n, b)]
            phases[name] = {
                "intervals": len(x),
                "hand_contact_ratio_LR": x.mean(0) if len(x) else None,
                "bilateral_ratio": float(x.all(1).mean()) if len(x) else None,
                "mean_occupancy_LR": occ.mean(0) if len(x) else None,
            }
        allworld.append(
            {
                "world": w,
                "max_Box_rise_m": float(rise.max()),
                "max_horizontal_displacement_m": float(horizontal.max()),
                "max_unsupported_lift_duration_s": longest(flying),
                "max_bilateral_unsupported_carry_duration_s": longest(carrying),
                "native_lift_criterion": longest(flying) >= 0.2,
                "native_carry_criterion": longest(carrying) >= 0.2,
                "peak_hand_force_LR_N": act["magnitude_N"].max((0, 1)),
                "tail_max_COM_speed_m_s": float(np.linalg.norm(cv, axis=1).max()),
                "tail_max_angular_speed_rad_s": float(np.linalg.norm(omega, axis=1).max()),
                "final_Box_displacement_m": float(
                    np.linalg.norm(state[-1, w, 71:74] - initial[w, 71:74])
                ),
                "phases": phases,
                "penetration": penetration_summary([c for c in cs if c["env"] == w], len(samples)),
            }
        )
    return {
        "result": read(case / "result.json"),
        "worlds": allworld,
        "external_Box_wrench": False,
        "free_robot_root": True,
        "postreset_Box_teleport": False,
        "criteria": "lift>=.03m unsupported for>=.2s; carry additionally>=.03m lateral and bilateral contact for>=.2s; final tail stability separately reported",
    }


def physical(out):
    summaries = read(E / "physical_02/summary.json")
    results = []
    for name in sorted({x["candidate"] for x in summaries}):
        r = [x for x in summaries if x["candidate"] == name]
        c = candidate(name)
        coms = []
        for case in r:
            path = E / "physical_02" / name / case["case"]["name"] / "trajectory.json"
            t = read(path)
            q = np.array([x["qpos"] for x in t])
            com = q[:, :3] + rot(q[:, 3:7]).apply(
                np.tile(c["inertia"]["com_pos_root_m"], (len(q), 1))
            )
            coms.append(
                {
                    "case": case["case"]["name"],
                    "time_s": [x["time_end_s"] for x in t],
                    "COM_world_m": com,
                }
            )
        dump_gz(out / (name + "_COM_trajectories.json.gz"), coms)
        results.append(
            {
                "candidate": name,
                "cases": len(r),
                "support_cases_stable": all(
                    x["stable_tail"] for x in r if x["case"]["kind"] in ("settle", "drop", "lift")
                ),
                "analytic_max_error": max(
                    max(
                        x["analytical"]["first_velocity_error"],
                        x["analytical"]["first_omega_error"],
                    )
                    for x in r
                    if x["analytical"]
                ),
                "max_support_relative_force_error": max(
                    x["support_balance_relative_error"]
                    for x in r
                    if x["support_balance_relative_error"] is not None
                ),
                "external_lift_rise_m": next(
                    x["max_rise_m"] for x in r if x["case"]["name"] == "support0_lift"
                ),
                "max_penetration_m": max(
                    x["penetration"]["all_box"]["max_penetration_m"] for x in r
                ),
                "max_solver_niter": max(x["solver_iterations_max"] for x in r),
                "warnings": sum(sum(x["warnings"].values()) for x in r),
            }
        )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-suffix", default="01")
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    save(a.output / "physical_comparison.json", physical(a.output))
    for name in ("original", "candidate"):
        rows, summary = replay(E / f"integrated_{name}_{a.run_suffix}")
        dump_gz(a.output / (name + "_rows.json.gz"), rows)
        save(a.output / (name + "_agreement.json"), summary)
    save(a.output / "independent.json", independent(E / f"integrated_independent_{a.run_suffix}"))
