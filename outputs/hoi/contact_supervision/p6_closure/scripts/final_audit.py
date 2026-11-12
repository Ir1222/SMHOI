"""Recompute closure invariants; report physical failures without weakening gates."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from closure_common import E, P6, ROOT, save
from contact_parameters import decode
from supermimic.hoi.contact_validation import actual_contact
import numpy as np
from scipy.spatial.transform import Rotation


def read(p):
    return json.loads(p.read_text())


def gz(p):
    with gzip.open(p, "rt") as f:
        return json.load(f)


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = E / "final_audit"
    out.mkdir(exist_ok=True)
    checks = {}
    baseline = read(E / "baseline.json")
    changed = []
    missing = []
    for row in baseline["files"]:
        p = Path(row["path"])
        current = digest(p) if p.is_file() else None
        if current != row.get("sha256"):
            changed.append({**row, "current": current})
        if current is None:
            missing.append(row)
    checks["preservation"] = {
        "files": len(baseline["files"]),
        "changed": changed,
        "preexisting_missing": missing,
    }
    assert not changed, changed
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == baseline["head"]
    )
    assert not subprocess.check_output(["git", "diff", "--name-only"], text=True).strip()
    assert not subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], text=True
    ).strip()
    orig = np.load(ROOT / "outputs/hoi/asset_audit/part_b/reference/reference.npz")
    cand = np.load(E / "reference_candidate_v1/reference.npz")
    cm = read(E / "reference_candidate_v1/reference_manifest.json")
    assert digest(E / "reference_candidate_v1/reference.npz") == cm["sha256"]
    for k in [
        "object_pos_w",
        "object_quat_w",
        "object_contact",
        "joint_names",
        "body_names",
        "fps",
    ]:
        np.testing.assert_array_equal(orig[k], cand[k])
    for k in ["joint_pos", "joint_vel"]:
        ix = [i for i, n in enumerate(orig["joint_names"]) if n not in cm["changed_joints"]]
        np.testing.assert_array_equal(orig[k][:, ix], cand[k][:, ix])
    for k in cm["fields"]:
        if k.startswith("body_") and k != "body_names":
            ix = [i for i, n in enumerate(orig["body_names"]) if n not in cm["changed_body_arrays"]]
            np.testing.assert_array_equal(orig[k][:, ix], cand[k][:, ix])
    checks["reference_preserved_fields"] = "PASS; only explicit arm candidate fields changed"
    geometry = read(E / "geometry_02/summary.json")
    assert len(geometry) == 20 and all(
        x["frames"] == 840 and x["ON_frames"] == 290 for x in geometry
    )
    checks["geometry"] = {"candidates": 20, "frames_each": 840, "ON_each": 290}
    physical = read(E / "analysis_02/physical_comparison.json")
    assert len(physical) == 8 and sum(x["cases"] for x in physical) == 104
    assert all(
        x["support_cases_stable"] and x["warnings"] == 0 and x["analytic_max_error"] < 1e-5
        for x in physical
    )
    checks["physical_cases"] = physical
    for name in ["ik_single_02", "ik_shell_02"]:
        s = read(E / name / "summary.json")
        assert s["ON_feasible"] == 290 and s["continuous_all_ON"] and s["optimizer_success"]
        assert max(x["relative_error"] for x in read(E / name / "jacobian_check.json")) < 1e-7
    checks["coupled_IK"] = "PASS both candidates290/290; kinematic only"
    for folder in ["solver_guard_final", "solver_candidate_guard_final"]:
        rs = [read(p) for p in (E / folder).glob("*/result.json")]
        assert len(rs) == 6 and all(r["status"] == "PASS" and r["samples"] == 600 for r in rs)
    paired = read(E / "failure_replay_02/results.json")
    assert all(r["acceptance_next_qpos_lt_1e_4_qvel_lt_1e_3"] for r in paired)
    assert all(
        not any(r["overflow"]) if r["precision_guard"] else any(r["overflow"]) for r in paired
    )
    checks["solver_precision"] = "PASS exact paired states and12 bounded3s scenes; opt-in only"
    calibration = read(E / "integrated_calibration_02/results.json")
    assert len(calibration) == 9 and all(r["definition_check"] for r in calibration.values())
    tau = read(P6 / "calibration.json")["thresholds_N_LR"]
    checks["calibration"] = {
        "threshold_N_LR": tau,
        "occupancy_strictly_above": 0.25,
        "fixture_checks": 9,
    }
    aggregate = []
    for profile in ["original", "candidate"]:
        a = read(E / f"analysis_02/{profile}_agreement.json")
        assert (
            a["valid_rows"] == 2696
            and not a["invalid_rows"]
            and a["distinct_reached_reference_frames"] == 599
        )
        aggregate.append({k: v for k, v in a.items() if k != "windows"})
    ar = gz(E / "analysis_02/original_rows.json.gz")
    br = gz(E / "analysis_02/candidate_rows.json.gz")
    assert [(r["episode"], r["frame_start"], r["frame_end"], r["intent"]) for r in ar] == [
        (r["episode"], r["frame_start"], r["frame_end"], r["intent"]) for r in br
    ]
    checks["reference_agreement"] = aggregate
    replay_geometry = read(E / "replay_geometry/summary.json")
    assert all(x["rows"] == 2696 for x in replay_geometry.values())
    checks["actual_replay_geometry"] = replay_geometry
    # Independently recompute P1/P2 aggregation on all final C4 intervals.
    intervals = 0
    max_errors = {k: 0.0 for k in ("mean_box", "peak", "impulse", "occupancy")}
    for folder in [
        "integrated_original_02",
        "integrated_candidate_02",
        "integrated_independent_02",
        "integrated_calibration_02",
    ]:
        for path in (E / folder).glob("*/intervals.json"):
            for r in read(path):
                raw = np.array(r["force_raw_world"])
                net = raw.sum(3)
                q = np.array(r["box_quat_solve"])
                box = np.stack(
                    [
                        Rotation.from_quat(q.reshape(-1, 4), scalar_first=True)
                        .inv()
                        .apply(net[:, :, h].reshape(-1, 3))
                        .reshape(4, 4, 3)
                        for h in range(2)
                    ],
                    axis=2,
                )
                expected = {
                    "mean_box": box.mean(1),
                    "peak": np.linalg.norm(net, axis=-1).max(1),
                    "impulse": net.sum(1) * 0.005,
                    "occupancy": (np.linalg.norm(net, axis=-1) > 0.001).mean(1),
                }
                fields = {
                    "mean_box": "force_mean_box",
                    "peak": "force_peak",
                    "impulse": "impulse_world",
                    "occupancy": "occupancy",
                }
                for key, field in fields.items():
                    error = float(np.max(np.abs(np.array(r[field]) - expected[key])))
                    max_errors[key] = max(max_errors[key], error)
                    np.testing.assert_allclose(r[field], expected[key], rtol=2e-5, atol=2e-5)
                if "transition_valid" in r:
                    assert all(r["transition_valid"])
                intervals += len(raw)
    checks["aggregation"] = {
        "world_intervals": intervals,
        "max_abs_errors": max_errors,
        "stored_occupancy_uses_P2_diagnostic_threshold": 0.001,
        "actual_binary_uses_P6_calibration": tau,
    }
    allparams = read(E / "contact_physical_01/summary.json") + read(
        E / "contact_physical_gap_01/summary.json"
    )
    assert len(allparams) == 40 and not any(any(r["warnings"]) for r in allparams)
    checks["material_CPU_cases"] = allparams
    audit = read(E / "contact_audit_04/compiled_audit.json")
    groups = audit["candidate_parameter_groups"]
    assert groups["friction_only"]["hand_probe"]["contacts"][0]["active_torsion"] is False
    assert groups["condim4_only"]["hand_probe"]["contacts"][0]["active_torsion"] is True
    assert groups["gap_only"]["positive_gap_probe"]["contacts"][0]["efc_address"] == -1
    assert groups["gap_after_margin"]["distance3mm_probe"]["contacts"][0]["efc_address"] == -1
    checks["compiled_contact_semantics"] = (
        "PASS actual dimensions/defaults and current-engine margin/gap"
    )
    toaster = audit["Toaster_configured"]["geoms"]
    assert toaster[2]["type"] == 7 and toaster[2]["contype"] == 1
    assert toaster[27]["type"] == 7 and toaster[27]["contype"] == 0
    pairs = read(E / "toaster_pair_audit.json")
    assert len(pairs["generated_native_capsule_contacts"]) == 4
    assert all(c["dim"] == 3 for c in pairs["generated_native_capsule_contacts"])
    checks["upstream_Toaster_edit_order_and_pairs"] = (
        "PASS fresh spec edited before compile;4 native-shaped capsule contacts"
    )
    import mjlab.utils.string as string_utils

    assert (
        Path(string_utils.__file__).read_text().strip()
        == (E / "upstream/mjlab_v1_4_string.py").read_text().strip()
    )
    checks["upstream_editor_regex_helpers"] = "Installed helpers equal pinned1.4 source"
    failures = read(E / "contact_gpu_failure_details/results.json")
    assert len(failures) == 8
    assert all(
        r["floor"]["status"] == "FAIL"
        and set(r["floor"]["final_overflow"]) <= {0, 1024}
        and any(r["floor"]["final_overflow"])
        for r in failures
    )
    checks["material_floor_failures"] = [
        {
            "group": r["group"],
            "samples": r["floor"]["samples"],
            "final_overflow": [decode(x) for x in r["floor"]["final_overflow"]],
        }
        for r in failures
    ]
    native = []
    for run in ["contact_gpu_01", "contact_gpu_02"]:
        for row in read(E / run / "results.json"):
            if "native_hold" not in row or row["native_hold"]["status"] != "PASS":
                continue
            case = E / run / row["group"] / "native_hold"
            rr = read(case / "intervals.json")
            raw = np.array([r["force_raw_world"] for r in rr])
            act = actual_contact(raw.reshape(-1, 4, 2, 2, 3), tau, [0.25, 0.25])
            cs = gz(case / "contacts.json.gz")
            native.append(
                {
                    "group": row["group"],
                    "intervals": len(rr) * 4,
                    "contact_ratio_LR": act["hands"].mean(0),
                    "mean_occupancy_LR": act["occupancy"].mean(0),
                    "peak_force_LR_N": act["magnitude_N"].max((0, 1)),
                    "max_hand_penetration_m": max(
                        (-c["dist"] for c in cs if c["category"] in ("left", "right")), default=0
                    ),
                }
            )
    assert len(native) == 8
    checks["material_native_hold"] = native
    overflow = read(E / "overflow_audit_01/summary.json")
    assert all(
        any(x["LS_ITERATIONS_bit_set"] for x in r["actual_overflow"])
        and all(set(x["active"]) <= {"LS_ITERATIONS"} for x in r["actual_overflow"])
        for r in overflow
    )
    checks["original_overflow_replay"] = overflow
    independent = read(E / "analysis_02/independent.json")
    assert not any(
        w["native_lift_criterion"] or w["native_carry_criterion"] for w in independent["worlds"]
    )
    checks["natural_manipulation"] = "FAIL all4worlds; retained physical gate"
    suite = ET.parse(E / "tests_05.xml").getroot()[0]
    assert (
        suite.attrib["tests"] == "16" and suite.attrib["failures"] == suite.attrib["errors"] == "0"
    )
    checks["unit_regressions"] = "16 passed"
    checks["decisions"] = {
        "OBJECT_CONSTRUCTION_VALIDITY": "PASS bounded simulation approximation",
        "BOX_MODEL_VALIDITY": "PASS selected approximation; original COM concern remains",
        "FLOOR_SOLVER_RELIABILITY": "PARTIAL; bounded precision guard passes, default remains failing",
        "REFERENCE_COMPATIBILITY": "PARTIAL; original fails, candidate kinematic pass/dynamic fail",
        "BILATERAL_CONTACT_FEASIBILITY": "PARTIAL; geometric pass, sustained dynamic fail",
        "PHYSICAL_VALIDITY": "PARTIAL",
        "P7_TRAINING_READINESS": "FAIL / NOT SATISFIED",
    }
    save(out / "verification.json", checks)
    print("Verification PASS; physical failures remain:", checks["decisions"])


if __name__ == "__main__":
    main()
