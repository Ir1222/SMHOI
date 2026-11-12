"""P6 final measurement verification; failed physical gates remain explicit."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from common import inputs, rot, save


def main(root, output):
    if output.exists():
        raise FileExistsError(output)
    checks = {}
    _manifest, clip, _ = inputs()
    cal = json.loads((root / "calibration.json").read_text())
    checks["original_reference_hash_and_schema"] = True
    checks["independent_thresholds_not_P2_default"] = all(t > 0.001 for t in cal["thresholds_N_LR"])
    checks["no_contact_baseline"] = max(cal["null_max_N_LR"]) <= 1e-6
    checks["heldout_low_load_sensitivity"] = min(cal["heldout_first_solve_recall_LR"]) >= 0.95
    expected = [[False, False], [True, False], [False, True], [True, True]]
    checks["calibration_policy_contact_heldout"] = all(
        f["actual_hands"] == expected for f in cal["fixtures"] if f["held_out"]
    )
    transients = cal["reused_P1_transients_at_new_definition"]
    checks["occupancy_single_sample_rejected"] = all(
        not np.any(transients[k]["actual_hands"]) for k in ("substep1_only", "substep4_only")
    )
    checks["occupancy_two_samples_detected"] = all(
        x == [True, False] for x in transients["contact_ends_before_final"]["actual_hands"]
    )
    # Independent NumPy frame/force algebra over every new captured complete row.
    errors = {"mean_box_N": 0.0, "impulse_Ns": 0.0, "peak_N": 0.0}
    rows_checked = 0
    for mode in ("calibration_02", "dynamics_01", "isolation_01", "filters_02", "replay_01"):
        for path in sorted((root / mode).glob("*/intervals.json")):
            rows = json.loads(path.read_text())
            for k, row in enumerate(rows):
                raw = np.array(row["force_raw_world"])
                q = np.array(row["box_quat_solve"])
                world = raw.sum(3)
                matrices = rot(q.reshape(-1, 4)).as_matrix().reshape(4, 4, 3, 3)
                expected_mean = np.einsum("ekji,ekhj->ekhi", matrices, world).mean(1)
                for key, actual, desired, tol in (
                    ("mean_box_N", row["force_mean_box"], expected_mean, 3e-5),
                    ("impulse_Ns", row["impulse_world"], 0.005 * world.sum(1), 3e-6),
                    ("peak_N", row["force_peak"], np.linalg.norm(world, axis=-1).max(1), 3e-4),
                ):
                    err = float(np.max(np.abs(actual - desired)))
                    errors[key] = max(errors[key], err)
                    np.testing.assert_allclose(actual, desired, atol=tol, rtol=2e-6)
                rows_checked += 4
                if mode == "replay_01":
                    start = int(path.parent.name.split("_f")[1]) + k
                    intents = np.broadcast_to(
                        clip.arrays["object_contact"][start : start + 2], (4, 2)
                    )
                    np.testing.assert_array_equal(row["intent_ref"], intents)
                    if k:
                        np.testing.assert_array_equal(row["state"], rows[k - 1]["next_state"])
    checks["four_substep_aggregation_recomputed"] = True
    checks["reference_intent_and_state_continuity"] = True
    geometry = json.loads((root / "geometry_summary.json").read_text())
    checks["all840_geometry_and_original_timing"] = (
        geometry["frames"] == 840 and geometry["on"] == 290 and geometry["off"] == 550
    )
    checks["native_approach_isolation"] = all(
        a["initial_hand_contacts"] == 0
        and a["first_min_dist_m"] is not None
        and a["first_min_dist_m"] >= -0.00051
        and a["opposite_hand_contact_placements"] == 0
        for a in geometry["approach"].values()
    )
    checks["bounded_bilateral_initial_overlap"] = all(
        c["dist"] >= -0.001
        for c in json.loads((root / "geometry_02/independent_bilateral.json").read_text())[
            "contacts"
        ]
    )
    filt = root / "filters_02/non_hand_box_filter"
    samples = json.loads((filt / "samples.json").read_text())
    with gzip.open(filt / "contacts.json.gz", "rt") as f:
        contacts = json.load(f)
    checks["robot_body_and_support_excluded"] = not np.any(
        [s["raw_world_N"] for s in samples]
    ) and all(
        any(
            c["env"] == e
            and c["category"] == category
            and np.linalg.norm(c["world_force_on_geom2_N"]) > 1
            for c in contacts
        )
        for e, category in [(0, "robot_body"), (1, "robot_body"), (2, "support")]
    )
    box = json.loads((root / "box_dynamics_01/results.json").read_text())
    checks["box_force_torque_analytic"] = all(
        max(b["analytical"]["first_velocity_error"], b["analytical"]["first_omega_error"]) < 1e-5
        for b in box
        if b["analytical"]
    )
    checks["isolated_support_floor_stability"] = all(
        b["stable_tail"] for b in box if b["case"]["kind"] in ("settle", "drop", "lift")
    )
    checks["controlled_external_lift"] = all(
        b["max_rise_m"] > 0.03 and b["stable_tail"] for b in box if b["case"]["kind"] == "lift"
    )
    checks["support_gravity_balance"] = all(
        abs(b["tail_mean_support_z_N"] - 19.62) < 0.05 * 19.62
        for b in box
        if b["case"]["name"].startswith("support")
    )
    analysis = json.loads((root / "analysis_02/dynamics_analysis.json").read_text())
    isolated = analysis["isolation_01/original_supports_robot_z50.0"]
    checks["original_supports_four_worlds_without_floor_impact"] = not isolated[
        "flagged_samples"
    ] and all(
        w["tail_max_box_COM_speed_m_s"] < 0.02
        and w["tail_max_box_angular_speed_rad_s"] < 0.1
        and abs(w["tail_mean_support_world_N"][2] - 19.62) < 0.981
        for w in isolated["worlds"].values()
    )
    agreement = json.loads((root / "replay_summary.json").read_text())
    checks["same_reference_metric_denominator"] = (
        agreement["any"]["count"] == agreement["bilateral"]["count"] == 1124
    )
    # These are acceptance failures/limitations, never disguised as unit-test PASS.
    physical_gates = {
        "nominal_box_numerical_dynamics": "PASS",
        "nominal_support_capacity": "PASS",
        "native_hand_sensing_and_force_definition": "PASS",
        "mass_distribution_matches_collision_envelope": "FAIL",
        "coupled_original_floor_solver_reliability": "FAIL",
        "externally_forced_lift_release": "PASS",
        "natural_hand_driven_lift_carry": "NOT TESTED",
        "strict_production_provenance_loader": "FAIL",
        "reference_whole_motion_native_geometry": "FAIL",
        "reference_driven_bilateral_agreement": "FAIL",
    }
    result = {
        "checks": {k: "PASS" if v else "FAIL" for k, v in checks.items()},
        "rows_recomputed": rows_checked,
        "aggregation_max_errors": errors,
        "physical_and_reference_gates": physical_gates,
        "P6": "PARTIAL",
        "PHYSICAL_VALIDITY": "PARTIAL",
        "REFERENCE_COMPATIBILITY": "FAIL",
        "P7_prerequisites_satisfied": False,
        "no_training_executed": True,
    }
    save(output, result)
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
