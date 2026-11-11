"""Read-only P3 evidence checks; never recollects, retrains or rewrites prior evidence."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from diagnose_support_context import compare
from p3_common import reference

from supermimic.hoi.asset_manifest import load_manifest
from supermimic.hoi.dataset import load_archive, partition_episodes, replay_from_archive
from supermimic.hoi.reference import DEFAULT_ASSET, REPO, sha256
from supermimic.hoi.validation import legacy_force_readiness_gate

OUT = REPO / "outputs/hoi/contact_supervision/force_model_validation"


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    baseline = read(REPO / "docs/hoi/force_contact/p3/baseline.json")
    changes = {
        name: {"before": digest, "after": sha256(REPO / name)}
        for name, digest in baseline["protected_sha256"].items()
        if sha256(REPO / name) != digest
    }
    expected = {
        "src/supermimic/world_model/world_model.py",
        "src/supermimic/hoi/replay.py",
        "tests/test_hoi_reconciliation.py",
    }
    assert set(changes) == expected, changes
    for name, digest in baseline["dependency_sha256"].items():
        assert sha256(Path(name)) == digest, name
    for name, version in baseline["versions"].items():
        assert importlib.metadata.version(name) == version, name
    clip = reference()  # Also verifies every accepted loader-manifest field hash.
    assert sha256(clip.path) == baseline["reference"]["sha256"]
    for name, digest in baseline["reference"]["manifests"].items():
        assert sha256(clip.path.parent / name) == digest
    data, meta = load_archive(OUT / "collection_02", reference_hashes=[clip.manifest["sha256"]])
    assert len(data["state"]) == 320 and meta["all_substep_capacity_guards_passed"]
    assert not meta["natural_HOI_training_data"]
    try:
        load_archive(OUT / "collection_01", reference_hashes=[clip.manifest["sha256"]])
    except ValueError as exc:
        assert "Failed or incomplete" in str(exc)
    else:
        raise AssertionError("Failed capture was accepted")
    # Reproduce the balanced pair comparison from the saved measurements only.
    support_meta = read(OUT / "support_02/metadata.json")
    assert support_meta["acceptance"] == "PASS"
    # Early support sidecar predates archive hashing. Hash independently matched
    # the immutable archive committed in 7a2faa4; do not rewrite that sidecar.
    support_hash = "c34ec7f9b9635ca28d1f5c7745cf379062e4ee1bbd808ceedc6b37cc5cd8915d"
    assert sha256(OUT / "support_02/transitions.npz") == support_hash
    with np.load(OUT / "support_02/transitions.npz", allow_pickle=False) as a:
        rows = [{k: a[k][i] for k in a.files} for i in range(len(a["state"]))]
    probe = compare(rows, load_manifest(DEFAULT_ASSET))
    saved_probe = read(OUT / "support_balanced_analysis.json")
    for name in probe["comparisons"]:
        for key, value in probe["comparisons"][name].items():
            np.testing.assert_allclose(value, saved_probe["comparisons"][name][key], rtol=1e-10)
    fit = read(OUT / "fit_01/results.json")
    train_keys = set(map(tuple, fit["train_episode_keys"]))
    valid_keys = set(map(tuple, fit["validation_episode_keys"]))
    assert not train_keys & valid_keys
    train_idx, valid_idx = partition_episodes(data, meta, valid_keys)
    assert (len(train_idx), len(valid_idx)) == (240, 80)
    windows = [
        int(replay_from_archive(data, meta, indices).window_mask(8).sum())
        for indices in (train_idx, valid_idx)
    ]
    assert windows == [72, 24]
    assert all(value > 0 for value in fit["parameter_delta_L2"].values())
    assert all(value > 0 for value in fit["last_head_gradient_L2"].values())
    gate = legacy_force_readiness_gate(
        fit["after_validation"],
        coverage=fit["unique_validation_coverage"],
        provenance=fit["provenance"],
        trained_updates=200,
    )
    assert gate == read(OUT / "readiness_review.json") and not gate["ready"]
    with np.load(OUT / "fit_01/predictions.npz", allow_pickle=False) as a:
        assert all(np.isfinite(a[k]).all() for k in a.files)
        assert a["state_prediction"].shape == (8, 24, 84)
        assert a["force_prediction_N"].shape == (8, 24, 2, 3)
        assert not a["target_force_N"][-1].any()  # No active eighth-step claims.
    migration = read(OUT / "migration_01/results.json")
    assert sha256(Path(migration["source_path"])) == migration["source_sha256"]
    checkpoint = OUT / "migration_01/diagnostic_joint_weights.pt"
    assert sha256(checkpoint) == migration["joint_checkpoint_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["metadata"]["readiness"] == gate
    assert payload["metadata"]["actor_force_optimization_enabled"] is False
    assert migration["legacy_robot_max_abs_error"] <= 2e-6
    assert migration["strict_joint_roundtrip_max_abs_error"] == 0
    assert "102 passed, 17 skipped" in (OUT / "tests_full_02.log").read_text()
    return {
        "verification": "PASS",
        "stage_acceptance": "PARTIAL",
        "protected_files_checked": len(baseline["protected_sha256"]),
        "protected_files_unchanged": len(baseline["protected_sha256"]) - len(changes),
        "intentional_existing_file_changes": changes,
        "dependency_source_hashes_verified": len(baseline["dependency_sha256"]),
        "versions_verified": baseline["versions"],
        "canonical_reference": baseline["reference"],
        "measured_archive_sha256": meta["archive_sha256"],
        "train_validation_rows": [240, 80],
        "train_validation_windows": windows,
        "support_comparison_reproduced": True,
        "support_archive_sha256": support_hash,
        "force_labels_recomputed_from_saved_substeps": True,
        "failed_capture_rejected": True,
        "learned_parameter_updates_verified": True,
        "readiness": gate,
        "real_checkpoint_source_unchanged": True,
        "regression": {"passed": 102, "skipped": 17, "failed": 0},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
