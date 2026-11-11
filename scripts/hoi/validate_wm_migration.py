"""Read-only legacy WM migration probe plus diagnostic HOI checkpoint round trip."""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path, PosixPath

import torch
from p3_common import reference

from supermimic.hoi.checkpoint import (
    joint_checkpoint,
    load_joint_weights,
    migrate_robot_state_dict,
    robot_config,
)
from supermimic.hoi.dataset import load_archive
from supermimic.hoi.force import ForceStatistics
from supermimic.hoi.reference import sha256
from supermimic.hoi.state_spec import ObjectSpecCfg
from supermimic.hoi.world_model import JointWorldModel, JointWorldModelCfg, LossWeights
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel

EXPECTED_SOURCE = "16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7"


def run(source, data_dir, fit_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source_hash = sha256(source)
    if source_hash != EXPECTED_SOURCE:
        raise ValueError("Explicit external P0 candidate hash mismatch")
    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    old_cfg = robot_config(checkpoint["config"]["world_model"])
    old = WorldModel(old_cfg, AnalyticHeads, "cpu")
    old.load_state_dict(checkpoint["wm"], strict=True)
    clip = reference()
    data, meta = load_archive(data_dir, reference_hashes=[clip.manifest["sha256"]])
    stats = ForceStatistics.fit(
        torch.tensor(data["force_mean_box"]),
        torch.tensor(data["occupancy"]),
        force_on_N=0.001,
        active_mass=0.5,
    )
    new_cfg = JointWorldModelCfg(
        **{k: v for k, v in asdict(old_cfg).items() if k != "state_spec"},
        state_spec=copy.deepcopy(old_cfg.state_spec),
        weights=LossWeights(1.0, 1.0, 1.0),
    )
    new = JointWorldModel(new_cfg, AnalyticHeads, stats, "cpu")
    mapping = migrate_robot_state_dict(checkpoint["wm"], old_cfg, new)
    s, a, xi = [torch.tensor(data[k][:32]) for k in ("state", "action_exec", "xi")]
    with torch.no_grad():
        old_next = old.step(s[:, :71], a, xi)
        new_next, force = new.predict(s, a, xi)
    error = float((old_next - new_next[:, :71]).abs().max())
    torch.testing.assert_close(old_next, new_next[:, :71], atol=2e-6, rtol=2e-6)
    assert force.count_nonzero() == 0
    # Package the already-trained small diagnostic, no additional learning.
    fitted = torch.load(fit_dir / "diagnostic_weights.pt", map_location="cpu", weights_only=True)
    cfg_map = dict(fitted["config"])
    robot_keys = set(asdict(old_cfg))
    fitted_robot = robot_config({k: cfg_map[k] for k in robot_keys})
    fitted_cfg = JointWorldModelCfg(
        **{k: v for k, v in asdict(fitted_robot).items() if k != "state_spec"},
        state_spec=fitted_robot.state_spec,
        object_spec=ObjectSpecCfg(**cfg_map["object_spec"]),
        weights=LossWeights(**cfg_map["weights"]),
        force_loss=cfg_map["force_loss"],
        force_delta=cfg_map["force_delta"],
        support_context=cfg_map["support_context"],
    )
    sd = fitted["wm"]
    fitted_stats = ForceStatistics(
        sd["force_statistics.scale_N"],
        sd["force_statistics.counts_inactive_active"],
        sd["force_statistics.median_active_N"],
        force_on_N=0.001,
        active_mass=fitted["force_active_mass"],
    )
    learned = JointWorldModel(fitted_cfg, AnalyticHeads, fitted_stats, "cpu")
    learned.load_state_dict(sd, strict=True)
    readiness = json.loads((fit_dir.parent / "readiness_review.json").read_text())
    payload = joint_checkpoint(
        learned,
        replay_metadata=meta["replay"],
        readiness=readiness,
        provenance={
            "kind": "controlled_P3_diagnostic",
            "updates": 200,
            "source_archive_sha256": meta["archive_sha256"],
        },
    )
    path = output / "diagnostic_joint_weights.pt"
    torch.save(payload, path)
    restored = JointWorldModel(
        copy.deepcopy(fitted_cfg), AnalyticHeads, copy.deepcopy(fitted_stats), "cpu"
    )
    load_joint_weights(
        restored, torch.load(path, weights_only=True), expected_replay_metadata=meta["replay"]
    )
    with torch.no_grad():
        prediction = learned.predict(s, a, xi)
        restored_prediction = restored.predict(s, a, xi)
    roundtrip = max(float((a - b).abs().max()) for a, b in zip(prediction, restored_prediction))
    assert roundtrip == 0 and sha256(source) == source_hash
    result = {
        "acceptance": "PASS",
        "source_path": str(source.resolve()),
        "source_sha256": source_hash,
        "source_provenance": "External P0 candidate; no embedded training Git/data SHA. Target-local candidate remains missing. Architecture compatibility only, not selected HOI policy.",
        "legacy_hidden": list(old_cfg.hidden),
        "legacy_robot_max_abs_error": error,
        "new_force_max_abs_N": float(force.abs().max()),
        "mapping": mapping,
        "strict_joint_roundtrip_max_abs_error": roundtrip,
        "joint_checkpoint_sha256": sha256(path),
        "source_unchanged": True,
        "actor_optimizer_migration": False,
        "joint_checkpoint_metadata": payload["metadata"],
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("mapping", "joint_checkpoint_metadata")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("source", "data", "fit", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.data, args.fit, args.output)
