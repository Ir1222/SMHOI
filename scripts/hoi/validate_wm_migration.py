"""Read-only named checkpoint migration and physical one/eight-step parity probes."""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path, PosixPath

import torch
from p3_common import reference

from supermimic.hoi.checkpoint import (
    contact_config,
    joint_checkpoint,
    legacy_joint_step,
    load_joint_weights,
    migrate_joint_weights,
    migrate_robot_state_dict,
    robot_config,
)
from supermimic.hoi.dataset import load_archive, replay_from_archive
from supermimic.hoi.reference import REPO, sha256
from supermimic.hoi.world_model import JointWorldModel, JointWorldModelCfg, LossWeights
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel

OLD = REPO / "outputs/hoi/contact_supervision/force_model_validation"
ROBOT_SHA256 = "16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7"


def run(source, data_dir, output, robot_source=None):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(402)
    source_hash = sha256(source)
    old = torch.load(source, map_location="cpu", weights_only=True)
    cfg = contact_config(old["metadata"]["config"], legacy=True)
    new = JointWorldModel(cfg, AnalyticHeads, "cpu")
    clip = reference()
    data, meta = load_archive(
        data_dir, reference_hashes=[clip.manifest["sha256"]], references=[clip]
    )
    mapping = migrate_joint_weights(old, new, expected_replay_metadata=meta["replay"])
    replay = replay_from_archive(data, meta, list(range(len(data["state"]))))
    e, t = replay.window_mask(8).nonzero(as_tuple=True)
    if not len(e):
        raise ValueError("No complete measured windows for parity probe")
    batch = replay._gather_windows(e[:16], t[:16], 8)
    errors = {}
    with torch.no_grad():
        for contact in (0.0, 1.0, 0.37):
            physical = batch["states"][0].clone()
            z = torch.cat((physical, torch.full_like(physical[:, :1], contact)), -1)
            per_step = []
            for action in batch["a_exec"]:
                physical = legacy_joint_step(old, new, physical, action, batch["xi"])
                z, logit = new.predict(z, action, batch["xi"])
                torch.testing.assert_close(physical, z[:, :84], atol=2e-6, rtol=2e-6)
                assert logit.count_nonzero() == 0 and (z[:, 84] == 0.5).all()
                per_step.append(float((physical - z[:, :84]).abs().max()))
            errors[str(contact)] = per_step
    payload = joint_checkpoint(
        new,
        replay_metadata=meta["replay"],
        readiness={"overall": "PARTIAL", "reason": "Migration only; contact untrained"},
        provenance={
            "source_path": str(source.resolve()),
            "source_sha256": source_hash,
            "migration": mapping,
            "updates": 0,
        },
    )
    path = output / "migrated_contact_weights.pt"
    torch.save(payload, path)
    restored = JointWorldModel(copy.deepcopy(cfg), AnalyticHeads, "cpu")
    load_joint_weights(
        restored, torch.load(path, weights_only=True), expected_replay_metadata=meta["replay"]
    )
    for key, tensor in new.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[key], atol=0, rtol=0)
    with torch.no_grad():
        predicted = new.predict(z, batch["a_exec"][0], batch["xi"])
        recovered = restored.predict(z, batch["a_exec"][0], batch["xi"])
    roundtrip = max(float((a - b).abs().max()) for a, b in zip(predicted, recovered))
    assert roundtrip == 0
    robot_probe = {"status": "NOT_REQUESTED"}
    if robot_source is not None:
        robot_hash = sha256(robot_source)
        if robot_hash != ROBOT_SHA256:
            raise ValueError("External P0 robot candidate hash mismatch")
        with torch.serialization.safe_globals([PosixPath]):
            robot_checkpoint = torch.load(robot_source, map_location="cpu", weights_only=True)
        robot_cfg = robot_config(robot_checkpoint["config"]["world_model"])
        original = WorldModel(robot_cfg, AnalyticHeads, "cpu")
        original.load_state_dict(robot_checkpoint["wm"], strict=True)
        joint_cfg = JointWorldModelCfg(
            **{k: v for k, v in asdict(robot_cfg).items() if k != "state_spec"},
            state_spec=copy.deepcopy(robot_cfg.state_spec),
            weights=LossWeights(1, 1, 1),
        )
        initialized = JointWorldModel(joint_cfg, AnalyticHeads, "cpu")
        robot_mapping = migrate_robot_state_dict(robot_checkpoint["wm"], robot_cfg, initialized)
        physical = batch["states"][0].clone()
        z = torch.cat((physical, torch.full_like(physical[:, :1], 0.37)), -1)
        robot = physical[:, :71].clone()
        robot_errors = []
        fp32_within_tolerance = []
        with torch.no_grad():
            for action in batch["a_exec"]:
                robot = original.step(robot, action, batch["xi"])
                z = initialized.step(z, action, batch["xi"])
                robot_errors.append(float((robot - z[:, :71]).abs().max()))
                fp32_within_tolerance.append(
                    bool(torch.allclose(robot, z[:, :71], atol=2e-6, rtol=2e-6))
                )
        # Diagnose numerical accumulation separately; never relabel a float32 failure.
        original.double()
        initialized.double()
        physical = batch["states"][0].double()
        z = torch.cat((physical, torch.full_like(physical[:, :1], 0.37)), -1)
        robot = physical[:, :71].clone()
        fp64_errors = []
        with torch.no_grad():
            for action in batch["a_exec"].double():
                robot = original.step(robot, action, batch["xi"].double())
                z = initialized.step(z, action, batch["xi"].double())
                torch.testing.assert_close(robot, z[:, :71], atol=2e-6, rtol=2e-6)
                fp64_errors.append(float((robot - z[:, :71]).abs().max()))
        assert sha256(robot_source) == robot_hash
        robot_probe = {
            "status": "PASS" if all(fp32_within_tolerance) else "PARTIAL",
            "source": str(robot_source),
            "source_sha256": robot_hash,
            "per_horizon_max_abs": robot_errors,
            "float32_per_horizon_within_approved_tolerance": fp32_within_tolerance,
            "float64_per_horizon_max_abs": fp64_errors,
            "float64_parity": "PASS",
            "mapping": robot_mapping,
            "provenance": "Qualified external P0 candidate; architecture compatibility only",
        }
    assert sha256(source) == source_hash
    result = {
        "implementation_acceptance": "PASS",
        "source_sha256": source_hash,
        "joint_per_horizon_max_abs": errors,
        "named_mapping": mapping,
        "strict_roundtrip_max_abs": roundtrip,
        "new_contact_probability": 0.5,
        "checkpoint_sha256": sha256(path),
        "original_robot_probe": robot_probe,
        "source_unchanged": True,
        "trained_model_claim": False,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("named_mapping", "original_robot_probe")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=OLD / "migration_01/diagnostic_joint_weights.pt"
    )
    parser.add_argument("--data", type=Path, default=OLD / "collection_02")
    parser.add_argument("--robot-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.data, args.output, args.robot_source)
