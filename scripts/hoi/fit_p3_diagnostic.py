"""One bounded contact-intent diagnostic via the original WorldModelTrainer.update."""

import argparse
import copy
import json
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from p3_common import reference

from supermimic.hoi.checkpoint import contact_config, joint_checkpoint, load_joint_weights
from supermimic.hoi.dataset import (
    episode_keys,
    load_archive,
    partition_episodes,
    replay_from_archive,
)
from supermimic.hoi.reference import sha256
from supermimic.hoi.validation import evaluate_windows, readiness_gate, unique_coverage
from supermimic.hoi.wm_trainer import JointWorldModelTrainer
from supermimic.hoi.world_model import JointWorldModel
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer


def diagnostic_partitions(data, metadata):
    """Accept only the approved four whole source episodes, never the all-ON archive."""
    if (
        metadata.get("kind") != "controlled_P3_contact_boundary_diagnostic"
        or len(data["state"]) != 32
    ):
        raise ValueError("Fitting requires the exact approved 32-transition contact diagnostic")
    episodes = metadata.get("episodes", [])
    expected = [
        ("train", "onset", 256),
        ("train", "release", 546),
        ("validation", "onset", 255),
        ("validation", "release", 545),
    ]
    if [(x["split"], x["regime"], x["initial_frame"]) for x in episodes] != expected or any(
        x["rows"] != 8 for x in episodes
    ):
        raise ValueError("Incomplete or altered boundary schedule")
    digest = metadata["replay"]["references"][0]["sha256"]
    keys = episode_keys(data, metadata)
    declared = {(digest, x["env"], x["episode"]) for x in episodes}
    if len(declared) != 4 or set(keys) != declared:
        raise ValueError("Boundary source episode identities do not match declared schedule")
    for episode in episodes:
        selected = np.array([k == (digest, episode["env"], episode["episode"]) for k in keys])
        if not np.array_equal(
            np.sort(data["frame"][selected]),
            np.arange(episode["initial_frame"], episode["initial_frame"] + 8),
        ):
            raise ValueError("Recorded episode frames differ from approved schedule")
    valid_keys = {(digest, x["env"], x["episode"]) for x in episodes if x["split"] == "validation"}
    train, valid = partition_episodes(data, metadata, valid_keys)
    for indices in (train, valid):
        coverage = unique_coverage(data, indices, metadata)
        if [coverage[f"pair_{p}"] for p in ("00", "01", "11", "10")] != [7, 1, 7, 1]:
            raise ValueError("Missing balanced classes or contact boundaries")
    return train, valid


def run(source, initial, output):
    clip = reference()
    data, metadata = load_archive(
        source, reference_hashes=[clip.manifest["sha256"]], references=[clip]
    )
    train_idx, valid_idx = diagnostic_partitions(data, metadata)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(404)
    result = {
        "scope": "bounded single-reference supervised diagnostic; no cross-clip generalization",
        "seed": 404,
        "source": str(source.resolve()),
        "source_archive_sha256": metadata["archive_sha256"],
        "initial_checkpoint": str(initial.resolve()),
        "initial_checkpoint_sha256": sha256(initial),
        "completed_updates": 0,
        "updates": [],
        "acceptance": "PARTIAL",
    }
    try:
        train = replay_from_archive(data, metadata, train_idx)
        valid = replay_from_archive(data, metadata, valid_idx)
        initial_payload = torch.load(initial, map_location="cpu", weights_only=True)
        cfg = contact_config(initial_payload["metadata"]["config"])
        if tuple(cfg.hidden) != (64, 64, 64) or cfg.weights.contact != 1.0:
            raise ValueError("Bounded fit requires approved hidden64x3 and contact coefficient1")
        wm = JointWorldModel(cfg, AnalyticHeads, "cpu")
        load_joint_weights(wm, initial_payload, expected_replay_metadata=metadata["replay"])
        trainer = JointWorldModelTrainer(
            wm,
            [clip],
            WMTrainerCfg(batch_size=32, batches_per_update=1, window=8, lr=1e-3, compile=None),
        )
        assert JointWorldModelTrainer.update is WorldModelTrainer.update
        # Validate every measured target, not only randomly sampled training windows.
        for buffer in (train, valid):
            e, t = buffer.window_mask(1).nonzero(as_tuple=True)
            trainer.predictive_states(buffer._gather_windows(e, t, 1))
        before_train, _ = evaluate_windows(wm, train)
        before_valid, _ = evaluate_windows(wm, valid)
        cv, _ = evaluate_windows(wm, valid, constant_velocity=True)
        parameters = {k: v.detach().clone() for k, v in wm.named_parameters()}
        keys = episode_keys(data, metadata)
        train_keys, valid_keys = {keys[i] for i in train_idx}, {keys[i] for i in valid_idx}
        provenance = {
            "kind": metadata["kind"],
            "episode_disjoint": not (train_keys & valid_keys),
            "reference_verified": True,
            "simulator_labels_verified": metadata["all_substep_capacity_guards_passed"],
            "reference_sha256": clip.manifest["sha256"],
            "asset_sha256": metadata["replay"]["asset_manifest_sha256"],
            "source_archive_sha256": metadata["archive_sha256"],
            "cross_clip_generalization": False,
            "shared_reference_boundary_frames": True,
        }
        result.update(
            config=asdict(cfg),
            trainer_config=asdict(trainer.cfg),
            trainer_update="supermimic.world_model.wm_trainer.WorldModelTrainer.update (inherited unchanged)",
            train_rows=len(train_idx),
            validation_rows=len(valid_idx),
            train_episode_keys=sorted(train_keys),
            validation_episode_keys=sorted(valid_keys),
            train_coverage=unique_coverage(data, train_idx, metadata),
            validation_coverage=unique_coverage(data, valid_idx, metadata),
            before_train=before_train,
            frozen_validation=before_valid,
            constant_velocity_validation=cv,
            provenance=provenance,
            loss_weight_selection="Physical/termination coefficients preserved from migration; contact=1; no calibration or validation tuning",
        )
        for update in range(200):
            metrics = trainer.update(train)
            if not all(np.isfinite(v) for v in metrics.values()):
                raise ValueError("Nonfinite optimizer update")
            result["completed_updates"] = update + 1
            if update in (0, 1, 19, 49, 99, 199):
                result["updates"].append({"update": update + 1, **metrics})
                print(
                    f"Update {update + 1}/200: loss={metrics['loss']:.6g}, contact_BCE={metrics['contact_bce']:.6g}",
                    flush=True,
                )
        after_train, _ = evaluate_windows(wm, train)
        after_valid, predictions = evaluate_windows(wm, valid)
        final = f"dynamics.mlp.net.{2 * len(cfg.hidden)}"
        delta = {k: p.detach() - parameters[k] for k, p in wm.named_parameters()}
        changes = {
            f"{name}_head": sum(
                float(delta[f"{final}.{kind}"][sl].square().sum()) for kind in ("weight", "bias")
            )
            ** 0.5
            for name, sl in [
                ("robot", slice(0, 70)),
                ("object", slice(70, 82)),
                ("contact", slice(82, 83)),
            ]
        }
        changes["trunk"] = (
            sum(
                float(v.square().sum())
                for k, v in delta.items()
                if k.startswith("dynamics.") and not k.startswith(final + ".")
            )
            ** 0.5
        )
        gradients = {
            name: float(dict(wm.named_parameters())[final + ".weight"].grad[sl].norm())
            for name, sl in [
                ("robot", slice(0, 70)),
                ("object", slice(70, 82)),
                ("contact", slice(82, 83)),
            ]
        }
        readiness = readiness_gate(
            after_valid,
            coverage=result["validation_coverage"],
            provenance=provenance,
            trained_updates=200,
            baseline=before_valid,
            parameter_updates=changes,
        )
        payload = joint_checkpoint(
            wm,
            replay_metadata=metadata["replay"],
            readiness=readiness,
            provenance={
                **provenance,
                "updates": 200,
                "initial_checkpoint_sha256": result["initial_checkpoint_sha256"],
            },
        )
        checkpoint = output / "diagnostic_contact_weights.pt"
        torch.save(payload, checkpoint)
        restored = JointWorldModel(copy.deepcopy(cfg), AnalyticHeads, "cpu")
        load_joint_weights(
            restored,
            torch.load(checkpoint, weights_only=True),
            expected_replay_metadata=metadata["replay"],
        )
        for name, value in wm.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
        np.savez_compressed(
            output / "predictions.npz", **{k: v.numpy() for k, v in predictions.items()}
        )
        result.update(
            acceptance=readiness["overall"],
            after_train=after_train,
            after_validation=after_valid,
            parameter_delta_L2=changes,
            last_head_gradient_L2=gradients,
            readiness=readiness,
            checkpoint_sha256=sha256(checkpoint),
            strict_checkpoint_roundtrip_max_abs=0.0,
        )
        assert sha256(initial) == result["initial_checkpoint_sha256"]
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["readiness"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("source", "initial", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.initial, args.output)
