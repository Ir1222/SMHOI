"""Bounded diagnostic fit via original WorldModelTrainer.update, never an Actor run."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from p3_common import reference

from supermimic.hoi.dataset import (
    episode_keys,
    load_archive,
    partition_episodes,
    replay_from_archive,
)
from supermimic.hoi.force import ForceStatistics
from supermimic.hoi.validation import evaluate_windows, readiness_gate, unique_coverage
from supermimic.hoi.wm_trainer import JointWorldModelTrainer
from supermimic.hoi.world_model import JointWorldModel, JointWorldModelCfg, LossWeights
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WMTrainerCfg


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(304)
    clip = reference()
    data, metadata = load_archive(source, reference_hashes=[clip.manifest["sha256"]])
    keys = episode_keys(data, metadata)
    valid_keys = {k for k in keys if k[2] >= 6}
    train_idx, valid_idx = partition_episodes(data, metadata, valid_keys)
    train = replay_from_archive(data, metadata, train_idx)
    valid = replay_from_archive(data, metadata, valid_idx)
    e, t = train.window_mask(8).nonzero(as_tuple=True)
    full_train = train._gather_windows(e, t, 8)
    stats = ForceStatistics.fit(
        full_train["force_mean_box"],
        full_train["occupancy"],
        force_on_N=metadata["replay"]["force_on_N"],
        active_mass=0.5,
    )
    cfg = JointWorldModelCfg(hidden=(64, 64, 64), weights=LossWeights(1.0, 1.0, 1.0))
    wm = JointWorldModel(cfg, AnalyticHeads, stats, "cpu")
    trainer = JointWorldModelTrainer(
        wm, [clip], WMTrainerCfg(batch_size=32, batches_per_update=1, lr=1e-3, compile=None)
    )
    with torch.no_grad():
        _, scales = trainer.loss_on(full_train)
    components = [
        float(scales[k])
        for k in ("robot_weighted_loss", "object_weighted_loss", "force_weighted_loss", "term_bce")
    ]
    if any(not np.isfinite(v) or v <= 0 for v in components):
        raise ValueError("Cannot calibrate weights from absent/nonfinite component scale")
    cfg.weights = LossWeights(*(1 / v for v in components[:3]))
    cfg.term_loss_weight = 1 / components[3]
    before_train, _ = evaluate_windows(wm, train)
    before_valid, _ = evaluate_windows(wm, valid)
    initial = {k: v.detach().clone() for k, v in wm.named_parameters()}
    log = []
    for update in range(200):
        metrics = trainer.update(train)  # sole original optimizer/allreduce/clip implementation
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite update")
        if update in (0, 1, 19, 49, 99, 199):
            log.append({"update": update + 1, **metrics})
    after_train, _ = evaluate_windows(wm, train)
    after_valid, predictions = evaluate_windows(wm, valid)
    coverage = unique_coverage(data, valid_idx, stats)
    provenance = {
        **metadata,
        "episode_disjoint": not ({keys[i] for i in train_idx} & {keys[i] for i in valid_idx}),
        "reference_verified": True,
        "simulator_labels_verified": True,
        "cross_clip_generalization": False,
    }
    readiness = readiness_gate(
        after_valid, coverage=coverage, provenance=provenance, trained_updates=200
    )
    last_name = f"dynamics.mlp.net.{2 * len(cfg.hidden)}.weight"
    delta = dict(wm.named_parameters())[last_name].detach() - initial[last_name]
    results = {
        "scope": "controlled supervised diagnostic only; not natural-HOI trained model",
        "seed": 304,
        "source": str(source.resolve()),
        "source_archive_sha256": metadata["archive_sha256"],
        "config": asdict(cfg),
        "trainer_config": asdict(trainer.cfg),
        "train_rows": len(train_idx),
        "validation_rows": len(valid_idx),
        "train_episode_keys": sorted({keys[i] for i in train_idx}),
        "validation_episode_keys": sorted(valid_keys),
        "reference_sha256": clip.manifest["sha256"],
        "asset_sha256": metadata["replay"]["asset_manifest_sha256"],
        "initial_component_scales_robot_object_force_term": components,
        "force_statistics": {k: v.cpu().tolist() for k, v in stats.state_dict().items()},
        "normalization_fit_scope": "all eligible training windows; reflects uniform-window sampling distribution, no validation rows",
        "unique_train_coverage": unique_coverage(data, train_idx, stats),
        "unique_validation_coverage": coverage,
        "before_train": before_train,
        "before_validation": before_valid,
        "after_train": after_train,
        "after_validation": after_valid,
        "updates": log,
        "parameter_delta_L2": {
            "trunk": sum(
                float((p.detach() - initial[k]).square().sum())
                for k, p in wm.named_parameters()
                if k.startswith("dynamics.") and ".net.6." not in k
            )
            ** 0.5,
            "robot_head": float(delta[:70].norm()),
            "object_head": float(delta[70:82].norm()),
            "force_head": float(delta[82:].norm()),
        },
        "last_head_gradient_L2": {
            name: float(dict(wm.named_parameters())[last_name].grad[sl].norm())
            for name, sl in [
                ("robot", slice(0, 70)),
                ("object", slice(70, 82)),
                ("force", slice(82, 88)),
            ]
        },
        "readiness": readiness,
        "provenance": provenance,
    }
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        output / "predictions.npz", **{k: v.numpy() for k, v in predictions.items()}
    )
    torch.save(
        {
            "wm": wm.state_dict(),
            "config": asdict(cfg),
            "force_active_mass": stats.active_mass,
            "source_archive_sha256": metadata["archive_sha256"],
            "reference_sha256": clip.manifest["sha256"],
            "readiness": readiness,
            "updates": 200,
        },
        output / "diagnostic_weights.pt",
    )
    print(
        json.dumps(
            {
                k: results[k]
                for k in (
                    "train_rows",
                    "validation_rows",
                    "parameter_delta_L2",
                    "unique_validation_coverage",
                    "readiness",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
