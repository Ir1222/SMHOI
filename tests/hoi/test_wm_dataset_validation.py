"""Archive, episode split, real optimizer-path and readiness contract tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.hoi.dataset import (
    episode_keys,
    load_archive,
    partition_episodes,
    replay_from_archive,
    validate_reference_intent,
)
from supermimic.hoi.reference import ReferenceClip
from supermimic.hoi.validation import (
    binary_metrics,
    evaluate_windows,
    intent_events,
    legacy_force_readiness_gate,
    readiness_gate,
    unique_coverage,
)
from supermimic.hoi.wm_trainer import JointWorldModelTrainer
from supermimic.hoi.world_model import JointWorldModel, JointWorldModelCfg, LossWeights
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WMTrainerCfg

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz"
DATA = ROOT / "outputs/hoi/contact_supervision/force_model_validation/collection_02"
HASH = "f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7"


def test_archive_episode_split_and_original_optimizer_update():
    clip = ReferenceClip(REF)
    data, meta = load_archive(DATA, reference_hashes=[HASH], references=[clip])
    keys = episode_keys(data, meta)
    train_idx, valid_idx = partition_episodes(data, meta, {k for k in keys if k[2] >= 6})
    assert (len(train_idx), len(valid_idx)) == (240, 80)
    assert not {keys[i] for i in train_idx} & {keys[i] for i in valid_idx}
    replay = replay_from_archive(data, meta, train_idx)
    assert replay.window_mask(8).sum() == 72
    batch = replay.sample_windows(4, 8)
    wm = JointWorldModel(
        JointWorldModelCfg(hidden=(16, 16), weights=LossWeights(1.0, 1.0, 1.0)),
        AnalyticHeads,
        "cpu",
    )
    tr = JointWorldModelTrainer(
        wm, [ReferenceClip(REF)], WMTrainerCfg(batch_size=4, batches_per_update=1, compile=None)
    )
    loss, _ = tr.loss_on(batch)
    modified = {k: v.clone() for k, v in batch.items()}
    modified["a_clean"].normal_(0, 50)
    modified["force_mean_box"].normal_(0, 100)
    modified["occupancy"].fill_(0.73)
    same, _ = tr.loss_on(modified)
    torch.testing.assert_close(loss, same, atol=0, rtol=0)
    modified["intent_ref"].fill_(99)
    with pytest.raises(ValueError, match="binary"):
        tr.loss_on(modified)
    modified["intent_ref"].zero_()
    with pytest.raises(ValueError, match="canonical reference"):
        tr.loss_on(modified)
    modified["intent_ref"] = batch["intent_ref"].clone()
    changed_data = {k: v.copy() for k, v in data.items()}
    changed_data["intent_ref"][0] = 0
    with pytest.raises(ValueError, match="canonical reference"):
        validate_reference_intent(changed_data, meta, [clip])
    refs = tr.references_for(batch)
    root = ReferenceClip(REF).arrays["body_names"].tolist().index("pelvis")
    np.testing.assert_array_equal(
        refs[0].root_pos_w.numpy(),
        ReferenceClip(REF).arrays["body_pos_w"][batch["frame"][0].numpy() + 1, root],
    )
    before = wm.dynamics.mlp.net[-1].weight.detach().clone()
    result = tr.update(replay)
    assert np.isfinite(result["loss"]) and (wm.dynamics.mlp.net[-1].weight - before).abs().sum() > 0
    assert tr.online_error(replay, n=4).shape == (8,)
    with pytest.raises(ValueError, match="whole source episodes"):
        replay_from_archive(data, meta, train_idx[1:])
    with pytest.raises(ValueError, match="reference identity"):
        load_archive(DATA, reference_hashes=["wrong"])
    with pytest.raises(ValueError, match="Failed or incomplete"):
        load_archive(DATA.parent / "collection_01", reference_hashes=[HASH])
    modified["transition_valid"][0, 0] = False
    with pytest.raises(ValueError, match="valid windows"):
        tr.loss_on(modified)


def test_binary_metrics_denominators_bce_and_deterministic_baselines():
    probability = torch.tensor([0.2, 0.8, 0.7, 0.1])
    truth = torch.tensor([0.0, 1.0, 0.0, 1.0])
    metrics = binary_metrics(probability, truth, torch.logit(probability))
    assert [metrics[k] for k in ("tp", "tn", "fp", "fn")] == [1, 1, 1, 1]
    assert all(metrics[k] == 0.5 for k in ("accuracy", "precision", "recall", "f1"))
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.logit(probability).double(), truth.double()
    )
    assert metrics["bce"] == float(expected)
    off = binary_metrics(torch.zeros_like(truth), truth)
    assert off["precision"] is None and off["denominators"]["precision"] == 0
    assert off["f1"] == 0 and off["bce"] is None and off["bce_infinite_count"] == 2
    assert off["bce_status"] == "infinite"
    empty = binary_metrics(torch.empty(0), torch.empty(0))
    assert empty["accuracy"] is None and empty["count"] == 0
    with pytest.raises(ValueError):
        binary_metrics(probability, truth + 0.1)


def test_events_include_first_last_transition_and_unique_counts():
    bits = torch.tensor([0.0, 1.0, 1.0, 0.0])[:, None].expand(-1, 2)
    pairs = torch.stack((bits[:-1], bits[1:]), -1)
    probs = torch.tensor([0.8, 0.9, 0.2])[:, None].expand(-1, 2)
    ids = [[("ref", 0, 1, j), ("ref", 0, 1, j)] for j in range(3)]
    result = intent_events(probs, pairs, ids)
    for event in result.values():
        assert event["true_count"] == event["exact_hits"] == 2
        assert event["unique_true_count"] == event["unique_exact_hit_count"] == 1
        assert event["false_events"] == 0 and event["mean_absolute_error_steps"] == 0
    missed = intent_events(torch.zeros_like(probs), pairs, ids)
    assert missed["onset"]["exact_hits"] == 0 and missed["release"]["precision"] is None
    assert missed["release"]["timing"][0]["error_steps"] is None
    with pytest.raises(ValueError, match="Invalid event"):
        intent_events(torch.full_like(probs, float("nan")), pairs)


def test_evaluation_covers_all_one_step_rows_and_complete_windows():
    data, meta = load_archive(DATA, reference_hashes=[HASH], references=[ReferenceClip(REF)])
    keys = episode_keys(data, meta)
    _, valid = partition_episodes(data, meta, {k for k in keys if k[2] >= 6})
    replay = replay_from_archive(data, meta, valid)
    wm = JointWorldModel(
        JointWorldModelCfg(hidden=(8, 8), weights=LossWeights(1, 1, 1)), AnalyticHeads, "cpu"
    )
    metrics, predictions = evaluate_windows(wm, replay)
    cv, _ = evaluate_windows(wm, replay, constant_velocity=True)
    assert metrics["one_step_rows"] == 80 and metrics["eligible_windows"] == 24
    assert predictions["ar_state_prediction"].shape == (8, 24, 84)
    assert predictions["one_step_contact_probability"].shape == (80,)
    assert metrics["horizon_1"]["state"] == cv["horizon_1"]["state"]
    assert metrics["horizon_8"]["state"] == cv["horizon_8"]["state"]
    coverage = unique_coverage(data, valid, meta)
    assert coverage["target_on"] == 80 and coverage["target_off"] == 0
    gate = readiness_gate(metrics, coverage=coverage, provenance={}, trained_updates=0)
    assert gate["overall"] == "PARTIAL" and not gate["ready"]
    assert any("coverage" in reason for reason in gate["gate_b"]["missing"])


def test_readiness_approved_criteria_and_fail_closed_missing_evidence():
    # Synthetic arithmetic/schema tests only, never reported as learned validation.
    metrics = {
        f"horizon_{h}": {
            "state": {"all": {"robot_nse": 0.9, "object_nse": 0.9}},
            "contact": {"accuracy": 1.0, "f1": 1.0, "target_on": 1, "target_off": 1},
        }
        for h in (1, 8)
    }
    metrics["contact_baselines"] = {"persistence": {"one_step": {"accuracy": 0.875, "f1": 0.875}}}
    metrics["autoregressive"] = {
        "events": {
            name: {"true_count": 1, "exact_hits": 1, "false_events": 0}
            for name in ("onset", "release")
        }
    }
    baseline = {
        f"horizon_{h}": {"state": {"all": {"robot_nse": 1.0, "object_nse": 1.0}}} for h in (1, 8)
    }
    kwargs = {
        "coverage": {k: 1 for k in ("target_on", "target_off", "onset", "release")},
        "provenance": {
            k: True for k in ("episode_disjoint", "reference_verified", "simulator_labels_verified")
        },
        "trained_updates": 200,
        "baseline": baseline,
        "parameter_updates": {k: 1.0 for k in ("robot_head", "object_head", "contact_head")},
    }
    assert readiness_gate(metrics, **kwargs)["overall"] == "PASS"
    worse = copy.deepcopy(metrics)
    worse["horizon_8"]["state"]["all"]["object_nse"] = 1.1
    worse["autoregressive"]["events"]["release"]["exact_hits"] = 0
    gate = readiness_gate(worse, **kwargs)
    assert gate["gate_a"]["status"] == gate["gate_b"]["status"] == "FAIL"
    kwargs["coverage"]["release"] = 0
    assert readiness_gate(metrics, **kwargs)["gate_b"]["status"] == "PARTIAL"


def test_historical_force_readiness_retains_its_original_meaning():
    old = DATA.parent
    fit = json.loads((old / "fit_01/results.json").read_text())
    expected = json.loads((old / "readiness_review.json").read_text())
    gate = legacy_force_readiness_gate(
        fit["after_validation"],
        coverage=fit["unique_validation_coverage"],
        provenance=fit["provenance"],
        trained_updates=200,
    )
    assert gate == expected and not gate["ready"]
    assert "P6" in gate["scope"]  # Historical semantics, not a new P3 requirement.
