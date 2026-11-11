"""Archive, episode split, real optimizer-path and readiness contract tests."""

from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.hoi.dataset import (
    episode_keys,
    load_archive,
    partition_episodes,
    replay_from_archive,
)
from supermimic.hoi.force import ForceStatistics
from supermimic.hoi.reference import ReferenceClip
from supermimic.hoi.validation import force_metrics, readiness_gate, transition_events
from supermimic.hoi.wm_trainer import JointWorldModelTrainer
from supermimic.hoi.world_model import JointWorldModel, JointWorldModelCfg, LossWeights
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WMTrainerCfg

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz"
DATA = ROOT / "outputs/hoi/contact_supervision/force_model_validation/collection_02"
HASH = "f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7"


def test_archive_episode_split_and_original_optimizer_update():
    data, meta = load_archive(DATA, reference_hashes=[HASH])
    keys = episode_keys(data, meta)
    train_idx, valid_idx = partition_episodes(data, meta, {k for k in keys if k[2] >= 6})
    assert (len(train_idx), len(valid_idx)) == (240, 80)
    assert not {keys[i] for i in train_idx} & {keys[i] for i in valid_idx}
    replay = replay_from_archive(data, meta, train_idx)
    assert replay.window_mask(8).sum() == 72
    batch = replay.sample_windows(4, 8)
    stats = ForceStatistics.fit(
        torch.tensor(data["force_mean_box"][train_idx]),
        torch.tensor(data["occupancy"][train_idx]),
        force_on_N=0.001,
        active_mass=0.5,
    )
    wm = JointWorldModel(
        JointWorldModelCfg(hidden=(16, 16), weights=LossWeights(1.0, 1.0, 1.0)),
        AnalyticHeads,
        stats,
        "cpu",
    )
    tr = JointWorldModelTrainer(
        wm, [ReferenceClip(REF)], WMTrainerCfg(batch_size=4, batches_per_update=1, compile=None)
    )
    loss, _ = tr.loss_on(batch)
    modified = {k: v.clone() for k, v in batch.items()}
    modified["a_clean"].normal_(0, 50)
    modified["intent_ref"].fill_(99)
    same, _ = tr.loss_on(modified)
    torch.testing.assert_close(loss, same, atol=0, rtol=0)
    refs = tr.references_for(batch)
    root = ReferenceClip(REF).arrays["body_names"].tolist().index("pelvis")
    np.testing.assert_array_equal(
        refs[0].root_pos_w.numpy(),
        ReferenceClip(REF).arrays["body_pos_w"][batch["frame"][0].numpy() + 1, root],
    )
    before = wm.dynamics.mlp.net[-1].weight.detach().clone()
    result = tr.update(replay)
    assert np.isfinite(result["loss"]) and (wm.dynamics.mlp.net[-1].weight - before).abs().sum() > 0
    with pytest.raises(ValueError, match="whole source episodes"):
        replay_from_archive(data, meta, train_idx[1:])
    with pytest.raises(ValueError, match="reference identity"):
        load_archive(DATA, reference_hashes=["wrong"])
    with pytest.raises(ValueError, match="Failed or incomplete"):
        load_archive(DATA.parent / "collection_01", reference_hashes=[HASH])
    modified["transition_valid"][0, 0] = False
    with pytest.raises(ValueError, match="valid windows"):
        tr.loss_on(modified)


def test_force_metrics_missing_strata_and_readiness_are_closed():
    f = torch.tensor([[[1.0, 0, 0], [0, 2.0, 0]], [[0.0, 0, 0], [0, 0.0, 0]]])
    o = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    stats = ForceStatistics.fit(f, o, force_on_N=0.001, active_mass=0.5)
    m = force_metrics(f, f, o, stats)
    assert m["left"]["all"]["mae_N"] == 0 and m["right"]["active"]["direction_cosine"] == 1
    assert m["left"]["inactive"]["direction_cosine"] is None
    # One synthetic onset and loss, exclusively testing event arithmetic.
    ff = torch.stack((f[1], f[0], f[1]))[:, None]
    oo = torch.tensor([[[0.0, 0.0]], [[1.0, 1.0]], [[0.0, 0.0]]])
    events = transition_events(ff, oo, stats)
    assert events["left"]["onset"]["recall"] == events["right"]["loss"]["recall"] == 1
    gate = readiness_gate({}, coverage={}, provenance={}, trained_updates=200)
    assert not gate["ready"] and not gate["actor_force_optimization_enabled"]
    assert any("coverage" in r for r in gate["reasons"])
    assert any("thresholds" in r for r in gate["reasons"])
