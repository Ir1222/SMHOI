"""Archive, physical batch adapter and historical integrity; no optimizer/evaluation."""

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
from supermimic.hoi.legacy_validation import legacy_force_readiness_gate
from supermimic.hoi.wm_trainer import JointWorldModelTrainer
from supermimic.hoi.world_model import JointWorldModel, JointWorldModelCfg, LossWeights
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WMTrainerCfg

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz"
DATA = ROOT / "outputs/hoi/contact_supervision/force_model_validation/collection_02"
HASH = "f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7"


def test_archive_episode_split_and_physical_loss_isolation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No optimizer update authorized")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    monkeypatch.syspath_prepend(str(ROOT / "scripts/hoi"))
    from p3_common import archived_reference

    clip = archived_reference()
    data, meta = load_archive(DATA, reference_hashes=[HASH], references=[clip])
    keys = episode_keys(data, meta)
    train_idx, valid_idx = partition_episodes(data, meta, {k for k in keys if k[2] >= 6})
    assert (len(train_idx), len(valid_idx)) == (240, 80)
    assert not {keys[i] for i in train_idx} & {keys[i] for i in valid_idx}
    replay = replay_from_archive(data, meta, train_idx)
    assert replay.window_mask(8).sum() == 72
    batch = replay.sample_windows(4, 8)
    wm = JointWorldModel(
        JointWorldModelCfg(hidden=(16, 16), weights=LossWeights(1.0, 1.0)),
        AnalyticHeads,
        "cpu",
    )
    tr = JointWorldModelTrainer(
        wm, [clip], WMTrainerCfg(batch_size=4, batches_per_update=1, compile=None)
    )
    loss, _ = tr.loss_on(batch)
    modified = {k: v.clone() for k, v in batch.items()}
    modified["a_clean"].normal_(0, 50)
    modified["force_mean_box"].normal_(0, 100)
    modified["occupancy"].fill_(0.73)
    same, _ = tr.loss_on(modified)
    torch.testing.assert_close(loss, same, atol=0, rtol=0)
    # Sidecar validation belongs to the archive boundary, not physical loss.
    modified["intent_ref"].zero_()
    same, _ = tr.loss_on(modified)
    torch.testing.assert_close(loss, same, atol=0, rtol=0)
    del modified["intent_ref"]
    same, _ = tr.loss_on(modified)
    torch.testing.assert_close(loss, same, atol=0, rtol=0)
    changed_data = {k: v.copy() for k, v in data.items()}
    changed_data["intent_ref"][0] = 0
    with pytest.raises(ValueError, match="canonical reference"):
        validate_reference_intent(changed_data, meta, [clip])
    refs = tr.references_for(batch)
    root = clip.arrays["body_names"].tolist().index("pelvis")
    np.testing.assert_array_equal(
        refs[0].root_pos_w.numpy(),
        clip.arrays["body_pos_w"][batch["frame"][0].numpy() + 1, root],
    )
    assert tr.physical_states(batch).shape == (9, 4, 84)
    with pytest.raises(ValueError, match="whole source episodes"):
        replay_from_archive(data, meta, train_idx[1:])
    with pytest.raises(ValueError, match="reference identity"):
        load_archive(DATA, reference_hashes=["wrong"])
    with pytest.raises(ValueError, match="Failed or incomplete"):
        load_archive(DATA.parent / "collection_01", reference_hashes=[HASH])
    modified["transition_valid"][0, 0] = False
    with pytest.raises(ValueError, match="valid windows"):
        tr.loss_on(modified)


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


def test_failed_contact_capture_remains_rejected():
    path = ROOT / "outputs/hoi/contact_supervision/contact_intent_model_validation/collection_01"
    with pytest.raises(ValueError, match="Failed or incomplete"):
        load_archive(path, reference_hashes=[HASH])
    saved = json.loads((path / "metadata.json").read_text())
    assert saved["transitions"] == 1 and saved["acceptance"] == "FAIL"
    assert "[512]" in saved["error"]
