"""Preflight authority migration, without training or physical asset promotion."""

import copy
import json
from pathlib import Path

import pytest
import torch

from supermimic.hoi import provenance
from supermimic.hoi.asset_manifest import load_manifest
from supermimic.hoi.reference import DEFAULT_ASSET, sha256
from supermimic.hoi.training_scene import ASSET_SHA, profile_manifest


def test_strict_loader_retires_only_planning_dependency(monkeypatch):
    before = DEFAULT_ASSET.read_bytes()
    original = Path.open

    def checked_open(path, *args, **kwargs):
        assert path.name != provenance.RETIRED_NAME, "Retired planning file must never be opened"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    loaded = load_manifest(DEFAULT_ASSET)
    raw = json.loads(before)
    for key in raw.keys() - {"sources"}:
        assert loaded[key] == raw[key]
    assert loaded["research_authority"] == provenance.research_authority()
    assert all(Path(s["path"]).name != provenance.RETIRED_NAME for s in loaded["sources"])
    assert DEFAULT_ASSET.read_bytes() == before and sha256(DEFAULT_ASSET) == ASSET_SHA
    assert not loaded["training_ready"]
    with pytest.raises(ValueError, match="training_ready"):
        profile_manifest("production")


def test_current_master_and_every_physical_source_remain_strict(monkeypatch, tmp_path):
    master = provenance.REPO / provenance.MASTER_RELATIVE
    original = provenance.sha256
    monkeypatch.setattr(provenance, "sha256", lambda p: "bad" if Path(p) == master else original(p))
    with pytest.raises(ValueError, match="Master Plan SHA256"):
        load_manifest(DEFAULT_ASSET)
    monkeypatch.setattr(provenance, "sha256", original)
    path = tmp_path / "physical.xml"
    path.write_text("physical original")
    sources = [{"path": str(path), "sha256": sha256(path)}]
    path.write_text("physical changed")
    with pytest.raises(ValueError, match="source changed"):
        provenance.verify_sources(sources)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        provenance.verify_sources(sources)
    with pytest.raises(ValueError, match="retired"):
        provenance.effective_sources([{"path": provenance.RETIRED_NAME, "sha256": "wrong"}])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native preflight requires CUDA")
def test_full_checkpoint_binds_authority_and_rejects_legacy_identity(tmp_path):
    from p5_fixtures import TEST_CFG

    from supermimic.hoi import training_checkpoint as checkpoint
    from supermimic.hoi.training import HOITrainer
    from supermimic.hoi.training_config import HOITrainConfig
    from supermimic.hoi.training_safety import SolverGuard

    cfg = HOITrainConfig(
        log_root=tmp_path, profile="p6-original", reward=TEST_CFG, sampler_g_max=18.2,
        execution="integration",
    )
    cfg.env.num_envs = 2
    cfg.schedule.updates_per_env_step = 0
    cfg.schedule.total_iterations = cfg.schedule.checkpoint_interval = 2
    with SolverGuard() as guard:
        trainer = HOITrainer(cfg, guard)
        try:
            trainer.run()
            saved = checkpoint.snapshot(trainer)
            assert saved["identity"]["research_authority"] == provenance.research_authority()
            assert saved["replay_metadata"]["research_authority"] == provenance.research_authority()
            assert trainer.learning_safety.steps == {"wm": 0, "actor": 0, "critic": 0}
            for altered in (None, {"sha256": "wrong"}):
                bad = dict(saved)
                bad["identity"] = copy.deepcopy(saved["identity"])
                if altered is None:
                    bad["identity"].pop("research_authority")
                else:
                    bad["identity"]["research_authority"] = altered
                with pytest.raises(ValueError, match="identity"):
                    checkpoint.restore(trainer, bad)
                assert torch.equal(
                    trainer.recorder.state.cpu(), saved["runtime"]["recorder"]["state"]
                )
            checkpoint.restore(trainer, saved)
            assert torch.equal(trainer.recorder.state.cpu(), saved["runtime"]["recorder"]["state"])
            assert any(x["reason"] == "resume_after_recapture" for x in guard.attestations)
        finally:
            trainer.close()
