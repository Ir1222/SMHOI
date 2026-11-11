"""Collector failure preservation and fit preflight; no simulator or fitting here."""

import importlib
import json
from pathlib import Path

import pytest

from supermimic.hoi.dataset import load_archive

ROOT = Path(__file__).resolve().parents[2]
HASH = "f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7"


def test_capture_failure_is_recorded_without_retry_or_overwrite(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/hoi"))
    collector = importlib.import_module("collect_p3_diagnostic")
    calls = []

    def fail(cfg):
        calls.append(cfg)
        raise RuntimeError("simulated construction failure for unit test")

    monkeypatch.setattr(collector, "make_physical_env", fail)
    output = tmp_path / "failed_capture"
    with pytest.raises(RuntimeError, match="construction failure"):
        collector.run_contact(output)
    meta = json.loads((output / "metadata.json").read_text())
    assert len(calls) == 1 and meta["acceptance"] == "FAIL" and meta["transitions"] == 0
    assert "construction failure" in meta["error"]
    assert not (output / "transitions.npz").exists()
    with pytest.raises(FileExistsError):
        collector.run_contact(output)
    assert len(calls) == 1


def test_existing_all_on_collection_cannot_enter_contact_fit(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/hoi"))
    fit = importlib.import_module("fit_p3_diagnostic")
    data, metadata = load_archive(
        ROOT / "outputs/hoi/contact_supervision/force_model_validation/collection_02",
        reference_hashes=[HASH],
    )
    with pytest.raises(ValueError, match="exact approved 32-transition"):
        fit.diagnostic_partitions(data, metadata)
