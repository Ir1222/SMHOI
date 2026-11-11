"""Retired entry points must stop before collection, training or output creation."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["collect_p3_diagnostic", "fit_p3_diagnostic"])
def test_retired_diagnostic_has_no_experiment_entry(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/hoi"))
    module = importlib.import_module(name)
    assert not hasattr(module, "run") and not hasattr(module, "run_contact")
    with pytest.raises(SystemExit, match="Retired P3 contact diagnostic"):
        module.main()
