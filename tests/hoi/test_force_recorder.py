"""Exercise actual installed callback ordering, not a mocked recorder sequence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="P2 physical interval capture requires CUDA"
)
def test_physical_interval_recorder_and_terminal_copy(tmp_path):
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "physical"
    command = [
        sys.executable,
        str(root / "scripts/hoi/diagnose_force_replay.py"),
        "--output",
        str(out),
    ]
    result = subprocess.run(
        command,
        check=False,
        cwd=root,
        env={**os.environ, "MUJOCO_GL": "egl", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / "diagnostics.json").read_text())
    assert summary["acceptance"] == "PASS"
    assert summary["transitions"] == 96 and summary["windows_checked"] == 512
    assert summary["active_terminal_probe"]["transitions"] == 8
    assert summary["active_terminal_probe"]["retained_copy_unchanged_after_reset"]
    assert summary["errors"]["action"] == 0
    assert summary["all_overflow_zero"]
