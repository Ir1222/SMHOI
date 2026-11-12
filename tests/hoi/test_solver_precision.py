import hashlib
from pathlib import Path

import pytest

from supermimic.hoi.solver_precision import (
    SOURCE_SHA256,
    bracket_precision_guard,
    patched_factory_source,
)


def test_unknown_solver_revision_fails_closed():
    with pytest.raises(ValueError, match="Unaudited"):
        patched_factory_source("def _linesearch_iterative_kernel(): pass\n")


def test_guard_restores_factory_on_failure_and_refuses_nesting():
    from mujoco_warp._src import solver

    original = solver._linesearch_iterative_kernel
    source_path = Path(solver.__file__)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert digest == SOURCE_SHA256
    with (
        pytest.raises(RuntimeError, match="deliberate test failure"),
        bracket_precision_guard() as profile,
    ):
        assert solver._linesearch_iterative_kernel is not original
        assert solver._linesearch_iterative_kernel.__name__ != original.__name__
        assert profile["source_sha256"] == digest
        with pytest.raises(RuntimeError, match="nested/concurrent"), bracket_precision_guard():
            pass
        raise RuntimeError("deliberate test failure")
    assert solver._linesearch_iterative_kernel is original
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == digest


def test_patched_factory_preserves_failure_reporting_and_iteration_limit():
    from mujoco_warp._src import solver

    fragment = patched_factory_source(Path(solver.__file__).read_text())
    assert "for _ in range(LS_ITERATIONS):" in fragment
    assert "wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.LS_ITERATIONS))" in fragment
    assert "lo[1] <= 0.0 and hi[1] >= 0.0" in fragment
    assert "and (lo[0] < 0.0 or hi[0] < 0.0)" in fragment
