"""Opt-in, hash-pinned MuJoCo Warp float32 line-search compatibility guard.

P6 closure isolated a sign-changing derivative bracket at float32 resolution
while the20-iteration search still raised LS_ITERATIONS. This guard adds only a
bracket-width stopping condition, retaining the best cost-improving endpoint.
It changes no physics or iteration settings and never suppresses overflow bits.

Keep the context active for the entire diagnostic environment lifetime. It is
process-global and is deliberately NOT enabled by default scene/training paths.
Installed dependency files are never written. Unknown source revisions fail
closed and require a new audit. Upstream integration is preferable long term.
"""

from __future__ import annotations

import ast
import hashlib
import linecache
import threading
from contextlib import contextmanager
from pathlib import Path

SOURCE_SHA256 = "1bb26d51451a065b9e183acacfbd5e59d85df1eb46f17951aa3d1727db6853ff"
_LOCK = threading.Lock()
_NEEDLE = """        ls_done = (
          converged
          or (not swap_lo and not swap_hi)"""
_REPLACEMENT = """        precision_bracket = (
          lo[1] <= 0.0 and hi[1] >= 0.0
          and wp.abs(hi_alpha - lo_alpha) <= _ALPHA_NOISE_EPS * wp.max(1.0, wp.max(wp.abs(lo_alpha), wp.abs(hi_alpha)))
          and (lo[0] < 0.0 or hi[0] < 0.0)
        )
        ls_done = (
          converged or precision_bracket
          or (not swap_lo and not swap_hi)"""


def patched_factory_source(source: str) -> str:
    """Validate the audited dependency and change precisely one condition."""
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unaudited MuJoCo Warp solver source; precision guard refused")
    tree = ast.parse(source)
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_linesearch_iterative_kernel"
    )
    start = min([function.lineno] + [d.lineno for d in function.decorator_list]) - 1
    fragment = "".join(source.splitlines(keepends=True)[start : function.end_lineno])
    if fragment.count(_NEEDLE) != 1:
        raise ValueError("Audited line-search condition not uniquely located")
    # cache_kernel keys by function NAME and arguments, not source identity.
    # A distinct name prevents a pre-existing baseline kernel from being reused,
    # and prevents the opt-in kernel from contaminating later baseline scenes.
    return fragment.replace(_NEEDLE, _REPLACEMENT).replace(
        "def _linesearch_iterative_kernel(", "def _smhoi_p6_linesearch_precision_kernel("
    )


@contextmanager
def bracket_precision_guard():
    """Temporarily use the tested factory; restore it even when a test fails."""
    from mujoco_warp._src import solver

    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Precision guard is process-global; nested/concurrent use refused")
    original = solver._linesearch_iterative_kernel
    try:
        source = Path(solver.__file__).read_text()
        fragment = patched_factory_source(source)
        filename = "<smhoi-p6-linesearch-" + hashlib.sha256(fragment.encode()).hexdigest() + ">"
        # Warp obtains nested-kernel source through inspect/linecache. Register
        # this exact versioned fragment rather than changing site-packages.
        linecache.cache[filename] = (
            len(fragment),
            None,
            fragment.splitlines(keepends=True),
            filename,
        )
        namespace = vars(solver).copy()
        exec(compile(fragment, filename, "exec"), namespace)  # noqa: S102 -- hash-pinned audited source only
        solver._linesearch_iterative_kernel = namespace["_smhoi_p6_linesearch_precision_kernel"]
        yield {
            "source_sha256": SOURCE_SHA256,
            "patch_sha256": hashlib.sha256(fragment.encode()).hexdigest(),
            "rule": "sign bracket + cost improvement + width<=8f32 ulps of unit ray anchor",
        }
    finally:
        solver._linesearch_iterative_kernel = original
        _LOCK.release()
