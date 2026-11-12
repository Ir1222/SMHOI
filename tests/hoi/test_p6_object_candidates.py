"""Independent geometric/inertial invariants for versioned closure candidates."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ROOT / "outputs/hoi/contact_supervision/p6_closure/candidates"


def load(name):
    return json.loads((CANDIDATES / f"{name}.json").read_text())


def membership(points, geoms):
    return np.stack(
        [np.all(np.abs(points - g["pos"]) < np.array(g["size"]), axis=1) for g in geoms], axis=1
    )


def test_disjoint_shell_preserves_original_solid_union_without_duplicate_mass():
    original, shell = load("original"), load("union_shell")
    points = np.random.default_rng(606).uniform([-0.1, -0.2, -0.02], [0.1, 0.2, 0.27], (50000, 3))
    a, b = membership(points, original["geoms"]), membership(points, shell["geoms"])
    np.testing.assert_array_equal(a.any(axis=1), b.any(axis=1))
    assert b.sum(axis=1).max() == 1
    assert (a.sum(axis=1) > 1).any()
    np.testing.assert_allclose(shell["mass_calculation"]["union_volume_m3"], 0.006072, atol=1e-15)
    np.testing.assert_allclose(
        original["mass_calculation"]["overlap_volume_m3"], 0.000656, atol=1e-15
    )


def test_selected_single_box_root_COM_and_homogeneous_tensor():
    c = load("single_body")
    np.testing.assert_array_equal(c["inertia"]["com_pos_root_m"], [0, 0, 0.125])
    np.testing.assert_allclose(c["oriented_dimensions_m"], [0.18, 0.34, 0.27], atol=1e-15)
    expected = np.diag([0.03141666666666667, 0.01755, 0.02466666666666667])
    np.testing.assert_allclose(
        c["mass_calculation"]["tensor_root_axes_at_COM_kg_m2"], expected, atol=1e-14
    )
    for path in CANDIDATES.glob("*.json"):
        item = json.loads(path.read_text())
        eig = np.array(item["inertia"]["diagonal_kg_m2"])
        assert (eig > 0).all() and 2 * eig.max() <= eig.sum()
        assert item["production_selected"] is False


def test_diagnostic_entrypoints_import_without_running_experiments(monkeypatch):
    """Catch removed re-exports and diagnostic entrypoint regressions."""
    monkeypatch.syspath_prepend(str(CANDIDATES.parent / "scripts"))
    for name in (
        "closure_common",
        "candidate_models",
        "geometry",
        "physical_compare",
        "solver_diagnosis",
        "trajectory_ik",
        "export_candidate",
        "independent_controller",
        "integrated",
        "failure_replay",
        "analyze_closure",
        "audit_details",
        "contact_parameters",
        "contact_gpu",
        "overflow_audit",
    ):
        assert importlib.import_module(name) is sys.modules[name]
