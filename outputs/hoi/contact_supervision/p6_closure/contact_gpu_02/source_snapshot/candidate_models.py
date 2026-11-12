"""Explicit candidate solids and exact non-overlapping-volume mass integration."""

import copy
import itertools
import numpy as np
from closure_common import E, inputs, save


def mass_properties(geoms, mass, union=True):
    lo = np.array([np.array(g["pos"]) - g["size"] for g in geoms])
    hi = np.array([np.array(g["pos"]) + g["size"] for g in geoms])
    assert all(np.allclose(g["quat"], [1, 0, 0, 0]) for g in geoms)
    cells = []
    if union:
        axes = [np.unique(np.r_[lo[:, i], hi[:, i]]) for i in range(3)]
        for ids in itertools.product(*(range(len(a) - 1) for a in axes)):
            l = np.array([axes[i][j] for i, j in enumerate(ids)])
            h = np.array([axes[i][j + 1] for i, j in enumerate(ids)])
            center = (l + h) / 2
            if np.any(np.all((center > lo) & (center < hi), axis=1)):
                cells.append((center, (h - l) / 2))
    else:
        cells = [((l + h) / 2, (h - l) / 2) for l, h in zip(lo, hi)]
    volumes = np.array([8 * np.prod(h) for c, h in cells])
    weights = mass * volumes / volumes.sum()
    com = sum(w * c for w, (c, h) in zip(weights, cells)) / mass
    I = np.zeros((3, 3))
    for w, (c, h) in zip(weights, cells):
        v = c - com
        I += np.diag(w / 3 * (np.sum(h * h) - h * h)) + w * (
            np.dot(v, v) * np.eye(3) - np.outer(v, v)
        )
    vals, vecs = np.linalg.eigh(I)
    if np.linalg.det(vecs) < 0:
        vecs[:, 0] *= -1
    return {
        "mass_kg": mass,
        "com_pos_root_m": com.tolist(),
        "diagonal_kg_m2": vals.tolist(),
        "principal_axes_wxyz": rot_from_matrix(vecs).tolist(),
    }, {
        "tensor_root_axes_at_COM_kg_m2": I.tolist(),
        "union_volume_m3": float(volumes.sum()),
        "component_volume_sum_m3": float(np.prod(hi - lo, axis=1).sum()),
        "overlap_volume_m3": float(np.prod(hi - lo, axis=1).sum() - volumes.sum())
        if union
        else None,
        "integration_cells": len(cells),
        "positive_definite": bool(np.all(vals > 0)),
        "triangle_margin_kg_m2": float(vals.sum() - 2 * vals.max()),
        "density_kg_m3": float(mass / volumes.sum()),
        "mass_assumption": "homogeneous density on union of stated solids"
        if union
        else "historical component sum; overlap double counted",
    }


def rot_from_matrix(r):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(r).as_quat(scalar_first=True)


def main():
    out = E / "candidates"
    out.mkdir(exist_ok=False)
    m = inputs()[0]
    original = m["object"]
    base = original["geoms"][0]

    def geom(name, pos, size):
        g = copy.deepcopy(base)
        g.update(name=name, pos=list(pos), size=list(size))
        return g

    disjoint = [
        geom("bottom", [0, 0, 0], [0.09, 0.17, 0.01]),
        geom("side1", [-0.08, 0, 0.135], [0.01, 0.17, 0.125]),
        geom("side2", [0.08, 0, 0.135], [0.01, 0.17, 0.125]),
        geom("side3", [0, 0.16, 0.135], [0.07, 0.01, 0.125]),
        geom("side4", [0, -0.16, 0.135], [0.07, 0.01, 0.125]),
        geom("handle_left", [0, 0.18, 0.21], [0.03, 0.01, 0.01]),
        geom("handle_right", [0, -0.18, 0.21], [0.03, 0.01, 0.01]),
    ]
    # Their set union exactly equals the original seven cuboids; no overlapping interiors.
    definitions = [
        ("original", original["geoms"], 2.0, "original inertial source parity"),
        ("p6_component_proposal", original["geoms"], 2.0, "historical overlap-counting proposal"),
        (
            "union_shell",
            disjoint,
            2.0,
            "original collision union, disjoint pieces and exact homogeneous shell mass",
        ),
        (
            "single_body",
            [geom("solid", [0, 0, 0.125], [0.09, 0.17, 0.135])],
            2.0,
            "solid approximation of main box, handles omitted",
        ),
        (
            "single_envelope",
            [geom("solid", [0, 0, 0.125], [0.09, 0.19, 0.135])],
            2.0,
            "solid full envelope, handle extents filled",
        ),
        (
            "smtrack_2kg",
            [geom("solid", [0, 0, 0.17], [0.15, 0.16, 0.18])],
            2.0,
            "SMTrack dimensions, HDMI root mapping and matched mass",
        ),
        (
            "smtrack_1p5kg",
            [geom("solid", [0, 0, 0.17], [0.15, 0.16, 0.18])],
            1.5,
            "SMTrack native mass; centre root mapped +.17m so bottom remains -.01m",
        ),
        (
            "grasp_pads",
            [
                geom("core", [0, 0, 0.125], [0.09, 0.15, 0.135]),
                geom("left_grasp", [0, 0.16, 0.14], [0.06, 0.01, 0.05]),
                geom("right_grasp", [0, -0.16, 0.14], [0.06, 0.01, 0.05]),
            ],
            2.0,
            "disjoint side pads at original targets; friction unchanged; simulation approximation",
        ),
    ]
    for name, gs, mass, note in definitions:
        inert, calc = mass_properties(gs, mass, union=name != "p6_component_proposal")
        if name == "original":
            inert = copy.deepcopy(original["inertia"])
        points = np.concatenate(
            [
                np.array(g["pos"])
                + np.array(g["size"]) * np.array(list(itertools.product((-1, 1), repeat=3)))
                for g in gs
            ]
        )
        save(
            out / (name + ".json"),
            {
                "schema": 1,
                "name": name,
                "production_selected": False,
                "simulation_approximation": True,
                "description": note,
                "root": "original HDMI bottom-plate centre; immutable original object pose trajectory",
                "geoms": gs,
                "inertia": inert,
                "mass_calculation": calc,
                "aabb_min_root_m": points.min(0),
                "aabb_max_root_m": points.max(0),
                "oriented_dimensions_m": points.max(0) - points.min(0),
                "source_manifest_sha256": "1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1",
            },
        )
    # One-axis-at-a-time sensitivity; geometry only, fixed 2kg and root lower surface.
    for axis, values in [
        (0, [0.07, 0.08, 0.10, 0.11]),
        (1, [0.15, 0.16, 0.18, 0.19]),
        (2, [0.115, 0.125, 0.145, 0.155]),
    ]:
        for h in values:
            half = np.array([0.09, 0.17, 0.135])
            half[axis] = h
            gs = [geom("solid", [0, 0, half[2] - 0.01], half)]
            inert, calc = mass_properties(gs, 2.0)
            name = f"sensitivity_a{axis}_{h:.3f}"
            save(
                out / (name + ".json"),
                {
                    "schema": 1,
                    "name": name,
                    "geoms": gs,
                    "inertia": inert,
                    "mass_calculation": calc,
                    "production_selected": False,
                    "geometry_sensitivity_only": True,
                },
            )
    print("Candidates", len(list(out.glob("*.json"))))


if __name__ == "__main__":
    main()
