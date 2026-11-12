"""Actual native-shaped capsule/Toaster pair generation; geometry-only probe."""

import json
from unittest.mock import patch
import mujoco
from closure_common import E, save
from contact_parameters import FIELDS, load_source


def main():
    up = E / "upstream"
    old = load_source("p6_pair_oldcfg", up / "mjlab_v1_4_spec_config.py")
    oldspec = load_source("p6_pair_oldspec", up / "mjlab_v1_4_spec.py")
    import mjlab.utils.spec_config as cfg
    import mjlab.utils.spec as utils

    with patch.object(cfg, "CollisionCfg", old.CollisionCfg):
        asset = load_source("p6_pair_toaster", up / "src/mjlab_g1/assets/toaster_constants.py")
    spec = asset.get_toaster_spec()
    with patch.object(utils, "disable_collision", oldspec.disable_collision, create=True):
        asset.FULL_COLLISION_TOASTER.edit_spec(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    overlap = []
    for side in ("left", "right"):
        marker = m.geom(side + "_grasp_marker_collision").id
        for g in range(m.ngeom):
            if g == marker or not m.geom_contype[g]:
                continue
            dist = float(mujoco.mj_geomDistance(m, d, marker, g, 1, None))
            if dist < 0:
                overlap.append(
                    {
                        "marker": side,
                        "geom": g,
                        "name": m.geom(g).name,
                        "dist": dist,
                        "same_body": bool(m.geom_bodyid[marker] == m.geom_bodyid[g]),
                    }
                )
    audit = json.loads((E / "contact_audit_04/compiled_audit.json").read_text())
    hand = next(g for g in audit["scene_counterparts"] if g["name"] == "robot/left_hand_collision")
    for side, sign in (("left", 1), ("right", -1)):
        marker = m.geom(side + "_grasp_marker_collision").id
        normal = d.xmat[m.geom_bodyid[marker]].reshape(3, 3)[:, 0] * sign
        pos = d.geom_xpos[marker] + normal * (0.005 + hand["size"][0] - 0.0005)
        b = spec.worldbody.add_body(name="probe_" + side, pos=pos)
        b.add_freejoint()
        g = b.add_geom(
            name="native_shape_" + side,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=hand["size"],
            mass=1,
        )
        for field in FIELDS:
            setattr(g, field, hand[field])
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    records = []
    for c in d.contact[: d.ncon]:
        if not any(m.geom(g).name.startswith("native_shape_") for g in [c.geom1, c.geom2]):
            continue
        records.append(
            {
                "geoms": [int(c.geom1), int(c.geom2)],
                "names": [m.geom(c.geom1).name, m.geom(c.geom2).name],
                "dist": c.dist,
                "pos": c.pos.copy(),
                "frame": c.frame.copy(),
                "dim": c.dim,
                "friction": c.friction.copy(),
                "solref": c.solref.copy(),
                "solimp": c.solimp.copy(),
                "efc_address": c.efc_address,
            }
        )
    save(
        E / "toaster_pair_audit.json",
        {
            "geometry_only_not_arm_reachability_or_carrying": True,
            "native_capsule_size": hand["size"],
            "marker_overlap_pairs": overlap,
            "generated_native_capsule_contacts": records,
            "explicit_pairs": m.npair,
            "body_excludes": m.nexclude,
            "full_scene_ncon": d.ncon,
            "full_scene_nefc": d.nefc,
        },
    )
    print("overlap pairs", len(overlap), "native capsule contacts", len(records))


if __name__ == "__main__":
    main()
