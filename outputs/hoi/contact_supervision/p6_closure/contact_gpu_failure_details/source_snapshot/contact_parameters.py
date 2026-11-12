"""C1.7 compiled contact audit and matched material-only CPU experiments."""

import argparse
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

from closure_common import E, cpu_model, dump_gz, manifest_for, save
from box_dynamics import model_for
import mujoco
import mujoco_warp as mjw
import numpy as np

GROUPS = {
    "baseline": {},
    "friction_only": {"friction": [1.0, 0.1, 0.1]},
    "condim4_only": {"condim": 4},
    "torsion_after_condim4": {"condim": 4, "friction": [1.0, 0.1, 0.1]},
    "solimp_only": {"solimp": [0.98, 0.99, 0.001, 0.5, 2.0]},
    "solref_only": {"solref": [0.001, 1.0]},
    "margin_only": {"margin": 0.002},
    "gap_only": {"gap": 0.002},
    "gap_after_margin": {"margin": 0.002, "gap": 0.002},
    "combined_followup": {
        "friction": [1.0, 0.1, 0.1],
        "condim": 4,
        "solimp": [0.98, 0.99, 0.001, 0.5, 2.0],
        "solref": [0.001, 1.0],
    },
}
FIELDS = (
    "friction",
    "solref",
    "solimp",
    "margin",
    "gap",
    "condim",
    "priority",
    "contype",
    "conaffinity",
    "solmix",
)


def decode(value):
    value = int(value)
    flags = {v.name: int(v) for v in mjw.OverflowType if int(v) and int(v) & (int(v) - 1) == 0}
    return {
        "value": value,
        "active": [n for n, bit in flags.items() if value & bit],
        "LS_ITERATIONS_bit_set": bool(value & int(mjw.OverflowType.LS_ITERATIONS)),
        "unknown_bits": value & ~sum(flags.values()),
    }


def set_group(spec, group):
    for g in spec.geoms:
        if g.name.startswith("box/"):
            for field, value in GROUPS[group].items():
                setattr(g, field, value)


def config_edit(group, existing=None):
    def change(cfg):
        if existing:
            existing(cfg)
        base = cfg.scene.spec_fn

        def edit(spec):
            base(spec)
            set_group(spec, group)

        cfg.scene.spec_fn = edit

    return change


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def geom_config(m, gid):
    return {
        "id": gid,
        "name": m.geom(gid).name,
        "body": m.body(m.geom_bodyid[gid]).name,
        "type": int(m.geom_type[gid]),
        "pos": m.geom_pos[gid].copy(),
        "size": m.geom_size[gid].copy(),
        "quat": m.geom_quat[gid].copy(),
        **{f: np.asarray(getattr(m, "geom_" + f)[gid]).tolist() for f in FIELDS},
    }


def probe_pair(a, b, distance=-0.001):
    """Compile a two-geom probe; contact mixing is independent of geom shape."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.005
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    plane = spec.worldbody.add_geom(name="a", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01])
    body = spec.worldbody.add_body(pos=[0, 0, 0.05 + distance])
    body.add_freejoint()
    sphere = body.add_geom(name="b", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0], mass=2)
    for g, data in [(plane, a), (sphere, b)]:
        for f in FIELDS:
            setattr(g, f, data[f])
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    result = {
        "pair": [a["name"], b["name"]],
        "distance_m": distance,
        "ncon": d.ncon,
        "nefc": d.nefc,
        "contacts": [],
    }
    for c in d.contact[: d.ncon]:
        result["contacts"].append(
            {
                "dist": c.dist,
                "dim": c.dim,
                "friction5": c.friction.copy(),
                "solref": c.solref.copy(),
                "solimp": c.solimp.copy(),
                "includemargin": c.includemargin,
                "efc_address": c.efc_address,
                "active_torsion": c.dim >= 4,
                "active_rolling": c.dim == 6,
                "pyramidal_rows": 1 if c.dim == 1 else 2 * (c.dim - 1),
                "effective_positive_timeconst_s": max(c.solref[0], 2 * m.opt.timestep),
            }
        )
    return result


def audit(out):
    out.mkdir(exist_ok=False)
    up = E / "upstream"
    sm = load_source("p6_upstream_smtrack", up / "src/mjlab_g1/tasks/smtrack/mdp/box.py")
    smspec = sm.get_box_spec()
    oldcfg = load_source("p6_mjlab14_spec_config", up / "mjlab_v1_4_spec_config.py")
    import mjlab.utils.spec_config as current_cfg

    with patch.object(current_cfg, "CollisionCfg", oldcfg.CollisionCfg):
        toaster = load_source(
            "p6_upstream_toaster", up / "src/mjlab_g1/assets/toaster_constants.py"
        )
    tspec = toaster.get_toaster_spec()
    raw = tspec.compile()
    oldspec = load_source("p6_mjlab14_spec", up / "mjlab_v1_4_spec.py")
    import mjlab.utils.spec as current_spec

    with patch.object(current_spec, "disable_collision", oldspec.disable_collision, create=True):
        toaster.FULL_COLLISION_TOASTER.edit_spec(tspec)
    applied = tspec.compile()
    (out / "toaster_after_upstream_CollisionCfg.xml").write_text(tspec.to_xml())
    (out / "smtrack_box.xml").write_text(smspec.to_xml())
    original = cpu_model("original")
    single = cpu_model("single_body")
    models = {
        "HDMI_original": original,
        "single_body": single,
        "SMTrack": smspec.compile(),
        "Toaster_XML": raw,
        "Toaster_configured": applied,
    }
    result = {}
    hand = geom_config(original, original.geom("robot/left_hand_collision").id)
    support = (
        geom_config(original, original.geom("support_start_geom").id)
        if "support_start_geom" in [original.geom(i).name for i in range(original.ngeom)]
        else next(
            geom_config(original, i)
            for i in range(original.ngeom)
            if "support" in original.geom(i).name
        )
    )
    for name, m in models.items():
        gids = (
            list(range(m.ngeom))
            if name.startswith("Toaster") or name == "SMTrack"
            else [i for i in range(m.ngeom) if m.geom(i).name.startswith("box/")]
        )
        geoms = [geom_config(m, i) for i in gids]
        result[name] = {
            "geoms": geoms,
            "nq": m.nq,
            "nv": m.nv,
            "total_mass": float(m.body_mass.sum()),
            "hand_parameter_probes": [probe_pair(hand, g) for g in geoms],
            "support_parameter_probes": [probe_pair(support, g) for g in geoms],
        }
    result["scene_counterparts"] = [
        geom_config(original, i)
        for i in range(original.ngeom)
        if original.geom_contype[i] and not original.geom(i).name.startswith("box/")
    ]
    selected = geom_config(single, single.geom("box/solid").id)
    result["candidate_parameter_groups"] = {}
    for name, changes in GROUPS.items():
        g = dict(selected, **changes)
        result["candidate_parameter_groups"][name] = {
            "changes": changes,
            "relative_to": "condim4_only"
            if name == "torsion_after_condim4"
            else "margin_only"
            if name == "gap_after_margin"
            else "baseline",
            "interpretation": "combined followup, no single-factor attribution"
            if name == "combined_followup"
            else "one parameter group comparison",
            "hand_probe": probe_pair(hand, g),
            "support_probe": probe_pair(support, g),
            "positive_gap_probe": probe_pair(hand, g, 0.001),
            "distance3mm_probe": probe_pair(hand, g, 0.003),
        }
    result["overflow_bits"] = {v.name: int(v) for v in mjw.OverflowType}
    result["solver"] = {
        "type": mujoco.mjtSolver(original.opt.solver).name,
        "cone": mujoco.mjtCone(original.opt.cone).name,
        "iterations": original.opt.iterations,
        "ls_iterations": original.opt.ls_iterations,
        "ls_tolerance": original.opt.ls_tolerance,
        "tolerance": original.opt.tolerance,
        "timestep": original.opt.timestep,
        "refsafe_enabled": not bool(
            original.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE
        ),
    }
    for file in ["solver.py", "types.py", "constraint.py"]:
        path = Path(mjw.__file__).parent / "_src" / file
        result.setdefault("installed_source_hashes", {})[str(path)] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    # Three-element XML solimp is compiled, not guessed.
    test = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1" solimp=".98 .99 .001"/></worldbody></mujoco>'
    )
    result["compiled_three_element_solimp"] = test.geom_solimp[0].copy()
    save(out / "compiled_audit.json", result)
    print("audit complete", flush=True)


def physical(out, groups=None):
    out.mkdir(exist_ok=False)
    summaries = []
    for group, changes in GROUPS.items():
        if groups and group not in groups:
            continue
        manifest = copy.deepcopy(manifest_for("single_body"))
        manifest["object"]["geoms"][0].update(changes)
        for mode in ("settle", "drop", "slide", "yaw"):
            m = model_for(manifest)
            d = mujoco.MjData(m)
            d.qpos[:3] = [0, 0, 0.11 if mode == "drop" else 0.01]
            d.qpos[3:7] = [1, 0, 0, 0]
            bid = m.body("box/bread_box").id
            mujoco.mj_forward(m, d)
            records = []
            for k in range(400):
                d.xfrc_applied[:] = 0
                if 100 <= k < 140:
                    if mode == "slide":
                        d.xfrc_applied[bid, 0] = 30.0
                    if mode == "yaw":
                        d.xfrc_applied[bid, 5] = 4.0
                mujoco.mj_step(m, d)
                contacts = []
                for i, c in enumerate(d.contact[: d.ncon]):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, wrench)
                    contacts.append(
                        {
                            "dist": c.dist,
                            "pos": c.pos.copy(),
                            "frame": c.frame.copy(),
                            "geom1": int(c.geom1),
                            "geom2": int(c.geom2),
                            "dim": c.dim,
                            "friction": c.friction.copy(),
                            "solref": c.solref.copy(),
                            "solimp": c.solimp.copy(),
                            "efc_address": c.efc_address,
                            "force_contact": wrench.copy(),
                            "force_world_on_geom2": c.frame.reshape(3, 3).T @ wrench[:3],
                        }
                    )
                records.append(
                    {
                        "time": d.time,
                        "qpos": d.qpos.copy(),
                        "qvel": d.qvel.copy(),
                        "COM": d.xipos[bid].copy(),
                        "applied": d.xfrc_applied[bid].copy(),
                        "contacts": contacts,
                        "ncon": d.ncon,
                        "nefc": d.nefc,
                        "solver_niter": d.solver_niter.copy(),
                        "warnings": d.warning.number.copy(),
                    }
                )
            folder = out / group
            folder.mkdir(exist_ok=True)
            dump_gz(folder / (mode + ".json.gz"), records)
            depths = [max(0, -c["dist"]) for r in records for c in r["contacts"]]
            npen = sum(any(c["dist"] < -0.001 for c in r["contacts"]) for r in records)
            last = records[-100:]
            summary = {
                "group": group,
                "mode": mode,
                "steps": len(records),
                "max_penetration_m": max(depths, default=0),
                "mean_penetration_m": float(np.mean(depths)) if depths else 0,
                "fraction_steps_penetration_gt_1mm": npen / len(records),
                "final_x_m": float(d.qpos[0]),
                "max_xy_speed": max(np.linalg.norm(r["qvel"][:2]) for r in records),
                "max_yaw_speed": max(abs(r["qvel"][5]) for r in records),
                "final_yaw_rad": float(2 * np.arctan2(d.qpos[6], d.qpos[3])),
                "tail_speed": max(np.linalg.norm(r["qvel"]) for r in last),
                "peak_normal_N": max(
                    (c["force_contact"][0] for r in records for c in r["contacts"]), default=0
                ),
                "tail_reaction_z_N": np.mean(
                    [sum(c["force_world_on_geom2"][2] for c in r["contacts"]) for r in last]
                ),
                "warnings": d.warning.number.copy(),
                "max_nefc": max(r["nefc"] for r in records),
                "max_iterations": max(int(r["solver_niter"].max()) for r in records),
            }
            summaries.append(summary)
            print(group, mode, summary["max_penetration_m"], flush=True)
    save(out / "summary.json", summaries)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["audit", "physical"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", nargs="+")
    a = p.parse_args()
    audit(a.output) if a.mode == "audit" else physical(a.output, a.groups)
