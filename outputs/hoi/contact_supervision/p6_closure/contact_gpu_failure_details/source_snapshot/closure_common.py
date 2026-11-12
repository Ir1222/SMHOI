"""Versioned P6 closure diagnostics; compose P1/P2 and preserve production."""

import copy
import gzip
import json
import sys
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
E = ROOT / "outputs/hoi/contact_supervision/p6_closure"
P6 = ROOT / "outputs/hoi/contact_supervision/contact_validation"
sys.path.insert(0, str(P6 / "scripts"))
from common import inputs
from common import rot, save, cpu_contacts
from supermimic.hoi import scene

__all__ = [
    "E",
    "P6",
    "ROOT",
    "inputs",
    "rot",
    "save",
    "cpu_contacts",
    "candidate",
    "manifest_for",
    "cpu_model",
    "gpu_env",
    "indices",
    "reference_qpos",
    "active_pairs",
    "dump_gz",
]


def candidate(name):
    return json.loads((E / "candidates" / (name + ".json")).read_text())


def manifest_for(name):
    m = copy.deepcopy(inputs()[0])
    c = candidate(name)
    m["object"]["geoms"] = c["geoms"]
    m["object"]["inertia"] = c["inertia"]
    return m


def cpu_model(name):
    """Fresh compilation rebuilds all collision bounds and inferred constants."""
    from mjlab.scene import Scene

    manifest = manifest_for(name)
    with patch.object(scene, "load_manifest", lambda _: manifest):
        cfg = scene.make_physical_env_cfg(scene.PhysicalSceneCfg(num_envs=1))
    s = Scene(cfg.scene, device="cpu")
    s.spec.option.timestep = 0.005
    s.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    s.spec.option.iterations = 10
    s.spec.option.ls_iterations = 20
    m = s.compile()
    cfg.sim.mujoco.apply(m)
    scene.validate_scene(m, manifest)
    return m


def gpu_env(name="original", num_envs=4, change=None):
    """Authorized diagnostic candidate construction; production loader untouched."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    import torch

    m = manifest_for(name)
    configure_torch_backends()
    with patch.object(scene, "load_manifest", lambda _: m):
        cfg = scene.make_physical_env_cfg(scene.PhysicalSceneCfg(num_envs=num_envs))
        if change is not None:
            change(cfg)
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
    scene.validate_scene(env.sim.mj_model, m)
    env.sim.expand_model_fields(("body_pos",))
    for s in m["supports"]:
        b = env.sim.mj_model.body(s["name"]).id
        g = env.sim.mj_model.geom(s["geom"]["name"]).id
        env.sim.model.body_pos[:, b] = (
            torch.tensor(s["root_pos_world"], device=env.device) + env.scene.env_origins
        )
        env.sim.data.geom_xpos[:, g] += env.scene.env_origins
    env.sim.forward()
    return env, m


def indices(m):
    boxj = m.body("box/bread_box").jntadr[0]
    bq, bv = int(m.jnt_qposadr[boxj]), int(m.jnt_dofadr[boxj])
    names = [
        m.joint(i).name.removeprefix("robot/")
        for i in range(m.njnt)
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    jq = np.array([m.jnt_qposadr[m.joint("robot/" + n).id] for n in names])
    jv = np.array([m.jnt_dofadr[m.joint("robot/" + n).id] for n in names])
    return bq, bv, names, jq, jv


def reference_qpos(m):
    clip = inputs()[1]
    a = clip.arrays
    bq, _, names, jq, _ = indices(m)
    q = np.tile(m.qpos0, (840, 1))
    pelvis = list(a["body_names"]).index("pelvis")
    q[:, :3] = a["body_pos_w"][:, pelvis]
    q[:, 3:7] = a["body_quat_w"][:, pelvis]
    order = [list(a["joint_names"]).index(n) for n in names]
    q[:, jq] = a["joint_pos"][:, order]
    q[:, bq : bq + 3] = a["object_pos_w"]
    q[:, bq + 3 : bq + 7] = a["object_quat_w"]
    return q


def active_pairs(m):
    box = m.body("box/bread_box").id
    obj = [i for i in range(m.ngeom) if m.geom_bodyid[i] == box and m.geom_contype[i]]
    robot = [i for i in range(m.ngeom) if m.geom(i).name.startswith("robot/") and m.geom_contype[i]]
    hands = [
        [m.geom("robot/" + s + "_" + p + "_collision").id for p in ("wrist", "hand")]
        for s in ("left", "right")
    ]
    non = [i for i in robot if i not in hands[0] + hands[1]]
    return obj, robot, hands, non


def dump_gz(path, value):
    with gzip.open(path, "wt") as f:
        json.dump(value, f, default=lambda x: x.tolist(), allow_nan=False)
