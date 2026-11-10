"""P1 scene and state contracts, including a real CUDA reset/step regression."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.scene import Scene

from supermimic.envs.state_io import StateIO
from supermimic.hoi.asset_manifest import load_manifest
from supermimic.hoi.reference import DEFAULT_ASSET, DEFAULT_OUTPUT, ReferenceClip
from supermimic.hoi.scene import (
    PhysicalSceneCfg,
    make_physical_env,
    make_physical_env_cfg,
    validate_scene,
)
from supermimic.hoi.state import JointStateIO, ReferenceResetCache, reference_object_twist
from supermimic.kinematics.rotations import quat_apply, quat_from_rotvec, quat_mul


@pytest.fixture(scope="module")
def compiled():
    cfg = make_physical_env_cfg(PhysicalSceneCfg(num_envs=2))
    # Deliberately reverse entity order: named indexing must still preserve robot71.
    cfg.scene.entities = dict(reversed(list(cfg.scene.entities.items())))
    scene = Scene(cfg.scene, "cpu")
    return cfg, scene, scene.compile()


def test_manifest_geometry_native_actions_and_static_supports(compiled):
    cfg, _scene, model = compiled
    manifest = load_manifest(DEFAULT_ASSET)
    validate_scene(model, manifest)
    box = model.body("box/" + manifest["object"]["name"]).id
    inertia = manifest["object"]["inertia"]
    assert model.body_mass[box] == inertia["mass_kg"]
    np.testing.assert_allclose(model.body_ipos[box], inertia["com_pos_root_m"], atol=1e-10)
    np.testing.assert_allclose(model.body_inertia[box], inertia["diagonal_kg_m2"], atol=1e-10)
    for geom in manifest["object"]["geoms"]:
        gid = model.geom("box/" + geom["name"]).id
        for field in ("pos", "quat", "size", "friction", "solref", "solimp"):
            np.testing.assert_allclose(
                getattr(model, "geom_" + field)[gid], geom[field], atol=1e-10
            )
        for field in ("contype", "conaffinity", "condim", "priority", "group"):
            assert getattr(model, "geom_" + field)[gid] == geom[field]
    for support in manifest["supports"]:
        bid = model.body(support["name"]).id
        np.testing.assert_allclose(model.body_pos[bid], support["root_pos_world"], atol=1e-10)
        np.testing.assert_allclose(model.body_quat[bid], support["root_quat_wxyz"], atol=1e-10)
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg

    original = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], action_noise_std=0))
    assert cfg.actions == original.actions
    native = Scene(original.scene, "cpu").compile()
    for field in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(model, field), getattr(native, field))
    assert cfg.sim.mujoco.timestep == 0.005 and cfg.decimation == 4
    assert not cfg.recorders and not cfg.commands and not cfg.rewards


def test_named_state_nonzero_origins_and_com_roundtrip(compiled):
    _, scene, model = compiled
    origins = torch.tensor([[3.0, -2.0, 0.0], [-4.0, 5.0, 0.0]])
    data = SimpleNamespace(
        qpos=torch.zeros(2, model.nq),
        qvel=torch.zeros(2, model.nv),
        qacc_warmstart=torch.ones(2, model.nv),
    )
    sim = SimpleNamespace(
        mj_model=model,
        data=data,
        model=SimpleNamespace(body_ipos=torch.tensor(model.body_ipos, dtype=torch.float32)[None]),
    )
    scene.terrain.env_origins = origins
    env = SimpleNamespace(sim=sim, scene=scene, device="cpu")
    io = JointStateIO(env)
    assert int(io.box_q[0]) < int(io.robot_q[0])  # ordering trap exercised
    clip = ReferenceClip(DEFAULT_OUTPUT / "reference.npz")
    cache = ReferenceResetCache(clip, io.joint_names, "cpu")
    state = cache.frames(torch.tensor([300, 500]))
    state[:, 74:78] = quat_from_rotvec(torch.tensor([[0.4, -0.3, 0.7], [-0.2, 0.3, 0.1]]))
    state[:, 81:84] = torch.tensor([[0.7, -0.4, 0.2], [0.3, 0.2, -0.8]])
    io.write(torch.arange(2), state, forward=False)
    torch.testing.assert_close(io.read(), state, atol=2e-6, rtol=0)
    for row in range(2):
        d = mujoco.MjData(model)
        d.qpos[:] = data.qpos[row].numpy()
        d.qvel[:] = data.qvel[row].numpy()
        mujoco.mj_forward(model, d)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, io.box_body_id, vel, 0)
        np.testing.assert_allclose(
            io.object_com_velocity_world()[row].numpy(), np.r_[vel[3:], vel[:3]], atol=2e-6
        )
        root_from_com = vel[3:] - np.cross(
            vel[:3], d.xipos[io.box_body_id] - d.xpos[io.box_body_id]
        )
        np.testing.assert_allclose(root_from_com, state[row, 78:81].numpy(), atol=2e-6)
    before = io.read().clone()
    changed = before[1:].clone()
    changed[:, 0] += 0.7
    io.write(torch.tensor([1]), changed, forward=False)
    torch.testing.assert_close(io.read()[0], before[0], atol=0, rtol=0)
    with pytest.raises(AssertionError):
        StateIO(env)  # unchanged original guard deliberately still rejects joint scenes


def test_reference_twist_endpoints_rotation_and_sign_flips():
    t = torch.arange(12, dtype=torch.float64) * 0.02
    v = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    w = torch.tensor([0.4, 0.2, -0.5], dtype=torch.float64)
    initial = quat_from_rotvec(torch.tensor([0.7, -0.3, 0.1], dtype=torch.float64))
    quat = quat_mul(initial.expand(12, 4), quat_from_rotvec(t[:, None] * w))
    quat[::2] *= -1
    twist = reference_object_twist(t[:, None] * v, quat, 0.02)
    torch.testing.assert_close(twist[:, :3], v.expand(12, 3), atol=1e-6, rtol=0)
    torch.testing.assert_close(twist[:, 3:], w.expand(12, 3), atol=1e-6, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="P1 physical reset requires CUDA")
def test_cuda_synchronized_reset_and_fixed_supports():
    import warp as wp

    wp.config.kernel_cache_dir = "/tmp/smhoi_p1_warp_cache"
    env = make_physical_env(PhysicalSceneCfg(num_envs=4))
    try:
        io = JointStateIO(env)
        cache = ReferenceResetCache(
            ReferenceClip(DEFAULT_OUTPUT / "reference.npz"), io.joint_names, env.device
        )
        ids = torch.arange(4, device=env.device)
        frames = torch.tensor([0, 300, 500, 839], device=env.device)
        io.reset_to_reference(ids, frames, cache)
        torch.testing.assert_close(io.read(), cache.frames(frames), atol=2e-6, rtol=0)
        assert io.read()[1, 78:84].norm() > 0.1
        support_ids = [
            env.sim.mj_model.body(s["name"]).id for s in load_manifest(DEFAULT_ASSET)["supports"]
        ]
        positions = env.sim.data.xpos[:, support_ids].clone()
        geom_ids = [
            env.sim.mj_model.geom(s["geom"]["name"]).id
            for s in load_manifest(DEFAULT_ASSET)["supports"]
        ]
        geom_positions = env.sim.data.geom_xpos[:, geom_ids].clone()
        torch.testing.assert_close(geom_positions, positions, atol=2e-6, rtol=0)
        manifest_pos = torch.tensor(
            [s["root_pos_world"] for s in load_manifest(DEFAULT_ASSET)["supports"]],
            device=env.device,
        )
        torch.testing.assert_close(
            positions - env.scene.env_origins[:, None],
            manifest_pos.expand(4, 2, 3),
            atol=2e-6,
            rtol=0,
        )
        obj = io.read()[:, 71:]
        torch.testing.assert_close(
            env.scene["box"].data.root_com_lin_vel_w,
            io.object_com_velocity_world()[:, :3],
            atol=2e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            env.scene["box"].data.root_link_ang_vel_w,
            quat_apply(obj[:, 3:7], obj[:, 10:13]),
            atol=2e-6,
            rtol=0,
        )
        env.step(torch.zeros(4, 29, device=env.device))
        before = io.read().clone()
        history = env.scene["left_hand_box"].data.force_history.clone()
        io.reset_to_reference(ids[:1], frames[:1], cache)
        torch.testing.assert_close(io.read()[1:], before[1:], atol=0, rtol=0)
        assert not env.scene["left_hand_box"].data.force_history[0].any()
        torch.testing.assert_close(env.scene["left_hand_box"].data.force_history[1:], history[1:])
        torch.testing.assert_close(env.sim.data.xpos[:, support_ids], positions, atol=0, rtol=0)
        torch.testing.assert_close(
            env.sim.data.geom_xpos[:, geom_ids], geom_positions, atol=0, rtol=0
        )
        assert env.sim.use_cuda_graph and env.sim.step_graph is not None
    finally:
        env.close()
