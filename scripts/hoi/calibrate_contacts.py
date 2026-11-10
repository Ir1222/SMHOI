"""P1 deterministic physical calibration; never trains or creates Replay labels.

Run with the project Python. Each --output directory must be new. Fixtures
reposition bodies deliberately to isolate sensing; they are not task rollouts.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import traceback
from pathlib import Path
from types import SimpleNamespace

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.types import ContactType
from scipy.spatial.transform import Rotation

from supermimic.hoi.contact import (
    HAND_BODIES,
    SENSOR_NAMES,
    assert_contact_capacity,
    force_box_local,
    hand_force_history_world,
    hand_forces_world,
)
from supermimic.hoi.reference import DEFAULT_ASSET, DEFAULT_OUTPUT, ReferenceClip, sha256
from supermimic.hoi.scene import PhysicalSceneCfg, make_physical_env
from supermimic.hoi.state import JointStateIO, ReferenceResetCache

NORMALS = {
    "bottom": [0, 0, -1],
    "side1": [-1, 0, 0],
    "side2": [1, 0, 0],
    "side3": [0, 1, 0],
    "side4": [0, -1, 0],
    "handle_left": [0, 1, 0],
    "handle_right": [0, -1, 0],
}


def raw_sample(env) -> dict:
    """Independent raw pair sums using solved Warp contact forces (all worlds)."""
    assert_contact_capacity(env)
    sim, model = env.sim, env.sim.mj_model
    count = int(sim.data.nacon[0])
    with wp.ScopedDevice(sim.wp_device):
        ids = wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32)
        force = wp.zeros(count, dtype=wp.spatial_vector)
        if count:
            mjwarp.contact_force(sim.wp_model, sim.wp_data, ids, True, force)
        forces = force.numpy()
    contact = sim.wp_data.contact
    geoms, worlds = contact.geom.numpy()[:count], contact.worldid.numpy()[:count]
    types = contact.type.numpy()[:count]
    distances = contact.dist.numpy()[:count]
    box = model.body("box/bread_box").id
    primary = [[model.body("robot/" + name).id for name in names] for names in HAND_BODIES.values()]
    direct = np.zeros((env.num_envs, 2, 2, 3))
    pairs = []
    for i, (g0, g1) in enumerate(geoms):
        if g0 < 0 or g1 < 0 or not (types[i] & int(ContactType.CONSTRAINT)):
            continue
        b0, b1 = int(model.geom_bodyid[g0]), int(model.geom_bodyid[g1])
        for side, bodies in enumerate(primary):
            for column, body in enumerate(bodies):
                if b0 == body and b1 == box:
                    direct[worlds[i], side, column] += forces[i, :3]
                elif b1 == body and b0 == box:
                    direct[worlds[i], side, column] -= forces[i, :3]
        pairs.append(
            {
                "env": int(worlds[i]),
                "geoms": [model.geom(g0).name, model.geom(g1).name],
                "distance_m": float(distances[i]),
                "force_on_geom1_world_N": forces[i, :3].tolist(),
            }
        )
    measured = np.stack([env.scene[n].data.force.cpu().numpy() for n in SENSOR_NAMES], axis=1)
    found = np.stack([env.scene[n].data.found.cpu().numpy() for n in SENSOR_NAMES], axis=1)
    np.testing.assert_allclose(measured, direct, atol=2e-3, rtol=2e-4)
    np.testing.assert_allclose(
        hand_forces_world(env.scene).cpu(), direct.sum(axis=2), atol=2e-3, rtol=2e-4
    )
    return {
        "force_columns_world_N": measured.tolist(),
        "found_columns": found.tolist(),
        "force_world_N": measured.sum(axis=2).tolist(),
        "contacts": pairs,
        "raw_sum_max_error_N": float(np.max(np.abs(measured - direct))),
        "overflow": sim.data.overflow[:].cpu().tolist(),
        "nacon": count,
        "nefc": sim.data.nefc[:].cpu().tolist(),
        "box_quat_at_solve_wxyz": sim.data.xquat[:, box].cpu().tolist(),
    }


class Placements:
    """Actual full native G1 and compound box, without collision-mask edits."""

    def __init__(self, env, io):
        self.model, self.io = env.sim.mj_model, io
        self.data = mujoco.MjData(self.model)
        rq, rv = io.robot_q.cpu().numpy(), io.robot_v.cpu().numpy()
        bq, bv = io.box_q.cpu().numpy(), io.box_v.cpu().numpy()
        self.indices = rq, rv, bq, bv
        self.base = np.r_[
            self.model.key_qpos[0, rq],
            np.zeros(35),
            [2.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            np.zeros(6),
        ]
        self.base[2] = 3.0

    def forward(self, state):
        rq, rv, bq, bv = self.indices
        self.data.qpos[rq], self.data.qvel[rv] = state[:36], state[36:71]
        self.data.qpos[bq], self.data.qvel[bv] = state[71:78], state[78:84]
        mujoco.mj_forward(self.model, self.data)

    def contact(self, side, part="hand", geom="side1", roll=0.0):
        state = self.base.copy()
        if part == "wrist":
            # Expose the pitch collider: at the default yaw pose the distal
            # hand can take the entire load from a redundant wrist constraint.
            yaw = self.io.joint_names.index(f"{side}_wrist_yaw_joint")
            state[7 + yaw] = -1.57 if side == "left" else 1.57
        self.forward(state)
        hand = self.model.geom(f"robot/{side}_{part}_collision").id
        axis = self.data.geom_xmat[hand].reshape(3, 3)[:, 2]
        radius, half = self.model.geom_size[hand, :2]
        tip = self.data.geom_xpos[hand] + axis * half * np.sign(axis[0]) + [radius, 0, 0]
        normal = np.array(NORMALS[geom])
        rotation, _ = Rotation.align_vectors([[-1, 0, 0]], [normal])
        rotation = Rotation.from_rotvec([roll, 0, 0]) * rotation
        gid = self.model.geom("box/" + geom).id
        face = self.model.geom_pos[gid] + normal * self.model.geom_size[gid]
        state[71:74] = tip - np.array([0.003, 0, 0]) - rotation.apply(face)
        state[74:78] = rotation.as_quat(scalar_first=True)
        return state

    def hand_ground(self):
        state = self.base.copy()
        self.forward(state)
        gid = self.model.geom("robot/left_hand_collision").id
        axis = self.data.geom_xmat[gid].reshape(3, 3)[:, 2]
        radius, half = self.model.geom_size[gid, :2]
        bottom = self.data.geom_xpos[gid, 2] - abs(axis[2]) * half - radius
        state[2] -= bottom + 0.003
        return state

    def box_support(self):
        state = self.base.copy()
        support = json.loads(DEFAULT_ASSET.read_text())["supports"][0]
        state[71:74] = np.array(support["root_pos_world"]) + [0, 0, 0.028]
        state[74:78] = support["root_quat_wxyz"]
        return state


def tensor_states(env, states):
    return torch.tensor(np.asarray(states), device=env.device, dtype=torch.float32)


def reset_states(env, io, states):
    env.reset()
    io.write(torch.arange(env.num_envs, device=env.device), tensor_states(env, states))


def one_substep(env):
    env.action_manager.process_action(torch.zeros(env.num_envs, 29, device=env.device))
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    env.sim.step()
    env.scene.update(env.physics_dt)
    return raw_sample(env)


def active(sample):
    return np.linalg.norm(sample["force_world_N"], axis=-1) > 1e-3


def fixed_robot_pushes(env, placements):
    """Weld complete native robot at its default pose in a disposable fixture.

    Bake FK transforms, remove robot joints/actuators ONLY from the copy. Keep
    every geom, sensor filter, box inertial parameter and support. Zero gravity
    isolates force/momentum; production scene and gains remain unchanged.
    """
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.sim.sim import Simulation

    placements.forward(placements.base)
    original, data = placements.model, placements.data
    spec = env.scene.spec.copy()
    for body in spec.bodies:
        if not body.name.startswith("robot/"):
            continue
        bid = original.body(body.name).id
        parent = int(original.body_parentid[bid])
        parent_rot = data.xmat[parent].reshape(3, 3)
        body.pos = parent_rot.T @ (data.xpos[bid] - data.xpos[parent])
        body.quat = Rotation.from_matrix(parent_rot.T @ data.xmat[bid].reshape(3, 3)).as_quat(
            scalar_first=True
        )
    for element in list(spec.actuators) + list(spec.keys):
        spec.delete(element)
    for joint in list(spec.joints):
        if joint.name.startswith("robot/"):
            spec.delete(joint)
    cfg = SimulationCfg(
        nconmax=256,
        njmax=1024,
        contact_sensor_maxmatch=64,
        mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, gravity=(0.0, 0.0, 0.0)),
    )
    sim = Simulation(num_envs=4, cfg=cfg, spec=spec, device=env.device)
    sensors = {name: env.scene[name].cfg.build() for name in SENSOR_NAMES}
    # edit_spec resolves slots on an expendable copy; original compiled sensors
    # already have the identical names and addresses in sim.mj_model.
    probe_spec = env.scene.spec.copy()
    for sensor in list(probe_spec.sensors):
        if sensor.name.startswith(SENSOR_NAMES):
            probe_spec.delete(sensor)
    for sensor in sensors.values():
        sensor.edit_spec(probe_spec, env.scene.entities)
        sensor.initialize(sim.mj_model, sim.model, sim.data, env.device)
    fixture = SimpleNamespace(sim=sim, scene=sensors, num_envs=4, cfg=SimpleNamespace(sim=cfg))
    states = [
        placements.contact(side, roll=roll)
        for side, roll in (("left", 0.0), ("right", 0.0), ("left", 0.63), ("right", 0.63))
    ]
    sim.reset()
    sim.data.qpos[:] = tensor_states(env, states)[:, 71:78]
    sim.data.qvel[:] = 0
    sim.data.qacc_warmstart[:] = 0
    sim.step()
    for sensor in sensors.values():
        sensor.update(0.005)
    result = raw_sample(fixture)
    forces = np.asarray(result["force_world_N"])
    qvel = sim.data.qvel[:].clone()
    quat = sim.data.qpos[:, 3:7].clone()
    from supermimic.kinematics.rotations import quat_apply

    omega = quat_apply(quat, qvel[:, 3:6])
    offset = torch.tensor(
        original.body_ipos[original.body("box/bread_box").id],
        device=env.device,
        dtype=torch.float32,
    ).expand(4, 3)
    com_velocity = qvel[:, :3] + torch.cross(omega, quat_apply(quat, offset), dim=-1)
    chosen = np.array([forces[i, i % 2] for i in range(4)])
    result.update(
        {
            "fixture": "full native robot welded at FK pose; gravity zero; box dynamic",
            "box_com_velocity_world_m_s": com_velocity.cpu().tolist(),
            "box_rotated_radians_about_world_x": [0.0, 0.0, 0.63, 0.63],
            "expected_push_direction_world": [1.0, 0.0, 0.0],
        }
    )
    assert np.all(chosen[:, 0] > 1e-3)
    assert bool(torch.all(com_velocity[:, 0] > 0))
    # Newton's law independently validates sign AND magnitude at the COM.
    expected_velocity = chosen * 0.005 / 2.0
    np.testing.assert_allclose(com_velocity.cpu(), expected_velocity, atol=2e-4, rtol=0.03)
    result["momentum_max_error_m_s"] = float(
        np.max(np.abs(com_velocity.cpu().numpy() - expected_velocity))
    )
    world = torch.tensor(forces, device=env.device, dtype=torch.float32)
    qsolve = torch.tensor(result["box_quat_at_solve_wxyz"], device=env.device)
    local = force_box_local(world, qsolve[:, None])
    roundtrip = quat_apply(qsolve[:, None].expand(4, 2, 4), local)
    torch.testing.assert_close(roundtrip, world, atol=2e-5, rtol=2e-6)
    result["force_box_local_N"] = local.cpu().tolist()
    result["frame_roundtrip_max_error_N"] = float((roundtrip - world).abs().max())
    return result


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "checks": {},
        "scenarios": {},
        "provenance": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "reference_sha256": sha256(DEFAULT_OUTPUT / "reference.npz"),
            "asset_manifest_sha256": sha256(DEFAULT_ASSET),
            "versions": {
                k: importlib.metadata.version(k)
                for k in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch")
            },
            "physics_dt_s": 0.005,
            "decimation": 4,
            "production_capacities": {"nconmax": 256, "njmax": 1024, "maxmatch": 64},
            "semantics": "world hand-on-box N; found is current constraint count; history newest-first",
        },
    }
    wp.config.kernel_cache_dir = "/tmp/smhoi_p1_warp_cache"
    env = make_physical_env(PhysicalSceneCfg(num_envs=4))
    io = JointStateIO(env)
    placements = Placements(env, io)
    cache = ReferenceResetCache(
        ReferenceClip(DEFAULT_OUTPUT / "reference.npz"), io.joint_names, env.device
    )

    def save():
        (output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")

    def check(name, fn):
        try:
            report["scenarios"][name] = fn()
            report["checks"][name] = "PASS"
        except Exception as error:  # noqa: BLE001 -- retain every failed calibration in evidence
            report["checks"][name] = "FAIL"
            report["scenarios"][name] = {
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "partial_evidence": report["scenarios"].get(name),
            }
        save()

    try:

        def coverage():
            rows = []
            report["scenarios"]["all_seven_geoms_both_hands_both_primaries"] = rows
            for side in HAND_BODIES:
                for part in ("wrist", "hand"):
                    for geom in NORMALS:
                        state = placements.contact(side, part, geom)
                        reset_states(env, io, [state] * 4)
                        sample = one_substep(env)
                        rows.append(
                            {
                                "side": side,
                                "part": part,
                                "target_geom": geom,
                                "initial_state84": state.tolist(),
                                "sample": sample,
                            }
                        )
                        expected = np.array([side == "left", side == "right"])
                        assert np.all(active(sample) == expected)
                        assert (
                            np.max(
                                np.linalg.norm(
                                    np.asarray(sample["force_world_N"])[:, int(side == "left")],
                                    axis=-1,
                                )
                            )
                            <= 1e-6
                        )
                        pair = {f"robot/{side}_{part}_collision", f"box/{geom}"}
                        matching = [c for c in sample["contacts"] if set(c["geoms"]) == pair]
                        assert {c["env"] for c in matching} == set(range(4))
                        loads = [
                            sum(
                                np.linalg.norm(c["force_on_geom1_world_N"])
                                for c in matching
                                if c["env"] == world
                            )
                            for world in range(4)
                        ]
                        assert min(loads) > 1e-3, (side, part, geom, loads)
            return rows

        check("all_seven_geoms_both_hands_both_primaries", coverage)

        bilateral = cache.states[260].cpu().numpy().copy()
        bilateral[36:71] = 0
        bilateral[78:84] = 0
        states = [
            placements.base,
            placements.contact("left"),
            placements.contact("right"),
            bilateral,
        ]

        def mixed():
            reset_states(env, io, states)
            sample = one_substep(env)
            np.testing.assert_array_equal(
                active(sample), [[False, False], [True, False], [False, True], [True, True]]
            )
            return sample

        check("no_left_right_bilateral_parallel", mixed)

        def exclusions():
            reset_states(
                env,
                io,
                [
                    placements.hand_ground(),
                    placements.box_support(),
                    placements.base,
                    placements.base,
                ],
            )
            sample = one_substep(env)
            report["scenarios"]["hand_ground_and_box_support_excluded"] = sample
            assert np.max(np.abs(sample["force_world_N"])) <= 1e-6
            assert any(
                c["env"] == 0
                and "robot/left_hand_collision" in c["geoms"]
                and any("terrain" in g for g in c["geoms"])
                for c in sample["contacts"]
            )
            assert any(
                c["env"] == 1
                and any(g.startswith("box/") for g in c["geoms"])
                and any("support" in g for g in c["geoms"])
                for c in sample["contacts"]
            )
            return sample

        check("hand_ground_and_box_support_excluded", exclusions)
        check("fixed_robot_known_and_rotated_pushes", lambda: fixed_robot_pushes(env, placements))

        def interval(pattern):
            contact = placements.contact("left")
            apart = contact.copy()
            apart[71] += 3.0
            reset_states(env, io, [contact if pattern[0] else apart] * 4)
            samples = []
            original = env.metrics_manager.compute_substep

            def capture():
                original()
                samples.append(raw_sample(env))
                if len(samples) < 4:
                    target = contact if pattern[len(samples)] else apart
                    io.write(
                        torch.arange(4, device=env.device),
                        tensor_states(env, [target] * 4),
                        forward=False,
                    )

            env.metrics_manager.compute_substep = capture
            start_time = env.sim.data.time[:].clone()
            try:
                env.step(torch.zeros(4, 29, device=env.device))
            finally:
                env.metrics_manager.compute_substep = original
            assert len(samples) == 4
            np.testing.assert_array_equal([active(s)[0, 0] for s in samples], pattern)
            history = hand_force_history_world(env.scene)
            expected = np.stack([s["force_world_N"] for s in samples[::-1]], axis=2)
            np.testing.assert_allclose(history.cpu(), expected, atol=0, rtol=0)
            torch.testing.assert_close(
                env.sim.data.time[:] - start_time,
                torch.full((4,), 0.020, device=env.device),
                atol=2e-7,
                rtol=0,
            )
            copied = history.clone()
            pointers = [env.scene[n].data.force_history.data_ptr() for n in SENSOR_NAMES]
            env.sim.forward()
            torch.testing.assert_close(hand_force_history_world(env.scene), copied, atol=0, rtol=0)
            one_substep(env)
            assert all(
                env.scene[n].data.force_history.data_ptr() != p
                for n, p in zip(SENSOR_NAMES, pointers)
            )
            torch.testing.assert_close(history, copied, atol=0, rtol=0)
            return {
                "pattern": pattern,
                "samples": samples,
                "history_N": copied.cpu().tolist(),
                "four_substeps": True,
                "no_fifth_forward_sample": True,
                "retained_copy_stable": True,
            }

        for name, pattern in (
            ("substep1_only", [1, 0, 0, 0]),
            ("substep4_only", [0, 0, 0, 1]),
            ("contact_ends_before_final", [1, 1, 0, 0]),
        ):
            check(name, lambda pattern=pattern: interval(pattern))

        def reset_isolation():
            reset_states(env, io, states)
            one_substep(env)
            before = hand_force_history_world(env.scene)
            state_before = io.read()
            io.reset_to_reference(
                torch.tensor([1], device=env.device), torch.tensor([300], device=env.device), cache
            )
            after = hand_force_history_world(env.scene)
            assert not after[1].any()
            keep = [0, 2, 3]
            torch.testing.assert_close(after[keep], before[keep], atol=0, rtol=0)
            torch.testing.assert_close(io.read()[keep], state_before[keep], atol=0, rtol=0)
            return {
                "before_N": before.cpu().tolist(),
                "after_N": after.cpu().tolist(),
                "reset_env": 1,
            }

        check("reset_immediately_after_contact_isolation", reset_isolation)

        def parity():
            values = []
            for graph in (True, False):
                env.sim.use_cuda_graph = graph
                reset_states(env, io, states)
                sample = one_substep(env)
                values.append((io.read().clone(), hand_force_history_world(env.scene), sample))
            env.sim.use_cuda_graph = True
            for index in (0, 1):
                torch.testing.assert_close(values[0][index], values[1][index], atol=2e-6, rtol=2e-5)
            return {
                "state_max_error": float((values[0][0] - values[1][0]).abs().max()),
                "history_max_error_N": float((values[0][1] - values[1][1]).abs().max()),
                "captured": values[0][2],
                "eager": values[1][2],
            }

        check("captured_eager_step_reset_parity", parity)

        def overflow():
            stress = placements.contact("left", "wrist", "handle_left")
            yaw = 7 + io.joint_names.index("left_wrist_yaw_joint")
            # Restore the default distal-hand pose for simultaneous compound
            # contacts; coverage's exposed-wrist pose has only one match.
            stress[yaw] = placements.base[yaw]
            reset_states(env, io, [stress] * 4)
            # Capacity option is consumed dynamically by the matching kernel.
            # Keep allocated arrays, but intentionally cap matches at one.
            old = env.sim.wp_model.opt.contact_sensor_maxmatch
            old_graph = env.sim.use_cuda_graph
            env.sim.use_cuda_graph = False
            env.sim.wp_model.opt.contact_sensor_maxmatch = 1
            try:
                env.sim.step()
                env.scene.update(0.005)
                flags = env.sim.data.overflow[:].cpu().tolist()
                try:
                    assert_contact_capacity(env)
                except RuntimeError as error:
                    assert any(f & int(mjwarp.OverflowType.CONTACT_MATCH) for f in flags)
                    return {"deliberate_maxmatch": 1, "flags": flags, "caught": str(error)}
                raise AssertionError("Deliberately overflowing match capacity was not rejected")
            finally:
                env.sim.wp_model.opt.contact_sensor_maxmatch = old
                env.sim.use_cuda_graph = old_graph
                env.reset()

        check("deliberate_overflow_rejected", overflow)
    finally:
        env.close()
        report["status"] = (
            "PASS"
            if report["checks"] and all(v == "PASS" for v in report["checks"].values())
            else "FAIL"
        )
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
