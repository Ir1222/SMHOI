"""Pair-filtered hand-on-box sensing and four-substep physical labels."""

from __future__ import annotations

import torch
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from supermimic.kinematics.rotations import quat_apply_inverse

HAND_BODIES = {
    side: (f"{side}_wrist_pitch_link", f"{side}_wrist_yaw_link") for side in ("left", "right")
}
SENSOR_NAMES = ("left_hand_box", "right_hand_box")


def hand_box_sensor_cfgs(box_body: str) -> tuple[ContactSensorCfg, ...]:
    return tuple(
        ContactSensorCfg(
            name=f"{side}_hand_box",
            primary=ContactMatch(mode="body", pattern=bodies, entity="robot"),
            secondary=ContactMatch(mode="body", pattern=box_body, entity="box"),
            secondary_policy="error",
            reduce="netforce",
            num_slots=1,
            history_length=4,
            fields=("found", "force"),
        )
        for side, bodies in HAND_BODIES.items()
    )


def hand_forces_world(scene) -> torch.Tensor:
    """Copy current net force [env, left/right, xyz], in N, hand on box.

    Sums the two simultaneous primary-body columns, never history/time. Read
    immediately after scene.update for an unambiguous solved-substep sample;
    a final forward can refresh native instantaneous data without history update.
    """
    return torch.stack([scene[name].data.force.sum(dim=1) for name in SENSOR_NAMES], dim=1)


def hand_force_history_world(scene) -> torch.Tensor:
    """Copy [env, left/right, newest-first substep, xyz], in N.

    Reacquires replaced native buffers on every call. No found history exists;
    these net vectors cannot reconstruct all geometric contacts or local loads.
    """
    return torch.stack([scene[name].data.force_history.sum(dim=1) for name in SENSOR_NAMES], dim=1)


def force_box_local(force_world: torch.Tensor, box_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Rotate same-sample vectors into Box axes; no sign change or time reduction.

    Caller supplies quaternion dimensions broadcastable to the force batch.
    """
    batch = torch.broadcast_shapes(box_quat_wxyz.shape[:-1], force_world.shape[:-1])
    quat = box_quat_wxyz.expand(*batch, 4)
    return quat_apply_inverse(quat, force_world.expand(*batch, 3))


def assert_contact_capacity(env) -> None:
    """Fail closed on any Warp overflow, saturated buffers or invalid force.

    Calibration calls this after EVERY physics solve, before a later forward
    resets overflow bits. This host synchronization is intentional in P1.
    """
    overflow = env.sim.data.overflow[:]
    if torch.any(overflow != 0):
        raise RuntimeError(f"MuJoCo Warp overflow flags: {overflow.cpu().tolist()}")
    if torch.any(env.sim.data.nefc[:] > env.cfg.sim.njmax):
        raise RuntimeError("MuJoCo constraint capacity exceeded")
    if int(env.sim.data.nacon[0]) > env.sim.wp_data.naconmax:
        raise RuntimeError("MuJoCo shared contact capacity exceeded")
    for name in SENSOR_NAMES:
        data = env.scene[name].data
        if torch.any(data.found > env.cfg.sim.contact_sensor_maxmatch):
            raise RuntimeError(f"Contact match capacity exceeded: {name}")
        if not torch.isfinite(data.force).all():
            raise RuntimeError(f"Nonfinite contact force: {name}")


def capacity_metric(env) -> torch.Tensor:
    """Existing mjlab substep metric hook; checks without replacing its loop."""
    assert_contact_capacity(env)
    return torch.zeros(env.num_envs, device=env.device)


def aggregate_interval(raw_world, solve_quat, found, *, force_on):
    """Four chronological solved samples; output physical units, hand order L/R.

    raw_world: [E,4,2,2,3], solve_quat: [E,4,4] wxyz,
    found: [E,4,2,2]. Found counts are separately captured, not force-derived.
    """
    n = raw_world.shape[0]
    if (
        raw_world.shape != (n, 4, 2, 2, 3)
        or solve_quat.shape != (n, 4, 4)
        or found.shape != (n, 4, 2, 2)
    ):
        raise ValueError("Expected four substeps, two hands and two primary bodies")
    if not 0 < force_on < float("inf"):
        raise ValueError("force_on must be an explicit positive finite threshold in N")
    if not all(torch.isfinite(x).all() for x in (raw_world, solve_quat, found)):
        raise ValueError("Nonfinite interval measurement")
    if torch.any((solve_quat.norm(dim=-1) - 1).abs() > 1e-5):
        raise ValueError("Expected unit solve-time Box quaternion")
    if torch.any(found < 0) or torch.any(found != found.round()):
        raise ValueError("Found must contain nonnegative constraint counts")
    world = raw_world.sum(dim=3)
    local = force_box_local(world, solve_quat[:, :, None])
    magnitude = world.norm(dim=-1)
    return {
        "force_mean_box": local.mean(dim=1),
        "impulse_world": 0.005 * world.sum(dim=1),
        "force_peak": magnitude.amax(dim=1),
        "occupancy": (magnitude > force_on).to(world.dtype).mean(dim=1),
        "found_final": found[:, -1].sum(dim=-1).clone(),
        "force_raw_world": raw_world.clone(),
        "box_quat_solve": solve_quat.clone(),
        "found_substeps": found.clone(),
    }


def torque_at_point(force, torque, centroid, point):
    """Combine primary-column wrenches about one common WORLD point, N m.

    Column dimension is -2. A rotation alone does not change a torque's origin.
    """
    return (torque + torch.cross(centroid - point.unsqueeze(-2), force, dim=-1)).sum(dim=-2)


class IntervalCapture:
    """Copy solved samples in the existing per-substep metric callback.

    xquat deliberately lags integrated qpos here: it is exactly the orientation
    used to solve the force. No dependency buffers or graph pointers are retained.
    """

    def __init__(self, env, state_io, force_on):
        if env.cfg.decimation != 4 or abs(env.physics_dt - 0.005) > 1e-12:
            raise ValueError("P2 requires .005 s x 4")
        self.env, self.io, self.force_on = env, state_io, force_on
        n, device = env.num_envs, env.device
        self.raw = torch.zeros(n, 4, 2, 2, 3, device=device)
        self.quat = torch.zeros(n, 4, 4, device=device)
        self.found = torch.zeros(n, 4, 2, 2, device=device)
        self.count = torch.zeros(n, dtype=torch.long, device=device)
        self.next_state = torch.zeros(n, 84, device=device)

    def sample(self):
        env = self.env
        assert_contact_capacity(env)
        if torch.any(self.count >= 4):
            raise RuntimeError("Unconsumed interval or extra substep")
        e = torch.arange(env.num_envs, device=env.device)
        k = self.count
        self.raw[e, k] = torch.stack([env.scene[name].data.force for name in SENSOR_NAMES], dim=1)
        self.found[e, k] = torch.stack([env.scene[name].data.found for name in SENSOR_NAMES], dim=1)
        self.quat[e, k] = env.sim.data.xquat[:, self.io.box_body_id]
        self.next_state.copy_(self.io.read())
        self.count += 1

    def consume(self, ids):
        if torch.any(self.count[ids] != 4):
            raise RuntimeError("Incomplete or stale four-substep interval")
        # Reacquire native history; its storage is replaced on every update.
        history = hand_force_history_world(self.env.scene)[ids].permute(0, 2, 1, 3).flip(1)
        if not torch.equal(history, self.raw[ids].sum(dim=3)):
            raise RuntimeError("Native history and captured action interval disagree")
        result = aggregate_interval(
            self.raw[ids], self.quat[ids], self.found[ids], force_on=self.force_on
        )
        self.count[ids] = 0
        return result

    def reset(self, ids):
        self.count[ids] = 0
        self.raw[ids] = 0
        self.quat[ids] = 0
        self.found[ids] = 0
