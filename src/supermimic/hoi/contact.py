"""Raw, pair-filtered hand-on-box sensing. No control-interval labels (P2)."""

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
