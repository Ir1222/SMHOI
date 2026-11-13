"""Diagnostic-only real-training evidence; no optimizer/sampling-policy changes."""

import json

import torch

from supermimic.hoi.training_safety import finite
from supermimic.kinematics.rotations import quat_conjugate, quat_mul, rotvec_from_quat


def append_json(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


@torch.no_grad()
def prediction_report(trainer, n=32):
    # Diagnostics must not consume the RNG used for future Replay/Actor updates.
    device = torch.device(trainer.device)
    with torch.random.fork_rng(devices=[device.index or 0]):
        batch = trainer.buffer.sample_windows(n, 8)
        truth = trainer.wm_trainer.physical_states(batch)
        learned = prior = truth[0]
        result = {"learned": {}, "zero_residual_prior": {}}
        for step in range(8):
            learned = trainer.wm.predict(learned, batch["a_exec"][step], batch["xi"])
            prior = trainer.wm.spec.integrate(prior, prior.new_zeros(n, 82))
            if step not in (0, 7):
                continue
            target = truth[step + 1]
            for name, state in (("learned", learned), ("zero_residual_prior", prior)):
                error = trainer.wm.spec.error_vector(state, target)
                prefix = f"{step + 1}step/"
                metrics = {
                    "robot_normalized_mse": error[:, :70].square().mean(),
                    "box_position_m": (state[:, 71:74] - target[:, 71:74]).norm(dim=-1).mean(),
                    "box_orientation_rad": rotvec_from_quat(
                        quat_mul(quat_conjugate(target[:, 74:78]), state[:, 74:78])
                    )
                    .norm(dim=-1)
                    .mean(),
                    "box_linear_velocity_m_s": (state[:, 78:81] - target[:, 78:81])
                    .norm(dim=-1)
                    .mean(),
                    "box_angular_velocity_rad_s": (state[:, 81:84] - target[:, 81:84])
                    .norm(dim=-1)
                    .mean(),
                }
                finite("prediction diagnostics", metrics)
                result[name].update({prefix + k: float(v) for k, v in metrics.items()})
    return {
        "iteration": trainer.iteration,
        "windows": n,
        "scope": "online Replay sample; fitting diagnostics, NOT held-out generalization",
        "errors": result,
    }


@torch.no_grad()
def reached_geometry_report(trainer):
    """Signed distances of actual reached states at log boundaries, before reset.

    CPU mj_forward only; sampling does not establish all-substep maximum/duration.
    Uses environment-local recorded states and the unchanged native model.
    """
    import mujoco

    model = trainer.env.raw.sim.mj_model
    data = mujoco.MjData(model)
    io = trainer.recorder.io
    rq, bq = io.robot_q.cpu().numpy(), io.box_q.cpu().numpy()
    rows = []
    for world, state in enumerate(trainer.recorder.last["next_state"].cpu().numpy()):
        data.qpos[rq], data.qpos[bq] = state[:36], state[71:78]
        mujoco.mj_forward(model, data)
        worst = min(data.contact, key=lambda c: c.dist, default=None)
        rows.append(
            {
                "world": world,
                "reached_frame": int(trainer.recorder.last["frame"][world]) + 1,
                "episode": int(trainer.recorder.last["episode"][world]),
                "min_distance_m": min(0.0, float(worst.dist)) if worst is not None else 0.0,
                "worst_pair": [model.geom(g).name for g in (worst.geom1, worst.geom2)]
                if worst is not None
                else [],
            }
        )
    return {
        "iteration": trainer.iteration,
        "scope": "actual reached states at log boundary, before auto-reset; CPU signed geometry; not every substep",
        "max_penetration_m": -min(r["min_distance_m"] for r in rows),
        "worlds": rows,
    }
