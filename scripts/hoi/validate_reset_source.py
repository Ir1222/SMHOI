"""Independent named-NPZ reset reconstruction against the actual training scene."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.hoi.reference import ReferenceClip
from supermimic.hoi.state import JointStateIO
from supermimic.hoi.training import HOITrainer
from supermimic.hoi.training_config import HOITrainConfig
from supermimic.hoi.training_safety import SolverGuard


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = HOITrainConfig(log_root=args.output.parent, run_name=args.output.name, check_only=True)
    cfg.env.num_envs = 2
    with SolverGuard() as guard:
        t = HOITrainer(cfg, guard)
        try:
            env, cache = t.env.raw, t.recorder.cache
            m, a = env.sim.mj_model, ReferenceClip(cfg.reference).arrays
            io = JointStateIO(env)
            d, independent = mujoco.MjData(m), mujoco.MjData(m)
            root = a["body_names"].tolist().index("pelvis")
            rb = m.body("robot/pelvis").id
            bb = io.box_body_id
            rq = int(m.jnt_qposadr[m.body_jntadr[rb]])
            bq = int(m.jnt_qposadr[m.body_jntadr[bb]])
            joints = [int(m.jnt_qposadr[m.joint("robot/" + n).id]) for n in a["joint_names"]]
            bodies = [m.body("robot/" + n).id for n in a["body_names"]]
            max_q, max_fk, max_dist = 0.0, 0.0, 0.0
            pairs, rows = {}, []
            for f in range(len(a["joint_pos"])):
                # This path does not use ReferenceResetCache to assign poses.
                independent.qpos[rq : rq + 3] = a["body_pos_w"][f, root]
                independent.qpos[rq + 3 : rq + 7] = a["body_quat_w"][f, root]
                independent.qpos[joints] = a["joint_pos"][f]
                independent.qpos[bq : bq + 3] = a["object_pos_w"][f]
                independent.qpos[bq + 3 : bq + 7] = a["object_quat_w"][f]
                mujoco.mj_forward(m, independent)
                state = cache.states[f].cpu().numpy()
                d.qpos[io.robot_q.cpu().numpy()] = state[:36]
                d.qpos[io.box_q.cpu().numpy()] = state[71:78]
                mujoco.mj_forward(m, d)
                max_q = max(max_q, float(np.abs(d.qpos - independent.qpos).max()))
                max_fk = max(
                    max_fk,
                    float(
                        np.linalg.norm(independent.xpos[bodies] - a["body_pos_w"][f], axis=-1).max()
                    ),
                )
                assert d.ncon == independent.ncon
                max_dist = max(
                    max_dist,
                    max(
                        (abs(x.dist - y.dist) for x, y in zip(d.contact, independent.contact)),
                        default=0,
                    ),
                )
                for c in independent.contact:
                    if c.dist >= 0:
                        continue
                    key = " / ".join(sorted([m.geom(c.geom1).name, m.geom(c.geom2).name]))
                    entry = pairs.setdefault(
                        key, {"minimum_distance_m": 0.0, "worst_frame": 0, "frames": set()}
                    )
                    entry["frames"].add(f)
                    if c.dist < entry["minimum_distance_m"]:
                        entry.update(minimum_distance_m=float(c.dist), worst_frame=f)
                if f in (0, 406, 460, 839):
                    ids = torch.arange(2, device=env.device)
                    target = cache.states[f].expand(2, -1)
                    io.write(ids, target)
                    error = float((io.read() - target).abs().max())
                    details = []
                    for c in independent.contact:
                        if c.dist < 0:
                            details.append(
                                {
                                    "distance_m": float(c.dist),
                                    "geoms": [m.geom(g).name for g in (c.geom1, c.geom2)],
                                    "geom_world_positions": [
                                        independent.geom_xpos[g].tolist()
                                        for g in (c.geom1, c.geom2)
                                    ],
                                    "geom_size": [
                                        m.geom_size[g].tolist() for g in (c.geom1, c.geom2)
                                    ],
                                    "geom_type": [int(m.geom_type[g]) for g in (c.geom1, c.geom2)],
                                }
                            )
                    rows.append(
                        {
                            "frame": f,
                            "two_world_state_roundtrip_max_abs": error,
                            "contacts": details,
                        }
                    )
            for entry in pairs.values():
                entry["frame_count"] = len(entry.pop("frames"))
            assert max_q == 0 and max_dist == 0 and max_fk < 2e-6
            assert max(r["two_world_state_roundtrip_max_abs"] for r in rows) < 1e-6
            result = {
                "reference_sha256": t.identity["reference_sha256"],
                "frames": 840,
                "physics_steps": 0,
                "independent_qpos_max_abs": max_q,
                "independent_contact_distance_max_abs": max_dist,
                "native_fk_vs_reference_max_m": max_fk,
                "pairs": pairs,
                "selected_frames": rows,
                "conclusion": "Overlap exists in independently mapped immutable reference/native collision geometry; cache mapping and world origins do not cause it.",
                "physical_validity": "NOT_ESTABLISHED; reset overlap remains",
            }
            (args.output / "reset_source.json").write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k not in ("pairs", "selected_frames")}
                )
            )
        finally:
            t.close()


if __name__ == "__main__":
    main()
