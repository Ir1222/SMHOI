"""Forward-only reset inspection of existing profiles; zero physical transitions."""

import argparse
import json

import mujoco
import numpy as np
import torch

from supermimic.hoi.provenance import research_authority
from supermimic.hoi.reference import DEFAULT_OUTPUT, ReferenceClip
from supermimic.hoi.scene import PhysicalSceneCfg, make_physical_env
from supermimic.hoi.state import JointStateIO, ReferenceResetCache
from supermimic.hoi.training_safety import SolverGuard
from supermimic.hoi.training_scene import profile_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    clip = ReferenceClip(DEFAULT_OUTPUT / "reference.npz")
    for profile in ("p6-original", "p6-single-body"):
        with SolverGuard() as guard:
            env = make_physical_env(
                PhysicalSceneCfg(num_envs=2), manifest=profile_manifest(profile)
            )
            try:
                guard.attest(env, "preflight_forward_only")
                io = JointStateIO(env)
                cache = ReferenceResetCache(clip, io.joint_names, env.device)
                ids = torch.arange(2, device=env.device)
                for frame in (0, 250, 350, 475, 550, 740):
                    env.reset()
                    target = cache.frames(
                        torch.full((2,), frame, device=env.device, dtype=torch.long)
                    )
                    io.write(ids, target)
                    error = float((io.read() - target).abs().max())
                    assert error < 1e-6
                    assert not env.sim.data.time[:].any()
                    for world in range(2):
                        model = env.sim.mj_model
                        data = mujoco.MjData(model)
                        data.qpos[:] = env.sim.data.qpos[world].cpu().numpy()
                        data.qvel[:] = env.sim.data.qvel[world].cpu().numpy()
                        origin = env.scene.env_origins[world].cpu().numpy()
                        data.qpos[io.robot_q[:3].cpu().numpy()] -= origin
                        data.qpos[io.box_q[:3].cpu().numpy()] -= origin
                        mujoco.mj_forward(model, data)
                        contacts = []
                        for contact in data.contact:
                            bodies = [
                                model.body(model.geom_bodyid[g]).name
                                for g in (contact.geom1, contact.geom2)
                            ]
                            contacts.append({"bodies": bodies, "distance_m": float(contact.dist)})
                        assert np.isfinite(data.qacc).all()
                        rows.append(
                            {
                                "profile": profile,
                                "frame": frame,
                                "world": world,
                                "reset_roundtrip_max_abs": error,
                                "finite_qacc": True,
                                "minimum_distance_m": min(
                                    (c["distance_m"] for c in contacts), default=None
                                ),
                                "contacts": contacts,
                            }
                        )
            finally:
                env.close()
    with open(args.output, "x") as stream:
        json.dump(
            {
                "research_authority": research_authority(),
                "physical_steps": 0,
                "reference": str(clip.path),
                "reference_sha256": clip.manifest["sha256"],
                "scope": "forward-only geometry; not transition data or physical-readiness approval",
                "resets": rows,
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
