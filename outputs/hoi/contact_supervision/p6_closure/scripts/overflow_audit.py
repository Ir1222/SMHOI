"""C2.1 exact first-failure replay with complete bit and contact decoding."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from closure_common import E, gpu_env, save
from solver_diagnosis import snapshot
from contact_parameters import decode
from contact_gpu import contact_snapshot


def main(out):
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir = "/tmp/smhoi_p6_warp_cache"
    results = []
    for mode in ("complete", "robot_floor", "reference0"):
        source = json.loads((E / "solver_01" / mode / "first_failure.json").read_text())
        before = source["before"]

        def change(cfg, mode=mode):
            if mode == "robot_floor":
                base = cfg.scene.spec_fn

                def edit(spec):
                    base(spec)
                    for g in spec.geoms:
                        if g.name.startswith("box/") or "support" in g.name:
                            g.contype = g.conaffinity = 0

                cfg.scene.spec_fn = edit

        env, _ = gpu_env("original", change=change)
        try:
            for k in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied"):
                dst = getattr(env.sim.data, k)
                dst.copy_(torch.tensor(before[k], dtype=dst.dtype, device=env.device))
            env.sim.forward()
            env.sim.data.qacc_warmstart.copy_(
                torch.tensor(before["qacc_warmstart"], device=env.device, dtype=torch.float32)
            )
            env.sim.data.overflow.zero_()
            env.sim.step()
            state = snapshot(env)
            contact = contact_snapshot(env)
            save(
                out / (mode + "_state.json"),
                {
                    "historical_before": before,
                    "actual_after": state,
                    "contacts": contact,
                    "geom_names": [
                        env.sim.mj_model.geom(i).name for i in range(env.sim.mj_model.ngeom)
                    ],
                },
            )
            summary = {
                "mode": mode,
                "historical_sample": source["sample"],
                "historical_overflow": source["after"]["overflow"],
                "actual_overflow": [decode(v) for v in state["overflow"]],
                "nefc": state["nefc"],
                "solver_niter": state["solver_niter"],
                "solver": int(env.sim.wp_model.opt.solver),
                "iterations": env.sim.wp_model.opt.iterations,
                "ls_iterations": env.sim.wp_model.opt.ls_iterations,
                "ls_tolerance": env.sim.wp_model.opt.ls_tolerance.numpy(),
                "tolerance": env.sim.wp_model.opt.tolerance.numpy(),
                "cone": int(env.sim.wp_model.opt.cone),
                "worlds": [],
            }
            for w in range(4):
                mask = contact["worldid"] == w
                dims = contact["dim"][mask]
                summary["worlds"].append(
                    {
                        "world": w,
                        "contact_count": int(mask.sum()),
                        "dim_counts": {str(v): int((dims == v).sum()) for v in np.unique(dims)},
                        "torsional_contact_count": int((dims >= 4).sum()),
                        "rolling_contact_count": int((dims == 6).sum()),
                        "min_distance": float(contact["dist"][mask].min()) if mask.any() else None,
                    }
                )
            results.append(summary)
        finally:
            env.close()
    save(out / "summary.json", results)
    print([(r["mode"], [x["value"] for x in r["actual_overflow"]]) for r in results])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
