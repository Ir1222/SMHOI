"""Matched CPU/GPU original-floor diagnosis, with states before first failure."""

import argparse
import copy
import json
import shutil
import traceback
from pathlib import Path

from closure_common import P6, dump_gz, gpu_env, inputs, save
import mujoco
import numpy as np
import torch
import warp as wp
from calibrate_contacts import Placements, reset_states
from run_p6 import Observer
from supermimic.hoi.contact import assert_contact_capacity
from supermimic.hoi.state import JointStateIO, ReferenceResetCache


def snapshot(env):
    fields = (
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "ctrl",
        "actuator_force",
        "qfrc_actuator",
        "qfrc_smooth",
        "qfrc_constraint",
        "qfrc_applied",
        "xfrc_applied",
        "nefc",
        "overflow",
    )
    result = {k: getattr(env.sim.wp_data, k).numpy().copy() for k in fields}
    result["solver_niter"] = env.sim.wp_data.solver_niter.numpy().copy()
    return result


def experiment(name, kind, out, seconds, tolerance=None):
    def change(cfg):
        if tolerance is not None:
            cfg.sim.mujoco.tolerance = tolerance
        original = cfg.scene.spec_fn

        def edit(spec):
            original(spec)
            for g in spec.geoms:
                disable = (
                    (kind == "robot_floor" and (g.name.startswith("box/") or "support" in g.name))
                    or (kind == "ablate_floor" and "terrain" in g.name)
                    or (kind == "box_support" and g.name.startswith("robot/"))
                )
                if disable:
                    g.contype = g.conaffinity = 0

        cfg.scene.spec_fn = edit

    # Original native hand validation must hold for all tests. For box_support,
    # retain hand masks: their remote pose cannot contact any object/floor in3s.
    if kind == "box_support":

        def change(cfg):
            if tolerance is not None:
                cfg.sim.mujoco.tolerance = tolerance

    env, manifest = gpu_env(name, change=change)
    io = JointStateIO(env)
    _placement = Placements(env, io)
    folder = out / kind
    folder.mkdir()
    observer = Observer(env, io)
    source = json.loads((P6 / "isolation_01/original_supports_robot_z3.0/initial.json").read_text())
    states = np.array(source["state84"])
    if kind == "box_support":
        states[:, 2] = 50.0
    if kind == "elevated":
        states = np.array(
            json.loads((P6 / "isolation_01/hand_release_no_floor/initial.json").read_text())[
                "state84"
            ]
        )
    if kind == "reference0":
        cache = ReferenceResetCache(inputs()[1], io.joint_names, env.device)
        states = np.tile(cache.states[0].cpu().numpy(), (4, 1))
    reset_states(env, io, states)
    save(
        folder / "initial.json",
        {
            "states84": states,
            "origins": env.scene.env_origins.cpu().numpy(),
            "kind": kind,
            "candidate": name,
            "dt": 0.005,
            "iterations": env.sim.mj_model.opt.iterations,
            "ls_iterations": env.sim.mj_model.opt.ls_iterations,
            "tolerance": env.sim.mj_model.opt.tolerance,
            "ls_tolerance": env.sim.mj_model.opt.ls_tolerance,
            "cpu_gpu_same_controls": True,
            "no_postreset_state_assignment": True,
        },
    )
    cpu = []
    start = snapshot(env)
    for world in range(4):
        m = copy.copy(env.sim.mj_model)
        for s in manifest["supports"]:
            m.body_pos[m.body(s["name"]).id] = (
                np.array(s["root_pos_world"]) + env.scene.env_origins[world].cpu().numpy()
            )
        d = mujoco.MjData(m)
        d.qpos[:] = start["qpos"][world]
        d.qvel[:] = start["qvel"][world]
        d.qacc_warmstart[:] = start["qacc_warmstart"][world]
        mujoco.mj_forward(m, d)
        cpu.append((m, d))
    records = []
    forward_records = []
    cpu_contacts = []
    native = env.sim.step
    native_forward = env.sim.forward
    metrics = env.metrics_manager.compute_substep

    def step():
        before = snapshot(env)
        observer.solve_state = io.read().cpu().numpy()
        native()
        after = snapshot(env)
        shadows = []
        for world, (m, d) in enumerate(cpu):
            d.ctrl[:] = before["ctrl"][world]
            d.qfrc_applied[:] = before["qfrc_applied"][world]
            d.xfrc_applied[:] = before["xfrc_applied"][world]
            mujoco.mj_step(m, d)
            shadows.append(
                {
                    "qpos": d.qpos.copy(),
                    "qvel": d.qvel.copy(),
                    "qacc": d.qacc.copy(),
                    "actuator_force": d.actuator_force.copy(),
                    "ncon": d.ncon,
                    "nefc": d.nefc,
                    "solver_niter": d.solver_niter.copy(),
                    "warnings": [int(w.number) for w in d.warning],
                    "solver_gradient": [
                        float(v.gradient) for v in d.solver[: int(d.solver_niter.max())]
                    ],
                    "solver_lineslope": [
                        float(v.lineslope) for v in d.solver[: int(d.solver_niter.max())]
                    ],
                }
            )
            if after["overflow"].any() or len(records) < 2:
                for c in d.contact[: d.ncon]:
                    cpu_contacts.append(
                        {
                            "sample": len(records),
                            "world": world,
                            "dist": c.dist,
                            "pos": c.pos.copy(),
                            "frame": c.frame.copy(),
                            "geoms": [int(c.geom1), int(c.geom2)],
                            "names": [m.geom(c.geom1).name, m.geom(c.geom2).name],
                        }
                    )
        records.append(
            {
                "sample": len(records),
                "time_end_s": (len(records) + 1) * 0.005,
                "before": before,
                "after": after,
                "cpu": shadows,
            }
        )

    def observe():
        observer.sample()
        metrics()

    def forward():
        before = snapshot(env)
        native_forward()
        forward_records.append(
            {"after_substeps": len(records), "before": before, "after": snapshot(env)}
        )

    env.sim.step = step
    env.sim.forward = forward
    env.metrics_manager.compute_substep = observe
    term = env.action_manager.get_term("joint_pos")
    hold = (
        torch.tensor(states[:, 7:36], device=env.device, dtype=torch.float32) - term._offset
    ) / term._scale
    result = {"candidate": name, "mode": kind, "status": "PASS"}
    try:
        for k in range(round(seconds / 0.02)):
            action = torch.zeros((4, 29), device=env.device)
            if kind == "elevated":
                action = hold
            if kind == "reference0":
                action = (cache.states[k, 7:36][None].expand(4, -1) - term._offset) / term._scale
            env.step(action)
            assert_contact_capacity(env)
    except Exception as error:  # noqa: BLE001 -- preserve failed experiment evidence
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        env.sim.step = native
        env.sim.forward = native_forward
        env.metrics_manager.compute_substep = metrics
        observer.write(folder)
        dump_gz(folder / "solver_states.json.gz", records)
        dump_gz(folder / "final_forward_states.json.gz", forward_records)
        dump_gz(folder / "cpu_contacts.json.gz", cpu_contacts)
        result.update(
            samples=len(records),
            first_flag_sample=next(
                (r["sample"] for r in records if r["after"]["overflow"].any()), None
            ),
            first_flag_final_forward=next(
                (r["after_substeps"] for r in forward_records if np.any(r["after"]["overflow"])),
                None,
            ),
            final_overflow=env.sim.wp_data.overflow.numpy().copy(),
            max_niter=int(max((r["after"]["solver_niter"].max() for r in records), default=0)),
            max_nefc=int(max((r["after"]["nefc"].max() for r in records), default=0)),
            cpu_warning_counts=[int(sum(w for w in d.warning.number)) for m, d in cpu],
            max_CPU_GPU_qpos_difference=float(
                max(
                    (
                        max(
                            np.max(np.abs(r["after"]["qpos"][w] - r["cpu"][w]["qpos"]))
                            for w in range(4)
                        )
                        for r in records
                    ),
                    default=0,
                )
            ),
            raw_sensor_max_error_N=observer.raw_error,
        )
        if records and result["first_flag_sample"] is not None:
            save(folder / "first_failure.json", records[result["first_flag_sample"]])
        flagged_forward = next((r for r in forward_records if np.any(r["after"]["overflow"])), None)
        if flagged_forward is not None:
            save(folder / "first_forward_failure.json", flagged_forward)
        save(folder / "result.json", result)
        env.close()
    print(kind, result["status"], result["samples"], result.get("error", ""), flush=True)
    return result


def instrument_solver(out):
    import hashlib
    from mujoco_warp._src import solver

    original = Path(solver.__file__)
    source = original.read_text()
    needle = "      if not ls_converged:\n        if wp.static(bool(warn_overflow & OverflowType.LS_ITERATIONS)):"
    replacement = '      if not ls_converged:\n        wp.printf("P6_LS_TRACE world=%d slope=%.9g curvature=%.9g gross=%.9g gtol=%.9g accept=%.9g alpha=%.9g lo_alpha=%.9g hi_alpha=%.9g lo_cost=%.9g lo_grad=%.9g hi_cost=%.9g hi_grad=%.9g\\n", worldid, p0[1], p0[2], q1_abs, gtol, gtol_accept, alpha, lo_alpha, hi_alpha, lo[0], lo[1], hi[0], hi[1])\n        if wp.static(bool(warn_overflow & OverflowType.LS_ITERATIONS)):'
    assert source.count(needle) == 1
    path = out / "solver_instrumented.py"
    path.write_text(source.replace(needle, replacement))
    save(
        out / "instrumentation.json",
        {
            "original": str(original),
            "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "instrumented_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "change": "printf only at existing failure condition; installed file unchanged",
        },
    )
    solver.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), solver.__dict__)  # noqa: S102 -- diagnostic copy of installed solver


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate", default="original")
    p.add_argument(
        "--modes",
        nargs="+",
        default=[
            "complete",
            "robot_floor",
            "ablate_floor",
            "box_support",
            "elevated",
            "reference0",
        ],
    )
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--tolerance", type=float)
    p.add_argument("--instrument", action="store_true")
    p.add_argument("--precision-guard", action="store_true")
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    shutil.copytree(
        Path(__file__).parent,
        a.output / "source_snapshot",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    torch.set_num_threads(1)
    torch.manual_seed(606)
    wp.config.kernel_cache_dir = "/tmp/smhoi_p6_warp_cache"
    if a.instrument:
        instrument_solver(a.output)
    from contextlib import nullcontext
    from supermimic.hoi.solver_precision import bracket_precision_guard

    with bracket_precision_guard() if a.precision_guard else nullcontext() as profile:
        if profile:
            save(a.output / "precision_guard.json", profile)
        save(
            a.output / "results.json",
            [experiment(a.candidate, k, a.output, a.seconds, a.tolerance) for k in a.modes],
        )
