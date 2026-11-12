"""Jointly optimized840-frame native contact IK, with temporal derivatives.

All840 arm vectors are variables of ONE sparse least-squares problem. Velocity
and acceleration rows couple adjacent frames. Root, legs, waist, Box trajectory
and global binary intent are fixed. No claim of dynamics from this computation.
"""

import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, diags, eye, kron, vstack

from closure_common import active_pairs, candidate, cpu_model, dump_gz, inputs, reference_qpos, save


class TrajectoryIK:
    def __init__(self, name, qref=None, intent=None):
        self.m = cpu_model(name)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qref = reference_qpos(m) if qref is None else qref.copy()
        self.intent = (
            inputs()[1].arrays["object_contact"] if intent is None else np.array(intent, bool)
        )
        self.N = len(self.qref)
        names = [
            m.joint(i).name
            for i in range(m.njnt)
            if any(w in m.joint(i).name for w in ("shoulder", "elbow", "wrist"))
        ]
        self.jids = np.array([m.joint(n).id for n in names])
        self.qa = m.jnt_qposadr[self.jids]
        self.va = m.jnt_dofadr[self.jids]
        self.names = names
        self.n = len(names)
        assert self.n == 14
        self.obj, self.robot, self.hands, self.non = active_pairs(m)
        self.box = m.body("box/bread_box").id
        self.target = np.array([[0, 0.17, 0.14], [0, -0.17, 0.14]])
        # Surface convention is selected per candidate, independent of score.
        cs = candidate(name)["geoms"]
        self.side_y = [
            max(g["pos"][1] + g["size"][1] for g in cs if abs(g["pos"][2] - 0.14) <= g["size"][2]),
            -min(g["pos"][1] - g["size"][1] for g in cs if abs(g["pos"][2] - 0.14) <= g["size"][2]),
        ]
        self.nr = 28
        self.ref = self.qref[:, self.qa].copy()
        self.last_x = None
        self.temporal = vstack(
            [
                kron(
                    diags(
                        [-np.ones(self.N - 1), np.ones(self.N - 1)],
                        [0, 1],
                        shape=(self.N - 1, self.N),
                    ),
                    eye(14),
                )
                / 0.1,
                kron(
                    diags(
                        [np.ones(self.N - 2), -2 * np.ones(self.N - 2), np.ones(self.N - 2)],
                        [0, 1, 2],
                        shape=(self.N - 2, self.N),
                    ),
                    eye(14),
                )
                / 0.08,
            ],
            format="csr",
        )

    def jac_point(self, bid, point):
        jp = np.zeros((3, self.m.nv))
        jr = jp.copy()
        mujoco.mj_jac(self.m, self.d, jp, jr, point, bid)
        return jp[:, self.va], jr[:, self.va]

    def distance(self, g, b):
        ft = np.zeros(6)
        dist = mujoco.mj_geomDistance(self.m, self.d, g, b, 10.0, ft)
        normal = (ft[:3] - ft[3:]) / dist if abs(dist) > 1e-9 else np.zeros(3)
        j1, _ = self.jac_point(self.m.geom_bodyid[g], ft[:3])
        j2, _ = self.jac_point(self.m.geom_bodyid[b], ft[3:])
        return dist, normal @ (j1 - j2)

    def frame(self, k, x):
        m, d = self.m, self.d
        d.qpos[:] = self.qref[k]
        d.qpos[self.qa] = x
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)
        r = np.zeros(28)
        J = np.zeros((28, 14))
        r[:14] = 5 * (x - self.ref[k])
        J[:14] = 5 * np.eye(14)
        R = d.xmat[self.box].reshape(3, 3)
        p = d.xpos[self.box]
        normal = R[:, 1]
        for side, hands in enumerate(self.hands):
            yaw = hands[1]
            sign = 1 if side == 0 else -1
            centre = d.geom_xpos[yaw]
            axis = d.geom_xmat[yaw].reshape(3, 3)[:, 2]
            local = (centre - p) @ R
            jp, jr = self.jac_point(m.geom_bodyid[yaw], centre)
            ay = float(axis @ normal)
            radius, half = m.geom_size[yaw, :2]
            plane = sign * local[1] - self.side_y[side] - radius - half * abs(ay)
            planej = sign * (normal @ jp) - half * np.sign(ay) * (np.cross(axis, normal) @ jr)
            if self.intent[k]:
                r[14 + side] = 1000 * (plane + 0.0003)
                J[14 + side] = 1000 * planej
                vals = [(*self.distance(yaw, b), b) for b in self.obj]
                ds, dj, _ = min(vals, key=lambda z: z[0])
                r[16 + side] = 1000 * (ds + 0.0003)
                J[16 + side] = 1000 * dj
                for ax, row in [(0, 22 + 2 * side), (2, 23 + 2 * side)]:
                    r[row] = 10 * (local[ax] - self.target[side, ax])
                    J[row] = 10 * (R[:, ax] @ jp)
            for part, g in enumerate(hands):
                vals = [self.distance(g, b) for b in self.obj]
                ds, dj = min(vals, key=lambda z: z[0])
                # Interior0.8mm guard leaves0.2mm numerical/soft-constraint
                # margin to the unchanged1mm independent acceptance boundary.
                if ds < -0.0008:
                    row = 18 + 2 * side + part
                    r[row] = 2000 * (ds + 0.0008)
                    J[row] = 2000 * dj
        badnon = []
        badself = []
        for c in d.contact[: d.ncon]:
            if c.dist >= -0.0005:
                continue
            pair = (int(c.geom1), int(c.geom2))
            if (pair[0] in self.non and pair[1] in self.obj) or (
                pair[1] in self.non and pair[0] in self.obj
            ):
                badnon.append((float(c.dist), pair))
            if pair[0] in self.robot and pair[1] in self.robot:
                badself.append((float(c.dist), pair))
        for row, contacts in [(26, badnon), (27, badself)]:
            if contacts:
                _, pair = min(contacts)
                ds, dj = self.distance(*pair)
                r[row] = 2000 * (ds + 0.0005)
                J[row] = 2000 * dj
        return r, J

    def evaluate(self, x):
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return
        xx = x.reshape(self.N, 14)
        rr = []
        jj = []
        for k in range(self.N):
            r, j = self.frame(k, xx[k])
            rr.append(r)
            jj.append(j)
        jarr = np.array(jj)
        f, row, col = np.nonzero(jarr)
        local = coo_matrix(
            (jarr[f, row, col], (f * 28 + row, f * 14 + col)), shape=(self.N * 28, self.N * 14)
        ).tocsr()
        self.residual = np.r_[np.array(rr).ravel(), self.temporal @ x]
        self.jacobian = vstack([local, self.temporal], format="csr")
        self.last_x = x.copy()

    def fun(self, x):
        self.evaluate(x)
        return self.residual

    def jac(self, x):
        self.evaluate(x)
        return self.jacobian

    def audit(self, q):
        m, d = self.m, self.d
        rows = []
        vel = np.gradient(q[:, self.qa], 0.02, axis=0)
        acc = np.gradient(vel, 0.02, axis=0)
        for k in range(self.N):
            d.qpos[:] = q[k]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            distances = []
            yaw_dist = []
            for hs in self.hands:
                distances.append(
                    min(
                        mujoco.mj_geomDistance(m, d, g, b, 10.0, None) for g in hs for b in self.obj
                    )
                )
                yaw_dist.append(
                    min(mujoco.mj_geomDistance(m, d, hs[1], b, 10.0, None) for b in self.obj)
                )
            non = min(
                mujoco.mj_geomDistance(m, d, g, b, 10.0, None) for g in self.non for b in self.obj
            )
            selfdist = min(
                (
                    float(c.dist)
                    for c in d.contact[: d.ncon]
                    if c.geom1 in self.robot and c.geom2 in self.robot
                ),
                default=0.0,
            )
            good = (
                all(-0.001 <= v <= 0.002 for v in distances)
                and non >= -0.001
                and selfdist >= -0.001
            )
            rows.append(
                {
                    "frame": k,
                    "intent": bool(self.intent[k]),
                    "joint_correction_rad": q[k, self.qa] - self.ref[k],
                    "hand_surface_distance_m": distances,
                    "yaw_surface_distance_m": yaw_dist,
                    "nonhand_min_m": non,
                    "self_min_m": selfdist,
                    "arm_velocity_rad_s": vel[k],
                    "arm_acceleration_rad_s2": acc[k],
                    "feasible": good,
                }
            )
        on = [r for r in rows if r["intent"]]
        good = np.array([r["feasible"] for r in on])
        padded = np.r_[False, good, False].astype(int)
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        summary = {
            "ON_feasible": int(good.sum()),
            "ON_total": len(on),
            "continuous_all_ON": bool(good.all()),
            "max_continuous_ON_frames": int(max(ends - starts, default=0)),
            "ON_failed_frames": [r["frame"] for r in on if not r["feasible"]],
            "max_arm_correction_rad": float(np.abs(q[:, self.qa] - self.ref).max()),
            "RMS_arm_correction_rad": float(np.sqrt(np.mean((q[:, self.qa] - self.ref) ** 2))),
            "max_arm_velocity_rad_s": float(np.abs(vel).max()),
            "max_arm_acceleration_rad_s2": float(np.abs(acc).max()),
            "min_hand_surface_distance_m": np.min(
                [r["hand_surface_distance_m"] for r in on], axis=0
            ),
            "max_hand_surface_gap_m": np.max([r["hand_surface_distance_m"] for r in on], axis=0),
            "min_nonhand_m": min(r["nonhand_min_m"] for r in rows),
            "min_self_m": min(r["self_min_m"] for r in rows),
            "temporal_optimization": "single jointly coupled840-frame sparse nonlinear least squares",
            "root_box_legs_waist_unchanged": True,
            "kinematic_only_not_carrying": True,
        }
        return rows, summary


def main(args):
    out = args.output
    out.mkdir(exist_ok=False)
    shutil.copytree(
        Path(__file__).parent, out / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__")
    )
    ik = TrajectoryIK(args.candidate)
    m = ik.m
    # Independent central-difference check of analytic collision/plane Jacobian.
    checks = []
    for k in (260, 400, 500):
        x = ik.ref[k].copy()
        _r, j = ik.frame(k, x)
        fd = np.zeros_like(j)
        for i in range(14):
            delta = np.zeros(14)
            delta[i] = 1e-6
            fd[:, i] = (ik.frame(k, x + delta)[0] - ik.frame(k, x - delta)[0]) / (2e-6)
        checks.append(
            {
                "frame": k,
                "max_abs_error": float(np.max(np.abs(fd - j))),
                "relative_error": float(np.linalg.norm(fd - j) / max(np.linalg.norm(fd), 1)),
            }
        )
    save(out / "jacobian_check.json", checks)
    if max(c["relative_error"] for c in checks) > 1e-3:
        raise ValueError("IK analytic Jacobian failed central-difference check")
    x = ik.ref.copy()
    lo = m.jnt_range[ik.jids, 0] + 1e-6
    hi = m.jnt_range[ik.jids, 1] - 1e-6
    if args.initial is not None:
        initial = np.array(json.loads(args.initial.read_text()))
        assert initial.shape == ik.qref.shape
        x = initial[:, ik.qa]
    x = np.clip(x, lo, hi)
    result = least_squares(
        ik.fun,
        x.ravel(),
        jac=ik.jac,
        bounds=(np.tile(lo, 840), np.tile(hi, 840)),
        tr_solver="lsmr",
        tr_options={"maxiter": 100},
        max_nfev=args.evaluations,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-5,
        verbose=2,
    )
    q = ik.qref.copy()
    q[:, ik.qa] = result.x.reshape(ik.N, 14)
    rows, summary = ik.audit(q)
    _, original = ik.audit(ik.qref)
    summary.update(
        candidate=args.candidate,
        optimizer_success=bool(result.success),
        optimizer_message=result.message,
        evaluations=result.nfev,
        cost=result.cost,
        optimality=result.optimality,
        original=original,
        original_reference_sha256="bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b",
    )
    save(out / "summary.json", summary)
    dump_gz(out / "frames.json.gz", rows)
    save(out / "qpos.json", q)
    print(json.dumps(summary, default=lambda x: x.tolist(), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evaluations", type=int, default=60)
    p.add_argument("--initial", type=Path)
    main(p.parse_args())
