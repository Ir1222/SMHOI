"""Named joint scene state access and reference-only pose/twist reset cache."""

from __future__ import annotations

import mujoco
import torch

from supermimic.envs.state_io import NQ, NV
from supermimic.envs.state_io import STATE_DIM as ROBOT_DIM
from supermimic.hoi.reference import ReferenceClip
from supermimic.kinematics.rotations import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    rotvec_from_quat,
)

OBJECT_DIM = 13
JOINT_DIM = ROBOT_DIM + OBJECT_DIM


def reference_object_twist(pos: torch.Tensor, quat: torch.Tensor, dt: float) -> torch.Tensor:
    """Reset-only [N,6]: world root-origin linear, current-body angular velocity.

    Central differences inside, one-sided endpoints, shortest quaternion log.
    Separate derived cache; never a substitute for measured simulator velocity.
    """
    if len(pos) < 2 or pos.shape != (len(quat), 3) or quat.shape[-1] != 4 or dt <= 0:
        raise ValueError("Expected at least two pose samples and positive dt")
    indices = torch.arange(len(pos), device=pos.device)
    lo, hi = (indices - 1).clamp_min(0), (indices + 1).clamp_max(len(pos) - 1)
    span = ((hi - lo) * dt).unsqueeze(-1)
    linear = (pos[hi] - pos[lo]) / span
    world_delta = quat_mul(quat[hi], quat_conjugate(quat[lo]))
    angular_world = rotvec_from_quat(world_delta) / span
    return torch.cat((linear, quat_apply_inverse(quat, angular_world)), dim=-1)


class ReferenceResetCache:
    """Read-only Part B inputs, named joints, derived object twist in memory."""

    def __init__(self, clip: ReferenceClip, joint_names: tuple[str, ...], device: str):
        a = clip.arrays
        source_names = a["joint_names"].tolist()
        if set(joint_names) != set(source_names) or len(joint_names) != 29:
            raise ValueError("Reference/native joint names differ")
        order = [source_names.index(name) for name in joint_names]
        root = a["body_names"].tolist().index("pelvis")
        tensors = {
            k: torch.tensor(a[k].copy(), dtype=torch.float64)
            for k in (
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "joint_pos",
                "joint_vel",
                "object_pos_w",
                "object_quat_w",
            )
        }
        q = tensors["body_quat_w"][:, root]
        obj_twist = reference_object_twist(
            tensors["object_pos_w"],
            tensors["object_quat_w"],
            1.0 / clip.fps,
        )
        self.states = torch.cat(
            (
                tensors["body_pos_w"][:, root],
                q,
                tensors["joint_pos"][:, order],
                tensors["body_lin_vel_w"][:, root],
                quat_apply_inverse(q, tensors["body_ang_vel_w"][:, root]),
                tensors["joint_vel"][:, order],
                tensors["object_pos_w"],
                tensors["object_quat_w"],
                obj_twist,
            ),
            dim=-1,
        ).to(device=device, dtype=torch.float32)

    def frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 1 or frames.dtype != torch.long:
            raise ValueError("Frames must be a 1-D long tensor")
        if torch.any((frames < 0) | (frames >= len(self.states))):
            raise IndexError("Reset frame outside reference")
        return self.states[frames.to(self.states.device)].clone()


class JointStateIO:
    """[robot71, object13], with environment-local positions and wxyz quaternions.

    Both free bodies use world root-origin linear velocity and body angular
    velocity, exactly MuJoCo free qvel semantics. Never concatenate global arrays.
    """

    def __init__(self, env):
        self.env, self.sim = env, env.sim
        self.origins = env.scene.env_origins.clone()
        model = self.sim.mj_model
        robot, box = env.scene["robot"], env.scene["box"]
        self.joint_names = tuple(robot.joint_names)
        q_indices, v_indices = [], []
        for body_name, joints in (
            ("robot/pelvis", self.joint_names),
            (f"box/{box.body_names[0]}", ()),
        ):
            body = model.body(body_name).id
            free_id = int(model.body_jntadr[body])
            if free_id < 0 or model.jnt_type[free_id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"Missing named free joint on {body_name}")
            qa, va = int(model.jnt_qposadr[free_id]), int(model.jnt_dofadr[free_id])
            qi, vi = list(range(qa, qa + 7)), list(range(va, va + 6))
            for name in joints:
                jid = model.joint(f"robot/{name}").id
                if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError(f"Expected scalar native joint: {name}")
                qi.append(int(model.jnt_qposadr[jid]))
                vi.append(int(model.jnt_dofadr[jid]))
            q_indices.append(torch.tensor(qi, device=env.device))
            v_indices.append(torch.tensor(vi, device=env.device))
        self.robot_q, self.box_q = q_indices
        self.robot_v, self.box_v = v_indices
        if len(self.robot_q) != NQ or len(self.robot_v) != NV:
            raise ValueError("Native robot state must retain nq36/nv35")
        self.box_body_id = model.body(f"box/{box.body_names[0]}").id

    def read(self) -> torch.Tensor:
        q, v = self.sim.data.qpos[:], self.sim.data.qvel[:]
        state = torch.cat(
            (q[:, self.robot_q], v[:, self.robot_v], q[:, self.box_q], v[:, self.box_v]), dim=-1
        )
        state[:, :3] -= self.origins
        state[:, ROBOT_DIM : ROBOT_DIM + 3] -= self.origins
        return state

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        state = state.to(device=self.env.device, dtype=self.sim.data.qpos.dtype).clone()
        if state.shape != (len(env_ids), JOINT_DIM) or not torch.isfinite(state).all():
            raise ValueError("Expected finite joint state [selected_envs,84]")
        for start in (3, ROBOT_DIM + 3):
            if torch.any((state[:, start : start + 4].norm(dim=-1) - 1).abs() > 1e-5):
                raise ValueError("Expected unit wxyz quaternion")
        state[:, :3] += self.origins[env_ids]
        state[:, ROBOT_DIM : ROBOT_DIM + 3] += self.origins[env_ids]
        rows = env_ids[:, None]
        self.sim.data.qpos[rows, self.robot_q] = state[:, :NQ]
        self.sim.data.qvel[rows, self.robot_v] = state[:, NQ:ROBOT_DIM]
        self.sim.data.qpos[rows, self.box_q] = state[:, ROBOT_DIM : ROBOT_DIM + 7]
        self.sim.data.qvel[rows, self.box_v] = state[:, ROBOT_DIM + 7 :]
        self.sim.data.qacc_warmstart[env_ids] = 0
        if forward:
            self.sim.forward()

    def reset_to_reference(
        self,
        env_ids: torch.Tensor,
        frames: torch.Tensor,
        cache: ReferenceResetCache,
    ) -> None:
        state = cache.frames(frames)
        if len(state) != len(env_ids):
            raise ValueError("One reference frame is required for each reset world")
        # Native reset clears selected solver/action/sensor histories; one write
        # initializes both bodies from the SAME frame, including object twist.
        self.env.reset(env_ids=env_ids)
        self.write(env_ids, state)

    def object_com_velocity_world(self) -> torch.Tensor:
        """[E,6] world COM linear and world angular velocity from fresh raw state."""
        obj = self.read()[:, ROBOT_DIM:]
        quat, linear, angular = obj[:, 3:7], obj[:, 7:10], obj[:, 10:13]
        omega = quat_apply(quat, angular)
        offset = self.sim.model.body_ipos[:, self.box_body_id].expand(len(obj), 3)
        com = linear + torch.cross(omega, quat_apply(quat, offset), dim=-1)
        return torch.cat((com, omega), dim=-1)
