"""Deterministic P5 test-only coefficients and synthetic dynamics, never defaults."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.hoi.imagination import HOIPolicyTrainer, HOIStartBatch
from supermimic.hoi.reference import ReferenceClip
from supermimic.hoi.reward_reference import ClipBinding, HOIReferenceRegistry
from supermimic.hoi.rewards import ProximityReward
from supermimic.hoi.state_spec import JointStateSpec
from supermimic.hoi.tracking_reward import HOIRewardCfg, HOITrackingReward
from supermimic.hoi.world_model import JointWorldModel, JointWorldModelCfg, LossWeights
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.heads import AnalyticHeads

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "outputs/hoi/asset_audit/part_b/reference/reference.npz"
TEST_CFG = HOIRewardCfg((0.2, 0.4, 1.0, 2.0), (0.7, 0.8, 0.9, 1.1), 0.0423, 0.6, 0.02)


def save_evidence(name, result):
    directory = os.environ.get("P5_EVIDENCE_DIR")
    if directory:
        path = Path(directory) / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


class DeterministicCoupledResidual(nn.Module):
    """Known nonzero 166->82 coupling fixture, NOT learned physical dynamics.

    Replaces only the residual mapping; original JointWorldModel.predict,
    support recomputation and physical integration still execute. Parameter gain
    exists to verify freezing, not to be fitted or stepped.
    """

    def __init__(self, device, dtype):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
        self.register_buffer(
            "action_map",
            0.025
            * torch.sin(torch.arange(29 * 82, device=device, dtype=dtype).reshape(29, 82) * 0.17),
        )
        self.register_buffer(
            "feature_map",
            0.0003
            * torch.cos(torch.arange(104 * 82, device=device, dtype=dtype).reshape(104, 82) * 0.11),
        )

    def forward(self, features, action, xi):
        return self.gain * (action @ self.action_map + features @ self.feature_map)


def fixture(
    dtype=torch.float64,
    device="cpu",
    coupled=True,
    horizon=3,
    anchor=False,
    compile_mode=None,
    backend="inductor",
):
    torch.set_num_threads(1)
    torch.manual_seed(501)
    model = load_g1_model()
    joint_names = tuple(model.joint(i).name for i in range(model.njnt) if model.jnt_qposadr[i] >= 7)
    q0 = torch.tensor(model.qpos0[7:], device=device, dtype=dtype)
    # P4 deliberately requires the exact FK tensor device, including CUDA index.
    fk = G1Kinematics(model, device=q0.device, dtype=dtype)
    limits = torch.tensor(model.jnt_range[1:], device=device, dtype=dtype)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, limits, q0)
    clip = ReferenceClip(REFERENCE)
    lib = MotionLibrary([REFERENCE], TRACKED_BODIES, q0, device=device, dtype=dtype)
    registry = HOIReferenceRegistry(
        lib,
        [ClipBinding(7, 0, clip)],
        library_sources=[REFERENCE],
        dt=0.02,
        joint_names=joint_names,
    )
    cfg = JointWorldModelCfg(hidden=(16, 16), term_hidden=(8, 8), weights=LossWeights(1.0, 1.0))
    wm = JointWorldModel(cfg, heads, device).to(dtype=dtype)
    wm.spec = JointStateSpec(cfg.state_spec, cfg.object_spec, device=device, dtype=dtype)
    if coupled:
        wm.dynamics = DeterministicCoupledResidual(device, dtype)
    proximity = ProximityReward(
        fk,
        ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        [[0, 0.17, 0.14], [0, -0.17, 0.14]],
        [[0.05, 0, 0], [0.05, 0, 0]],
        sigma_pos=TEST_CFG.sigma_pos,
    )
    reward = HOITrackingReward(heads, proximity, TEST_CFG)
    pcfg = PolicyCfg(
        encoder_hidden=(16, 16), decoder_hidden=(16, 16), goal_dim=689 if anchor else 680
    )
    policy = Policy(pcfg).to(device=device, dtype=dtype)
    critic = Critic(CriticCfg(hidden=(16, 8)), pcfg=pcfg).to(device=device, dtype=dtype)
    trainer = HOIPolicyTrainer(
        wm,
        policy,
        critic,
        registry,
        reward,
        PolicyTrainerCfg(
            horizon=horizon, batch_size=5, obs_noise=False, compile=compile_mode, anchor_obs=anchor
        ),
        backend=backend,
    )
    return SimpleNamespace(**locals())


def start_batch(fx, frames=(258, 259, 548, 549, 838), *, clip_ids=None, **boundaries):
    f = torch.tensor(frames, dtype=torch.long, device=fx.lib.device)
    c = (
        torch.full_like(f, 7)
        if clip_ids is None
        else torch.tensor(clip_ids, device=f.device, dtype=torch.long)
    )
    global_frame = fx.registry.global_frames(c, f)
    qpos, qvel = fx.lib.ref_qpos_qvel(global_frame)
    obj = fx.registry.gather(c, f).box
    state = torch.cat(
        (
            qpos,
            qvel,
            obj.position,
            obj.quaternion,
            obj.linear_world,
            quat_apply_inverse(obj.quaternion, obj.angular_world),
        ),
        -1,
    )
    state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
    # Nondegenerate, small tracking errors so each isolated reward has a gradient.
    state[:, 2] += 0.012
    state[:, 71:74] += state.new_tensor([0.015, -0.01, 0.005])
    xi = state.new_zeros(len(f), 33)
    xi[:, 0] = 1
    actions = state.new_zeros(10, len(f), 29)
    return HOIStartBatch(
        state,
        xi,
        global_frame.clone(),
        actions[-1],
        state[None].repeat(10, 1, 1),
        actions,
        c,
        f.clone(),
        global_frame.clone(),
        fx.registry.hashes(c),
        torch.arange(len(f), device=f.device),
        torch.zeros_like(f),
        **boundaries,
    )
