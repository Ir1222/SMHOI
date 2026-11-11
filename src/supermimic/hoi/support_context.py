"""Time-varying support geometry context, separate from episode xi33.

P3 support_02 / support_balanced_analysis establishes canonical input aliasing.
Encoding: manifest-order support root positions and rotation6 relative to current
robot root/heading, 2*(3+6)=18. Fixed extents/materials are bound to asset SHA256.
"""

import torch
from torch import nn

from supermimic.hoi.asset_manifest import load_manifest
from supermimic.hoi.reference import DEFAULT_ASSET, sha256
from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
)


class SupportContext(nn.Module):
    dimension = 18
    schema = "support-root-heading-p3-r6-v1"

    def __init__(self, manifest_path=DEFAULT_ASSET):
        super().__init__()
        manifest = load_manifest(manifest_path)
        supports = manifest["supports"]
        if len(supports) != 2:
            raise ValueError("This fixed-asset encoding requires exactly two supports")
        self.asset_sha256 = sha256(manifest_path)
        self.names = tuple(s["name"] for s in supports)
        self.register_buffer(
            "positions", torch.tensor([s["root_pos_world"] for s in supports], dtype=torch.float32)
        )
        self.register_buffer(
            "quaternions",
            torch.tensor([s["root_quat_wxyz"] for s in supports], dtype=torch.float32),
        )

    def forward(self, state):
        if state.shape[-1] != 84:
            raise ValueError("Support context needs joint physical state84")
        h = heading_quat(quat_normalize(state[..., 3:7]))[..., None, :].expand(
            *state.shape[:-1], 2, 4
        )
        p = self.positions.expand(*state.shape[:-1], 2, 3)
        q = self.quaternions.expand(*state.shape[:-1], 2, 4)
        relative = quat_mul(quat_conjugate(h), q)
        return torch.cat(
            (quat_apply_inverse(h, p - state[..., None, :3]), rot6d_from_quat(relative)), -1
        ).flatten(-2)
