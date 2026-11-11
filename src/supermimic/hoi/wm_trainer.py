"""HOI batch/reference adapter; the original optimizer and update loop are inherited."""

import torch

from supermimic.hoi.reference import ReferenceClip
from supermimic.world_model.heads import ReferenceFrame
from supermimic.world_model.wm_trainer import WorldModelTrainer


class JointWorldModelTrainer(WorldModelTrainer):
    def __init__(self, wm, references: list[ReferenceClip], cfg):
        if not references:
            raise ValueError("Explicit reference list required; no legacy default")
        self.reference_hashes = [clip.manifest["sha256"] for clip in references]
        device = next(wm.parameters()).device
        self.reference_roots = []
        for clip in references:
            root = clip.arrays["body_names"].tolist().index("pelvis")
            self.reference_roots.append(
                tuple(
                    torch.tensor(clip.arrays[k][:, root].copy(), device=device)
                    for k in ("body_pos_w", "body_quat_w")
                )
            )
        super().__init__(wm, None, cfg)

    def references_for(self, batch):
        k, n = batch["frame"].shape
        positions = torch.empty(k, n, 3, device=batch["states"].device)
        quaternions = torch.empty(k, n, 4, device=positions.device)
        for c in batch["clip"].unique().tolist():
            if not 0 <= c < len(self.reference_roots):
                raise ValueError("Unknown clip identity")
            selected = batch["clip"] == c
            target_frame = batch["frame"][selected] + 1
            pos, quat = self.reference_roots[c]
            if ((target_frame < 1) | (target_frame >= len(pos))).any():
                raise ValueError("Target frame outside reference; no clamping")
            positions[selected], quaternions[selected] = pos[target_frame], quat[target_frame]
        return [
            ReferenceFrame(None, None, None, None, positions[j], quaternions[j]) for j in range(k)
        ]

    def physical_states(self, batch):
        if not batch["transition_valid"].all():
            raise ValueError("P3 uses full valid windows, no unmasked padding")
        if not torch.equal(batch["states"][1:], batch["next_state"]):
            raise ValueError("State/transition target alignment failed")
        for name in ("env_id", "episode_id", "clip"):
            if not (batch[name] == batch[name][0]).all():
                raise ValueError(f"Window crosses {name}")
        if ((batch["frame"][1:] - batch["frame"][:-1]) != 1).any() or batch["transition_terminal"][
            :-1
        ].any():
            raise ValueError("Discontinuous or prematurely terminal window")
        states = batch["states"]
        if states.ndim != 3 or states.shape[-1] != 84 or not torch.isfinite(states).all():
            raise ValueError("Expected finite physical states [K+1,N,84]")
        return states

    def loss_on(self, batch):
        states = self.physical_states(batch)
        return self.wm.window_loss(
            states,
            batch["a_exec"],
            batch["xi"],
            batch["physical_failure"].float(),
            self.references_for(batch),
        )

    @torch.no_grad()
    def online_error(self, buffer, n=1024, k=8):
        batch = buffer.sample_windows(n, k)
        return self.wm.rollout_error(self.physical_states(batch), batch["a_exec"], batch["xi"])
