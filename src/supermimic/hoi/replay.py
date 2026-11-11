"""HOI transition sidecars on the original SuperMimic ring and window sampler."""

from __future__ import annotations

import torch

from supermimic.data.replay_buffer import ReplayBuffer

FORCE_SHAPES = {
    "force_mean_box": (2, 3),
    "impulse_world": (2, 3),
    "force_peak": (2,),
    "occupancy": (2,),
    "found_final": (2,),
    "force_raw_world": (4, 2, 2, 3),
    "box_quat_solve": (4, 4),
    "found_substeps": (4, 2, 2),
}


class HOIReplayBuffer(ReplayBuffer):
    """A row owns (s_t, a_t, s_next, interval_t, masks_t), never successor labels.

    Native terminal STATE rows remain for k+1 state windows but have no transition.
    Full windows only (default 8); newest nonterminal row waits for its successor.
    """

    def __init__(self, num_envs, capacity, device, *, force_on):
        if capacity < 9:
            raise ValueError("Capacity must hold eight transitions plus a target row")
        if not 0 < force_on < float("inf"):
            raise ValueError("Explicit positive finite force_on required")
        super().__init__(num_envs, capacity, 84, 29, 33, device)
        shapes = {"next_state": (84,), **FORCE_SHAPES, "intent_ref": (2,)}
        self.fields = {
            name: torch.zeros(num_envs, capacity, *shape, device=device)
            for name, shape in shapes.items()
        }
        for name in (
            "physical_failure",
            "timeout",
            "clip_end",
            "transition_terminal",
            "transition_valid",
        ):
            self.fields[name] = torch.zeros(num_envs, capacity, dtype=torch.bool, device=device)
        self.metadata = {
            "schema": "hoi-transition-v1",
            "force_frame": "contemporaneous_box_at_substep_solve",
            "force_sign": "hand_on_box",
            "hand_order": ["left", "right"],
            "primary_order": ["wrist_pitch", "wrist_yaw"],
            "substep_order": "chronological",
            "physics_dt_s": 0.005,
            "decimation": 4,
            "control_dt_s": 0.020,
            "force_unit": "N",
            "impulse_unit": "N s",
            "force_on_N": float(force_on),
            "found_final_time": "fourth_solve_before_final_forward",
            "frame": "transition_start; next=start+1",
            "intent_ref": "global[start,end]",
            "terminal": "physical_failure OR timeout OR clip_end; transition-indexed",
            "state": "robot71+object13; local positions; wxyz; root-world-linear/body-angular",
        }
        self._episode_start = torch.ones(num_envs, dtype=torch.bool, device=device)

    @torch.no_grad()
    def end_episode(self, env_ids):
        super().end_episode(env_ids)
        self._episode_start[env_ids] = True

    @torch.no_grad()
    def append(
        self,
        ids,
        *,
        state,
        next_state,
        action_clean,
        action_exec,
        xi,
        frame,
        clip,
        labels,
        intent_ref,
        physical_failure,
        timeout,
        clip_end,
    ):
        """Copy one measured transition; automatically add terminal state rows."""
        if (
            ids.ndim != 1
            or ids.unique().numel() != ids.numel()
            or torch.any((ids < 0) | (ids >= self.E))
        ):
            raise ValueError("Expected unique valid environment IDs")
        n = len(ids)
        expected = {
            "state": (n, 84),
            "next_state": (n, 84),
            "action_clean": (n, 29),
            "action_exec": (n, 29),
            "xi": (n, 33),
            "intent_ref": (n, 2),
        }
        values = locals()
        for name, shape in expected.items():
            value = values[name]
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {name}")
        for value in (frame, clip, physical_failure, timeout, clip_end):
            if value.shape != (n,):
                raise ValueError("Expected one identity/mask per transition")
        if (
            frame.dtype != torch.long
            or clip.dtype != torch.long
            or torch.any(frame < 0)
            or torch.any(clip < 0)
        ):
            raise ValueError("Nonnegative integer reference identities required")
        if any(x.dtype != torch.bool for x in (physical_failure, timeout, clip_end)):
            raise ValueError("Terminal reason masks must be boolean")
        if set(labels) != set(FORCE_SHAPES):
            raise ValueError("Incomplete force schema")
        for name, shape in FORCE_SHAPES.items():
            if labels[name].shape != (n, *shape) or not torch.isfinite(labels[name]).all():
                raise ValueError(f"Invalid {name}")
        p = self.ptr[ids].clone()
        prev = (p - 1) % self.T
        continuing = ~self._episode_start[ids]
        if continuing.any():
            e, t = ids[continuing], prev[continuing]
            if (
                not torch.equal(self.fields["next_state"][e, t], state[continuing])
                or not torch.equal(self.xi[e, t], xi[continuing])
                or not torch.equal(self.frame[e, t] + 1, frame[continuing])
                or not torch.equal(self.clip[e, t], clip[continuing])
            ):
                raise ValueError("Discontinuous state/context/frame/clip; begin a new episode")
        terminal = physical_failure | timeout | clip_end
        payload = {
            "next_state": next_state,
            **labels,
            "intent_ref": intent_ref,
            "physical_failure": physical_failure,
            "timeout": timeout,
            "clip_end": clip_end,
            "transition_terminal": terminal,
            "transition_valid": torch.ones_like(terminal),
        }
        # All validation precedes ring mutation. Assignment owns each input copy.
        self.add(
            ids,
            state,
            action_clean,
            action_exec,
            xi,
            torch.zeros(n, device=self.device),
            physical_failure.float(),
            frame,
            clip,
            torch.zeros_like(terminal),
            self._episode_start[ids],
        )
        for name, value in payload.items():
            self.fields[name][ids, p] = value
        self._episode_start[ids] = False
        if terminal.any():
            e = ids[terminal]
            tail = self.ptr[e].clone()
            z = torch.zeros(len(e), 29, device=self.device)
            self.add(
                e,
                next_state[terminal],
                z,
                z,
                xi[terminal],
                torch.zeros(len(e), device=self.device),
                torch.zeros(len(e), device=self.device),
                frame[terminal] + 1,
                clip[terminal],
                torch.ones(len(e), dtype=torch.bool, device=self.device),
                torch.zeros(len(e), dtype=torch.bool, device=self.device),
            )
            for field in self.fields.values():
                field[e, tail] = 0
            self.end_episode(e)

    @torch.no_grad()
    def window_mask(self, k=8):
        if not 1 <= k < self.T:
            raise ValueError("Window length must be positive and smaller than capacity")
        ok = super().window_mask(k)
        for j in range(k + 1):
            ok &= self._rolled(self.clip, j) == self.clip
            ok &= self._rolled(self.frame, j) == self.frame + j
            if j < k:
                ok &= self._rolled(self.fields["transition_valid"], j)
            if j < k - 1:
                ok &= ~self._rolled(self.fields["transition_terminal"], j)
        return ok

    @torch.no_grad()
    def sample_windows(self, n, k=8):
        # Reuse original eligible-row distribution and ring offset convention.
        e, t = self._sample_from_mask(self.window_mask(k), n)
        return self._gather_windows(e, t, k)

    @torch.no_grad()
    def _gather_windows(self, e, t, k):
        """Gather already-validated start rows; also used by exhaustive evaluation."""
        tt = (t[None, :] + torch.arange(k + 1, device=self.device)[:, None]) % self.T
        ee = e[None, :].expand_as(tt)
        er, tr = ee[:-1], tt[:-1]
        batch = {name: value[er, tr] for name, value in self.fields.items()}
        batch.update(
            states=self.s[ee, tt],
            a_clean=self.a[er, tr],
            a_exec=self.a_exec[er, tr],
            xi=self.xi[e, t],
            frame=self.frame[er, tr],
            clip=self.clip[er, tr],
            env_id=er.clone(),
            episode_id=self.ep_id[er, tr],
            episode_start=self.ep_start[er, tr],
        )
        return batch
