"""Immutable derivative caches and explicit Replay/local/library frame binding.

Only construction/gathering runs here, outside the compiled imagined step.
ReferenceClip remains the production NPZ/manifest validator. No file is written.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from supermimic.hoi.reference import ReferenceClip, sha256
from supermimic.hoi.rewards import reference_contact_at
from supermimic.hoi.tracking_reward import BoxReference, safe_quat_box_minus, unit_quaternion_copy
from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.world_model.wm_trainer import reference_at

DERIVATIVE_VERSION = "central-world-log-v1"


def derive_reference_twist(position, quaternion, dt):
    """FP64 centered world increments, one-sided endpoints; normalized copies.

    The two-frame stencil must also stay below the shortest-log pi branch cut.
    Pose samples cannot exclude unobserved between-sample full rotations.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Positive finite dt required")
    p = torch.as_tensor(position).detach().to(dtype=torch.float64).clone()
    q = torch.as_tensor(quaternion).detach().to(device=p.device, dtype=torch.float64).clone()
    if p.ndim != 2 or p.shape != (len(q), 3) or q.shape != (len(p), 4) or len(p) < 2:
        raise ValueError("At least two aligned pose samples required")
    if not torch.isfinite(p).all():
        raise ValueError("Nonfinite reference position")
    q = unit_quaternion_copy(q)
    adjacent = safe_quat_box_minus(q[1:], q[:-1]).norm(dim=-1)
    stencil = safe_quat_box_minus(q[2:], q[:-2]).norm(dim=-1)
    if torch.any(adjacent >= math.pi - 1e-12) or torch.any(stencil >= math.pi - 1e-12):
        raise ValueError("Reference increment reaches the pi branch cut")
    f = torch.arange(len(p), device=p.device)
    lo, hi = (f - 1).clamp_min(0), (f + 1).clamp_max(len(p) - 1)
    span = ((hi - lo).to(torch.float64) * dt)[:, None]
    v = (p[hi] - p[lo]) / span
    omega = safe_quat_box_minus(q[hi], q[lo]) / span
    return p, q, v, omega, quat_apply_inverse(q, omega)


def _digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


class RewardReferenceCache:
    """Private constant tensors; gathers and public snapshots never alias storage."""

    def __init__(self, clip: ReferenceClip, *, dt, device, dtype):
        if not isinstance(clip, ReferenceClip):
            raise TypeError("Production cache requires a validated ReferenceClip")
        if (
            not math.isfinite(clip.fps)
            or clip.fps <= 0
            or not math.isclose(1 / clip.fps, dt, abs_tol=1e-12, rel_tol=0)
        ):
            raise ValueError("Reference/control timestep mismatch")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Reward cache supports FP32/FP64 only")
        if not _digest(clip.manifest.get("sha256")) or sha256(clip.path) != clip.manifest["sha256"]:
            raise ValueError("Missing or changed reference SHA")
        a = clip.arrays
        p, q, v, w, b = derive_reference_twist(
            a["object_pos_w"].copy(), a["object_quat_w"].copy(), dt
        )
        self._values = tuple(x.to(device=device, dtype=dtype) for x in (p, q, v, w, b))
        self._contact = torch.tensor(a["object_contact"].copy(), device=device, dtype=torch.bool)
        self.n_frames, self.dt = len(p), dt
        self.device, self.dtype = self._contact.device, dtype
        self._provenance = MappingProxyType(
            {
                "source_path": str(clip.path),
                "source_sha256": clip.manifest["sha256"],
                "manifest_sha256": sha256(clip.path.parent / "reference_manifest.json"),
                "frames": len(p),
                "fps": clip.fps,
                "dt_s": dt,
                "derivation": DERIVATIVE_VERSION,
                "compute_dtype": "torch.float64",
                "storage_dtype": str(dtype),
                "quaternion": "wxyz; normalized copies only",
                "position_velocity": "root origin; environment-local world axes; m and m/s",
                "angular_velocity": "world axes; rad/s; body audit uses R(q_f)^T",
            }
        )

    @property
    def provenance(self):
        return dict(self._provenance)

    def snapshot(self):
        return {
            k: x.clone()
            for k, x in zip(
                (
                    "position",
                    "quaternion",
                    "linear_world",
                    "angular_world",
                    "angular_body",
                    "contact",
                ),
                (*self._values, self._contact),
            )
        }

    def gather(self, local_frame):
        contact = reference_contact_at(self._contact, local_frame)
        return BoxReference(*(x[local_frame].clone() for x in self._values[:4]), contact.clone())


@dataclass(frozen=True)
class ClipBinding:
    replay_id: int
    library_id: int
    reference: ReferenceClip


@dataclass(frozen=True)
class HOIReferenceFrame:
    robot: object
    box: BoxReference
    clip_id: torch.Tensor
    local_frame: torch.Tensor
    global_frame: torch.Tensor


class HOIReferenceRegistry:
    """Validate the actual library tensors as well as its explicit ordered sources.

    Duplicate source bytes are allowed only with distinct explicit replay/library
    IDs (P2 can record two identities for one motion). Duplicate ID bindings fail.
    Packed/compact robot libraries require their own provenance contract and are
    deliberately rejected here; the accepted Part B is a full NPZ library.
    """

    def __init__(self, library, bindings, *, library_sources, dt, joint_names):
        self.lib, self.dt = library, dt
        # The original library can retain the unindexed alias "cuda" while its
        # tensors live on "cuda:0". Compare actual resolved tensor devices.
        self.device = library.goal_feat.device if not library.compact else library.device
        self.bindings = tuple(bindings)
        if library.compact:
            raise ValueError("HOI requires a full source-verified NPZ library")
        if not self.bindings or len(library_sources) != library.num_clips:
            raise ValueError("Explicit ordered library sources/bindings required")
        ids = [b.replay_id for b in self.bindings]
        lids = [b.library_id for b in self.bindings]
        if (
            len(set(ids)) != len(ids)
            or any(type(i) is not int or i < 0 for i in ids)
            or sorted(lids) != list(range(library.num_clips))
        ):
            raise ValueError("Missing, duplicate or ambiguous clip identity")
        if len(joint_names) != 29 or len(set(joint_names)) != 29:
            raise ValueError("Explicit native joint names/order required")
        self._by_id = {b.replay_id: b for b in self.bindings}
        self.caches = {}
        first_body_names = None
        for b in self.bindings:
            clip = b.reference
            if not isinstance(clip, ReferenceClip):
                raise TypeError("Validated ReferenceClip required")
            source = Path(library_sources[b.library_id]).resolve()
            if sha256(source) != clip.manifest["sha256"]:
                raise ValueError("Robot/Box reference source SHA mismatch")
            a = clip.arrays
            names = tuple(a["body_names"].tolist())
            if tuple(a["joint_names"].tolist()) != tuple(joint_names) or (
                first_body_names is not None and names != first_body_names
            ):
                raise ValueError("Joint/body names or ordering mismatch")
            first_body_names = names
            info = library.clips[b.library_id]
            if info.length != clip.n_frames or int(library.clip_len[b.library_id]) != clip.n_frames:
                raise ValueError("Reference/library length mismatch")
            frames = torch.arange(info.start, info.end, device=library.device)
            if (
                int(library.clip_start[b.library_id]) != info.start
                or not (library.frame_clip[frames] == b.library_id).all()
            ):
                raise ValueError("Inconsistent library global frame mapping")
            pelvis = names.index("pelvis")
            expected = np.concatenate(
                (
                    a["body_pos_w"][:, pelvis],
                    a["body_quat_w"][:, pelvis],
                    a["joint_pos"],
                    a["body_lin_vel_w"][:, pelvis],
                    a["body_ang_vel_w"][:, pelvis],
                    a["joint_vel"],
                ),
                -1,
            ).astype(np.float32)  # original library's exact cast
            if not torch.equal(
                library.ref71(frames),
                torch.tensor(expected, device=library.device, dtype=library.dtype),
            ):
                raise ValueError("Library robot tensors do not match bound source")
            from supermimic.envs.mdp.motion_library import goal_features

            expected_goal = goal_features(library.ref(frames), library.default_joint_pos)
            # Raw source tensors above remain bit-exact. Derived goals may vary
            # by CUDA TorchScript profiling/fusion roundoff on recomputation.
            # Use the specification's existing dtype algebra tolerances, without
            # modifying the stored goals or relaxing compiled parity tolerances.
            atol, rtol = (5e-6, 5e-6) if library.dtype == torch.float32 else (1e-10, 1e-8)
            if not torch.allclose(library.goal_feat[frames], expected_goal, atol=atol, rtol=rtol):
                raise ValueError("Library goals do not match bound source")
            idx = [names.index(n) for n in library.body_names]
            for key, actual in zip(
                ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"),
                (
                    library.body_pos_w,
                    library.body_quat_w,
                    library.body_lin_vel_w,
                    library.body_ang_vel_w,
                ),
            ):
                expected_body = torch.tensor(
                    a[key][:, idx].copy(), device=library.device, dtype=library.dtype
                )
                if not torch.equal(actual[frames], expected_body):
                    raise ValueError("Library body tensors do not match bound source")
            self.caches[b.replay_id] = RewardReferenceCache(
                clip, dt=dt, device=library.device, dtype=library.dtype
            )

    def _check(self, clip_id, local_frame):
        if (
            clip_id.shape != local_frame.shape
            or clip_id.dtype != torch.long
            or local_frame.dtype != torch.long
            or clip_id.device != self.device
            or local_frame.device != self.device
        ):
            raise ValueError("Aligned long clip/local-frame tensors required on library device")
        for c in clip_id.unique().tolist():
            if c not in self._by_id:
                raise ValueError(f"Unknown replay clip identity {c}")
            f = local_frame[clip_id == c]
            if torch.any((f < 0) | (f >= self.caches[c].n_frames)):
                raise IndexError("Local reference frame outside clip; no clamping")

    def global_frames(self, clip_id, local_frame):
        self._check(clip_id, local_frame)
        result = torch.empty_like(local_frame)
        for c in clip_id.unique().tolist():
            selected = clip_id == c
            lid = self._by_id[c].library_id
            result[selected] = self.lib.clip_start[lid] + local_frame[selected]
            if not (self.lib.frame_clip[result[selected]] == lid).all():
                raise ValueError("Global frame belongs to a different clip")
        return result

    def lengths(self, clip_id):
        self._check(clip_id, torch.zeros_like(clip_id))
        out = torch.empty_like(clip_id)
        for c in clip_id.unique().tolist():
            out[clip_id == c] = self.caches[c].n_frames
        return out

    def hashes(self, clip_id):
        self._check(clip_id, torch.zeros_like(clip_id))
        return tuple(self.caches[c].provenance["source_sha256"] for c in clip_id.tolist())

    def gather(self, clip_id, local_frame):
        global_frame = self.global_frames(clip_id, local_frame)
        fields = [
            torch.empty(*local_frame.shape, w, device=self.lib.device, dtype=self.lib.dtype)
            for w in (3, 4, 3, 3)
        ]
        contact = torch.empty_like(local_frame, dtype=torch.bool)
        for c in clip_id.unique().tolist():
            selected = clip_id == c
            ref = self.caches[c].gather(local_frame[selected])
            for out, value in zip(
                fields, (ref.position, ref.quaternion, ref.linear_world, ref.angular_world)
            ):
                out[selected] = value
            contact[selected] = ref.contact
        return HOIReferenceFrame(
            reference_at(self.lib, global_frame),
            BoxReference(*fields, contact),
            clip_id.clone(),
            local_frame.clone(),
            global_frame,
        )

    def metadata(self):
        return [
            {
                "replay_id": b.replay_id,
                "library_id": b.library_id,
                "global_start": self.lib.clips[b.library_id].start,
                **self.caches[b.replay_id].provenance,
            }
            for b in self.bindings
        ]
