import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from p5_fixtures import REFERENCE, fixture, save_evidence, start_batch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.hoi.reference import ReferenceClip, array_info, sha256
from supermimic.hoi.reward_reference import (
    ClipBinding,
    HOIReferenceRegistry,
    RewardReferenceCache,
    derive_reference_twist,
)
from supermimic.hoi.tracking_reward import safe_quat_box_minus
from supermimic.kinematics.rotations import quat_apply, quat_from_rotvec, quat_mul, quat_normalize


def test_all_840_provenance_endpoints_signs_and_immutable_gathers():
    fx = fixture()
    cache = fx.registry.caches[7]
    snap = cache.snapshot()
    assert (
        cache.provenance["source_sha256"]
        == "bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b"
    )
    assert sha256(REFERENCE) == cache.provenance["source_sha256"]
    assert cache.n_frames == 840 and int(snap["contact"].sum()) == 290
    p = torch.tensor(fx.clip.arrays["object_pos_w"].copy(), dtype=torch.float64)
    q = torch.tensor(fx.clip.arrays["object_quat_w"].copy(), dtype=torch.float64)
    signs = torch.where(torch.arange(840) % 2 == 0, 1.0, -1.0)[:, None]
    derived = derive_reference_twist(p, q, 0.02)
    flipped = derive_reference_twist(p, q * signs, 0.02)
    for x, y in zip(derived[2:], flipped[2:]):
        assert torch.equal(x, y)
    _, qn, v, w, b = derived
    for f, lo, hi in ((0, 0, 1), (839, 838, 839), (260, 259, 261), (550, 549, 551)):
        torch.testing.assert_close(
            v[f], (p[hi] - p[lo]) / ((hi - lo) * 0.02), atol=1e-10, rtol=1e-8
        )
        torch.testing.assert_close(
            w[f], safe_quat_box_minus(qn[hi], qn[lo]) / ((hi - lo) * 0.02), atol=1e-10, rtol=1e-8
        )
    torch.testing.assert_close(quat_apply(qn, b), w, atol=1e-10, rtol=1e-8)
    allref = cache.gather(torch.arange(840))
    assert all(
        not x.requires_grad
        for x in (allref.position, allref.quaternion, allref.linear_world, allref.angular_world)
    )
    allref.position.zero_()
    snap["position"].fill_(9)
    cache.gather(torch.tensor(0)).position.fill_(8)
    torch.testing.assert_close(cache.snapshot()["position"], p, atol=0, rtol=0)
    for bad in (-1, 840):
        with pytest.raises(IndexError):
            cache.gather(torch.tensor([bad]))
    with pytest.raises(ValueError):
        RewardReferenceCache(fx.clip, dt=0.01, device="cpu", dtype=torch.float64)
    save_evidence(
        "reference_840",
        {
            "provenance": cache.provenance,
            "frames_checked": 840,
            "contact_on": int(cache.snapshot()["contact"].sum()),
            "contact_off": 550,
            "all_finite": bool(torch.isfinite(torch.cat((v, w, b), -1)).all()),
            "quaternion_sign_invariance": "exact",
            "scalar_and_batch_gather_isolation": True,
            "endpoints": {"v": v[[0, -1]].tolist(), "omega_world": w[[0, -1]].tolist()},
            "maximum_speed": float(v.norm(dim=-1).max()),
            "maximum_angular_speed": float(w.norm(dim=-1).max()),
        },
    )


def test_constant_twist_and_convergence():
    dtype = torch.float64
    base = quat_normalize(torch.tensor([0.6, 0.2, -0.3, 0.7], dtype=dtype))
    t = torch.arange(41, dtype=dtype) * 0.02
    vel = torch.tensor([0.2, -0.3, 0.5], dtype=dtype)
    omega = torch.tensor([0.4, -0.6, 0.2], dtype=dtype)
    p = 1 + t[:, None] * vel
    q = quat_mul(quat_from_rotvec(t[:, None] * omega), base.expand(len(t), 4))
    _, _, v, w, _ = derive_reference_twist(p, q, 0.02)
    torch.testing.assert_close(v, vel.expand_as(v), atol=5e-13, rtol=0)
    torch.testing.assert_close(w, omega.expand_as(w), atol=5e-13, rtol=0)
    errors = []
    for h in (0.02, 0.01, 0.005):
        t = torch.arange(round(1 / h) + 1, dtype=dtype) * h
        p = torch.stack((t**3, t**2, 0.4 * t), -1)
        vtrue = torch.stack((3 * t**2, 2 * t, torch.full_like(t, 0.4)), -1)
        ax = torch.tensor([0.8, 0, 0], dtype=dtype)
        ay = torch.tensor([0, -0.7, 0], dtype=dtype)
        qx = quat_from_rotvec(t[:, None] * ax)
        qy = quat_from_rotvec(t[:, None] * ay)
        q = quat_mul(quat_mul(qx, qy), base.expand(len(t), 4))
        wtrue = ax + quat_apply(qx, ay.expand(len(t), 3))
        _, _, v, w, _ = derive_reference_twist(p, q, h)
        errors.append(
            [
                float((x[sl] - y[sl]).abs().max())
                for x, y, sl in (
                    (v, vtrue, slice(1, -1)),
                    (w, wtrue, slice(1, -1)),
                    (v, vtrue, [0, -1]),
                    (w, wtrue, [0, -1]),
                )
            ]
        )
    for i in (0, 1):
        for j in range(4):
            ratio = errors[i][j] / errors[i + 1][j]
            assert (3.9 < ratio < 4.1) if j < 2 else (1.9 < ratio < 2.1)


def synthetic_clip(tmp_path, original):
    """Independent 840-frame source+manifest fixture; never overwrite accepted NPZ."""
    arrays = {k: v.copy() for k, v in original.arrays.items()}
    arrays["object_pos_w"] += np.array([3, 2, 1], dtype=np.float32)
    arrays["body_pos_w"] += np.array([1, 2, 3], dtype=np.float32)
    arrays["object_contact"] = ~arrays["object_contact"]
    path = tmp_path / "second"
    path.mkdir()
    np.savez(path / "reference.npz", **arrays)
    manifest = dict(
        original.manifest,
        sha256=sha256(path / "reference.npz"),
        fields={k: array_info(v) for k, v in arrays.items()},
    )
    (path / "reference_manifest.json").write_text(json.dumps(manifest))
    return ReferenceClip(path / "reference.npz")


def test_two_clip_identity_mapping_and_rejections(tmp_path):
    fx = fixture()
    second = synthetic_clip(tmp_path, fx.clip)
    sources = [second.path, REFERENCE]
    lib = MotionLibrary(sources, fx.lib.body_names, fx.q0, device="cpu", dtype=torch.float64)
    bindings = [ClipBinding(7, 1, fx.clip), ClipBinding(42, 0, second)]
    kwargs = {"library_sources": sources, "dt": 0.02, "joint_names": fx.joint_names}
    reg = HOIReferenceRegistry(lib, bindings, **kwargs)
    ids = torch.tensor([7, 42, 42, 7])
    frames = torch.tensor([260, 260, 550, 550])
    r = reg.gather(ids, frames)
    assert r.global_frame.tolist() == [1100, 260, 550, 1390]
    assert r.box.contact.tolist() == [True, False, True, False]
    from supermimic.hoi.imagination import HOIPolicyTrainer
    from supermimic.imagination.policy_trainer import frozen_parameters

    fx.lib, fx.registry = lib, reg
    trainer = HOIPolicyTrainer(fx.wm, fx.policy, fx.critic, reg, fx.reward, fx.trainer.cfg)
    start = start_batch(fx, (259, 549, 838, 838), clip_ids=(7, 42, 7, 42))
    with frozen_parameters(fx.wm, fx.critic):
        _, _, ro, _, _ = trainer.actor_loss(start)
    assert ro.diagnostics["evaluated_global_frame"][0].tolist() == [1100, 550, 1679, 839]
    assert ro.diagnostics["contact"][0].tolist() == [True, True, False, True]
    assert ro.lengths.tolist() == [3, 3, 1, 1]
    save_evidence(
        "multi_clip",
        {
            "global_frames": ro.diagnostics["evaluated_global_frame"].tolist(),
            "local_frames": ro.diagnostics["evaluated_local_frame"].tolist(),
            "contact": ro.diagnostics["contact"].tolist(),
            "valid": ro.valid.tolist(),
            "registry": reg.metadata(),
        },
    )
    torch.testing.assert_close(
        r.box.position[1] - r.box.position[0],
        torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    )
    for cs, fs in (([7], [-1]), ([42], [840]), ([99], [0])):
        with pytest.raises((IndexError, ValueError)):
            reg.gather(torch.tensor(cs), torch.tensor(fs))
    for bad in ([bindings[0], bindings[0]], [replace(bindings[0], library_id=0), bindings[1]]):
        with pytest.raises(ValueError):
            HOIReferenceRegistry(lib, bad, **kwargs)
    with pytest.raises(ValueError, match="SHA"):
        HOIReferenceRegistry(lib, bindings, **{**kwargs, "library_sources": sources[::-1]})
    with pytest.raises(ValueError, match="order"):
        HOIReferenceRegistry(lib, bindings, **{**kwargs, "joint_names": fx.joint_names[::-1]})
    lib.body_pos_w[0, 0, 0] += 1
    with pytest.raises(ValueError, match="body tensors"):
        HOIReferenceRegistry(lib, bindings, **kwargs)


@pytest.mark.parametrize(
    "case", ["short", "nan", "zero", "norm", "dt", "pi_adjacent", "pi_stencil"]
)
def test_invalid_derivative_inputs(case):
    p = torch.zeros(3, 3, dtype=torch.float64)
    q = torch.zeros(3, 4, dtype=torch.float64)
    q[:, 0] = 1
    dt = 0.02
    if case == "short":
        p, q = p[:1], q[:1]
    elif case == "nan":
        p[0, 0] = float("nan")
    elif case == "zero":
        q[0] = 0
    elif case == "norm":
        q[0, 0] = 2
    elif case == "dt":
        dt = float("inf")
    elif case == "pi_adjacent":
        q[1] = torch.tensor([0.0, 1.0, 0, 0])
    else:
        q = quat_from_rotvec(
            torch.tensor([[0.0, 0, 0], [math.pi / 2, 0, 0], [math.pi, 0, 0]], dtype=torch.float64)
        )
    with pytest.raises(ValueError):
        derive_reference_twist(p, q, dt)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_reference_registration_and_eager_rollout(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from supermimic.imagination.policy_trainer import frozen_parameters

    fx = fixture(dtype, "cuda")  # exercise unindexed library device alias
    start = start_batch(fx)
    assert fx.registry.device == start.state.device
    with frozen_parameters(fx.wm, fx.critic):
        loss, _, ro, _, _ = fx.trainer.actor_loss(start)
        loss.backward()
    assert ro.lengths.tolist() == [3, 3, 3, 3, 1]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fx.policy.parameters())
    fx.lib.goal_feat[0, 0] += 0.1
    with pytest.raises(ValueError, match="goals"):
        HOIReferenceRegistry(
            fx.lib,
            [ClipBinding(7, 0, fx.clip)],
            library_sources=[REFERENCE],
            dt=0.02,
            joint_names=fx.joint_names,
        )
