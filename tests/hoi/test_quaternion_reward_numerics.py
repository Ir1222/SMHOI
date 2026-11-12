"""Approved P5 numerical matrix, including the original known identity NaN."""

import math

import pytest
import torch
from mjlab.utils.lab_api.math import quat_box_minus
from p5_fixtures import save_evidence

from supermimic.hoi.tracking_reward import safe_squared_orientation_error as sq
from supermimic.kinematics.rotations import quat_from_rotvec, quat_mul, quat_normalize


def quaternion_matrix(dtype):
    torch.manual_seed(501)
    axis = torch.tensor([1.0, -2.0, 3.0], dtype=dtype)
    axis /= axis.norm()
    base = quat_normalize(torch.tensor([0.6, 0.2, -0.3, 0.7], dtype=dtype))
    identity = torch.tensor([1.0, 0, 0, 0], dtype=dtype)
    rows = []
    cases = [(identity, 0.0, s) for s in (1, -1)] + [(base, 0.0, s) for s in (1, -1)]
    cases += [
        (base, t, s)
        for t in (1e-9, 1e-7, 0.999e-6, 1.001e-6, 1e-5, 0.2, 1.3, math.pi - 0.001, math.pi + 0.001)
        for s in (1, -1)
    ]
    for ref, theta, sign in cases:
        x = (theta * axis).clone().requires_grad_()
        q = sign * quat_mul(quat_from_rotvec(x), ref)
        value = sq(ref, q)
        grad = torch.autograd.grad(value, x, retain_graph=True)[0]
        old = quat_box_minus(ref, q).square().sum(-1)
        old_grad = torch.autograd.grad(old, x)[0]
        angle = theta if theta < math.pi else theta - 2 * math.pi
        assert torch.isfinite(grad).all()
        tolerance = 5e-6 if dtype == torch.float32 else 1e-12
        assert abs(value.item() - angle**2) < tolerance
        assert (grad - 2 * angle * axis).abs().max() < tolerance
        assert torch.equal(value, old)
        rows.append(
            {
                "theta": theta,
                "sign": sign,
                "value": value.item(),
                "gradient_error": float((grad - 2 * angle * axis).abs().max()),
                "old_gradient_finite": bool(torch.isfinite(old_grad).all()),
            }
        )
    a = quat_normalize(torch.randn(128, 4, dtype=dtype)).requires_grad_()
    b = quat_normalize(torch.randn(128, 4, dtype=dtype)).requires_grad_()
    value = sq(a, b)
    ga, gb = torch.autograd.grad(value.sum(), (a, b))
    for sa, sb in ((-1, 1), (1, -1), (-1, -1)):
        aa = (sa * a.detach()).requires_grad_()
        bb = (sb * b.detach()).requires_grad_()
        v = sq(aa, bb)
        gaa, gbb = torch.autograd.grad(v.sum(), (aa, bb))
        assert torch.equal(v, value) and torch.equal(ga, sa * gaa) and torch.equal(gb, sb * gbb)
    exact = torch.cat((torch.zeros(1, dtype=dtype), axis)).requires_grad_()
    assert torch.isfinite(torch.autograd.grad(sq(identity, exact), exact)[0]).all()
    fd = []
    for theta in (0.0, 1e-5, 0.2, 1.3, math.pi - 0.01, math.pi + 0.01):
        x = (theta * axis).clone().requires_grad_()
        fun = lambda y: sq(base, quat_mul(quat_from_rotvec(y), base))
        g = torch.autograd.grad(fun(x), x)[0]
        eps = 1e-3 if dtype == torch.float32 else 1e-6
        eye = torch.eye(3, dtype=dtype) * eps
        estimate = torch.stack(
            [(fun(x.detach() + e) - fun(x.detach() - e)) / (2 * eps) for e in eye]
        )
        fd.append(float((g - estimate).abs().max()))
    assert max(fd) < (2e-3 if dtype == torch.float32 else 5e-9)
    if dtype == torch.float64:
        for theta in (0.0, 1e-7, 0.2, 1.3, math.pi - 0.001, math.pi + 0.001):
            x = (theta * axis).clone().requires_grad_()
            assert torch.autograd.gradcheck(
                lambda y: sq(base, quat_mul(quat_from_rotvec(y), base)),
                (x,),
                eps=1e-6,
                atol=1e-7,
                rtol=1e-5,
            )
    assert not rows[0]["old_gradient_finite"]
    return {
        "cases": rows,
        "max_fd_error": max(fd),
        "random_sign_pairs": 128,
        "exact_pi_finite_branch": True,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quaternion_matrix(dtype):
    save_evidence(f"quaternion_{str(dtype).split('.')[-1]}", quaternion_matrix(dtype))
