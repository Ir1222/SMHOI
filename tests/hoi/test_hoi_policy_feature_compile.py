"""Full-graph proof for new pure feature kernels, separate from legacy FK guards."""

import pytest
import torch
from p7_policy_fixtures import fixture, save_evidence, start_batch

from supermimic.hoi.policy_features import critic_features


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
def test_new_feature_kernels_fullgraph(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; feature compilation NOT TESTED")
    torch._dynamo.reset()
    fx = fixture(dtype, device, horizon=1)
    st = start_batch(fx, (300, 301))
    ref = fx.registry.gather(st.clip_id, st.local_frame)
    adapter = fx.trainer.observation_adapter

    def features(s, a, xi):
        obs = adapter.observation(s, a, xi)
        hist = adapter.flatten_history(obs[None].expand(10, -1, -1))
        feat = critic_features(s, a, ref, fx.wm.spec, fx.wm.support)
        return obs, hist, feat

    compiled = torch.compile(features, backend="inductor", fullgraph=True)
    runs = []
    for fn in (features, compiled):
        s = st.state.clone().requires_grad_()
        a = st.prev_action.clone().requires_grad_()
        out = fn(s, a, st.xi)
        gradients = torch.autograd.grad(sum(x.square().sum() for x in out), (s, a))
        runs.append((*out, *gradients))
    atol, rtol = (5e-6, 5e-6) if dtype == torch.float32 else (1e-10, 1e-8)
    errors = {}
    for name, x, y in zip(
        ("observation", "history", "critic_features", "state_gradient", "action_gradient"), *runs
    ):
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol, msg=name)
        errors[name] = float((x - y).detach().abs().max())
    save_evidence(
        f"feature_fullgraph_{device}_{str(dtype).split('.')[-1]}",
        {
            "status": "PASS",
            "backend": "inductor",
            "fullgraph": True,
            "device": device,
            "dtype": str(dtype),
            "atol": atol,
            "rtol": rtol,
            "max_absolute_errors": errors,
            "optimizer_steps": 0,
        },
    )
