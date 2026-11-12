"""Actually executed torch.compile/Inductor parity; no physical simulation."""

import os

import pytest
import torch
from p5_fixtures import fixture, save_evidence, start_batch

from supermimic.hoi.imagination import HOIImagStep
from supermimic.imagination.policy_trainer import frozen_parameters


def run_parity(dtype, device):
    # Each device/dtype is an independent experiment, not another specialization
    # consuming a previous experiment's Dynamo recompile budget.
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    assert not torch._dynamo.config.suppress_errors
    fx = fixture(dtype, device, horizon=3)
    eager = fx.trainer.step_fn
    compiled = HOIImagStep(fx.wm, fx.q0, fx.reward, compile_mode="default", backend="inductor")
    atol, rtol = (2e-5, 2e-5) if dtype == torch.float32 else (1e-9, 1e-7)
    result = {
        "device": device,
        "dtype": str(dtype),
        "backend": "inductor",
        "mode": "default",
        "dynamo_recompile_limit": torch._dynamo.config.recompile_limit,
        "atol": atol,
        "rtol": rtol,
        "test_model": "DeterministicCoupledResidual; no physical fidelity claim",
        "horizons": {},
        "optimizer_steps": 0,
    }
    for H in (1, 3):
        fx.trainer.cfg.horizon = H
        st = start_batch(fx)
        histories = []
        for step in (eager, compiled):
            fx.trainer.step_fn = step
            fx.policy.zero_grad(set_to_none=True)
            with frozen_parameters(fx.wm, fx.critic):
                loss, J, ro, _, _ = fx.trainer.actor_loss(st)
                # All steps' actions influence their own and later rewards/observations.
                for action in (ro.actions,):
                    action.retain_grad()
                loss.backward()
                grads = [p.grad.detach().clone() for p in fx.policy.parameters()]
            histories.append((loss.detach(), J.detach(), ro, grads))
        (le, je, re, ge), (lc, jc, rc, gc) = histories
        errors = {}

        def check(name, a, b, errors=errors):
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)
            errors[name] = float((a - b).detach().abs().max())

        check("loss", le, lc)
        check("objective", je, jc)
        for key in ("states", "actions", "rewards", "dhat", "survival", "goals", "e_root"):
            check(key, getattr(re, key), getattr(rc, key))
        for key in ("valid", "lengths", "bootstrap", "continuation", "frames"):
            assert torch.equal(getattr(re, key), getattr(rc, key)), key
        for key in re.diagnostics:
            a, b = re.diagnostics[key], rc.diagnostics[key]
            if a.dtype in (torch.bool, torch.long):
                assert torch.equal(a, b), key
            else:
                check(key, a, b)
        for i, (a, b) in enumerate(zip(ge, gc)):
            check(f"actor_gradient_{i}", a, b)
        # Direct action and state gradients + next obs are checked separately.
        st1 = start_batch(fx, (259, 549, 838))
        ref = fx.registry.gather(st1.clip_id, st1.local_frame + 1)
        pair = []
        with frozen_parameters(fx.wm):
            for step in (eager, compiled):
                s = st1.state.clone().requires_grad_()
                a = s.new_full((3, 29), 0.031).requires_grad_()
                outputs = step(s, a, st1.prev_action, st1.xi, ref)
                gs, ga = torch.autograd.grad(outputs[1].sum(), (s, a))
                pair.append((outputs, gs, ga))
        for i, (a, b) in enumerate(zip(pair[0][0], pair[1][0])):
            check(f"step_output_{i}", a, b)
        check("state_gradient", pair[0][1], pair[1][1])
        check("action_gradient", pair[0][2], pair[1][2])
        result["horizons"][str(H)] = {
            "status": "PASS",
            "max_absolute_errors": errors,
            "valid_lengths": re.lengths.tolist(),
            "evaluated_frames": re.diagnostics["evaluated_local_frame"].tolist(),
            "actor_gradient_norm": float(torch.stack([g.square().sum() for g in ge]).sum().sqrt()),
        }
    result["compiled_unique_graphs"] = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    result["compiler_unimplemented"] = dict(torch._dynamo.utils.counters["unimplemented"])
    assert result["compiled_unique_graphs"] > 0
    assert not any(
        "recompile" in key or "cache_size_limit" in key for key in result["compiler_unimplemented"]
    ), "Compiler budget fallback must not count as compiled parity"
    result["geometry_diagnostics_compiled"] = True
    # Explain the same tensor signature; graph breaks are expected from preserved guards.
    st = start_batch(fx, (259, 549, 838))
    ref = fx.registry.gather(st.clip_id, st.local_frame + 1)
    r, b = ref.robot, ref.box
    with frozen_parameters(fx.wm):
        explanation = torch._dynamo.explain(eager._step)(
            st.state,
            st.prev_action,
            st.prev_action,
            st.xi,
            r.body_pos_w,
            r.body_quat_w,
            r.body_lin_vel_w,
            r.body_ang_vel_w,
            r.root_pos_w,
            r.root_quat_w,
            b.position,
            b.quaternion,
            b.linear_world,
            b.angular_world,
            b.contact,
        )
    result["graph_count"] = explanation.graph_count
    result["graph_break_count"] = explanation.graph_break_count
    result["graph_break_reasons"] = [str(x.reason) for x in explanation.break_reasons]
    result["single_full_graph_claim"] = False
    return result


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inductor_parity(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; GPU parity NOT TESTED")
    # Optional selection for separately retained runs; no backend fallback.
    requested = os.environ.get("P5_PARITY_DEVICE")
    if requested and requested != device:
        pytest.skip("Device tested in a separate invocation")
    # Several active batch widths, differentiable inputs, and separately compiled
    # no-grad diagnostics legitimately specialize the shared normalization helper
    # more than eight times. Permit those graphs, rather than silently accepting
    # Dynamo's default eager fallback. This validation-only scope restores the
    # original compiler setting and changes no numerical tolerance or production
    # training default.
    with torch._dynamo.config.patch(recompile_limit=64):
        result = run_parity(dtype, device)
    save_evidence(f"parity_{device}_{str(dtype).split('.')[-1]}", result)
