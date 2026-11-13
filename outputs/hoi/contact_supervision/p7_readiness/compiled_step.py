"""One-signature CUDA FP32 P5 parity probe; synthetic coupling, backward only."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/hoi"))
from p5_fixtures import fixture, start_batch

from supermimic.hoi.imagination import HOIImagStep
from supermimic.imagination.policy_trainer import frozen_parameters

torch._dynamo.reset()
torch._dynamo.utils.counters.clear()
assert not torch._dynamo.config.suppress_errors
fx = fixture(torch.float32, "cuda", horizon=1)
start = start_batch(fx, (259, 549))
ref = fx.registry.gather(start.clip_id, start.local_frame + 1)
compiled = HOIImagStep(fx.wm, fx.q0, fx.reward, compile_mode="default")
results = []
with frozen_parameters(fx.wm):
    for step in (fx.trainer.step_fn, compiled):
        state = start.state.clone().requires_grad_()
        action = torch.full((2, 29), 0.031, device=state.device, requires_grad=True)
        outputs = step(state, action, start.prev_action, start.xi, ref)
        gradients = torch.autograd.grad(outputs[1].sum(), (state, action))
        results.append((*outputs, *gradients))
errors = []
for a, b in zip(*results):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    errors.append(float((a - b).detach().abs().max()))
graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
assert graphs > 0
unimplemented = dict(torch._dynamo.utils.counters["unimplemented"])
assert not any("recompile" in k or "cache_size_limit" in k for k in unimplemented)
report = {
    "status": "PASS",
    "device": "cuda",
    "dtype": "float32",
    "batch": 2,
    "horizon": 1,
    "backend": "inductor",
    "atol": 2e-5,
    "rtol": 2e-5,
    "max_errors_outputs_then_state_action_gradients": errors,
    "unique_graphs": graphs,
    "unimplemented": unimplemented,
    "fullgraph_claim": False,
    "optimizer_updates": 0,
    "model": "P5 synthetic coupled residual; not trained dynamics",
}
print(json.dumps(report, indent=2))
(Path(__file__).parent / "compiled_step.json").write_text(json.dumps(report, indent=2) + "\n")
