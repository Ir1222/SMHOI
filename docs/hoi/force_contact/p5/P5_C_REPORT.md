# P5-C — BPTT, boundaries and validation

Software validation: **PASS** — 41 eager acceptance tests, four actual Inductor
cases, and 69 selected regressions; zero final failures/skips.

This report records bounded software verification only. No Actor/Critic Adam step,
WM update, physical HOI rollout, P6 or P7 execution is part of these results.
The final acceptance decision and test totals are in [REPORT](REPORT.md).

## Boundary and return implementation

For start `f`, horizon `H`, known remaining timeout `K` when supplied, and an
authoritative reached hard-failure step `F` when supplied:

`L=min(H,T-1-f,K,F)`, omitting unavailable events. `L>=1` is required.
`valid[k,n]=(k<L_n)`. Natural completion and hard failure have `b_L=0`;
horizon/timeout before completion have `b_L=1`. Coincident reasons remain separately
visible; natural completion or physical failure takes precedence over timeout
bootstrap. The inherited soft failure prediction is not thresholded into an event.
Hard failure sets that reached step's `d=1`. Natural completion does not do so.

The original recurrence remains `C_0=1`, `C_(k+1)=C_k*(1-d_(k+1))` and

\[
J=\operatorname{mean}_n\left[\sum_{k<L_n}\gamma^k C_{k,n}r_{k+1,n}
  +\gamma^{L_n}C_{L_n,n}b_{L_n,n}V_{L_n,n}\right],\qquad loss=-J/H.
\]

Nominal `H` still scales the Actor loss. The shared TD(lambda) recursion starts
each row at `G_L=b_L*V_L` and uses

\[
G_k=r_{k+1}+\gamma(1-d_{k+1})B_{k+1}
  [(1-\lambda)V_{k+1}+\lambda G_{k+1}],
\]

where `B=0` at natural completion/hard failure, otherwise 1. Both bootstrap
branches are masked. Padded rows reset the reverse recursion to that row's boundary
value and never become Critic samples. Masking selects operands before arithmetic,
so nonfinite padded values cannot contaminate valid results. Natural-end Actor
tails do not evaluate Critic at all. Tiny valid Critic batches use nonempty splits;
the original no-mask minibatch partition and optimizer order remain unchanged.

Tests check hand-calculated returns/targets, exact all-valid equality with the
original equations, timeout/horizon/natural/hard/coincident reasons, frame839
ineligibility, the single 838→839 reward, and no padded Actor/WM/reference evaluation.
They corrupt padded state/goal/root/reward/termination with NaNs and verify unchanged
valid Critic samples, targets and MSE. A separate Actor-gradient check gives invalid
rewards a large explicit Actor-weight dependency and invalid survival NaNs; masking
must preserve both the scalar and every Actor gradient exactly.

## Gradient evidence

`frozen_parameters(wm,critic)` preserves each parameter's previous flag and restores
it in `finally`, including exceptions and already-frozen parameters. No `no_grad`
wraps imagined dynamics or the Actor's Critic tail. References and detached logging
copies are constants; state, Box, FK, support, observations, previous actions,
survival and Critic input paths remain differentiable. Critic fitting separately
uses detached inputs/rewards/termination and fixed TD targets.

The deterministic fixture replaces only the Joint WM residual mapping with a
known nonzero action/state coupling. Existing support construction, integration,
FK, reward, Actor and Critic paths still execute. It has no learned physical
fidelity. FP64 measured isolated gradients:

| Component | Action gradient norm | Actor parameter gradient norm |
|---|---:|---:|
| Robot root position | 4.02231e-6 | 4.23409e-6 |
| Robot root orientation | 4.50881e-7 | 4.74617e-7 |
| Robot body position | 1.05744e-5 | 1.11312e-5 |
| Robot body orientation | 1.17125e-6 | 1.23291e-6 |
| Robot linear velocity | 4.39862e-5 | 4.63020e-5 |
| Robot angular velocity | 8.19389e-5 | 8.62524e-5 |
| Box position | 4.14218e-6 | 4.36024e-6 |
| Box orientation | 6.22350e-7 | 6.55124e-7 |
| Box linear velocity | 5.02145e-5 | 5.28579e-5 |
| Box angular velocity | 4.02334e-5 | 4.23517e-5 |
| Weighted proximity | 7.62583e-5 | 8.02729e-5 |

Separate branch tests isolate robot→EEF and Box→target action gradients. OFF
proximity gradients are exactly zero; a deliberate detached-state control has no
action connection. All four Box terms and proximity pass action finite differences
at `eps=1e-6, atol=1e-7, rtol=1e-5`. Multi-step checks establish later reward→earlier
action/history reachability, previous-action regularization, survival and tail paths.
`recurrence_paths.json` separately measures future-reward, survival and frozen
Critic-tail gradients to the first action: `2.8104164570e-4`, `6.3272103514e-5`
and `1.7421568529e-4`, respectively. These checks do not rely on the total reward
being nonzero to infer survival or bootstrap reachability.

The mixed-length H=3 full return gives Actor gradient norm `1.8901432098e-4` and
objective `0.2600760097`. WM/Critic weights do not change during Actor backward;
their gradient buffers remain absent. After clearing Actor buffers, detached
Critic MSE backward produces Critic gradients and **no Actor or WM gradients**.
Flags restore on success and an intentional exception. No optimizer step occurs.

The **actual initialized JointWorldModel**, with its zero output layer, separately
has zero action→Box Jacobian and zero four-Box-reward action gradients. This is
reported in `initialized_model.json`. It is not a trained model or a failure of
graph wiring. P3/P7-A learned prediction and gradient usefulness remain unvalidated.

## Quaternion and reference numerics

The executed FP32/FP64 matrix includes identity, antipodal identity, nonidentity
agreement and its antipode; angles `1e-9,1e-7,0.999e-6,1.001e-6,1e-5,0.2,1.3`,
and both sides of pi, with both signs. It also covers 128 random quaternion pairs,
gradient sign covariance, the finite exact-pi computational branch, tangent finite
differences and six FP64 gradchecks. Both robot orientation rewards attain maxima
with finite zero tangent gradients at agreement/antipode. The original helper's
exact-match nonfinite gradient is retained as a known baseline limitation.

Algebra tolerances remain FP32 `atol=rtol=5e-6`, FP64 `atol=1e-10, rtol=1e-8`.
Quaternion finite-difference maximum errors must remain below `2e-3` / `5e-9`.
No pi-cut smoothness or physical velocity-truth claim is made. All 840 reference
frames and derivative/sign/endpoint tests are covered, including distinct-source
multi-clip identities and local/global mapping.

## Actual compiled validation

`test_imagination_parity.py` runs actual `torch.compile`, backend `inductor`, mode
`default`, `dynamic=False`, on CPU and the available NVIDIA GeForce RTX 4090 D,
in FP32 and FP64. Each case compares H=1 and H=3 with starts spanning 260/550/839.
The six-item step and a separate tensor-signature diagnostic method are compiled.
Geometry comparison therefore covers compiled EEF/target positions and distances,
not merely eager diagnostics of a compiled state.

Fields include physical states, each of 13 reward components, total, observations,
anchors, bool gates, frame/masks, actions, survival, goals, root errors, direct
state/action gradients and every Actor parameter gradient. Bool/masks/indices
must be exact; FP64 uses `atol=1e-9, rtol=1e-7`, FP32 `atol=rtol=2e-5`.
All gradients must be finite. These tolerances do not replace the historical P3
FP32 migration tolerance `2e-6`.

Each device/dtype starts a fresh Dynamo experiment, while reusing on-disk Inductor
artifacts. The validation-only `recompile_limit=64` is recorded and restored after
each case. The default budget of 8 was actually exhausted by active batch widths,
input gradient flags and separately compiled no-grad geometry sharing the same
quaternion helper. Raising this budget permits more actual compiled graphs; it
does not change formulas, tolerances or production training/compiler defaults.
The default-budget path can fall back to eager and is a performance limitation.
Error suppression is disabled. Tests require actual compiled graphs
and reject a compiler recompile-budget fallback being counted as parity. The
retained P4/WM data-dependent guards cause graph breaks. Per-case graph counts,
reasons and maximum errors are recorded in `compiled_final_v3/measurements/`; no
single-full-graph, speedup, CUDA-graph or performance claim is made. Final executed
counts and the compiled result table are recorded in the main report.

## Failure preservation and repairs

All earlier evidence remains under `outputs/hoi/contact_supervision/p5_validation/`.

| Evidence | Observed issue | Root cause and repair |
|---|---|---|
| `infrastructure_01.log` | 2 failed, 14 passed | Synthetic quaternion fixture assumed broadcasting unsupported by original `quat_mul`; explicitly expand the fixture. A real FP64 Actor tail computed `gamma**integer_lengths` in FP32; cast lengths to reward dtype before exponentiation. No tolerance change. |
| `integration_02.log` | 22 passed | Repaired reference/boundary/integration checks. |
| `compiled_01.log` | 1 passed, 1 failed, then interrupted | Initial CUDA registry compared unindexed library `cuda` with actual tensor `cuda:0`. Resolve tensor devices. The in-progress FP64 cases were not counted as passed. |
| `cuda_diagnosis_02.log` | 1 failed | Derived goals recomputed after CUDA TorchScript profiling differ by `8.1956386566e-8` in FP32. Raw source tensors retain exact checks; derived goal validation uses the already approved algebra tolerances. Stored Actor goals are unchanged. |
| `cuda_goal_diagnosis.json` | Device metadata `cuda` vs tensor `cuda:0` | New cache/registry/replay adapter compares resolved tensor devices. No original-library mutation. |
| `cuda_repair_01.log` | 2 failed | Test FK used unindexed `cuda`, violating P4's exact-device guard. Construct fixture FK on `q0.device`; preserve P4 guard/source. |
| `cuda_repair_02.log` | 2 passed | Both repaired CUDA eager paths pass; 10 unrelated cases deselected in this targeted run. |
| `cuda_diagnosis_01.log` | 4 deselected | Initial targeted selector used dtype names instead of pytest's dtype0/dtype1 IDs. No executed-test pass claim. Corrected selector retained above. |
| `compiled_final/`, `cpu_diagnosis_01.log` | Strict compiler-coverage assertion failed | Default Dynamo recompile budget 8 exhausted in `unit_quaternion_copy`. Explicit validation-only budget 64 allows all specializations, with the no-fallback assertion retained. Initial failed runs were interrupted for diagnosis; their partial results are not final acceptance. |

Early Ruff logs are preserved; safe import ordering, unused imports/noqa and loop
closure binding were corrected. The final static check covers every new and
modified Python source/test. No assertion was removed to convert a failing
acceptance test into a pass.
Early long-running processes held earlier imported modules while independent work
continued; displayed traceback source-line text can therefore reflect a later file
revision. The exceptions/counts are preserved, and targeted diagnosis/reruns record
the fixes independently. The final matrix was rerun against the completed checkout.
Captured `.log` and JUnit `.xml` traceback whitespace is preserved verbatim using
an attributes file scoped to this evidence directory; source and document
whitespace checks remain enabled. The first staged whitespace check identified
trailing spaces inside captured failure tracebacks, not source/document defects.

## Reproduction and execution limits

Run from the repository root using the existing `.venv`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/validate_p5.py --suite eager --output /tmp/p5-eager-new
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/validate_p5.py --suite compiled --output /tmp/p5-compiled-new
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/validate_p5.py --suite regression --output /tmp/p5-regression-new
```

Output directories must not exist, preserving previous failures. The harness
records exact pytest arguments, every test result, XML/logs, measurements, versions,
hardware and forbidden-action attempts. It guards Adam/SGD steps and MuJoCo stepping
for these tests and restores those test-only guards afterward. Production HOI
modules install no global monkeypatch. Regression selection deliberately excludes
legacy tests that train/update parameters or execute physical HOI rollouts.

The selected 69 P4/P3/P2/shared regressions pass. They cover P4 target/reward tests,
P2 force aggregation/Replay metadata, joint state/support/WM architecture and
migration interfaces, dataset validation, robot FK and original return equations.
Historical physical tests and failed P3 learned/multistep migration gates are not
rerun or changed by that result. Full optimizer-training tests, P6 physical metrics,
P7 learning, full HOI physical rollouts and performance benchmarking are **NOT
TESTED**, as required by this task's execution boundary.
