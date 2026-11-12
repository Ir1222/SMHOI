# P5 — complete software implementation and validation

**Acceptance decision: PASS for complete P5 software implementation and validation.**
P5-A/B/C are complete, with **114 final tests passed, zero failures and zero skips**.
This decision is restricted to software integration; physical task success and
training readiness are not established.

## Baseline, authorization and preservation

Implementation started on `main` at approved specification commit
`3b86b1526735fc3d46b408200ed700650aad6dfc`. Initial status contained only untracked
`.vscode/`. No applicable `AGENTS.md` was found in the repository or its ancestors.
The complete Master Plan/specification, supporting P4/P3 reports and implementation,
and applicable P0–P2 contracts/evidence were inspected before source edits.

The original interfaces were checked against `/home/han/pyx/SuperMimic-Reproduced`
at `2ea2664bc40cc5d73dfff811a561073ae31463b9`; eight named reward/imagination/network/
training sources were initially byte-identical. Installed dependency hashes matched
P0. `baseline.json` inventories 642 files, including tracked sources/docs,
NPZs/assets/checkpoints, historical outputs, dependency sources and user settings.
The final `preservation.json` lists authorized changes and any unexpected change.
The approved P5 specification is preserved as an exact prefix with Appendix B added;
historical P3/P4 reports are unchanged. The Master Plan changes only its P5 scope,
deliverables and implementation evidence. There was no reset, clean or push.

The user authorized all P5 work and the local commit, while forbidding optimizer
updates, WM fitting, P6/P7 and physical HOI rollouts. Those boundaries were honored.

## Implemented files

| File | Responsibility |
|---|---|
| `src/supermimic/hoi/tracking_reward.py` | Required coefficients; safe HOI quaternion error; four Box rewards; complete scalar and bound |
| `src/supermimic/hoi/reward_reference.py` | Immutable FP64-derived cache; source-verified clip registry and local/global gathering |
| `src/supermimic/hoi/imagination.py` | HOI start/Replay/context/step/trainer components; projections, valid masks, diagnostics and initialization metadata |
| `src/supermimic/data/replay_buffer.py` | Shared start-selection/history/envelope extension hooks |
| `src/supermimic/envs/mdp/rewards.py`, `world_model/heads.py` | Optional HOI orientation-error injection; original default retained |
| `src/supermimic/imagination/rollout.py` | One original recurrence with optional HOI references, projection and masks |
| `src/supermimic/imagination/objective.py` | Same Actor return and TD(lambda) recursion with optional finite-task boundaries |
| `src/supermimic/imagination/policy_trainer.py` | Shared loss/target construction, frozen-parameter restoration, valid Critic minibatches and diagnostics |
| `tests/hoi/p5_fixtures.py` and seven P5 test modules | Deterministic fixtures and algebra/reference/gradient/boundary/checkpoint/compiled acceptance |
| `scripts/hoi/validate_p5.py` | Reproducible bounded suites, forbidden-update/physics guards, per-test JSON/XML/log evidence |
| `P5_A_REPORT.md`, `P5_B_REPORT.md`, `P5_C_REPORT.md` | Equations/configuration, exact integration, numerical/gradient/boundary evidence and repairs |

Original Policy/Critic networks, `Trainer.run`, robot `ImagStep`, P3 Joint WM,
state/integration/support contracts, and P4 proximity code remain unchanged.
No contact/force head, observation dimension, active contact BCE or force loss was
introduced. Scheme A remains selected.

## Reward, reference and training path

[P5-A](P5_A_REPORT.md) gives the complete equations. In compact form, for Box
position, shortest orientation log, world linear velocity and world angular velocity:

\[
r_j=dt\,\beta_j e^{-e_j/\sigma_j^2},\quad
(e_p,e_q,e_v,e_\omega)=
(\|p-\bar p\|^2,\|D(\bar q,q)\|^2,
 \|v^w-\bar v^w\|^2,\|R(q)\omega^b-\bar\omega^w\|^2).
\]

`total = six positive robot terms + four Box terms + dt*beta_c*P4_proximity
+ original action-rate and unsquared joint-limit penalties`, each penalty once.
The original robot equations/reanchoring remain; both HOI robot orientation terms
use the approved safe helper. The global robot-only helper is unchanged.

There are **no selected final training coefficients**. Construction requires
explicit finite valid parameters. The only numerical fixtures are labeled test-only
in tests and evidence. The original 840-frame, 50-Hz NPZ is immutable, SHA256
`bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`.
Its derivative cache uses centered interior and one-sided endpoint world pose
increments, with body angular audit `R(q_f)^T*omega_world`. All frames, signs,
provenance and explicit two-clip mapping were checked.

[P5-B](P5_B_REPORT.md) describes the exact path. `HOIPolicyTrainer` supplies hooks
to the original `PolicyTrainer.actor_loss/update` and shared `imagine`. Inside
`HOIImagStep._step`, the existing Joint WM produces physical84; the complete reward
is computed immediately afterward and returned in the original scalar slot.
Original BPTT propagates that scalar through frozen WM input Jacobians to Actor.
The shared detached Critic path fits the same scalar returns using the original
Critic and TD(lambda) algorithm. Neither optimizer update was executed.

At transition `k`, Actor goal progression remains `f+k`; every reached reward,
termination feature and stored next Critic root feature uses **`f+k+1`**. The 260/550
contact boundaries and 838→839 final transition are tested. Frame839 cannot start
another transition. Natural clip end includes the final reward once with zero
tail; timeout before the end keeps bootstrap; hard failure includes its reached
reward with pre-step survival and zero tail. Padding has no reward or Critic loss.
Separate reason flags are retained even when boundaries coincide.

Dimensions remain physical84/canonical86/residual82/input166/support18/action29/xi33.
Robot FK/observations/Critic receive robot71. Actor remains 930/680/29 and Critic788;
the existing optional anchor setting remains 689/797. Strict loading of the actual
original `ckpt_0012000.pt` passed, including all parameter and normalization-buffer
values. No optimizer/sampler resume is implied by shape compatibility.

## Executed validation

Environment: torch `2.14.1+cu130`, numpy `2.5.4`, mjlab `1.6.0`, MuJoCo and
mujoco-warp `3.16.0`, warp `1.18.0`; CPU and NVIDIA GeForce RTX 4090 D.
The latest evidence is under `outputs/hoi/contact_supervision/p5_validation/`.

| Suite / check | Result | Evidence |
|---|---|---|
| P5 eager acceptance, including CPU FP32/FP64 and CUDA eager regression | **41 passed**, 0 failed/skipped | `eager_final_v5/summary.json`, `tests.xml`, `pytest.log`, `measurements/` |
| P4/P3/P2/shared selected regressions | **69 passed**, 0 failed/skipped | `regression_final/summary.json`, `tests.xml`, `pytest.log` |
| CPU/CUDA × FP32/FP64 Inductor, H=1/H=3 | **4 passed**, 0 failed/skipped | `compiled_final_v3/summary.json`, XML/log and per-case measurements |
| Ruff on all 18 new/modified Python files | PASS | `lint_final.log` |
| Python AST syntax and source hashes | PASS | `source_manifest.json` |
| Protected files / append-only specification | PASS at final verification | `preservation.json` |
| Full diff/whitespace/document links | Recorded at final verification | `review.json` |

Actual final commands (each output directory was new):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/validate_p5.py --suite eager --output outputs/hoi/contact_supervision/p5_validation/eager_final_v5
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/validate_p5.py --suite regression --output outputs/hoi/contact_supervision/p5_validation/regression_final
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/validate_p5.py --suite compiled --output outputs/hoi/contact_supervision/p5_validation/compiled_final_v3
```

Each summary includes the full expanded pytest command and every test result.
The selected regression set includes geometric reward/target audit, force aggregation
and Replay, joint state/WM/support/architecture, dataset validation and checkpoint
migration, FK, and original GAE. It excludes optimizer-update and physical-rollout
tests to enforce the authorized boundary. No selected test was skipped.

The deterministic nonzero-coupling model gives finite, nonzero action and Actor
gradients for all six positive robot terms, four Box terms and proximity separately.
The H=3 full Actor gradient norm is `1.8901432098e-4`. OFF proximity gradients are
zero; separate EEF/Box-target branches, finite differences and detached controls
verify the graph. Frozen WM/Critic flags and weights are preserved and restored;
detached Critic loss produces no Actor/WM gradients. The **actual initialized model**
has zero action→Box gradient, reported separately. Neither result establishes
learned physical dynamics fidelity.

Original robot-only matched-RNG rollouts/actions/noise/rewards/gradients and Replay
sampling match exactly. All-valid returns/TD targets match the original formulas.
The legacy exact-match quaternion NaN remains explicitly documented, not repaired
globally or counted as a newly passing behavior.

Compiled validation uses actual Inductor with explicit validation-only Dynamo
budget64; the default budget8 was exhausted and can fall back to eager. Tests retain
the no-budget-fallback assertion and all numerical tolerances. Graph breaks and
eager regions caused by preserved guards/FK enum access remain; there is no full
graph or speedup claim. Per-case maximum forward/gradient errors and graph details
are recorded in JSON. The final measured maximum absolute differences across
H=1/H=3 are:

| Device/dtype | State | 13 components | State gradient | Action gradient | Actor gradient |
|---|---:|---:|---:|---:|---:|
| CPU FP32 | 5.96046e-08 | 6.51926e-09 | 2.27708e-07 | 8.73115e-11 | 6.63931e-11 |
| CPU FP64 | 0 | 1.73472e-18 | 7.28584e-17 | 8.13152e-20 | 2.71051e-20 |
| CUDA FP32 | 8.19564e-08 | 1.35042e-08 | 5.47152e-07 | 5.96629e-10 | 8.00355e-11 |
| CUDA FP64 | 2.39392e-16 | 2.42861e-17 | 9.33525e-16 | 2.43945e-19 | 8.13152e-20 |

All cases pass the specified dtype tolerances; bool gates, indices and masks
match exactly. Each case produced 100 compiled graphs across its specializations;
step explanation reports 24 graphs and 23 breaks. Detailed geometry/observation/
survival errors and break reasons remain in the per-case JSON. The earlier complete
`compiled_final_v2` run also passed all four cases; the final cached run confirms
the completed checkout after the constructor dimension guard was tightened.

Earlier failures and repairs are detailed in [P5-C](P5_C_REPORT.md): the FP64 discount
dtype bug, an unsupported quaternion fixture broadcast, CUDA device alias/FK setup,
derived goal rounding, and compiler specialization-budget exhaustion. Failed logs,
interrupted partial runs and targeted reruns are retained. They are not relabeled
PASS or hidden in the final totals. Final acceptance uses only the completed suites.

## Remaining blockers, risks and untested work

1. **P3 prediction validation remains open.** The actual initialized Box Jacobian
   is zero. Learned physical joint-dynamics fidelity/useful gradients require P7-A.
   Historical FP32 multistep checkpoint migration failures at `2e-6` remain unchanged;
   P5's distinct compiled tolerances do not supersede them.
2. **M6.5 remains `training_ready=false`, `M6_5_NO_GO_M7`.** P4's internal EEF marker
   geometry and support/overlap failures remain. No asset or target was changed and
   no physical gate was marked passed. Historical P3/P4 PARTIAL results remain.
3. Strict physical loading still lacks historical V3 provenance; refined-reference
   sidecars are absent. No provenance artifact was fabricated. P5 uses the accepted
   original NPZ; no full physical HOI rollout or P6 was executed.
4. Final reward coefficients, Critic return calibration, HOI return normalization
   and sampler scaling remain unselected/unvalidated. Initialization metadata says
   `training_ready=false`; robot-only optimizer moments/sampler statistics are not
   valid HOI resume state.
5. Actor/Critic direct Box observability and Box-only failure prediction remain
   absent by the approved dimension contract. Authoritative optional timeout/cause
   metadata must be supplied by future physical wiring; mask tests do not predict
   object failure.
6. Quaternion pi discontinuity, P4 zero-distance cusp, finite-difference reference
   approximation and compiled graph breaks remain explicit limitations. CUDA graph
   capture, distributed training, speed/performance and additional compiler modes
   were **NOT TESTED**.
7. Actor/Critic optimizer steps, WM fitting, P7 training and physical task success
   were **NOT TESTED and not executed**, as required. Backward validation alone
   does not establish training readiness or pick-carry-place success.

## Final acceptance and handoff

**P5-A PASS; P5-B PASS; P5-C PASS; overall P5 software PASS.** Required reward,
reference, imagination, BPTT, numerical, boundary, checkpoint and regression gates
passed. Final verification found 634 unchanged baseline files and exactly eight
authorized changes out of 642, with no missing/unexpected changes.
P5 stops with the final evidence, synchronized plan and local commit. No push,
P6 or P7 is authorized by this report. The final commit hash and worktree status
are reported with the delivered commit rather than embedded as a self-reference.
