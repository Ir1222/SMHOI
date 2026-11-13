# P7.0.4 training integration specification

Authority: the current user request, at `d3c3db10bce01b6da4bb5c18d759787680e14afc`
on `main`. Implement integration, bounded deterministic tests, documentation and
a local commit only. No P7-A warm-up, P7-B training, PPO, asset/reference promotion
or push. The complete current Master Plan was read. Its exact SHA and the initial
tracked-file/user-settings hashes are in `outputs/hoi/contact_supervision/p7_0_4/baseline.json`.
The only pre-existing change is untracked `.vscode/`.

The accepted P7 architecture report and readiness audit, P6/closure reports and
specifications, relevant P0 audits, P1/P2 scene/recorder contracts, final P3 cleanup,
P4 reward and P5 integration specifications/reports constrain this implementation.
Historical force-head and PPO proposals are superseded by the current Master.

| Requirement / Master | Verified evidence and current implementation | Decision, files, contracts and validation |
|---|---|---|
| Single original lifecycle, §§1.3.6/12 | P0 architecture audit: fixed construction in `training/loop.py`; P7 accepted report §7 | Extract default-preserving construction hooks; HOI subclass overrides construction/checkpoint only and inherits `Trainer.run`. Separate CLI/config under `hoi`; regression verifies original construction and run identity. |
| Physical collection, §§6/7/12 | P1 scene/state; P2 recorder captures fourth-solve state before reset, rejects commands/events; P0 dependency callback order | Compose native `ManagerBasedRlEnv.step` with an HOI recorder/reset and observation wrapper. No copied step loop. Rebind selected worlds after native reset; one synchronized reference write at reset only. Retain P2 substep measurements/executed action. |
| Features, §4.6 | `HOIObservationAdapter`, `HOIPolicyReplayBuffer`, registry | Physical84/canonical86/WM166→82; ten causal state/action history frames give Actor1080, synchronized goals830, Critic1015. Eight-transition WM windows remain full and boundary-safe. Check noise/scales/reference and reset backfill. |
| Reward/termination, §10 | P5 spec §§5/6/9; original command sampler | Evaluate existing full reward on captured reached state and reference f+1, dt once. Reuse original return sampler methods with reached-frame bins and terminal Critic B+ value before reset. Robot height/tilt thresholds remain .25/.8; no new learned failure head. Timeout bootstraps; clip/failure zero tail. Box failure criteria require explicit configuration and remain a research limitation until reviewed; no invented contact-success rule. |
| Replay, §§7/12 | Original ring plus P2 transition-owned sidecars | Install policy-capable Replay directly; preserve independent raw force/found/orientation diagnostics. Add task reward, episode age/remaining timeout and separate physical causes. Reject invalid solves before insertion. Test ring/history/terminal pre-reset and sampling. |
| Full checkpoint, §12.7 | P0 and P7 audits identify missing Replay/RNG/UTD | New strict HOI training schema, distinct from weights-only migration. Save all models/optimizers, Replay, sampler, physical simulator arrays, action/sensor/history/counters, RNG, schedule/iteration/UTD, episode accumulators and identities. In-place restore preserves graph pointers; recapture under guard, then restore exact state. Compare split/resume with uninterrupted execution. Exclusive snapshots, no historical overwrite. |
| Solver safety | P6 closure hash-pinned `bracket_precision_guard`, P7 readiness instrumentation | Lifetime guard around actual entry construction/capture/resume/steps; attest guarded kernel and graph launches. Per-world overflow/nonfinite/action-map/external-wrench checks each substep; finite losses/gradients/optimizer states before updates. Unknown solver hash fails closed. Preserve warnings and solver/material defaults. |
| P6 disposition | P6/closure remains PARTIAL, candidate not production; original strict loader lacks V3 source | Default reference is original. Explicit diagnostic profiles retain exact P6 missing-document exception, original or separately selected single-body candidate; never modify production loader policy, reference labels, calibration or training_ready. Strict production mode remains blocked by its historical prerequisites. Document each blocker independently. |
| Acceptance | User's eight categories and Master §§15/17 | Deterministic software checks first; at most four physical worlds and 64 steps per integration invocation; physical tests/backward only, zero physical-data optimizer updates. Isolated synthetic one-update tests may validate inherited optimizers and populated optimizer resume; no fitting experiment. Report every failure/skip and numerical evidence. |

Implementation-level adjustments: construction hooks, recorder extension points,
explicit profile loader, safe checkpoint serialization and lifecycle wrappers only.
No research architecture/reward/solver-parameter change is proposed. Original
Robot classes/defaults/checkpoints remain intact. Default HOI integration mode
disables updates and enforces a bounded budget; future alternating execution
requires an explicitly supplied readiness/authorization record matching identities.
No such authorization record is created by this task.

Implementation adjustment A1: installed mjlab managers deepcopy their configuration.
Live model/registry/guard objects therefore live on the runtime environment and are
resolved by the recorder, rather than being deepcopied in term parameters. Plain
function callbacks retain the process guard. No manager/step algorithm changes.

Implementation adjustment A2: optimizer pre-step hooks cannot detect an infinite
gradient norm after clipping has turned finite large gradients into zeros. Add
optional, default-None numerical callbacks to the original WM/Policy trainers at
scalar loss and gradient-norm boundaries, including Critic MSE. This is needed for
fail-fast behavior before optimizer updates and leaves Robot defaults unchanged.

New files planned: `hoi/training_config.py`, `training.py`, `training_env.py`,
`training_checkpoint.py`, `training_safety.py`, `training_scene.py`, `train.py`,
focused integration/checkpoint tests and bounded evidence runner. Shared changes:
`training/loop.py` construction hooks only, `pyproject.toml` entry registration;
HOI-local scene/recorder/Replay optional interfaces only as needed. Final file map
will enumerate actual paths in the implementation report.

Predeclared checks: exact schema/identity/mask/action/frame matching; physical
state/history roundtrip FP32 atol1e-6; resume tensors ideally bit-identical on
the pinned same-device stack (otherwise report measured differences and do not
claim exact resume); rewards match direct P5 recomputation atol/rtol5e-6; every
environment must have zero overflow and finite state/action/reward/loss/gradient.
Existing accepted architecture parity is reused unless its numerical path changes.
Synthetic Jacobians do not establish learned WM readiness. Checkpoint provenance
includes actual Box, support, reference, calibration, solver source/patch, code,
dependency and configuration identities, with candidate readiness false.

Remaining research gates: original COM/inertia, missing source provenance,
reference/native collision mismatch, reset penetration, sustained manipulation,
reward/return calibration, Box-only failure prediction and learned WM fidelity.
None is silently converted into PASS by software integration. Stop after the
requested local commit and report; no subsequent training stage is authorized.

Implementation adjustment A3: mjlab allocates its returned reward/reset buffers
on the first step. HOI construction allocates those same buffers before loading
a checkpoint; native termination/metric dictionaries and all 185 Warp workspace
arrays are serialized. Runtime values are restored in place after guarded graph
recapture. The compiled native model, source code, versions and schema are pinned.

Implementation adjustment A4: mjlab `TorchArray` is a proxy, not a `torch.Tensor`.
The physical monitor explicitly takes `[:]` tensor views before finite checks.
Independent per-world NaN injection found and verified this correction. No
warning mask or solver parameter is changed; required `OverflowType.ALL` is 4095
on this installed stack (the old readiness prose's `warn_overflow=1` is not used).

Resume acceptance clarification: immediate restoration retains the predeclared
exact/1e-6 check, and populated-Adam continuation is required to be exact. The
first continuation attempt failed an exact physical trajectory comparison.
Subsequent checks record every absolute difference, require exact discrete
metadata/masks/RNG and a 1e-4 ceiling on continued physical state, actions,
observations and rewards. This is an additional bounded drift check, **not** a
replacement PASS for exact physical continuation; that acceptance remains
PARTIAL. The original failing test/log is retained. No deterministic solver mode
or parameter was substituted to improve this result.

Final validation includes one inherited update per learner on an artificial
batch; a tiny populated-Adam continuation/failure test; and a full-width model
serialization probe with constant artificial gradients (two effective steps plus
one replayed step per model). These are isolated tests, consume no collected
physical training target, and are followed by no physical execution. All actual
`smHOI-track` runs and all physical-data backward checks have zero optimizer steps.
Box height termination is explicitly diagnostic; alternating mode rejects that
option until its counterfactual imagination/failure semantics are resolved.
