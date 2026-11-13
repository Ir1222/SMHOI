# P7.0.4 training integration report

**P7.0.4 implementation delivered; overall acceptance PARTIAL. Training readiness:
NOT READY.** The approved HOI architecture executes through the original
`Trainer.run()` and its existing WM / Actor BPTT / Critic TD(lambda) interfaces.
The actual `smHOI-track` entry constructs, collects, checkpoints and resumes.
Exact checkpoint restoration and populated-optimizer continuation pass. Continued
physical trajectories are close but not bit-identical. Strict production loading
remains blocked by missing historical provenance, and P6 research gates remain.

This task performed no P7-A warm-up or P7-B alternating training. All physical
collection runs and physical-data gradient probes used **zero optimizer steps**.
Only isolated synthetic unit tests exercised optimizers. No PPO, force/contact
reward, prediction head, reference edit, asset promotion or push was introduced.

Baseline: `d3c3db10bce01b6da4bb5c18d759787680e14afc`. Specification:
[P7_0_4_IMPLEMENTATION_SPEC.md](P7_0_4_IMPLEMENTATION_SPEC.md).
Evidence root, abbreviated **E** below:
`outputs/hoi/contact_supervision/p7_0_4/`.
The full current Master Plan, P7 architecture/spec/readiness, P6/closure, and
relevant accepted P0–P5 reports/specifications were read. The current Master
supersedes the historical readiness document's PPO proposal.

## Implementation and lifecycle

`HOITrainer` specializes construction and checkpoint hooks; it does not override
`run` or `act`. The original Robot constructor moved into `construct_components`
without changing its components/defaults. AST comparisons against the accepted
commit confirm original `run`, `act`, Robot `save` and Robot `load` are identical.
Optional numerical callbacks in the shared learners default to `None` for Robot.

The full approved schema is `hoi-actor-b-critic-bplus-v1`: physical84,
canonical86, Joint WM166→82, Actor observation1080/goals830, Critic1015,
action29. The real constructor uses full approved network widths, P1 physics,
P2 interval recorder, P4/P5 reward, policy Replay, reference registry, observation
adapters and original learners. The original sampler's selection, reward-bin
accumulation and episode-finalization methods are reused outside the native
command manager, so its Robot-only reset writes never run.

Native mjlab owns physical stepping and autoreset. A recorder subclass computes
the existing 13 additive reward terms and terminal Critic value from captured
reached physical state at reference frame t+1, before resetting any done world.
Reward already includes dt=.020 exactly once; the empty native reward manager
does not rescale it. Previous-action observations and action-rate reward use clean
actions, as in SuperMimic; WM targets use the actually executed noisy action.
Every substep checks its exact native actuator mapping.

Selected-world reset performs one synchronized reference initialization and binds
episode/clip/frame metadata. Ten-frame policy history is backfilled at reset;
the policy Replay supplies valid eight-transition WM windows and boundary-safe
Actor starts. The P2 pre-reset reached state, four force samples, found counts and
solve orientations remain transition-owned. Bad solves abort before insertion.
No state write, reference teleport, external wrench or command reset runs inside
an episode. Partial resets leave other worlds' state/history/episode unchanged.

Physical Robot failure uses the original root height/tilt thresholds .25/.8 on
fresh reached state. Timeout and natural clip end retain distinct flags, including
simultaneous endings. Failure/clip end remove bootstrap and continuation; timeout
retains them. Sampler timeout bootstrap uses pre-reset Critic B+ value. Logged
failure from a recorded action is not imposed on counterfactual imagined actions.
An optional explicit Box-height boundary is tested only as a diagnostic; there
is no approved Box-only imagined failure model, so alternating mode rejects it.

Integration mode enforces at most four worlds, 64 steps, and zero updates. The
default profile is strict production and fails closed here. Executed tests select
`p6-original` explicitly; its only provenance exception is the exact missing V3
path/hash previously documented by P6. `p6-single-body` is a separately named,
hash-checked, in-memory candidate profile. Neither changes the production manifest
or reference. No P7-B approval record was created. Future alternating execution
requires a matching explicit authorization/readiness record and approved WM or
full-resume checkpoint; that branch was not run in this task.

## Checkpoint and safety evidence

New schema: `hoi-training-lifecycle-v1`, distinct from Robot and weights-only
formats. It contains:

- Joint WM, Actor B and Critic B+ state dictionaries and all three Adam states.
- Full Replay tensors/sidecars/pointers/episode counters/metadata; sampler bin
  return/memory, frame, episode logs, gamma powers and metrics.
- Native simulator state including all **185** Warp workspace arrays, warmstarts,
  action buffers, sensors, termination and metric buffers, selected reference,
  clean/action/observation histories and pending transition data.
- Python, NumPy, Torch CPU and every visible CUDA device RNG; iteration, original
  warm-up origin, UTD accumulator, update count, episode statistics and full
  scheduler configuration. Resume does not restart warm-up.
- Reference/manifest, Box, supports, contact-target/calibration, solver source and
  precision fragment, code, dependency, compiled native model and feature-schema
  identities. Physical profile and false readiness status remain explicit.

Restore validates identities, keys/shapes/dtypes, fixed buffers, optimizer
structure/moments, counters, RNG and finite authoritative state before mutation.
It recaptures graphs under the guard, restores arrays in place, and restores RNG
last. The inherited `run()` receives restored observations instead of resetting
the loaded episode. Six corrupt-checkpoint cases reject without changing
models/runtime/RNG. Existing run directories and checkpoint files are refused;
exclusive files are never used to overwrite Robot or historical checkpoints.

At iteration8, all serialized state restores **bit-exactly** in a new trainer.
Models, optimizers, discrete masks/metadata, schedule counters and RNG remain
exact through the eight resumed steps. Final physical state/action/observation/
reward maximum absolute drift is **9.149312973e-6**, with raw/contact-force maximum
drift **2.74658203125e-4 N**. Every path's difference is saved in
`measurements/physical_resume.json`. The first exact-trajectory test failed and
is preserved in `integration_02.log`; it was not relabeled PASS. GPU reduction
ordering is a plausible explanation, not a proven root-cause claim. No change to
baseline solver settings or warning behavior was made to seek determinism.

The full-width populated-Adam checkpoint test uses constant artificial gradients
1e-5, saves after one step, and compares the next step against load/resume:
all models, optimizer states, runtime, counters and RNG are **bit-exact**. It uses
no physical target and performs no subsequent physical execution. A separate
small-Adam test verifies nonfinite gradients cannot step any of the three models.

`SolverGuard` encloses actual entry construction, capture, stepping, checkpointing
and resume. Launch instrumentation observes the hash-pinned precision kernel and
rejects the original iterative kernel, lost guard, missing graph or reduced
overflow warning mask. On this stack `OverflowType.ALL=4095`; earlier prose saying
`warn_overflow=1` is not the installed bitmask contract. All worlds independently
undergo overflow, finite state/acceleration/control/wrench, executed-action and
force checks each physics substep. NaN/overflow probes each reject in both worlds;
guard-loss injection also rejects. Learners check loss and gradients before
clipping/step, and optimizer parameters/state before and after step.

The final 16-step physical test checks **128 world-substeps**, 32 transitions,
one natural clip end and one timeout, with zero overflow. An explicit one-step
overlap fixture verifies two physical failures + two timeouts + one clip end;
all terminal states precede reset and bootstrap values are zero. No artificial
physics-state write is used for this overlap fixture. Fault-injection tests write
temporary NaNs only for direct monitor calls and restore them without stepping.

## Tests, numerical evidence and known failure

| Test / evidence | Result |
|---|---|
| Initial affected component suite, `components_01.log` | 51 PASS |
| Broad affected regression, `tests_final_01.log` / XML | **95 PASS, 1 FAIL**, 25.92 s; no skips |
| Final integration after direct reward/history and populated-optimizer checks, `integration_final_02.log` / XML | **12 PASS**, 12.71 s; no skips |
| Registered `smHOI-track`, original diagnostic profile, `entry_final_01.log` | PASS; 2 worlds ×16 steps; zero updates; snapshots at8/16 |
| Registered entry resume, `entry_resume_01.log` | PASS; load8 →16; guarded recapture; zero updates |
| Explicit single-body diagnostic constructor, `entry_candidate_01.log` | PASS; 2 worlds ×2 steps, original reference, zero updates; no promotion |
| Original Robot Trainer + checkpoint and environment regression (within broad suite) | PASS; original Actor/Critic/WM classes, 930/680 dimensions, checkpoint schema and exact loaded policy |
| New-code lint and both CLI help commands | PASS; `lint_final.log`, `entry_help_final.txt`, `robot_entry_help.txt` |
| Historical/user-file preservation | PASS; 3,122 initial paths checked; only six intended existing files changed; no missing/unintended paths |

The sole broad-suite failure is
`test_force_recorder.py::test_physical_interval_recorder_and_terminal_copy`:
its legacy strict production subprocess fails on missing
`SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md` before collection. The source/loader is
unchanged, and P6 already documents this prerequisite failure. The equivalent
actual collection/reset/replay behavior passes through the explicitly identified
diagnostic profile. No skip, fake source document or global loader bypass was added.

Final measured values (`measurements/physical_resume.json`): executed action-noise
std **.03068234399** for configured .03; WM eight-step loss **180.0487213**, gradient
norm **75.21167755**; Actor H=2 loss **−.1002309620**, gradient norm **.001286476152**;
Critic loss **.03423961625**. These are untrained-network interface checks, not
useful prediction/control evidence. All are finite. Physical-data optimizer counts
are WM0/Actor0/Critic0. Direct t+1 reward and additive-term checks pass atol/rtol5e-6;
current reached state and latest history agree atol1e-6.

One inherited update per learner on an artificial CPU batch returns WM loss
**5.399609089**, WM gradient norm **20.96197891**, Actor gradient norm
**.004437382333**, Critic loss **.03682804108**. This verifies the actual inherited
`update()` interfaces; it is not a training experiment or WM-readiness result.

Independent retained P2 diagnostics use unchanged thresholds
(.0743401046875112, .07375101817770177) N and strict occupancy>.25. For the 32-row
test, OR confusion is [[27,2],[3,0]], AND [[28,1],[3,0]], and peak net force is
about131.9114 N. Contact F1=0 for both definitions on this tiny random-policy
sample. None of those forces/decisions enters a loss or reward, and no contact,
lift or carry success is claimed.

Initial failures are retained: `probe_01` exposed manager deepcopy of a live guard;
`probe_02` exposed the incorrect warning-mask expectation; `integration_01`
exposed first-step native buffer allocation; `integration_02` exposed nonexact
physical continuation; `integration_04`/`safety_05` exposed the native tensor-proxy
finite-check omission. `safety_06` and the final suite verify the corrected guard.
The existing TorchScript deprecation warning remains visible. Console `nan`
for absent WM/imagination metrics and no-completed-episode return in zero-update
smokes is the unchanged original logger's missing-data display, not a nonfinite
physical reward/loss accepted by the guard.
Source/document whitespace checks pass. Raw pytest failure logs/XML retain their
original traceback whitespace; whole-evidence `git diff --check` therefore reports
those lines. They were not edited to make the historical failure output cleaner.

## P6 blocker disposition

| Blocker | Classification | Effect on this profile and disposition |
|---|---|---|
| Missing historical V3 source; production M6.5 false/NO_GO | **Hard software/provenance blocker** | Strict production construction and one legacy test FAIL. Exact diagnostic exception only; no forged source or status change. |
| LS1024 floor-impact failures / historical ITERATIONS512 | **Hard physical blocker without guard; bounded experimental limitation with guard** | Actual entry requires the approved source-pinned guard. Bounded current solves pass; historical failures and lack of universal reliability remain. |
| Original Box COM190 mm below collision envelope | **Unresolved research-level decision** | `p6-original` preserves it and tests software plumbing only. Separate single-body candidate is available explicitly but not production-selected. |
| Original reference native-collider overlap/timing mismatch | **Reference compatibility issue** | Original HDMI reference/labels remain unchanged. P6 original agreement FAIL persists; integration does not resolve it. |
| Candidate passes geometric ON frames but dynamic replay/carry fails | **Reference compatibility issue and unresolved research decision** | Candidate NPZ remains `training_ready=false`; it is not used in these entry runs and is not promoted. |
| Known reset penetration / finite does not imply penetration-free | **Bounded experimental limitation** | No pose shift or post-reset assistance. Long physical safety and permissible reset overlap need separate disposition. |
| Independent natural lift/carry failed4/4 in closure | **Unresolved research-level decision** | No manipulation-success or training-ready claim from current random-policy smokes. |
| Reward coefficients/return scale and Box-only failure semantics | **Unresolved research-level decision** | Explicit test coefficients/g_max only; no new production defaults. Box-height diagnostic cannot enter alternating mode. |
| Learned WM held-out fidelity/action-to-Box sensitivity absent | **Hard prerequisite for P7-B; research decision/experiment pending** | Finite backward paths and synthetic updates do not satisfy it. No WM fitting undertaken. |
| Physical continuation not bit-identical; only single-process restore tested | **Bounded experimental limitation** | Exact restore/Adam/schedule pass; measured continued-state drift retained; distributed/device-changing resume unsupported. |

References: [P6 report](../p6/REPORT.md), [closure report](../p6/closure/REPORT.md),
[P7 readiness](P7_TRAINING_READINESS.md). No original reference, contact label,
production asset manifest, contact calibration, installed solver source or baseline
solver parameter was changed. Candidate reference SHA remains
`f38071e5347403f9eff993dd37884bf374e0c003400156dc4cb87d3d2b856cb0`, readiness false.

## Acceptance decisions and final readiness checklist

| Requested acceptance requirement | Decision |
|---|---|
| Separate entry and full HOI construction | **PASS** for explicit diagnostic profiles; **FAIL** for currently blocked strict production provenance |
| Reuse original `Trainer.run` / alternating algorithm; no PPO/second loop | **PASS**; AST and method-identity checks |
| Actual stepping, executed actions, no episode teleport/wrench | **PASS** in bounded native tests |
| Per-world reset, observations/history/reference/episode alignment | **PASS**; partial-reset isolation and reached-state checks |
| P4/P5 Robot/Box/proximity reward, t+1 and dt once | **PASS**; direct recomputation and existing reward regressions |
| Failure/timeout/clip masks and terminal Critic value | **PASS** for implemented physical/authoritative boundaries; **PARTIAL** for unresolved Box-only imagined failure modeling |
| Policy Replay insertion, eight-step windows, reset backfill, terminal states and invalid-target exclusion | **PASS** in diagnostic profile; strict legacy P2 subprocess still **FAIL** before construction |
| Joint WM update interface | **PASS**; physical backward + inherited synthetic single update |
| Actor B imagination/BPTT interface | **PASS**; full physical-history backward, frozen WM and inherited synthetic update |
| Critic B+ TD(lambda) interface | **PASS**; boundary suite and inherited synthetic update |
| All checkpoint contents, exact load, populated optimizers, RNG/schedule/sampler | **PASS** on pinned single-process/same-device configuration |
| Save/resume equivalence against uninterrupted physical execution | **PARTIAL**; exact restoration/discrete/RNG/optimizer continuation, small measured physical drift; no bit-exact trajectory claim |
| Actual-entry solver guard after capture and resume; all-world nonfinite/overflow abort | **PASS** in bounded and injected-fault tests |
| Original `sm-train` regression/defaults/checkpoint compatibility | **PASS**; original lifecycle and checkpoint schema retained |
| P6 disposition without hidden changes/promotion | **PASS** for audit/preservation; physical/reference training gates remain **PARTIAL/FAIL** |
| Bounded tests only; no P7-A/P7-B, no optimizer-intensive experiment | **PASS**; zero physical-data optimizer updates, isolated synthetic unit checks only |

- [x] Reviewable P7.0.4 code/spec/report and file map delivered.
- [x] Bounded actual-entry collection, reset, Replay, backward interfaces and resume checked.
- [x] Physical/synthetic evidence distinguished; failures and warnings preserved.
- [x] Robot behavior/checkpoints and existing user `.vscode/` preserved.
- [ ] Strict production provenance and mass-model disposition resolved.
- [ ] Reference/reset/contact/carry limitations explicitly approved for a training profile.
- [ ] Reward/sampler calibration and Box-only failure policy approved.
- [ ] Learned Joint-WM readiness demonstrated on held-out physical episodes.
- [ ] Bit-exact physical or broader device/distributed resume established.
- [ ] P7-A or P7-B explicitly authorized. **Neither stage started.**

## Complete file-change map

| Path | Change |
|---|---|
| `pyproject.toml` | Add `smHOI-track`; preserve `sm-train`. |
| `src/supermimic/training/loop.py` | Two default-preserving construction hooks; original run/act/save/load unchanged. |
| `src/supermimic/world_model/wm_trainer.py` | Optional default-None finite-loss/gradient callback. |
| `src/supermimic/imagination/policy_trainer.py` | Same optional checks for original Actor/Critic optimizers. |
| `src/supermimic/hoi/scene.py` | Optional validated manifest/configuration injection; default strict behavior retained. |
| `src/supermimic/hoi/recorder.py` | Optional recorder subclass selection; P2 recorder unchanged. |
| `src/supermimic/hoi/training_config.py` | Separate explicit bounded config, schema constraints and future authorization gate. |
| `src/supermimic/hoi/training_scene.py` | Hash-pinned diagnostic profiles and physical/code/compiled-model identities. |
| `src/supermimic/hoi/training_safety.py` | Lifetime precision guard, native finite/overflow/action checks and learner hooks. |
| `src/supermimic/hoi/training_env.py` | Native physical reset/reward/history/Replay/sampler bridge. |
| `src/supermimic/hoi/training_checkpoint.py` | Exclusive whole-system save, validation and in-place guarded restore. |
| `src/supermimic/hoi/training.py` | Full HOI constructor and checkpoint hooks on original Trainer. |
| `src/supermimic/hoi/training_diagnostics.py` | Independent retained P2 calibrated contact export. |
| `src/supermimic/hoi/train.py` | CLI, lifecycle guard, failure evidence and diagnostics. |
| `tests/hoi/test_training_integration.py` | Twelve bounded construction/physical/history/mask/safety/checkpoint/synthetic-interface tests. |
| `tests/hoi/test_training_robot_regression.py` | Original full Trainer and historical checkpoint-schema regression. |
| `docs/hoi/force_contact/p7/P7_0_4_IMPLEMENTATION_SPEC.md` | Preimplementation contracts and explicit evidence-driven adjustments. |
| `docs/hoi/force_contact/p7/P7_0_4_IMPLEMENTATION_REPORT.md` | This report, disposition, commands and acceptance. |
| `outputs/hoi/contact_supervision/p7_0_4/` | Baseline/preservation, logs/XML (including failures), measurements, exact CLI metadata/checkpoint hashes, command ledger and inventory. Every file enumerated in `files.json`. |

The local editable project registration was refreshed offline with no dependency
installation/change. New diagnostic `.pt` snapshots are under `/tmp/smhoi_p704_final`
and pytest temporary directories, not historical paths or Git. Persistent manifests
record their sizes/hashes/schema/array counts; model binaries are not committed.
Wall-clock timing/console aggregation is intentionally not resumed; learning,
physical, sampler and schedule state is. Exported contact report covers retained
Replay rows, not a long-run success history.

Exact principal commands and all test/run variants are in
E/`commands.json`; commands use new output/run names when repeated. Shell execution
required the approved escalated-shell fallback because sandbox `bwrap` could not
initialize loopback. No automatic-review rejection occurred. Work stops at the
requested local P7.0.4 commit; no push and no subsequent training stage.
