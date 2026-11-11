# P2 implementation specification

Scope: §7 data acquisition, interval statistics and Replay only. Authority remains
[the Master Plan](../SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md); this document records
concrete implementation decisions, not replacement requirements. No P3, Force Head,
rewards, imagination, training, checkpoint migration or entry-point integration.

Baseline: `main`, `a562d936541ac7a73c6fa70fcf727449e426f28a` (P1 implementation),
preceded by specification commit `61c2760`. Master working-copy SHA256
`5b0c26e3c3d8c721e5e1f76ffe194f6141994a9eba4894ce7f43d7266e9be82f`.
P1 reports retain historical pre-reconciliation commit IDs; they are not rewritten.
The present user request accepts P1 as the prerequisite and authorizes P2 and commits
after each group; it does not approve P3 or pushing. This explicit commit instruction
supersedes the default §15.2 wait-before-commit workflow for this turn.

Read all 1,834 master lines; P0 coverage/report/closure/file map, architecture,
dependency, dataset/asset and reference implementation audits; P1 specification,
report, validation and reuse audit. P0's documented accepted IDs remain historical;
current Git history has reconciled IDs, so preservation is checked by file hashes
rather than claiming absent IDs are current ancestors. `baseline.json` records
actual HEAD/status, versions, all 126 P1 protected hashes and ten matching dependency
hashes, plus a snapshot of existing files. P1 source and evidence hashes are checked
against its validation summary before implementation. User-owned master typo fix,
`.vscode/`, untracked V3 and `reference_refined/` are excluded from commits.

## Decisions grounded in current code

| Master requirement / acceptance obligation | P0 finding and evidence | Actual implementation and decision | Changes / reusable interfaces | Contracts and validation |
|---|---|---|---|---|
| §7.1 both primary bodies, force/impulse units and Box frame | Dependency audit “Four-substep timing” and “CUDA graph boundary”; P1 calibration raw solve quaternion | mjlab 1.6.0 `step` calls `MetricsManager.compute_substep` after `scene.update`; `data.xquat` is the solve's pre-integration pose, not final qpos. Capture there, test against pre-step qpos. Use preferred contemporaneous frame, not interval anchoring. | Extend `hoi/contact.py`; original sensor config and functions preserved. | Raw `[E,4,2,2,3]` chronological substep/hand/primary/xyz; quats `[E,4,4]`. Sum primary, inverse rotate per sample, then mean. World impulse `.005*sum`; peak norm; occupancy. Algebra float64 `1e-12`; physical float32 `atol=2e-5,rtol=2e-6`. |
| §7.2 distinct found, net force and occupancy | No built-in found history; force cancellation, dependency audit “Native netforce semantics” | Same hook copies found columns each substep as explicit additional bookkeeping. Final-found means fourth solve before final forward, never post-forward current data. | Capture buffers in contact module, physical-unit Replay sidecars. | Found `[E,4,2,2]` raw counts, plus final per-hand sum; no geometric load equivalence. `F_on` required configuration, no reward threshold. Diagnostic choice initially `1e-3 N` from P1 inactive≤`1e-6 N` and controlled active>`1e-3 N`; retain distributions and sensitivity. |
| §7.3 joint state/action/context/identity/masks | Architecture “Recorder and replay”; state global ordering trap | Reuse P1 JointStateIO84 and SimContext33, original NoisyJointPositionAction.executed_raw29. Store explicit next state sidecar as well as original state rows. No support context until P3 identifiability test. | New `hoi/replay.py`, subclass original ReplayBuffer. Reuse allocation/add/end_episode/rolled/window mask/random selection. | state/next84 split robot71/object13; action clean/exec29; xi33; frame=start frame t, next frame t+1; global intent at both endpoints. Environment is ring row index, episode original counter, clip explicit. |
| §7.4 exact interval and before-reset copying | Dependency callback trace, P1 history replacement/terminal reset evidence | New RecorderTerm uses native pre-reset/post-step callbacks. Substep capture snapshots state at the fourth integration, force, quats and actions before reset/forward; validates exactly four samples and native reversed history. | New `hoi/recorder.py`; install via existing manager APIs on P1 factory environment. No copied stepping loop. Explicit `begin_episode` after caller's physical/reference reset binds frame and clip; auto-reset leaves rows uninitialized until rebound. | Fail closed on incomplete/duplicate intervals, overflow, stale samples, unsynchronized starts. Reference intent never supervises force. Raw simulator next state; no reference velocity as target. |
| §7.5 terminal discrepancy, separate causes | `architecture_checks.json` demonstrates `[1,1]` instead of `[0,1]` | HOI reads all transition labels at row t. Original extra terminal state row retained solely for k+1 state windows and marked non-transition. Physical failure, timeout and natural completion are independent flags; terminal is their OR. | HOI-local gathering only; no shared terminal fix. | Tests expect `[0,1]`, one terminal transition, no phantom reset row, all three reasons and simultaneous reasons. Explicit external reset increments episode and cannot bridge windows. |
| §7.6 eight transitions, no boundary crossing | Original `window_mask` covers episode/write pointer but not clip/consecutive frames | Extend native mask with clip, frame continuity and transition validity. Original k+1 state scheme means newest nonterminal row awaits successor before sampling; explicit next state remains stored immediately. Full windows only, no padding. | HOI `sample_windows` gathers sidecars at identical selected indices using native `_sample_from_mask`; no second sampling algorithm. | Default k=8; all returned validity true. Final transition allowed only at last window slot. Tests ring wrap, partial resets, invalid rows, clip changes, gaps, xi and state continuity. |
| §7.7 diagnostics / §15 report | P0 raw evidence and P1 reusable physical fixtures | Reuse calibration Placements and raw-contact probe in diagnostic script; controlled short native action sequences, independent action schedule and no optimizers. | New `scripts/hoi/diagnose_force_replay.py`, aggregation/replay/recorder tests; JSON/NPZ/logs under prescribed output root. | Per hand/bilateral activity, distributions, onset/loss with episode boundaries excluded, grouped clip/env; raw examples and numeric alignment errors. Physical results are fixtures, not task success. |
| §§1.3.6,15 preservation/reuse | P1 reuse audit; original mainline untouched | No shared-source change is necessary. Runtime additions stay under hoi; no second Trainer/WM/Actor/Critic loop. | Full existing test suite and P1 robot CUDA Replay regression, protected hashes, complete diff review. | No mandatory physical skip counts as PASS. Missing LAFAN1 tests remain explicit existing skips. |

All decisions are implementation-level adaptations or choices explicitly allowed
by §7. No research amendment is proposed. Actual capture reliability remains a
mandatory physical gate until tested. Torque is not configured by P1 and is not
added to the force target. A diagnostic wrench helper, if used, must first shift
primary torques by `(centroid - common_point) cross force`; its algebra is tested.

## Runtime boundary and unresolved risks

P1 has no motion command/reset curriculum. P2's recorder binds explicit reference
metadata after a caller performs the synchronized P1 reset; it advances once per
recorded transition and registers natural clip completion at 838→839. Reset frame839
has no outgoing valid transition and is rejected. This is data collection plumbing,
not a task controller or P7 training integration. Existing termination terms, if
provided, supply physical failure/timeouts; P2 adds no failure policy or rewards.
Step/interval state-mutating events and command state writes are unsupported for
this physical-only collector: reject them rather than silently misalign targets.
Use auto-reset=true; manual reset mode requires separately validated lifecycle work.

P1's force sign/filter calibration, ten dependency hashes and static support fix are
reused. P0 reference-source audit informs the prohibition on importing foreign
sensors/labels; no external algorithm needs rereading or execution for P2. Historical
M6.5 FAILs/training_ready=false, missing target checkpoint, netforce cancellation and
bounded contact capacity remain. No physical/task success follows from Replay PASS.

Acceptance requires executed GPU timing/copy/reset tests and original regression,
with deterministic algebra and identity checks. Missing physical evidence yields
PARTIAL, not an invented threshold or PASS. Exact commands/results and final file
list will be recorded in VALIDATION.md, REPORT.md and the reuse audit.
