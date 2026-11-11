# P2 — force aggregation and Replay supervision

**PASS for P2 review.** Implemented and tested measured four-substep force labels,
transition-indexed HOI Replay, and native recorder integration. Final suite:
**90 passed, 17 existing missing-LAFAN1 skips, 0 failures**. All P2 physical tests ran.
No P3, Force Head, rewards, imagination, training, checkpoint migration or push.

## Implementation and prerequisite verification

The complete Master Plan and required P0/P1 documents were read before source edits.
The concrete [specification](IMPLEMENTATION_SPEC.md) was committed first as `bb6cb0c`.
Baseline is `main` / `a562d936541ac7a73c6fa70fcf727449e426f28a`, the current P1
implementation commit; P1 specification is `61c2760`. Historical P0/P1 report IDs
were preserved, not rewritten. Old P0 objects exist but are not ancestors of current
HEAD; current reconciled P0 closure is `a5301e4`. File/hash verification establishes
reused evidence integrity without pretending historical IDs are current ancestry.

The working Master Plan remains SHA256
`5b0c26e3c3d8c721e5e1f76ffe194f6141994a9eba4894ce7f43d7266e9be82f`.
The user's instruction explicitly authorized P2 and commits after each group;
no separate commit approval was needed. P3 still requires explicit approval.

Versions remain mjlab1.6.0 at `8bb2bc9e187797689ec82c78c35019afd2fd0376`,
MuJoCo/MuJoCo Warp3.16.0, Warp1.18.0, Torch2.14.1+cu130, RTX4090D.
All ten P0 dependency hashes and P1 implementation/evidence hashes matched before
implementation. Final preservation: 420/421 baseline files unchanged; the sole
intentional extension is `hoi/contact.py`, whose six original functions are
AST-identical. All P0/P1 reports and raw evidence remain untouched. Part B still hashes
to `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`;
M6.5 manifest to `1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`.

- `contact.py` captures both primary columns, solved Box quaternion and found counts
  in the existing per-substep metric hook. It calculates mean Box-local N, world
  impulse N·s, peak N and occupancy. Copies are owned; native history is reacquired
  and checked for exact chronological agreement. Overflow still fails closed.
- `replay.py` subclasses the original ReplayBuffer, retaining its ring storage,
  writes, episode counters and random-selection mechanisms. Extra fields live in
  sidecars. Transition masks use the action's row; terminal state rows have no
  outgoing transition. Windows enforce clip/frame/episode validity.
- `recorder.py` uses native pre-reset/post-step callbacks, original noisy actions,
  SimContext33 and P1 JointStateIO84. Call `install_recording` on the fresh P1 physical
  environment; after each synchronized physical reset, bind frames/clip IDs with
  `begin_episode`. Natural completion occurs at 838→839. No outgoing frame839 action
  is recorded. All other termination terms retain separate cause flags.

## Tensor and timing contract

State/next state84 = robot71 + object13. Positions remain environment-local, wxyz
quaternions, world root-origin linear velocities and body angular velocities.
Clean/executed actions are normalized29, xi33 remains episode-constant. Force is an
auxiliary label, never part of integrated state. No support context is frozen in P2.

Raw arrays: `[E,4,2,2,3]` chronological substep/hand/primary/xyz; solve quaternion
`[E,4,4]`; captured found counts `[E,4,2,2]`. Hands are left/right, primaries pitch/yaw.
Mean force and impulse `[E,2,3]`; peak, occupancy and final-found `[E,2]`. Found-final
means fourth **solved** substep, before final forward; found history is explicit
bookkeeping, not a built-in sensor field or force-derived geometric-contact claim.

The state at frame t and clean/executed action t precede four 5 ms solves. Each force
is rotated using that solve's Box orientation (`xquat`, verified against qpos before
the solve), then averaged. Integrated end-state is copied after the fourth substep.
The terminal callback writes labels before reset; the nonterminal callback checks
that no event changed the physical target and copies the same captured interval.
Global Part B intent stores `[c_ref[t], c_ref[t+1]]` separately from actual contact.

Masks are independent `physical_failure`, `timeout`, `clip_end`; transition-terminal
is their OR. Original storage-terminal means an extra state-only row, explicitly
invalid as a transition. All sampled windows contain eight consecutive transitions,
nine states and true validity masks, with no padding. Latest nonterminal row waits
for a successor row under original sampling semantics; its explicit next state is
already stored. Metadata records frame/sign/units/threshold/sensors/reference/asset
provenance for later checkpoint integration; P2 writes no training checkpoint.

## Executed numerical evidence

Final physical artifacts: `outputs/hoi/contact_supervision/replay_diagnostics/run_03/`.
The main diagnostic collects 24 control steps × 4 worlds = **96 transitions**.
Additional active-terminal graph/eager probes collect **8 transitions**. In total,
104 vector substep callbacks represent 416 individual world/substep samples.
Two clip IDs deliberately reference the same accepted motion to exercise identity
boundaries; this is not evidence of motion diversity. Fixtures are elevated contact
placements and prescribed noisy actions, not tracking or task-success rollouts.

| Check | Result |
|---|---|
| Both primary bodies, changing Box orientation, units | Deterministic CPU tests pass; max independent matrix mean-force error `3.07967e-6 N` physically |
| Solve orientation vs pre-step qpos | Max quaternion component error `1.19209e-7` |
| Independent raw-contact sums vs sensor columns | Max error `1.75834e-6 N` |
| Action, raw-force, next-state and native-history alignment | Exact agreement; maximum recorded errors 0 |
| No-contact tolerance | Measured max force 0 N, tolerance `1e-6 N` |
| Noisy executed actions | Observed `(a_exec-a_clean)` std `0.0302866` versus configured `.03` |
| Terminal cause counts, main dataset | Physical4, timeout4, natural clip-end5; union9 (causes overlap) |
| Active terminal copy after reset | Left-only peak `1.538914 N`, right-only `1.971448 N`, bilateral `[43.838249,32.259716] N`; stored raw copies unchanged after further reset |
| Graph/eager parity, active terminal probe | Max state difference `9.53674e-7`, mean-force `5.72205e-6 N`, raw-force `1.47820e-5 N`, impulse `1.19209e-7 N·s` |
| Window identities/alignment | 512 physical eight-step windows, zero boundary violations; CPU ring wrap, clip/frame gaps, invalid rows and reset tests pass |
| P0 terminal-label discrepancy | HOI two-step labels `[0,1]` for each terminal cause; no successor-row double label |
| Capacity | All 104 final-run vector samples have zero overflow flags |
| Full tests | 90 passed / 17 existing LAFAN1-data skips; original robot CUDA Replay regression passes |

Main-dataset diagnostics at explicit `F_on=.001 N` (diagnostic threshold, not a
reward coefficient):

| Statistic | Left | Right |
|---|---:|---:|
| Active interval ratio (occupancy >0) | 10.4167% | 8.3333% |
| Mean interval-force norm, N | .757089 | .531016 |
| 95th percentile interval-force norm, N | 4.628196 | 1.101860 |
| Maximum interval peak, N | 43.006920 | 30.906328 |
| Mean impulse norm, N·s | .015141 | .010617 |
| Maximum impulse norm, N·s | .449118 | .305976 |
| Mean occupancy | .088542 | .067708 |
| Onset / loss count across 87 valid adjacent pairs | 0 / 4 | 0 / 5 |

Bilateral interval ratio is 4.1667%. Loss rates are 2.29885 / 2.87356 Hz over the
1.74 s of valid adjacency; onset rates are zero because active fixtures begin with
contact. Resets are excluded. Full min/median/p95/max distributions, threshold
sensitivity, per-environment/per-clip statistics and six aligned examples are in
`diagnostics.json`; complete samples/labels are in `raw_samples.json` and NPZs.

## Failure history, limitations and adjustments

`tests_cpu_01`: 20 passed. `run_01`: passed initial timing checks but inspection found
all terminal force labels zero; added an explicit nonzero-terminal probe instead
of claiming that coverage. `run_02`: passed active-terminal checks. First complete
suite: **1 failed, 89 passed, 17 skipped**, rejecting bilateral-world solver flag1024
(`LS_ITERATIONS`). That failure remains in `tests_full_01.log` and the copied failure
JSON. Its partial raw samples were not flushed by the original harness; later
failures now preserve partial samples and flags.

Spec A1 documents the diagnostic-only correction: move both bilateral free roots
+2 m together to isolate contact recording from floor impacts. The precise source
of the original intermittent solver limit is not established. `run_03` and the
second full suite pass with unchanged solver iterations10/line-search20, assets,
actions, gravity, overflow rejection and tolerances. No research amendment or
shared-source change was required. The original floor-contact fixture failure is
not reclassified as physical success.

Historical M6.5 contact/support FAILs and `training_ready=false` remain. Net force can
cancel opposing local loads. Capacity/success claims cover these fixtures only.
P2 collector intentionally requires auto-reset and no commands/state-mutating events;
P7 must integrate validated command/reset lifecycle before training. Short-window
padding and exact stochastic checkpoint resume are not implemented. Seventeen
original tests require absent LAFAN1 NPZ/CSV files; none is relabeled as passing.

## Files, Git review and stop

Implementation commit: `5730be4d0c07862c7e91da75803ea4626384f7c4`; specification
commit: `bb6cb0c`. Final report/evidence commit follows this report, under the user's
explicit group-commit authorization. No push. [VALIDATION.md](VALIDATION.md) records
commands and exact review exceptions; [validation_summary.json](validation_summary.json)
records all artifact/source hashes and results.

Stage diff: **32 files: 31 new, one existing HOI file extended; zero shared-source
changes**. Every changed/new file is listed in [files.json](files.json). Groups:

- Extended `src/supermimic/hoi/contact.py`; new `hoi/replay.py`, `hoi/recorder.py`.
- New `scripts/hoi/diagnose_force_replay.py` and three `tests/hoi/test_force_*.py` files.
- New P2 specification, baseline, report, validation, reuse audit, summary and file inventory.
- New diagnostic runs01–03 (logs, JSONs, transition archives; active-terminal archives
  in runs02–03), CPU/full-suite logs and preserved first-full-suite failure JSON.

The complete source/test/diagnostic diff was inspected; machine evidence was checked
structurally and numerically, including recomputation from saved NPZs. Raw failed
pytest output retains nine trailing-space lines verbatim; scoped diff whitespace
check passes for all other P2 files. Ignored NPZs are explicitly added in the final
evidence commit. User changes remain exactly the master typo correction, `.vscode/`,
untracked V3 and `reference_refined/`; none is included in P2 commits.

Training Framework Reuse Audit: **no shared modifications, no duplicated training
algorithm**, original Replay mechanisms reused; see
[TRAINING_FRAMEWORK_REUSE_AUDIT.md](TRAINING_FRAMEWORK_REUSE_AUDIT.md).

**P2 is complete and stops here for explicit review. No P3 or later work has started.**
