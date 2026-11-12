# P6 closure report

**Overall: PARTIAL.** The authorized C1–C5 implementation, diagnosis and bounded
validation work is complete for review. Physical success gates remain unsatisfied.
No P7/P8, WM fitting, Actor/Critic updates, viewer, video, commit or push occurred.

Baseline: `main` at `b4d412479d8b7dd5a174e6f176a23a07c525e1f6`.
Evidence root E=`outputs/hoi/contact_supervision/p6_closure`.

| Independent decision | Status | Scope and remaining limitation |
|---|---|---|
| OBJECT_CONSTRUCTION_VALIDITY |PASS, bounded approximation|Selected one-collider2kg HDMI-sized candidate is documented and independently tested; production remains original.|
| BOX_MODEL_VALIDITY |PASS for candidate; original concern remains|Explicit homogeneous COM/inertia are plausible simulation assumptions. Original COM190mm below the envelope is unreconciled as a real mass model.|
| FLOOR_SOLVER_RELIABILITY |PARTIAL|LS1024 bitmask cause reproduced; opt-in precision guard passes bounded and exact-state tests. Default production remains failing; no universal guarantee.|
| REFERENCE_COMPATIBILITY |PARTIAL|Original native geometry fails. Separate coupled candidate passes290/290 ON kinematic frames, but native replay fails sustained bilateral contact.|
| BILATERAL_CONTACT_FEASIBILITY |PARTIAL|Geometric feasibility established; dynamic lift/carry fails in all four independent worlds.|
| PHYSICAL_VALIDITY |PARTIAL|Isolated rigid-body/support tests pass; natural manipulation and production provenance are not ready.|
| P7_TRAINING_READINESS |FAIL / NOT SATISFIED|No training authorization or physical-readiness gate is inferred from diagnostic completion.|

## What changed and why

The selected diagnostic Box is .18×.34×.27m,2kg, COM(0,0,.125)m in the original
bottom-plate root, root-axis inertia diag(.031416667,.01755,.024666667)kg m².
It preserves main exterior surfaces and omits handles/fills the cavity as an
explicit simulation approximation. It needs smaller, smoother arm corrections
than the exact-union shell. Extra grasp-marker colliders are unnecessary for
demonstrated native bilateral geometry. SMTrack's .30×.32×.36m dimensions cause
non-hand overlap in151/290 original ON frames and were not transplanted. Its1.5kg
mass is an independent comparison, not the selected nominal mass.

The previous uniform component proposal double-counted .000656m³ overlap. New
candidate integration accounts for the exact union. Original assets, accepted
Part B NPZ, manifest and historical reports/failures are preserved. Candidate
assets and reference have separate names and hashes; none replaces production.

The installed `LS_ITERATIONS=1024` is a bitmask, while the actual line-search
limit is20. Exact original failure replays show only this bit, with no capacity
overflow. Source/instrumentation isolate a cost-improving derivative bracket at
float32 resolution that misses the slope termination test. A hash-pinned opt-in
termination guard fixes the measured failures without iteration/material/dt
changes. Three exact-state guarded replays agree with CPU within declared
tolerances, and12 original/selected3s scenarios finish without overflow. The
default production path is deliberately not switched to this diagnostic profile.

C1.7 compiles upstream source and tests ten contact parameter profiles. Dim3 makes
stored torsional/rolling changes inactive. Dim4 adds torsion and more constraints;
solref/solimp materially affect forces/penetration but are uncalibrated. At5ms,
refsafe clamps higher-priority Box solref.001 to.01s. Current-engine gap semantics
were measured directly. Nonzero-margin GPU cases are explicitly NOT TESTED after
Warp rejects the fixed MULTICCD configuration. Baseline materials are recommended
for continued bounded work, not arbitrary stiff/high-friction settings.

## Reused and new validation

Reused P1/P2 native scene/actions, hand–Box filtering, force aggregation, state/
reset utilities, per-world support offsets and Replay recording. Reused P4/P5
coordinate/contact-intent contracts and previous P6 force calibration. No new
force prediction, decoder/BCE/regression loss, force Actor reward or imagined
measured-force input was introduced. State84/canonical86/residual82, WM166/82 and
original Actor/Critic dimensions remain unchanged.

| New execution | Evidence | Result |
|---|---|---|
|20 constructions×840 FK frames|`geometry_02`, `audit_details`|Complete; original motion fails continuous bilateral band|
|8 constructions×13 dynamics cases|`physical_02`|104 executed;56 support/drop/lift/recontact settling gates pass;0 CPU warnings|
|Original/isolated floor, support and elevated CPU/GPU comparison|`solver_01`, `solver_trace_01`|Historical failures reproduced and retained|
|Exact before-failure CPU/GPU paired correction|`failure_replay_02`, `overflow_audit_01`|3 baseline LS-only failures;3 guarded passes|
|Final guarded original/selected scenes|`solver_guard_final`, `solver_candidate_guard_final`|12×600 physics solves, PASS bounded|
|Two coupled840-frame IK candidates|`ik_single_02`, `ik_shell_02`|290/290 ON geometry each; earlier failed attempts retained|
|Separate candidate NPZ|`reference_candidate_v1`|Original intent/root/legs/waist/Box unchanged; candidate training_ready=false|
|9×4 null/left/right/bilateral calibration fixtures|`integrated_calibration_02`|All classification checks PASS|
|Original/candidate bounded native PD replay|`integrated_original_02`, `integrated_candidate_02`, `analysis_02`|2696 valid transitions each; required bilateral physical agreement FAIL|
|Independent4.98s contact/lift/carry/release attempt|`integrated_independent_02`|Solver completion PASS; actual lift/carry FAIL in4/4 worlds|
|40 matched CPU contact-material cases|`contact_physical_01`, `contact_physical_gap_01`|Executed;0 CPU warnings; behavior differences and transient penetration retained|
|8 supported GPU material profiles|`contact_gpu_01`, `contact_gpu_02`, `contact_gpu_failure_details`|All baseline-floor trials LS-only FAIL; guarded floor/native-hold trials complete|
|2 GPU nonzero-margin profiles|`contact_gpu_02/*/unsupported.json`|NOT TESTED dynamically; model conversion rejects baseline CCD configuration|
|Affected software/metric regressions|`tests_05.log`, `tests_05.xml`|16 PASS; one upstream Torch deprecation warning|
|Independent artifact/aggregation/preservation verification|`final_audit/verification.json`|PASS integrity checks; physical FAIL decisions retained|

Force thresholds remain L.0743401N/R.0737510N with strict occupancy>.25(≥2/4
substeps). They are validated on the candidate fixtures, not fitted to reference
labels. Final6424 world intervals reproduce the P1/P2 aggregate measurements.
Selected coupled IK maximum/RMS correction is .133715/.026120rad; max speed
3.44945rad/s, acceleration33.9474rad/s², minimum non-hand clearance23.996mm.

Original/candidate reference OR recall is4.127%/26.349%, but required AND recall
is1.270%/.317%. Both metrics are reported; no favorable aggregate was selected.
Candidate maximum dynamic hand penetration11.023mm remains significant despite
improvement from38.148mm. Independent native carrying fails: world0 maximum rise
23.668mm, no qualifying unsupported lift/carry, and subsequent floor settling.

## Failures, limitations and next steps

Preserved failures include original P6 evidence; early IK acceptance failures;
the initial XML-save API fixture failure; diagnostic guard-cache activation
failure; upstream CollisionCfg import incompatibility; GPU margin conversion
rejection; the Toaster raw-compile-before-editor audit-order error;
and every LS/non-carrying result. The incomplete geometry run was a
diagnostic performance issue, not a physical pass. A final command initially
used a nonexistent test filename; it ran no tests and was corrected. Lint cleanup
removed diagnostic re-exports; they were restored explicitly and an entrypoint
import regression test added. These harness issues are distinguished from model,
solver and reference failures. No old FAIL is silently relabeled PASS.

The original missing V3 planning source hash
`168465ae0a2e17153902a337963574942551909680d0d8dd36f7ffb22102b05c`
still blocks strict production provenance. The robot-only checkpoint path was
already absent in the initial inventory; no checkpoint was deleted. Current
upstream source/configuration and compiled-model evidence do not recover every
historical trained checkpoint/runtime. Material parameters and density remain
simulation assumptions. Bounded tests do not establish long-horizon robustness.

Recommended next work, before any P7: recover/reconcile missing provenance under
the approved amendment process; review the candidate construction/reference and
precision correction; establish sustained unassisted bilateral load-bearing
tracking, lift/carry/placement under native dynamics; then repeat affected
physical/contact/solver gates. A production asset/reference or default solver
promotion should be based on those explicit decisions. Training remains stopped.

## Files, commands and Git review

New application source: `src/supermimic/hoi/solver_precision.py` only; opt-in and
unused by default training/scene paths. New tests:
`tests/hoi/test_solver_precision.py`, `tests/hoi/test_p6_object_candidates.py`.
New specifications/reports are in this `docs/hoi/force_contact/p6/closure/`
directory; all versioned diagnostic scripts, candidate assets/configurations,
reference, upstream snapshots, failures and numerical evidence are under E.
`E/file_inventory.json` enumerates every new artifact with size/hash; source diff
and Git review files are archived beside it. `E/commands.json` contains exact
shell commands and orchestration inputs after this closure authorization.

All1236 baseline inventory paths retain their original hash/missing status;
two were already missing. No tracked baseline file changes, staged changes,
accepted-NPZ changes, production asset/checkpoint changes or dependency changes
occurred. Final Git status consists of pre-existing `.vscode/` plus the new
closure docs/evidence/source/tests. Nothing is committed or pushed.

Detailed findings: [C1 construction](C1_OBJECT_COMPARISON.md),
[C1.7 contact parameters](C1_7_CONTACT_PARAMETERS.md),
[C2 solver](C2_SOLVER_DIAGNOSIS.md), [C3 feasibility](C3_REFERENCE_FEASIBILITY.md),
[C4 integrated validation](C4_INTEGRATED_VALIDATION.md).
