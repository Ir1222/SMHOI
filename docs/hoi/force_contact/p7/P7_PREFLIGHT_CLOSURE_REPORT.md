# P7 preflight closure

**PREFLIGHT = PARTIAL. Stop for physical-profile and bounded-collection review.**
The obsolete planning dependency is retired. Strict current-Master provenance,
checkpoint rejection, actual-entry construction and guarded resume work. The
production physical gate remains closed; natural lift/carry coverage is not
established. No P7-A warm-up, P7-B training or P8 was started. Every optimizer
executed **zero steps** during this preflight, including the native smoke tests.

Baseline: `c86a483c200f445b9f565acf3c78fb017e584c84`, 2026-11-13.
Evidence root **E**: `outputs/hoi/contact_supervision/p7_preflight/`.
This report implements execution Part 1 of the current user request; it is not a
replacement research plan. Accepted P0–P7.0.4 code and historical evidence are reused.

## Authority and strict provenance

The sole authority is
[`SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md`](../SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md),
SHA256 `25554b0987d70e778b1c41b33cc15512b876e3ee9f359536a850545d63305841`.
Its bytes are unchanged. The legacy attribution in its header is superseded by
the user's explicit current authority decision, without rewriting the Master.
No old V3 file was restored, fabricated, opened or made a prerequisite.

The named migration `retire-incorrect-v3-use-current-master-v1` replaces the exact
retired planning record in the **effective in-memory source list** with the
current Master path/hash. The old filename/hash is recognized only to identify
that historical metadata record. An unknown retired hash fails closed. Every
nonplanning source hash remains mandatory; missing or changed physical sources
still fail. Historical manifests, baselines, reports and snapshots remain unchanged.

The shared strict loader now serves physical scene construction, support context,
reward geometry audit and diagnostic profiles. Future audit/build/reconciliation
tools use the same migration, including indirect dependencies through archived
baseline dictionaries. Historical audit generators were **not** rerun over their
old output paths. Historical freeze checks retain their original nonplanning
hash expectations; this migration does not rewrite history to claim they pass.

New full-training, Joint-WM and Actor/Critic checkpoints bind current authority.
Full checkpoints also retain reference/manifest, Box, support, contact targets,
calibration, compiled model, solver, dependency, feature schema and code identities.
Replay metadata now carries current authority. Missing/wrong checkpoint authority
is rejected before state mutation. Legacy weights remain available only through
the existing explicit named initialization migrations; this does not migrate old
optimizer/sampler state into a current HOI resume. No historical checkpoint was
modified or automatically blessed by replacing its metadata.

`hoi-joint-physical-wm-v3` is the accepted **model schema**, unrelated to the
retired planning document. It remains unchanged. Physical84/canonical86,
WM166→82, support18/xi33/action29, Actor B1080/830 and Critic B+1015 are preserved.

Strict loading of the original asset now succeeds on provenance. Production
training correctly rejects its unchanged `training_ready=false` physical gate;
missing V3 is no longer the error. This is **not production physical approval**.

## Physical profile requiring review

The concrete, unapproved proposal is
[`E/physical_profile_proposal_v1.json`](../../../../outputs/hoi/contact_supervision/p7_preflight/physical_profile_proposal_v1.json),
ID `p7-single-body-original-ref-v1-proposed`. It records full object/support
parameters and identities, with `approved=false`, `production_selected=false`
and `training_ready=false`. No new production profile was enabled.

| Item | Existing original | Proposed bounded profile |
|---|---|---|
| Box construction | Seven collision boxes, original handles/cavity approximation | Existing P6 `single_body`, homogeneous solid, one box collider; cavity filled, handles omitted |
| Mass | 2 kg | 2 kg |
| Root convention | Original HDMI bottom-plate centre | Same; original object pose trajectory unchanged |
| COM in root coordinates | `(0,0,-0.2)` m, 190 mm below collision envelope | `(0,0,0.125)` m |
| Root-axis inertia at COM | Original declared principal inertia `.0334,.02167,.04167` kg m² | `diag(.0314166667,.01755,.0246666667)` kg m² |
| Stored principal inertia / axes | Original | `(.01755,.0246666667,.0314166667)`, quaternion `(.5,.5,.5,.5)` |
| Collision extent | Main envelope z `[-.01,.26]` m | Full dimensions `.18×.34×.27` m, centre `(0,0,.125)`, same z envelope |
| Status | Unreconciled physical mass model | Independently tested simulation approximation; requires explicit experimental approval |

The recommended candidate follows the existing
[C1 selection](../p6/closure/CANDIDATE_SELECTION.md) and
[construction audit](../p6/closure/C1_OBJECT_COMPARISON.md). It is not a measured
real Box density model. The exact-union shell is another documented candidate,
with COM z `.1083794466` m and principal inertia
`(.0233188583,.0360563900,.0412730085)` kg m²; it is not silently substituted.
Handle or interior manipulation would invalidate the solid approximation.

Supports remain `.20×.20×.04` m with original orientations and centres
`(-1.466734,-.159754,.660550)` and `(1.381318,-.284395,.625123)` m.
The original reference endpoints retain their approximately 10 mm support gap.
Native Robot collision shapes/actuators, contact targets and material parameters
are unchanged. The accepted C1 evidence contains 104 matched rigid-body cases,
56 support/drop/lift/recontact settling passes and analytic velocity error
`1.29e-8`. Its externally applied 24 N lift is **not natural HOI data**.

| Identity | SHA256 |
|---|---|
| Original production manifest | `1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1` |
| Existing single-body candidate file | `6c3f1a9c601336cb264cd0f979ec4d039ed380e76df6ee1a0a33358d5a82e331` |
| Original accepted reference | `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b` |
| Source HOI NPZ | `d89da14d49cc5f86398f3a86ad6037e777f5588972eaf7441917833da4b04dbe` |
| Contact calibration | `34a47e06454c64f71b456bb428110fceff65302f6c6517dc85a95e151d400266` |
| Support JSON identity | `be2f474c19e52a0caf1ccaf5287caacf42cde61fd44bb633229ba70eb01e9df7` |
| Experimental reference candidate, still unapproved | `f38071e5347403f9eff993dd37884bf374e0c003400156dc4cb87d3d2b856cb0` |

## Solver, reset and invalid-transition audit

The actual entry requires the existing guard throughout construction, graph
capture, stepping and resume. Source hash:
`1bb26d51451a065b9e183acacfbd5e59d85df1eb46f17951aa3d1727db6853ff`;
patched fragment hash:
`41c06e49c60862d0d40e69c74cae8c0f2310d300516b096b5464607a4efe6dcd`.
Physics remains 5 ms × 4 substeps, iterations 10, line-search iterations 20;
friction/solref/solimp are unchanged. Overflow warnings remain `ALL` (4095).
Installed dependencies are not patched on disk. Unknown source revisions fail closed.

Actual `smHOI-track` single-body/original-reference execution completed four
control steps in two worlds, then a fresh process resumed checkpoint 2 through
step 4. The guard observed its named kernel after initial capture and resume
recapture; it checked 32 and 16 world-substeps respectively. Both runs have zero
overflow and zero WM/Actor/Critic updates. Eight retained intervals are all
reference-OFF and no hand contact; this smoke establishes no manipulation coverage.
Final physical-state difference versus uninterrupted execution is
`1.5050172805786133e-6`; model weights, Replay discrete indices and identities match.
Exact state restoration is separately verified by the new checkpoint test.
This does not supersede P7.0.4's non-bit-exact physical continuation limitation.
See [entry evidence](../../../../outputs/hoi/contact_supervision/p7_preflight/entry_evidence.json).

Injected failures exercise all worlds independently: overflow, nonfinite native
state, applied wrench and invalid action mapping abort before Replay insertion.
Existing loss/gradient checks and original optimizer hooks remain. No finite
check is interpreted as geometric acceptance. Original physical failure, timeout,
clip-end, reached-state capture, bootstrap and continuation semantics are unchanged.

New reset inspection uses two worlds × six original reference frames × two
existing profiles: **24 resets, zero physical steps**, CPU forward geometry after
native reset binding. All qacc are finite; maximum reset state round-trip error
is `1.1920928955078125e-7`. Minimum signed distance (world 0; world 1 agrees in the
saved report) is:

| Reference frame | Original Box, mm | Single-body Box, mm | Deepest pair |
|---|---:|---:|---|
| 0 | -12.235645 | -12.235645 | Left foot / floor |
| 250 | -11.840940 | -11.840940 | Left foot / floor |
| 350 | -28.126243 | -19.707578 | Right wrist / Box |
| 475 | -35.000000 | -35.000000 | Left wrist / Box |
| 550 | -33.116139 | -35.000000 | Left wrist / Box |
| 740 | -4.875866 | -4.875866 | Right foot / floor |

Full pair records are in
[`resets_01.json`](../../../../outputs/hoi/contact_supervision/p7_preflight/resets_01.json).
These contact-frame overlaps are substantial, not proof of safe resets. No pose
offset, reference teleport, contact parameter adjustment or new penetration
threshold was introduced. Current validity masks reject invalid solves/nonfinite
transitions and preserve boundaries, but **do not certify acceptable penetration**.
Admitting the known frame-0 foot overlap for a bounded experiment requires review.
The proposal excludes contact-frame resets rather than manufacturing a grasp.

The unblocked legacy P2 subprocess exposed a numerical parity failure: graph
versus eager force histories differ in 2/192 entries, maximum absolute difference
`4.57763671875e-5 N`, against its existing `atol=2e-5, rtol=2e-6` check. Earlier
same-run capture/terminal assertions reached that comparison successfully. The
tolerance was not relaxed and the failing test was not rerun to select a pass.
Its cause is not established; it remains a bounded measurement/reproducibility
issue requiring explicit disposition or diagnosis before claiming full P2 parity.

## Reference compatibility and P6 disposition

Original HDMI motion, labels and targets remain unchanged. Both sides of the
existing C4 original/candidate reference comparison use the **single-body Box**;
“integrated_original” denotes the reference, not the original seven-geom asset.
Each has 2696 transitions from six overlapping windows and four worlds.

| Reference / measured aggregate | TN, FP, FN, TP | F1 |
|---|---|---:|
| Original / any hand | 1420,16,1208,52 | .078313 |
| Original / bilateral | 1428,8,1244,16 | .024922 |
| Candidate / any hand | 1395,41,928,332 | .406614 |
| Candidate / bilateral | 1430,6,1256,4 | .006299 |

The candidate's geometric 290/290 ON-frame feasibility did not produce sustained
dynamic carrying. Independent native lift/carry failed in **4/4 worlds**; maximum
world-0 rise was 23.668 mm and qualifying unsupported lift/carry duration was zero.
Its `.586 m` horizontal movement was falling, not carrying. Historical penetration
and solver failures remain in [C4](../p6/closure/C4_INTEGRATED_VALIDATION.md).
Reference mismatch alone does not invalidate independently sound physical WM
transitions; neither good reference agreement nor finite stepping proves physics.

| Blocker | Classification and disposition |
|---|---|
| Retired V3 prerequisite | Software/provenance blocker **resolved** by explicit current-authority migration; no file recovery required |
| Production M6.5 false/NO_GO and original COM | **Unresolved research-level asset decision**; strict production remains blocked; versioned solid approximation awaits review |
| Historical LS1024 / iteration failures | **Hard physical blocker without guard; bounded limitation with guard**; bounded guard evidence reused, no universal reliability claim |
| Reset hand/foot penetration | **Reference compatibility issue and physical-admission decision**; finite states do not settle it; frame-0-only collection proposed |
| Original native reference mismatch | **Reference compatibility issue**; retain all labels/frames and report both aggregates; no automatic relabeling |
| Candidate dynamic bilateral failure | **Reference compatibility/control limitation**; `reference_candidate_v1.training_ready=false` remains |
| Natural unsupported lift/carry missing | **Unresolved data/control research issue**; collection may still fail to provide it; no substitute external-wrench examples |
| Legacy P2 graph/eager force parity | **Unresolved bounded numerical limitation**; current failure retained, exact cause undiagnosed |
| Non-bit-exact physical resume | **Bounded numerical limitation**; exact restore and identities pass; continued trajectories are close, not identical |
| Reward scales, return bins, Box-only imagined failure | **P7-B research/calibration decisions**; test coefficients are not production settings; no new head or termination rule introduced |
| Learned WM fidelity and action sensitivity | **Hard P7-B prerequisite**; P7-A has not run, so this remains unverified |

## Physical Replay inventory and coverage

Read-only inventory covers all repository `outputs/hoi` NPZ/transition traces and
surviving checkpoints under that tree, `/tmp/smhoi*` and `/tmp/pytest-of-han`.
No external dataset is assumed. Final inventory: **15 NPZ, 97 interval JSON,
4 policy JSONL and 28 checkpoint files** (10 whole-HOI lifecycle checkpoints).
Temporary pytest generations can expire; inventory hashes preserve what was
present at audit time. Earlier inventory versions and parser failures remain.
See [audit_final](../../../../outputs/hoi/contact_supervision/p7_preflight/audit_final/summary.json).

| Source | Rows / source segments | Calibrated contact none/L-only/R-only/both | Admission |
|---|---:|---|---|
| P3 controlled collection_02 | 320 / 32 | 272/20/12/16 | Explicit `natural_HOI_training_data=false`; sensor/dynamics fixture only |
| P3 support_02 | 64 / 64 one-step fixtures | 32/16/16/0 | Paired controlled fixture; no eight-step window |
| P2 replay run_01/02/03 | 96 / 9 each | 85/4/3/4; 85/4/3/4; 84/5/3/4 | Repeated controlled diagnostics; missing xi |
| Earlier failed captures | 48 / 4 and 1 / 1 | 43/2/1/2 and 0/1/0/0 | Historical FAIL, retained and excluded from automatic admission |
| Original P6 reference replay | 1124 / 24 world-window segments | 975/56/29/64 | Two failed solver windows with recorded prefixes; incomplete training archive |
| C4 single-body / original reference | 2696 / 24 | 2628/16/28/24 | Controller-driven dynamics; potentially useful after provenance-complete conversion and physical approval |
| C4 single-body / candidate reference | 2696 / 24 | 2323/149/214/10 | Experimental reference unapproved; separate lineage |
| C4 independent native controller | 996 / 4 | 986/2/0/8 | Controlled native maneuver; lift/carry failed 4/4 |
| P7 readiness physical_final | 50 / 25 vector steps | 46/2/0/2 | Candidate-reference policy diagnostic; earlier 02/03 are related reruns, 01 empty failure |
| P7.0.4 continuous final checkpoint | 32 / 4 episode segments | See checkpoint inventory | Complete physical Replay but only 7 valid eight-transition windows; intermediate/resumed snapshots overlap |
| New preflight actual-entry smoke | 8 / 2 partial episodes | 8/0/0/0 | Software regression only; zero eight-transition windows |

Counts across rows/snapshots must **not** be summed as independent demonstrations.
Multiple worlds share starts/controllers; C4 windows overlap in reference time.
Checkpoint transition-content hashes expose identical snapshots; earlier/later
snapshots also overlap without being identical. A future split must group complete
source lineage before windowing, not split these copies across train/validation.

The final inventory contains per-source state/action speed quantiles, frame
continuity, state-link error, terminal fields, finite/mask counts, available
eight-transition windows, raw-force classification and missing fields. No saved
validity mask is reported as **unknown**, not silently certified by finiteness.
For example C4 original-reference frame-250 window has Box linear speed
median/max `.19758/3.80516 m/s`, angular speed `.45804/6.78157 rad/s`, and Robot
root speed `.07542/1.11967 m/s`; action min/median/max `-3.43757/0/1.97428`.
These distributions establish motion, not stable locomotion or carrying.

P6 intervals include state84/action/reached-state and independent raw substep
forces/overflow, but do not persist xi33 and separately identified clean/executed
actions. Audit-local world/episode/frame identities are explicitly reconstructed
from file identity, saved start and row order; unbound fixtures use an ordinal,
not a reference frame. This is not a training archive conversion. P7 readiness
JSONL also lacks authoritative episode/clip/xi/action-execution metadata. No
missing field was fabricated or silently inferred for optimizer targets.

Coverage decision: **INSUFFICIENT for complete P7-A readiness.** Stationary,
translation/rotation, contact/loss, falling/free motion and support interaction
have physical evidence, with controlled and controller-driven categories kept
separate. Stable locomotion and representative natural manipulation distribution
are not established. Hand-supported lift and carry are unverified/failed;
external lift and FK placement cannot fill that gap. Terminal/reset interfaces
pass bounded checks, not broad regime coverage. No dataset was admitted and no
source-episode train/validation partition was executed in this preflight.

## Proposed bounded collection, not executed

This is the concrete Part 1.10 protocol requiring approval, not another high-level
plan. It is conditional on the versioned profile and reset/solver limitations
above being explicitly accepted for a bounded experiment.

1. Use the proposed single-body profile, original reference/labels, baseline
   native G1 PD mapping and active pinned guard. Reuse the existing P6
   joint-position reference controller as a collection action source. Roots and
   Box remain free. No optimizer updates, learned force heads, pose assistance,
   external wrench, new collision geometry or post-reset state writes.
2. Four worlds, seeds 0 and 1, at most 839 control steps per seed: **6712 world
   transitions maximum**, plus a global 600 s wall-clock cap. This is a ceiling,
   not a minimum or automatic extension. Proposed action noise std `.03` is an
   explicit collection setting already exercised in P7.0.4, not a calibrated
   production default; no reset noise/domain randomization. All normal resets
   bind original frame 0. No starts inside the deeply overlapping contact frames.
3. Preserve original physical Robot failure, time-limit and natural clip-end
   semantics and every reached terminal state. Normal per-world resets create
   new source-episode IDs. Retain every failure denominator; do not discard
   transitions merely for disagreement with reference labels. The known frame-0
   foot overlap requires explicit bounded approval; no new “safe depth” is invented.
4. Use original `Trainer.run()` collection scheduling and accepted P2/HOI recorder,
   with an explicit approval-gated collection configuration/action adapter.
   This mode is not yet implemented: current integration mode's 64-step ceiling
   was not bypassed, and alternating mode must not be misused for collection.
5. Persist physical84 start/reached states, clean and executed action29, xi33,
   clip/frame/environment/source-episode IDs, full validity and termination masks,
   elapsed time and reset metadata. Persist four-substep independent force,
   contact geometry/depth/duration, support load path, overflow/solver data and
   all profile/controller/source/checkpoint/RNG/code hashes. Use exclusive new
   paths; archive full episodes before making eight-transition windows.
6. Abort immediately on any invalid solve, overflow or nonfinite value in any
   world. Keep the failed interval and its physical snapshot outside training
   targets. Quarantine any ambiguous alignment or incomplete interval. Preserve
   prior measured prefixes separately for explicit admission review. Record
   overlap distributions; an unexplained worsening of the approved physical
   limitations stops collection for review rather than triggers silent tuning.
7. Predeclare whole seed-0 episode lineages as training and seed-1 as held-out;
   deduplicate/group identical trajectories before window construction. All
   descendants/resets and repeated starts from a lineage stay in its partition.
   Both use one HDMI clip, so this cannot establish new-motion generalization.
8. Re-audit every §12 regime and actual action/state coverage. Reuse C4's
   unsupported lift/carry definitions (≥3 cm rise for ≥.2 s; carry additionally
   ≥3 cm lateral travel with bilateral contact ≥.2 s), preserving support/contact
   evidence and failure counts. These are existing diagnostic definitions,
   not new WM readiness thresholds. If the controller still lacks natural
   lift/carry, stop with incomplete coverage; do not fabricate contact episodes
   or automatically increase budget. Controller research/reference changes
   require a separate decision.

Collection approval alone does not establish preflight acceptance or WM readiness.
After collection/admission review, P7-A must predeclare a bounded WM update budget
and evidence-based prediction/gradient criteria from the available data. No
numeric WM threshold or convergence claim is selected here. P7-A must still
deliver source-episode partitions, named initialization, Robot/Box one-/multi-step
held-out errors, initialized and constant-velocity baselines, action-to-Box
Jacobians, reward-to-Actor gradients and full checkpoint provenance. Actor/Critic
optimizers stay disabled. **WM_READINESS is NOT EVALUATED: P7-A has not started.**
P7-B remains separately gated on explicit approval and its own budget/checkpoint
schedule, calibration, actual updates and A/B/C ablations. P8 is not authorized.

## Verification, failures and acceptance

Exact commands and outcomes are in
[`E/commands.json`](../../../../outputs/hoi/contact_supervision/p7_preflight/commands.json).
Commands use `.venv/bin/python`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, bounded
timeouts and fresh output paths. No packages were installed. Shell execution
needed the reviewed escalation fallback because sandbox loopback setup fails;
there was no automatic approval rejection.

| Execution | Result |
|---|---|
| `tests_01`: provenance, Joint architecture, support, migrations, policy checkpoint, P2 recorder, guard, all-world safety, original Robot lifecycle | **31 PASS, 1 FAIL**: P2 force graph/eager tolerance above; provenance now reaches physical execution |
| `tests_02`: changed Replay authority and explicit WM/policy/full-checkpoint negatives | **18 PASS, 1 harness FAIL**: missing metadata correctly rejected, but newly added test expected wrong error text |
| `tests_03`: corrected missing-metadata assertion | **1 PASS**; no loader relaxation |
| `historical_freeze_01`: migrated old M6.5 freeze test | **FAIL**: 170 identities match; 13 old source-code hashes differ after accepted later implementations and this migration; no missing V3 prerequisite. Historical expectations were not rewritten to force PASS |
| `resets_01` | **PASS measurement**, **PARTIAL physical admission**: 24 finite/aligned forward-only resets, overlaps reported |
| `entry_01` and `entry_resume_01` | **PASS bounded execution/guard/identity**, **PARTIAL trajectory equivalence**, zero optimizer updates |
| `audit_final` | **PASS inventory**, **FAIL required natural coverage** |
| Focused lint and source whitespace | **PASS** |

The two failure classes in the final regression record are kept distinct:
unresolved physical numerical parity, and a historical stage-freeze assertion
that still expects pre-P0–P7 source hashes. The latter is not used to require a
retired planning file or to undo accepted implementations. Current preservation
is checked against the requested `c86a483` baseline separately.

Earlier `audit_01` failed on an unbound diagnostic's `reference_start=None`;
`audit_03` read its own inventory as raw intervals; `audit_04/05` exposed the
JSONL non-vector reward-summary format. The corrected inventory distinguishes
all of these without changing source data. Failed logs remain intact. TorchScript
deprecation and deliberately injected overflow warnings remain visible. No
physics tolerance, warning suppression or failure relabeling was used.

| Part 1 requirement | Decision | Remaining condition |
|---|---|---|
| 1. Master/provenance migration | **PASS** | Current path/hash authoritative; historical records preserved |
| 2. Strict production loader and checkpoint identity | **PASS software; PARTIAL production readiness** | Physical false/NO_GO gate intentionally rejects production |
| 3. Resolve/approve versioned Box profile | **PARTIAL** | Concrete proposal delivered; user research approval required |
| 4. COM/inertia/support/collision audit | **PASS audit; PARTIAL physical choice** | Original COM unresolved; solid approximation bounded |
| 5. Guard and fail-fast | **PASS bounded** | Long-horizon reliability not established; force parity issue retained |
| 6. Reset penetration/invalid rejection | **PARTIAL** | Solve-invalid targets rejected; geometric admissibility not automatically inferred |
| 7. P6 reference compatibility review | **PASS audit; PARTIAL compatibility** | Original unchanged; candidate not promoted; dynamic bilateral failure remains |
| 8. Available physical Replay/provenance audit | **PASS inventory; PARTIAL admission** | Missing metadata, duplicate lineage and physical approval prevent automatic use |
| 9. Determine natural coverage sufficiency | **PASS determination; FAIL sufficiency** | No verified representative natural lift/carry coverage |
| 10. Bounded collection protocol if insufficient | **PASS proposal; PARTIAL execution gate** | Explicit profile/limitation/budget approval required before implementation/run |
| Original `sm-train` / architecture regression | **PASS** | Existing Robot lifecycle, defaults and checkpoints unchanged |
| Full P7 acceptance (§12/§17) | **NOT SATISFIED** | No P7-A/B learned readiness, alternating updates, calibration or ablations claimed |

## File-change map and preservation

| Files | Purpose |
|---|---|
| `src/supermimic/hoi/provenance.py` (new) | Shared current-Master identity and exact metadata migration |
| `asset_manifest.py`, `support_context.py`, `training_scene.py` in the same directory | Strict shared source validation; remove missing-V3 bypass; bind authority into physical identity |
| `checkpoint.py`, `policy_checkpoint.py`, `training_checkpoint.py` | Bind and validate current authority on new weights/full snapshots |
| `recorder.py` | Current authority in new Replay provenance |
| `scripts/hoi/baseline.py`, `build_manifest.py`, `run_audit.py`, `reconcile/common.py`, `verify_p3_evidence.py`, `audit_proximity_reward.py` | Migrate direct and indirect obsolete planning dependencies without regenerating old evidence |
| `scripts/hoi/audit_p7_preflight.py` (new) | Read-only archive/checkpoint/lineage/contact/profile inventory |
| `scripts/hoi/audit_p7_resets.py` (new) | Forward-only two-profile/two-world reset geometry audit |
| `tests/hoi/test_preflight_provenance.py` (new) | No retired-file access; strict physical/current-plan failures; production gate; full-checkpoint atomic rejection/restore |
| `tests/hoi/test_joint_architecture.py`, `test_checkpoint_migration.py`, `test_hoi_policy_checkpoint.py` | Shared-validation interception and explicit authority negatives |
| `tests/test_hoi_reconciliation.py` | Migrate planning record only; preserve historical nonplanning freeze expectations and its failure |
| This report | Decisions, numerical evidence, concrete collection proposal and review boundary |
| E | Baseline/preservation, inventory, reset and entry evidence, proposal, commands, logs/XML including failures, review helper and per-file hash map |

`E/files.json` enumerates all new evidence files; `E/preservation.json` records
every changed baseline path and all preserved inputs. Existing `.vscode/`, Master,
source references, candidate reference, assets, calibration, historical outputs
and checkpoints are preserved. No original Robot source/default/config file was
changed in this preflight. New full checkpoint binaries remain in
`/tmp/smhoi_p7_preflight`, with paths/hashes captured in E; they are integration
artifacts, not a P7-A trained WM. Changes are committed locally; no push occurs.

- [x] Sole current-Master authority and strict physical provenance implemented.
- [x] Actual entry/guard/resume and zero-update regression evidence recorded.
- [x] Physical/reference/data blockers classified without altering their evidence.
- [x] Versioned profile and capped collection proposal made reviewable.
- [ ] User approves a bounded physical profile and the known reset/solver limitations.
- [ ] User approves collection budget/protocol; valid natural data is collected/admitted.
- [ ] Preflight accepted; P7-A budget/criteria declared and WM readiness demonstrated.
- [ ] Explicit P7-B approval, budget and checkpoint schedule received.
- [ ] Actual alternating updates, reward/return calibration and A/B/C evidence delivered.

**Review requested:** approve or revise the proposed single-body/original-reference
profile and frame-0-only bounded collection, including its documented physical
limitations. This stop follows the user's Part 1 instruction and Master §12's
requirement for PHYSICAL_VALIDITY PASS or explicit bounded-experiment approval.
It does not request recovery of the retired plan. No later stage begins pending
that decision.
