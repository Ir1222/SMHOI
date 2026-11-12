# P6 implementation specification

Authority: complete current Master Plan, especially §§1.3, 3, 4, 9, 11, 14–17;
user authorization covers all P6-A/P6-B implementation and bounded validation as
one stage. No P7/P8, optimization, checkpoint migration, commit or push.

Baseline: main at e4f574185597d23570fad462224f76f741b36538; Master SHA256
68142af38edb0a9a443aac5815491aaae90dba696acc84d30fc54223c6e4a9cc.
No applicable AGENTS.md exists in the repository or its ancestor directories.
Pre-existing untracked `.vscode/` is protected. `baseline.json` in the evidence root
hashes 770 paths, including ignored references/checkpoints and external sources.
Installed mjlab1.6.0 (8bb2bc9), MuJoCo/Warp3.16.0, Warp1.18.0, Torch2.14.1+cu130;
all ten P0 dependency hashes match. P5 software PASS is already committed; P3/P4
historical PARTIAL and M6.5 NO-GO remain. P0 superseded force-head designs do not
apply to this stage. No prior passing P1/P2 suite is scheduled for repetition.

Evidence root: `outputs/hoi/contact_supervision/contact_validation/` (abbreviated E).
New diagnostics/configuration live in E; reusable evaluation mathematics in
`src/supermimic/hoi/contact_validation.py`; focused tests in
`tests/hoi/test_physical_contact_agreement.py` and `test_collision_diagnostics.py`.
Four required reports live here. Existing shared/training/asset/dependency files
are not expected to change. Every run uses a new directory and preserves failures.

## Evidence inspected and reused

P0 CLOSURE, REQUIREMENT_COVERAGE, REPORT, FILE_CHANGE_MAP, architecture,
dependency/contact, dataset/asset and reference implementation audits; P1/P2
reports and actual scene/contact/state/recorder/Replay/calibration implementations;
P3 report, cleanup and failed contact evidence; P4 report and CONTACT_TARGET_AUDIT;
P5 report and final suite summaries; M6.5 reconciliation, dynamics protocol and
failed dynamics records. The current installed environment step confirms exactly
four solve → sensor → history → metric callbacks, then one final forward.

P1 run_05 already establishes all 28 hand/primary/Box-geom combinations, world
hand-on-box force sign, rotated-Box frame, raw sums (3.82e-6 N error), isolated
support/ground filtering, histories, reset/parallel isolation, graph parity and
intentional capacity rejection. P2 run_03 establishes contemporaneous quaternion
aggregation, interval/action/next-state alignment, terminal copies, Replay windows
and reset isolation (raw sum error 1.76e-6 N). Reuse these results with their original
scope, source hashes and failures. New tests cover new physical regimes and newly
calibrated binary definitions, not a rerun of their acceptance suites.

## Implementation adjustment A1: bounded diagnostic provenance

The exact manifest is SHA256 1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1.
Only its declared historical V3 planning text (SHA256 168465ae0a2e17153902a337963574942551909680d0d8dd36f7ffb22102b05c)
is absent; all physical source files remain verifiable. Strict production loading
must continue to reject the missing source. The P6 harness will verify the exact
manifest, structural schema, all available source hashes and precisely this one
allowed missing planning filename/hash, then scope a loader adapter around the
unchanged P1 factory. This is an implementation-only diagnostic construction
adapter, permitted by the user's bounded P6 authorization. It does not set
training_ready, rewrite sources or approve learning. Any different missing or
changed source fails closed. Report strict-loader failure separately. No physical
parameter, sensor, control or solver setting changes in this adapter.

## P6-A traceability and concrete procedure

| Requirement | Existing evidence / actual implementation | New work and exact utilities | Setup, measurements and predeclared acceptance | Risks |
|---|---|---|---|---|
| A.1 native geoms, masks, transforms, pair identification | P0 DEPENDENCY_CONTACT_AUDIT; P1 scene.validate_scene, contact.HAND_BODIES, calibrate_contacts.Placements | E/run_p6.py compiled inventory and geometry approach scan | Original four pitch/yaw capsules, seven Box geoms, enabled masks, parent/world transforms, raw pair IDs and EEF marker distance; approach from +20 mm to shallow overlap, left/right and independent bilateral initial states; separate detection and dynamics | EEF marker is internal; FK scan is not lifting |
| A.1 separated/approach/first contact/displacement/release | P1 all-pair coverage reused | E/run_p6.py geometry scan and independent IK fixture from native keyframe, never reference260 | Save every initialized state and contact.dist/pos/frame/geom IDs; separated states have no hand pair, intended contacts appear near surface, opposite hand remains separated in unilateral tests | A fixture may fail; preserve and correct only fixture |
| A.2 force calibration and policy occupancy | P1 calibrated sensors, P2 aggregate_interval and IntervalCapture | E/run_p6.py calibration, E/analyze_p6.py; contact_validation.py | Per-side measured null and shallow loaded force distributions; thresholds selected independently by half the gap between max null and 5th-percentile positive load, require separation and held-out checks; no copied .001 N threshold | Net vector cancellation and low-load contact are explicitly limited |
| A.2 sign/frame/mean/peak/impulse/filtering/history/reset/overflow | P1 run_05 and P2 run_03 reused; actual source hashes pinned | New runs call existing capture/aggregation and capacity guard, add contact geometry and warnings before guard throws | Four .005 s samples; mean N, peak N, impulse .005 sum N s; independent raw contacts vs sensors; verify new fixture histories, support-only force separation and no hidden solver flags | Overflow is failure, never silently ignored |
| A.2 occupancy | P1 transient [1,0,0,0], [0,0,0,1], [1,1,0,0] samples reusable | Reanalyze raw transient evidence at calibrated thresholds; strict occupancy > rho | Proposed rho=.25 (at least 2/4 force-bearing solves) separates single-solve impulses from persistent force. Require measured two/four-solve detection; save sensitivities rho=0,.25,.5,.75. No tuning against reference agreement | Definition excludes genuine short impacts; retain raw force/occupancy |
| A.3 mass/root/COM/inertia/geoms/free joint | M6.5 nominal 2 kg, COM(0,0,-.2), inertia(.0334,.02167,.04167), build_audit_scene/_box_spec | E/run_p6.py physical inventory and force/torque experiment; E/analyze_p6.py mass-distribution proposal | Check compiled equality, positive inertia/triangle, root at bottom plate, COM vs envelope, known COM force F/m and torque I^-1 tau, gravity -9.81 at COM; analytic one-step error <=1e-5 CPU /2e-4 GPU | Mathematical consistency is not plausible bread-box mass distribution |
| A.3 reconcile COM concern | M6.5 explicitly retained for source parity only | Report uniform-density seven-box mass/COM/full inertia as a proposal, not applied | Compute volume-weighted centroid and parallel-axis inertia; note overlapping parts and missing measured material distribution; retain original scene | Asset revision requires separately approved versioned profile/regressions |
| A.4 support/static/release/origins/reset | P1 fixed-support cache offset fix, M6.5 terminal settle | E/run_p6.py independent Box at original support0/support1, floor, translated world fixtures | Gravity settle and recontact for 3 s; tail .5 s support vertical force within 5% of 19.62 N, speeds <.02 m/s and .1 rad/s, orientation drift <.05 rad, displacement <.02 m for static settle; same support origins across worlds | Isolated support capacity is not achieved delivery |
| A.4 historical solver failures | P2 full_suite_failure_01 LS1024; P3 ITERATIONS512 | E/run_p6.py original floor and elevated diagnostic cases, capture flags before failure; CPU replay comparison where useful | Same dt .005, decimation4, iterations10/ls20; report max iteration, bit values, contact counts, penetration and full state on failure; no solver retuning | Nonreproduction cannot resolve an intermittent failure |
| A.5 known translation/rotation/lift/gravity/release/recontact | P1 fixed-robot impulse and M6.5 support-only evidence | E/run_p6.py independent native hand contact dynamics and explicitly forced Box lift from original support | Apply documented world COM wrench, then zero all applied forces; no post-reset Box teleport. Require lift >.03 m, support loss/recontact, tail stability; report force/impulse, pose/twist and contact slip. Separate hand release from externally assisted lift | External lift is not natural hand carrying; no successful grasp assumed |
| A.6 penetration | P0 sensor netforce dist unusable; M6.5 large overlap failures | contact_validation.py per-category raw-contact summary; E/run_p6.py raw records | Save dist,pos,full frame,geom1/2 at each solve, count,min signed,max/mean overlap, duration/frequency. Diagnostic bands .1 mm/1 mm/5 mm; >1 mm persisting >=.1 s is flagged significant, not erased. Report solver limits independently | Contact manifold records cannot prove absence of missed/tunnelled pairs |

Independent fixtures use the native keyframe/FK geometry, support transforms and
known forces, not HDMI contact agreement. A separately labeled reference260 floor
probe may reproduce a historical failure but cannot be used as independent success.
Bilateral placement may use bounded numerical IK in a disposable diagnostic state;
no asset/reference modification. Dynamics logs distinguish fixed robot fixtures,
free actuated robot, isolated free Box and known external wrench experiments.

## P6-B traceability and concrete procedure

| Requirement | Existing evidence / implementation | New work / files | Acceptance and limitations |
|---|---|---|---|
| B.1 original geometry/timing/reachability | P4 geometry_01, audit_proximity_reward; rewards.ContactGeometry; ReferenceClip/ReferenceResetCache | E/run_p6.py all840 native collision geometry audit, E/analyze_p6.py geometry summary | Original accepted NPZ hash bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b; preserve 290 ON260–549/550 OFF; EEF/target per-frame trajectory and native pair distances; reuse P4 FK checks. No unconstrained IK claimed as whole-motion force feasibility |
| B.2 bounded original-PD physical replay | P2 recorder/Replay; M6.5 original PD and actions | E/run_p6.py bounded native GPU replay episodes; E/analyze_p6.py | Reset at specified frames (0,250,400,530,550,800), original reference joint commands, free robot and Box; no root aid or Box wrench. 40–60 policy steps per episode, never beyond838→839. Stop failed world/episode with raw failure snapshot. No learned-policy or full-task evaluation |
| B.2 exact alignment/validity | P2 transition s_t,a_t→s_t+1 with c[t],c[t+1]; P5 next-state reference | P2 install_recording reused where valid; capture owns four solve samples | Row stores start frame t, end frame t+1; force interval compared to original c[t+1] consistently. State/geometry at end state; also store solve states. No shift/smoothing. Same mask all metrics; incomplete/overflow rows explicitly invalid, failed physical tracking remains visible |
| B.3 independent actual contact | A calibrated tau_L/R and rho_L/R | contact_validation.py binary + confusion/phase/duration metrics | Strict norm>tau, occupancy>rho; independent L/R, OR, AND; TN/FP/FN/TP, accuracy/precision/recall/F1/balanced accuracy; undefined denominators null, ON/OFF durations, occupancy, onset/loss across adjacent rows only |
| B.3 aggregate interpretation | P4 CONTACT_TARGET_AUDIT cites manifest bilateral_force_required=true | E/analyze_p6.py reports both aggregates regardless of score | AND task-required only ON260–549; OR mandatory for unintended OFF contact. No invented per-hand reference label |
| B.4 classification | P4 internal marker / M6.5 overlap and support10mm gap / A physical evidence | REFERENCE_COMPATIBILITY_REPORT.md and E summaries | Classify hand/Box reconstruction, coordinate mismatch, surface mismatch, timing, control, physical/solver, insufficient evidence; distinguish evidence from inference. No automatic retarget or asset correction |

No arbitrary target agreement percentage is invented as a physical validity gate.
Reference compatibility decision follows measured geometry and contact failures;
software/measurement completion and task feasibility have separate statuses.

## Validation and stop criteria

Focused tests cover strict threshold/occupancy inequalities, identical validity
masks, aligned indices, undefined metrics, contact episodes/durations, pair category
and penetration summaries. New physical fixture guards cover saved initial states,
no post-reset Box assignment in dynamic trials, correct 4-sample timing, finite
state and fail-closed solver diagnostics. Do not rerun old passing suites without
an affected interface. Hash preservation proves unchanged original training paths;
new evaluation code is never imported into reward/WM/Actor.

Failures retain original output and partial measurements. Classify implementation,
fixture, physical-model, solver or reference issue; minimal implementation/fixture
repairs may proceed with failed-test and affected-regression reruns. Research-level
asset/solver/reference changes require approval; finish all independent work first.
Final reports distinguish PASS/PARTIAL/FAIL/NOT TESTED, give separate
PHYSICAL_VALIDITY/REFERENCE_COMPATIBILITY, exact commands/files/evidence and P7
prerequisite disposition. Review full diff and every preservation hash; no commit.

## Executed reconciliation

All work packages were executed without an intermediate approval gate. Final
machine evidence: `calibration_02`, `geometry_02`, `box_dynamics_01`, `dynamics_01`,
`isolation_01`, `filters_02`, `replay_01`, `analysis_02`, `verification_02.json`, and
`tests_final.log` in E. The earlier calibration/geometry/IK/filter attempts and
physical failures remain intact. Only disposable fixtures and evaluation tooling
were corrected; no production scene/source/solver parameter was changed.

Actual new utility files: E/scripts/common.py, run_p6.py, box_dynamics.py,
analyze_p6.py, physical_analysis.py, verify_evidence.py, export_commands.py and
ik_fixture_probe.py; source/test paths remain as specified. The latter probe is
preserved diagnostic development evidence, not an acceptance runner.

A2 (implementation/fixture): initial bilateral IK's success flag did not imply
small target residual or pitch-collider clearance. The corrected independent
fixture explicitly guards both, with residuals <1.4e-9 m and ≤0.5 mm overlap.
A3 (diagnostic isolation): original-support tests failed when a distant robot
landed. A paired remote-robot z=50 m test keeps original Box/support coordinates;
all three seconds complete. The z=3 m floor failure is retained, not replaced.
A4 (fixture): the first torso filtering load was 0.663 N, below the verification
criterion >1 N. Diagnostic compression 0.5→2 mm produces 2.709 N. Sensors remain
zero. `verification_01` FAIL and corrected `verification_02` PASS are distinct.

Final calibrated τL/R=0.0743401046875/0.0737510181777 N; strict ρL/R=.25.
The missing planning-source adaptation remains diagnostic-only. P1/P2 prior hash
differences are explained by already accepted P2 contact additions and P3 Replay
method extraction, not P6 changes. Exact force/interval arithmetic and reference
intent/state continuity were checked on the new captures.

Research-level proposals remain UNAPPLIED: Box mass-distribution correction and
any solver/asset/reference changes. Physical validity PARTIAL, reference
compatibility FAIL; original-floor reliability and M6.5 readiness remain blocked.
Natural hand-driven lifting/carrying is NOT TESTED as a successful task. No P7/P8,
training, commit or push. Reports and final preservation review complete this stage.
