# P6 closure implementation specification

Authority: current Master Plan §11 and the user's C1–C5 closure authorization,
following committed P6 b4d412479d8b7dd5a174e6f176a23a07c525e1f6. This is a
versioned diagnostic extension, not P7 or a change to learning semantics.
Original P6 reports/evidence, production assets, accepted NPZ and source hashes
remain preserved. No viewer, videos, training, automatic commit or push.

Evidence root E = outputs/hoi/contact_supervision/p6_closure. Sources and exact
candidate definitions live in E/scripts and E/candidates. Previous P1/P2 scene,
state, sensor, interval and reset implementations are reused, including the
previously documented pinned diagnostic provenance adapter. Missing historical
V3 provenance remains a production gate.

| Requirement | Implementation and controlled validation | Acceptance / unresolved risk |
|---|---|---|
| Phase 0 | E/baseline.json hashes tracked and protected ignored/external files; re-read current production implementations and P6 failures; pinned upstream source snapshots | No historical evidence changes; only new diagnostic candidates until selection is justified |
| C1.1 | Pin humanoid-amp-ppo commit and inspect SMTrack Box, generation, train/eval configuration separately from released DualArm Toaster/grasp configuration | Public source identifies configuration; do not assert inaccessible checkpoint metadata or transplant non-HDMI trajectory |
| C1.2–4 | candidate_models.py: original seven, prior overlapping-density proposal, exact disjoint-union mass model, HDMI-sized single solids, SMTrack dimensional/mass control, optional disjoint grasp pads | Explicit root mapping, mass, COM, full tensor, principal axes; positive inertia/triangle; overlap counted once; approximation labeled |
| C1.3 | geometry.py: all840 FK, all290 ON, capsule-to-cuboid signed distances, pair identities, intended targets, non-hand and thigh/knee/shin clearances, dimension sensitivity | Original global intent and trajectories unchanged; [−1,+2]mm hand band, >1mm non-hand/self overlap flagged; no score-only resizing |
| C1.5–6 | physical_compare.py: matched support gap/orientation, free drop, external lift/release/recontact, 4N translation, .02Nm torque; nominal mass-fixed comparisons plus explicit1.5kg control | Tail .5s speed<.02m/s, angular speed<.1rad/s, drift<.05rad, balance within5%; analytic first-step error<1e-5; no warnings. External force is not carrying |
| C2 | solver_diagnosis.py: source/kernel/bit trace, exact historical state/controls, complete scene/robot-floor/floor ablation/support-only/elevated; CPU shadow and GPU snapshot replay | Preserve every warning and before-failure state, qpos/qvel/qacc/ctrl/actuator force/contact/constraint/solver data. Isolate one variable; diagnose before correction. No arbitrary bulk retuning |
| C3 | trajectory_ik.py: reference-near arm IK with hard joint bounds, native collider distance/side constraints, self/non-hand collision penalties and coupled consecutive-frame first/second differences | Report all frames, no independent-frame continuity claim; bilateral band and non-hand overlap criteria plus actual joint velocities/accelerations. Any exported candidate has separate provenance and unchanged bool intent |
| C4 | integrated.py: selected model and solver variant consistently; original/candidate reference under identical native actuation; bounded contact/lift/carry/release attempts | No Box/root teleport after reset, no Box wrench for natural manipulation; force/occupancy, motion and unintended/lost contact reported even on failure. Failures prevent readiness |
| C5 | separate object construction, Box model, floor solver, reference, bilateral and P7 decisions; exact commands, files, hashes and affected tests | Software completion never implies physical success. No P7 starts regardless of diagnostic outcome |

The user explicitly authorizes candidate object/reference reconstruction and the
smallest evidence-backed solver correction in this task. This supersedes the prior
stage's requirement to pause before these experiments; it does not authorize an
unvalidated production asset replacement. Diagnostic changes remain isolated and
versioned. Any production correction requires demonstrated causal evidence and
affected regression tests. Candidate selection must precede integrated comparison.

Original contracts remain state84/canonical86/residual82, WM166/82, unchanged
Actor/Critic observation sizes and global reference intent. Force remains an
independent physical measurement. Calibration from P6 is initially reused with
new-model null/loaded checks; thresholds are never optimized for agreement.

Physical caveats to resolve: no measured material/mass specification, unknown
HDMI density, potential native capsule mismatch, original reference support gap,
robot-floor numerical convergence distinct from contact-buffer capacity, finite
bounded experiments cannot establish universal solver reliability.

## C1.7 and C2.1 additive audit

`scripts/contact_parameters.py` compiles the original scene, upstream SMTrack
factory and complete Toaster XML/assets. The Toaster's actual pinned mjlab1.4
CollisionCfg editor is applied separately from the raw XML; local mjlab1.6 cannot
be substituted because its required structural arguments differ. Compilation uses
the installed MuJoCo3.16 and is a current-engine reconstruction, not an assertion
that public checkpoints were compiled with this version. All upstream assets are
Git-blob verified. Pair probes retain compiled masks, priority, condim, friction,
solmix, solref/solimp, margin/gap; raw stored coefficients and active constraints
are reported separately. Full meshes are retained because the editor can enable
previously non-colliding geoms.

Ten candidate parameter profiles use the selected single-body geometry and2kg
inertia. One group at a time: friction, condim4, friction conditional on condim4,
solimp, solref, margin, gap, gap conditional on margin; the combined profile is
explicitly a follow-up without single-factor attribution. Four matched CPU cases
per profile:2s static floor support,0.1m drop,30N lateral pulse and4Nm yaw pulse
from0.5–0.7s. Same initial state/control/dt/solver. Contact stability, actual
wrenches, slip/rotation, penetration and solver counts are retained every step.
Current-version margin/gap behavior and the enabled refsafe clamp are verified
by compiled probes and installed source rather than inferred from older docs.

`scripts/contact_gpu.py` runs every profile on the same original complete floor
fixture for1s, both unmodified and precision-guard solver profiles, plus a0.5s
elevated native PD hold with null/left/right/bilateral initial contacts. No Box
wrench or post-reset teleport occurs. These trials diagnose material response;
none establishes natural carrying or measured material accuracy. They preserve
full before/after state, contact dim/friction/solver parameters, and all overflow
bits. Thresholds are not tuned per material to improve agreement.

C2.1 explicitly checks `overflow & OverflowType.LS_ITERATIONS`, records full
integer values and every active/unknown bit, and distinguishes the1024 flag from
the actual20-iteration limit. Main Newton limit10, line-search tolerance.01,
main tolerance1e-8, pyramidal cone and active friction rows are recorded. Proposed
corrections are accepted only with paired exact-state CPU comparison and retained
physical-behavior evidence, never from a reduced flag rate alone.
