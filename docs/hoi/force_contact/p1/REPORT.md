# P1 — physical scene and hand–box sensing

**PASS.** The actuated native G1–box–support scene, named joint state/reset adapter
and independent left/right contact sensors are implemented and physically calibrated.
The final calibration passes all ten groups. Repository tests: **69 passed,
17 skipped** because existing LAFAN1 NPZ/CSV inputs are absent. All P1 physical
and GPU checks ran. **No P2, training-mainline change, training or push occurred.**

## Scope, provenance and implementation

Read the complete 1,834-line master, including §§1.3, 1.3.4, 1.3.6 and 6; reused P0
closure `b5c604a` and its preceding `579d1a8` / `e895731` evidence. The user explicitly
started P1 and authorized commits. Initial branch/HEAD: `main` /
`f2cfac76e1977d3321199a57d961da517e855e2b`. Master working-copy SHA256:
`5b0c26e3c3d8c721e5e1f76ffe194f6141994a9eba4894ce7f43d7266e9be82f`.
The spec was written before source implementation. Its commit is `f08b52b`, authored
by `Ir1222 <Yiqian.Pan23@student.xjtlu.edu.cn>`. The earlier attempt without a configured
identity failed; no commit used another person's identity.

Pinned environment: mjlab 1.6.0 / `8bb2bc9e187797689ec82c78c35019afd2fd0376`,
MuJoCo/MuJoCo Warp 3.16.0, Warp 1.18.0, Torch 2.14.1+cu130; RTX 4090 D.
All ten dependency-source hashes audited in P0 still match. Only the newly uncovered
static-geom cache behavior required additional dependency inspection.

- `hoi/scene.py` composes native G1, the manifest's dynamic seven-geom box, two
  genuinely fixed supports and plane. Native actuator/action configuration is reused.
  It invokes the existing mjlab environment without commands, rewards or recorders.
- `hoi/state.py` resolves named free joints and robot scalar joints; it never treats
  global qpos/qvel concatenation as joint state. Resets use one reference frame for
  both bodies, with a separate in-memory object twist cache and selected-row clearing.
- `hoi/contact.py` configures exact pitch+yaw primaries, single box-body secondary,
  `secondary_policy="error"`, netforce, one slot and four history samples. It exposes
  raw simultaneous hand sums, same-sample Box rotation and explicit overflow checks.

Implementation adjustments are recorded in [IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md),
especially A1: Warp skips recomputing static collision-geom transforms. The new HOI
factory translates both body positions and cached geom positions per world. This
fix preserves fixed supports, graph pointers and source assets. No shared/dependency
patch or research-level change was needed. Construct the runtime through
`make_physical_env`; `make_physical_env_cfg` alone does not apply this runtime cache
initialization. Environment origins are fixed for the lifetime of this P1 scene.

## Contracts and physical evidence

State `[E,84] = [robot71,object13]`; original robot order unchanged. Object13 is
position3, wxyz4, root-origin world linear velocity3, body angular velocity3. Both
positions subtract environment origins. COM conversion uses the rotated offset and
`v_COM = v_root + omega_world × offset_world`. Reference-derived velocity is reset
initialization only; simulator velocities remain the physical measurement.

Per hand: force `[E,2,3]`, found `[E,2]`, history `[E,2,4,3]`. Two columns are pitch/yaw;
the hand-level helper sums only that simultaneous body axis. History is newest-first,
one sample after each 5 ms solve; four samples span one 20 ms action. Calibration
captures the solve-time Box orientation alongside each sample. Final forward adds
no fifth sample. Returned copies survive later native history-buffer replacement.
Forces are **hand on box**, world N; Box-local force is inverse Box rotation with no
extra negation. No temporal mean/impulse/occupancy, Replay or Force Head was added.

Final raw evidence: [run_05/calibration.json](../../../../outputs/hoi/contact_supervision/sensor_calibration/run_05/calibration.json).
Compact checks/hashes: [validation_summary.json](validation_summary.json).

| Required validation | Executed result |
|---|---|
| Native actions, audited geometry/inertia/supports | PASS; nq43/nv41/nu29; original actuator arrays and action map equal; manifest geometry tests |
| Named state, nonzero origins, root/COM frames | PASS; reverse entity order trap, MuJoCo body-velocity comparison, pose/twist round trip within `2e-6` |
| Synchronized reset | PASS at frames 0/300/500/839; moving-frame object twist nonzero; other worlds preserved |
| Both hands, both collision parents, all seven geoms | PASS, 28 placements × 4 worlds = 112 cases; minimum target-pair load `1.323077 N` |
| No / left-only / right-only / bilateral parallel worlds | PASS; expected independent activity pattern; inactive hand force zero |
| Hand-ground and box-support exclusions | PASS with actual raw excluded contacts and zero filtered hand-box force |
| Independent raw-contact sum vs sensor columns | Max error `3.8146973e-6 N` across 45 recorded samples |
| Fixed-robot left/right pushes and rotated box | PASS; positive known-direction force and COM response; maximum momentum error `1.3426319e-5 m/s` |
| Box-frame transform | Max inverse-rotation round-trip error `5.9604645e-8 N` |
| Substep1-only / substep4-only / contact ends before final | PASS; measured chronological patterns `[1,0,0,0]`, `[0,0,0,1]`, `[1,1,0,0]`; exact reversed-history equality |
| Reset immediately after contact | PASS; selected history zero, other worlds' state/history exactly preserved |
| CUDA graph / eager step and reset | PASS; maximum state difference `1.1920929e-6`, history difference `1.5258789e-5 N` |
| Capacity under normal calibration | All overflow flags zero; max total contacts50 / shared capacity1024; max constraints184 / per-world capacity1024; max matches3 / capacity64 |
| Deliberate capacity overflow | PASS; maxmatch1 fixture raised `CONTACT_MATCH` flag64 in all four worlds and the guard rejected it |
| Original robot-only path | PASS; real CUDA HDMI environment, state/action/context and original Replay smoke check |
| Full tests / lint / format | 69 passed, 17 data-dependent skips; lint/format PASS on all seven new Python files |

The fixed-robot sign fixture welds a disposable copy of the complete native robot
at its FK pose and removes gravity to isolate box momentum. Production remains
actuated with its original gains and gravity. Transient fixtures deliberately
reposition bodies between solves to control event timing; they are not policy
trajectories or claims about spontaneous transient dynamics.

## Failed attempts retained and resolved

| Attempt | Result and correction |
|---|---|
| Initial six-test run | One CPU test-fixture Float64/Float32 mismatch; corrected fixture dtype. Production CUDA test already passed. |
| run_01 | Nine groups failed on diagnostic `ContactType` import; installed enum is in `_src.types`. Overflow rejection passed. |
| run_02 | Eight groups passed. Coverage wrongly demanded force at every redundant contact point; fixed per-geom load criterion. Real support collision failed, exposing static cached geom positions. |
| run_03 | Nine groups passed after cache fix. Default yaw capsule carried the entire wrist-bottom load; changed only controlled native wrist pose to expose pitch collider. |
| run_04 | All normal physical groups passed. Exposed-wrist pose no longer overflowed the deliberately tiny match capacity; restored the original multi-contact pose for that negative test. |
| run_05 | All ten groups PASS; final acceptance evidence. |

All five JSONs and logs are retained. Historical P0/M6.5 FAIL records are untouched.
Final full test log lists each of the 17 skips: four analytic-head tests, six original
environment tests, one imagination test, four motion-conversion tests, one compact
library test and one WM overfit test. Their required LAFAN1 assets are absent. No
mandatory P1 test is counted as a skip or an unexecuted PASS.

## Training Framework Reuse Audit

**No shared source changed; no training functionality duplicated.** Reused native
G1/action configuration, mjlab step/reset/sensors, manifest/reference loaders,
original state constants and rotation math. Trainer, Replay, WM fitting, Actor BPTT,
Critic TD(lambda), optimizer, scheduler and `sm-train` defaults remain byte-identical.
The original robot-only HDMI smoke check passed. Full audit:
[TRAINING_FRAMEWORK_REUSE_AUDIT.md](TRAINING_FRAMEWORK_REUSE_AUDIT.md).

## Files, commands and Git review

All P1 paths are new relative to the initial checkout; only the new spec was updated
after its first commit. No previously existing source file changed.
Diff scope across the two commits: 26 new files — seven Python files (425 runtime,
575 calibration and 280 test lines), six stage documents/evidence files, and
13 raw calibration/test artifacts. Most patch lines are retained raw JSON samples.

| Group | Every file |
|---|---|
| Runtime | `src/supermimic/hoi/scene.py`, `state.py`, `contact.py` |
| Calibration | `scripts/hoi/calibrate_contacts.py` |
| Tests | `tests/hoi/test_joint_scene.py`, `test_contact_sensor.py`, `test_robot_only_regression.py` |
| P1 documentation/evidence | `docs/hoi/force_contact/p1/IMPLEMENTATION_SPEC.md`, `baseline.json`, `REPORT.md`, `TRAINING_FRAMEWORK_REUSE_AUDIT.md`, `VALIDATION.md`, `validation_summary.json` |
| Calibration raw JSONs | `outputs/hoi/contact_supervision/sensor_calibration/run_01/calibration.json`, `run_02/calibration.json`, `run_03/calibration.json`, `run_04/calibration.json`, `run_05/calibration.json` |
| Calibration logs in the same root | `run_01.log`, `run_02.log`, `run_03.log`, `run_04.log`, `run_05.log` |
| Test logs in the same root | `tests_02.log`, `tests_full_01.log`, `tests_full_02.log` |

[VALIDATION.md](VALIDATION.md) records exact substantive commands, diagnostic
corrections and review procedure. All source/test diffs were read in full, all raw
JSON structures/numbers were checked, and `git diff --check` passed. Preservation
verification matched all 126 baseline file hashes and ten P0 dependency hashes.
Part B still hashes to `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`;
M6.5 still hashes to `1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`.

Pre-implementation-commit HEAD is `f08b52b7a00f805e86cd4409922decf6b130a282` on `main`.
Only the P1 files above are included. User-owned changes remain excluded:
the master plan's one-line typo correction, `.vscode/` and untracked
`SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md`. The implementation commit and post-commit
Git status are reported to the user after creation; no push is authorized.

## Limitations and stop point

This PASS establishes sensing and scene/state plumbing for the tested fixtures.
M6.5 `physical_hand_contact=FAIL`, `scene_support_validity=FAIL`,
`training_ready=false` and `M6_5_NO_GO_M7` remain historical facts. Controlled sensing
does not establish full-motion task feasibility, physical release or training
readiness. Netforce cancellation can hide local loads; found has no built-in history.
The new scene requires fixed origins and the pinned dependency's static-geom behavior.
Capacity calibration covers the recorded cases, not arbitrary future policies.
The absent target-local checkpoint and P0's source/provenance limitations remain.

**P1 only is complete. Stop after the explicitly authorized commit and wait.
P2 and all later stages were not started.**
