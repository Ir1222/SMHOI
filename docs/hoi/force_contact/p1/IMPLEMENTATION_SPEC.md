# P1 implementation specification

Scope: physical scene, joint pose/twist reset and raw hand–box sensing only.
No P2 interval statistics/Replay, learned model, rewards, optimization, CLI or training
mainline changes. The user's current request accepts the P0 baseline for P1 and
explicitly authorizes commits; it does not authorize P2 or a push.

Baseline: `main`, `f2cfac76e1977d3321199a57d961da517e855e2b`; master SHA256
`5b0c26e3c3d8c721e5e1f76ffe194f6141994a9eba4894ce7f43d7266e9be82f` (1,834 lines).
P0 accepted evidence: `579d1a8`, `e895731`, `b5c604a`, especially its final
`REQUIREMENT_COVERAGE.md`, `REPORT.md`, `CLOSURE.md` and `FILE_CHANGE_MAP.md`.
All ten inspected dependency source hashes still match P0. Versions and protected
file hashes are in `baseline.json`. Pre-existing plan typo correction, `.vscode/`
and untracked V3 remain user-owned and excluded from P1 commits.

## Decisions and traceability

| Master obligation | P0 evidence and actual implementation | P1 decision / files / validation |
|---|---|---|
| §6.1 actuated native robot + seven-geometric box + two static supports | `DATASET_ASSET_AUDIT.md`, “Accepted Part B and physical profile”; `asset_manifest.py:load_manifest/_add_geom`; installed `get_g1_robot_cfg`, `SceneCfg.spec_fn` | New `hoi/scene.py`: native robot entity, dynamic box entity, fixed supports added directly to scene spec (avoids Entity's automatic mocap wrapping). Manifest is geometry/inertia/contact authority. Retain native action scale/offset and `NoisyJointPositionActionCfg`. Verify compiled parameters, 29 actuators, masks and support placement. |
| §6.2 named state and synchronized reset | `ARCHITECTURE_AUDIT.md`, “Loading, command and state acquisition”; `audit_checks.json`, `coordinate_closure_checks.json`; existing `StateIO` asserts 36/35 | New `hoi/state.py`: resolve free joints from named bodies, scalar robot joints in native order; reuse constants and rotation primitives. State `[E,84]=[robot71,object13]`; both positions environment-local, wxyz, root-origin world linear and body angular velocities. New in-memory reference twist cache uses central/one-sided pose differences and shortest quaternion log, never writes Part B. Test pose/twist round trip, nonzero reset twist, COM conversion, selective reset and nonzero origins. |
| §6.3–4 independent pair filters | Dependency audit “Match expansion” and `audit_checks.json` | New `hoi/contact.py`: `left_hand_box`, `right_hand_box`; exact pitch+yaw bodies; box secondary body; `error`, `netforce`, slot1, history4, found/force. Assert actual colliders and all seven secondary geoms. Sum only simultaneous primary force columns; no temporal statistics. Test both columns and every box geom physically. |
| §6.5 four 5 ms samples and capacities | Dependency audit “Four-substep timing”, “CUDA graph boundary”, native `ManagerBasedRlEnv.step` | Reuse native step loop unchanged through minimal physical-only environment config (empty rewards/commands/recorders/terminations). Start calibration capacities nconmax256, njmax1024, maxmatch64, subject to measured stress. `assert_contact_capacity` checks actual overflow bits, counts and finite force; inspect every calibration substep, including forced-overflow detection. Histories reacquired/cloned, newest-first. Graph pointers remain stable; per-world static body positions use supported model-field expansion/recapture. |
| §6.6 sign/frame | Dependency audit “Force sign and coordinates”; P0 probe only tested a sphere/plane | Force is hand-on-box world N without negation; inverse wxyz rotation gives Box-local N. Known-direction left/right pushes and rotated-box loads on actual native colliders; compare raw-contact sum and box COM response. |
| §6.7 transient/exclusion/reset/parallel cases | P0 history probe is synthetic, not physical acceptance | New `scripts/hoi/calibrate_contacts.py` and `tests/hoi/test_contact_sensor.py`: physical no/left/right/bilateral, hand-ground, box-support, each compound geom, rotated, substep1-only, substep4-only, ending-before-final, reset-after-contact, mixed parallel worlds and graph/eager parity. Controlled fixture pose assignments are explicitly calibration interventions, not policy rollout. |
| §1.3.6 / §6 acceptance original framework | `FILE_CHANGE_MAP.md`; existing `tracking_env_cfg`, actions, StateIO, robot StateSpec, rotations and all training components | No existing source file changes planned. `tests/hoi/test_joint_scene.py` plus original CPU suites and an original robot-only HDMI environment smoke check; missing LAFAN1 remains an explicit skip. Report Training Framework Reuse Audit and inspect complete diff. |

All choices above are **implementation-level adjustments**, preserving §6 semantics.
Static supports remain fixed at manifest pose plus each environment origin; their
positions are expanded once using installed `Simulation.expand_model_fields`, which
recaptures graphs. No dependency patch, asset change or research amendment is proposed.
Calibration may use a fixed-robot fixture derived from the identical scene: lock native
robot coordinates only in the fixture, preserve collision geometry and box inertia;
production remains fully actuated. Record any fixture intervention in evidence.

### A1: static support collision cache (discovered during P1 calibration)

Original obligation §6.1–2: both fixed support collision poses must equal manifest
poses plus environment origins. Initial `body_pos` expansion satisfied body pose
checks but **failed actual box-support collision** in run_02. Installed
`mujoco_warp/_src/smooth.py:198–201` skips `_geom_local_to_global` for static geoms;
`io.py:1999–2027` initializes their world poses only when allocating data. Thus a
forward does not update their collision transforms after changing body positions.

Implementation-level correction, presented before affected code: additionally
translate each support's cached `data.geom_xpos` in place once at scene construction.
Keep `geom_xmat` unchanged (origins translate only), keep supports truly fixed,
and test collision plus geom/body poses through selected reset and graph stepping.
No array replacement, dependency patch, asset mutation, frame change or research
amendment is needed. Runtime origin changes are not supported by this P1 scene.
P0's sensor evidence remains valid; this is previously uncovered static-scene
behavior, not a rewrite of its historical findings.

Run_01 also exposed that `ContactType` is exported only from the installed
`mujoco_warp._src.types`, unlike `OverflowType`; calibration uses that exact import.
Run_02's coverage assertion incorrectly required every redundant contact point to
carry force. The requirement is that each target geom has measured load in each
world: sum the target pair's local force magnitudes per world and require >`1e-3 N`.
Retain all individual contacts, including zero-load constraints, as raw evidence.

## Contracts and predeclared validation criteria

- Raw per-hand force `[E,2,3]`, raw history `[E,2,4,3]`, simultaneous hand sum
  `[E,2,3]` with hand order left/right; combined history `[E,2,4,3]` after summing
  primary bodies only. No interval mean, impulse, occupancy or transition label in P1.
- History index0 is substep4; history reset zeros selected rows. `found` describes
  only the current solved constraint sample, not geometric-contact history. No
  geometric-event reconstruction or cancellation-free contact guarantee is claimed.
- Calibration captures each force alongside the pre-integration box orientation
  (the pose used by that substep solve). P2 must select its own label convention.
- Float32 state/frame round trips: absolute tolerance `2e-6`; manifest CPU geometry
  comparison `1e-10`; quaternion synthetic derivative `1e-6` in float64.
  Raw-contact/sensor sum: `atol=2e-3 N, rtol=2e-4` for GPU accumulation.
  These are numerical comparison tolerances, not learned contact/reward thresholds.
- Inactive filtered force must be at most `1e-6 N`; controlled active contacts must
  exceed `1e-3 N` with matched constraint count >0. Known push requires positive
  force projection and positive COM acceleration/velocity response along that
  direction after gravity removal; force rotation round trip within `2e-5 N`
  absolute plus `2e-6` relative. Report magnitudes, not just booleans.
- Exactly four callbacks and `.020 s` per action, raw/history reversal equality;
  no fifth history sample after forward; retained copies unchanged across updates.
  All normal calibration overflow flags must be zero. Deliberately undersized
  capacity must raise visibly. No PASS on absent GPU or skipped physical tests.
- Outputs: `outputs/hoi/contact_supervision/sensor_calibration/` (fresh run dirs,
  JSON raw measurements/provenance and logs); final `p1/REPORT.md` includes exact
  commands, every changed file, outcomes, limitations and reuse audit.

## Risks retained

Historical M6.5 hand-contact/support FAILs and `training_ready=false` remain unchanged;
P1 sensing cannot establish full-task feasibility. Reference-derived object twist is
reset-only, separate from simulator supervision. Force cancellation and missing found
history remain limitations. Real fixture contact coverage, nonzero-origin support
placement, GPU capacities and graph parity are untested until execution; any mandatory
failure blocks P1 PASS. Reference implementation audit is applicable to rejecting
foreign assets/sensors/trainers; P4 reward/source mechanisms are read but not ported.
