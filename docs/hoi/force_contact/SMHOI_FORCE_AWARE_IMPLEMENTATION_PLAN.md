# SMHOI Part C — Force-aware World-Model-Based Humanoid–Object Interaction
## Single Master Implementation Plan for Codex

**Target repository:** https://github.com/Ir1222/SMHOI  
**Overall architecture plan:** `SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md`  
**Primary source implementation:** https://github.com/Ir1222/supermimic_carry  
**Original algorithm:** https://github.com/CuboiLeo/SuperMimic-Reproduced

**Research references**
- HDMI paper: https://arxiv.org/abs/2509.16757
- HDMI code: https://github.com/LeCAR-Lab/HDMI
- PhysHSI paper: https://arxiv.org/abs/2510.11072
- PhysHSI code: https://github.com/InternRobotics/PhysHSI
- SUGAR paper: https://arxiv.org/abs/2605.20373
- SUGAR code: https://github.com/tianshuwu/SUGAR
- Workhorse: https://arxiv.org/abs/2610.09117

---

# 1. How to read and execute this plan

This is the **single source of truth** for this implementation. Codex must read this document and execute the stages in order. Do not create another high-level plan before starting.

The labels P0–P8 are execution stages, not model components and not the same thing as V3's C0–C5 architecture sections.

| Stage | Plain-language meaning | Main question answered | Code changes? | Required result |
|---|---|---|---|---|
| **P0** | Inspect what already exists | What is actually in this checkout, and what must change? | No training-code changes; audit docs only | Verified audit and file-change map |
| **P1** | Make the physical HOI scene measurable | Can the simulator create the correct robot-box-support scene and measure each hand's real box contact? | Yes | Validated left/right contact sensors |
| **P2** | Turn substep measurements into training data | Can every action transition be paired with the correct force label and stored in Replay? | Yes | Correct force aggregation and HOI Replay |
| **P3** | Teach the World Model to predict object motion and hand forces | Can one joint WM predict robot, box and left/right force from state and action? | Yes | Trained/validated joint WM and Force Head |
| **P4** | Define the interaction incentives | Do rewards guide approach, real contact, sustained interaction and release without hard-coded phases? | Yes | Tested differentiable reward functions |
| **P5** | Put the new model and rewards inside imagination | Can Actor BPTT optimize actions through predicted box motion and predicted force? | Yes | Eager/compiled differentiable HOI rollout |
| **P6** | Check whether the learned force gradients are physically useful | Does the Force Head give the Actor a directionally meaningful optimization signal? | Validation code/experiments | Gradient-fidelity report and go/no-go |
| **P7** | Connect everything to the actual trainer | Can the original SuperMimic training loop train HOI without breaking robot-only tracking? | Yes | Checkpoint migration and successful short HOI training |
| **P8** | Test the complete task | Does the selected policy complete the full 840-frame pick-carry-place-release task physically? | Evaluation code/experiments | Full-task report |

## 1.1 The execution rule

Codex executes **one stage per user instruction**.

For every stage:

1. Read this whole master plan, then focus on the current stage.
2. Inspect the actual checkout and the previous accepted stage's report/commit.
3. Confirm prerequisites.
4. Implement only the current stage.
5. Run the tests and experiments required by that stage.
6. Inspect the complete Git diff.
7. Write the stage report in the path specified below.
8. Report `PASS`, `PARTIAL`, or `FAIL`.
9. Stop and wait for the user's explicit approval.

A stage finishing does **not** authorize the next stage. Do not automatically commit, push, or start another stage.

## 1.2 What the labels mean

- **P0 is an audit stage.** It finds and verifies existing code, data, assets, dependency behavior and gaps. It does not implement the training system.
- **P1–P5 build the physical sensing, data, model, reward and imagination pipeline.**
- **P6 validates whether the learned force model is a trustworthy optimization model, not merely a predictor.**
- **P7 migrates checkpoints and integrates the validated components into SuperMimic training.**
- **P8 evaluates the final selected checkpoint on the complete task.**

Do not confuse:
- `P0–P8`: Codex execution order in this document.
- `C0–C5`: broader architecture topics in `SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md`.
- `M0–M6.5`: previously completed physical asset audit.
- `Part B / B1–B3`: previously completed reference processing and acceptance.

## 1.3 P0 Audit Integration and Adaptive Implementation Protocol

This section extends the execution rules in Sections 1.1 and 14–16 for every stage P1–P8. The Master Plan remains the single source of truth for research objectives, contracts and acceptance criteria. The complete P0 audit evidence is a mandatory implementation constraint, not background reading. Do not repeat an implementation detail from the original plan when verified repository or dependency behavior contradicts it; resolve the discrepancy through the amendment procedure below.

The accepted P0 baseline comprises commits `579d1a8c661985c91926a365c7505488abe03bb8`, `e89573171c86be7f40c8d120dc813cee0274a152` and final closure `b5c604a0a3924787e553a265a994bc9a12e3cc65`. Preserve their reports, raw evidence and Git history. Earlier missing-plan, pre-commit and master-hash observations describe their historical snapshots; do not rewrite them to describe later state.

### 1.3.1 Required reading and current-state verification

Before starting each stage:

1. Read the complete Master Plan as required by Section 1.1, focusing on the current stage, cross-stage contracts and review gates.
2. Read [P0 REQUIREMENT_COVERAGE.md](p0/REQUIREMENT_COVERAGE.md), [REPORT.md](p0/REPORT.md), [CLOSURE.md](p0/CLOSURE.md) and [FILE_CHANGE_MAP.md](p0/FILE_CHANGE_MAP.md).
3. Read the relevant [architecture audit](p0/ARCHITECTURE_AUDIT.md), [dependency/contact audit](p0/DEPENDENCY_CONTACT_AUDIT.md), [dataset/asset audit](p0/DATASET_ASSET_AUDIT.md) and [reference implementation audit](p0/REFERENCE_IMPLEMENTATION_AUDIT.md), including the supporting JSON, probes and validation records needed for the stage. Record why any audit category is not applicable.
4. Read the previous accepted stage's report and inspect its accepted commit, including documented corrective commits. For P1, use the final P0 closure and the three commits above. Record acceptance and authorization separately; a commit does not authorize the next stage.
5. Inspect current branch, HEAD, user changes and the relevant actual code/dependency implementations. Compare their paths, symbols, versions or hashes with the evidence being reused. Recheck only changed or uncovered behavior; do not repeat completed audits without a reason. Record new observations separately and explain any superseded finding.

### 1.3.2 Stage implementation specification

Before any source-code implementation, create `docs/hoi/force_contact/p<N>/IMPLEMENTATION_SPEC.md` for the authorized stage. This is a concrete implementation specification derived from this Master Plan, not a replacement high-level plan or a separate Plan/Goal Mode pass. Keep it current as implementation decisions change.

Record the stage scope, exclusions, Master Plan revision, checkout/dependency baseline, previous accepted report/commit and applicable user approvals. Include a traceability matrix with one row per requirement or justified group:

| Required field | Content |
|---|---|
| Original requirement | Master Plan section, requirement and unchanged acceptance obligation |
| P0 finding | Exact report section and supporting evidence; distinguish observation, recommendation and untested hypothesis |
| Actual implementation | Existing file/symbol, dependency version and verified behavior |
| Revised decision | Concrete implementation choice, rationale, adjustment class and any amendment/approval reference |
| File changes | Files to create/modify, reusable components and minimal shared interfaces |
| Contracts | Tensor shapes/order, units, coordinate/reference-point conventions, timing, action/transition alignment and reset/validity semantics |
| Validation | Tests/experiments, numerical acceptance criteria, evidence paths and prerequisites |

List unresolved risks and pending decisions with the affected stage/functionality, required evidence and blocking condition. Do not invent a numerical threshold, freeze an untested candidate or mark a missing physical test PASS merely to complete the specification. Where the plan leaves a choice open, state the proposed criterion before running the test.

### 1.3.3 Adjustments, amendments and unresolved risks

Classify every departure or open issue explicitly:

- **Implementation-level adjustment:** file organization, installed API adaptation, named indexing or a narrow interface extension that preserves approved research semantics and acceptance criteria. Document and present the rationale before affected code; it may proceed within an already authorized stage without an additional approval cycle. An option expressly allowed by the Master Plan must still be selected and recorded explicitly.
- **Research-level design change:** a change to the approved model/learning architecture, state or force meaning, required frame/timing contract, supervision, reward semantics, optimization path, assets/reference assumptions or acceptance gates. Propose it and obtain explicit user approval before implementing affected functionality. Do not disguise such a change as an API fix.
- **Unresolved physical or experimental risk:** retain the evidence, uncertainty, consequence, resolution test and blocking condition in the specification and stage report. Historical failures remain failures; a P0 audit PASS or a P1 sensing PASS does not establish task success or training readiness.

For a conflict between the Master Plan and verified P0 evidence, first present a precise amendment: identify the original clause, cite the observed code/probe evidence, give the proposed replacement wording or implementation interpretation, explain why it is necessary, and state contract, compatibility, test and risk impacts. Classify it using the rules above and record approval status before affected implementation. If the impact on research semantics is uncertain, leave that functionality pending clarification; independently authorized work may continue. Incorporate accepted amendments into this Master Plan with provenance, without changing historical P0 evidence or committing automatically.

### 1.3.4 Mandatory P0 findings mapped to P1–P8

The following is a minimum mapping, not a replacement for the full audits or stage acceptance criteria. Audit names refer to the linked reports above; exact code paths and evidence belong in each stage specification.

| Stage | P0 constraint and implementation consequence | Required validation or visible risk |
|---|---|---|
| P1 | Architecture/dependency/assets: existing StateIO asserts robot-only `nq=36,nv=35`; the reference viewer is display-only. Use a physical scene and named robot/object state adapter. Each native hand needs pitch and yaw collision parents; one box body covers all seven geoms. Configure the audited mjlab 1.6.0 API with independent filtered sensors, `secondary_policy="error"`, one netforce slot and history length four; each hand has force `[E,2,3]`, history `[E,2,4,3]`. | Calibrate actual colliders, left/right isolation, all seven geoms, force sign/frame, four substeps, transients, overflow, joint pose/twist reset and parallel worlds. Preserve M6.5 assets/supports and historical contact/support FAILs. |
| P2 | Dependency/architecture: four newest-first samples span `.005 × 4 = .020` s; final forward overwrites instantaneous data, history buffers are replaced, and reset clears selected rows. Capture terminal history before reset; reacquire/copy interval data. No built-in found history exists. Replay's successor-row done lookup yielded `[1,1]` instead of `[0,1]`. | Test executed-action/force/terminal alignment, validity and episode/clip boundaries. Distinguish mean N, impulse N·s and occupancy. Record the permitted Box-orientation sampling convention; shift diagnostic torques to a common point before summing. |
| P3 | Architecture/dataset: preserve robot71/action29/xi33; target joint state84, canonical86, physical residual82 and auxiliary force6. Integrate only physical residuals. Root-origin/world-linear versus body-angular velocity, wxyz and right-multiplied rotation increments need explicit adapters/tests. Support-relative context is a time-varying candidate, separate from xi33; WM named checkpoint migration begins here. | Validate one/eight-step joint/force predictions, simulator supervision, frame/integration contracts and migration. Run the prescribed support-context identifiability/ablation before freezing its encoding. Reference-derived velocity stays in a separate cache and never replaces simulator targets. |
| P4 | Reference/dataset audits: HDMI's outside-intent baseline, PhysHSI's unfiltered body forces/stage gates and SUGAR's bilateral boolean/three-sample maximum differ from this plan. Part B provides global intent, not per-hand truth or measured force. | Adapt reusable geometry/math to independent smooth per-hand approach/contact/persistence/release rewards. Keep intent, actual contact and predicted confidence distinct; test saturation, gradients and support-aware release without rewriting Part B. |
| P5 | Architecture: current imagination clamps beyond clip end and can repeat final-frame rewards. Dynamic goals must use the current imagined object. Minimal default-preserving rollout/compiled/objective/critic hooks may be needed here, before P7. | Test final-transition/validity/bootstrap masks, eager/compiled parity and action-to-object/force gradients while retaining original BPTT and TD(lambda). Force-based Actor optimization remains gated by P6. |
| P6 | P0's sensor and coordinate probes establish conventions, not learned-force gradient fidelity. Low force prediction error is not a physical gradient guarantee. | Use validated P1/P2 measurements and repeatable state/action perturbations across prescribed contact regimes and multiple epsilon values; report directional/magnitude evidence and explicit go/no-go. Keep contact discontinuities and failed regimes visible. |
| P7 | Architecture/checkpoint audit: Trainer construction hard-codes components; retain `Trainer.run()` with narrow hooks. Action/xi input columns move; migration requires named maps and original scales/history order. Existing checkpoints omit Replay/RNG state. The target-local checkpoint is missing and the external candidate has qualified provenance. | Test actor parity, robot-only regression, initialization versus resume and metadata/buffer refill; do not claim exact stochastic resume or silently substitute a checkpoint. Register `smHOI-track`, preserve `sm-train` defaults and run only explicitly authorized training. |
| P8 | Dataset/assets: accepted Part B has 840 samples at 50 Hz. Historical physical contact/support failures and `training_ready=false` were not resolved by P0. | Evaluate the explicitly selected checkpoint from frame 0 through the full task, including release and documented post-release stability. Report every failed episode and provenance; reference replay, resets or a terminal-only settle do not establish policy success. |

Workhorse's approved paper/available-repository fallback remains a source limitation, not evidence of verified Workhorse code or force supervision. External algorithms, thresholds, assets or deployment augmentation are not automatically approved for adoption.

### 1.3.5 Preserved objectives and execution sequence

Preserve the approved joint robot-object World Model, auxiliary Force Prediction Head supervised by actual MuJoCo forces, soft approach/contact/persistence/release rewards, differentiable imagination and Actor BPTT, and original SuperMimic training mainline. Keep the original Critic/TD(lambda), accepted Part B reference, source datasets, checkpoints, M6.5 evidence and robot-only defaults protected under Sections 2 and 15.

For each separately authorized stage: **read and verify evidence → write IMPLEMENTATION_SPEC.md → present/classify amendments and resolve required approvals → implement only the authorized scope → run stage tests → reconcile results with the specification → inspect the complete diff → report PASS/PARTIAL/FAIL → stop for review**. Update the specification/report when new evidence changes a decision. An unresolved mandatory acceptance failure prevents PASS and progression. Commit only after explicit approval under Section 15.2; never automatically push or start the next stage.

Adding this protocol is a documentation-only change. It does not start P1, create a P1 implementation specification, authorize source edits or approve any later experiment.

---

# 2. Project objective and non-negotiable constraints

## 2.1 Objective

Implement **Force-aware, World-Model-Based Humanoid–Object Interaction Tracking** in the existing SuperMimic system.

The target learning loop is:


MuJoCo physical interaction
    ↓
Pair-filtered left/right hand–box contact measurement
    ↓
Four-substep force aggregation per control interval
    ↓
Joint robot–object–force Replay supervision
    ↓
Joint Robot–Object World Model + Force Prediction Head
    ↓
Differentiable soft interaction reward
    ↓
Joint robot–object imagination
    ↓
Actor BPTT + original Critic/TD(lambda)

The model must learn how humanoid actions produce contact forces and how those forces influence object dynamics.

This is not:
- a contact bonus added to ordinary PPO;
- an AMP-only solution;
- a separate HOI trainer copied from SuperMimic;
- an object pose predictor disconnected from robot dynamics;
- a force classifier that only predicts whether contact should happen.

## 2.2 Non-negotiable constraints
1. Preserve original sm-train behavior and defaults.
2. Add/maintain a separate smHOI-track entry.
3. Reuse the original Trainer.run() and SuperMimic optimization structure.
4. Preserve the original robot-only tracking ability and verified checkpoint compatibility.
5. Preserve the accepted Part B reference byte-for-byte.
6. Preserve the M6.5 physical asset profile and historical audit results.
7. Keep HOI-specific logic primarily under src/supermimic/hoi/.
8. Do not rewrite the original Replay, WM, imagination or Trainer wholesale.
9. Do not introduce six hard-coded one-hot phases: approach, grasp, lift, carry, place, release.
10. Use actual MuJoCo measurements for force supervision; use predicted force inside imagination.
11. Never use reference object pose as the current imagined object state.
12. Do not enable force-based Actor optimization before Force Head validation and gradient-fidelity checks.

# 3. Existing repositories, datasets, assets and checkpoints to inspect
These locations are provided so Codex can audit the real artifacts instead of guessing or regenerating them. They are inputs to inspection, not permission to modify them.
## 3.1 Repositories and checkouts
Role	Location	Instruction
Target repository	https://github.com/Ir1222/SMHOI	Actual current checkout is authoritative
Overall implementation plan	SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md	Read from the target checkout
Source HOI implementation	https://github.com/Ir1222/supermimic_carry	Inspect original HOI data conversion, assets, manifests and code
Ubuntu source checkout	~/pyx/humanoid-amp-ppo	Inspect actual HEAD/status; do not overwrite
Original SuperMimic	https://github.com/CuboiLeo/SuperMimic-Reproduced	Algorithm and training-mainline reference
Official HDMI	https://github.com/LeCAR-Lab/HDMI	Original motion, physical assets and interaction implementation
PhysHSI	https://github.com/InternRobotics/PhysHSI	Contact consistency and approach shaping reference
SUGAR	https://github.com/tianshuwu/SUGAR	Contact labels and object-aware reward reference

Do not assume these checkouts share the same commit, environment, dependencies or generated files. Record their actual HEAD and working-tree state if accessible.
## 3.2 Original HDMI motion
Recorded source:

data/motion/data_for_sim/carry_and_place_bread_box-0829/motion.npz

Repository: official HDMI checkout.

Motion identifier: carry_and_place_bread_box-0829.

Known properties:
- 840 frames.
- 50 Hz.
- G1 29-DoF robot.
- Bread-box pick/carry/place motion.
This is the original HDMI reference motion, not a PD-controller rollout. Earlier PD replay was a physical feasibility check, not the source dataset.

Codex must locate the actual HDMI checkout and verify this path. Do not invent an absolute path.
## 3.3 Existing datasets under /home/han/pyx
Inspect these exact paths:
Dataset	Exact path	Role
Robot-only HDMI reference	/home/han/pyx/hdmi_bread_box_robot_only_50hz_supermimic_pyx.npz	Robot-only reference used for SuperMimic tracking compatibility
Converted HOI reference	/home/han/pyx/hdmi_bread_box_hoi_v1.npz	Existing HDMI-derived HOI reference

Treat both as read-only. Inspect existence, size, SHA256, keys, shapes, dtypes and FPS. Do not overwrite, smooth, retarget, normalize or regenerate them in place.

If unavailable, report MISSING or INACCESSIBLE; do not silently substitute another similarly named file.
## 3.4 Existing files in supermimic_carry
Inspect these paths relative to the actual source checkout:
Relative path	Role
dataset/tracking/dual_arm1_box.npz	Original SuperMimic Lift robot/object tracking dataset
dataset/hoi/generated/hdmi_bread_box_pcp_v2.npz	Converted HDMI Pick–Carry–Place intermediate
dataset/hoi/generated/hdmi_bread_box_hoi_v1.npz	Derived HOI training reference
dataset/hoi/manifests/hdmi_bread_box_hoi_v1.json	HOI reference manifest
dataset/hoi/manifests/hdmi_bread_box_pcp_v2.json	PCP conversion manifest
dataset/hoi/manifests/hdmi_usd_contact_assets.json	Contact asset manifest
dataset/hoi/manifests/hdmi_usd_support_assets.json	Support asset manifest

Some generated files may be gitignored. Check the filesystem and manifests; absence from git ls-files does not imply absence from disk. Do not regenerate them just because they are ignored or untracked.
## 3.5 HDMI assets and code
Within the actual HDMI checkout, inspect:
- active_adaptation/assets_mjcf/objects/bread_box/bread_box.xml
- cfg/task/G1/hdmi/carry_and_place_bread_box.yaml
- active_adaptation/envs/mdp/commands/hdmi/command.py
- active_adaptation/envs/mdp/commands/hdmi/rewards.py
These are reference implementations. The target SMHOI scene must use the accepted M6.5 experimental asset profile, not silently replace it with an unverified source asset.
## 3.6 Accepted Part B artifact
In the target SMHOI repository:
Artifact	Path
Accepted minimal HOI reference	outputs/hoi/asset_audit/part_b/reference/reference.npz
Part B acceptance report	outputs/hoi/asset_audit/part_b/DATA_ACCEPTANCE.md
M0–M6.5 and Part B audit root	outputs/hoi/asset_audit/

Expected Part B reference SHA256:

bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b

Verify against the committed manifest and actual file before asserting. The committed manifest is authoritative if a transcription discrepancy exists.

The accepted reference has 12 fields:

Robot:
- joint_pos [840,29]
- joint_vel [840,29]
- body_pos_w [840,30,3]
- body_quat_w [840,30,4]
- body_lin_vel_w [840,30,3]
- body_ang_vel_w [840,30,3]
- joint_names [29]
- body_names [30]
- fps [1]
Object:
- object_pos_w [840,3]
- object_quat_w [840,4]
Interaction:

- object_contact [840]

The reference does not contain object velocity or measured force. Do not fabricate either. Derive reference object velocity in a separate cache if needed for tracking. Simulator state supplies velocity for WM supervision.
## 3.7 M6.5 physical assets and checkpoint
Inspect the actual committed M6.5 manifests under outputs/hoi/asset_audit/; discover the exact filename rather than guessing.

The accepted experimental profile records:
- seven compound box collision geometries;
- box mass, COM and inertia;
- G1 hand/wrist collision geometry;
- contact targets and EEF offsets;
- two static support geometries;
- contact alignment and penetration diagnostics;
- known physical discrepancies.
Preserve historical FAIL records. Do not claim PhysX equivalence.

Candidate robot-only checkpoint:

logs/hdmi_robot_only/ckpt_0012000.pt

Use only if it exists and its provenance/hash/architecture are verified. Never overwrite it.
## 3.8 Mandatory dataset inventory during P0
For every path above, report:
- EXISTS, MISSING, or INACCESSIBLE;
- file size and SHA256;
- NPZ keys, shapes, dtypes and FPS;
- JSON manifest provenance and hashes;
- which entry point currently loads it;
- whether it is original source, derived intermediate, accepted reference, asset manifest or checkpoint.
Do not redo M0–M6.5, Part B processing, reference visualization or dataset conversion.

# 4. Architecture contracts to preserve across all stages
These are candidate baseline dimensions. P0 must verify them against the actual code. If the actual implementation differs, report the discrepancy before changing the contract.
## 4.1 State and dynamics
Quantity	Intended dimension	Meaning
Robot physical state	71	Original SuperMimic state
Object physical state	13	position 3 + quaternion 4 + linear velocity 3 + angular velocity 3
Joint physical state	84	Robot + object
Robot canonical	71	Preserve original semantics
Object canonical	15	relative position 3 + relative orientation 6D + relative linear velocity 3 + angular velocity 3
Joint canonical	86	71 + 15
Action	29	G1 action
Existing domain context	33	Original context semantics
Robot residual output	70	Original robot dynamics
Object residual output	12	New object integration residual
Physical dynamics output	82	70 + 12
Force output	6	left 3D + right 3D force
Combined heads	88	82 physical + 6 auxiliary force

Force is not part of physical state and is never integrated.

Support-relative context is a separate, time-varying physical context candidate. Do not silently append it to the existing 33-D episode-constant randomization vector. P0/P3 must determine its representation and use an ablation/identifiability test before freezing it.
## 4.2 Time
Known current environment settings:
- physics timestep: 0.005 s (200 Hz);
- decimation: 4;
- control timestep: 0.020 s (50 Hz);
- one action spans four physics substeps.
Verify these settings in the actual checkout.

The first Force Head predicts average force over one 20 ms interval, not a 200 Hz force sequence.
## 4.3 Coordinate conventions
Document and test:
- robot root position frame;
- robot root linear and angular velocity frames;
- object free-joint root frame;
- object COM versus root-origin velocity;
- quaternion multiplication order;
- Box-local force convention;
- force-on-hand versus force-on-box sign.
Do not assume MuJoCo free-joint qvel is automatically the same as the root-origin velocity used by the canonical state. Verify it.
## 4.4 Reference velocity
Reference object velocity is derived from adjacent reference poses, not simulator supervision.

Linear velocity may use central differences for interior frames and one-sided differences at endpoints.

Angular velocity must use quaternion relative rotation/log map, not component-wise quaternion differences. Verify the angular-velocity frame and sign convention.

Do not write derived velocity back into the accepted NPZ.
## 4.5 Three contact signals
Always keep separate:
1. contact_intent_ref: global desired-contact intent from Part B.
2. contact_actual_sim: actual left/right hand-box contact/force-bearing contact measured in MuJoCo.
3. contact_pred_imagination: differentiable confidence inferred from predicted force and geometry.
Part B provides one global contact label, not separate left/right labels. Do not invent per-hand reference labels.

# 5. P0 — Repository, Dataset and Dependency Audit
What P0 means
P0 does not implement the HOI system. It answers: what code, data, assets, dependencies and checkpoints actually exist, and what exactly must be changed?

This is the first stage because Codex must not design against assumed paths or APIs.
Tasks
P0.1 Repository safety and state
Inspect:
- AGENTS.md
- SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md
- current branch, HEAD, recent commits and git status;
- all existing uncommitted/untracked user changes;
- running training/evaluation processes if safely observable;
- existing checkpoints and relevant experiment outputs.
Do not stop or kill processes. Do not reset, clean or overwrite anything.
P0.2 Dataset and asset inventory
Audit every exact path in Section 3. Record existence, hashes, schema and provenance.

Verify Part B against its committed manifest and acceptance report.

Confirm which dataset is actually loaded by each current entry point.
P0.3 Existing code architecture
Read the actual implementations of:
src/supermimic/training/loop.py

src/supermimic/envs/state_io.py
src/supermimic/envs/tracking_env_cfg.py
src/supermimic/envs/mdp/commands.py
src/supermimic/envs/mdp/observations.py
src/supermimic/envs/mdp/rewards.py
src/supermimic/envs/mdp/motion_library.py

src/supermimic/data/recorder.py
src/supermimic/data/replay_buffer.py

src/supermimic/world_model/state_spec.py
src/supermimic/world_model/dynamics.py
src/supermimic/world_model/heads.py
src/supermimic/world_model/world_model.py
src/supermimic/world_model/wm_trainer.py

src/supermimic/imagination/rollout.py
src/supermimic/imagination/compiled.py
src/supermimic/imagination/objective.py
src/supermimic/imagination/policy_trainer.py

src/supermimic/policy/nets.py
src/supermimic/hoi/


Trace real tensor dimensions, state integration, recorder timing, replay sampling, BPTT, compiled execution and checkpoint mapping.
P0.4 Dependency and ContactSensor audit
Inspect actual installed/locked mjlab and MuJoCo Warp versions.

A previously inspected mjlab commit was:

8bb2bc9e187797689ec82c78c35019afd2fd0376

Verify whether this is the actual dependency.

Inspect:
- mjlab/sensor/contact_sensor.py
- sensor update and history implementation;
- mjlab/scene.py;
- environment stepping order;
- recorder callback order;
- CUDA graph capture;
- reduce="netforce" semantics;


- `secondary_policy` and ambiguity handling;
- contact capacity and overflow behavior;
- sensor history reset semantics;
- force and torque coordinate frames;
- whether instantaneous sensor data is overwritten by the post-step forward pass.

### P0.5 Reference implementation audit

Inspect the actual source implementations, not only paper descriptions.

HDMI:
- `active_adaptation/envs/mdp/commands/hdmi/command.py`
- `active_adaptation/envs/mdp/commands/hdmi/rewards.py`

PhysHSI:
- `legged_gym/legged_gym/envs/g1/carrybox.py`
- `legged_gym/legged_gym/envs/g1/carrybox_config.py`

SUGAR:
- `source/sugar_rl/sugar_rl/tasks/locomanip/mdp/rewards.py`
- relevant G1 carry-box refiner configuration

Workhorse:
- inspect task-space targets and deployment-error augmentation;
- do not misrepresent it as a force-supervision method.

Record which mechanisms are directly reusable, which require adaptation, and which are incompatible with the target SuperMimic/mjlab version.

### P0.6 P0 deliverables

Create:
- `docs/hoi/force_contact/p0/REPORT.md`
- `docs/hoi/force_contact/p0/DATASET_ASSET_AUDIT.md`
- `docs/hoi/force_contact/p0/DEPENDENCY_CONTACT_AUDIT.md`
- `docs/hoi/force_contact/p0/ARCHITECTURE_AUDIT.md`
- `docs/hoi/force_contact/p0/FILE_CHANGE_MAP.md`
- supporting machine-readable evidence where useful.

The file-change map must identify for P1–P8:
- existing files to reuse;
- existing files requiring minimal changes;
- new files;
- tests;
- dependencies and risks.

### P0 acceptance criteria

P0 is PASS only when:
- actual checkout, branch, HEAD and dependency versions are recorded;
- all listed dataset/asset/checkpoint paths are marked EXISTS/MISSING/INACCESSIBLE;
- Part B integrity is verified against its committed manifest;
- actual code tensor shapes and integration semantics are traced;
- sensor update/history/recording order is understood;
- the source implementation audit is complete;
- the P1–P8 file-change map is grounded in actual code;
- no training source, datasets, assets, checkpoints or accepted Part B artifacts were modified.

**P0 stops here. Do not start P1 until the user reviews and approves P0.**

---

# 6. P1 — Physical HOI Scene and Real Contact Sensors

## What P1 means

P1 makes the physical task measurable. It creates or completes the actuated robot–box–support scene and establishes reliable, independently filtered left/right hand–box contact measurements.

P1 does not implement Replay, the Force Head, rewards, imagination or Actor optimization.

## Prerequisites

- P0 is approved.
- Use actual body/geom names and dependency APIs discovered during P0.
- Use the accepted M6.5 physical asset profile. Do not replace it with an unverified source asset.
- Keep the existing offline `ReferenceScene` and Part B viewer behavior unchanged.

## P1.1 Build the actuated HOI scene

If the target repository does not already contain an actuated HOI scene, add one under `src/supermimic/hoi/`.

It must contain:
- native G1 29-DoF robot and its original actuator/action configuration;
- the audited seven-geometry compound bread box as one dynamic rigid body;
- both audited static supports;
- ground and verified collision filters.

The existing offline reference viewer is kinematic and display-only. Do not repurpose it as the training environment.

Keep the accepted asset manifest as the source of geometry, mass, COM, inertia, contact targets, EEF offsets and support transforms.

## P1.2 Implement joint physical state access and reset

Create or extend an HOI-specific state interface.

Explicitly identify robot and box free-joint qpos/qvel addresses. Do not concatenate global MuJoCo qpos/qvel and pretend the result is `[robot71, object13]`.

Verify:
- robot root position is environment-local where expected;
- robot linear velocity is root-origin world velocity;
- robot angular velocity uses the documented body/world frame;
- box free-joint root pose and twist are extracted correctly;
- box COM velocity is converted to/from root-origin velocity correctly;
- quaternion order is wxyz;
- reset initializes robot and box from the same reference frame;
- box velocity is initialized consistently rather than silently set to zero;
- support poses remain fixed and environment-local.

Preserve the original robot-only `StateIO` path and its current nq/nv assertions. Use a separate HOI adapter or a narrowly scoped compatible extension.

## P1.3 Verify hand collision bodies

Use P0's compiled native G1 model and M6.5 manifest.

The previous P0 audit identified these native collision parent bodies:

Left:
- `left_wrist_pitch_link`
- `left_wrist_yaw_link`

Right:
- `right_wrist_pitch_link`
- `right_wrist_yaw_link`

Verify these names and collision enablement against the current constructed scene. Do not assume the audit's names remain valid after asset/configuration changes.

Do not accidentally include forearm or unrelated contacts.

## P1.4 Create independent filtered ContactSensors

Create:
- `left_hand_box`
- `right_hand_box`

Each hand sensor must include both verified hand collision bodies on its primary side and the single box rigid body on its secondary side.

The P0 audit found that mjlab expands primary matches into separate columns; it does not automatically sum multiple primary bodies into one hand wrench. Therefore explicitly aggregate the two primary-body netforces for each hand.

Use:
- `reduce="netforce"` if the installed API confirms its audited semantics;
- `num_slots=1`;
- `secondary_policy="error"` where supported;
- `history_length=4`;
- at least `found` and `force`, plus torque/position if needed for diagnostics.

The secondary body match must cover the complete compound box. Do not match only one box geom. Do not use a policy that silently selects the first ambiguous match or disables the secondary filter.

## P1.5 Verify sensor update timing and capacity

Verify that each sensor is updated after each of the four 5 ms physics substeps.

Confirm history ordering and that the latest four values correspond to exactly one 20 ms action interval.

Inspect:
- contact sensor match capacity;
- MuJoCo `nconmax` / `njmax`;
- Warp contact capacity and overflow flags;
- behavior with multiple box geoms contacting one hand;
- reset clearing of selected environment histories;
- CUDA graph capture and pointer stability.

Do not infer absence of overflow from quiet/suppressed logs. Add explicit overflow checks or assertions for calibration runs.

Do not patch installed dependency source unless an actual API limitation is demonstrated and the user approves the scope.

## P1.6 Calibrate force sign and coordinate frame

The audited netforce convention is primary-on-secondary. With hand primary and box secondary, it should represent force exerted by the hand on the box.

Still verify this in the actual HOI scene:
1. fixed robot, left hand pushes in a known direction;
2. repeat for right hand;
3. rotate the box and repeat;
4. compare sensor force with box motion and known contact geometry;
5. verify Box-local transformation.

Canonical target:

`force_hand_on_box_world`, then `force_hand_on_box_box_local`.

Do not apply a second sign inversion if the actual API already returns primary-on-secondary.

## P1.7 Calibration scenarios

Run controlled scenarios:
- no hand-box contact;
- left-only contact;
- right-only contact;
- bilateral contact;
- each hand contacting different compound box geoms;
- hand-ground without hand-box;
- box-support without hand-box;
- rotated box;
- contact only during substep 1;
- contact only during substep 4;
- contact ending before the final substep;
- reset immediately after contact;
- multiple parallel environments with different contact states.

Because the installed ContactSensor has no `found_history`, distinguish:
- geometric/constraint contact found at the current sample;
- force-bearing contact inferred from force history;
- any additional transient geometric-contact bookkeeping, if genuinely required.

Do not claim that force history reconstructs every geometric contact event. Netforce cancellation can also hide opposing local contact loads.

## P1 deliverables

- actuated HOI scene and joint state/reset interface;
- independent left/right hand-box ContactSensors;
- calibration scripts and tests;
- `docs/hoi/force_contact/p1/REPORT.md`;
- raw calibration evidence under `outputs/hoi/contact_supervision/sensor_calibration/`.

## P1 acceptance criteria

- correct scene and audited assets;
- correct robot/box state extraction;
- synchronized robot/box reset;
- verified left/right pair filtering;
- both primary hand bodies included per side;
- all seven box geoms covered;
- force sign and frames verified physically;
- four-substep history verified;
- no unhandled contact overflow;
- reset and parallel-environment isolation pass;
- original robot-only tests pass.

**P1 stops here. Do not implement Replay or World Model changes. Report and wait for approval.**

---

# 7. P2 — Control-Interval Force Aggregation and HOI Replay

## What P2 means

P2 turns raw substep contact measurements into correctly aligned training data. Every 20 ms action transition must be paired with the force generated during exactly that action interval.

P2 does not implement the Force Head.

## Prerequisites

P1 is approved. Force sign, pair filtering, Box-local frame and sensor history semantics are frozen.

## P2.1 Force aggregation

Implement/complete `src/supermimic/hoi/contact.py`.

For each hand and each substep:
1. read the two primary-body netforce columns;
2. sum them into one hand-level net wrench, after verifying the force convention;
3. convert each substep world force into the Box-local frame using that substep's Box orientation;
4. compute interval statistics.

For hand \(i\), four substeps \(k\):

\[
\bar F_i^{box}=\frac14\sum_{k=1}^{4}F_{i,k}^{box}
\]

Unit: N. This is the primary Force Head target.

\[
J_i^{world}=0.005\sum_{k=1}^{4}F_{i,k}^{world}
\]

Unit: N·s.

\[
F_i^{peak}=\max_k\|F_{i,k}^{world}\|
\]

Unit: N.

\[
\rho_i=\frac14\sum_{k=1}^{4}
\mathbb{1}[\|F_{i,k}^{world}\|>F_{on}]
\]

Unitless occupancy in [0,1].

Keep raw substep measurements available for diagnostics. Do not sum force values and call the result force. Do not use peak force as the first Force Head target.

### Box-frame anchoring decision

The force history stores four force vectors but may not store the four corresponding Box orientations.

Therefore P2 must not silently rotate all four samples using the post-step Box orientation.

Use one of these approaches, based on the actual P1 implementation:
- capture Box orientation alongside each substep force and rotate each force into its contemporaneous Box frame before aggregation; or
- define a single documented interval anchor frame and rotate the interval force into that frame.

The first is preferred when the additional state capture is reliable and inexpensive. Record the exact convention in the Replay schema and checkpoint metadata.

## P2.2 Contact semantics

Keep distinct:
- `found`: current matched constraint-contact count;
- force-bearing contact: thresholded measured force;
- contact occupancy: fraction of four substeps above threshold.

A nonzero `found`, a nonzero net force and a positive sum of local normal loads are not equivalent. Use raw fields and calibrated thresholds; do not collapse them into one undocumented boolean.

## P2.3 HOI Replay schema

Create an HOI-specific adapter/schema, preferably `src/supermimic/hoi/replay.py`, reusing original ReplayBuffer storage and sampling mechanisms.

One transition means:

\[
(s_t,a_t^{exec})\rightarrow(s_{t+1},F_{t:t+1}^{sim})
\]

Store:
- robot state and next robot state;
- object state and next object state;
- clean action and executed action;
- domain context;
- support-relative context only if separately justified;
- left/right mean Box-local force;
- left/right world impulse;
- left/right peak force;
- left/right contact occupancy;
- left/right contact-found information with clear sample-time semantics;
- reference frame, clip ID, environment ID and episode ID;
- physical failure, time-limit truncation, natural clip-end, terminal and episode-start masks.

Do not store a current-sample `found` value as if it were a four-substep history label.

## P2.4 Temporal alignment

Inspect the actual callback sequence from P0/P1.

The force label covers the four substeps during which `action_exec_t` was applied.

Document:
- transition start state;
- action execution;
- each physics substep;
- substep force and Box orientation;
- transition end state;
- reference frame convention;
- desired-contact intent over that interval.

The terminal path must copy force history before reset clears it. The nonterminal path must copy the correct interval history after the step. Do not retain a pointer to a history tensor that is rolled/replaced on the next update; clone or copy the interval label.

## P2.5 Resolve existing recorder terminal-label discrepancy

P0 identified that the current recorder writes `done_next=1` on the final transition and also on an extra terminal-state row, while the replay window reader consumes successor-row labels. This can yield `[1,1]` where transition labels should be `[0,1]`.

Do not blindly copy this behavior into HOI Replay.

Define one unambiguous transition-indexed terminal schema and test it. If a shared recorder/replay fix is necessary, keep it narrowly scoped and prove original `sm-train` behavior is unchanged.

Keep physical failure, timeout truncation and natural clip-end separate.

## P2.6 Window sampling

Preserve the original WM autoregressive window length of 8.

Every sampled window must:
- remain within one environment and episode;
- remain within one clip;
- contain consecutive valid transitions;
- never cross reset;
- obey physical terminal, timeout and clip-end masks;
- align state, action, force and masks at identical indices.

If short windows or padding are supported, include an explicit validity mask. Do not silently treat padded samples as real transitions.

## P2.7 Diagnostics and deliverables

Report:
- active contact ratio per hand;
- bilateral ratio;
- mean/peak force distributions;
- impulse distribution;
- occupancy distribution;
- onset/loss frequency;
- zero-force tolerance;
- per-clip/per-environment statistics;
- force label/action interval alignment examples.

Create:
- Replay extension;
- aggregation and temporal-alignment tests;
- `docs/hoi/force_contact/p2/REPORT.md`;
- outputs under `outputs/hoi/contact_supervision/replay_diagnostics/`.

## P2 acceptance criteria

- correct per-hand aggregation of both primary bodies;
- correct Box-frame convention;
- correct force/impulse units;
- exact action-to-force temporal alignment;
- terminal force copied before reset;
- no stale sensor history;
- no cross-episode or cross-clip windows;
- terminal/timeout/clip-end semantics tested;
- original robot-only Replay tests pass.

**P2 stops here. Do not implement the Force Head. Wait for approval.**

---

# 8. P3 — Joint Robot–Object World Model and Force Prediction Head

## What P3 means

P3 teaches one learned dynamics model to predict robot motion, box motion and left/right hand forces jointly. This is the first stage that changes the learned physical dynamics model.

It is not merely adding six output numbers to the old robot-only WM.

## Prerequisites

P2 is approved. Joint physical state and force Replay labels are validated.

## P3.1 Extend state representation

Preserve the original Robot StateSpec and exact feature order.

Add:
- 13-D object physical state;
- 15-D object canonical feature;
- joint 84-D physical state;
- joint 86-D canonical feature.

Verify actual qpos/qvel addresses, root versus COM, quaternion order, angular velocity frame and environment-local coordinates.

Do not use global `[qpos|qvel]` concatenation as the joint state.

## P3.2 Object canonicalization

Candidate object canonical feature:

\[
\phi_o=[
p_{o,rel}^{h}(3),
R_{o,rel}^{6D}(6),
v_o^{h}(3),
\omega_o^{body}(3)]
\]

where relative position/orientation and linear velocity are expressed relative to the robot heading frame, and angular velocity frame is explicitly documented.

This is a candidate contract, not a license to ignore P0's frame findings.

Test invariance and round-trip behavior.

## P3.3 Object integration

Add a separate object integration path with a constant-velocity prior.

The 12-D residual is:

- linear velocity residual 3;
- angular velocity residual 3;
- position correction 3;
- orientation correction 3.

Use the correct root-origin/COM velocity relationship. Do not directly regress quaternion components. Verify right/left quaternion multiplication against the chosen angular-velocity frame.

Zero residual must produce the documented constant-velocity prediction.

## P3.4 Support-relative context identifiability

P0 found that robot and robot-relative object canonical features discard common global XY/yaw. Static supports can therefore create different object dynamics for otherwise identical canonical features.

Do not silently append support features to the existing 33-D episode-constant `xi`.

Perform a paired identifiability experiment:
- construct physically corresponding robot/object states with identical joint canonical features;
- vary their support-relative placement;
- compare one-step object and force prediction errors with and without support-relative context.

If context is required, represent support geometry relative to the current robot/root heading or another explicitly justified invariant frame. It is time-varying physical context, separate from episode randomization.

Freeze its exact encoding/dimension only after the experiment and report.

## P3.5 Joint WM architecture

Use a shared joint dynamics trunk with:
- robot residual head: 70-D;
- object residual head: 12-D;
- auxiliary force head: 6-D.

Combined head width: 88-D. Only 82-D is integrated into physical state.

Reuse original DynamicsMLP and WM training machinery. Preserve the original robot output semantics and constant-velocity prior.

Do not create two independent WMs that cannot represent interaction coupling.

## P3.6 Force prediction

Predict:

`[left_force_box_xyz, right_force_box_xyz]`

for the next 20 ms action interval.

Input:
- joint physical/canonical state;
- executed action during WM fitting;
- relevant physical context;
- support context only if justified.

Do not feed reference desired-contact intent into the physical Force Head as a shortcut. Physical force is determined by state, action, geometry and scene.

## P3.7 Force normalization and loss

Keep Replay force in physical units. Use separate reversible normalization and checkpoint metadata.

Use robust regression, e.g. Huber loss.

Balance active/inactive examples so zero-force samples do not dominate.

\[
L_F=\lambda_{active}L_{active}+\lambda_{inactive}L_{inactive}
\]

Overall:

\[
L_{WM}=\lambda_rL_r+\lambda_oL_o+\lambda_FL_F+\lambda_dL_{termination}
\]

Weights must be configurable and chosen after measuring component scales, not guessed.

Report left/right, active/inactive and low/high force bins.

## P3.8 Validation and readiness

Use episode/clip-aware validation to avoid adjacent transition leakage.

Evaluate:
- left/right force MAE;
- force magnitude error;
- force direction cosine similarity;
- onset/loss prediction;
- active/inactive calibration;
- robot/object one-step prediction;
- 8-step autoregressive prediction;
- contact versus non-contact prediction quality.

Implement a configurable readiness gate based on validation evidence, not a fixed iteration count.

Do not enable Force Head reward in Actor imagination in P3.

## P3.9 Checkpoint migration

Use explicit named field/column mappings.

Preserve original robot WM trunk and robot output where semantically compatible. Map action and context columns by named offsets; do not copy the first N columns blindly.

Initialize object and force branches intentionally.

Do not use `strict=False` as a migration strategy.

## P3 deliverables

- joint StateSpec/canonicalization/integration;
- shared-trunk robot/object/force WM;
- force loss and normalization;
- support-context identifiability report;
- checkpoint migration unit tests;
- `docs/hoi/force_contact/p3/REPORT.md`;
- outputs under `outputs/hoi/contact_supervision/force_model_validation/`.

## P3 acceptance criteria

- robot-only state semantics preserved;
- object integration and zero-residual prior validated;
- support-context decision evidence-based;
- robot/object/force heads have correct shapes;
- force normalization reversible;
- active/inactive imbalance handled;
- validation is clip/episode-aware;
- Force Head predicts nontrivial left/right force;
- original robot-only WM regression passes.

**P3 stops here. Do not enable force-based Actor optimization. Wait for approval.**

---

# 9. P4 — Soft Contact and Interaction Reward Design

## What P4 means

P4 defines and tests what good physical interaction should be rewarded for: approaching the reference contact target, establishing physically meaningful contact, maintaining useful contact and releasing naturally.

It does not integrate the rewards into long Actor imagination yet.

No hard-coded six-phase one-hot labels.

## Prerequisites

P3 is approved. Force Head validation is available. Synthetic force predictions may be used for unit tests, but force reward in Actor BPTT remains gated by readiness and P6.

## P4.1 Separate contact signals

Reference intent:

\[
c_t^{ref}\in[0,1]
\]

Source: global Part B `object_contact`.

Actual simulator contact:

\[
c_{t,i}^{sim}\in\{0,1\}
\]

Source: independently filtered MuJoCo left/right hand-box measurement.

Predicted imagination contact:

\[
c_{t,i}^{pred}\in[0,1]
\]

Source: predicted force and differentiable contact geometry.

Never equate these signals. Part B has no independent left/right desired-contact labels.

## P4.2 Smooth reference contact

Keep original bool NPZ unchanged.

Optionally derive a bounded smooth transition signal in a separate cache:
- preserve original active interval;
- do not shift contact onset/offset;
- document interpolation parameters;
- retain original bool signal for exact evaluation.

## P4.3 Pre-contact approach shaping

Mandatory.

Compute differentiable EEF positions using verified FK and audited wrist-to-EEF offsets.

Compute contact targets using current predicted Box pose and audited Box-local contact geometry.

\[
d_{t,i}=\|p_{EEF,i}-p_{target,i}\|
\]

\[
\Phi(d_i)=\exp(-d_i/\sigma_a)
\]

Build future-contact lookahead:

\[
g_t^{pre}=(1-c_t^{ref})\sum_{k=1}^{K_a}w_kc_{t+k}^{ref}
\]

where \(w_k\geq0\) and \(\sum_k w_k=1\).

Choose lookahead from the actual 840-frame reference approach geometry, not an invented approach duration.

\[
r_{approach}=g_t^{pre}\frac12[\Phi(d_{t,L})+\Phi(d_{t,R})]
\]

Original robot tracking stays active. Approach reward must not cause arbitrary deviation from robot reference motion.

Do not add an approach-direction classifier unless diagnostics establish a specific need.

## P4.4 Contact position reward

\[
r_{pos,i}=\exp\left(-\frac{\max(d_{t,i}-d_{tol},0)}{\sigma_p}\right)
\]

Use actual EEF geometry, not wrist origin without the verified transform.

## P4.5 Bounded force reward

Use predicted Box-local force in imagination:

\[
F_i=\|\hat F_{t,i}\|
\]

\[
r_{force,i}=\exp\left(-\frac{\max(F_{min}-F_i,0)}{\sigma_F}\right)
\]

This HDMI-inspired bounded force shaping:
- insufficient force lowers reward;
- sufficient force saturates;
- excessive force gains no additional reward.

\[
r_{contact}=c_t^{ref}\frac12\sum_{i=L,R}r_{pos,i}r_{force,i}
\]

Calculate hands independently, then average. Do not require simultaneous bilateral contact in v1.

Physical diagnostic uses actual simulator force; imagination uses predicted force. Never query MuJoCo inside differentiable imagination.

## P4.6 Predicted soft contact confidence

\[
c_{t,i}^{pred}=
\sigma\left(\frac{\|\hat F_{t,i}\|-F_{on}}{\tau_F}\right)
\sigma\left(\frac{d_{on}-d_{t,i}}{\tau_d}\right)
\]

This is a differentiable confidence, not actual measured contact.

## P4.7 Persistence

\[
r_{persist}=c_t^{ref}\frac12\sum_i c_{t-1,i}^{pred}c_{t,i}^{pred}
\]

Do not penalize all force variation. Changing force is necessary for manipulation.

Log contact loss separately. Do not impose a strong loss penalty before evidence supports continuous per-hand contact.

## P4.8 Release

\[
g_t^{release}=c_{t-1}^{ref}(1-c_t^{ref})
\]

\[
r_{release}=g_t^{release}\left(1-\frac12\sum_i c_{t,i}^{pred}\right)
\]

Couple release with box-support stability. Do not reward hand separation if the box falls or slides.

## P4.9 Support stability

Measure box-support contact separately from hand-box contact.

Evaluate:
- support contact;
- box position and orientation;
- linear and angular velocity;
- slip;
- post-release stability.

Use differentiable predicted object state/geometry in imagination and actual simulator measurements in physical evaluation.

## P4.10 Combined reward

\[
\begin{aligned}
r_{HOI}={}&\lambda_r r_{robot}+\lambda_o r_{object}\\
&+\lambda_a r_{approach}+\lambda_c r_{contact}\\
&+\lambda_s r_{persist}+\lambda_{rel}r_{release}\\
&+r_{support}+r_{safety}
\end{aligned}
\]

Robot and object tracking remain active throughout. Do not automatically lower robot tracking weight when contact activates.

Log raw and weighted contributions of every reward. Calibrate weights from measured scales and gradient diagnostics; do not invent final coefficients.

## P4.11 Unit tests

Test controlled cases:
- far from target;
- approaching before desired contact;
- near target but no force;
- correct position but insufficient force;
- correct position and sufficient force;
- excessive force;
- continuous contact;
- unexpected contact loss;
- reference release transition;
- hands separated but box unstable;
- box stable on support after release.

## P4 deliverables

- shared pure-Torch reward functions;
- unit tests;
- `docs/hoi/force_contact/p4/REPORT.md`;
- outputs under `outputs/hoi/contact_supervision/reward_validation/`.

## P4 acceptance criteria

Correct activation, finite outputs, correct gradients, position and force both influence contact reward, force reward saturates, persistence does not demand constant force, release is coupled to stability, and no hard-coded six-phase labels.

**P4 stops here. Do not integrate full imagination. Wait for approval.**

---

# 10. P5 — Differentiable Joint Imagination and Actor BPTT

## What P5 means

P5 connects validated joint WM and reward functions to the original SuperMimic imagination path. It proves actions can influence predicted Box motion and force, and both affect Actor gradients.

## Prerequisites

P4 is approved. Force Head readiness is established. Force-based Actor reward remains gated pending P6 go/no-go.

## P5.1 Joint imagined state

Maintain predicted robot and object physical states. Never use reference Box pose as the current imagined state.

## P5.2 Dynamic future goals

At every imagined step:
1. read predicted robot state;
2. read predicted object state;
3. read future reference robot/object target;
4. transform future object goal relative to the current predicted robot frame;
5. recompute contact target from the current predicted Box pose;
6. include reference desired-contact intent.

Do not precompute robot-relative object goals once and reuse them after imagined robot drift.

## P5.3 Joint WM step

Predict robot residual, object residual and left/right force.

Integrate only robot/object physical residuals. Force is auxiliary and is not integrated.

## P5.4 Differentiable contact geometry

Recompute FK, EEF positions and contact target geometry at every imagined step.

No MuJoCo contact query inside imagination. No NumPy conversion. No detach of predicted object state or force.

## P5.5 Persistence state

Carry previous predicted contact confidence through the rollout. Initialize/reset explicitly and preserve the intended gradient path.

## P5.6 Clip-end and terminal masks

Distinguish:
- robot physical failure;
- object physical failure;
- natural clip-end;
- time-limit truncation.

Do not repeat final-frame reward indefinitely.

Apply correct survival and critic bootstrap semantics. Natural task completion must not be treated as physical failure.

## P5.7 BPTT gradient checks

Verify:

\[
a_t\rightarrow\hat s_{t+1}^{object}\rightarrow r_{object}
\]

\[
a_t\rightarrow\hat F_{t+1}\rightarrow r_{contact}
\]

WM parameters may be frozen, but gradients through WM to action must remain intact.

## P5.8 Eager/compiled parity

Compare:
- robot state;
- object state;
- predicted force;
- predicted contact confidence;
- every reward component;
- total reward;
- persistence state;
- action gradients.

Compilation must not change semantics.

## P5 deliverables

- HOI imagination integration;
- minimal shared rollout/compiled/policy-trainer changes;
- gradient and parity tests;
- `docs/hoi/force_contact/p5/REPORT.md`.

## P5 acceptance criteria

Object and force reward gradients reach Actor; no unintended detach; correct clip-end behavior; eager/compiled values and gradients agree within documented tolerances; original robot-only imagination regression passes.

**P5 stops here. Do not launch expensive training. Wait for approval.**

---

# 11. P6 — Force Gradient Fidelity Validation

## What P6 means

P6 checks whether Force Head gradients are useful for policy optimization, not merely whether force prediction error is low.

A model may predict force accurately while providing a misleading derivative with respect to action.

## Prerequisites

P5 is approved.

## P6.1 Deterministic simulator experiments

Use repeatable initial states for:
- left-only contact;
- right-only contact;
- bilateral contact;
- low-force contact;
- sustained contact;
- contact onset/loss.

Keep initial state, reference frame, randomization and unperturbed action dimensions fixed.

## P6.2 Finite-difference reference

For selected action dimension \(j\):

\[
\frac{\partial F}{\partial a_j}\approx
\frac{F(a+\epsilon e_j)-F(a-\epsilon e_j)}{2\epsilon}
\]

Use multiple epsilon values to identify numerical instability near non-smooth contact boundaries.

## P6.3 Model/simulator comparison

Calculate:
- model \(\nabla_a\hat F\);
- model \(\nabla_a r_{contact}\);
- simulator finite-difference force response;
- simulator finite-difference reward response where meaningful.

Report:
- cosine similarity;
- sign agreement;
- relative magnitude error;
- per-action-dimension results;
- left/right results;
- contact-regime-specific results;
- epsilon sensitivity.

Do not claim exact differentiability of MuJoCo contact. Identify unstable boundary cases.

## P6.4 Go/no-go

If force prediction is accurate but gradients are systematically misleading:
- do not enable unrestricted force-based Actor BPTT;
- retain Force Head for diagnostics/supervision;
- report evidence;
- propose bounded/staged alternatives for user approval.

Do not silently change the algorithm.

## P6 deliverables

- deterministic finite-difference scripts;
- machine-readable gradient results;
- `docs/hoi/force_contact/p6/REPORT.md`;
- outputs under `outputs/hoi/contact_supervision/gradient_fidelity/`.

## P6 acceptance criteria

Reproducible comparisons, separate left/right and contact-regime results, documented limitations and explicit go/no-go recommendation.

**P6 stops here. Do not start integrated training. Wait for approval.**

---

# 12. P7 — Checkpoint Migration and Integrated SuperMimic Training

## What P7 means

P7 connects validated HOI components to the actual SuperMimic training entry point and proves original robot-only tracking remains compatible.

## Prerequisites

P6 is approved. Force-gradient go/no-go decision is explicit.

## P7.1 Separate entry point

Provide `smHOI-track`.

Reuse original `Trainer.run()` and training scheduler.

Do not change default `sm-train` behavior.

## P7.2 Explicit checkpoint migration

Explicitly map:
- original robot WM trunk and robot output;
- original actor observation/goal inputs and output;
- critic-compatible features where appropriate.

Initialize object and force branches intentionally.

Do not use `strict=False` as the migration strategy.

## P7.3 Actor parity

Where new Actor inputs are added, zero-initialize their contribution and verify:

\[
\max|\pi_{old}(o,g)-\pi_{new}([o,0],[g,0])|<\epsilon
\]

Use actual input layout and named field maps; do not assume concatenation order.

## P7.4 Critic

Extend Critic only as required by the actual HOI objective.

Decide whether force prediction is privileged Critic information based on the rollout/training contract. Actor and Critic need not have identical inputs.

Preserve TD(lambda), terminal and truncation semantics.

## P7.5 Optimizers, Replay and checkpoint metadata

Create new HOI optimizer/Replay state where schema changes require it.

Persist:
- state/feature schema versions;
- force frame/sign convention;
- force normalization;
- sensor configuration;
- physics/control timestep and decimation;
- reward configuration;
- Force Head readiness;
- reference hash;
- asset manifest hash.

Do not reuse incompatible optimizer state. Never overwrite the original checkpoint.

## P7.6 Staged short training

Run only explicitly approved short experiments:

1. robot-only regression;
2. joint robot-object WM;
3. Force Head supervised warm-up;
4. approach reward short run;
5. position-force reward short run;
6. full soft interaction reward short run;
7. joint Actor BPTT and Critic training.

Do not launch long training without explicit user approval.

## P7.7 Verify actual updates

Demonstrate actual updates to:
- robot WM;
- object WM;
- Force Head;
- Actor;
- Critic.

Report gradient norms and parameter deltas.

Verify checkpoint save/resume and Replay schema restoration.

## P7 deliverables

- separate `smHOI-track` entry;
- training configuration;
- explicit migration tests;
- short-run logs/checkpoints;
- `docs/hoi/force_contact/p7/REPORT.md`;
- outputs under `outputs/hoi/contact_supervision/training/`.

## P7 acceptance criteria

Original robot-only regression passes; Actor parity passes; intended modules update; checkpoint save/resume works; no NaN/Inf; original `sm-train` behavior unchanged; short HOI training completes with interpretable metrics.

**P7 stops here. Do not run full task evaluation until approved.**

---

# 13. P8 — Full 840-frame Pick–Carry–Place–Release Evaluation

## What P8 means

P8 evaluates the selected trained checkpoint on the complete physical task. It is evaluation, not another training stage.

## Prerequisites

P7 is approved. The user explicitly identifies the checkpoint to evaluate.

## P8.1 Evaluation setup

Record:
- checkpoint path and SHA256;
- Part B reference path and SHA256;
- M6.5 asset manifest hash;
- simulator/dependency versions;
- physics timestep and decimation;
- deterministic evaluation settings.

Start from frame 0.

## P8.2 Full task

Evaluate:
- approach;
- contact establishment;
- pickup/lift;
- carry;
- placement;
- release;
- post-release stability.

These may be descriptive evaluation labels, but must not become hard-coded training phase one-hot variables.

## P8.3 Metrics

Contact:
- left/right actual contact ratio;
- bilateral ratio;
- onset/loss timing;
- mean/peak force;
- impulse;
- occupancy;
- EEF-target distance.

Object:
- position/orientation error;
- linear/angular velocity error;
- slip;
- support contact;
- post-release motion/stability.

World Model:
- robot one-step and 8-step error;
- object pose/twist error;
- force MAE and direction similarity;
- contact onset/loss prediction;
- rollout drift.

Task:
- pickup success;
- carry success;
- placement success;
- release success;
- full completion;
- post-release stability.

Frame 839 alone is not proof of physical success. The Box must remain physically stable for a documented interval after release.

## P8.4 Failure analysis

Report every failed episode:
- failure frame;
- failure category;
- force/contact history;
- robot/object trajectories;
- likely cause;
- whether the issue is model, policy, reference, asset or simulator related.

Do not discard failed episodes from success-rate calculation.

## P8 deliverables

- evaluation scripts/configuration;
- trajectory/contact logs;
- metrics and plots;
- `docs/hoi/force_contact/p8/REPORT.md`;
- outputs under `outputs/hoi/contact_supervision/evaluation/`.

## P8 acceptance criteria

Evaluation starts at frame 0, traverses all 840 reference frames, evaluates real release and post-release support stability, and records checkpoint/reference/asset/simulator provenance.

**P8 stops after reporting. Do not automatically launch more training or modify the accepted checkpoint.**

---

# 14. Expected code organization and change boundaries

Prefer HOI-specific implementation under:

```text
src/supermimic/hoi/
    scene.py
    state.py
    contact.py
    replay.py
    world_model.py
    rewards.py
    imagination.py
    config.py
    checkpoint.py
    evaluate.py
```

Tests should be placed under:

```text
tests/hoi/
    test_joint_scene.py
    test_contact_sensor.py
    test_force_aggregation.py
    test_force_replay.py
    test_joint_world_model.py
    test_force_head.py
    test_soft_contact_rewards.py
    test_release_reward.py
    test_force_gradient.py
    test_imagination_parity.py
    test_checkpoint_migration.py
```

These are proposed locations. P0's actual file-change map takes precedence if equivalent modules already exist.

Shared files should receive only necessary interface extensions.

Do not:
- copy the complete Trainer;
- copy the complete ReplayBuffer;
- duplicate the original robot StateSpec;
- add an independent PPO loop;
- introduce unnecessary contact/phase classifiers;
- rewrite Part B;
- rebuild the reference Viewer;
- reprocess source datasets;
- change `sm-train` defaults.

---

# 15. Mandatory stage report, review and commit protocol

## 15.1 Report after every stage

Every stage must end with a report containing:

1. Stage and objective.
2. What was implemented.
3. Every new and modified file.
4. Tensor dimensions, data flow, units, coordinate frames and temporal alignment affected.
5. Exact commands executed.
6. Tests executed, passed, failed and skipped with reasons.
7. Numerical evidence and output paths.
8. Known limitations and deviations from this plan.
9. Git branch, HEAD, status, changed-file list and diff summary.
10. Acceptance: `PASS`, `PARTIAL` or `FAIL`.
11. Explicit confirmation that no next stage was started.

Do not claim a test passed unless it was actually executed. Compilation or tensor-shape agreement alone is not proof of physical validity.

## 15.2 Commit after explicit user approval

Do not commit automatically.

After the user reviews a PASS report and explicitly approves:
1. inspect the complete diff again;
2. ensure only current-stage changes are included;
3. create one primary commit for the stage;
4. report commit hash and final Git status;
5. do not push;
6. do not start the next stage until separately authorized.

Suggested commit messages:

| Stage | Commit |
|---|---|
| P0 | `docs(hoi): audit force-aware implementation inputs` |
| P1 | `hoi: add joint scene and filtered hand-box sensing` |
| P2 | `hoi: add force aggregation and replay supervision` |
| P3 | `hoi: add joint world model force prediction` |
| P4 | `hoi: implement soft contact and release rewards` |
| P5 | `hoi: integrate force-aware differentiable imagination` |
| P6 | `hoi: validate force gradient fidelity` |
| P7 | `hoi: integrate force-aware SuperMimic training` |
| P8 | `hoi: add full interaction evaluation` |

Corrective commits within a stage are allowed but must be documented.

Do not push unless separately requested.

## 15.3 Failure and design-change protocol

If a mandatory test fails:
- remain in the current stage;
- diagnose and report;
- propose a minimal correction;
- do not proceed.

If actual code or dependency behavior contradicts this plan:
- do not silently change state dimensions, coordinate conventions, force semantics, reward equations or architecture;
- explain the discrepancy and impact;
- propose a change;
- wait for user approval for research-level changes.

Small API adaptations are permitted only when semantics remain unchanged and are documented.

## 15.4 Protect existing work

Throughout all stages:
- preserve all existing user modifications;
- never reset or clean the repository;
- never overwrite checkpoints;
- never alter accepted Part B or source datasets;
- never weaken tests or rewrite historical audit results;
- never modify unrelated experiments;
- never launch expensive training without explicit approval;
- never automatically push.

---

# 16. Exact Codex execution instructions

Use one request per stage. Do not require a separate Plan Mode or Goal Mode pass: this master plan is the approved execution specification.

## First request — P0

```text
Read the complete document:
docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md

Execute P0 only.

P0 means repository, dataset, asset, dependency and reference-code audit.
It is read-only with respect to training source code.

Inspect all exact dataset and asset paths in Section 3. Verify the Part B reference against its committed manifest. Trace actual SuperMimic WM, Replay, Imagination, Trainer and checkpoint code. Verify installed mjlab/MuJoCo Warp ContactSensor API and update order.

Produce all P0 reports and the implementation file-change map.

Do not reprocess Part B, regenerate datasets, modify checkpoints, change training source code or start training.

Follow Section 15. Report PASS/PARTIAL/FAIL and STOP. Do not start P1.
```

## Later requests — P1 through P8

```text
Read the complete document:
docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md

Execute P<N> only, following the corresponding section.

Inspect the actual current checkout and confirm all prerequisites and previous-stage acceptance/commit status.

Preserve all existing work. Implement only the current stage. Run its required tests and experiments. Inspect the complete Git diff.

Provide the full stage report required by Section 15, including exact commands, numerical evidence, modified files, Git status and PASS/PARTIAL/FAIL.

Do not commit or push automatically.
Do not start the next stage.
Stop and wait for my explicit approval.
```

For P7/P8, obtain explicit approval before expensive training or full evaluation.

---

# 17. Final research acceptance

The overall project is complete only when:

1. Real left/right hand–box forces are measured with verified filtering and sign.
2. Four-substep force aggregation is correctly aligned with each action transition.
3. Replay preserves joint robot-object state and force supervision.
4. Joint WM predicts robot and object dynamics.
5. Force Head predicts actual simulator measurements.
6. Pre-contact approach shaping guides hands toward reference targets while preserving robot tracking.
7. Contact reward considers both position and force.
8. Persistence does not demand constant force.
9. Release is learned without hard-coded six-stage one-hot labels.
10. Support stability is measured independently.
11. Object and force predictions remain differentiable in imagination.
12. Actor receives meaningful object and force reward gradients.
13. Force gradient fidelity is measured and limitations are documented.
14. Eager and compiled paths are consistent.
15. Robot-only checkpoint migration and tracking regression pass.
16. Original `sm-train` behavior remains unchanged.
17. Full 840-frame evaluation starts at frame 0 and includes physical release and post-release stability.

The final research contribution is a **joint physical interaction model** in which humanoid actions produce predicted contact forces, those forces influence object dynamics, and the Actor learns to optimize the resulting interaction through differentiable SuperMimic imagination.

Do not reduce this task to adding a contact bonus to an ordinary tracking policy.
