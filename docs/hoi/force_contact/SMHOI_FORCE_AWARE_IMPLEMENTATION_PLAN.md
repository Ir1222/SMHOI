# SMHOI Part C — Reference-Conditioned World-Model-Based Humanoid–Object Interaction
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
| **P2** | Preserve action-aligned physical interaction data | Can each action transition retain aligned robot/object state and physical contact diagnostics in Replay? | Yes | Validated HOI Replay and physical measurements |
| **P3** | Learn joint robot-object dynamics | Can one joint WM predict robot and box motion from state, action and physical context? | Yes | Final 166D/82D joint WM architecture; training/validation deferred to P7-A |
| **P4** | Define reference-conditioned HDMI-style proximity reward | Does the unchanged binary reference label gate differentiable EEF-to-current-Box-target proximity while preserving robot/object tracking? | Yes; reward and geometric audit/tests only | Target/EEF/collision correspondence audit, justified target participation, reward-scale audit and tested reward; no imagination integration or training |
| **P5** | Integrate joint imagination and proximity reward | Do reference-aligned robot/object tracking and proximity gradients reach Actor through the joint WM? | Yes; integration and controlled tests only | Eager/compiled reward and gradient parity, terminal/bootstrap checks; no Actor training |
| **P6** | Validate geometry and independent physical contact agreement | Do geometric gradients behave correctly, and does measured contact agree with reference ON/OFF intent? | Validation code/experiments | Independent left/right, any-hand and bilateral metrics under P4.1 semantics; penetration/solver diagnostics, asset/support reconciliation and go/no-go |
| **P7** | Train the joint WM, then Actor/Critic | Can P7-A validate joint dynamics before P7-B policy training using the original SuperMimic loop? | Yes; separately authorized experiments | Held-out WM validation, short policy training and proximity ablations with physical evaluation; robot-only compatibility |
| **P8** | Evaluate the complete physical task | Does the selected policy complete all 840 frames from frame 0, including release and post-release stability? | Evaluation code/experiments | Full-task tracking, physical binary contact agreement, independent hand metrics, penetration diagnostics, post-release stability and all failures |

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

- **P0 is an audit stage.** It verifies the existing repository, dependencies, datasets, assets and implementation requirements.
- **P1–P2 establish the physical HOI scene, contact sensing and action-aligned physical interaction data.**
- **P3 defines the final joint robot-object World Model architecture.** The dedicated contact-classifier diagnostic is skipped. Joint WM predictive training and validation remain deferred to P7-A.
- **P4 defines the reference-conditioned physical interaction reward.** It combines the original binary reference contact condition with differentiable EEF-to-object-target proximity guidance. Actual simulator contact is retained as a separate physical measurement and evaluation signal.
- **P5 integrates the joint robot-object dynamics and proximity reward into differentiable imagination and Actor BPTT.** Integration and controlled gradient tests do not authorize Actor training.
- **P6 validates geometric reward behavior and compares reference contact intent with actual MuJoCo hand-box contact across the complete evaluated reference interval.** It reports independent hand measurements, both aggregate measurements and penetration diagnostics under the P4.1 audit contract.
- **P7 trains and validates the joint WM, then trains the Actor/Critic using the approved reference-conditioned proximity reward.** P7-A must pass before P7-B.
- **P8 evaluates the complete physical Pick-Carry-Place-Release task, including actual contact agreement and post-release object stability.**

Do not confuse:
- `P0–P8`: Codex execution order in this document.
- `C0–C5`: broader architecture topics in `SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md`.
- `M0–M6.5`: previously completed physical asset audit.
- `Part B / B1–B3`: previously completed reference processing and acceptance.

The final research design does not add a Contact Prediction Head, Force Prediction Head, Contact BCE, force regression, force-based Actor reward or additional contact-related observation/state dimensions.

Reference contact labels are used directly as known temporal conditions for the differentiable proximity reward. Actual MuJoCo hand-box contact labels are independently derived from filtered physical contact measurements and used for physical validation, not as differentiable Actor reward inputs.


## 1.3 P0 Audit Integration and Adaptive Implementation Protocol

This section extends the execution rules in Sections 1.1 and 14–16 for every stage P1–P8. The Master Plan remains the single source of truth for research objectives, contracts and acceptance criteria. The complete P0 audit evidence is a mandatory implementation constraint, not background reading. Do not repeat an implementation detail from the original plan when verified repository or dependency behavior contradicts it; resolve the discrepancy through the amendment procedure below.

The accepted P0 baseline comprises commits `579d1a8c661985c91926a365c7505488abe03bb8`, `e89573171c86be7f40c8d120dc813cee0274a152` and final closure `b5c604a0a3924787e553a265a994bc9a12e3cc65`. Preserve their reports, raw evidence and Git history. Earlier missing-plan, pre-commit and master-hash observations describe their historical snapshots; do not rewrite them to describe later state.

### 1.3.1 Required reading and current-state verification

Before starting each stage:

1. Read the complete Master Plan as required by Section 1.1, focusing on the current stage, cross-stage contracts and review gates.
2. Read [P0 REQUIREMENT_COVERAGE.md](p0/REQUIREMENT_COVERAGE.md), [REPORT.md](p0/REPORT.md), [CLOSURE.md](p0/CLOSURE.md) and [FILE_CHANGE_MAP.md](p0/FILE_CHANGE_MAP.md).
3. Read the relevant [architecture audit](p0/ARCHITECTURE_AUDIT.md), [dependency/contact audit](p0/DEPENDENCY_CONTACT_AUDIT.md), [dataset/asset audit](p0/DATASET_ASSET_AUDIT.md) and [reference implementation audit](p0/REFERENCE_IMPLEMENTATION_AUDIT.md), including the supporting JSON, probes and validation records needed for the stage. Record why any audit category is not applicable.
4. Read the previous accepted stage's report and inspect its accepted commit, including documented corrective commits. For P1, use the final P0 closure and the three commits above. Record acceptance and authorization separately; a commit does not authorize the next stage. Historical reports/file-change maps may describe superseded force/contact-head designs; preserve that evidence and apply the current Master Plan's architecture and stage boundaries to new work.
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

The following is a minimum mapping, not a replacement for the full audits or stage acceptance criteria. Exact code paths and evidence belong in each stage specification.


| Stage | P0 constraint and implementation consequence | Required validation or visible risk |
|---|---|---|
| P1 | Use a physical scene and named robot/object state adapter. Each native hand needs pitch and yaw collision parents; one box body covers all seven geoms. Configure the audited mjlab API with independent filtered sensors, secondary_policy=error, one netforce slot and history length four. | Calibrate colliders, left/right isolation, box geoms, force sign/frame, four substeps, transients, overflow, reset and parallel worlds. Preserve M6.5 historical failures. |
| P2 | Four newest-first samples span .005 × 4 = .020 s. Preserve accepted action/transition alignment and Replay episode/clip boundaries. Actual force/contact measurements remain physical diagnostic data, not active WM targets or Actor reward inputs. | Test executed-action/state/terminal alignment, validity and episode/clip boundaries. Distinguish force magnitude (N), impulse (N·s), contact occupancy and binary contact labels. |
| P3 | Final joint state84, canonical86, residual82 and support context18. One shared dynamics trunk with 166D input and 82D output. No Contact Prediction Head, Contact BCE, Force Prediction Head or force regression. | Validate state/frame/integration contracts and checkpoint migration. Skip dedicated contact-boundary classifier fitting. Defer learned joint WM validation to P7-A. Preserve all existing dimensions and original robot-only compatibility. |
| P4 | Preserve the original global binary reference contact label. Use it to gate the HDMI-style differentiable EEF-to-object-target proximity reward. Do not introduce additional temporal classifiers, force prediction or new observation dimensions. | Audit EEF offsets, object-local targets, collision geometry correspondence, FK and distance scales. Verify exact reference-frame alignment and reward gradients. Actual contact labels remain outside the differentiable reward. |
| P5 | Integrate the proximity reward into the original joint imagination and Actor BPTT. Recompute EEF and Box-local target geometry from current imagined robot/object states. | Test action-to-geometry gradients, reference alignment, terminal/bootstrap masks and eager/compiled parity. Preserve original Actor/Critic interfaces and state dimensions. |
| P6 | Compare original reference binary contact labels against independently measured actual MuJoCo hand-box contact at every valid evaluated timestep. Keep left/right measurements independent and preserve support-contact separation. | Report contact agreement, confusion matrices, precision/recall, any-hand and bilateral contact under the P4.1/P6.2 aggregation contract, contact duration, force diagnostics and collision penetration diagnostics. Do not use actual contact measurements as differentiable reward inputs. |
| P7 | Retain Trainer.run(), joint WM architecture, original Actor/Critic optimization and the approved proximity reward. | Train/validate joint WM before Actor BPTT. Perform reward ablations and physical rollout evaluation. Preserve checkpoint compatibility and sm-train defaults. |
| P8 | Accepted Part B has 840 samples at 50 Hz. M6.5 physical asset limitations remain visible until resolved or explicitly accepted for bounded evaluation. | Evaluate from frame 0 through complete pick-carry-place-release, including physical contact agreement, object stability, release and post-release behavior. Report all failures and provenance. |


Workhorse's approved paper/available-repository fallback remains a source limitation, not evidence of verified Workhorse code. External algorithms, thresholds, assets or deployment augmentation are not automatically approved for adoption.

### 1.3.5 Preserved objectives and execution sequence

Preserve the approved joint robot-object World Model, original SuperMimic training mainline, reference-conditioned geometric contact reward, differentiable imagination and Actor BPTT. The original Critic/TD(lambda), accepted Part B reference, source datasets, checkpoints, M6.5 evidence and robot-only defaults remain protected under Sections 2 and 15.

The final research architecture does not contain a Contact Prediction Head, Force Prediction Head, contact BCE, force regression, force-based Actor reward or force-gradient fidelity module. Physical hand-box measurements and force aggregation from P1/P2 remain available as diagnostics and physical evaluation evidence; they are not targets or inputs to the final joint WM loss.

The current 167D/83D Contact-Head implementation is transitional. P3 architecture cleanup must restore the final 166D/82D contract before P4 implementation. The dedicated P3 contact-boundary collection and classifier diagnostic are skipped. P3 remains historically PARTIAL; joint robot-object predictive training/validation is deferred to P7-A.

For each separately authorized stage: read and verify evidence → write IMPLEMENTATION_SPEC.md → present/classify amendments and resolve required approvals → implement only the authorized scope → run stage tests → reconcile results with the specification → inspect the complete diff → report PASS/PARTIAL/FAIL → stop for review. Commit only after explicit approval under Section 15.2; never automatically push or start the next stage.

This protocol does not authorize source edits, P3 cleanup, P4 implementation or later experiments by itself.

### 1.3.6 Minimal-Invasive Implementation and Original Framework Preservation

cThe objective is to extend SuperMimic for joint humanoid–object interaction, not to build a second training framework.

**Preserve and reuse the original SuperMimic implementation wherever its semantics remain valid.**

1. Keep `Trainer.run()` as the shared training mainline. Both `sm-train` and `smHOI-track` must reuse the original training scheduler, UTD update cycle, distributed synchronization, logging and checkpoint lifecycle.

2. Do not create duplicate implementations of the original Trainer, ReplayBuffer, World Model training loop, Actor BPTT optimizer or Critic TD(lambda). Extend or compose existing mechanisms instead.

3. Prefer HOI-specific modules and adapters under `src/supermimic/hoi/`. Modify shared modules only when a concrete HOI requirement cannot be supported through an HOI-local adapter or composition.

4. Every shared training-core modification must have a specific justification in the current stage's IMPLEMENTATION_SPEC.md, identify the smallest necessary interface change, and include regression evidence that the original robot-only path remains unchanged.

5. Preserve the default behavior and checkpoint compatibility of `sm-train`. Avoid widespread task-specific conditionals throughout shared training code. Prefer separate HOI configuration and component construction with narrowly defined shared interfaces.

6. Production training code should remain modular and avoid duplicated algorithms or unnecessary abstraction layers. Extensive tests, physical calibration scripts, diagnostics and evaluation tools are encouraged and are not subject to this restriction.

7. Every stage report must include a brief **Training Framework Reuse Audit** listing reused original components, modified shared files and their justification, any duplicated functionality, and robot-only regression results.

Do not impose arbitrary line-count limits. The acceptance criterion is minimal architectural disruption, maximum reuse of the original learning algorithm, and preservation of existing robot-only behavior.

---

# 2. Project objective and non-negotiable constraints

## 2.1 Objective

Implement **Reference-Conditioned, World-Model-Based Humanoid–Object Interaction Tracking** in the existing SuperMimic system.

The target learning loop is:

Reference motion (robot + object + binary desired-contact label)
    ↓
Physical humanoid-object scene and measured simulator diagnostics
    ↓
Action-aligned joint robot-object Replay
    ↓
Joint Robot–Object World Model
    ↓
Reference-conditioned differentiable geometric contact reward
    ↓
Joint robot-object imagination
    ↓
Actor BPTT + original Critic/TD(lambda)

The joint World Model learns robot and object physical dynamics:

\[
\hat{s}_{t+1}^{joint}=f_\theta(s_t^{joint},a_t^{exec},\xi,\text{support context})
\]

The reference motion independently supplies the binary desired-contact condition:

\[
c_t^{ref}=\text{object\_contact}[t]\in\{0,1\}
\]

The binary label directly gates the geometric contact reward. It is not predicted by the World Model and is not inferred from measured force.

The interaction reward uses differentiable EEF-to-Box-target geometry. Actual MuJoCo contact and force measurements remain separate physical diagnostics.

This is not:
- a force-prediction model;
- a contact classifier;
- an onset/release event detector;
- a phase-conditioned reward system;
- a contact bonus disconnected from object tracking;
- an AMP-only solution;
- a separate HOI trainer copied from SuperMimic;
- an object pose predictor disconnected from robot dynamics.

## 2.2 Non-negotiable constraints

1. Preserve original sm-train behavior and defaults.
2. Add/maintain a separate smHOI-track entry.
3. Reuse the original Trainer.run() and SuperMimic optimization structure.
4. Preserve original robot-only tracking ability and verified checkpoint compatibility.
5. Preserve the accepted Part B reference byte-for-byte.
6. Preserve the M6.5 physical asset profile and historical audit results.
7. Keep HOI-specific logic primarily under src/supermimic/hoi/.
8. Do not rewrite the original Replay, WM, imagination or Trainer wholesale.
9. Do not introduce hard-coded phase one-hot variables or additional contact/phase classifiers.
10. Use the original binary reference object_contact[t] directly as the contact-reward gate.
11. Keep reference desired contact, actual MuJoCo contact and geometric EEF-target proximity as distinct signals.
12. Do not use reference object pose as the current imagined object state.
13. Do not introduce Contact Prediction Head, Force Prediction Head, force regression, force normalization or force-based Actor reward.
14. Preserve physical force/contact measurements as diagnostics where already supported; do not use them as final WM supervision targets.
15. Do not begin Actor training until the joint robot-object WM has passed P7-A validation and physical asset limitations have an explicit disposition.

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

| Quantity | Intended dimension | Meaning |
|---|---:|---|
| Robot physical state | 71 | Original SuperMimic state |
| Object physical state | 13 | Position 3 + quaternion 4 + linear velocity 3 + angular velocity 3 |
| Joint physical state | 84 | Robot + object |
| Robot canonical | 71 | Preserve original semantics |
| Object canonical | 15 | Relative position 3 + relative orientation 6D + relative linear velocity 3 + angular velocity 3 |
| Joint canonical | 86 | 71 + 15 |
| Support-relative context | 18 | Time-varying physical context, recomputed from current joint physical state |
| Action | 29 | G1 executed action |
| Existing domain context | 33 | Original episode context semantics |
| Robot residual output | 70 | Original robot dynamics |
| Object residual output | 12 | Object integration residual |
| Physical dynamics output | 82 | 70 + 12 |
| Final WM input | 166 | Canonical86 + support18 + action29 + xi33 |
| Final WM output | 82 | Robot70 + object12 |

No contact probability/logit or force output is part of the final WM state or output.

Support-relative context remains separate from the episode-constant 33D xi. Preserve the verified support representation and recompute it from the current physical state.

## 4.2 Time

Known current environment settings:
- physics timestep: 0.005 s (200 Hz);
- decimation: 4;
- control timestep: 0.020 s (50 Hz);
- one action spans four physics substeps.

Verify these settings in the actual checkout.

The reference motion runs at 50 Hz. Contact reward uses the binary label aligned to the exact reference frame being evaluated. Do not add a separate temporal lookahead, phase schedule, onset/release detector or smoothed contact signal.

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
## 4.5 Three distinct interaction signals

Always keep separate:

1. **contact_intent_ref:** global binary desired-contact label from Part B object_contact[t]. This is the direct condition that gates the geometric contact reward.
2. **contact_actual_sim:** actual left/right hand-box contact measured independently in MuJoCo. This is used for physical diagnostics and evaluation.
3. **contact_geometry:** differentiable EEF-to-Box-target proximity computed from current robot and imagined/physical Box states. This supplies the Actor's geometric reward gradient.

Do not equate these signals.

Part B provides one global binary desired-contact label, not independent left/right labels. Left/right target participation must be decided from the audited asset/reference contract, not inferred by inventing per-hand labels.

P4.1 must establish whether the reference/task requires bilateral physical contact and document the applicable interval/evidence. Only that finding permits bilateral AND to serve as the task-required contact aggregate. Otherwise preserve and report any-hand (OR) and bilateral (AND) measurements separately, without assuming either is the uniquely correct interpretation of the global label. Independent left/right measurements are always retained; neither physical aggregation changes the target-average proximity reward in P4.4. See P6.2–P6.3 and P8.3 for evaluation.

The final architecture has no predicted contact confidence or force-prediction signal.

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

Unit: N. Retain this interval mean as a physical diagnostic; it is not an active WM prediction target or Actor reward input.

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

Keep raw substep measurements available for diagnostics. Do not sum force values and call the result force. Peak force is a separate physical diagnostic, not a prediction target or Actor reward input.

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

# 8. P3 — Joint Robot–Object World Model Architecture

## What P3 means

P3 defines the final joint robot-object World Model architecture. The model predicts robot and Box physical dynamics jointly.

Reference contact intent is already provided by the reference motion. It is not a learned World Model output.

The final architecture contains no Contact Prediction Head and no Force Prediction Head.

## Prerequisites

- P2 implementation and Replay contracts are preserved.
- P0 support-context and coordinate findings are reconciled.
- Existing state, canonicalization and integration code is inspected.

## P3.1 Final state representation

Preserve the original Robot StateSpec and exact feature order.

Use:
- robot physical state: 71D;
- object physical state: 13D;
- joint physical state: 84D;
- robot canonical: 71D;
- object canonical: 15D;
- joint canonical: 86D;
- support-relative context: 18D;
- action: 29D;
- episode context xi: 33D.

The support context remains separate from xi33 and is recomputed from the current physical state.

## P3.2 Object canonicalization

Preserve the audited object canonical representation:

\[
\phi_o=[
p_{o,rel}^{h}(3),
R_{o,rel}^{6D}(6),
v_o^{h}(3),
\omega_o^{body}(3)]
\]

where relative position/orientation and linear velocity are expressed in the robot heading frame, and angular velocity frame is explicitly documented.

Use the existing verified frame conventions. Test invariance and round-trip behavior.

## P3.3 Object integration

Preserve the existing separate object integration path with a constant-velocity prior.

The 12D residual contains:
- linear velocity residual 3;
- angular velocity residual 3;
- position correction 3;
- orientation correction 3.

Use the correct root-origin/COM velocity relationship. Do not directly regress quaternion components. Verify quaternion multiplication against the chosen angular-velocity frame.

Zero residual must produce the documented constant-velocity prediction.

## P3.4 Support-relative context

Preserve the accepted support-relative context representation and its evidence.

Do not append support geometry to xi33. Recompute support context from each current or imagined physical state.

If the actual implementation contradicts the accepted support audit, report the discrepancy and obtain approval before changing the contract.

## P3.5 Final joint WM architecture

Use one shared DynamicsMLP trunk with:
- robot residual head: 70D;
- object residual head: 12D.

Final input:

\[
86+18+29+33=166
\]

Final output:

\[
70+12=82
\]

Only these 82 physical residual values are integrated into the 84D joint physical state.

No Contact Prediction Head, contact logit, contact probability, contact BCE, Force Prediction Head or force regression is permitted.

The reference contact label is not a WM input or prediction target.

Reuse the original DynamicsMLP and World Model training machinery. Preserve the original robot output semantics and object constant-velocity prior.

Do not create two independent WMs that cannot represent interaction coupling.

## P3.6 Reference contact semantics

The accepted Part B object_contact[t] is a known binary reference condition.

It is retrieved by the reference motion interface at the frame being evaluated. It is used later by P4/P5 to gate geometric contact reward.

Do not:
- predict the next contact label;
- classify ON/OFF states;
- construct onset/release training windows;
- add a phase classifier;
- use force as a substitute label.

## P3.7 Checkpoint migration and architecture cleanup

The current 167D/83D Contact-Head implementation is transitional and is not the final architecture.

Perform explicit named mappings:
- Robot-only checkpoint to the joint robot-object model.
- Historical 166D/88D force checkpoint, if reused, maps only the verified robot/object input columns and 82 physical output rows; discard the six force-output rows explicitly.
- Transitional 167D/83D Contact checkpoint must not be treated as a final model checkpoint.

Do not use strict=False as a migration strategy. Do not migrate incompatible optimizer state. Preserve original checkpoints.

The final model must strictly validate:
- input 166D;
- output 82D;
- physical state 84D;
- canonical state 86D;
- support context 18D;
- action 29D;
- xi 33D.

## P3.8 Verification scope and deferred training

The dedicated P3 contact-boundary diagnostic is explicitly skipped.

Do not execute:
- the special 32-transition onset/release capture;
- contact-specific balanced train/validation episode construction;
- bounded Contact-Head fitting;
- binary contact event detection;
- contact prediction readiness gates.

P3 architecture cleanup may run shape, migration, state-integration and original robot-only regression tests. These are software/contract checks, not learned predictive performance.

Joint robot-object WM training and predictive validation are deferred to P7-A. P3 remains historically PARTIAL; do not rewrite its existing reports or failed capture evidence.

## P3 deliverables

- final 166D/82D joint WM architecture;
- removal of the active Contact Prediction Head and contact BCE path;
- explicit checkpoint migration tests;
- original robot-only regression tests;
- updated P3 architecture report documenting the skipped diagnostic and deferred predictive validation.

## P3 acceptance criteria

- final input/output dimensions are 166D/82D;
- physical state remains 84D;
- canonical state remains 86D;
- support-relative context remains 18D and separate from xi33;
- robot/object residual semantics and integration are correct;
- no active Contact Prediction Head or Force Prediction Head remains;
- transitional checkpoints are not silently reused;
- strict checkpoint loading works;
- original robot-only WM regression passes.

P3 stops after architecture cleanup and reporting. Do not train the joint WM here. Joint WM training/validation belongs to P7-A.


# 9. P4 — Reference-Conditioned Physical Interaction Reward

## What P4 means

P4 defines and validates the differentiable physical interaction guidance used by the original SuperMimic Actor training pipeline.

The training reward combines:
1. The original binary reference contact condition.
2. Differentiable EEF-to-object-target proximity guidance.
3. The existing robot and object tracking rewards.

Actual MuJoCo hand-box contact measurements are retained as an independent physical evaluation signal. They are not used as differentiable Actor reward inputs.

The design follows the position-proximity component of HDMI's interaction reward while retaining the original SuperMimic joint robot-object World Model and differentiable imagination architecture.

P4 does not introduce:
- Contact Prediction Head;
- Force Prediction Head;
- Binary Contact Decoder;
- Contact BCE or force regression;
- Additional Actor/Critic observation dimensions;
- Additional World Model state/output dimensions;
- Force-based Actor reward;
- Contact phase classifier, onset/release detector or future-contact predictor.

P4 defines and tests the reward only. Imagination integration belongs to P5; Actor training belongs to P7-B after P7-A validation.

## Prerequisites

- P1 physical scene/state access is available.
- P2 reference/action/physical-state alignment is available.
- Accepted Part B reference remains unchanged.
- M6.5 asset and contact-target audit evidence is read.
- Final P3 physical-state contract is 84D.
- Original SuperMimic robot/object tracking reward implementations are verified.

## P4.1 Contact target and EEF audit

Before implementing the final reward, inspect:

- `data/assets/hdmi_bread_box/asset_manifest_m6_5.json`;
- `docs/hoi/asset_audit/contact_points.csv`;
- `src/supermimic/hoi/reference.py`;
- `src/supermimic/hoi/state.py`;
- native G1 FK implementation and wrist/EEF transforms;
- original HDMI contact-target and EEF conventions;
- native G1 hand collision geometries and the seven Box collision geometries.

The current candidate asset contract records:

~~~python
target_root_m = [
    [0.0,  0.17, 0.14],  # left
    [0.0, -0.17, 0.14],  # right
]

eef_offset_wrist_m = [
    [0.05, 0.0, 0.0],   # left
    [0.05, 0.0, 0.0],   # right
]
~~~

Verify:

- left/right ordering;
- Box-root coordinates;
- wrist-local EEF offset;
- FK frame and transform convention;
- target transformation under current Box rotation;
- EEF-target distances throughout the reference;
- correspondence between EEF markers, native hand collision capsules and Box collision surfaces;
- physical reachability and consistency of the candidate contact targets.

The EEF marker is a kinematic reference point and must not automatically be interpreted as the physical hand collision surface.

The existing audit records both target slots as desired during the same global contact interval for the current 840-frame reference. This supports a two-target candidate but does not establish independently annotated left/right reference labels or physically validated bilateral contact.

Record separately (a) the valid target set for the P4.4 proximity average and (b) whether the reference/task evidence establishes required bilateral physical contact, with its applicable reference interval. Two candidate targets alone do not establish a bilateral requirement. P6/P8 may use bilateral AND as the task-required aggregate only when this audit establishes that requirement. If it remains unestablished, retain independent left/right measurements and report any-hand and bilateral measurements separately under P6.2–P6.3; do not invent per-hand labels or silently select a global aggregate.

The original `object_contact[t]` remains the only reference contact label. Do not create new per-hand reference labels from assumptions about bilateral manipulation.

The current M6.5 manifest has `training_ready=false` and `M6_5_NO_GO_M7`. Preserve its physical validation failures. Marker alignment is not proof of collision/contact validity.

Deliverable:

`docs/hoi/force_contact/p4/CONTACT_TARGET_AUDIT.md`

## P4.2 Reference binary contact condition

Use the original binary reference label:

\[
c_t^{ref}=\text{object\_contact}[t]\in\{0,1\}
\]

The label is retrieved from the accepted reference at the exact frame being evaluated.

The same binary condition applies throughout the complete reference:

- `1` means the reference expects object interaction/contact.
- `0` means the reference does not expect object interaction/contact.

Do not treat ON and OFF as separate training systems or separate temporal classifiers.

Do not:
- shift or modify the original binary signal;
- introduce future-contact lookahead;
- smooth or predict the reference label;
- create independent left/right reference labels without source annotation;
- add onset/release classifiers or phase encoding.

Keep the original reference NPZ byte-for-byte unchanged.

## P4.3 Differentiable contact geometry

For each audited target i:

\[
p_{target,i}^{world}
=
p_{box}^{world}
+
R_{box}p_{target,i}^{box}
\]

\[
p_{EEF,i}^{world}
=
p_{wrist,i}^{world}
+
R_{wrist}p_{offset,i}^{wrist}
\]

\[
d_{t,i}
=
\left\|
p_{EEF,i}^{world}
-
p_{target,i}^{world}
\right\|_2
\]

Use the current physical or imagined Box pose, not a reference Box pose substituted for the current imagined state.

Use differentiable Torch operations.

No MuJoCo contact query, measured force, contact threshold operation or detached physical sensor result is permitted inside the differentiable Actor reward.

The EEF-target distance is a proximity measure, not a direct measurement of collision or physical contact.

P4.1 must verify that the selected EEF marker and target conventions are physically meaningful relative to the native collision geometry. Do not assume that zero EEF-target distance means zero collision-surface distance.

## P4.4 HDMI-style reference-conditioned proximity reward

Use the position-proximity component of HDMI's interaction reward.

For each valid audited contact target:

\[
r_{contact,i,t}
=
c_t^{ref}
\cdot
\exp\left(
-\frac{d_{t,i}}{\sigma_{pos}}
\right)
\]

Aggregate the valid contact targets:

\[
r_{proximity,t}
=
\frac{1}{N_c}
\sum_{i\in\mathcal E}
r_{contact,i,t}
\]

where:

- \(\mathcal E\) is the audited set of valid contact targets;
- \(N_c=|\mathcal E|\);
- \(\sigma_{pos}>0\) is a configurable distance scale;
- \(c_t^{ref}\) is the original binary reference contact condition.

When `object_contact[t] = 1`, the reward encourages the corresponding EEFs to approach their Box-local targets.

When `object_contact[t] = 0`, this proximity reward is zero. This does not imply that actual physical contact is absent; unintended contact is evaluated independently using the actual simulator contact measurements in P6 and P8.

Do not add force magnitude, force threshold, force saturation or force-based confidence to this differentiable reward.

Do not add a separate non-contact repulsion reward in P4. Such a term requires independent evidence that unintended proximity/contact is a systematic problem and must be proposed as a separate research-level amendment.

Do not invent final values for `sigma_pos` or reward weights. Determine candidate values from audited reference-distance distributions and controlled reward-scale experiments.

If a target is invalid or physically meaningless, do not silently retain it in the average. Record the evidence and obtain approval for a target-set change.

## P4.5 Combined tracking objective

Keep the original robot and object tracking rewards:

\[
r_{HOI}
=
\lambda_r r_{robot}
+
\lambda_o r_{object}
+
\lambda_c r_{proximity}
+
r_{regularization}
\]

The original robot tracking reward remains active throughout the task.

Object tracking remains active throughout the task.

The proximity reward is gated by the original reference binary contact label.

Do not automatically reduce robot tracking weight when contact becomes active.

Do not replace object tracking with contact reward.

Reward coefficients must be selected from actual reward-scale measurements and documented. Do not invent final weights.

## P4.6 Physical contact is an independent measurement

P4 must explicitly distinguish:

1. Reference contact intent:
   `object_contact[t]`.
2. Differentiable geometric proximity:
   `d_t,i`.
3. Actual MuJoCo hand-box contact:
   independently measured filtered contact force and contact state.

Actual hand-box contact labels are derived from the P1/P2 physical sensor pipeline using calibrated thresholds and valid measurement intervals.

They are not inputs to the differentiable proximity reward.

They are not WM prediction targets.

They are not additional Actor/Critic observations.

They are not used to modify the original reference NPZ.

P4 may reuse existing P1/P2 measurement utilities for deterministic diagnostic experiments, but full reference-versus-actual contact validation belongs to P6.

## P4.7 Unit tests

Test:

- exact binary reference gating;
- proximity reward is zero when reference contact is OFF;
- proximity reward is positive and increases monotonically as audited EEF-target distance decreases when reference contact is ON;
- left/right target ordering;
- rotated Box target transforms;
- current imagined Box pose is used;
- batched environment independence;
- gradients to EEF joint coordinates;
- gradients to imagined Box pose;
- finite outputs and gradients;
- independence from MuJoCo sensor data and measured force;
- original robot-only tracking reward regression.

These are mathematical and implementation tests, not proof of physical contact success.

## P4 deliverables

- audited contact-target contract;
- pure-Torch contact geometry and proximity reward functions;
- unit tests;
- reward-scale and reference-distance audit;
- `docs/hoi/force_contact/p4/REPORT.md`;
- outputs under `outputs/hoi/contact_supervision/reward_validation/`.

## P4 acceptance criteria

- original reference binary label is used without modification;
- reference-frame alignment is exact;
- no additional temporal classifier or phase system exists;
- target transforms and EEF FK are correct;
- EEF marker and collision geometry correspondence is documented;
- proximity reward has the intended gradient;
- valid target set is explicitly justified;
- original robot/object tracking remains intact;
- no measured force or actual contact label enters differentiable Actor reward;
- no new observation or World Model dimensions are introduced;
- original robot-only regression passes.

P4 stops after reporting. Do not integrate full imagination.


---


# 10. P5 — Differentiable Joint Imagination and Actor BPTT

## What P5 means

P5 integrates the joint robot-object World Model and the approved reference-conditioned proximity reward into the original SuperMimic imagination pipeline.

The Actor learns to track the humanoid and object while approaching audited Box-local contact targets during reference-defined interaction intervals.

Actual MuJoCo contact measurements remain outside differentiable imagination and Actor BPTT.

No new Contact Decoder, Contact Prediction Head, Force Prediction Head or additional observation/state dimensions are introduced.

## Prerequisites

P4 is approved; final 84D/86D/82D contract is fixed; contact target audit and proximity reward tests pass; original imagination and Critic interfaces are verified.

A trained P3 model is not required to implement imagination interfaces or run controlled gradient tests. Use an explicitly identified test model where necessary.

## P5.1 Joint imagined state

Maintain only robot physical state 71D and object physical state 13D.

\[
s_t^{joint}\in\mathbb R^{84}
\]

There is no additional contact-probability state or force state.

Preserve the existing canonical86, support-context18, action29 and xi33 contracts.

## P5.2 Reference lookup and alignment

At each imagined step, retrieve corresponding reference robot state, object state and original binary contact label using the original motion-library frame progression.

The contact label must correspond to the state at which its reward is evaluated.

For an action transition:

\[
s_t\xrightarrow{a_t}\hat{s}_{t+1}
\]

evaluate the next-state reward against reference frame t+1:

\[
r_{proximity,t+1}
=
c_{t+1}^{ref}
\cdot
r_{geometry}(\hat{s}_{t+1})
\]

This is direct per-frame reference alignment, not future lookahead or a new time-dependent classifier.

Do not introduce future contact prediction, reuse stale Box-relative geometry, clamp invalid clip indices or treat terminal padding as valid reference data.

## P5.3 Joint World Model rollout

At each imagined step:

1. Read current imagined physical84.
2. Compute canonical86.
3. Compute support-relative context18.
4. Obtain Actor executed action29.
5. Supply episode context xi33.
6. Predict robot/object residual82.
7. Integrate joint physical state84.
8. Evaluate robot/object tracking and reference-conditioned proximity rewards.

Only physical robot/object state is integrated.

## P5.4 Differentiable geometry and reward

At every imagined step recompute:

- robot FK;
- EEF positions;
- current imagined Box pose;
- Box-local target transforms;
- EEF-target distances;
- exact reference binary contact gate;
- proximity reward.

Use the current imagined Box pose.

Do not use NumPy conversion, detached imagined Box states, detached FK outputs, MuJoCo contact queries or measured simulator force as an Actor gradient source.

Actual physical contact matching is evaluated independently in P6/P8.

## P5.5 Terminal and clip-end semantics

Preserve separate handling for physical robot failure, physical object failure, time-limit truncation and natural reference clip-end.

A natural clip-end is not a physical failure. No repeated terminal-frame proximity reward is permitted.

Preserve original survival and Critic bootstrap semantics.

## P5.6 Actor BPTT gradient checks

Verify:

- action gradients through predicted robot state to robot tracking reward;
- predicted object state to object tracking reward;
- predicted robot state/FK to EEF proximity reward;
- predicted object state to Box target geometry and proximity reward.

The World Model can be frozen while gradients through it to Actor remain intact.

No gradient is expected or required through MuJoCo contact sensors.

## P5.7 Eager/compiled parity

Compare:

- robot/object state;
- EEF positions;
- Box target positions;
- contact distances;
- reference binary gate;
- proximity reward;
- total reward;
- Actor action gradients.

Compilation must preserve reward and gradient semantics.

## P5 deliverables

- HOI imagination integration;
- minimal shared rollout/compiled/policy-trainer changes;
- gradient and eager/compiled parity tests;
- `docs/hoi/force_contact/p5/REPORT.md`.

## P5 acceptance criteria

- correct 84D imagined state and reference frame alignment;
- proximity reward activates only when reference contact is true;
- reference OFF produces zero proximity reward;
- robot/object state gradients and proximity geometry gradients reach Actor;
- no unintended detach;
- correct clip-end/bootstrap behavior;
- eager/compiled parity;
- no MuJoCo force/contact sensor dependency inside differentiable imagination;
- original robot-only imagination regression passes.

P5 stops after reporting. Do not start P6 or Actor training.

---


# 11. P6 — Physical Contact and Proximity Validation

## What P6 means

P6 validates the approved differentiable proximity reward and independently evaluates whether simulated physical contact agrees with the original reference binary contact label.

The central physical validation is a frame-aligned comparison:

\[
c_t^{ref}
\quad\text{versus}\quad
c_t^{sim}
\]

for every valid evaluated reference timestep, including both reference contact ON and OFF frames. Here \(c_t^{sim}\) denotes an explicitly named physical aggregate under P6.2–P6.3, not an assumed bilateral global label.

P6 distinguishes:

1. Reference contact intent.
2. Differentiable EEF-to-object-target proximity.
3. Actual MuJoCo hand-box physical contact.

These signals must never be treated as interchangeable.

P6 does not introduce a learned contact classifier, force prediction model, force-based Actor reward or new observation dimensions.

## Prerequisites

P5 is approved.

The P1/P2 physical contact measurement pipeline is available and calibrated.

The accepted Part B reference and M6.5 physical asset evidence are available.

Tests requiring a failed/unverified asset gate are diagnostic, not accepted task evidence.

## P6.1 Geometric reward sensitivity

Use deterministic robot/object states and controlled action perturbations.

Evaluate:

- EEF-target distance;
- reward response to joint and Box-pose perturbations;
- gradients with respect to action/state;
- left/right target contributions;
- reference ON/OFF gating;
- reward scale across the complete reference.

Use finite differences of geometric reward only to verify local numerical behavior.

This is not a comparison against predicted or measured force gradients.

## P6.2 Actual MuJoCo hand-box contact labels

Use the existing P1/P2 filtered hand-box contact measurements.

For each hand i and policy timestep t, aggregate the four physics substeps.

Let:

\[
F_{i,t,k}
=
\left\|
f_{i,t,k}^{hand-box}
\right\|_2
\]

where k indexes the physics substep within one policy step.

Use the calibrated contact threshold \(\tau_i\):

\[
c_{i,t,k}^{sim}
=
\mathbb{1}[F_{i,t,k}>\tau_i]
\]

Derive the per-policy-step contact occupancy:

\[
o_{i,t}
=
\frac{1}{4}
\sum_{k=1}^{4}
c_{i,t,k}^{sim}
\]

Derive the binary per-hand contact state using a calibrated occupancy criterion:

\[
c_{i,t}^{sim}
=
\mathbb{1}[o_{i,t}>\rho_i]
\]

The force threshold and occupancy criterion must be justified using P1/P2 calibration evidence. Do not invent or copy thresholds from another simulator.

Preserve independent left/right contact measurements and both derived physical measurements:

\[
c_t^{sim,any}=c_{L,t}^{sim}\lor c_{R,t}^{sim}
\]

\[
c_t^{sim,bilateral}=c_{L,t}^{sim}\land c_{R,t}^{sim}
\]

Use bilateral AND as the task-required aggregate only when P4.1 establishes that bilateral contact is required by the reference/task, citing its evidence and applicable interval. Computing the bilateral measurement for diagnostics does not establish that requirement.

Otherwise report any-hand and bilateral measurements separately; do not silently define a single global contact label as bilateral AND or choose the aggregate that maximizes agreement. Retain both measurements even when a bilateral requirement is established, so that unilateral unintended contact during reference OFF is not hidden by AND aggregation.

These are derived physical evaluation metrics, not newly annotated reference labels. They do not change the P4.4 reward's average over audited targets.

Do not assume that the original global `object_contact[t]` provides independent left/right reference annotations.

## P6.3 Reference-versus-actual binary contact agreement

Compare each separately named physical aggregate from P6.2 with the original reference binary label at every valid evaluated timestep:

\[
c_t^{ref}
=
\text{object\_contact}[t]
\]

\[
m\in\{\mathrm{any},\mathrm{bilateral}\}
\]

Compute:

\[
A_{contact}^{m}
=
\frac{1}{N}
\sum_{t\in\mathcal V}
\mathbb{1}
[c_t^{ref}=c_t^{sim,m}]
\]

where \(\mathcal V\) contains valid, correctly aligned physical evaluation timesteps, \(N=|\mathcal V|\), with the same validity mask for both aggregates. Name the aggregate for every reported metric. Only interpret bilateral agreement as task-required agreement within the scope established by P4.1; otherwise report both comparisons without claiming a unique task-required aggregate.

Report the full binary confusion matrix for each aggregate:

- True Positive: reference ON, actual contact ON.
- True Negative: reference OFF, actual contact OFF.
- False Positive: reference OFF, actual contact ON.
- False Negative: reference ON, actual contact OFF.

Report:

- contact accuracy;
- precision;
- recall;
- F1 score;
- balanced accuracy;
- left-hand contact ratio;
- right-hand contact ratio;
- any-hand contact ratio;
- bilateral contact ratio;
- contact occupancy and duration.

Evaluate left/right contact ratios, occupancy and duration independently, including separate reference ON/OFF breakdowns. These are measurements conditioned on global reference intent, not accuracy against independently annotated per-hand ground truth.

Evaluate both ON and OFF frames. Do not calculate agreement only over reference-active frames.

Preserve the original reference labels and do not shift, smooth or relabel them to improve agreement.

If a reference segment has ambiguous or invalid physical alignment, report it explicitly. Do not silently remove mismatching frames.

## P6.4 Physical contact and object behavior

For reference ON intervals, report:

- actual left/right hand-box contact ratio;
- any-hand and bilateral contact ratios, with bilateral requirement interpreted only under P4.1 evidence;
- EEF-target distance;
- hand-box force magnitude and impulse;
- contact persistence;
- Box position/orientation tracking;
- Box slip and stability.

For reference OFF intervals, report:

- unintended left/right hand-box contact;
- unintended bilateral contact;
- Box-support contact separately;
- object stability and unintended object motion.

Keep hand-box, box-support and robot-body-box contacts separate.

Measured force is physical diagnostic data only. Do not introduce force-based Actor rewards.

## P6.5 Collision and penetration diagnostics

MuJoCo collision constraints do not guarantee mathematically zero geometric penetration at every timestep.

Use MuJoCo contact records to inspect actual collision behavior.

For relevant hand-box contact pairs, record:

- contact geom identifiers;
- contact position;
- contact frame;
- signed contact distance;
- contact count;
- minimum contact distance;
- maximum negative contact distance;
- penetration duration and frequency.

Interpretation:

- positive contact distance: separated surfaces within the contact detection margin;
- zero contact distance: geometric touching;
- negative contact distance: geometric overlap/penetration.

Distinguish small transient numerical penetration from persistent or physically significant penetration.

Do not add an SDF model, differentiable penetration reward or new collision geometry by default.

If significant penetration is observed, investigate:

- collision geometry and geom transforms;
- contact parameters such as solref and solimp;
- simulation timestep and substeps;
- solver convergence and iteration limits;
- high-speed motion and possible tunneling.

Any proposed changes to collision geometry or solver configuration require evidence and a separate implementation-level or research-level amendment according to Section 1.3.3.

## P6.6 Asset and support validation

Reconcile unresolved M6.5 gates.

Verify:

- enabled native hand collision geometries;
- Box collision geometry;
- hand-box sensor filtering;
- support/contact separation;
- contact force sign and frame;
- Box stability during carrying;
- post-release stability.

Preserve failed historical results.

Do not replace the accepted asset profile without explicit approval.

## P6 deliverables

- deterministic geometric reward-gradient diagnostics;
- frame-aligned reference-versus-actual binary contact report;
- independent left/right and separately reported any-hand/bilateral contact metrics;
- contact occupancy and force diagnostics;
- collision/penetration diagnostics;
- asset/support gate reconciliation;
- `docs/hoi/force_contact/p6/REPORT.md`;
- outputs under `outputs/hoi/contact_supervision/contact_validation/`.

## P6 acceptance criteria

- proximity reward gradients are finite and directionally consistent;
- actual MuJoCo contact is independently measured and correctly filtered;
- original reference binary labels are preserved;
- reference and actual contact are compared across all valid ON/OFF timesteps;
- left/right contact is reported independently;
- any-hand and bilateral measurements are preserved separately; bilateral AND is used as the task-required aggregate only where P4.1 establishes the requirement;
- box-support contact is not confused with hand-box contact;
- penetration and solver limitations are explicitly reported;
- no force model or force-based Actor reward is introduced;
- physical asset/support limitations remain visible;
- a clear go/no-go decision is reported for integrated policy training.

P6 stops after reporting. Do not start P7 or training.


---


# 12. P7 — Joint World Model Training and Integrated SuperMimic Policy Training

## What P7 means

P7 completes deferred P3 joint robot-object World Model training and integrates the approved reference-conditioned proximity reward into the original SuperMimic training mainline.

P7 contains two sequential gates:

- P7-A: train and validate joint robot-object WM.
- P7-B: train Actor/Critic using differentiable joint imagination and the approved proximity reward.

Actual MuJoCo contact measurements remain physical evaluation data. They are not training targets for the active World Model or Actor reward.

No Contact Prediction Head, Force Prediction Head, Binary Contact Decoder, Contact BCE or force regression is introduced.

## Prerequisites

P4, P5 and P6 are approved; M6.5 physical asset limitations are resolved or explicitly approved for a bounded research experiment; final P3 contract is 166D/82D; reference and Replay provenance are verified; each training experiment is explicitly authorized.

## P7.1 Separate training entry point

Provide `smHOI-track`.

Reuse original Trainer.run(), training scheduler, WM optimizer/update structure, Actor BPTT, Critic/TD(lambda), logging and checkpoint lifecycle.

Do not change `sm-train` defaults or create a separate PPO implementation.

## P7.2 P7-A: Joint robot-object WM training

Train 166D/82D joint dynamics using measured robot/object physical states, executed action, episode context and verified support-relative context.

Use original WM prediction loss for robot/object dynamics.

Existing P2 measured force/contact archives may remain for diagnostics and physical evaluation, but force labels must not enter WM loss.

Reference `object_contact` remains outside the WM prediction target.

Do not introduce Contact Prediction Head, Contact BCE, Force Prediction Head, force regression, force normalization or force-specific readiness.

## P7.3 Deferred P3 model validation

Evaluate joint WM before Actor BPTT.

Report robot/object one-step and multi-step prediction, object position/orientation error, linear/angular velocity error, constant-velocity and frozen initialized-model baselines, rollout drift and episode/clip-aware validation.

Use complete Replay windows and correct terminal/clip masks.

Distinguish training fit from held-out prediction performance.

P7-A must pass before P7-B.

## P7.4 Checkpoint migration

Use explicit named mappings from robot-only to joint robot-object model and into final 166D/82D schema.

No Contact Head or Force Head migration into active architecture. Do not reuse incompatible optimizer state.

Persist state/feature schema, action/xi mappings, support-context schema, reference/asset hashes, simulator/dependency versions, physics/control timestep, reward configuration, training seed and validation metrics.

Never overwrite original checkpoints.

## P7.5 Actor and Critic integration

Preserve the original Actor and Critic observation contracts and dimensions.

The proximity reward is calculated from existing imagined robot/object states and differentiable FK. It does not require new Actor/Critic observations or additional physical state dimensions.

Do not introduce privileged force information.

Preserve original Actor input dimensions, Critic input dimensions, TD(lambda), survival and bootstrap semantics.

Verify original Actor output parity and checkpoint compatibility.

## P7.6 P7-B: Staged short training and reward ablation

Run separately authorized short experiments after P7-A has passed.

Use controlled experiments with identical reference, asset, simulation settings, random seeds and training budgets wherever applicable.

1. **Experiment A — Tracking Baseline**
   - Robot tracking reward ON.
   - Object tracking reward ON.
   - Proximity reward OFF.

2. **Experiment B — Reference-conditioned Proximity**
   - Preserve all tracking rewards.
   - Enable the approved reference-conditioned proximity reward.
   - Keep the original binary reference label unchanged.

3. **Experiment C — Proximity Configuration Ablation**
   - Preserve Experiment B.
   - Evaluate controlled candidate proximity distance scales and reward weights.
   - Do not introduce additional reward branches or force supervision.

For every experiment, record reward components, reward scales, policy gradients, training stability and checkpoint provenance.

Evaluate selected checkpoints using the P6 physical contact metrics.

The physical contact agreement is an evaluation outcome, not a training reward.

Each experiment has its own configuration, checkpoint and report.

Do not launch long training or automatic sweeps without explicit approval.

## P7.7 Verify actual parameter updates

Report robot WM, object WM, Actor and Critic parameter updates, gradient norms, optimizer steps, loss/reward components and checkpoint save/resume behavior.

Verify that the proximity reward contributes gradients through imagined robot/object geometry to the Actor.

There must be no Contact Head, Force Head or additional contact-related observation/state update.

## P7 deliverables

- `smHOI-track`;
- final joint WM configuration;
- Actor/Critic integration;
- checkpoint migration tests;
- joint WM training/validation reports;
- proximity reward ablation reports;
- short-run logs/checkpoints;
- physical contact evaluation results;
- `docs/hoi/force_contact/p7/REPORT.md`;
- outputs under `outputs/hoi/contact_supervision/training/`.

## P7 acceptance criteria

- original `sm-train` behavior unchanged;
- joint WM uses 166D/82D;
- robot/object prediction validation passes agreed criteria;
- Actor/Critic receive actual optimizer updates;
- proximity reward influences Actor gradients;
- no measured MuJoCo force/contact sensor dependency inside differentiable Actor reward;
- checkpoint save/resume works;
- no NaN/Inf;
- no unintended force/contact prediction branch;
- original robot-only regression passes;
- physical contact agreement and task performance are reported for each selected experiment.

P7 stops after reporting. Do not run full task evaluation until approved.

---


# 13. P8 — Full 840-frame Pick-Carry-Place-Release Evaluation

## What P8 means

P8 evaluates the selected trained policy on the complete physical task and whether the humanoid tracks reference robot/object motion while realizing reference-conditioned physical interaction.

It is not another training stage.

## Prerequisites

P7 is approved, user selects checkpoint, asset/physical scene are approved for evaluation, and exact reference/asset hashes are recorded.

## P8.1 Evaluation setup

Record checkpoint/reference paths and SHA256, asset manifest hash, simulator/dependency versions, physics timestep/decimation, environment count, random seed and deterministic settings.

Start from frame 0.

## P8.2 Full task evaluation

Evaluate the complete 840-frame reference, including approach, desired contact, pickup/lift, carry, placement, release and post-release stability.

These are descriptive task intervals only. Do not introduce hard-coded phase inputs or classifiers.

The original binary reference contact label remains the only reference contact condition.

## P8.3 Reference-versus-actual contact metrics

Compare the original reference binary label against actual MuJoCo hand-box contact at every valid evaluated timestep, including reference ON and OFF frames.

Report:

- reference contact ON/OFF frame counts;
- actual hand-box contact ON/OFF frame counts;
- contact confusion matrix;
- overall contact agreement;
- precision, recall, F1 and balanced accuracy;
- left-hand contact ratio;
- right-hand contact ratio;
- any-hand and bilateral contact ratios, reported separately;
- unintended contact ratio during reference OFF;
- missed contact ratio during reference ON;
- contact occupancy and duration;
- EEF-target proximity statistics.

Do not treat the global reference label as independently annotated left/right labels.

Evaluate left/right actual contact independently, including reference ON/OFF ratios, occupancy and duration. Report any-hand (OR) and bilateral (AND) comparisons separately using P6.2–P6.3. Bilateral AND may serve as the task-required aggregate only where P4.1 establishes that bilateral contact is required by the reference/task. Without that evidence, do not collapse the two measurements into one assumed global label. Always report any-hand contact during reference OFF so unilateral unintended contact remains visible.

Use the calibrated P6 physical contact thresholds and occupancy criteria. Preserve the same definitions across evaluation experiments.

## P8.4 Physical interaction and object stability

Report:

- measured left/right hand-box contact force;
- contact impulse and duration;
- peak contact force;
- Box position/orientation tracking error;
- Box linear/angular velocity error;
- Box slip during carrying;
- Box stability during transport;
- Box-support contact during placement;
- release success;
- post-release object displacement and velocity.

Keep hand-box contact, Box-support contact and robot-body-box contact separate.

Force values are physical evaluation diagnostics only. Do not require learned force predictions, force MAE, learned-force calibration or force-gradient similarity. Physical sensor calibration from P1/P2 and the contact criteria in P6.2 remain required.

## P8.5 Collision and penetration evaluation

Record relevant MuJoCo hand-box contact distances and penetration statistics.

Report maximum penetration depth, average penetration depth during contact, penetration duration and solver warnings/errors.

Distinguish numerical contact behavior from persistent physically significant interpenetration.

Do not claim that enabled collision geometry guarantees zero penetration.

Any collision/solver configuration change requires a separate documented justification and approval.

## P8.6 Robot and object metrics

Robot:

- root position/orientation error;
- body pose error;
- joint tracking error;
- velocity tracking error.

Object:

- position/orientation error;
- linear/angular velocity error;
- support contact;
- slip;
- post-release motion/stability.

World Model:

- robot/object one-step and multi-step prediction error;
- rollout drift.

## P8.7 Task success

Report pickup, carry, placement, release, full task completion and post-release stability.

Frame 839 alone is not proof of successful physical release.

The Box must remain physically stable for a documented interval after release.

## P8.8 Failure analysis

Report every failed episode:

- failure frame/category;
- robot/object trajectories;
- reference contact label;
- actual left/right contact history;
- EEF-target distance;
- measured physical contact diagnostics;
- penetration/solver diagnostics where relevant;
- likely cause and whether the issue is policy, WM, reference, asset or simulator related.

Do not discard failed episodes from success-rate calculation.

## P8 deliverables

- full evaluation scripts/configuration;
- robot/object/contact trajectory logs;
- contact agreement and task metrics;
- penetration/solver diagnostics;
- evaluation plots;
- `docs/hoi/force_contact/p8/REPORT.md`;
- outputs under `outputs/hoi/contact_supervision/evaluation/`.

## P8 acceptance criteria

- evaluation starts at frame 0;
- all 840 reference frames are evaluated;
- robot/object tracking is reported;
- reference-versus-actual binary contact is evaluated across all valid ON/OFF timesteps;
- left/right contact is reported independently;
- actual contact measurements use the validated P6 definitions;
- any-hand and bilateral agreement are reported separately, with task-required bilateral aggregation justified by P4.1;
- release and post-release stability are physically verified;
- collision/penetration limitations are documented;
- every failure is documented;
- checkpoint/reference/asset/simulator provenance is complete.

P8 stops after reporting. Do not automatically launch additional training or modify the selected checkpoint.


# 14. Expected code organization and change boundaries

Prefer HOI-specific implementation under:

~~~text
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
~~~

The active World Model must contain only joint robot-object dynamics.

The final architecture preserves:

- physical state84;
- canonical state86;
- residual output82;
- support context18;
- action29;
- xi33.

Do not introduce additional contact-related observation/state dimensions or modify the original Actor/Critic input contracts for the proximity reward.

The approved training reward uses the original binary reference contact label and differentiable EEF-to-object-target proximity.

Actual MuJoCo contact measurements remain independent physical diagnostics and evaluation signals. They must not be imported into the differentiable Actor reward or active WM prediction loss.

Tests should be placed under:

~~~text
tests/hoi/
    test_joint_scene.py
    test_contact_sensor.py
    test_force_aggregation.py
    test_force_replay.py
    test_joint_world_model.py
    test_contact_target_audit.py
    test_geometric_contact_reward.py
    test_imagination_parity.py
    test_checkpoint_migration.py
    test_physical_contact_agreement.py
    test_collision_diagnostics.py
~~~

Historical force utilities/tests may remain where required to reproduce P1/P2 diagnostics and old evidence.

Reuse existing physical contact aggregation and sensor utilities. Do not duplicate the P1/P2 measurement implementation.

Physical contact matching, force statistics and penetration diagnostics belong to evaluation/diagnostic paths, not differentiable training reward paths.

These are proposed locations. Reuse equivalent existing modules identified by P0's actual file-change map, while applying the current stage semantics. Superseded force/contact-head entries in historical maps do not authorize restoring those designs.

Shared files should receive only necessary interface extensions.

Do not:

- copy the complete Trainer;
- copy the complete ReplayBuffer;
- duplicate the original robot StateSpec;
- add an independent PPO loop;
- introduce unnecessary contact/phase classifiers;
- add a Contact Prediction Head, Binary Contact Decoder or Force Prediction Head;
- add force regression or Contact BCE;
- add measured MuJoCo contact/force as Actor reward inputs;
- add new Actor/Critic observation dimensions;
- add new World Model state/output dimensions;
- introduce an SDF network or penetration reward without evidence and approval;
- rewrite Part B;
- rebuild the reference Viewer;
- reprocess source datasets;
- change sm-train defaults.


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
| P3 | `hoi: finalize joint robot-object world model architecture` |
| P4 | `hoi: implement reference-conditioned proximity reward` |
| P5 | `hoi: integrate joint imagination and proximity reward` |
| P6 | `hoi: validate physical contact agreement and geometry` |
| P7 | `hoi: train joint world model and integrate HOI policy` |
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
- do not silently change state dimensions, coordinate conventions, reference-label semantics, target geometry, reward equations or architecture;
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

~~~text
Read the complete document:
docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md

Execute P0 only.

P0 means repository, dataset, asset, dependency and reference-code audit.
It is read-only with respect to training source code.

Inspect all exact dataset and asset paths in Section 3. Verify the Part B reference against its committed manifest. Trace actual SuperMimic WM, Replay, Imagination, Trainer and checkpoint code. Verify installed mjlab/MuJoCo Warp ContactSensor API and update order.

Produce all P0 reports and the implementation file-change map.

Do not reprocess Part B, regenerate datasets, modify checkpoints, change training source code or start training.

Follow Section 15. Report PASS/PARTIAL/FAIL and STOP. Do not start P1.
~~~

## Later requests — P1 through P8

~~~text
Read the complete document:
docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md

Execute P<N> only, following the corresponding section.

Inspect the actual current checkout and confirm all prerequisites and previous-stage acceptance/commit status.

Preserve all existing work. Implement only the current stage. Run its required tests and experiments. Inspect the complete Git diff.

Before implementation, create the current stage IMPLEMENTATION_SPEC.md and follow the amendment and approval rules in Section 1.3 and the report protocol in Section 15.

Provide the full stage report required by Section 15, including exact commands, numerical evidence, modified files, Git status and PASS/PARTIAL/FAIL.

Do not commit or push automatically.
Do not start the next stage.
Stop and wait for my explicit approval.
~~~

## P3 architecture cleanup request

~~~text
Read the complete Master Plan, especially Sections 1, 2, 4 and 8.

Execute P3 architecture cleanup only.

The final joint WM contract is 166D input / 82D output with 84D physical state. Remove the active 167D/83D Contact Prediction Head path and contact BCE. Do not run the dedicated 32-transition contact-boundary collection or classifier fit. Preserve historical reports, failed collection artifacts and original checkpoints.

Implement only final architecture cleanup, named checkpoint compatibility and required software/regression tests. Do not train the joint WM; its predictive training/validation is deferred to P7-A.

Follow Section 15 and stop for review.
~~~

## P4 request

~~~text
Read the complete Master Plan and execute P4 only, following all of Section 9 (P4.1–P4.7, deliverables and acceptance criteria) and Sections 1.3 and 15.

Confirm the current checkout, prior-stage acceptance and explicit authorization before implementation. Write docs/hoi/force_contact/p4/IMPLEMENTATION_SPEC.md with requirement traceability, verified P0/prior-stage evidence, contracts, validation criteria and unresolved risks.

Preserve physical84/canonical86/residual82, the final 166D/82D joint WM, support18/action29/xi33, original Actor/Critic observation dimensions, original robot/object tracking and existing P1/P2 contact sensing. Preserve user changes, historical reports/evidence, accepted reference NPZ, assets and checkpoints.

First produce docs/hoi/force_contact/p4/CONTACT_TARGET_AUDIT.md. Verify left/right ordering and target participation, wrist-local EEF offsets, FK, Box-root transforms under rotation, reference-distance distributions, marker/collision-surface correspondence and target reachability. Preserve M6.5 training_ready=false and M6_5_NO_GO_M7 evidence. Separately document whether P4.1 establishes a bilateral physical-contact requirement and its applicable interval; otherwise retain separate any-hand and bilateral evaluation measurements. Do not infer per-hand reference labels from two target slots. Obtain approval for any target-set or research-level change before affected implementation.

Implement only pure-Torch geometry and the HDMI-style position-proximity reward from P4.3–P4.5: r_proximity = object_contact[t] * mean_i(exp(-d_i / sigma_pos)) over the valid audited target set. Use the unchanged binary label at the exact evaluated reference frame, differentiable EEF FK and the current physical/imagined Box pose. Preserve robot/object tracking throughout the task. Reference OFF must yield zero proximity reward. Determine candidate distance scales and coefficients from audited distances and controlled reward-scale measurements; do not invent final values.

Actual MuJoCo force/contact remains an independent physical diagnostic, never a differentiable Actor reward input, active WM prediction target or additional observation. Reuse P1/P2 utilities only for required bounded deterministic diagnostics; full physical binary-contact agreement validation belongs to P6.

Do not add force rewards, contact/force prediction heads, Contact BCE, Binary Contact Decoder, future lookahead, smoothed labels, phase classifiers, onset/release detectors, non-contact repulsion, SDF/penetration rewards or new dimensions.

Complete every P4.7 test: exact frame/binary gating and OFF-zero behavior; ON-positive monotonic proximity; left/right ordering; rotated targets; current Box pose; batch independence; finite EEF-joint and Box-pose gradients; independence from measured force/contact; and original robot-only tracking reward regression. Record reward-scale/reference-distance evidence under outputs/hoi/contact_supervision/reward_validation/. Mathematical tests alone do not prove physical contact success.

Inspect the complete diff, reconcile IMPLEMENTATION_SPEC.md with results, and produce docs/hoi/force_contact/p4/REPORT.md under Section 15, including commands, tests, numerical evidence, framework reuse audit, limitations and PASS/PARTIAL/FAIL.

Do not start P5, P6, imagination integration, WM/Actor training or long experiments. Commit only when explicitly authorized, do not push, and stop for explicit review/approval.
~~~

For P7/P8, obtain explicit approval before expensive training or full evaluation.

# 17. Final research acceptance

The overall project is complete only when:

1. The native robot and object physical scene is valid.
2. Left/right hand-box contact sensing is physically verified.
3. Physical Replay and transition alignment are correct.
4. The final joint World Model uses 166D input and 82D output.
5. Support-relative context is correctly represented.
6. The joint WM predicts robot and object dynamics with validated one-step and multi-step performance.
7. The unchanged original binary reference contact label gates the reward at the exact evaluated reference frame.
8. Left/right contact targets, EEF geometry and their collision-surface correspondence are audited under P4.1, with target participation and any bilateral task requirement explicitly justified.
9. The P4.4 HDMI-style differentiable proximity reward uses current robot/Box geometry, yields zero for reference OFF and preserves physical84/canonical86/residual82 and original Actor/Critic observation dimensions.
10. Robot and object tracking remain active throughout the task.
11. Contact geometry reward produces meaningful Actor gradients.
12. Joint imagination preserves physical state and reference alignment.
13. Eager and compiled implementations are consistent.
14. Actor/Critic training uses the original SuperMimic optimization structure.
15. Original sm-train behavior remains unchanged.
16. Full evaluation starts at frame 0 and covers all 840 reference frames, demonstrating robot-object tracking and physical interaction; all failed episodes remain in the results and success-rate denominator.
17. Physical binary contact agreement is evaluated against the original reference label at every valid, correctly aligned ON/OFF timestep under P6.2–P6.3 and P8.3. Report confusion matrices, agreement, precision, recall, F1, balanced accuracy, unintended OFF contact and missed ON contact, with explicit validity masks and calibrated measurement criteria.
18. Left/right physical contact is evaluated independently, including ON/OFF ratios, occupancy and duration. Preserve and report any-hand (OR) and bilateral (AND) measurements separately. Bilateral AND serves as the task-required aggregate only where P4.1 establishes that requirement; the global reference label is never presented as independent per-hand annotation.
19. Penetration and solver diagnostics under P6.5/P8.5 include signed contact distances, maximum and average penetration depth, duration/frequency and solver warnings/errors. Separate hand-box, box-support and other robot-body-box contacts, and document persistent penetration and unresolved physical limitations.
20. Physical release and post-release object stability are verified under P8.4/P8.7 over a documented interval, with object displacement and linear/angular velocity evidence. Reaching frame 839 or matching EEF markers alone is not proof of release or stable placement.

Physical force/contact measurements are required independent diagnostics for the physical acceptance checks above. They are not final World Model prediction targets, differentiable Actor reward inputs/terms or force-gradient acceptance gates. Preserve historical M6.5 failures and report the disposition of unresolved asset/support limitations; a geometric reward test PASS alone does not satisfy physical task acceptance.

The final research contribution is a **reference-conditioned humanoid-object tracking framework that combines a learned joint robot-object dynamics model with differentiable geometric contact reinforcement, allowing the Actor to learn physically meaningful interaction while following demonstrated reference motion.**

No Contact Prediction Head, Force Prediction Head, force regression, force-based Actor reward or force-gradient fidelity module is part of the final architecture.

