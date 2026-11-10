# P0.5 reference implementation audit

Scope: the complete master, Section 5 / P0.5. This closes the reference-code gap in `579d1a8` and `e895731`. Static inspection only; no foreign training framework was installed, imported or run. Exact source hashes, file sizes and relevant AST symbol ranges are in [reference_implementation_checks.json](reference_implementation_checks.json).

## Sources and provenance

| Source | Inspected checkout / revision | Scope and limitation |
|---|---|---|
| HDMI | `/tmp/supermimic_hoi_audit_20261104/HDMI`, `main`, `32282f6dcf26cae70b814d585ceb12cc38aa1b60` | Official remote; clean. Exact `command.py`, `rewards.py` and bread-box task YAML inspected. |
| PhysHSI | `/home/han/pyx/PhysHSI-carrybox`, `pose_amp_vel`, `c75efd8ce8ded98031281b5a9b00ed71390dbccd` | Local Ir1222 fork; not asserted identical to official InternRobotics code. Exact `legged_gym/legged_gym/envs/g1/carrybox.py` and `carrybox_config.py` inspected. Existing untracked `torch` and `yaw_left.csv` retained. |
| SUGAR | `/tmp/supermimic_hoi_audit_20261104/SUGAR`, `main`, `01fe1234b57188412ec38775150f008f133e0ad4` | Official remote; clean. `locomanip/mdp/rewards.py`, `commands.py`, G1 `carry_box_refiner_env_cfg.py` and inherited `base_refiner_env_cfg.py` inspected. |
| Workhorse | arXiv `2610.09117v2`, Sections IV-A–IV-D; official project page, accessed 2026-11-10 | Public page states code is unreleased. User explicitly authorized using the paper and available papers/repositories instead. No source-code verification claimed. |

The user instruction was: “if workhorse code not released, just use paper and other papers and repo”. This exception applies only to the unavailable Workhorse code, not to any implementation stage or acceptance test.

## HDMI: measured position and force shaping

In `command.py:395`, each EEF resolves sensor instances and body indices. `command.py:441` loads the motion's global `object_contact` and `command.py:567` selects current/future labels. The active loop resolves `contact_eef_body_name`; the separate `contact_frc_eef_body_name` argument is not a guarantee that all pitch/yaw collision bodies are included. The target must use the audited native collider groups explicitly.

`command.py:570` computes contact targets from the **current simulated object body** and root-local offsets, and EEF positions from robot link pose plus wrist-local offsets. It sums `force_matrix_w` across selected sensors and uses inverse object rotation to express the force in object axes. The YAML supplies the same `(0,±.17,.14)` contact targets and `(.05,0,0)` EEF offsets recorded in M6.5.

`rewards.py:367`, `eef_contact_exp`, uses a tolerance-clamped distance and a bounded force deficit: the scalar-threshold branch multiplies `exp(-position_error/pos_sigma)` by `exp(min(norm(force)-threshold,0)/frc_sigma)`, then averages hands. Its outside-intent value is **1**, with a configurable gain on the active interval. The target master instead defines an intent-gated interaction contribution, so the source baseline/gain must not be copied silently. `eef_contact_exp_max` selects the best hand; `eef_contact_all` thresholds position and force. Neither is the prescribed independent smooth per-hand average by default.

Directly reusable concepts: local-point transforms, per-hand geometry and bounded force-deficit algebra. Adaptation required: pure Torch FK on predicted states, predicted force in imagination, independently calibrated native hand groups, four-substep labels and master reward gating. Incompatible wholesale imports: IsaacLab scene/sensor APIs, source robot hand/foot replacements, wrist gain overrides, object randomization and any source dataset-writing path. Current simulator force is diagnostic/supervision, not a differentiable Actor input.

## PhysHSI local fork: approach, contact and hold consistency

`carrybox.py:1058` acquires the Isaac Gym **net contact force tensor**, then reshapes it by body. `carrybox.py:2012` and `2032` threshold hand-body force magnitudes. These are not independently pair-filtered hand–box measurements: another object or the ground can contribute. The `rubber_hand` collision names in `carrybox_config.py` also differ from the target native pitch/yaw capsules. Root quaternion storage is xyzw, unlike the target wxyz contract.

`_reward_walk_task` at line 1975 combines approach speed and heading, with distance/success gates. `_reward_carryup_task` at 1996 uses the mean hand position relative to the box and a saturated height deficit. `_reward_hold_contact` at 2032 requires near/lifted/bilateral contact and combines it with robot–box horizontal velocity agreement. The configuration enables these local-fork pickup/hold rewards, while relocation and stand-up scales are zero. The presence of functions does not imply they are active.

`_reward_hand2object_pos` at 1949 subtracts box translation and clamps against box half extents without inverse box rotation; it is not general rotated compound-box distance. `_reward_relocation_task` and `_reward_standup_task` contain stage/success gates. They are inspection references, not a permitted phase policy for this task.

Reusable concepts: reward approach before lifting, distinguish contact from relative-motion consistency, and evaluate release separately. Adaptation: use the master's future-intent lookahead, verified EEF/compound-box geometry, independent hand–box force and support-aware release. Incompatible imports: PPO/AMP loop, hard task-stage gates, bilateral reward requirement, source reset curriculum, asset geometry and unfiltered force labels. No thresholds, coefficients or successful-task claims are transferred from this fork.

## SUGAR: label consistency and object-relative tracking

`mdp/rewards.py:142`, `hands_contact`, reads left/right filtered `force_matrix_w_history`, takes the maximum force norm across its history, thresholds each hand, combines them with logical AND and rewards equality with a single reference label. Thus an inactive label can receive reward when only one hand contacts: equality with a bilateral boolean is not independent per-hand agreement.

`base_refiner_env_cfg.py:71` configures three history samples; the hand sensors at lines 94–110 use `left_rubber_hand` / `right_rubber_hand` filtered to `Obj`. `carry_box_refiner_env_cfg.py:99` sets the contact threshold to 0.1 N and selects `SMALLBOX_CFG`. The environment has 5 ms physics and decimation 4, but three sensor samples and max reduction do not establish the required four-substep interval mean.

`commands.py:50` loads `robot_50hz.npz`, object motion and `contact_labels_50hz.npy` into a global bool label. Its rollout logging at line 755 separately derives a bilateral hand-contact label from measured maxima. These source labels are not substitutes for accepted Part B intent. `rewards.py:74–139` scores object pose/twist and object-relative body transforms; the base config keeps robot and object tracking terms active.

Reusable concepts: explicitly separate reference labels and measured contact, filtered sensor naming, and object-relative pose/twist rewards. Adaptation: target native geoms, full compound box, four samples, force normalization, independent soft per-hand confidence, and predicted-state transforms in imagination. Incompatible imports: hard bilateral equality, three-sample maximum as a Force Head target, IsaacLab runtime objects, small-box assets and privileged source policy assumptions.

## Workhorse: task-space interface and deployment errors

Workhorse specifies torso, wrist and foot targets through measured rigid offsets. Targets are expressed in torso frames; deployment transforms the planner's observation-time frame into the tracker's current frame. Its planner augmentation perturbs past poses and associated images while retaining clean future targets. Tracker augmentation instead introduces command drift and resets at replanning, with rewards following the perturbed commands. These mechanisms address deployment errors; they do not supply the target's measured-force labels or a force-prediction head. [Paper, Sections IV-A–IV-D](https://arxiv.org/html/2610.09117v2)

The official project describes command drift up to 10 cm horizontally and 10° yaw, reset after 0.2–1 s, and states that code is not released. These are source settings, not adopted SMHOI hyperparameters. [Official project](https://hybridrobotics.github.io/workhorse/)

Potential adaptation is limited to verified task-space transforms and future robustness diagnostics. Adding a visual planner, changing accepted motion, or importing deployment augmentation into training is outside P0 and is not authorized by this comparison.

## Reuse decision

| Mechanism | Decision for target | Stage boundary |
|---|---|---|
| Local EEF/target transforms | Reuse audited math and existing FK, with target frame tests | P1 state/inspection; P4 differentiable geometry |
| HDMI bounded position×force shaping | Adapt to exact master equations and predicted force | P4; Actor use gated through P6 |
| PhysHSI approach/relative-motion ideas | Adapt to continuous future-intent approach and physical diagnostics | P4/P8; no phase classifier |
| SUGAR object-relative tracking and label separation | Adapt; keep current imagined state distinct from reference | P3–P5 |
| SUGAR bilateral binary contact equality | Incompatible with v1 independent per-hand smooth reward | Do not copy |
| Workhorse task-space/deployment-error concepts | Paper-supported context only; no force-supervision claim | No new component authorized |
| Any external trainer, asset or simulator implementation | Incompatible as a wholesale port | Preserve original SuperMimic loop, M6.5 and mjlab pin |
