# M6.5 Physical Reference Reconciliation

**Decision: NO-GO for M7.** The investigation and controlled tests are complete. No tested minimal target, EEF, collider or static support-height correction reconciles the preserved reference across the motion. The support can physically hold a released box after a small gravity settle, but that conditional result does not repair the original reference geometry or demonstrate successful delivery. Stop here for review; no M7–M9 or HOI training has started.

The original plan, M6 report, validation record and asset manifest were read before work. All pre-existing tracked target files, historical audit outputs and referenced source assets/data were frozen by SHA256. Both NPZs, robot joint/body trajectories, old manifest, checkpoints, experiments and `sm-train` remain untouched. All new measurements are in [m6_5/](m6_5/); no old result was overwritten. [Baseline](m6_5/baseline.json), [final integrity check](m6_5/integrity_final.json).

## A. Hand–box root causes

### Coordinate convention is consistent; point semantics are not physical surface constraints

The original M4 measurements were reproduced to 1e-12 m for both native and HDMI profiles. M4's target-G1 wrist FK agreement (maximum 8.19e-8 m) remains valid under unchanged inputs. Official `active_adaptation/envs/mdp/commands/hdmi/command.py:573–577` uses body **link** poses and exactly the current formulas:

- `target_world = box_root_world + R_box * [0, ±.17, .14]`
- `EEF_world = wrist_yaw_link_world + R_wrist * [.05, 0, 0]`

These targets lie on actual side3/side4 outer walls, not on handles or at COM. No quaternion-order, joint-order, root/COM or wrist-offset transcription error was found. The existing bread-box seven-part geometry matches official XML/USD; replacing it with a different solid cuboid would change the task.

Crucially, official `cfg/task/base/hdmi-base.yaml:220` sets `eef_contact_exp.pos_sigma=0.3 m` and a separate force threshold of 10 N. The reward uses a soft exponential point error and actual contact force. It is **not** a constraint that the recorded EEF must coincide exactly with the target. The task's fixed EEF marker is inside the hand collision volume. Interpreting marker coincidence as nonpenetrating surface contact was the semantic mismatch; relabeling a point alone cannot move any physical surface.

The remaining large errors are measurable properties of the recorded wrist–object relative poses. During desired contact `[260,550)`, left/right RMS error is **48.87/51.71 mm**. Normal-to-wall RMS is **47.14/40.86 mm**; tangential RMS is **12.86/31.69 mm**. The current +50 mm local-X offset is correctly implemented. An offset required to match the target exactly would vary strongly with time, especially laterally: left wrist-local Y ranges **−92.24 to +2.14 mm**, right Y **+3.61 to +84.64 mm**. That is not a constant frame-calibration error.

### Point-definition candidates

All fits are diagnostic; none was written into either reference or the original manifest. RMS values below cover the 290 desired-contact frames. The constant-offset fit is the global least-squares optimum for a fixed target under orthogonal wrist rotations: its nonzero residual directly rules out a perfect repair by *any constant offset alone*.

| Candidate | Left RMS (mm) | Right RMS (mm) | Interpretation |
|---|---:|---:|---|
| Official +50 mm local X | 48.87 | 51.71 | Correct official convention |
| Wrist origin, no offset | 64.28 | 69.19 | Worse |
| +80 mm local X | 61.17 | 61.69 | Worse |
| +100 mm local X | 75.12 | 74.57 | Worse |
| Best unconstrained constant 3D offset | 28.35 | 28.33 | Still inconsistent; invents large lateral offsets |
| Fit fixed target constrained to true wall | 48.68 | 44.71 | Does not resolve normal mismatch |
| Bounded offset + wall target, fit first half | 28.85 | 27.74 | Second-half RMS 34.88/34.37 mm; fit reaches bounds |

Best unconstrained offsets are `(42.44, −36.78, 13.20)` mm left and `(46.13, 41.93, −9.91)` mm right. These are data fits, not verified palm surface definitions. A time-varying target equal to recorded EEF could make error zero by construction but would no longer certify contact with an actual box surface; it is rejected.

[Per-frame candidate errors](m6_5/eef_candidates.csv), [fit values](m6_5/eef_candidates.json), [side-by-side chart](m6_5/eef_comparison.png).

### Which geometry and orientation cause penetration?

Penetration occurs against real side walls, not an erroneous OBB approximation. Worst native contacts are `left_hand_collision`–`side3` at frame 466 and `right_hand_collision`–`side4` at frame 438. Worst HDMI contacts are left proximal box–side3 at frame 436 and right middle box–side4 at frame 446. The problem is not solely the native pitch-wrist capsule: HDMI removes that capsule yet still has deep hand–wall intersection.

The table uses a documented near-contact band **[−1,+2] mm** for primitive-pair distance, distinct from M4's 0.1 mm penetration-reporting threshold. Counts therefore differ from M4 without changing the underlying distances. A valid band is only a geometric screening condition; it still does not establish supporting force.

| Hand geometry | Worst penetration L/R (mm) | Frames penetration >1 mm L/R | Frames gap >2 mm L/R | Both hands in band /290 |
|---|---|---|---|---:|
| Original native capsules | 37.22 / 44.99 | 189 / 228 | 86 / 54 | 0 |
| HDMI hand-only boxes | 31.07 / 57.89 | 134 / 203 | 140 / 75 | 3 |
| Native radius 35→30 mm diagnostic | 32.22 / 39.99 | 169 / 215 | 109 / 67 | 0 |
| HDMI local-Y half-size ×0.8 diagnostic | 27.73 / 56.32 | 116 / 195 | 161 / 82 | 1 |

Reducing collider size trades penetration for separation; neither small reduction passes, and neither has source-asset provenance. They are rejected as physical corrections. Native wrist/hand inertias and feet were preserved in all dynamics; changing hand collision shapes does not make the model dynamically identical to HDMI.

An orientation-only diagnostic kept original wrist positions but froze each wrist orientation relative to the box at frame 260. Worst right HDMI overlap decreased **57.89→30.94 mm**, while substantial gaps/overlaps remained. Thus the changing wrist orientation contributes, but is not the only cause; relative translation and the finite collider envelope also matter. This counterfactual violates robot FK and was never used as a corrected trajectory or in dynamics. Scaling the official box geometries to 0.9 or 1.1 also fails: at 0.9, worst native/HDMI right overlap remains 38.08/48.09 mm, with larger gaps. Nominal scale 1 is retained.

[All-frame geometry comparisons](m6_5/hand_candidates.csv), [orientation/scale ablations](m6_5/hand_ablations.csv), [distance plot](m6_5/hand_geometry_comparison.png), [actual collision surfaces and markers at frame 446](m6_5/collision_surface_snapshots.png).

### Reachability with the robot reference fixed

With robot poses, box poses and either source-supported hand profile all held exactly as recorded, there is no free variable that can remove the measured intersections; changing point labels cannot help. We also explored whether small **object-only** pose adjustments might allow near-contact while keeping the robot fixed. This did not create a new object reference.

A multistart local SLSQP probe tested 13 representative frames per profile, constraining each translation component to ±30 mm and each rotation-vector component to ±5°. These are component bounds, not a 30 mm translation-norm or 5° angle-norm bound. Feasibility requires both hands' minimum distances within [−1,+2] mm, with 0.01 mm optimizer residual tolerance. Only **4/13** frames found feasible configurations for each profile. The search ignores support and force equilibrium and does not enforce continuity; failures are not a proof of global infeasibility outside those bounds. Even successes therefore cannot authorize a continuous physical trajectory. [Search results](m6_5/box_pose_feasibility.csv).

The justified conclusion is narrower and actionable: **no tested minimal, fixed calibration/geometry correction establishes a physically consistent whole-motion reference without changing its robot poses**. A broader object/contact trajectory reconciliation could be explored explicitly after review, but is not silently substituted here.

## B. Box–support reconciliation

### Root conventions and actual footprint

Both official support MJCFs specify half extents `(.1,.1,.02)` at local position zero; they agree with USD and the M6 manifest. The support root is its geometric center. The box root is the bottom-plate center, whose lowest face is −.01 m. At both endpoints, the recorded box-root/support-root vertical separation is .04 m; actual contact would require .03 m for level surfaces. This directly explains the **10 mm** reference gap. Source metadata does not identify why the motion producer used .04 m, so an upstream authoring intent is not asserted.

A full 840-frame recomputation uses actual seven-part MuJoCo pair distances and convex-polygon intersection of the projected bottom plate and support footprint. Initial and final overlap areas are **.036 m²**, out of bottom projection **.0612 m²** (about 58.8%); COM projects to the center of the support, with 100 mm margin to its XY edges. The endpoint gap is real, not a comparison against the wrong remote support or missing horizontal overlap. [Footprints](m6_5/support_footprints.csv), [top views](m6_5/support_footprints.png).

### Full-trajectory static height candidates

Positive correction means raising each fixed support root, preserving its collider-local center. A matching local-geom Z shift would move the surface identically but would misleadingly leave the motion root convention unresolved; neither convention change is silently applied.

| Raise | Endpoint gap | Minimum pair distance support0 / support1 | Frames penetration >1 mm support0 / support1 |
|---|---:|---|---|
| 0 mm | +10.000 mm | +7.146 / +4.306 mm | 0 / 0 |
| +4 mm | +6.000 mm | +3.146 / +0.306 mm | 0 / 0 |
| +7 mm | +3.000 mm | +0.146 / −2.694 mm | 0 / 8 |
| +10 mm | approximately 0 | −2.854 / −5.694 mm | 8 / 15 |

The +10 mm correction produces support0 penetration during approach around frame **253**, before desired contact starts, and support1 penetration during placement around frame **544**, before release intent at 550. Full per-frame rows include approach, pickup, carry, placement and release; no phase was omitted or replaced by an endpoint check.

All-frame bisection bounds the maximum *common* raise with at most 1 mm penetration at **5.3057–5.3086 mm**. Endpoint gap ≤1 mm needs raise **9.0000–11.0000 mm**. These intervals do not overlap. Independent support heights also cannot solve the final-support contradiction: support1 alone is limited by frame 544. A +10 mm shift is consequently rejected for the strict reference gate; +4 mm is safe from penetration but does not reconcile endpoint contact.

[Complete sweep](m6_5/support_candidates.csv), [constraints and extrema](m6_5/support_candidates.json), [full-motion height comparison](m6_5/support_height_comparison.png).

### Actual released-box stability

The original supports **can** support the nominal free box. In isolated 3 s tests starting at original frame 0 or 839, with no actuation, it settles approximately **10.108 mm** onto the original support and remains stable; final mean support normal force is **19.62 N**, matching 2 kg × 9.81. +4/+10 mm controls settle about 6.108/0.108 mm respectively. The extra ~0.108 mm is compliant MuJoCo contact penetration, not a rewritten reference pose.

Stability requires the last .5 s continuously supported >1 N, linear speed <.02 m/s, angular speed <.1 rad/s, orientation error <.05 rad and final position error <.02 m. This explicitly permits the observed gravity settle; it **does not pass the original 1 mm reference consistency gate**. At frame 839, where hands are already far away, coupled robot+box tests also retain a stable box even though the unassisted PD robot falls. At release-intent frame 550, the still-intersecting hands disturb the box and fail placement. Conditional terminal stability is therefore established; actual delivery-and-release success is not.

[Gravity-settle traces and forces](m6_5/terminal_stability.png). These conclusions retain the official unusual COM below the collision envelope; they describe that nominal simulated asset, not validated real-object mass distribution.

## C. Controlled short dynamics tests

The full protocol is [DYNAMICS_PROTOCOL.md](m6_5/DYNAMICS_PROTOCOL.md). **36 primary cases and 6 half-step sensitivity cases** completed, each 1–3 simulated seconds. Source NPZ samples remain intact. The original Reproduced `get_g1_robot_cfg → Entity` constructs all 29 built-in position actuators; compiled stiffness, damping, armature and torque limits are checked against the original. Action affine mapping reconstructs the exact original joint setpoint at 50 Hz; no velocity feedforward or gain retuning is introduced. Noise and domain randomization are disabled for paired tests, with original evaluation foot friction 1.0.

Primary MuJoCo CPU settings use 5 ms steps, implicitfast, pyramidal cone, iterations=10 and ls_iterations=20. Half-step checks use 2.5 ms, preserving 50 Hz command holding and gains. The robot root and seven-part box are free, supports fixed, ground present. **No root assistance, box actuation, weld/attachment, mocap or post-reset state rewriting.** External applied force arrays are asserted zero every step. This is original-PD reference playback, not a trained balance policy or policy-success test.

Representative original-support cases, native versus HDMI hand-only:

| Start / duration | Final object error native / HDMI (m) | Actual bilateral force fraction during intent | Peak hand normal force L/R native; HDMI (N) | Slip path L/R native; HDMI (mm) |
|---|---|---|---|---|
| 250 / 1.2 s, pickup | 1.148 / 1.015 | 0.0% / 2.5% | 49/77; 59/165 | 43.6/58.3; 53.8/32.1 |
| 400 / 1 s, carry | 1.015 / 1.037 | 11.4% / 15.4% | 85/187; 92/36 | 132.6/141.3; 157.6/134.9 |
| 530 / 2 s, placement | .672 / .698 | 12.5% / 11.3% | 2067/112; 778/308 | 38.9/16.3; 15.7/13.7 |
| 550 / 3 s, release intent | .692 / .650 | N/A: intent already false | 2013/99; 938/559 | 41.7/45.7; 63.5/15.9 |
| 839 / 3 s, terminal reset | .0101 / .0101 | N/A; hands clear | 0/0; 0/0 | 0/0; 0/0 |

“Force” here is the sum of actual `mj_contactForce` normal magnitudes on each hand's box contacts, not net Cartesian force or motor torque. Slip is the force-weighted relative contact-point tangential speed integrated while normal force >1 N; object trajectory/drift is logged separately. Resetting a frame with pre-existing penetration produces solver transients: frame-530 initial maximum hand overlap is 35.00 mm native / 27.77 mm HDMI. The high forces are evidence of an inconsistent initial physical state, **not strong successful grasping**.

At frame 400 the robot pelvis falls below .35 m around .89 s; box tracking also fails. At frame 530 object errors grow despite the robot remaining above that fall threshold. Raising supports by 4 or 10 mm does not fix the failures: frame-530 final errors are .660/.699 m for +4 mm and .688/.699 m for +10 mm (native/HDMI). No motion success is inferred from joint tracking, clip progress, high force or support reset stability.

Half-step repeats preserve the qualitative conclusions: frame-400 final errors **1.018/1.033 m**, frame-530 **.668/.672 m**; original/+10 mm isolated final drops remain **10.108/.108 mm**, stable. Thus the observed failure is not removed by halving dt, although exact contact peaks and trajectories remain solver-dependent. [Primary summary](m6_5/dynamics_v1/summary.json), [half-step summary](m6_5/dynamics_half_step/summary.json), [side-by-side dynamics](m6_5/dynamics_comparison.png). Each case directory contains its full per-step trajectory, contacts and compiled PD parameters.

Three preliminary smoke cases are preserved but excluded from decision evidence: they exposed a command-index rounding issue, fixed to exactly four/eight physics steps per reference sample before the official runs. [Smoke status](m6_5/dynamics_smoke/STATUS.md).

## D. Selected configuration and M7 recommendation

The versioned review manifest is [asset_manifest_m6_5.json](../../../data/assets/hdmi_bread_box/asset_manifest_m6_5.json). The original `asset_manifest.json` remains byte-identical. No physical correction is accepted merely to improve a diagnostic number: selected native-hand worst overlap remains 37.22/44.99 mm before/after, and selected original-support endpoint gap remains 10.000 mm before/after. What changed is the explicit semantic/physical separation and the measured stability evidence, not the reference.

Selected **review baseline**, not a certified learning configuration:

- Official nominal seven collision boxes, mass 2 kg, COM `(0,0,-.2)`, explicit official inertia, scale 1; existing documented priority=1 adaptation retained.
- Native Reproduced hand/wrist, feet and inertias unchanged. HDMI hand-only remains an explicitly tested alternative, not blindly selected; size reductions and orientation changes rejected.
- Official fixed target and +50 mm EEF retained as semantic markers. Physical validation separately uses actual surface separation/penetration, bilateral forces and slip. A zero marker error is never a physical-contact pass.
- Original support geometry and recorded root poses retained. +4/+7/+10 mm shifts rejected as full-reference fixes. Conditional settling stability recorded as its own result, not substituted for strict reference consistency.

| Requirement | M6.5 result |
|---|---|
| Immutable robot/reference/history | PASS |
| Target/EEF source and coordinate conventions | PASS; semantics clarified |
| Whole-motion hand–box consistency | FAIL |
| Static support correction valid across original trajectory | FAIL; incompatible height constraints |
| Original support can hold an independently released terminal box | CONDITIONAL PASS |
| Unassisted original-PD pickup/carry/placement/release | FAIL in tested windows |
| M7 readiness | **NO-GO** |

Remaining uncertainties and limits:

1. A local pose search is not a global reachability proof and did not establish a continuous, force-feasible object trajectory. The provided clip has no source-generation/contact-force manifest establishing collision-constrained authoring.
2. Pure position-PD playback with a free root is not a learned balance controller. Its failures reject these tests, not every possible future policy. No pretrained policy was silently substituted.
3. PhysX–MuJoCo material/solver equivalence is unresolved; raw HDMI mass randomization and hand inertias differ from the nominal controlled configuration. Half-step testing is a sensitivity check, not complete engine validation.
4. Terminal resets test support capacity, not successful arrival. The original COM lies outside the collider envelope and is retained only for source parity.

The smallest defensible next decision is to keep the validated robot reference frozen and explicitly authorize an independent object/contact trajectory reconciliation or another reviewed physical demonstration/controller test, with named tolerances and full-motion force/geometry evidence. A new object/contact artifact would need its own provenance; it must not overwrite either original NPZ. No such replacement or M7 implementation is performed here.

## Verification, reproduction and commits

**7 focused tests passed**, covering prior hashes, exact footprint intersection, original actuator gains/limits/armature, free-root/no-assistance structure, exact command holding, complete 840-frame support data, terminal force balance and reproduction of M4 distances. [Test log](m6_5/tests.txt). These checks validate the harness and records, not failed physical gates. The 12 original M6 asset checks also passed: **19 combined tests passed**, with one upstream Torch deprecation warning. [Combined validation log](m6_5/validation_all.txt).

From `/home/han/pyx/SMHOI`:

```bash
.venv/bin/python scripts/hoi/reconcile/geometry.py
.venv/bin/python scripts/hoi/reconcile/dynamics.py --output dynamics_review_new
.venv/bin/python -m pytest tests/test_hoi_reconciliation.py tests/test_hoi_asset_audit.py -q
```

Dynamics refuses an existing output directory, preserving experiments. Do not run the old M0–M6 aggregate command to generate this report: it rewrites historical audit outputs. `sensitivity.py` and `decision.py` also require their new output paths not to exist. The scripts and exact data used are recorded in [m6_5/WORK_LOG.md](m6_5/WORK_LOG.md).

**Paused for review after this report.**
