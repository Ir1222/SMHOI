# P6-A independent physical validation

**PHYSICAL_VALIDITY = PARTIAL.** Native collision/sensing and the nominal Box's
bounded dynamics are measurable and internally consistent. Original coupled
floor-contact solver reliability and plausible Box mass distribution remain
unresolved. This is not full physical training readiness.

All paths below are relative to `outputs/hoi/contact_supervision/contact_validation/`
(E). The approved stage was executed on main/e4f5741 without training, asset edits,
solver retuning or reference modification. The exact compiled model, active geom
transforms, masks, materials, joints, options, per-world support roots and sources
are in `geometry_02/inventory.json` and `model.mjb`.

## Preserved construction and reused evidence

P1 `sensor_calibration/run_05` already validated all 28 native hand/primary/Box-geom
combinations (112 placements across four worlds), force-on-Box sign, world/Box
frames, independent raw sums, transient history, selected reset, deliberate
capacity rejection and graph/eager parity. P2 `replay_diagnostics/run_03` validated
four-solve aggregation, terminal-before-reset copies, action alignment and Replay
windows. Their original results and failures were reused, not rerun.

The ten P0 dependency-source hashes still match. Current P1 `contact.py` includes
its accepted P2 extensions; current P2 `replay.py` includes the accepted P3 extraction
of `_gather_windows` from `sample_windows`, with the gathering body unchanged.
These explain the two historical whole-file hash differences in
`analysis_02/evidence_reuse.json`; neither file changed during P6. New P6 capture
uses those actual current implementations, including fail-closed capacity checks
and exact native-history equality.

The strict production loader still rejects the absent historical V3 document.
The diagnostic adapter described in specification A1 pins the original M6.5
manifest and verifies every physical source, allowing only that exact missing
planning filename/hash. It leaves production loading and M6.5 readiness unchanged.
This limited construction is authorized for the present bounded P6 experiments.

## Native collision surfaces and independent initialization

Both hands retain pitch and yaw capsules, masks 1/1. Pitch radius/half-length are
35/35 mm, center (25,0,0) mm. Yaw radius/half-length are 35/41.231 mm, centers
(110,−10,0)/(110,+10,0) mm. Full local quaternions, parent transforms and all seven
Box pairs are saved in the inventory. Box priority1/condim3 and native hand
priority0/condim1 remain unchanged. No contact pair or geometry was added.

Four approach scans contain **204 FK placements**: left/right × pitch/yaw ×51
steps of 0.5 mm. Each begins separated; no opposite-hand contact appears. First
yaw contact is within floating-point zero distance; first pitch contact is at
−0.5 mm, consistent with scan resolution. These are geometric tests only.

Independent bilateral initialization uses native-keyframe arm IK and actual
capsule surface clearance, without HDMI poses. Final position residuals are
1.26e-9/1.37e-9 m; pitch/yaw overlaps are bounded to 0.5 mm (guard ≤1 mm).
`geometry_02/independent_bilateral.json` stores full state and contacts. This proves
that bounded bilateral surface contact can be constructed, not that a free robot
can lift or carry the Box.

The initial orientation-constrained IK was a **failed fixture**: target residual
34.14 mm and approximately 97 N initial bilateral loads. `calibration_01` and
`geometry_01` remain intact. A position-only probe still intersected pitch
colliders by 3.33 mm. Adding actual pitch clearance and explicit residual/penetration
guards corrected only the disposable fixture. `calibration_02` and `geometry_02`
are the accepted replacements; the original attempts are not physical passes.

## Force calibration and binary definition

Fifteen calibration intervals × four worlds = **60 world-transitions**, each with
four 5 ms solves. Worlds are separated, left-only, right-only and independent
bilateral contact. Unilateral initial compression is 0.25, 0.5, 1, 2 or 3 mm;
Box rolls are 0, 0.31 or 0.63 rad. Full initial states, force columns, solve-time
quaternions, found counts, contacts, overflow and next state are preserved.

Training calibration uses rolls 0 and 0.63; roll 0.31 is held out. For each hand,
select the midpoint between maximum measured null force and the fifth percentile
of first-solve shallow-contact load. This rule was specified before measurements
and uses no reference label or agreement score.

| Quantity | Left | Right |
|---|---:|---:|
| Maximum null force | 0 N | 0 N |
| Training positive-load p05 | 0.1486802094 N | 0.1475020364 N |
| Calibrated threshold τ | **0.07434010469 N** | **0.07375101818 N** |
| Held-out first-solve detection | 5/5 | 5/5 |
| Occupancy cutoff ρ | 0.25 | 0.25 |

At each substep, sum the two simultaneous primary vectors, then threshold their
norm using strict `F > τ`. A policy interval is active only if occupancy `>0.25`,
thus **at least two of four solves**. Both hands remain independent. All held-out
policy-step no/left/right/bilateral decisions match their controlled setups.
Reanalysis of P1's actual stored transients at these thresholds rejects one-solve
pulses and accepts the two-solve pulse. This is a persistence definition; genuine
single-solve impacts, subthreshold loads and cancelling local forces remain visible
in raw diagnostics but need not be binary ON. It is not a universal sensor-noise
model or a reward coefficient.

`calibration.json` contains the rule, distributions, holdout outcomes and transient
reanaly­sis. P2's scalar 0.001 N remains only the unchanged recorder's legacy raw
occupancy field. Final P6 contact is recomputed from saved force columns with the
above independent thresholds, never taken from that legacy field.

New pelvis/torso–Box and Box-support exclusion fixtures have nonzero raw excluded
contact loads while both hand sensors remain exactly zero. They deliberately
initialize body contact and are filtering tests, not manipulation demonstrations.
The first torso fixture produced only 0.663 N, failing the verifier's >1 N load
criterion; increasing only its diagnostic compression to 2 mm passes in
`filters_02`. `filters_01` and `verification_01` retain the failure. The pelvis
filter fixture has a deliberately large initial overlap; it is not a physical
stability acceptance case.

Independent saved-data recomputation checks **2,616 world-transitions**. Maximum
errors are 7.62e-6 N for contemporaneous Box-frame mean, 1.48e-7 N·s for impulse,
and 9.75e-6 N for peak. Four-solve timing/history guards remain active. No new
force feature enters state, WM loss, imagination, Actor reward or observations.

## Box model, analytic response and proposed correction

Nominal model: one free rigid body, seven collision boxes, **2 kg**, root at the
bottom-plate center, COM **(0,0,−0.2) m**, identity principal axes and diagonal
inertia **(0.0334,0.02167,0.04167) kg·m²**. Inertia is positive and satisfies the
triangle inequalities. Collision envelope is [−.09,−.19,−.01] to [.09,.19,.26] m.
The COM is **190 mm below the envelope**. This cannot represent positive mass
confined to those seven solids; it remains a source-parity model, not validated
bread-box mass distribution.

Six independent 0.1 s CPU experiments apply gravity, a 4 N world-X COM force or a
0.02 N·m world-X torque, each at yaw 0/0.6 rad. No state is reassigned after reset.
First-solve COM-velocity error is at most **1.29e-8 m/s**, angular-velocity error
at most **1.75e-18 rad/s** against F/m and the rotated inertia tensor. Thus the
simulator responds consistently to the declared inertial model; that does not
make the mass distribution physically plausible.

`box_dynamics_01/mass_proposal.json` gives an **unapplied proposal**, assuming
uniform density across seven separately weighted cuboids (overlapping part
volumes counted per component): mass remains 2 kg; COM would be
**(0,0,0.1074910820) m**, and inertia about that COM in root axes would be
**diag(0.04311170713,0.02364345505,0.03797970670) kg·m²**. Component masses and
principal axes are included. This is a calculation under stated assumptions,
not measured material evidence. Moving COM changes gravitational/contact torque
arms and the current pendulum-like stability. Approval must specify a versioned
asset profile/material assumption and require P1 filtering/sign/reset plus P6
force/torque/support/solver/lift/release regressions and a separate reference
comparison. **No correction was applied.**

## Support, lift, release and solver findings

Eight 3 s isolated CPU cases use unchanged original supports: settle/drop/lift at
each support, a floor drop, and a support0 world-offset control. Along with the six
analytic cases, **14 CPU cases** ran. Static settle starts 10 mm above contact;
drop starts 100 mm above contact. External-lift cases first settle, apply **24 N
upward at COM for 0.25 s** (6 N·s impulse), then remove all applied forces.
Both lift cases rise **83.6245 mm**, lose support, fall and recontact. Every case
ends stable for the final 0.5 s: support/ground force **19.62 N**, COM speed below
3.11e-10 m/s and angular speed below 4.12e-9 rad/s. No CPU warnings; at most two
solver iterations. This validates externally forced lifting/release of the nominal
Box, **not native-hand lifting or natural carrying**.

GPU support tests retain the original Box/support heights and per-environment
origins. With the distant robot initialized at z=50 m to avoid a floor impact
within 3 s, all four worlds settle **10.1079 mm**, final mean support force is
**19.620002–19.620003 N**, maximum tail COM speed **3.83e-6 m/s**, no hand force,
no solver flags. Only the remote robot is elevated; the Box and supports are not.

The same original-support setup with robot z=3 m fails at **0.700 s**, world1,
`LS_ITERATIONS=1024`. Contact evidence contains concurrent foot–floor overlap of
about **33.73 mm**; the Box remains supported with no hand contact. The paired
z=50 control isolates floor impact as the condition associated with this failure.
It does not prove which internal line-search numerical condition is defective.

Independent native no/left/right/bilateral hand-contact dynamics at z≈3 m fail at
**0.835 s**, world2, also bit1024 during landing. A +10 m common robot/Box offset
completes 1 s without flags: contacts release, the Box falls under gravity, and
measured bilateral contact lasts 0.06 s per hand with roughly 1.92e-5 m slip.
Unilateral contact lasts 0.02 s, with about 3.98e-5 m slip. These are diagnostic
release/freefall cases, not stable carrying or support placement by the hands.

P2-like frame260 floor/elevated deterministic controls both complete 24 steps
(0.48 s) here. They use zero action noise, unlike P2's historical noisy run, and
are not exact reproductions of that intermittent failure. Historical P2 LS1024
and P3 ITERATIONS512 failures remain unresolved; no limit was increased. New
LS1024 cases have available contact/constraint capacity, so they must not be
misreported as buffer saturation. Full flags/counts/iterations are in
`analysis_02/dynamics_analysis.json`. The production guard stops on every warning;
partial final intervals are retained, not reused as valid force labels.

## Penetration and acceptance limits

Every solve logs signed distance, position, full frame, geom IDs/names and type.
All recorded relevant GPU contacts in these runs have constraint type1. Summaries
retain record counts, unique-sample duration, minimum signed distance, maximum and
mean depth, >0/.1/1/5 mm frequency and maximum continuous duration per hand/world.
Positive distance denotes separation, zero touching, negative overlap.

| Independent scenario | Maximum depth | Mean depth during contact / interpretation |
|---|---:|---|
| Original-support 10 mm settle, CPU | 2.399 mm | 0.1465 mm; final compliance ≈0.108 mm |
| Support/floor 100 mm drop, CPU | 10.481 mm | ≈0.290 mm; transient impact, final stable |
| External lift and recontact, CPU | 5.711 mm | 0.2228 mm; transient landing |
| Native unilateral release fixture | 3.000 mm | intentional initial compression; full traces retained |
| Corrected bilateral initialization | 0.500 mm | bounded initial surface placement |
| Original coupled foot/floor failure | ≈33.73 mm | solver failure; not accepted stability |

Transient impact penetration is not zero geometric penetration. Persistent >1 mm
for ≥0.1 s is explicitly flagged; stable settling tails are much smaller. There
was no SDF, collision reward, new geometry, dt change or solver tuning. A solver
proposal would require separate approval and matched original-floor regressions;
these findings do not justify silently changing iterations or material parameters.

**Decision:** bounded sensing, analytic response and nominal support capacity PASS;
full physical validity PARTIAL, with explicit COM/provenance and coupled-floor
blockers. Native-hand dynamic lifting/carrying is NOT TESTED as a successful task.
M6.5 remains `training_ready=false` / `M6_5_NO_GO_M7`.
