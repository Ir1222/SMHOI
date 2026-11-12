# C1 — Object construction and physical reconstruction

Status: **PASS for the selected, bounded simulation approximation**. The original
HDMI asset remains unchanged and is not physically reconciled by this result.
Evidence root **E** = `outputs/hoi/contact_supervision/p6_closure`.

## Baseline and provenance

Checkout `main`, HEAD `b4d412479d8b7dd5a174e6f176a23a07c525e1f6`;
pre-existing untracked `.vscode/` preserved. `E/baseline.json` inventories1236
protected paths, including historical evidence, assets, checkpoints, dependencies
and the accepted reference. No applicable AGENTS.md was found in the checkout or
ancestors. The current Master Plan and P0–P6/M6.5 records were inspected before
candidate work. Original P6 failures remain authoritative for their configurations.

The accepted840-frame NPZ SHA256 is
`bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`.
Asset manifest SHA256 is
`1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`.
The missing historical V3 planning source still prevents strict production
provenance validation. Experiments reuse P6's pinned diagnostic adapter; this
does not remove the M6.5 gate.

## Upstream constructions are different tasks

Sources are pinned to humanoid-amp-ppo commit
`b5087b7c8b3a123135b094b9cece3b2cefef26bc` in `E/upstream`. The primary
[SMTrack Box factory](https://github.com/shuayyy/humanoid-amp-ppo/blob/b5087b7c8b3a123135b094b9cece3b2cefef26bc/src/mjlab_g1/tasks/smtrack/mdp/box.py)
creates one free body and one box collider: half-extents(.15,.16,.18)m,1.5kg,
friction(1,.005,.0001), condim3, priority1. MuJoCo derives homogeneous-box inertia.
Its reference generator uses a different dual-arm mocap, derives the Box path
from hand midpoints, adds a forward offset and clamps the floor. That trajectory
is not an HDMI reconstruction.

The inspected SMTrack train/eval entrypoints, environment, RL configuration and
batch scripts map Lift to this factory. Training randomizes nominal mass/friction
by .8–1.2; evaluation pins nominal values. Public README results describe trained
Lift policies, but corresponding full run/checkpoint metadata was not available
in the pinned tree. This is source/configuration evidence, not independent
verification of every reported training run.

The [DualArm Toaster configuration](https://github.com/shuayyy/humanoid-amp-ppo/blob/b5087b7c8b3a123135b094b9cece3b2cefef26bc/src/mjlab_g1/assets/toaster_constants.py)
identifies the released v16 R3/model4999 policy and original spawn(.375,.1,.125)m,
yaw90°. README specifies8192 training environments, while current defaults differ.
The full XML has1.10kg main-body inertia plus .01/.03/.07/.09kg components:
total1.30kg. Button, knob and lever have joints. It is not a single rigid compound
Box. Two grasp-marker boxes have half-extents(.005,.06,.05)m at local
(±.09,.04,.055), with a marker-only friction override(2,.5,.05). Explicit body
inertias prevent collider volume from automatically adding mass. The training
configuration also contains assistance/reward machinery that was not adopted.
Its virtual assistance scale is zero in evaluation. See C1.7 for inherited
contact settings and the actual compiled collision editor effects.

## Original HDMI model and candidate mass properties

The original root is the bottom-plate centre, not the COM. Seven axis-aligned
cuboids form a hollow body with side handles. Local envelope is
[-.09,.09]×[-.19,.19]×[-.01,.26]m; oriented dimensions .18×.38×.27m.
Root rotation changes world AABB; `geometry_02/*/frames.json.gz` records all840
frame measurements. Full source/compiled values are in
`audit_details/original_construction.json` and candidate specifications.

| Construction | Mass kg | COM z relative to root m | Tensor diagonal in root axes at COM, kg m² |
|---|---:|---:|---|
| Original seven cuboids |2|−.200000|(.033400,.021670,.041670)|
| Prior P6 component-density proposal |2|.107491|(.043112,.023643,.037980)|
| Exact union shell, seven disjoint pieces |2|.108379|(.041273,.023319,.036056)|
| Selected single body, .18×.34×.27m |2|.125000|(.031417,.017550,.024667)|
| SMTrack dimensions, .30×.32×.36m |2|.170000|(.038667,.036600,.032067)|
| Same SMTrack dimensions |1.5|.170000|(.029000,.027450,.024050)|

Off-diagonal terms, eigenvalues and principal-axis quaternions are explicitly
stored in `E/candidates/*.json`; all tensors are positive definite and satisfy
rigid-body triangle inequalities. The selected principal moments are
(.01755,.024666667,.031416667), quaternion(.5,.5,.5,.5), correctly rotated back
to the root-axis tensor above. The single-body homogeneous baseline is
Ixx=m(hy²+hz²)/3, with cyclic permutations. Collision solid and mass assumption
are explicit; there is no real measured bread-box material/density specification.

The original COM lies190mm below its entire collision envelope. Positive inertia
alone does not make this a plausible mass distribution for the stated object.
The M6.5 concern is therefore substantiated, not dismissed by settling tests.
The earlier P6 proposal counted overlaps per component: component volume sum
.006728m³ versus actual union .006072m³, a .000656m³ duplication. The disjoint
shell integrates the union exactly once and preserves the original solid.

The selected2kg mass holds the original nominal mass fixed while comparing
geometry/inertia.1.5kg is an explicit independent control, not a score-selected
replacement: weight decreases19.62→14.715N and the same force gives4/3 the
translational acceleration. Homogeneous solid density for the selected model is
about121kg/m³; it is an effective simulation approximation, not an identified
material. A hollow shell remains a valid alternative if interior interaction is
later required.

## HDMI geometry and selection

Twenty configurations were evaluated over all840 frames, including all290 ON
frames260–549: eight constructions plus twelve one-axis dimension sensitivities.
EEF separation during ON ranges .339593–.515613m, median .417814m. Original Box
root ranges x[−1.467521,1.388164], y[−.348682,.229014], z[.659429,.927355]m;
maximum root speed1.363064m/s and angular speed2.353772rad/s.

Original motion yields0/290 bilateral frames in the unchanged [−1,+2]mm native
surface band for both original and selected geometry. SMTrack-sized geometry
gives only4/290 and non-hand overlap in151/290 ON frames. A narrower y sensitivity
gets9/290; that isolated score does not justify changing the task dimensions.
Original thigh/knee-shin/torso minimum clearances are241.17/370.35/185.29mm;
the minimum non-hand arm clearance is19.76mm. Per-phase and per-frame distances,
approach directions, EEF/target coordinates and AABBs remain in `geometry_02` and
`audit_details`. No original joint-limit violations were found.

The selected collider preserves the original main wall extents and bottom/top,
fills the cavity, and omits protruding handles. Those changes are suitable only
for the specified exterior bilateral carry task. Coupled IK reaches290/290 ON
frames with smaller arm corrections and accelerations than the exact shell
(C3). Separate grasp-marker colliders are not geometrically necessary.
Selection was documented in `CANDIDATE_SELECTION.md` before integrated tests;
no production asset was replaced.

## Matched physical comparison

`physical_02`:104 CPU cases, eight constructions×13. Each uses the same .005s
timestep, implicitfast integrator, Newton10/line-search20, pyramidal cone and
original contact material. Six .1s analytic experiments per candidate:
gravity-only,4N translation and .02Nm torque, each at yaw0/.6. Seven3s contact
experiments: settle/drop/external-lift at both supports plus floor drop. All
initial poses and inputs are archived per case, with actual COM, angular motion,
contact wrenches, distances and solver results.

All56 contact cases meet the declared final settling criteria; CPU warnings0.
Maximum analytic velocity discrepancy is1.29e-8; settled reaction balances
candidate weight to better than3e-11 relative. A24N vertical pulse for.25s after
.5s settling lifts2kg candidates83.6245mm and the1.5kg control315.417mm. Removal
of the force produces recontact and stable settling. This is external lifting,
not natural hand carrying. Peak floor/drop penetration around10.48mm is transient;
passing tail stability does not mean penetration was absent.

Supports remain fixed .20×.20×.04m boxes at
(−1.466734,−.159754,.660550) and(1.381318,−.284395,.625123)m, original yaw/frames
retained. Reference endpoint lower surfaces begin10.000021mm above support tops.
Per-world support offsets, reset and original floor experiments are independent
of elevated diagnostic hand fixtures. No support transform was silently changed.

Recommendation: retain the versioned single-body2kg candidate for further
bounded control work, with baseline contact parameters. Production promotion
requires the remaining C4 and provenance gates. Neither successful isolated
dynamics nor a simpler collider establishes P7 readiness.
