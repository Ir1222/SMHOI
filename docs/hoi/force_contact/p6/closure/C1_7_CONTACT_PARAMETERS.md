# C1.7 — Source, compiled pair parameters and controlled response

Status: **PASS for the audit and supported controlled trials; PARTIAL GPU margin
coverage due to an explicit engine limitation.** No material or collision setting
was promoted to production. E=`outputs/hoi/contact_supervision/p6_closure`.

`contact_audit_04/compiled_audit.json` contains every original Box geom, native
robot/support counterpart, upstream SMTrack geom, raw/configured Toaster geom,
candidate group, and compiled two-geom pair probe. Probes preserve masks,
priority, dimensionality, friction, solmix, solref/solimp, margin/gap; pair mixing
does not depend on the probe shapes. Actual shape contacts and manifold counts
are independently recorded in the dynamic experiments.

## Effective settings

All values below are actually compiled in installed MuJoCo3.16. The upstream
lock pins mjlab1.4, MuJoCo3.10, mujoco-warp3.8.0.2 and Warp1.15. Thus the complete
Toaster/SMTrack compilations are current-engine reconstructions of pinned source,
not recovered historical checkpoint model binaries. That version limitation is
explicit. The exact upstream mjlab1.4 editor was loaded in isolation, because
installed mjlab1.6 requires structural fields the upstream code omits. No installed
package was changed. All five meshes and texture were fetched at the pinned
commit and verified against Git blob hashes; the complete asset was compiled.

| Geom/profile | friction(sliding,torsional,rolling) | solref | solimp | condim / priority | masks |
|---|---|---|---|---|---|
| Original HDMI seven geoms |(1,.005,.0001)|(.02,1)|(.9,.95,.001,.5,2)|3 /1|1 /1|
| Selected solid baseline |same|same|same|3 /1|1 /1|
| SMTrack source factory |(1,.005,.0001)|(.02,1), default|(.9,.95,.001,.5,2), default|3 /1|1 /1|
| Toaster marker after editor |(2,.5,.05)|(.004,1)|(.998,.998,.001,.5,2)|3 /0|1 /1|
| Toaster ordinary collision class |(1,.005,.0001)|(.004,1)|(.998,.998,.001,.5,2)|3 /0|1 /1|
| Toaster knob colliders |(100,.5,.01)|(.004,1)|(.998,.998,.001,.5,2)|3 /0|1 /1|
| Native hand geoms |(1,.005,.0001)|(.02,1)|(.9,.95,.001,.5,2)|1 /0|1 /1|
| Supplied combined proposal |(1,.1,.1)|(.001,1)|(.98,.99,.001,.5,2)|4 /1|1 /1|

Default margin/gap are0 and solmix1. The three supplied solimp entries compile
with trailing(.5,2). In hand–HDMI/SMTrack contacts, object priority1 supplies dim3,
friction5=(1,1,.005,.0001,.0001), and its solver parameters. Four pyramidal rows
are active. Stored torsional/rolling numbers do not create constraints at dim3.
For native hand–Toaster marker equal-priority probes, dim3 and maximum friction
produce(2,2,.5,.05,.05), while equal solmix blends solref to(.012,1) and solimp
to(.949,.974,.001,.5,2). The marker's .004 time constant is not automatically the
effective pair value. The actual source G1 hand structural settings agree with
the native counterpart used in these probes.

The upstream editor's `.*` selection also enables named region boxes and, through
unnamed lookup on this engine, the main Body001 visual mesh. Other visual
meshes remain disabled. This observed behavior is captured in the configured XML
and all28 compiled geom records. It must not be confused with marker-only friction
assignment, nor asserted as verified historical MuJoCo3.10 runtime behavior.
Markers overlap the compound object's occupied region; explicit inertias mean
these colliders do not add mass. Same-body contact filtering prevents internal
self-contact, but multiple overlapping external colliders can generate redundant
external contacts. The geometry-only native-shaped capsule probe in
`toaster_pair_audit.json` detects nine marker/occupied-geom overlap pairs and
generates four capsule-marker contact points at .5mm penetration, dim3, without
claiming arm reachability or carrying. The HDMI component overlap is quantified in C1; the candidate
solid has no internal geom overlap and the shell/pad candidates use disjoint pieces.

Audits02/03 compiled the raw spec before editing the same object; that changes
unnamed lookup in this engine and incorrectly enabled Body002 instead. They are
preserved as superseded fixture evidence. Audit04 uses independent fresh specs
and reproduces upstream Entity's editor-before-compile order. Other profile and
dynamic trial results are unaffected.

MuJoCo pair mixing and dimensionality are documented in the
[official contact parameter reference](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters).
Installed Warp `constraint.py:95–125` enforces positive timeconst≥2dt when
refsafe is enabled. For the higher-priority Box/native-hand pair, proposed
solref(.001,1) is stored as written but acts at10ms for P6 dt5ms. Equal-priority
Box/support mixing instead gives .0105s before the clamp; every support-pair
probe is recorded separately. Torsional coefficients have a length scale; choosing .1 without
material/contact-patch evidence is not an accuracy calibration.

## Margin and gap: current engine semantics

In installed3.16, detection uses margin+gap and force inclusion uses margin.
Direct probes verify: margin0/gap2mm detects a1mm-separated pair with no constraint
rows; margin2mm/gap0 activates that pair; margin2mm/gap2mm detects a3mm-separated
pair with no rows. Increasing gap alone does not create supporting force.
This differs from the stale margin-minus-gap wording in the installed mjlab
configuration docstring. Installed Warp `collision_core.py` and the current
[MuJoCo collision driver](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_driver.c)
agree with the observed behavior. Pinned local source hashes accompany the audit.

## Matched experiments and results

Ten profiles change one parameter group at a time: baseline; friction only;
condim4 only; friction after condim4; solimp only; solref only; margin2mm only;
gap2mm only; gap2mm after margin2mm; combined proposal as a separate follow-up.
The combined trial has no single-factor causal attribution. Geometry,2kg mass,
COM/inertia, state, controls, dt5ms, Newton10/LS20 and pyramidal cone are fixed.

Forty CPU experiments each run2s: flat-floor settling, .1m drop,30N lateral pulse,
4Nm yaw pulse. Pulses run .5–.7s. Full state/contact wrench/solver data are in
`contact_physical_01` and `contact_physical_gap_01`. No CPU warnings occurred;
final reaction is19.62N throughout. Transient penetration remains significant.

| Profile | Drop max penetration mm | Lateral test max penetration mm | Final lateral displacement m | Peak yaw speed rad/s |
|---|---:|---:|---:|---:|
| Baseline |10.481|10.617|.176776|6.0746|
| Friction only, dim3 |10.481|10.617|.176776|6.0746|
| Dim4 only |10.444|11.108|.175234|6.0272|
| Larger torsion after dim4 |10.444|11.108|.175234|5.5945|
| Solimp only |10.719|14.662|.172968|5.2813|
| Solref only |6.684|6.862|.171561|7.3097|
| Margin2mm |4.209|9.197|.178566|6.0746|
| Gap2mm only |10.481|10.617|.176776|6.0746|
| Margin2mm plus gap2mm |4.209|9.197|.178566|6.0746|
| Combined follow-up |6.684|4.689|.146746|6.3754|

Friction-only dim3 trajectories are identical to baseline: the changed dimensions
are inactive. Dim4 increases rows/contact4→6 and changes twist/constraint response;
rolling remains inactive. Solimp does not monotonically reduce penetration under
dynamic lateral loading. Solref changes response and drop penetration, subject to
refsafe. Margin reduces measured geometric penetration partly by activating force
at a positive surface separation; that is not proof of improved realism. Gap-only
leaves these force trajectories unchanged while allowing inactive contacts.

Eight supported profiles also run the same complete floor scene on GPU for1s,
both solver variants, and .5s elevated null/left/right/bilateral native PD holds.
Every original-solver floor profile fails with LS1024 only. Every guarded floor
and native hold finishes; native force aggregation agrees exactly with direct
contact extraction. Full dimensions/forces/occupancies/penetration and first
failure states are in `contact_gpu_01`, `contact_gpu_02`,
`contact_gpu_failure_details` and `final_audit/verification.json`.

Both nonzero-margin GPU profiles are rejected during model conversion by the
baseline MULTICCD Box–Box restriction. Their CPU results are valid, but GPU
dynamics are **NOT TESTED**, not PASS. No CCD flags, timestep or solver tolerances
were altered to bypass the restriction.

Recommendation: keep baseline contact parameters for the selected candidate.
Larger stored torsional/rolling coefficients are unnecessary at dim3; activating
torsion needs a physical contact-patch justification. New solref/solimp materially
change response but are uncalibrated. Grasp score, reduced penetration or fewer
warnings cannot identify real contact parameters. None of these trials establishes
natural hand-driven carrying.
