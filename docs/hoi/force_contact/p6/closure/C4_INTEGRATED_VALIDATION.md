# C4 — Integrated independent physics and reference replay

**Sustained native bilateral lift/carry: FAIL.** Simulation completion and improved
kinematic geometry do not change this decision. No training or learned-policy
performance was evaluated. E=`outputs/hoi/contact_supervision/p6_closure`.

All integrated tests consistently use selected `single_body`,2kg, baseline contact
parameters and the hash-pinned precision guard. Original native G1 actuators,
free robot/Box roots, P1 filtered sensors and P2 interval recording are reused.
The Box receives no external wrench or post-reset state assignment. No root
assistance, force reward, observation change or differentiable force path is added.

## Calibration and measurement reuse

P6 measured thresholds are retained: **left .0743401046875112N, right
.07375101817770177N**. Actual contact requires net hand-force magnitude strictly
above its threshold in strictly more than25% of the four5ms physics substeps:
at least2/4. This is a contact detector, not a load-support threshold. The stored
P2 diagnostic occupancy at.001N is independently recomputed but is not used as
the calibrated binary label.

The original measured null maxima were0N. Thresholds are half the low-load
training5th percentiles(.148680/.147502N), with100% first-contact detection at
held-out roll.31rad. The occupancy cutoff rejects one-substep transients and
retains two-substep events. This provides a measured noise/load separation;
genuine subthreshold or one-substep impacts and net-force cancellation remain
explicit detector limitations.

`integrated_calibration_02`: nine rotations/depths×four null/left/right/bilateral
fixtures all reproduce the expected binary classification with the new object.
Raw filtered sensor force agrees exactly with contact-force extraction. The
final audit independently recomputes6424 world intervals of Box-frame mean force,
world impulse, peak magnitude and diagnostic occupancy. Maximum errors are
2.13e-5N,2.05e-7Ns,1.53e-5N and0 respectively, within float32 tolerances. Four
substeps are aligned before final-forward diagnostics; no threshold is optimized
against reference agreement. Original P1/P2 sign/frame/filter/reset/history tests
are reused, rather than rerun without a regression reason.

## Reference-driven replay, original versus candidate

Both use identical six(start,count) windows:(0,100),(250,125),(350,125),
(475,125),(550,100),(740,99), four worlds each. These are bounded native-PD
reference-driven tests, not a continuous840-frame policy rollout. All requested
windows finish with zero overflow. Each comparison has2696 valid transitions,
1260 reference ON and1436 OFF, reaching599 distinct end frames. Overlapping
windows and repeated worlds are not independent statistical samples. The same
validity mask is checked against recorded transition validity. Each [t,t+1]
interval is compared to unchanged `object_contact[t+1]`. No shifts, smoothing,
relabeling or mismatching-frame exclusion occur.

P4.1 establishes bilateral intent for260–549; AND is therefore the required
aggregate there. Both OR and AND are reported regardless of which looks better.
Confusion matrices use reference rows OFF/ON and actual columns OFF/ON.

| Reference / aggregate | TN,FP,FN,TP | Accuracy | Precision | Recall | F1 | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|
| Original OR |1420,16,1208,52|.545994|.764706|.041270|.078313|.515064|
| Original AND |1428,8,1244,16|.535608|.666667|.012698|.024922|.503564|
| Candidate OR |1395,41,928,332|.640579|.890080|.263492|.406614|.617470|
| Candidate AND |1430,6,1256,4|.531899|.400000|.003175|.006299|.499498|

The candidate improves any-hand recall but reduces required bilateral recall.
It is not a successful physical reconstruction for carrying. Overall left/right
actual-contact ratios are .014837/.019288 original versus .058976/.083086
candidate. ON left/right mean substep occupancy is .019841/.024603 original and
.104762/.138294 candidate; OFF occupancy is .006267/.006964 and .006442/.020891.

Across the25.2 world-seconds of ON intervals, OR contact duration is1.04s original
versus6.64s candidate; bilateral duration .32s versus .08s. Across28.72 world-seconds
OFF, unintended OR duration is .32s versus .82s and AND .16s versus .12s.
OR onset/loss counts are24/36 versus60/72; AND0/12 versus2/10. Counts respect
episode boundaries. Full per-hand durations and force-bearing substep durations
are in `analysis_02/*_agreement.json`, with every transition in compressed rows.

Native PD loses posture/Box tracking: maximum final-window Box position error is
2.398m original and2.540m candidate. This is substantial control execution failure,
not an accurate carried object trajectory. Peak left/right forces are
158.16/152.38N original and272.65/137.32N candidate; higher force is not a success
metric. State, EEF/target geometry and actual contact are kept separate.

`replay_geometry` reconstructs geometry from all5392 recorded end states, without
integrating or changing them. ON median EEF-to-actual-Box-target distances are
(.5119,.5441)m original and(.3958,.4386)m candidate. Distances to the unchanged
reference Box targets are(.6726,.6905)m and(.6614,.7014)m. Per-frame actual EEF,
both target sets, native surface distance, non-hand clearance and measured contact
are archived. End-state FK is distinguished from the four-substep force interval.

## Penetration and unintended body contact

| Replay | Hand contact records | Max depth mm | Mean depth over hand contact records mm | >1mm world-duration s | Longest >1mm episode s |
|---|---:|---:|---:|---:|---:|
| Original |752|38.148|13.515|1.000|.075|
| Candidate |1556|11.023|.927|1.425|.065|

Durations count a world/substep once, not once per manifold point. Mean depth is
contact-record weighted and is not a duration average. The original starts with
large geometric overlaps; these cannot be dismissed as tiny numerical noise.
The candidate reduces depth but still violates1mm under dynamic execution.
Box-support peak depths are7.961/13.308mm, Box-ground24.392/20.871mm and non-hand
robot–Box10.018/11.442mm original/candidate. Non-hand >1mm durations are2.170/3.175
world-seconds, longest .120/.125s. Full category-specific min distance, contact
positions/frames/IDs, depth distributions and frequencies are retained. No
penetration reward or new SDF is introduced.

## Independent native manipulation attempt

`independent_controller_01` solves a separate temporally coupled250-frame arm
trajectory at original support0: .5s contact,1s planned8cm lift,1s planned6cm
lateral carry,1s lower,1.5s release. Its175 planned-contact frames are geometrically
feasible; that does not move the simulated Box. `integrated_independent_02` executes
249 policy intervals/4.98s in four worlds using native PD and free roots.

All four worlds fail the declared lift criterion(≥3cm rise unsupported for≥.2s)
and carry criterion(additionally≥3cm lateral travel with bilateral contact for
≥.2s). World0 rises only23.668mm, has0 qualifying unsupported-lift/carry duration,
and loses both hands throughout planned carry. Peak hand forces are68.63/73.64N.
The Box falls off the support and eventually settles on the floor. Its horizontal
travel .586m and final displacement .678m are fall motion, not carrying or correct
placement. Final COM speed≈2.1e-6m/s establishes floor settling only.

External24N lift/release/recontact in C1 remains a separate physical-response
test. Kinematic IK and external forcing are never counted as natural carrying.
The remaining limitation is sustained free-root tracking and load-bearing
bilateral control. No training was started to mask this gap.
