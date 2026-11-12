# P6-B reference compatibility

**REFERENCE_COMPATIBILITY = FAIL for the tested native scene/reference pairing.**
The evaluation implementation and measurements are complete, but whole-motion
native geometry and bounded original-PD replay do not support required bilateral
contact. This decision is separate from P6-A's PARTIAL physical-validity decision.

Evidence root E: `outputs/hoi/contact_supervision/contact_validation/`.
The original accepted Part B NPZ remains byte-identical:
`bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`.
All 840 samples retain the original global bool contact: ON260–549 (290), OFF550
samples. No per-hand labels, shifts, smoothing, target edits or retargeting.

## Geometry and timing

P4/P5 geometry and frame conventions are reused. The full 840-frame native
collision audit is in `geometry_02/reference_geometry.json` and its contact log;
`geometry_summary.json` contains compact results. P4's FK-vs-recorded wrist
agreement and 15 mm internal EEF marker finding remain unchanged.

| Original-reference ON geometry | Left | Right |
|---|---:|---:|
| EEF-target median | 41.802 mm | 42.610 mm |
| EEF-target RMS | 48.867 mm | 51.710 mm |
| EEF-target maximum | 96.097 mm | 118.220 mm |
| Worst native pair overlap | 37.223 mm, frame466 | 44.995 mm, frame438 |
| ON frames with overlap >1 mm | 189/290 | 228/290 |
| ON frames with gap >2 mm | 86/290 | 54/290 |

**Zero of 290 ON frames** place both hands within M6.5's [−1,+2] mm surface band.
The targets remain on side3/side4 walls; the EEF marker is not the collision
surface. Independent P6-A shallow bilateral IK succeeds for a different diagnostic
pose, so these reference discrepancies cannot establish that native bilateral
contact is universally impossible. No continuous force-feasible reference
reachability or learned-policy success is claimed. Historical support endpoints
remain 10 mm above physical contact; nominal gravity settling is not exact
reference-pose consistency.

P4.1 cites the manifest's explicit `bilateral_force_required=true`, establishing
AND as the task-required aggregate during ON260–549. Both OR and AND are reported
regardless of their scores; any-hand OFF contact remains visible.

## Bounded reference-driven physical replay

Six deterministic native-PD episodes use original joint targets and unchanged
actuator/action mapping, free robot root and free Box. Reset poses/twists come
from the original accepted reference; object twist is a separate P1 reset cache.
No root aid, Box force, attachment, post-reset Box assignment, learned policy or
optimizer is used. Each episode runs in four translated worlds. This is bounded
reference-driven replay, not P8 full-task evaluation.

| Reset frame | Requested steps | Completed steps per world | Result |
|---:|---:|---:|---|
| 0 | 50 | 35 | LS1024 at solve144, world0, 0.720 s |
| 250 | 60 | 60 | no solver flags |
| 400 | 50 | 50 | no solver flags |
| 530 | 60 | 60 | no solver flags |
| 550 | 60 | 37 | LS1024 at solve151, world1, 0.755 s |
| 800 | 39 | 39 | reaches frame839; no outgoing frame839 action |

There are **1,124 valid world-transitions**, covering **244 distinct reached
reference frames**. Four translated worlds are repeated controlled episodes, not
four independent demonstrations. Duration totals below sum world-episode time.
Requested coverage was 1,276 world-transitions; eight belong to two incomplete or
solver-invalid final intervals, and 144 were not executed after the stops.
No missing frame is silently treated as a match or mismatch.

For each complete interval, P2 records `(state_t, action_t) → state_t+1`, four
solved force samples and `[object_contact[t],object_contact[t+1]]`. Evaluation uses
**the reached frame t+1** for both end-state geometry and the global reference
comparison. The verifier checks stored intent and state continuity exactly.
All complete solver-valid measurements remain in the denominator, even when
tracking is poor. Both aggregates use the identical validity mask. Failed final
samples and reasons remain in per-case `samples.json`/`result.json`; the incomplete
force history never becomes a contact label.

`replay_rows.json` contains frame IDs, actual robot/Box state84, EEF/target positions,
distance, original global intent, independent contact, force peaks, occupancy and
validity. Raw per-case intervals retain mean force, impulse and all primary vectors;
contacts include all required MuJoCo fields. `replay_summary.json` includes events,
durations and fixed-definition sensitivity; `analysis_02` adds slip/penetration.

## Calibrated contact agreement

Use P6-A τL=**0.07434010469 N**, τR=**0.07375101818 N**, and strict occupancy
`>0.25` (at least two of four solves). These values were selected independently
of reference results. There are **480 valid ON** and **644 valid OFF** rows.

Confusion matrices use reference rows [OFF,ON], measured columns [OFF,ON].

| Aggregate | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Any hand (OR) | 624 | 20 | 351 | 129 | 66.99% | 86.58% | 26.88% | 41.02% | 61.88% |
| Bilateral (AND) | 632 | 12 | 428 | 52 | 60.85% | 81.25% | 10.83% | 19.12% | 54.48% |

| Measurement | Left | Right | Any | Bilateral |
|---|---:|---:|---:|---:|
| All-row contact ratio | 10.68% | 8.27% | 13.26% | 5.69% |
| ON contact ratio | 20.83% | 16.88% | 26.88% | 10.83% |
| OFF contact ratio | 3.11% | 1.86% | 3.11% | 1.86% |
| ON binary-contact duration | 2.00 s | 1.62 s | 2.58 s | 1.04 s |
| OFF binary-contact duration | 0.40 s | 0.24 s | 0.40 s | 0.24 s |
| Mean ON substep occupancy | 0.20052 | 0.13281 | — | — |
| Mean OFF substep occupancy | 0.02640 | 0.01863 | — | — |
| Contact onsets / losses | 28 / 40 | 26 / 38 | 36 / 48 | 12 / 24 |

ON observation duration is 9.60 world-seconds; OFF is 12.88. Force-bearing substep
durations ON are 1.925/1.275 s, OFF 0.340/0.240 s. Onset/loss counts never cross
resets or gaps. **73.13%** of ON rows miss any-hand contact; **89.17%** miss required
bilateral contact. Unintended any-hand OFF contact occurs on **20/644** rows.
The global reference label is not presented as independent per-hand ground truth.

## Penetration, tracking and mismatch classification

The following uses constraint records, includes retained failure diagnostics,
and reports maximum depth across worlds; mean depth and >1 mm duration are world0.

| Reset | Max hand depth | Mean contact depth | Time >1 mm | Max continuous >1 mm |
|---:|---:|---:|---:|---:|
| 0 / 800 | 0 | no hand contacts | 0 | 0 |
| 250 | 6.504 mm | 0.971 mm | 0.095 s | 0.040 s |
| 400 | 30.088 mm | 3.353 mm | 0.180 s | 0.050 s |
| 530 | 38.817 mm | 17.633 mm | 0.140 s | **0.135 s** |
| 550 | 36.826 mm | 16.710 mm | 0.075 s | 0.075 s |

Frame530 replay contains persistent significant overlap under the predeclared
>1 mm/≥0.1 s diagnostic criterion. High force is not successful grasping. Peak
hand forces across the bounded cases reach approximately **145.92 N left /
139.64 N right**. Last completed Box-position errors in one translated world are
0.900 m (reset250), 1.021 m (400), 0.631 m (530), 0.081 m (550); endpoint-only
resets settle about 0.0101 m. Full per-world states and slip remain available.

| Classification | Evidence and limit |
|---|---|
| Reference hand-pose reconstruction | Native FK reproduces accepted wrist poses; no new conversion/index error found. Recorded relative poses intersect native colliders. An upstream reconstruction/retargeting cause is plausible, not proven. |
| Box trajectory reconstruction | Original endpoint support gap and trajectory-dependent hand/Box separation persist. The source-authoring cause is insufficiently documented; no corrected trajectory inferred. |
| EEF/target coordinate mismatch | No new transform/order mismatch found; P4/P5 exact conventions reused. Root is not COM. |
| Collision-surface mismatch | Confirmed: internal EEF marker versus finite capsule and wall surface; zero marker error cannot certify collision-free contact. |
| Contact timing discrepancy | Measured missed ON and unintended OFF contact are explicit; cannot separate source timing from pose/control errors with this evidence. No label shift attempted. |
| Insufficient tracking/control execution | Confirmed in unassisted PD replay: large Box errors, falling robot and contact loss. This is not a trained balancing controller. |
| Physical scene/solver issue | P6-A independently identifies COM plausibility and coupled floor LS failures. Replay0/550 also stop on LS1024. These findings do not explain every geometric mismatch. |
| Insufficient evidence | No measured real-object inertial model, continuous force-feasible source reconstruction or learned-policy demonstration. No uniquely causal per-frame diagnosis claimed. |

The measurement pipeline is validated; the compatibility outcome remains FAIL.
No aggregate was selected to maximize agreement, and no failed episode or
unmatched valid frame was removed to improve a metric. No automatic reference,
target, solver or asset correction was made.
