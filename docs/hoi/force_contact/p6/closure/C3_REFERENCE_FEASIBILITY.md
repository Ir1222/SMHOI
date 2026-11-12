# C3 — Native G1 reference feasibility

**Original reference: FAIL for continuous native bilateral surface compatibility.**
**Versioned candidate: PASS for bounded kinematic feasibility; dynamic execution
remains FAIL/PARTIAL as detailed in C4.** No per-hand reference labels were created.

Evidence root E=`outputs/hoi/contact_supervision/p6_closure`. Original840-frame
HDMI NPZ, original global binary intent260–549, and all P4/P5 semantics remain.
The original EEF uses wrist-yaw body plus rotated(.05,0,0)m; Box targets are
(0,±.17,.14)m in the bottom-plate root frame. The COM is not used as the target
origin. Reconstructed FK, target rotations and native pitch/yaw colliders were
checked independently; no uniform coordinate shift explains the failures.

`geometry_02` measures all20 constructions over840 frames. The original collider
and original motion have min hand distances−37.223/−44.995mm and ON maximum gaps
28.835/28.319mm. The selected solid preserves side-target positions, but original
right-hand overlap reaches50.749mm. Point-marker agreement therefore cannot
stand in for collision-surface correspondence. Native capsules, wrist geometry,
hand-pose reconstruction and trajectory timing all matter.

## Contact-constrained trajectory solve

`scripts/trajectory_ik.py` solves one jointly coupled840×14-arm-variable sparse
nonlinear least-squares problem. It minimizes deviation from the original arms,
targets native capsule surfaces on both sides, penalizes excess overlap and
non-hand/self collision, and couples consecutive first/second differences.
Hard joint bounds apply. Robot root, legs, waist and Box poses stay unchanged.
No frame is independently optimized and then called a continuous trajectory.
Analytic contact Jacobians were checked against finite differences at260/400/500;
relative errors below1.8e-10 for the selected solve.

Acceptance is unchanged: both native surfaces in[−1,+2]mm throughout all290 ON
frames, hard joint limits, no non-hand/self penetration worse than1mm, and reported
full-sequence velocities/accelerations. These kinematic gates do not establish
actuator tracking or force-bearing stability. The optimizer uses a stricter
.8mm interior overlap penalty to leave room inside the1mm acceptance boundary.

| Result | Selected solid | Exact union shell |
|---|---:|---:|
| Continuous ON feasible frames |290/290|290/290|
| Maximum arm correction rad |.133715|.218467|
| RMS correction rad |.026120|.029167|
| Maximum arm speed rad/s |3.44945|recorded per frame|
| Maximum arm acceleration rad/s² |33.9474|112.8234|
| Worst surface penetration mm |.82731|.96585|
| Minimum non-hand clearance mm |23.9956|20.554|

The selected solve converged by its ftol condition. Maximum correction is7.66°.
No enabled self-contact was detected; a recorded minimum self-distance0 is the
empty-contact sentinel, not measured zero clearance. Initial attempts are retained:
`ik_single_01` reached185/290 and `ik_shell_01`173/290, with slight gate violations.
They are not reclassified as successful. `ik_single_02` and `ik_shell_02` use the
same stricter penalty and unchanged independent acceptance.

The separately versioned `reference_candidate_v1/reference.npz` SHA256 is
`f38071e5347403f9eff993dd37884bf374e0c003400156dc4cb87d3d2b856cb0`.
Only14 arm joint positions/velocities and associated body FK/velocity arrays
change. Root/leg/waist, Box trajectories, frame count/fps and original bool
`object_contact` remain byte-identical. Its manifest sets `training_ready=false`.
This file is a candidate reconstruction, not the accepted original Part B NPZ.

## Mismatch classification

| Category | Evidence and conclusion |
|---|---|
| Reference hand-pose reconstruction | Native FK has large gaps/overlaps; small coupled arm changes remove them geometrically. Supported contributor. |
| Box trajectory reconstruction | Original root/trajectory retained.10mm endpoint support gap and high speeds are documented; absent real-object ground truth prevents unique causal attribution. |
| EEF/target coordinate mismatch | Root/COM/EEF transforms explicitly verified. No evidence supports a global frame correction. Semantic point markers are not surface constraints. |
| Collision-surface/native hand mismatch | Direct capsule–cuboid measurements show this. Candidate IK uses actual surfaces, not marker coincidence. |
| Joint-space reachability | Candidate reaches all ON frames within bounds; original motion itself does not satisfy native surface constraints. |
| Temporal continuity | One coupled solve with adjacent velocity/acceleration terms; all840 derivatives reported. Continuous geometry is established only for the candidate. |
| Contact timing discrepancy | Original ON/OFF frames unchanged; physical onset/loss and missed/unintended contacts measured in C4. No shifting, smoothing or relabeling. |
| Tracking/control execution | Free-root native PD loses bilateral contact even with feasible targets. Strong supported limitation; no learned controller was tested. |
| Physical scene/solver | Original COM problem is independent of reference mismatch; LS failures independently reproduced and isolated in C2. Corrected candidate still fails natural carrying. |
| Insufficient evidence | No real material identification, no verified long-horizon natural carry, no complete public training provenance. |

The feasibility result supports preserving the original humanoid motion as far as
possible and making small explicit arm corrections in a separate artifact. It
does not justify silently retargeting production or changing contact semantics.
