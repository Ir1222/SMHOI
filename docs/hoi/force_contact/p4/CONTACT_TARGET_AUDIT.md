# P4 contact target and EEF audit

Geometric/software checks PASS; physically feasible exact-marker contact remains
unvalidated. No target/offset/asset/reference was changed. This audit precedes the
reward implementation; its evidence is `reward_validation/geometry_01/geometry.json`
and all 840 rows in `distances.csv` under `outputs/hoi/contact_supervision/`.

## Inputs and conventions

Original accepted Part B: `outputs/hoi/asset_audit/part_b/reference/reference.npz`,
SHA256 `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`.
`ReferenceClip` verifies the manifest and every array. It contains 840 frames at
50 Hz, global bool contact ON at 260–549 inclusive (290 ON, 550 OFF).
The historically refined NPZ is not used or modified.

Exact M6.5 manifest SHA256:
`1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`.
Every available declared source hash is verified; the missing historical V3 file
is recorded in JSON. The strict physical-scene loader is unchanged and remains
blocked by that missing provenance. Offline geometry does not approve that loader.

The installed native G1 is compiled through the existing `audit_robot.target_spec`
(same collision edits as native configuration). Joint order is resolved by names.
EEF body order is left/right wrist yaw; offsets `(0.05,0,0)` are wrist-local.
Box targets `(0,+0.17,0.14)` and `(0,-0.17,0.14)` are root-local, not COM-local.
Quaternions are wxyz; positions/distances are meters. Both geometry operands must
use the same world/environment-local frame. Current Box translation and rotation
transform the targets; reference Box poses are only inputs to this reference audit.

The local official HDMI checkout `32282f6` was inspected at
`/tmp/supermimic_hoi_audit_20261104/HDMI`: task YAML, `command.py:570` and
`rewards.py:eef_contact_exp`. Its transforms agree with this contract. Only its
position exponential is adopted: no force multiplier, tolerance clamp, gain or
outside-intent baseline is imported. The source's robot profile differs from native
G1, so its collision/contact success is not transferable.

## Measured geometry

| Check | Result |
|---|---:|
| Torch EEF versus independent MuJoCo FK, all frames/hands | max 1.7766e-15 m |
| Native FK wrist versus recorded float32 reference | max 8.1811e-8 m |
| Each target's exposed Box surface distance | 8.6736e-18 m |
| Left target surface | side3, outward +Y, below handle |
| Right target surface | side4, outward -Y, below handle |
| Each EEF point's signed distance inside yaw capsule | -0.015 m |
| Left ON EEF-target distance: min / median / max | 0.003437 / 0.041802 / 0.096097 m |
| Right ON EEF-target distance: min / median / max | 0.012056 / 0.042610 / 0.118220 m |

The existing `contact_points.csv` has 840 rows for each hand; its desired flags
match the global reference at every frame. Recomputed distances match within the
declared 2e-6 m float32 criterion. JSON records exact errors and ON/OFF distributions.
Candidate sigma values are pooled ON p25/p50/p75:
0.0252139293, 0.0423003221, 0.0628937157 m. These are test candidates, not final settings.

Both pitch and yaw bodies have enabled native collision capsules; all four are
recorded with compiled transforms in JSON. The yaw capsule starts at local x=.07
and has radius .035; the marker at x=.05 is internal by .015 m. The pitch capsule
is also relevant to actual hand-box collision and remains in P1/P2 sensing.
All seven box geoms are inspected as the same accepted compound rigid body.

## Target participation, bilateral semantics and physical limitation

Retain both original slots as explicitly supplied semantic proximity candidates:
their source ordering, surface location, reference approach and FK correspondence
are verified. No slot is dropped, remapped or made dependent on measured contact.
This justifies the two-target mathematical audit; it does not establish physically
feasible zero-distance targets or authorize training with the asset profile.

The manifest's `contact.physical_validation.bilateral_force_required=true` explicitly
establishes the task's bilateral physical-validation requirement during the global
ON interval 260–549. This is task evidence beyond merely having two target slots.
It is not independently annotated left/right ground truth or evidence that the
requirement was achieved. P6/P8 may interpret AND as task-required in that scope;
they must still preserve independent hand and any-hand/bilateral measurements,
especially any-hand unintended contact on OFF frames. P4 implements no physical
aggregate and never uses AND to replace the mean geometric reward.

Physical reachability is unresolved: a marker internal to a capsule placed on an
exposed wall entails overlap; closeness along a recorded trajectory is not proof
of collision-free attainability. Reuse the unchanged M6.5/contact-summary evidence:
native worst overlaps 37.22/44.99 mm and exact-marker-alignment overlap probes.
Those probes are geometric translations, not joint-reachability proofs. No IK,
trajectory repair, physics stepping or new physical contact evaluation was run.

`training_ready=false`, `M6_5_NO_GO_M7`, physical_hand_contact FAIL and
scene_support_validity FAIL remain unchanged. The generic reward can be tested
with these explicitly identified candidates, but a physically validated target
contract and any proposed target/offset change require further evidence and user
review under §1.3.3/P4.4. This audit does not silently approve a replacement set.
