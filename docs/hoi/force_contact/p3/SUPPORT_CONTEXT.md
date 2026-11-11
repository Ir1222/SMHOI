# P3.4 support-context evidence and decision

**Identifiability PASS; learned Force Head readiness is not established.**
64 simulator transitions (32 pairs) were captured through the unchanged P2
recorder with measured xi33 and refined reference hash. Native assets, gravity,
actions and solver settings remain unchanged. Within each pair both roots translate
1m in X, leaving canonical86/action29/xi33 equal while fixed supports stay put.
16 pairs have no hand contact, 8 left-hand and 8 right-hand fixtures. Initial states
are controlled placements, not natural motion. No state is overwritten within a
control interval. All capacity guards passed. Full raw labels/quaternions/states,
identities, actions and xi are in support_02/transitions.npz with metadata.

Canonical difference max2.38419e-7. Paired object differences reach .00358564m,
.00238539rad, .263008m/s, .134734rad/s; measured force component differences reach
18.20434N left and18.01209N right. Identical canonical inputs cannot uniquely predict
these outcomes. This is physical evidence, unlike P0's algebraic translation probe.

A fixed ridge diagnostic (regularization1e-4, train-only standardization) predicts
one-step residuals from joint canonical/action/xi, with and without candidate18.
24 whole pairs train;8 whole pairs validate. It is an identifiability diagnostic,
not the production WM, not an additional optimization framework and not a substitute
for P3.8 joint-model validation. Object errors are against the constant-twist prior;
force labels are the actual P2 interval means.

| Held-out metric | No support context | Candidate18 |
|---|---:|---:|
| Object normalized MSE | .00486310774 | .000124138570 |
| Position RMSE m | .000840153 | .000218867 |
| Orientation RMSE rad | .000309496 | .000243811 |
| Linear velocity RMSE m/s | .0680351 | .00607503 |
| Angular velocity RMSE rad/s | .0254047 | .0179951 |
| Left force component MAE N | 1.25843 | 1.97066 |
| Right force component MAE N | 1.24432 | 1.93464 |

The linear diagnostic worsens force prediction; retain this negative result. It
cannot model all support×hand interactions. Context removes a demonstrated physical
ambiguity and improves held-out object prediction; this does not guarantee better
Force Head generalization. Later learned model validation must still pass separately.

**Decision after experiment:** select18 = two manifest-order support root transforms,
position3 relative to current robot root in heading axes and orientation6 relative
to that heading. Recompute on every predicted state. Geometry/material constants
are fixed by M6.5 asset SHA256; reject incompatible asset metadata on load. xi33 stays
unchanged. This is the option expressly allowed by P3.4, not a research amendment.
No assertion that this is the smallest or universally best encoding is made.

Failure/correction history: support_01 failed before simulation because the diagnostic
omitted the mandatory load_manifest path; fixed to DEFAULT_ASSET. support_02 completed
all64 transitions. Its initial modulo4 validation split accidentally included only
right-hand contact pairs. Reanalysis holds out pair IDs6,7,14,15,22,23,30,31 (last two
of each block8), covering both sides; training IDs and both comparisons are saved in
support_balanced_analysis.json. Original support_02 metadata/log remain unchanged.
No simulator data or old P0–P2 evidence was overwritten.
