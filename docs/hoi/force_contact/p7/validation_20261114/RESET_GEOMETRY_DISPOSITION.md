# Unresolved reference/collision compatibility — reviewable disposition

The current task explicitly protects the reference and collision assets. None was
changed. Numerical resource fixes and finite training do not eliminate overlap.

Evidence: independent named assignment of all840 NPZ frames exactly reproduces
training-cache qpos and signed contacts; native FK/reference maximum discrepancy
8.3563e-8 m. The maximum penetration is44.1007 mm at406 (right hand/Box);
frame0 foot/ground is12.2356 mm; foot/ground reaches24.7182 mm at155. Exact
immutable reference and native collision geometry therefore prescribe incompatible
reset poses. Raising solver iterations only permits the dynamics to solve these
initial contacts. Sparse reached-state observations still contain penetration.

Within current authorization, retain the inputs, record the limitations, and run
bounded engineering validation. This is the path executed. The unchanged pose
cannot simultaneously become geometrically nonpenetrating merely through a
correct reset mapping. Changing reset targets, filtering frames or removing
colliders would change the protected task and has not been done.

A possible follow-up, requiring explicit approval, is a separately versioned
reference-compatibility correction, preserving the current NPZ and all evidence:

1. Keep native collision assets, mass/inertia, supports, reward and binary contact
   labels fixed. Produce a candidate NPZ/manifest under a new directory; never
   overwrite the accepted refined NPZ or silently bind it to the existing profile.
2. Solve a minimum-motion-change reference projection over offending robot root
   and joint coordinates, with native FK/contact-distance constraints. Keep the
   Box trajectory fixed initially. Evaluate all840 frames and all collision pairs,
   including foot-ground and hand-Box; use the existing1 mm reporting band without
   increasing it. This is a proposed feasibility experiment, not a guaranteed fix.
3. Check joint limits, frame-to-frame velocity/acceleration, contact-intent timing,
   EEF-target correspondence, Box support and tracking deviations. Recompute the
   affected body transforms/velocities consistently, quantify the changes and
   display before/after frames0/155/406/447/530. If fixed Box/reward targets make the
   constraints infeasible, stop and present that evidence rather than adjusting
   them or the collision shape implicitly.
4. Independently validate CPU signed geometry and multi-world native resets, then
   repeat bounded Actor-only interaction, optimizer and play checks. A new profile
   and strict identity are required; existing checkpoints cannot silently resume.

This follow-up is not implemented or authorized by the current commit. It would
change the reference (and therefore the task distribution), so approval must
precede affected work. No new high-level Master Plan is introduced.
