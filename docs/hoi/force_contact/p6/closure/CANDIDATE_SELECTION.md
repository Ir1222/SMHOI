# Candidate selection before integrated validation

Selected diagnostic construction: **single_body**,2kg homogeneous solid,
half extents(.09,.17,.135)m, geom/COM position(0,0,.125)m relative to the
unchanged HDMI bottom-plate root. Dimensions .18×.34×.27m. Original root trajectory,
support transforms and contact intent remain fixed. This is an explicit simulation
approximation, not a claim about a measured real bread box. Production assets are
not replaced.

The SMTrack single-free-body/one-collider construction is useful as a baseline;
its .30×.32×.36m dimensions are unsuitable here:151/290 original ON frames have
non-hand overlap. Its1.5kg mass is an independent control, not selected by contact
score. Keeping2kg isolates construction/inertia changes from the nominal baseline.

Eight constructions completed104 matched CPU physical cases. All support/drop/
external-lift/recontact cases satisfy the declared settling criteria; analytic
force/torque errors are below1.3e-8. This eliminates no motion feasibility issue
by itself. The exact disjoint seven-piece shell is a valid alternative: its mass
properties count the original solid union once, removing0.000656m³ duplicate
component volume from the previous0.006728m³ component sum.

Both the single-body and exact-union shell reach290/290 geometric ON frames under
the same coupled840-frame arm IK, joint bounds, original root/Box motion and
unchanged [−1,+2]mm independent surface gate. Single-body maximum/RMS arm
correction is.133715/.026120rad; shell is.218467/.029167rad. Maximum arm acceleration
is33.95 versus112.82rad/s². No active self-collision or excessive non-hand overlap
is found; single-body minimum non-hand clearance is23.996mm.

The single-body candidate is preferred for this outer-surface carrying task:
it preserves the original main wall extents and bottom/top, avoids redundant
corner contact surfaces, needs less arm correction and yields a smoother feasible
candidate motion. Filling the cavity and omitting external handles is a deliberate
approximation. It would be unsuitable if handle use or interior interaction became
a task requirement. The full-envelope solid unnecessarily fills handle protrusions;
the pad construction adds unsupported narrow grasp surfaces. **Separate grasp-marker
collision geoms are not needed for the demonstrated native bilateral geometry.**

Evidence: E/candidates/*.json, E/geometry_02, E/physical_02, E/ik_single_02 and
E/ik_shell_02, where E=outputs/hoi/contact_supervision/p6_closure. Earlier failed
IK attempts remain. The chosen candidate is kinematically feasible; dynamic
hand-driven lift/carry and original-floor reliability still require C4 testing.

Solver profile for the integrated comparison is the opt-in, source-hash-pinned
`supermimic.hoi.solver_precision.bracket_precision_guard`. It adds a cost-improving,
sign-bracketed float32 precision termination condition to the line search. Baseline
dt=.005, iterations10, ls_iterations20, friction, solref and solimp remain. Original
and selected-object baseline failures show construction alone does not remove
LS1024. Production defaults and dependency files remain unchanged.
