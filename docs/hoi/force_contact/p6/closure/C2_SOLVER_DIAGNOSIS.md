# C2 / C2.1 — Original floor solver diagnosis

**FLOOR_SOLVER_RELIABILITY: PARTIAL.** The failure is reproduced and a narrowly
scoped precision correction passes the bounded tests. The default production
solver remains unchanged and still fails; universal reliability is not claimed.
Evidence root E=`outputs/hoi/contact_supervision/p6_closure`.

## Exact meaning of1024

Installed versions: mjlab1.6.0, MuJoCo/mujoco-warp3.16.0, Warp1.18.0,
torch2.14.1+cu130, RTX4090D. Installed solver.py SHA256:
`1bb26d51451a065b9e183acacfbd5e59d85df1eb46f17951aa3d1727db6853ff`.
`mujoco_warp.OverflowType.LS_ITERATIONS = 1 << 10 = 1024` is a bit flag.
It is **not a count of1024 iterations**. The actual baseline options are Newton,
main limit10, line-search limit20, main tolerance1e-8, line-search tolerance.01,
dt.005s, implicitfast and pyramidal contacts. `Option.ls_iterations` supplies
the compile-time loop bound in `_linesearch_iterative_kernel`. When the loop
exhausts without the accepted termination condition, the kernel atomically ORs
the LS bit into `data.overflow`; its warning says to increase the limit beyond20.
The warning text is not a justified remedy by itself.

All active flags are decoded individually in `overflow_audit_01/summary.json` and
`final_audit/verification.json`. Enum bits are NEFC1, NJMAX_NNZ2, BROADPHASE4,
NARROWPHASE8, CCD16, HFIELD32, CONTACT_MATCH64, NVMAX128, EPA_HORIZON256,
ITERATIONS512, LS_ITERATIONS1024, TACTILE2048. The three exact original failing
states reproduce these complete world bitmasks:

| Original mode | Actual overflow by world | ncon/world | nefc/world | solver_niter | Active contact dimensions |
|---|---|---:|---:|---:|---|
| Complete scene |[0,0,1024,0]|32|128|6|32×dim3|
| Robot-floor isolation |[1024,0,0,0]|30|116|3|28×dim3,2×dim1|
| Reference frame0 start |[0,0,1024,0]|16|64|6|16×dim3|

`overflow & LS_ITERATIONS` is true exactly in the failing worlds. No other bit
or unknown bit is active. Sliding friction is active at dim3; torsional and
rolling constraints are absent. Capacity is ample. Full qpos/qvel/qacc,
warmstart, controls, actuator forces, applied/constraint forces, contact identities,
distance/position/frame, effective friction/solref/solimp and constraint addresses
are archived before/after the replay. CPU gradients/line slopes and warnings are
also retained in `solver_01`.

## Controlled isolation

All baseline experiments retain original solver/material options, with four
worlds and equivalent CPU model/state/control shadows.

| Mode | Original solver | Guarded original model | Guarded selected model |
|---|---|---|---|
| Complete original-support scene |FAIL at140 solves|PASS600|PASS600|
| Robot-floor, Box/support contacts disabled |FAIL167|PASS600|PASS600|
| Floor-contact ablation |PASS600|PASS600|PASS600|
| Box-support, robot remote at z50m |PASS600|PASS600|PASS600|
| Elevated hand/Box release |FAIL318 after eventual floor impact|PASS600|PASS600|
| Original reference frame0 start |FAIL158|PASS600|PASS600|

600 solves=3s. Elevation does not establish original floor success. Complete
diagnostic initial robot elevation is3m; reference0 uses its original floor state.
Original support placements and per-world offsets are retained. Original native
colliders include seven capsules per foot; many simultaneous foot contacts are
present at failure. The failing minimum signed distances are−33.730mm in the
high-drop complete fixture,−7.894mm robot-floor and−2.986mm reference0. These are
actual penetrations, not evidence of a clean static standing controller. The
reference0 reproduction shows a large initial drop is not necessary for the
numerical failure. Original COM is also not necessary: robot-floor isolation and
the selected Box reproduce it. Selected-object baseline complete/robot-floor/
reference0 fail at140/164/144 solves respectively.

## Causal evidence and smallest correction

Printf-only instrumentation in `solver_trace_01` shows a convex derivative sign
bracket with decreasing cost, but its alpha width has reached float32 resolution.
Examples: complete bracket width1.2e-7, slopes−27.5/+1.375, acceptance slope
1.0739; robot-floor width1.1e-7, slopes−11.47/+.2656 versus.1827; reference0
width1.34e-7, slopes−.559/+.01416 versus.00739. Further steps cannot reliably
satisfy the slope test. The regularized constraint condition numbers in paired
CPU reconstruction reach about1904/6006/1834, with positive mass-matrix eigenvalues.
This identifies finite-precision line-search termination as the reproducible
failure path; contact combinations and conditioning expose it. It does not prove
that every physical penetration or future solver failure has the same cause.

`src/supermimic/hoi/solver_precision.py` supplies an **opt-in**, process-scoped,
source-hash-pinned guard. It accepts a sign-changing derivative bracket only when
width≤the existing8-float32-epsilon alpha scale and at least one endpoint improves
cost. It retains the existing best endpoint and all failure reporting. Main/line
iterations, tolerances, cone, timestep, inertia and material remain unchanged.
It fails closed on any unaudited dependency revision, refuses nesting/concurrency,
and restores the factory on exit. No dependency file is patched on disk.

`failure_replay_02` starts CPU and GPU from each exact saved pre-failure state,
including warmstart and identical controls. All three baseline cases flag1024;
all guarded cases clear it. Guarded maximum next-qvel errors against CPU are
8.02e-6,1.45e-5,6.54e-6; all next-qpos errors<1e-4 and qvel errors<1e-3.
The first paired replay revealed a diagnostic activation bug: Warp caches this
factory by function name. The corrected factory has a distinct name so a cached
baseline cannot replace it or contaminate later baseline scenes. The failing
`failure_replay_01` is preserved. Final guard experiments were rerun afterward.

Long trajectories are not numerically identical: high-impact elevated drops
diverge substantially after impact, including about1.72 maximum qpos difference
over3s. The exact one-step CPU agreement is the correction check; long-horizon
contact chaos is a separate limitation. Instrumentation also changes rounding
and which world/solve first flags. First-failure time is not used as a material
accuracy ranking.

## Parameter controls and final-forward failures

Eight supported C1.7 parameter profiles reproduce LS-only failure with original
solver settings; all eight guarded1s complete-floor runs and .5s native holds
finish without overflow. `contact_gpu_failure_details` saves full states for
these failures. Condim4 profiles can fail early during Box-support interaction,
so additional friction constraints can expose the same numerical path even
before robot-floor impact. The original floor failure is not cured by object
construction or by the proposed material settings.

Some initial parameter runs flag during mjlab's final forward solve, after its
four physics substeps. Diagnostics now separately wrap this forward stage and
archive `final_forward_states.json.gz` as well as substep states. A null
`first_flag_sample` in the earlier run with a nonzero reported exception is not
a successful simulation. Reruns preserve all failures and capture whichever
stage first flags. The baseline observation/force substep order is unchanged.

Nonzero-margin GPU profiles are **NOT TESTED dynamically** under the fixed
baseline collision configuration: Warp rejects Box–Box margins with MULTICCD
(and has an additional NATIVECCD restriction). The rejection and traceback are
retained; no CCD flags were silently changed to make those trials run.

Recommendation: review/upstream the precision termination correction and retain
its opt-in status until broader high-impact/contact regression coverage and
production provenance are resolved. The bounded evidence supports this numerical
correction, not a blanket declaration of physical training readiness.
