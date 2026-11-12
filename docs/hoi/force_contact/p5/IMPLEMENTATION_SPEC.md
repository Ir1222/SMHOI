# P5 implementation specification — reward and integration contracts

Date: 2026-11-12. **Specification for user review; source implementation is not authorized.**
This change adds only this document. The user explicitly requested the specification,
isolated numerical validation, a commit, and a stop before implementation. No P5
runtime code, training, optimizer step or physical rollout was executed.

The selected specification is Scheme A: original robot tracking, four Box tracking
terms, unchanged P4 reference-conditioned proximity, and the original two
regularizers. Real MuJoCo contact/force remains independent diagnostic evidence.
There is no Contact Decoder, Contact/Force Prediction Head, Contact BCE, extra
state, or Actor/Critic observation dimension.

## 1. Baseline, authority and disposition

- Checkout: `main`, preparation HEAD `e2451cb652dcdc703799c2845784610696a0cc15`
  (`docs: review HOI reward integration before P5`). No applicable `AGENTS.md`
  was found in ancestors or the inspected repository directories.
- Master: [SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md](../SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md),
  SHA256 `268047aa4e431866bf708c6da2b271ef48d44dfec561458d09e11f76df563a41`.
  All 2,381 lines were read. This specification does not edit the master.
- Previous implementation: P4 commit `fd8d28a`, [REPORT](../p4/REPORT.md),
  **PARTIAL**. Read the complete [Reward Integration Review](../p4/REWARD_INTEGRATION_REVIEW.md).
  P4 software success does not establish physical target validity.
- Read the complete historical [P3 REPORT](../p3/REPORT.md) and current
  [architecture-cleanup report](../p3/architecture_cleanup/REPORT.md), commit
  `bab5132`. The active model is 166D/82D; historical force-era results remain
  historical. P3 stays **PARTIAL**, including FP32 multi-step migration failures.
- Read P0 [coverage](../p0/REQUIREMENT_COVERAGE.md), [report](../p0/REPORT.md),
  [closure](../p0/CLOSURE.md), [file map](../p0/FILE_CHANGE_MAP.md), and relevant
  architecture, dependency, dataset/asset and reference-implementation audits.
  Their superseded force/head/observation-extension proposals are not adopted.
  Sensor mechanics are relevant only to preserving the diagnostic boundary;
  no new sensor calibration or external-paper investigation is needed here.
- Numerical environment: local `.venv/bin/python`, Torch `2.14.1+cu130`, mjlab
  `1.6.0`, NumPy `2.5.4`; CPU, one Torch thread, seed 501. Installed mjlab math
  SHA256 `85e267ee202dc8dc2b340be07ec7abe9b10da9b82c3bf33e16889f422fcf9ca8`.
- Original accepted reference, explicitly selected for this specification:
  `outputs/hoi/asset_audit/part_b/reference/reference.npz`, SHA256
  `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`.
  It is distinct from P3's historical refined reference `f6ba446c…`.
  No loader/default or historical experiment is redirected to a different NPZ.
- M6.5 asset SHA256 remains
  `1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`.
  `training_ready=false`, `M6_5_NO_GO_M7`, hand overlap and support failures remain.
- The pre-existing untracked `.vscode/` belongs to the user and is excluded.

**Decision status:** mathematical definitions and interface decisions below are
selected for review. This is not a P5 implementation PASS or approval of training
hyperparameters. Research choices are exposed, not silently installed. User review
of this document is the next gate before any source change.

| ID | Prior clause / observed discrepancy | Selected resolution and classification |
|---|---|---|
| D1 | P4.5 says preserve an existing object reward; none exists | Define the four Box terms in §3. Research-level completion of missing reward semantics; proposed here, pending implementation approval. |
| D2 | P5.2 requests next-frame scoring; original `ImagStep` uses current-frame scoring | HOI scores every reached-state term and termination feature at `f+1`; robot-only retains current-frame scoring. Explicit HOI timing decision, not a global bug fix. |
| D3 | Original orientation helper produces NaN gradients at agreement | Denominator-safe, formula-preserving helper, injected only into HOI robot reward and used by Box reward. Numerical implementation adjustment, explicitly reviewed; no dependency patch or default robot change. |
| D4 | Original rollout clamps past clip end without masking | HOI valid transitions stop at frame 839; separate physical survival and task-boundary continuation. Natural clip end has zero tail value, timeout/horizon can bootstrap. Boundary interpretation of P5.5, detailed in §6. |
| D5 | P0 identified fixed robot FK/Critic slices; Replay start loses clip identity | Named robot projection and clip-aware start/reference adapters; narrow shared hooks default to original behavior. Implementation adjustments, no dimension changes. |

## 2. Physical, frame and reference contracts

Let `s_k` be the current imagined physical84 and `a_k` the clean Actor action29.
Imagination uses that action as the dynamics action, as in the original path;
it does not add a new action-noise model. Measured WM fitting continues to use
executed actions. Control timestep `h = dt = 0.020 s`, physics `.005 s × 4`.
Reject a reference/control-timestep mismatch rather than resampling it implicitly.

| Slice / tensor | Meaning and convention |
|---|---|
| `s[...,0:71]` | Original robot physical state: qpos36, qvel35 |
| `s[...,71:74]` | Box **root-origin** position `p`, environment-local world axes, m |
| `s[...,74:78]` | Box unit quaternion `q`, **wxyz**, body-to-world active rotation |
| `s[...,78:81]` | Box root-origin linear velocity `v`, world axes, m/s |
| `s[...,81:84]` | Box angular velocity `omega_b`, Box body axes, rad/s |
| `R(q) omega_b` | The same angular velocity in world axes |
| WM input / output | canonical86 + support18 + action29 + normalized xi33 = 166; residual82 |
| Other contracts | support18 recomputed from current state; xi33 episode-constant; physical84 unchanged |

`JointStateIO` already subtracts each environment origin from both root positions.
References must use the same local world origin. Do not subtract the origin twice.
Any separately approved scene rotation/translation must transform robot, Box,
references and supports consistently. Translation leaves velocities unchanged;
rotation rotates world velocities. There is no new scene transform in this spec.

COM velocity is **not** the tracked quantity. For body COM offset `r_com`,
`v_COM = v_root + omega_world × (R(q) r_com)`. The derivative of NPZ
`object_pos_w` supplies `v_root`; applying a COM correction to just one side is wrong.
Canonical86 and P3's normalized object prediction errors are not reward errors.
In particular, the WM's body-component angular error remains unchanged.

### 2.1 Immutable reference derivative cache

The original 840-frame NPZ contains position, quaternion and global bool contact,
but no Box velocities. Derive constants in a **separate in-memory cache** from
this NPZ, never from imagined motion or simulator measurements. No persistent
cache or new NPZ was created in this task.

For clip-local frame `f`, length `T >= 2`, define

\[
\ell=\max(f-1,0),\quad u=\min(f+1,T-1),\quad \tau=(u-\ell)h,
\]
\[
 v_f^{ref}=\frac{p_u^{ref}-p_\ell^{ref}}{\tau},\qquad
 \omega_{w,f}^{ref}=\frac{\operatorname{LogRot}
 (q_u^{ref}\otimes(q_\ell^{ref})^{-1})}{\tau},\qquad
 \omega_{b,f}^{ref}=R(q_f^{ref})^T\omega_{w,f}^{ref}.
\]

`LogRot` is the shortest rotation vector, radians. Compute world increments by
**left relative multiplication** `q_u * conjugate(q_lo)`. The alternative
`conjugate(q_lo) * q_u` expresses the increment in the **lo-frame**; rotate that
vector by `R(q_lo)` to recover world components. It is not generally the angular
velocity expressed in the body frame at `f`. The latter always uses `R(q_f)^T`.

At interior frames this is a centered, second-order estimate for smooth resolved
motion. Endpoints use first-order one-sided estimates over one interval. For a
changing rotation axis the log of the two-sample increment is an approximation
to instantaneous velocity; it is exact for constant spatial twist below the log
branch limit. It is not componentwise quaternion differentiation and not a
promise of exact derivatives for arbitrary motion.

Validate finite positions/quaternions, finite positive FPS, at least two frames,
and unit norm within the existing reference tolerance `1e-5`; reject degenerate
or invalid quaternions. Normalize **copies** in FP64, then calculate derivatives
in FP64; construct `tau` in FP64 explicitly (integer indices times a Python float
otherwise produce FP32 in this environment). Cast the completed cache once to
the runtime device/dtype. Runtime cache tensors have `requires_grad=False`.
The cache contains `p[840,3], q[840,4], v_world[840,3], omega_world[840,3]` and
unchanged `contact[840]` bool; body omega may be retained for audit, not subtracted
directly in the reward.

Independent sign flips of any source quaternion leave all rewards/derivatives
unchanged away from the unavoidable pi cut. No temporal sign unwrapping, smoothing,
clipping, endpoint zeroing, extrapolation or per-hand relabeling is applied. Check
both adjacent **and two-frame stencil** rotation magnitudes below pi; adjacent
angles below pi alone do not guarantee an unambiguous centered stencil. Pose
samples cannot rule out unobserved rotations between samples.

Cache provenance must include source/manifest SHA256, clip identity/order, frame
mapping, FPS/dt, root-origin/world conventions, wxyz, normalized-copy policy,
derivation version (`central-world-log-v1`), compute/storage dtype and units.
No reuse of a cache with mismatched metadata. The existing reset-only
`reference_object_twist` and `ReferenceResetCache` remain untouched.

## 3. Complete reward specification

All quantities in this section are evaluated at the **same reached state**
`s_{k+1}` and the **same evaluated reference frame** `f_k+1`.

### 3.1 Four always-active Box rewards

\[
\begin{aligned}
 E_p &= \|p-p^{ref}\|_2^2,\\
 E_R &= \|\operatorname{LogRot}(q^{ref}\otimes q^{-1})\|_2^2,\\
 E_v &= \|v-v^{ref}\|_2^2,\\
 E_\omega &= \|R(q)\omega_b-\omega_w^{ref}\|_2^2,\\
 r_{o,j}&=dt\,\beta_{o,j}\exp(-E_j/\sigma_{o,j}^{2}),
 \quad j\in\{p,R,v,\omega\},\\
 r_{Box}&=r_{o,p}+r_{o,R}+r_{o,v}+r_{o,\omega}.
\end{aligned}
\]

The position and linear velocity use Box root origins and world axes; orientation
uses the squared shortest geodesic angle; angular velocity compares two **world**
vectors. If only reference body omega is stored, convert it with **reference**
orientation, `R(q_ref) omega_ref_body`, while predicted body omega uses **predicted**
orientation. Equal body components at different orientations are not equal world
angular velocities. Conversely, equal world angular velocities must have zero
error even when body components differ.

`beta_o,j > 0` and `sigma_o,j > 0`, all finite; sigma units are respectively
m, rad, m/s, rad/s. Squared vector errors sum three coordinates, without division
by 3 or averaging over robot bodies. No robot-relative Box reanchoring, product
of scores, contact gate, WM-loss scale or force factor applies. Tracking stays
active during all ON/OFF/release frames. Use normalized quaternion copies for
Box geometry/reward; invalid inputs fail validation, not an epsilon-normalized
zero quaternion silently interpreted as a pose.

The four terms are differentiable on their valid domain except the orientation
cut at exact pi. The zero-error squared-angle derivative is finite and zero.
Angular velocity reward deliberately also has an orientation gradient through
`R(q) omega_b`; do not detach that rotation.

### 3.2 Original robot terms and regularization, exactly once

Reuse [tracking_reward_terms](../../../../src/supermimic/envs/mdp/rewards.py)
and [AnalyticHeads](../../../../src/supermimic/world_model/heads.py), retaining
these existing defaults and equations:

| Positive robot term | Error `E` | sigma | weight |
|---|---|---:|---:|
| Global anchor position | squared distance | .3 m | .5 |
| Global anchor orientation | squared shortest angle | .4 rad | .5 |
| Relative body position | mean of body squared distances | .3 m | 1 |
| Relative body orientation | mean of body squared shortest angles | .4 rad | 1 |
| World body linear velocity | mean of body squared differences | 1 m/s | 1 |
| World body angular velocity | mean of body squared differences | 3.14 rad/s | 1 |

Each contributes `dt * weight * exp(-E / sigma²)`. The configured anchor is
**torso_link**, not the pelvis free root. Preserve the original XY/yaw reanchoring
of reference body poses; reference height/roll/pitch and world velocity targets
retain their original semantics. `r_robot_positive` sums these six terms.

\[
 r_{reg}=dt\left[-0.1\|a_k-a_{k-1}\|_2^2
 -2\sum_j\left(\max(q_j^{lo}-q_j,0)+\max(q_j-q_j^{hi},0)\right)\right].
\]

Here `q_j` denotes reached robot joint position, not Box quaternion. Joint-limit
violation is **not squared**. Preserve configured weights if the user explicitly
loads a nondefault original reward configuration. Previous imagined actions and
states retain their graphs; only the replay initial action is a constant.

### 3.3 P4 proximity is unchanged

Reuse [hoi/rewards.py](../../../../src/supermimic/hoi/rewards.py) byte-for-byte:

\[
 P_f(s)=c_f^{ref}\frac{1}{N_c}\sum_{i\in\mathcal E}
 \exp\left(-\frac{\|p_{EEF,i}(s)-[p+R(q)t_i^{Box}]\|_2}{\sigma_{pos}}\right).
\]

`c_ref` is the original single global bool label: ON frames260–549 inclusive,
290 ON and 550 OFF. Use current imagined robot FK and current imagined Box pose.
P4's two candidate targets are `[0,+.17,.14]` and `[0,-.17,.14]` m, ordered left,
right, with wrist-local offsets `[.05,0,0]` m. No target/offset/participation change
is authorized. Their physical validity remains limited: the EEF markers lie
15 mm inside native capsules. P5 tests can exercise this audited candidate geometry
without declaring it physically accepted for training.

Raw `P_f` is dimensionless, bounded in [0,1], with **no dt**. OFF gives exact zero
value and gradients. P4 uses a finite zero subgradient at exact zero distance;
the norm proximity is not classically differentiable there. The target mean is
not a physical bilateral-AND measurement. No actual contact/force, OFF repulsion,
phase gate, future intent or reference Box substitution is introduced.

### 3.4 One complete scalar and its scale contract

\[
\boxed{r_{total}=r_{robot\_positive}+r_{Box}+dt\,\beta_c P_f+r_{reg}.}
\]

**Selected composition convention:** call the unchanged P4 combiner with
`robot=robot_terms['total']`, `object=r_Box`, `proximity=P_f`,
`regularization=zeros_like(robot_total)`, `robot_weight=1`, `object_weight=1`,
`contact_weight=dt * beta_c`. Original `robot_total` already includes both
regularizers. Do not add `r_reg` again, multiply any already scaled term by dt,
or reduce robot tracking weight during contact. Log the two penalties separately.
The zero tensor above is bookkeeping, not removal of regularization.

The master notation is resolved as `lambda_r=1`, `lambda_o` absorbed in the four
effective `beta_o,j`, and `lambda_c=dt*beta_c`. Proximity `beta_c > 0` when enabled;
zero is permitted only for a named proximity-OFF ablation. All four Box weights
remain positive in that ablation.

**Required configuration, no implicit research defaults:** the four Box sigmas,
four Box weights, `sigma_pos` and `beta_c` must be supplied and persisted with the
reward version. Missing/nonfinite/nonpositive settings fail construction, with
only the named proximity-OFF exception. The reward formula and parameter contract
are finalized here; final experimental coefficients are **not** fabricated.
Master P4.4–P4.5 explicitly requires measured scale selection, and no trained-state
or Actor-gradient scale evidence exists yet. Synthetic fixture values may test
arithmetic but must be labeled test-only, never registered as training defaults.

For implementation review, report each physical error distribution, each weighted
contribution and state/action/Actor gradient norm before selecting experimental
coefficients. At error norm `sigma`, each normalized score is `exp(-1)`; its maximum
radial sensitivity is `sqrt(2/e) / sigma` at error `sigma/sqrt(2)`. Multiplication
by `dt*beta` gives the actual gradient scale. This is a check of units and
sensitivity, not evidence that a particular sigma is suitable for the task.

P4's measured candidate `sigma_pos` values remain .0252139293/.0423003221/
.0628937157 m, with raw ON means .24483194/.40023489/.52512410; original positive
robot ON mean .0999999033. They are candidates, not a selected deployment setting.
The positive per-step bound is

\[
 B_f=dt\left(5+\sum_j\beta_{o,j}+\beta_c c_f^{ref}\right),\quad
 B_{max}=dt\left(5+\sum_j\beta_{o,j}+\beta_c\right).
\]

Replace 5 by the actual original positive weight sum for a nondefault config.
Regularization is nonpositive, so `r_total <= B_f`; there is no finite lower bound
from this equation. Do not clamp negative rewards or normalize the training
return/TD targets by this bound.

## 4. Reference lookup and original network compatibility

### 4.1 Clip identity is mandatory

[HOIReplayBuffer](../../../../src/supermimic/hoi/replay.py) stores `(clip, local_frame)`;
[MotionLibrary](../../../../src/supermimic/envs/mdp/motion_library.py) concatenates
clips and uses global frames. Original `StartBatch` has no clip field. Passing
its HOI frame directly to the library is therefore wrong for clips after clip0.

Create an HOI start envelope retaining explicit `clip_id`, `local_frame`, library
`global_frame`, reference SHA, environment/episode identity and boundary metadata.
These are control metadata, **not neural observations or physical state fields**.
Use shared Replay row selection/history gathering through the small extraction
hooks in §9; do not copy Replay sampling or guess clip identity from a local index.

At construction, bind each HOI clip to a library clip using an explicit ordered
registry with matching SHA, length, FPS, joint/body names and order. No matching
by filename stem alone. Reject missing, duplicate/ambiguous or mismatched identities.
For each row:

\[
 g=\mathrm{clip\_start}[c]+f,\quad 0\le f<T_c,\quad
 \mathrm{frame\_clip}[g]=c.
\]

The registry may map external replay clip IDs to different library IDs, but that
mapping must be explicit and validated. Use the same reference source for robot,
Box and label; mixing original robot data with a refined robot sidecar is rejected.
This specification selects the original Part B reference only for its P5 checks.
Future use of the refined reference requires its own validated provenance binding.

For transition `k`, gather **only valid** `(c,f+k+1)` rows for all reward terms and
termination features. Call P4 `reference_contact_at` with the clip-local label
tensor and that exact local index. Gather robot `ReferenceFrame` with the matching
global index. Do not clamp an invalid reward index, concatenate clips and cross
into the next one, or count repeated last frames as transitions. The cache gather
and registry validation occur outside the compiled step; pass tensor references
and bool gates explicitly into it.

### 4.2 Action goal and scoring reference are distinct

| Operation at step k | State | Reference |
|---|---|---|
| Actor observation/history | current robot state/history | original proprioception |
| Actor goal | current frame `f+k` | original `goal_window(g_k,10)` |
| WM dynamics/support | current physical84 and action | no reference contact input |
| Robot/Box/proximity reward | reached `s[k+1]` | common `f+k+1` |
| Termination height/tilt | reached robot root | pelvis ref at `f+k+1` |
| Stored next Critic root error/goal | reached `s[k+1]` | `f+k+1` |

Keep the original Actor goal progression and its existing final-frame repetition
inside the fixed-width **goal window**. Such goal padding does not authorize any
reward frame past the clip boundary. Distinguish goal-window padding from invalid
reward transitions. Original robot-only `imagine` continues to pass `ref_at(k)`
to its existing step and to use its original clamp behavior; HOI selection must
be explicit, never inferred from a changed global default.

P3 `JointWorldModelTrainer.references_for` already uses local `frame+1` and pelvis
reference pose, and trains its termination BCE using `physical_failure`, excluding
timeouts/clip ends. HOI imagination must match this training convention. Do not
shift P3 termination targets or use torso anchor data as pelvis termination data.

### 4.3 Exact physical84 → robot71 projection

Use the named `ROBOT = slice(0,71)` projection at **every** original robot consumer:

- `heads.robot_bodies(robot_next)`, `heads.reward(robot_next, ...)` and initial
  anchor FK. Passing all84 into `robot_bodies` supplies 48 qvel entries to FK
  instead of the required35 because it slices `s[...,36:]`.
- `analytic_observation(robot_state, prev_action, xi, q0)` for every replay-history
  and reached-state sample; preserve term-major flattening and noise semantics.
- Critic input is `wm.spec.robot.canonical(ro.states[...,ROBOT])`, preserving all
  leading time/batch dimensions. Do **not** call joint `canonical` and feed86 or
  take the first71 physical coordinates as though they were canonical features.
- Root-error and termination APIs receive the projected robot state/reference.
  Their current root slices happen to fit84, but explicit projection prevents
  hidden coupling to joint layouts.

Actor remains obs93 × history10 = **930**, goal **680**, output **29**, latent32;
Critic remains canonical71 + goal680 + xi33 + root-error4 = **788**. Existing
`anchor_obs=True` keeps its existing goal689/Critic797 option; do not enable it
or change checkpoint configuration implicitly. Policy/Critic parameter names,
weight shapes, normalization buffers and strict loading stay unchanged.

The unchanged Actor cannot observe Box-only deviations directly. The unchanged
Critic aliases joint states with the same robot/goal/xi/root-error and different
Box states. It estimates total return conditional on those existing inputs; it
is not a fully observed joint-state value. Its direct terminal gradient with
respect to Box state is zero; earlier Box→robot dynamics can still influence its
value. This approximation is an explicit risk, not permission to widen inputs.

## 5. Exact insertion point and reuse of the original loop

The insertion is the HOI adaptation of
[ImagStep._step](../../../../src/supermimic/imagination/compiled.py), immediately
**after `s_next = wm.step(s,a,xi)` and before selecting/returning the scalar reward**
(the current lines39–43). A new HOI step adapter will:

1. Receive validated evaluated-frame robot tensors, Box pose/world twist and bool
   contact in a plain-tensor compiled signature, plus `s,a,a_prev,xi`.
2. Call the existing joint `wm.step`: canonical86, support18 recomputed from `s`,
   action29, normalized xi33, shared residual82, existing physical84 integration.
3. Project robot71 and call existing robot FK/reward with the HOI-selected safe
   orientation-error function in §7. Reuse original body reanchoring/regularizers.
4. Compute four Box terms and call unchanged P4 `ProximityReward(s_next,c_ref)`.
   Recompute geometry each step. Its separate FK call may duplicate computation;
   retain that small cost rather than alter P4 or cache stale geometry.
5. Combine exactly as §3.4; return `r_total` in the existing reward slot with its
   graph retained. `term_vec` has the eight original keys followed by
   `box_pos, box_ori, box_lin_vel, box_ang_vel, proximity_weighted` (13 components).
   Report `total`, `box_total`, `robot_positive`, `regularization` and raw proximity
   separately; they are redundant diagnostic aggregates, never resummed as terms.
6. Evaluate termination against the same next-frame pelvis reference; produce the
   original obs93 and tracked anchor pose7 using robot71. Preserve the original
   six-item step return structure; expose geometric diagnostics for tests through
   an HOI diagnostic method, not additional learned observations.

`PolicyTrainer.__init__` currently constructs `ImagStep` directly; `update`
hard-calls `imagine`, and `_critic_inputs` uses the WM's spec. Supply explicit
optional component hooks/default methods for the step, rollout context and robot
Critic projection. The HOI adapter supplies them; the original defaults remain.
Use **one** shared `imagine` recurrence, Actor optimizer/update path, TD(lambda)
recursion and Critic optimizer loop. Do not copy `PolicyTrainer.update`,
`Trainer.run`, the Replay algorithm or the robot reward into an HOI fork.

The rollout context supplies reference batches, validity/boundary metadata and
term names. The default context keeps the original operations/order. The HOI
context uses §§4/6. Shared dynamic term logging must use the selected step's term
names, instead of the hard-coded eight-name tuple. No global monkeypatching.

`torch.compile` wraps the HOI step with explicit tensor arguments and fixed
configuration. Existing P4 and joint-WM tensor-dependent validation may cause graph
breaks. Preserve the checks/P4 source and measure/report graph breaks; do not claim
`fullgraph=True` success or change numerical formulas to force compilation.
Initial supported contract is actual eager/compiled parity, not a single graph
or guaranteed speedup. CPU tests here do not establish compiled/GPU parity.

## 6. Valid transitions, survival, terminal and bootstrap semantics

### 6.1 Stop at the last valid reached frame

For a start at local frame `f`, nominal horizon `H`, and optional **known** remaining
time budget `K`, use `L=min(H,T-1-f,K)` (omit K when unavailable). Require `L>=1`;
reject/resample terminal state rows and frame839 starts. Do not resample away
starts838 or late contact-boundary starts simply to avoid boundary tests.

`valid[k,n]=(k<L_n)` is metadata of shape `[H,N]`; states remain `[H+1,N,84]`.
Reward/reference lookup executes only for valid rows. At/after boundary, stop
advancing that row's state/history/frame; any rectangular storage padding has
zero reward and is excluded from term averages and Critic MSE. A neutral finite
placeholder may be used for compiled batching, but **never** by clamping an
out-of-range reference and counting it as real. Validate before gathering; avoid
computing invalid values and hoping multiplication by zero will repair NaNs.

From frame0 there are **839 transitions**, reaching frames1…839. Frame0 is an
initial-state diagnostic, not an extra action or reward. This does not rewrite
P8's obligation to evaluate all 840 states. From838 there is exactly one reward
against839; no transition839→840 and no repeated reward at839.

Replay terminal state rows have no outgoing transition. Their history may serve
a valid previous transition's target, never an imagination start. Keep inherited
within-episode history/backfill/previous-clean-action semantics and strengthen
HOI eligibility with clip continuity and transition validity. Do not use the
recorded next physical failure from a **different recorded action** as a known
failure of the counterfactual imagined action.

### 6.2 Physical failure and boundary masks are different

Let `d[k+1]=sigmoid(termination_logits)` be the existing predicted physical-failure
probability. Keep pre-step survival weighting:

\[
 C_0=1,\qquad C_{k+1}=C_k(1-d_{k+1}).
\]

A supplied, authoritative hard physical-failure event ends the valid segment at
its first reached state: shorten L to that step and set subsequent continuation
to zero. Retain its robot/object reason metadata separately. Do not threshold the
soft prediction to manufacture such an event. There is no differentiation through
physical sensors and no new failure head/feature dimension.

| End reason | Reward for the valid reached state | Tail value | Physical-failure label |
|---|---|---|---|
| Imagination horizon before clip end | included | `V(s_L,g_L)` | unchanged predicted d |
| Known time-limit truncation before clip end | included | `V(s_L,g_L)` | not set by timeout |
| Natural reference clip end | final reward included once | **zero** | not set by clip end |
| Known physical robot/object failure | included, weighted by pre-step survival | **zero** | true physical failure |
| Padding after any end | none | none | not a training label |

If timeout and clip end coincide, retain both reasons and zero the tail because
the task clip ended. Physical failure takes precedence for failure reporting.
This follows original `commands.py::_finalize_episodes`, which bootstraps timeout
only when `~terminated & ~clip_end`. Original imagination itself lacks this clip
mask; its repeated clamped terminal rewards are not a semantic precedent to copy.
The HOI extension preserves original survival/TD recursions within valid segments
and truncation bootstrapping, while giving finite-reference completion an explicit
zero tail. Robot-only imagination remains unchanged, including its historical gap.

Define `b_L=1` for horizon/time truncation with a remaining reference transition,
`b_L=0` for natural completion or hard failure. The HOI Actor objective is

\[
 J=\frac1N\sum_n\left[\sum_{k=0}^{L_n-1}\gamma^k C_{k,n}r_{k+1,n}
 +\gamma^{L_n} C_{L_n,n}b_{L_n,n}V_{L_n,n}\right],\qquad
 L_{Actor}=-J/H.
\]

Use **nominal configured H** for the original loss scale, not per-row division
by L (which would change late-clip sampling weights). Report valid lengths so
shorter tails are visible. `use_critic=False` removes only the terminal value.
For full valid horizons with `b=1`, this is exactly the original objective.

TD(lambda) uses the same scalar rewards, detached, with boundary value
`G_L=b_L V_L`. For valid `k<L`,

\[
 G_k=r_{k+1}+\gamma(1-d_{k+1})B_{k+1}
 \left[(1-\lambda)V_{k+1}+\lambda G_{k+1}\right],
\]

where `B=0` at natural clip end or a hard failure, otherwise1. Mask **both** the
one-step `V` term and recursive term at true task completion. Timeout leaves B=1.
Fit Critic only on valid pre-transition states. Do not encode natural completion
as a positive WM physical-failure label. In rectangular storage, the backward
recursion must initialize each row at its own L; invalid rows can reset the
running target to the stored boundary value and are excluded from the loss.
Original `gae_targets` keeps its existing no-mask fast path exactly.

### 6.3 Existing limitations that masks cannot fix

P2 stores aggregate `physical_failure`, timeout and clip_end; it does not record
separate robot/object failure causes or an always-available remaining timeout
budget. Do not infer an episode age across an overwritten ring or silently add
an arbitrary timeout. P5's default imagination is bounded by H and clip; the
optional K/reason interface can be tested with explicit fixtures. P7 physical
wiring must provide authoritative timeout/cause metadata if it uses those events.

The inherited termination head observes only pelvis height/tilt relative to the
reference (four scaled/signed-magnitude inputs). It cannot distinguish Box-only
failures for otherwise identical robot states. P5 must expose this limitation:
mask tests can prove handling of a supplied object-failure event, but cannot prove
prediction of that event. A new object-failure predictor/threshold would require
separate design approval and is outside this specification. P7/P8 must not claim
complete joint-failure coverage from the existing head.

## 7. Quaternion NaN: diagnosis and formula-preserving correction

Installed `quat_box_minus(q1,q2)` computes
`axis_angle_from_quat(q1 * conjugate(q2))`. The latter canonicalizes sign, computes
`angle=2*atan2(norm(xyz),w)` and selects between `sin(angle/2)/angle` and the series
`0.5-angle²/48`. At identity the unselected quotient is **0/0**. Autograd still
encounters that invalid derivative through `torch.where`; masking its result is
too late. This behavior matches PyTorch's documented
[division-by-zero autograd example](https://docs.pytorch.org/docs/2.15/notes/autograd.html#division-by-zero-in-autograd).
The installed version and actual local probe, rather than that newer documentation
version, establish the result here.

Use this local correction, preserving the installed sign rule, threshold, series,
multiplication order and all selected forward arithmetic:

```python
def safe_quat_box_minus(q1, q2):
    q = quat_mul(q1, quat_conjugate(q2))
    q = q * (1.0 - 2.0 * (q[..., :1] < 0.0))
    mag = torch.linalg.vector_norm(q[..., 1:], dim=-1)
    half = torch.atan2(mag, q[..., 0])
    angle = 2.0 * half
    large = angle.abs() > 1e-6
    safe_angle = torch.where(large, angle, torch.ones_like(angle))
    ratio = torch.where(
        large, torch.sin(half) / safe_angle, 0.5 - angle * angle / 48
    )
    return q[..., 1:] / ratio[..., None]
```

Only the **unused** branch denominator is changed. It is never zero; the selected
branch and its derivative away from the singularity remain the installed formula.
At identity the rotation-vector map has xyz Jacobian `2I`, and its squared norm
has zero first derivative. There is no reward epsilon floor, `acos` clamp,
quaternion-component MSE, detached gradient, `nan_to_num` or straight-through rule.

The existing local `rotvec_from_quat` also guards its quotient and is a valid
candidate on normalized inputs, but uses a different small-angle threshold and
series organization. Select the correction above to minimize numerical departure
from the **original robot formula**. Do not change the existing helper, WM loss,
P4 implementation or installed dependency to achieve this.

Future wiring, **only after review**: add an optional orientation-error callable
to `tracking_reward_terms` and `AnalyticHeads`, defaulting to the current
`_sq_quat_err`. The HOI component explicitly injects the safe squared-log error
for **both** anchor and relative-body orientation terms. The shared robot reward
body remains the sole implementation. No global replacement/monkeypatch is allowed.
Original robot-only values, nondegenerate gradients and behavior remain unchanged;
its known identity-gradient defect remains documented until a separate global fix
is authorized. P5 cannot claim finite HOI BPTT if it fixes only Box orientation
and leaves HOI's robot orientation terms on the defective path.

Quaternion sign invariance refers to values and physical tangent derivatives;
raw gradient coordinates at `-q` change sign relative to those at `q`. At exactly
pi the shortest-log axis is ambiguous and the squared geodesic loss is not smooth.
The implementation returns a finite branch value/derivative; this is not a unique
mathematical derivative. Near pi, test both sides without finite-difference steps
crossing the cut. FP16/BF16 and second derivatives are not covered by these checks.

## 8. Numerical evidence executed for this specification

Only isolated CPU tensor calculations and read-only NPZ/source inspection ran.
No model fitting, optimizer construction/update, physical environment or rollout
was used. Temporary scripts/results are under `/tmp/smhoi_p5_spec/`; complete
probe sources are embedded in Appendix A so the committed evidence is reproducible.

Commands (from the repository root):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/smhoi_p5_spec/probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/smhoi_p5_spec/supplement.py
```

### 8.1 Quaternion validation

Each dtype exercised 22 tangent cases: identity, negative identity, nonidentity
agreement and antipodal agreement; nonidentity-reference angles `1e-9, 1e-7,
.999e-6, 1.001e-6, 1e-5, .2, 1.3, pi-.001, pi+.001`, each with both predicted
quaternion signs. A separate exact-pi case checked finite output/gradient only.
Another 128 random pairs checked independent sign flips of either/both quaternions
and raw gradient sign covariance. FD used six tangent cases, including both sides
of pi with .01 rad clearance. FP64 gradcheck covered six cases including identity.

The probe assertions required safe gradients finite, forward agreement with the
installed helper exact in the 22 cases, analytic squared-angle/tangent errors below
`5e-6` FP32 / `1e-12` FP64, exact sign invariance, FD absolute errors below `2e-3`
FP32 / `5e-9` FP64. FD epsilon was `1e-3`/`1e-6`, respectively. These tolerances
are numerical checks in the stated dtype, not physical tracking acceptance limits.

| Measurement | FP32 | FP64 |
|---|---:|---:|
| Safe vs installed forward max difference, 22 cases | 0 | 0 |
| Max squared-angle error vs analytic value | 2.90324e-6 | 7.10543e-15 |
| Max tangent-gradient error vs analytic value | 2.38419e-6 | 3.55271e-15 |
| Max tangent FD gradient error | 1.37091e-3 | 2.62595e-9 |
| Independent sign-flip value / gradient covariance error | 0 / 0 | 0 / 0 |
| Exact-pi squared angle | 9.8696041107 | 9.8696044011 |
| Safe first gradients, all cases | finite | finite |
| FP64 gradcheck (`atol=1e-7, rtol=1e-5`) | not applicable | 6/6 passed |

Installed helper produced NaNs at identity/negative identity in **both** dtypes.
FP32 also failed nonidentity matches and the `1e-9` perturbations that rounded to
agreement. FP64 nonidentity products retained tiny residuals and had finite
installed gradients in this fixture; do not claim every equal-quaternion product
necessarily hits exactly zero in floating point. The correction handles both.

### 8.2 Derivative convention and accuracy checks

Constant translation and nonidentity constant world rotation, 41 samples at 50 Hz:

| Maximum component error | FP32 | FP64 |
|---|---:|---:|
| Linear velocity vs known translation | 1.14441e-5 m/s | 1.15463e-14 m/s |
| World omega vs known rotation | 8.34465e-6 rad/s | 2.04836e-14 rad/s |
| Body→world round trip | 2.08616e-7 rad/s | 8.32667e-16 rad/s |
| Alternating quaternion signs | 0 | 0 |

Assertions used component tolerances `2e-5` FP32 / `5e-13` FP64 and exact
sign-invariance checks. To test changing axes, the supplemental FP64 trajectory was
`q(t)=Exp(.8t e_x) Exp(-.7t e_y) q0`, whose spatial omega is
`.8 e_x + R(Exp(.8t e_x))(-.7 e_y)`. Translation was `(t³,t²,.4t)`.

| h (s) | Interior v error | Endpoint v error | Interior omega error | Endpoint omega error |
|---|---:|---:|---:|---:|
| .020 | 4.0000e-4 | .05960 | 5.22691e-5 | .00560 |
| .010 | 1.0000e-4 | .02990 | 1.30668e-5 | .00280 |
| .005 | 2.5000e-5 | .014975 | 3.26668e-6 | .00140 |

Halving h gives approximately 4× interior and 2× endpoint improvement, as asserted
(ratios 3.9–4.1 and 1.9–2.1). This supports the stated derivative accuracy and frame
convention. It does not validate unobserved physical velocities in the source NPZ.

### 8.3 Actual original 840-frame NPZ

All derived values were finite. Original quaternion maximum norm error was
`3.40621e-8`; no adjacent sign flips occurred. Injecting alternating signs gave
exactly unchanged derived world/body angular velocities. Reconstructing world
omega via the lo-frame body increment differed by at most `1.67842e-14 rad/s`.

| Quantity | Median | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|
| Adjacent rotation (rad) | 3.38231e-16 | .0353387 | .0422880 | .0470754 |
| Centered stencil rotation (rad) | 3.38231e-16 | .0708487 | .0848510 | .0940369 |
| Derived root speed (m/s) | 0 | .858666 | 1.347138 | 1.363064 |
| Derived world angular speed (rad/s) | 1.96262e-15 | 1.770594 | 2.120673 | 2.350921 |

Largest adjacent position displacement is `.0272733 m` on transition336→337;
largest adjacent rotation is on352→353. Largest change between neighboring
forward-difference velocities is `.0650930 m/s` at frame454; for angular velocity
it is `.188408 rad/s` at458. These are observed sampled-motion changes, not a
physical discontinuity diagnosis or grounds to smooth the reference.

Both endpoint linear velocities are exactly zero; endpoint angular components
are at most `2.78e-15 rad/s`, as produced by the formula, not forced to zero.
Endpoint results equal their one-sided stencils using the same normalized copies.
The maximum difference from the existing normalized-input reset helper is
`5.25472e-8` across twist components, attributable to its span dtype/finite
arithmetic; that helper remains unchanged. Direct FP32 vs FP64 derivation differs
by up to `5.96046e-8 m/s` and `5.96068e-6 rad/s` world omega. This motivates deriving
once in FP64 before storage casting, not silently modifying the reset path.

**Probe corrections recorded:** an initial exploratory run formed time spans in
Torch's default FP32 even for FP64 poses; its results were retained separately and
superseded by explicit pose-dtype spans. The first supplemental endpoint
bit-equality assertion failed because the independent oracle normalized a copy
once while `derive` normalized it again. The oracle now uses the identical
normalized copy; no numerical tolerance was loosened. These were probe issues,
not source/dependency edits or failed physical experiments. An existing
TorchScript deprecation warning appeared. No other final probe assertion failed.

## 9. Future file-change map and training-framework reuse

**This commit changes only this document.** The following is a reviewable map for
a later authorized P5 implementation; filenames below do not imply files exist.

| Future file | Minimal responsibility / justification |
|---|---|
| New `hoi/tracking_reward.py` | Four Box terms, required config, safe quaternion error, composition calling unchanged P4 combiner; all no-sensor Torch code. |
| New `hoi/reward_reference.py` | Immutable FP64-derived reference cache; explicit clip registry and checked evaluated-frame gathers. Reuse `ReferenceClip` validation and `MotionLibrary` robot references; no dataset conversion. |
| New `hoi/imagination.py` | HOI step and rollout-context adapters, robot projection, metadata/boundary selection, diagnostics. No duplicate rollout/optimizer/TD algorithm. |
| `envs/mdp/rewards.py` | One optional quaternion-error function argument, default original `_sq_quat_err`; same reward body/weights/reanchoring/penalties. Needed to remove HOI robot NaNs without copying reward or changing robot default. |
| `world_model/heads.py` | Optional error-function forwarding/configuration for HOI; defaults original. No FK/observation/termination formula change. |
| `data/replay_buffer.py` | Extract existing start-mask/row-history gather into reusable methods; optional metadata construction hook with original `StartBatch` default. Required to retain exact selected-row clip identity without a second sampler. Keep original selection distribution/history results. |
| `imagination/rollout.py` | Optional context for references, robot projection, validity/boundaries and term names; optional metadata fields in `ImagRollout`. Keep one recurrence and the unchanged default path. |
| `imagination/objective.py` | Optional lengths/boundary/validity inputs for the same Actor return and TD recursion. Preserve the original no-mask branch and `@no_grad` target behavior. |
| `imagination/policy_trainer.py` | Optional construction/rollout/projection hooks; boundary-value gather and valid Critic filtering; original Adam, allreduce, clips, update order, epochs and minibatches reused. Restore previous freeze flags safely. |
| New `tests/hoi/test_box_tracking_reward.py`, `test_reward_reference.py`, `test_quaternion_reward_numerics.py` | Equation/units, frame mapping and identity/sign/near-pi validation. |
| New `tests/hoi/test_imagination_parity.py`, `test_imagination_boundaries.py`, `test_policy_adapter.py` | Graph/parity, exact next-frame boundary sequence, masked returns, original network/Replay compatibility. |
| Future P5 `REPORT.md` / numerical evidence | Actual integration results under a new `outputs/hoi/contact_supervision/p5_validation/` directory; no rewriting historical results. |

P4 `hoi/rewards.py`, P4 tests/scripts/docs/evidence, current original `ImagStep`,
robot `StateSpec`, policy network definitions, existing P2 recorder/force fields,
P3 WM architecture/integration/loss, installed dependencies, references and assets
are unchanged. A subclass of existing HOI Replay can supply the new policy-start
metadata hooks without changing its P2 append/window semantics. No new learned
parameter is introduced by these adapters.

`training/loop.py`, shared schedules and CLI/config entry wiring are **P7 work**:
component factories must reuse `Trainer.run()` and supply the selected HOI
PolicyTrainer adapter. P5 exercises its interfaces with controlled fixtures and
backward calls only. Do not instantiate a physical Trainer or call a training
update to demonstrate this spec. Later P5 integration must not implement a second
training mainline or silently register an enabled HOI training configuration.

### 9.1 Reward logging and return-scale compatibility

P5 imagined logging uses valid rows only and retains units. Log all 13 additive
terms, total, raw/weighted proximity, ON/OFF counts, Box physical errors, valid
lengths, physical survival, and boundary reasons; zero ON count means undefined
conditional mean, not a successful zero. Detach **only logging copies**. Check
`sum(13 additive terms)==r_total` within dtype rounding.

Critic fits raw total returns in the same units as the Actor objective; no reward
standardization, new clipping or hidden scaling. A robot-only Critic's weights
are shape-compatible initialization, **not** an already calibrated HOI value
function. Do not restore robot-only optimizer moments or sampler return statistics
as though they were an HOI resume. Fresh HOI metadata must identify reward equation,
coefficients, derivative version and next-frame policy. P7 must calibrate Critic
values and validate bootstrap behavior before policy learning claims.

For eventual P7 real logging, recompute the same reward from captured **pre-reset
reached physical84**, action/previous action and evaluated reference. Simulator
reward-manager robot values follow the old command timing and cannot be relabeled
as this HOI total. Never add dt twice through the manager. Retain distinctly named
legacy robot-only telemetry if needed. Replay's P2 zero reward placeholder is not
the Actor reward source; imagination recomputes the total.

Original `Trainer.max_step_reward` is .1 under defaults and normalizes episode
returns by `.1 * length`. For HOI its injected bound must be `B_max`; also report
raw return and, optionally, `sum(r)/sum(B_f)` with its own label. Negative normalized
returns are valid. No unqualified comparison to original robot-normalized return.
Original return-based sampling uses `softmax(-temperature * bin_return/g_max)`:
HOI `bin_return` and any timeout bootstrap must use the **same total-return units**
as the HOI Critic, and sampler bins must record the evaluated reward frame.
Reset incompatible saved sampler state. Configure/record HOI `g_max` explicitly;
a documented scale-preserving candidate is
`g_max_HOI = g_max_robot * B_max / B_robot`, subject to measured return validation.
Do not silently feed expanded HOI returns to the old robot scale. These are P7
construction contracts, not permission to change `Trainer.run()` or sample now.

### 9.2 Freezing and Actor gradient propagation

During Actor BPTT, freeze WM/Critic **parameters** with `requires_grad=False`,
preserving and restoring their previous flags in `try/finally`. Do not wrap
imagination/terminal Critic evaluation in `no_grad`/inference mode. Do not detach
`s_next`, Box pose, FK, support context, observations, previous actions, survival
or Critic inputs. Reference/geometry constants alone are fixed.

For Actor parameters psi and frozen WM theta,

\[
\frac{d s_{k+1}}{d\psi}=
\frac{\partial F_\theta}{\partial s_k}\frac{d s_k}{d\psi}
+\frac{\partial F_\theta}{\partial a_k}\frac{d a_k}{d\psi}.
\]

Rewards, later observations/actions, differentiable survival and terminal Critic
input derivatives contribute to Actor gradients. The original action-rate penalty
also has direct derivatives through both adjacent imagined actions. Frozen weights
do not prevent input Jacobians. Existing stale `.grad` buffers are not proof of a
new gradient: clear or snapshot them in tests and assert no accumulation/weight
change during Actor backward. Restore flags even on exceptions.

The subsequent Critic path uses detached states/rewards/termination/goals/xi/error
and fixed TD(lambda) targets; Critic MSE must not update Actor/WM. WM fitting remains
a separate supervised step on physical data. P5 tests stop at backward and graph
checks, without Adam steps. P7 retains original optimizer/allreduce/clipping order.

A freshly zero-initialized dynamics output layer, including a migrated zero Box
head, may have zero action→Box Jacobian. Therefore nonzero state reward gradients
alone are not an Actor-gradient PASS. Use an explicitly labeled deterministic,
nonzero-coupling test model in future P5 tests; report the frozen initialized model
as a separate zero-gradient baseline. Neither fixture establishes learned physical
dynamics fidelity, which remains P7-A work.

## 10. Traceability and acceptance gates

| Requirement | P0 / prior evidence and current symbol | Decision / files | Verification obligation |
|---|---|---|---|
| Master4.3–4.4, P4.5; Box reward and velocities | P0 architecture “Reference object velocity”; P0 coordinate closure; Review§2; `reference_object_twist` reset-only | §§2–3, D1; new reward/cache modules | Four terms, root/world conventions, sign invariance, centered/end stencils, cache hash validation |
| P4.2–P4.5, P5.4; preserve proximity | P4 report/target audit, `ProximityReward`, original bool labels | §3.3; no P4 edits | Exact gate, current Box, two-target ordering; OFF value/gradient zero; no sensors |
| P5.2 exact next-frame alignment | P0 architecture “Clip-end gap”; Review§1.3; `references_for` already +1 | §§4/6, D2/D4 | Robot/Box/gate/termination/root error common frame; original robot path unchanged |
| P5.1/P7.5 interfaces | P0 verified dimensions/FK warning; `robot_bodies`, `_critic_inputs`, `StartBatch` | §§4–5/9, D5 | Physical84→robot71, canonical71, Actor930/680/29 and Critic788 strict shape/value parity |
| P5.5 terminal/bootstrap | P0 recorder discrepancy; P2 transition masks; original sampler timeout rule | §6; optional shared masks | Final reward once, no invalid gather/loss, natural zero tail, timeout bootstrap, separate physical reasons |
| P5.6 differentiability | Review§5; installed math source; frozen original `PolicyTrainer` | §§7–9, D3 | Identity/sign/small/nonidentity/pi checks; component→Actor paths, unchanged frozen weights |
| P5.7 compiled parity | Existing `ImagStep` tensor signature; P4 validation guards | §5; HOI adapter | Eager/compiled states/geometry/reward/gradients; report graph breaks |
| Master1.3.6/2/15 preservation | P0 file map/closure; P3/P4 PARTIAL; M6.5 FAIL | §9; only narrow hooks later | One original optimizer/TD/Trainer path, no new dimensions/head/physical data modification |

Acceptance tests required **after separate source-implementation authorization**:

1. **Reference and frame identity:** check all 840 source frames and unchanged NPZ
   hash; synthetic two-clip registry with different positions/labels must detect
   local/global mismatches even when numerical indices happen to be in range.
   Reject missing SHA/FPS/order, T<2, nonfinite/degenerate inputs, frame<0 or>=T.
   Never allow cross-clip derivative stencils or reference lookup.
2. **Boundary table:** starts258/259/548/549 must reach259 OFF /260 ON /549 ON /
   550 OFF respectively. Start838 produces one reward at839; start839 is ineligible.
   Log actual evaluated indices for every reward/termination/root feature. Original
   robot-only fixture retains its prior current-frame reward values/gradients.
3. **Reward algebra:** exact matching Box gives four maxima `dt*beta`; error=sigma
   gives `dt*beta/e`; monotonic controlled position/orientation/velocity perturbations;
   global angular-vector tests with differing body frames; tracking active OFF;
   original penalties once; component sum equals total. Robot reanchoring, weights,
   non-squared joint-limit penalty and action-rate gradient unchanged.
4. **Numerics:** rerun §8 matrix in actual implementation; finite identity/antipodal
   gradients for both robot orientation terms and all Box terms; identity must
   neither reduce reward maximum nor yield a spurious nonzero tangent gradient.
   Near-pi checks on both sides, no smoothness claim at pi. Unit FP64 checks use
   `atol=1e-10, rtol=1e-8`; FP32 algebra/analytic gradients use `atol=rtol=5e-6`.
   Retain the separately stated FD tolerances in §8.
5. **Projection/history/checkpoints:** invalid-width input is rejected; batched
   time layouts preserved; term-major original obs/noise/scales/action outputs
   match exactly for identical robot input and RNG. Original Policy/Critic strict
   state_dict loading requires no new keys or sizes. Alias tests demonstrate the
   unchanged Actor/Critic's lack of direct Box observability.
6. **Survival/TD:** tiny hand-computed cases for horizon, timeout, clip completion,
   hard robot/object failure, coincident flags and mixed lengths; reward uses C_k,
   bootstrap uses C_L; final boundary masks both TD branches. Padding perturbations
   cannot change reward, valid TD target, Actor gradient or Critic loss. All-valid
   no-boundary path equals original return/targets. Exclude padded Critic samples
   before minibatch formation; handle small batches without zero-size minibatches.
7. **Gradient paths without training:** component-isolated backward checks through
   deterministic joint dynamics for robot tracking, each Box term, robot→EEF and
   Box→target proximity paths; finite nonzero Actor/action gradients in nondegenerate
   coupled fixtures. Proximity OFF gradients exactly zero. Finite differences and
   deliberate detach controls distinguish graph reachability from merely nonzero
   regularization. WM/Critic flags/weights unchanged; no Actor/WM gradient from
   detached Critic-loss test. No optimizer or physical rollout is necessary.
8. **Eager/compiled:** compare state, EEF, target, distances, bool gate, each of 13
   components, total, obs, survival, action and Actor gradients at identical inputs,
   H=1 and a multi-step horizon including260/550/839 boundaries. Require gate/masks
   exact, FP64 `atol=1e-9, rtol=1e-7`, FP32 `atol=rtol=2e-5`, all gradients finite.
   These are new compiled integration tolerances, **not** replacements for P3's
   failed `2e-6` migration gate. Record device/backend, compiler mode, graph breaks
   and maximum per-field errors; unavailable backends are untested, not PASS.
9. **Framework/preservation:** original robot-only regression with matched RNG,
   complete diff inspection, hash checks for P4/reference/assets/dependencies,
   unchanged network state_dicts and original default behavior. Original known
   exact-match NaN is an explicitly recorded limitation, not a newly passing test.
   Reuse P1/P2 evidence without launching physics. No source validation or P5
   acceptance test above is claimed executed merely because the spec lists it.

## 11. Remaining risks and blocking conditions

| Risk | Consequence / required evidence | Gate |
|---|---|---|
| P4 target markers internal to capsules; M6.5 overlaps/support failures | Proximity success may require penetration or conflict with tracking. Preserve targets and failures; separate physical audit/approved disposition required. | Physical acceptance and P7 training blocked; isolated P5 software tests possible after approval. |
| Missing historical V3 provenance and refined-reference sidecars | Strict physical scene load still fails; do not fabricate files or change provenance to pass. | Physical setup remains blocked under current strict loader. |
| P3 unvalidated learned joint dynamics and FP32 multi-step migration failures | Controlled graph tests cannot show a useful physical action→Box derivative. | P7-A validation/explicit migration disposition required before P7-B. |
| Final sigmas/weights and HOI return calibration unselected | Formula is specified, experimental sensitivity/relative weighting remains unknown. | No enabled training defaults; measured component/Actor-gradient audit and reviewed config required. |
| Actor/Critic omit Box input, termination head omits Box failure features | State aliasing, biased tail values, missing Box-only failure prediction. | Report in P5; evaluate in P7-A/B. No dimension/head change authorized. |
| Reference derivatives approximate sampled motion | Endpoint accuracy lower; FP32 source quantization and between-frame motion unknown. | Formula validated for this reference; do not claim measured velocity truth. |
| Quaternion pi cut and P4 zero-distance cusp | Finite branch/subgradient does not imply global smoothness. | Test/report; no silent alternate loss. |
| Unmodified robot-only exact-match quaternion NaN | Original default backward can still fail at agreement. | HOI injection required for P5; separate approval for a global robot fix. |
| Compiled data-dependent guards and GPU arithmetic | Graph breaks/performance or gradient mismatch may appear. | Actual P5 compiled validation required; no installed-code/P4 patch workaround. |
| Clip-end interpretation changes missing HOI boundary behavior | A zero-tail finite task must not be confused with timeout continuation. | Review D4 and test masks; original robot-only behavior stays unchanged. |

The specification resolves the reward equations, frame alignment, insertion,
projection, mask/return semantics and a tested numerical correction. It does not
turn historical P3/P4 PARTIAL results or M6.5 failures into PASS, and does not
establish physical or training readiness. **Commit this document only, then stop
for user review before source implementation.**

## Appendix A. Reproducible temporary numerical probes

Final documentation checks before commit: all **568** pre-existing protected files
in the hash inventory remained unchanged, including the original/refined NPZs,
P4 files, installed quaternion dependency and user settings. All eight original
reward/imagination/training/network source files named in the Reward Integration
Review remain byte-identical to the local original checkout. Local document links
and whitespace checks pass. Only this specification is staged; `.vscode/` remains
untouched. The commit hash is reported to the user after committing, without a
self-referential hash in this document.

Temporary probe SHA256 values:

- `probe.py`: `d88e335f3160c4df3582728240fdfacbb21e3eebe314348e79119739d089e117`.
- `supplement.py`: `5d44df6bfa8b5d4ca7567cd504d2c3206578f0cc6fcca65164780a4359d745de`.

These are investigation scripts, not production P5 implementations. Save the
following blocks as `/tmp/smhoi_p5_spec/probe.py` and `supplement.py`, respectively,
then run the commands in §8 from this checkout. They write only temporary evidence;
all source/reference inputs are read-only. The supplementary script reads helper
definitions from the first script. The asserted thresholds are visible in code.

### probe.py

```python
import hashlib, importlib.metadata, json, math, pathlib, subprocess
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_box_minus
from supermimic.kinematics.rotations import quat_mul, quat_conjugate, quat_from_rotvec, quat_normalize, quat_apply, quat_apply_inverse, rotvec_from_quat
from supermimic.hoi.state import reference_object_twist

torch.set_num_threads(1)
torch.manual_seed(501)
ROOT = pathlib.Path('/home/han/pyx/SMHOI')
OUT = pathlib.Path('/tmp/smhoi_p5_spec')

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
paths = [ROOT / p for p in subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0') if p]
paths += [ROOT / p for p in ('outputs/hoi/asset_audit/part_b/reference/reference.npz', 'outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz', '.vscode/settings.json', '.venv/lib/python3.12/site-packages/mjlab/utils/lab_api/math.py')]
preserved = {str(p): digest(p) for p in paths if p.is_file()}
(OUT / 'preservation_before.json').write_text(json.dumps(preserved, indent=2))

def safe_minus(a, b):
    q = quat_mul(a, quat_conjugate(b))
    q = q * (1.0 - 2.0 * (q[..., :1] < 0.0))
    mag = torch.linalg.vector_norm(q[..., 1:], dim=-1)
    half = torch.atan2(mag, q[..., 0])
    angle = 2.0 * half
    large = angle.abs() > 1e-6
    safe_angle = torch.where(large, angle, torch.ones_like(angle))
    ratio = torch.where(large, torch.sin(half) / safe_angle, 0.5 - angle * angle / 48)
    return q[..., 1:] / ratio[..., None]

def sq(a, b, fn=safe_minus): return fn(a, b).square().sum(-1)

def quat_suite(dtype):
    axis = torch.tensor([1., -2., 3.], dtype=dtype); axis /= axis.norm()
    base = quat_normalize(torch.tensor([.6, .2, -.3, .7], dtype=dtype))
    ident = torch.tensor([1., 0., 0., 0.], dtype=dtype)
    rows = []
    for name, ref, theta, sign in [('identity', ident, 0., 1), ('negative_identity', ident, 0., -1), ('nonidentity_match', base, 0., 1), ('nonidentity_negative_match', base, 0., -1)] + [(f'angle_{a:g}', base, a, s) for a in (1e-9, 1e-7, .999e-6, 1.001e-6, 1e-5, .2, 1.3, math.pi-.001, math.pi+.001) for s in (1,-1)]:
        x = (theta * axis).clone().requires_grad_()
        q = sign * quat_mul(quat_from_rotvec(x), ref)
        value = sq(ref, q)
        grad = torch.autograd.grad(value, x, retain_graph=True)[0]
        old = sq(ref, q, quat_box_minus)
        old_grad = torch.autograd.grad(old, x)[0]
        target_angle = theta if theta < math.pi else theta-2*math.pi
        expected = 2*target_angle*axis
        rows.append(dict(case=name, sign=sign, value=float(value.detach()), value_error=abs(float(value.detach())-target_angle**2), forward_diff=float(abs(value-old).detach()), grad_error=float((grad-expected).abs().max()), finite=bool(torch.isfinite(grad).all()), old_finite=bool(torch.isfinite(old_grad).all()), old_grad_diff=float((grad-old_grad).abs().max()) if torch.isfinite(old_grad).all() else None))
    # Exact pi is a cut locus: demand finite values/one-sided computational derivative only.
    exact = torch.cat((torch.zeros(1, dtype=dtype), axis)).requires_grad_()
    pi_value = sq(ident, exact)
    pi_grad = torch.autograd.grad(pi_value, exact)[0]
    # Independent signs, raw quaternion gradient covariance, random pairs away from cut.
    a = quat_normalize(torch.randn(128,4,dtype=dtype)).requires_grad_()
    b = quat_normalize(torch.randn(128,4,dtype=dtype)).requires_grad_()
    v = sq(a,b); ga,gb = torch.autograd.grad(v.sum(), (a,b))
    sign_value = 0.; sign_grad = 0.
    for sa,sb in ((-1,1),(1,-1),(-1,-1)):
        aa=(sa*a.detach()).requires_grad_(); bb=(sb*b.detach()).requires_grad_()
        vv=sq(aa,bb); gaa,gbb=torch.autograd.grad(vv.sum(),(aa,bb))
        sign_value=max(sign_value,float((v-vv).detach().abs().max()))
        sign_grad=max(sign_grad,float((ga-sa*gaa).abs().max()),float((gb-sb*gbb).abs().max()))
    # Finite difference tangent gradients (do not cross pi).
    fd_errors=[]
    for theta in (0., 1e-5, .2, 1.3, math.pi-.01, math.pi+.01):
        x=(theta*axis).clone().requires_grad_()
        fun=lambda y: sq(base,quat_mul(quat_from_rotvec(y),base))
        g=torch.autograd.grad(fun(x),x)[0]
        eps=1e-3 if dtype==torch.float32 else 1e-6
        eye=torch.eye(3,dtype=dtype)*eps
        fd=torch.stack([(fun(x.detach()+e)-fun(x.detach()-e))/(2*eps) for e in eye])
        fd_errors.append(float((fd-g).abs().max()))
    gradchecks=[]
    if dtype==torch.float64:
        for theta in (0., 1e-7, .2, 1.3, math.pi-.001, math.pi+.001):
            x=(theta*axis).requires_grad_()
            gradchecks.append(torch.autograd.gradcheck(lambda y: sq(base,quat_mul(quat_from_rotvec(y),base)),(x,),eps=1e-6,atol=1e-7,rtol=1e-5))
    tol=5e-6 if dtype==torch.float32 else 1e-12
    assert all(r['finite'] and r['value_error']<tol and r['grad_error']<tol and r['forward_diff']==0 for r in rows)
    assert sign_value==0 and sign_grad==0 and torch.isfinite(pi_grad).all()
    assert max(fd_errors)<(2e-3 if dtype==torch.float32 else 5e-9)
    return dict(rows=rows, max_fd_error=max(fd_errors), sign_value_error=sign_value, sign_gradient_error=sign_grad, exact_pi_value=float(pi_value.detach()), exact_pi_grad_finite=bool(torch.isfinite(pi_grad).all()), gradchecks=gradchecks)

def derive(p,q,h):
    q=quat_normalize(q)
    f=torch.arange(len(p)); lo=(f-1).clamp_min(0); hi=(f+1).clamp_max(len(p)-1)
    span=((hi-lo).to(p.dtype)*h)[:,None]
    v=(p[hi]-p[lo])/span
    w=safe_minus(q[hi],q[lo])/span
    return v,w,quat_apply_inverse(q,w)

def twist_suite(dtype):
    h=.02; t=torch.arange(41,dtype=dtype)*h
    base=quat_normalize(torch.tensor([.6,.2,-.3,.7],dtype=dtype))
    vel=torch.tensor([.2,-.3,.5],dtype=dtype)
    omega=torch.tensor([.4,-.6,.2],dtype=dtype)
    p=torch.tensor([1.,2.,3.],dtype=dtype)+t[:,None]*vel
    q=quat_mul(quat_from_rotvec(t[:,None]*omega),base.expand(len(t),4))
    v,w,b=derive(p,q,h)
    signs=torch.where(torch.arange(len(t))%2==0,1.,-1.).to(dtype)[:,None]
    vf,wf,bf=derive(p,q*signs,h)
    dtol=2e-5 if dtype==torch.float32 else 5e-13
    assert float((v-vel).abs().max())<dtol and float((w-omega).abs().max())<dtol
    assert torch.equal(v,vf) and torch.equal(w,wf) and torch.equal(b,bf)
    return dict(linear_error=float((v-vel).abs().max()), world_angular_error=float((w-omega).abs().max()), body_roundtrip_error=float((quat_apply(q,b)-w).abs().max()), sign_error=float((w-wf).abs().max()))

result={'versions':{x:importlib.metadata.version(x) for x in ('torch','mjlab','numpy')},'quaternion':{},'synthetic_twist':{}}
for dtype in (torch.float32,torch.float64):
    result['quaternion'][str(dtype)]=quat_suite(dtype)
    result['synthetic_twist'][str(dtype)]=twist_suite(dtype)

with np.load(ROOT/'outputs/hoi/asset_audit/part_b/reference/reference.npz',allow_pickle=False) as z:
    p=torch.tensor(z['object_pos_w'].copy(),dtype=torch.float64)
    raw=torch.tensor(z['object_quat_w'].copy(),dtype=torch.float64)
    q=quat_normalize(raw); h=1/float(z['fps'][0])
    assert len(p)==840 and h==.02 and z['object_contact'].dtype==np.bool_
    v,w,b=derive(p,q,h)
    signs=torch.where(torch.arange(len(p))%2==0,1.,-1.)[:,None]
    vf,wf,bf=derive(p,q*signs,h)
    adjacent=safe_minus(q[1:],q[:-1]).norm(dim=-1)
    stencil=safe_minus(q[2:],q[:-2]).norm(dim=-1)
    reset=reference_object_twist(p,q,h)
    # Body quaternion difference at lo must be rotated using q_lo, not q_f.
    f=torch.arange(len(p));lo=(f-1).clamp_min(0);hi=(f+1).clamp_max(len(p)-1)
    body_at_lo=safe_minus(quat_mul(quat_conjugate(q[lo]),q[hi]),torch.tensor([1.,0.,0.,0.],dtype=q.dtype).expand(len(q),4))
    from_body=quat_apply(q[lo],body_at_lo)/(((hi-lo).to(p.dtype)*h)[:,None])
    def quant(x): return [float(y) for y in torch.quantile(x,torch.tensor([0.,.5,.95,.99,1.],dtype=x.dtype))]
    result['reference']=dict(norm_error=float((raw.norm(dim=-1)-1).abs().max()),adjacent_angles_rad=quant(adjacent),stencil_angles_rad=quant(stencil),v_norm=quant(v.norm(dim=-1)),w_norm=quant(w.norm(dim=-1)),adjacent_sign_flips=int(((q[1:]*q[:-1]).sum(-1)<0).sum()),sign_error=float((w-wf).abs().max()),reset_error=float((torch.cat((v,b),-1)-reset).abs().max()),body_relative_world_error=float((from_body-w).abs().max()),endpoints={'v':v[[0,-1]].tolist(),'omega_world':w[[0,-1]].tolist()},largest_position_step_frame=int((p[1:]-p[:-1]).norm(dim=-1).argmax()+1),largest_rotation_step_frame=int(adjacent.argmax()+1),all_finite=bool(torch.isfinite(torch.cat((v,w),-1)).all()),contact_on=int(z['object_contact'].sum()))
(OUT/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
```

### supplement.py

```python
import json, math, pathlib
import torch
# Read helper definitions only, avoiding execution of probe's main block.
ns={}
s=pathlib.Path('/tmp/smhoi_p5_spec/probe.py').read_text()
exec(s[:s.index("result={'versions'")],ns)
derive=ns['derive']; safe_minus=ns['safe_minus']
quat_mul=ns['quat_mul']; exp=ns['quat_from_rotvec']; apply=ns['quat_apply']; inverse=ns['quat_apply_inverse']; normalize=ns['quat_normalize']
rows=[]
for h in (.02,.01,.005):
 t=torch.arange(round(1/h)+1,dtype=torch.float64)*h
 p=torch.stack((t**3,t**2,.4*t),-1)
 v_true=torch.stack((3*t**2,2*t,torch.full_like(t,.4)),-1)
 ax=torch.tensor([.8,0.,0.],dtype=t.dtype); ay=torch.tensor([0.,-.7,0.],dtype=t.dtype)
 qx=exp(t[:,None]*ax); qy=exp(t[:,None]*ay)
 base=normalize(torch.tensor([.6,.2,-.3,.7],dtype=t.dtype))
 q=quat_mul(quat_mul(qx,qy),base.expand(len(t),4))
 w_true=ax+apply(qx,ay.expand(len(t),3))
 v,w,b=derive(p,q,h)
 rows.append({'h':h,'v_interior_max':float((v[1:-1]-v_true[1:-1]).abs().max()),'v_endpoint_max':float((v[[0,-1]]-v_true[[0,-1]]).abs().max()),'omega_interior_max':float((w[1:-1]-w_true[1:-1]).abs().max()),'omega_endpoint_max':float((w[[0,-1]]-w_true[[0,-1]]).abs().max())})
for name in ('v_interior_max','omega_interior_max'):
 assert all(3.9 < rows[i][name]/rows[i+1][name] < 4.1 for i in (0,1))
for name in ('v_endpoint_max','omega_endpoint_max'):
 assert all(1.9 < rows[i][name]/rows[i+1][name] < 2.1 for i in (0,1))
import numpy as np
with np.load('/home/han/pyx/SMHOI/outputs/hoi/asset_audit/part_b/reference/reference.npz') as z:
 p=torch.tensor(z['object_pos_w'],dtype=torch.float64);q=normalize(torch.tensor(z['object_quat_w'],dtype=torch.float64))
 v,w,b=derive(p,q,.02)
 slope=(p[1:]-p[:-1])/.02
 qn=normalize(q)  # Match derive's normalized computation copy.
 spin=safe_minus(qn[1:],qn[:-1])/.02
 jumps={}
 for name,seq in (('v_forward_change',slope[1:]-slope[:-1]),('omega_forward_change',spin[1:]-spin[:-1])):
  val,idx=seq.norm(dim=-1).topk(5)
  jumps[name]=[(int(i)+1,float(v)) for i,v in zip(idx,val)]
 dtype_compare={}
 for name,x,y in zip(('linear','world_angular','body_angular'),derive(p.float(),q.float(),.02),(v,w,b)):
  dtype_compare[name]=float((x.double()-y).abs().max())
 t2=derive(p[-2:],q[-2:],.02)
 assert all(torch.isfinite(x).all() for x in t2)
 # Full versus each source interval, proves boundary use of one-sided endpoints.
 assert torch.equal(v[0],slope[0]) and torch.equal(v[-1],slope[-1])
 assert torch.equal(w[0],spin[0]) and torch.equal(w[-1],spin[-1])
 data={'convergence':rows,'pose_slope_jumps':jumps,'fp32_vs_fp64':dtype_compare,'max_position_step_m':float((p[1:]-p[:-1]).norm(dim=-1).max()),'endpoint_one_sided_exact':True}
pathlib.Path('/tmp/smhoi_p5_spec/supplement.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
```
