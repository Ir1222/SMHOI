# P5-A — reward and reference infrastructure

Software result: **PASS**. Final training coefficients are **not selected**. This
report implements the approved [specification](IMPLEMENTATION_SPEC.md), based on
commit `3b86b1526735fc3d46b408200ed700650aad6dfc`; it does not replace that contract.
See [P5-C](P5_C_REPORT.md) for complete validation and [REPORT](REPORT.md) for scope
and unresolved gates.

## Implementation

- `src/supermimic/hoi/tracking_reward.py`: required coefficient configuration,
  named physical84→robot71 projection, safe squared quaternion log, four Box
  terms, unchanged robot/P4 composition and maximum positive step reward.
- `src/supermimic/hoi/reward_reference.py`: immutable derivative cache and explicit
  replay/local/library/global identity registry. Construction validates the actual
  library state, body and goal tensors against the bound full NPZ sources.
- `envs/mdp/rewards.py` and `world_model/heads.py`: optional orientation-error
  injection. Their default still selects the original robot implementation.
- `hoi/rewards.py`, reference NPZs, physical assets and
  installed quaternion code were not changed.

## Equations and units

All terms below are evaluated at the reached physical state against the same next
reference frame. The Box position is its root origin, in environment-local world
axes. The stored Box linear velocity is world-frame; angular velocity is body-frame.
Let `D(q1,q2)` be the shortest rotation-vector difference after sign selection.

\[
\begin{aligned}
e_p&=\|p-\bar p\|^2,&
e_q&=\|D(\bar q,q)\|^2,\\
e_v&=\|v^w-\bar v^w\|^2,&
e_\omega&=\|R(q)\omega^b-\bar\omega^w\|^2,\\
r_j&=\Delta t\,\beta_j\exp(-e_j/\sigma_j^2),
&j&\in\{p,q,v,\omega\}.
\end{aligned}
\]

The sigmas have units m, rad, m/s and rad/s. Box tracking is always active,
including contact-OFF frames. Quaternion normalization acts on computation copies;
input norm errors above `1e-5`, nonfinite values and incompatible shapes/dtypes are
rejected. Sign flips of either quaternion do not change the squared error.

The complete scalar is

\[
r=\sum_{i=1}^{6}r_i^{robot,+}+\sum_jr_j^{Box}
 +\Delta t\,\beta_c r_{P4}+r_{action-rate}+r_{joint-limit}.
\]

The original robot terms retain root/body reanchoring, means, sigmas and weights.
The penalties retain `dt*w_action_rate*sum((a-a_prev)^2)` and
`dt*w_joint_limit*sum(relu(lo-q)+relu(q-hi))`; the latter is **not squared**.
`heads.reward(...)["total"]` already contains both penalties. The P4 combiner
receives that total once and a zero additional regularization input. Thirteen
additive components sum to the scalar; redundant diagnostic aggregates are never
summed again. There is no reward clipping, standardization or extra dt factor.

The unchanged P4 reward is

\[
r_{P4}=c^{ref}\frac12\sum_{h\in\{L,R\}}
  \exp(-\|p_{EEF,h}(s^{robot})-[p_B+R(q_B)t_h]\|/\sigma_{pos}).
\]

It uses the existing left/right targets, wrist-local EEF offsets and **current
imagined Box** pose. OFF reward and gradients are exactly zero. Target locations
remain those audited in P4; geometric closeness does not establish physical contact.

## Formula-preserving quaternion correction

For the sign-selected relative quaternion `(w,v)`, use `mag=||v||`,
`half=atan2(mag,w)`, `angle=2*half`. The original ratio is selected as
`sin(half)/angle` when `abs(angle)>1e-6`, otherwise `0.5-angle^2/48`.
The HOI helper substitutes denominator `1` **only in the unused division branch**
at small angle. The result remains `v/ratio`; it neither adds an epsilon to the
loss nor changes its maximum. This corrects `torch.where` backward contamination
from an unused `0/0` while preserving the selected forward formula.

Both robot orientation terms receive this helper only through the HOI injection.
The original robot-only exact-match NaN remains a recorded legacy limitation.
The quaternion pi cut and P4 zero-distance cusp are not claimed smooth.

## Immutable reference cache

Source: `outputs/hoi/asset_audit/part_b/reference/reference.npz`.
SHA256: `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`.
All **840 frames**, at **50 Hz**, were checked. Contact is the original bool label:
290 ON frames `[260,550)` and 550 OFF frames.

For each frame `f`, let `lo=max(f-1,0)`, `hi=min(f+1,T-1)` and
`span=(hi-lo)*dt`. Construction computes in FP64:

\[
\bar v_f^w=(p_{hi}-p_{lo})/span,\quad
\bar\omega_f^w=D(q_{hi},q_{lo})/span,\quad
\bar\omega_f^b=R(q_f)^T\bar\omega_f^w.
\]

Interior differences are centered; endpoints are one-sided. Adjacent and two-frame
increments at the pi cut are rejected. No stencil crosses clips. Cache metadata
records source/manifest hashes, FPS, dt, derivative version `central-world-log-v1`,
units, conventions and storage dtype. Source arrays are never written. Public
snapshots and scalar/batched gathers own copies, so caller writes cannot mutate
cached constants. FP32 storage is a cast of the FP64 derivation.

Measured maximum speeds are `1.3630637385822617 m/s` and
`2.3509212780269064 rad/s`. Alternating quaternion sign flips give exact invariant
derivatives. Endpoint linear velocities are zero; angular endpoints are roundoff
at approximately `1e-14 rad/s`. These are sampled-pose derivatives, not measured
physical velocity truth. Synthetic smooth-motion tests verify second-order
interior and first-order endpoint convergence.

## Configuration and scale

`HOIRewardCfg` has no production defaults for Box sigmas/weights, `sigma_pos`,
`beta_c` or dt. Missing values fail construction. Values must be finite and
positive; `beta_c=0` is accepted only with named `proximity-off` ablation.
No training CLI default was enabled or changed.

The explicit **test-only** fixture is sigmas `(0.2,0.4,1.0,2.0)`, Box weights
`(0.7,0.8,0.9,1.1)`, `sigma_pos=0.0423`, `beta_c=0.6`, `dt=0.02`.
It exists only in `tests/hoi/p5_fixtures.py`. Its positive bound is
`B_max=B_robot+dt*(sum(beta_box)+beta_c)=0.182`, versus the original default `0.1`.
Production exposes this bound without changing `Trainer.run` or sampler defaults.
P7 must explicitly configure same-unit return normalization, Critic calibration
and sampler `g_max`; robot-only optimizer/sampler state is not an HOI resume.

## Evidence

`outputs/hoi/contact_supervision/p5_validation/eager_final_v5/measurements/` contains:

- `reference_840.json`: all-frame provenance, signs, endpoints and cache isolation.
- `multi_clip.json`: distinct source positions/labels, explicit reordered mapping,
  and reached local/global frame checks.
- `quaternion_float32.json`, `quaternion_float64.json`: 22-case angle/sign matrix,
  128 random pairs, finite differences, exact-pi branch behavior and FP64 gradchecks.
- `scale_audit_840.json`: physical squared errors, all 13 contributions and state
  gradient distributions for explicit small perturbations of all 840 reference
  states. This is a scale audit, not an imagined or physical task rollout.

Algebra tests verify maxima, `1/e` at one sigma, monotonicity, world/body conversion,
both quaternion signs, penalty counting and action/previous-action gradients.
P4 regression checks its existing formula and target definitions without edits.
