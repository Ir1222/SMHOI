# Reward integration review — 2026-11-12

**Status: proposal for review; no P5 implementation or training authorization.**
Reviewed `main` at `fd8d28a190cd57f317b088aa581c57b63dec435e`, the current
[P4 report](REPORT.md), [P4 implementation](../../../../src/supermimic/hoi/rewards.py),
and Master Plan P4.5/P5.2–P5.6/P7. This document is the only repository addition
from this review. Existing P4, training code, dimensions, reference data, assets,
user changes and historical evidence are preserved.

The original analytic reward can supply robot tracking and its two existing
regularizers. It does **not** supply Box tracking. The joint World Model's object
prediction loss is a separate supervised dynamics loss, not an Actor reward.
The proposed missing term is four continuously active, differentiable Box tracking
scores, using the original squared-error exponential convention. Their weights
and tolerances remain unselected.

Three integration issues require explicit treatment before implementation:

- Original imagination evaluates the reached state against the **current** target
  frame. P5.2 requires the **next** reference frame for the next-state reward.
- The joint model's canonical86 cannot be passed to the unchanged Critic's
  canonical71 input. An explicit robot projection is necessary.
- A bounded CPU probe found NaN orientation gradients at exact agreement in the
  installed original quaternion helper. This is recorded below, not repaired here.

## 1. Original reward calculation and training path

The following eight source files are byte-identical to their counterparts in the
local original checkout `/home/han/pyx/SuperMimic-Reproduced`. Line numbers refer
to the reviewed SMHOI commit.

| Source | Exact role |
|---|---|
| [world_model/heads.py](../../../../src/supermimic/world_model/heads.py), 101–131 | `AnalyticHeads.robot_bodies` runs differentiable FK; `reward` passes predicted and reference link quantities to `tracking_reward_terms`. |
| [envs/mdp/rewards.py](../../../../src/supermimic/envs/mdp/rewards.py), 24–128 | Defines errors, exponentials, weights, timestep scaling, regularizers and `terms["total"]`. |
| [imagination/compiled.py](../../../../src/supermimic/imagination/compiled.py), 37–48 | Predicts `s_next`, calls the reward head, selects `terms["total"]` as the imagined scalar reward. Same calculation in eager and compiled modes. |
| [imagination/rollout.py](../../../../src/supermimic/imagination/rollout.py), 68–129 | Calls Actor, carries predicted states/observation history forward, retains reward graphs and accumulates survival. |
| [imagination/objective.py](../../../../src/supermimic/imagination/objective.py), 10–35 | Forms the Actor return and the detached TD(lambda) Critic targets. |
| [imagination/policy_trainer.py](../../../../src/supermimic/imagination/policy_trainer.py), 60–123 | Freezes WM/Critic parameters for Actor BPTT, calls backward and Actor Adam; subsequently fits Critic. |
| [training/loop.py](../../../../src/supermimic/training/loop.py), 144–198 | Collects real transitions, then schedules WM fitting and policy updates after warmup. |
| [policy/nets.py](../../../../src/supermimic/policy/nets.py), 72–96 | Actor maps robot observation history and motion goal to action29; Critic consumes canonical71, goal, xi33 and root-error4. |

### 1.1 The original robot equations

For each tracking component, the source computes

\[
r_j^{robot}=\Delta t\,w_j\exp(-E_j/\sigma_j^2).
\]

These are existing defaults from `TrackingRewardCfg`, not proposed Box settings:

| Component | Error before division by sigma squared | Existing sigma | Existing weight |
|---|---|---:|---:|
| Global anchor position | Squared Euclidean distance | 0.3 m | 0.5 |
| Global anchor orientation | Squared shortest rotation angle | 0.4 rad | 0.5 |
| Relative body position | Mean over tracked bodies of squared Euclidean distance | 0.3 m | 1.0 |
| Relative body orientation | Mean over tracked bodies of squared shortest rotation angle | 0.4 rad | 1.0 |
| Global body linear velocity | Mean over tracked bodies of squared world-velocity difference | 1.0 m/s | 1.0 |
| Global body angular velocity | Mean over tracked bodies of squared world-angular-velocity difference | 3.14 rad/s | 1.0 |

The two keys called `motion_global_root_*` use the configured **torso_link
anchor**, not the pelvis free-joint root (`tracking_env_cfg.py:49–66`). Relative
body pose targets are reanchored to the predicted torso's XY position and yaw;
reference height, roll and pitch remain. Velocity targets are compared in world
axes without this pose reanchoring. FK returns link-origin linear velocities and
world angular velocities (`kinematics/g1_fk.py:107–164`). Each three-vector error
sums its coordinate squares; only the body terms then average across bodies.

Let `r_robot_positive` be the sum of those six terms. The same function also
computes

\[
r_{reg}=\Delta t\left[
w_{ar}\|a_k-a_{k-1}\|^2+
w_{jl}\sum_j\left(\max(q_j^{lo}-q_j,0)+\max(q_j-q_j^{hi},0)\right)
\right].
\]

Here `q_j` is the reached joint position, and the existing defaults are
`w_ar=-0.1`, `w_jl=-2.0`, `dt=0.02`. The joint-limit violation is **not squared**.
Thus `terms["total"] = r_robot_positive + r_reg` already. Adding a second copy of
`r_reg` to this total would change the original objective.

The six positive terms have an existing maximum of `0.02 * 5 = 0.1` per step;
regularization can lower the total. The environment's 0.005 s physics timestep
and decimation4 give the same 0.02 s control interval.

### 1.2 Where the actual training reward is produced

There are two calculations with distinct uses:

1. **Real collection:** `Trainer.act` is under `torch.no_grad()` (lines 144–146).
   `Trainer.run:164–165` calls it and then `env.step(action)`. The environment
   reward manager evaluates the tracking functions configured in
   [tracking_env_cfg.py](../../../../src/supermimic/envs/tracking_env_cfg.py):203–212,
   including the two penalties; its default `scale_rewards_by_dt=True` applies
   the timestep. This is the mjlab simulator reward path, not a call to
   `AnalyticHeads.reward`. The returned reward feeds episode-return logging,
   return-based reference sampling (`envs/mdp/commands.py:194–243`) and the replay
   reward field (`data/recorder.py:68–82`). It is not a differentiable simulator
   path back to Actor parameters.
2. **Actor imagination:** `ImagStep._step:39` predicts the next state;
   lines 40–41 run FK and `AnalyticHeads.reward`; line 42 takes `terms["total"]`.
   `imagine:113` appends this tensor without detaching it. This recomputed reward,
   not the replay reward field, enters the Actor objective and its Critic targets.
   Only the logging copy `term_vec` is detached (`rollout.py:104`).

After warmup, `Trainer.run:181–198` schedules
`WorldModelTrainer.update(buffer)` and then the configured number of
`PolicyTrainer.update(buffer)` calls. WM fitting uses replay transition/state and
termination targets (`world_model/wm_trainer.py:45–61`), not an object reward.
Changing only a simulator reward or replay field would therefore not install the
new objective in Actor BPTT.

### 1.3 Actual frame convention versus the requested HOI convention

For a replay start at frame `f`, the original rollout gathers
`frames_all[k] = min(f+k, last_frame)` (`rollout.py:78–88`). At iteration `k`:

- Actor sees the goal beginning at `f+k`.
- `step_fn(..., ref_at(k))` computes the reward for `s[k+1]` against `f+k`.
- Only after that call does the loop advance to `ref_at(k+1)`.
- The next state's stored Critic root-error features use this advanced reference.

This agrees with `AnalyticHeads.reward:114`, which explicitly describes reaching
`s_{t+1}` under target `g_t`. It also agrees with the simulator computing reward
before advancing the motion command. It must not be described as an existing
next-frame reward implementation or silently changed in the robot-only path.

**Proposal:** the HOI path should deliberately evaluate robot, Box and proximity
rewards for `s[k+1]` using the common evaluated frame `f+k+1`, following Master
Plan P5.2. Keep the Actor's original goal interface/progression; distinguish the
goal used to choose an action from the frame used to score its result. The
reference-dependent termination features also need an explicit alignment check
against the WM's termination-training targets. Changing only the reward reference
while accidentally shifting a trained termination head's targets is insufficient.

Use explicit clip identity/global-to-local frame mapping for the object sidecar
and original bool label. Do not borrow another clip's samples or repeat a final
frame as a valid reward. The current rollout clamps final frames; P5.5 will need
an explicit valid-step/clip-end boundary with the original survival/bootstrap
semantics preserved. No truncation rule or bootstrap policy is implemented here.
P4's `reference_contact_at` correctly rejects out-of-range indices, but cannot
detect an index that a caller has already clamped to a valid last frame.

## 2. Proposed Box tracking reward

### 2.1 State, frames and normalization

The current contract is defined by
[hoi/state_spec.py](../../../../src/supermimic/hoi/state_spec.py):21–58,74–117,141–174
and [hoi/state.py](../../../../src/supermimic/hoi/state.py):92–137.

| Physical84 slice | Quantity | Frame/unit |
|---|---|---|
| `0:71` | Original robot state | Original robot contract |
| `71:74` | Box root-origin position `p_o` | Environment-local world axes, m |
| `74:78` | Box root orientation `q_o` | Unit wxyz, body-to-world rotation |
| `78:81` | Box root-origin linear velocity `v_o` | World axes, m/s |
| `81:84` | Box angular velocity `omega_o_body` | Box body axes, rad/s |

`JointStateIO.read` subtracts the environment translation from **both** robot and
Box positions. The reference and prediction must share that origin convention;
do not subtract an environment origin twice. World velocities are unaffected by
a constant environment translation. Apply any explicit scene transform to both
reference and prediction consistently.

Box linear velocity is at the root origin, not the center of mass. The existing
diagnostic `object_com_velocity_world` adds `omega_world × R(q) * body_ipos`
(`state.py:172–179`). That COM conversion must not be mixed with a reference
derivative of root-origin position.

Canonical86 is a **model input representation**: robot71 plus Box position
relative to the robot root in robot-heading axes, heading-relative rotation6,
absolute Box linear velocity in robot-heading axes, and Box-body angular velocity.
It is not a moving-frame velocity derivative; robot velocity is not subtracted.
The reward should use physical84 in physical units, not an MSE over canonical86.
Otherwise motion of the robot's heading/origin can change the object error being
optimized. Existing `ObjectSpecCfg.in_scale`, `out_scale`, `err_scale` and
`LossWeights.object` normalize or weight WM inputs/residuals/supervised errors;
none defines an approved object reward tolerance or weight.

### 2.2 Reference velocities are missing and must be derived explicitly

The accepted original reference
`outputs/hoi/asset_audit/part_b/reference/reference.npz` contains Box position and
quaternion, but **no Box linear/angular velocity arrays**. It has 840 frames at
50 Hz, so `h=0.02 s`. SHA256:
`bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`.

`reference_object_twist` (`hoi/state.py:23–37`) already derives a **reset-only**
twist cache from these poses. Its numerical convention is a suitable proposal
for a separate immutable reward-reference cache, subject to validation. Its
present reset-only use does not establish accepted velocity supervision.

For reference frame `f`, set `l=max(f-1,0)`, `u=min(f+1,T-1)` within that clip,
and `tau=(u-l)h`. For unit reference quaternions:

\[
v^{ref}_{o,f}=\frac{p^{ref}_{o,u}-p^{ref}_{o,l}}{\tau},\qquad
\omega^{ref,world}_{o,f}=
\frac{\operatorname{LogRot}(q^{ref}_{o,u}\otimes(q^{ref}_{o,l})^{-1})}{\tau}.
\]

`LogRot` returns a shortest rotation vector in radians. The multiplication order
above gives a world-frame increment. For the existing reset representation,
`omega_ref_body = R(q_ref_f)^T * omega_ref_world`. Use central differences
internally and one-sided differences at the endpoints. These are derivative
stencils within valid reference data, not extra reward frames or contact-label
lookahead. Record endpoint accuracy and any pose discontinuities; do not silently
smooth, clamp, or zero velocities. Quaternion sign changes must not produce
spurious turns; sampling must resolve inter-sample rotations below the shortest-
rotation ambiguity. Future tests must include constant translation/rotation,
sign-flipped equivalent quaternions and clip boundaries.

Normalize valid quaternion copies for computation, reject nonfinite/degenerate
inputs, and retain the NPZ byte-for-byte. Store source hash, FPS, frame mapping,
derivation version, dtype and units with any future cache. Reference derivatives
are constants for Actor autograd and never substitutes for measured simulator
velocities. No such cache or new reference array was created in this review.

### 2.3 Four errors and the proposed equation

All quantities below refer to the reached predicted state and its **same evaluated
reference frame**. Let `R(q)` map body vectors to world axes, and use unit wxyz
quaternions. Define

\[
\begin{aligned}
E_p &= \|p_o-p_o^{ref}\|_2^2,\\
E_R &= \|\operatorname{LogRot}(q_o^{ref}\otimes q_o^{-1})\|_2^2,\\
E_v &= \|v_o-v_o^{ref}\|_2^2,\\
E_\omega &= \|R(q_o)\omega_o^{body}-\omega_o^{ref,world}\|_2^2.
\end{aligned}
\]

If reference angular velocity is stored in its own body frame, the final target
is `R(q_ref) * omega_ref_body`. Subtracting predicted and reference body-frame
vectors directly compares components expressed in different orientations.
`JointStateSpec.object_error_vector` currently uses that componentwise body
difference for its **WM loss**; this review does not change that P3 contract or
reuse it as an Actor angular-velocity reward.

The proposed object contribution is

\[
\boxed{r_{object}=\Delta t\sum_{j\in\{p,R,v,\omega\}}
\beta_{o,j}\exp(-E_j/\sigma_{o,j}^2).}
\]

All four `beta_o,j` are explicit positive, **unselected** coefficients, and all
four `sigma_o,j` are explicit positive, **unselected** tolerances in m, rad, m/s,
and rad/s respectively. No numerical object settings are prescribed here.
Tracking remains active for both contact ON and OFF, including post-release
reference motion. The contact gate does not multiply this term.

This adopts the original reward's squared Euclidean/geodesic error and
`exp(-E/sigma^2)` shape, sum over vector coordinates, and control-timestep scaling.
There is one rigid Box, so no average over robot bodies. It uses absolute Box
root pose and world velocities, analogous to the original global anchor pose
and world body-velocity terms. Reanchoring the Box to a drifting robot would
weaken tracking relative to the fixed support surfaces. Quaternion MSE,
rotation6 MSE, raw norm exponentials, a product of four rewards, and a new
robot-relative Box objective are not implicit parts of this proposal.

The orientation equation is invariant to `q` versus `-q` and has the same
squared-angle meaning as `_sq_quat_err`. The shortest-angle loss has a branch
ambiguity at 180 degrees; it is not globally smooth there. Computational handling
of identity rotations needs the numerical qualification in Section 5.

### 2.4 Scale selection remains an explicit decision

The robot tolerances above are provenance, not automatic Box defaults. Neither
the WM error scales nor the P4 proximity-distance quantiles establish suitable
Box pose/velocity tolerances. Before activating this proposed term, report
component error distributions, reference derivative quality, controlled physical-
unit perturbations, reward contributions and state/action gradient norms. Choose
and document tolerances first, then weights relative to the existing robot
tracking and regularization scales. Relevant predicted-state distributions and
Actor gradients require a separately authorized integration validation.

At perfect tracking each exponential is one, so the object maximum is
`dt * sum(beta_o,j)`. This bounds its value but does not determine its sensitivity
or justify a coefficient. P4 measured mean positive robot reward
`0.0999999033` on reference ON states and raw ON proximity means approximately
`0.245–0.525` across its three candidates. Those observations support a scaling
audit only; no final sigma, object weight or contact weight was selected.

## 3. Complete proposed reward and exact integration point

Let the unchanged P4 raw proximity be

\[
P_{f}(s)=c_f^{ref}\frac{1}{N_c}\sum_i
\exp\left(-\frac{\|p_{eef,i}(s)-[p_o+R(q_o)p_{target,i}^{box}]\|}{\sigma_{pos}}\right).
\]

It uses the **current imagined** Box pose, differentiable robot FK and the
original global binary reference label. It has no implicit timestep factor.
Retain P4's mean over the audited target set; it is not a physical bilateral AND
measurement. Existing target-validity limitations in the P4 report remain.

The complete proposed per-step reward is

\[
\boxed{r_{total}=
r_{robot\_positive}+r_{object}+\Delta t\,\beta_c P_f+r_{reg}.}
\]

This preserves the original robot weights and regularization exactly. In Master
Plan P4.5 notation, the robot outer multiplier is its existing identity; the
object outer multiplier is absorbed into the effective `beta_o,j`; and the
coefficient multiplying **raw** proximity is `lambda_c = dt * beta_c`.
`beta_c` remains unselected (zero only for an explicitly configured proximity-OFF
ablation). This avoids inventing redundant weight settings or applying `dt` twice.

There are two equivalent uses of the existing P4 combiner:

| Argument | Separate regularization convention | Existing robot-total convention |
|---|---|---|
| `robot` | Sum of the six positive robot terms | `robot_terms["total"]` |
| `regularization` | The two existing weighted, dt-scaled penalties | Zero tensor |
| `object` | The proposed already weighted, dt-scaled object contribution | Same |
| `proximity` | P4 raw `P_f` | Same |
| `robot_weight`, `object_weight` | Identity multipliers, since weights are already included | Same |
| `contact_weight` | `dt * beta_c` | Same |

These identity multipliers preserve the supplied terms; they are not newly
selected research coefficients. Per-component logging should retain the separate
regularizers regardless of which equivalent composition is used.

**Exact insertion point:** the HOI adaptation of
`imagination/compiled.py:39–43`, after `wm.step` has returned predicted physical84
and before the scalar reward is returned to `imagine`. Preserve the original
robot-only path. If approved, the future HOI integration would:

1. Select the evaluated-frame robot reference, Box pose/twist reference and global
   bool contact label outside the compiled step; pass tensors explicitly.
2. Predict physical84 using canonical86 + support18 + action29 + xi33, integrate
   residual82, and take `robot_next = s_next[..., :71]`.
3. Call the existing robot FK and `AnalyticHeads.reward` on **robot_next** with
   the aligned robot reference. `robot_bodies` uses `s[...,36:]` as qvel, so
   passing physical84 directly would incorrectly supply 48 rather than 35 values.
4. Calculate the four proposed object terms from `s_next[...,71:84]` and the
   aligned reference; calculate P4 proximity from full `s_next` and `c_ref`.
5. Return `r_total` in the existing reward slot. Extend HOI diagnostic term names
   for the four object terms/proximity/total; the current tuple lists only eight
   robot/regularizer terms. Logging must not detach the actual reward tensor.
6. Continue the original analytic robot observation/history, survival accumulation,
   Actor return, backward and optimizer flow. No reward term is an Actor input.

`PolicyTrainer.__init__:47` currently constructs `ImagStep`, and
`PolicyTrainer.update:68` calls `imagine`; these are the future adapter-selection
points. `_critic_inputs:55–58` needs a robot projection for the HOI case, such as
the original robot spec's `canonical(ro.states[..., :71])`, retaining the time
and batch layout. Blindly invoking `JointStateSpec.canonical` there yields86 and
is incompatible with the existing Critic. This describes required interfaces,
not a new trainer or duplicated optimizer loop. No adapter was implemented.

Actor remains observation93 × history10 = 930, motion goal680 under the default
configuration, output29. Critic remains canonical71 + goal680 + xi33 + root-error4
= 788 inputs. The existing optional anchor-goal configuration adds9 to the goal
and hence Critic input; this review neither enables nor changes it. The joint WM
retains input166/output82 and physical84/canonical86. Object references, contact
labels and support context are not appended to Actor/Critic observations.

Consequently, the unchanged Critic cannot distinguish two joint states with
identical robot/goal/xi/root-error inputs and different Box states. It approximates
the total return conditional on its existing information, not a fully observed
joint-state value. Its terminal value has no direct Box-input gradient; direct
object rewards and joint-dynamics coupling still provide Actor gradients. This
limitation should be reported in later evaluation, not hidden by widening inputs.

When real HOI reward logging is eventually integrated, use the same approved
equations and frame policy for comparable measurements. `Trainer.max_step_reward`
currently comes from the robot-only bound (`loop.py:97,172`), and return-based
sampling has its own scale. A total HOI return must not silently be labeled a
robot-normalized return or fed to unchanged sampling scales. These are future
integration checks, not changes in this review.

## 4. Reward values, gradient propagation and parameter updates

The intended data flow, with the proposed reward occupying the existing scalar
slot, is:

```text
Replay start + robot observation history + original motion goal
    -> Actor(action29)
    -> frozen joint WM + physical integration -> predicted physical84
         -> robot71 -> FK -> six original robot tracking terms
         -> Box13 + aligned Box reference -> four proposed object terms
         -> robot FK + current Box geometry + c_ref -> P4 proximity
         -> action/previous action/joints -> two original regularizers
    -> sum r_total -> rollout.rewards (autograd graph retained)
    -> original survival-discounted return + terminal frozen Critic value
    -> Actor loss.backward() -> gradient reduction/clipping -> Actor Adam.step()

Same imagined rollout, detached
    -> TD(lambda) targets -> Critic MSE.backward() -> Critic Adam.step()

Real MuJoCo contact/force -> independent physical diagnostics only
```

Let `psi` be Actor parameters, `theta` WM parameters and `phi` Critic parameters.
During Actor improvement,

\[
a_k=\pi_\psi(o_{hist,k},g_k),\qquad
s_{k+1}=F_\theta(s_k,a_k,\xi),\qquad
d_{k+1}=\operatorname{sigmoid}(\ell_\theta(s_{k+1},ref)),\qquad
c_0=1,\quad c_{k+1}=c_k(1-d_{k+1}).
\]

The joint model computes support18 from the current state inside this transition.
References/geometry constants are fixed; state, action and geometric computation
remain on the Torch graph. Original `policy_objective` computes

\[
J(\psi)=\frac1N\sum_n\left[
\sum_{k=0}^{H-1}\gamma^k c_{k,n}r_{total,k+1,n}
+\gamma^H c_{H,n}V_\phi(x_{H,n})\right],\qquad
L_{Actor}=-J/H.
\]

The immediate reward is weighted by **pre-step** survival `c_k`, not `c_{k+1}`.
The terminal term is omitted only in the existing `use_critic=False` option.

`PolicyTrainer.update:66–67` sets WM and Critic parameters' `requires_grad=False`.
It does **not** wrap imagination or terminal Critic evaluation in `no_grad` and
does not detach their inputs. For example, the state-mediated reward derivative
contains

\[
\frac{\partial r_{k+1}}{\partial s_{k+1}}
\left(\frac{\partial F_\theta}{\partial s_k}\frac{d s_k}{d\psi}
+\frac{\partial F_\theta}{\partial a_k}\frac{d a_k}{d\psi}\right).
\]

Action-rate regularization also has direct derivatives through current and
previous imagined actions. Further paths run through predicted observation
history into later actions, differentiable survival probabilities, and terminal
Critic input derivatives. `a_prev = a` and `s = s_next` are not detached between
steps. Thus frozen WM weights still transmit derivatives from robot/object
reward through their action/state Jacobians to Actor parameters. Nonzero geometry
gradients alone do not prove a useful Actor gradient: the learned dynamics must
also transmit the relevant action influence.

These operations are distinct:

| Operation | Source | Effect |
|---|---|---|
| Calculate reward and return | `ImagStep`, `imagine`, `policy_objective` | Builds tensor values and the differentiable computation graph; changes no model parameters. |
| Propagate Actor gradient | `policy_trainer.py:76–80` | Clears Actor grads, calls `loss.backward()`, all-reduces and clips norm/value. WM and Critic weights are frozen. |
| Update Actor parameters | `policy_trainer.py:81` | Actor Adam `self.opt.step()` changes Actor encoder/decoder parameters. |
| Build Critic targets | `policy_trainer.py:99–109`, `objective.py:23–35` | Uses detached rollout rewards, termination probabilities and inputs under `no_grad`. |
| Update Critic parameters | `policy_trainer.py:117–123` | Critic MSE backward, all-reduce, norm clipping, then `critic_opt.step()`. Does not backpropagate into Actor or WM. |
| Update WM parameters | `wm_trainer.py:52–61`, scheduled separately | Supervised dynamics/termination loss backward and WM Adam step; not Actor reward maximization. |

Although the helper is named `gae_targets`, it returns the following TD(lambda)
**value targets**, not an advantage multiplied by an Actor log probability:

\[
G_H=V_H,\qquad
G_k=r_{total,k+1}+\gamma(1-d_{k+1})
\left[(1-\lambda)V_{k+1}+\lambda G_{k+1}\right].
\]

The Critic minimizes MSE against these fixed targets after the Actor update, using
the same already generated rollout. It is not a second path from the Critic loss
into Actor parameters. Installing `r_total` in `rollout.rewards` supplies both
Actor return and these targets without inventing a new optimization objective.

## 5. Numerical finding: original orientation helper at exact agreement

The installed mjlab `quat_box_minus(q1,q2)` computes
`axis_angle_from_quat(q1 * conjugate(q2))`
(`.venv/lib/python3.12/site-packages/mjlab/utils/lab_api/math.py:592–606`). The
axis-angle function chooses the shortest quaternion sign and a small-angle
series, but its other `torch.where` branch evaluates `sin(half_angle)/angle`
at zero (`math.py:500–508`). In the installed version, backward encounters this
undefined division even when the forward series branch is selected.

A read-only CPU autograd probe of existing functions produced:

| Input | Original `_sq_quat_err` value | Original quaternion gradient | Existing local `rotvec_from_quat` squared-log gradient |
|---|---:|---|---|
| Identity versus identity, FP32 and FP64 | 0 | NaN in all four components | Finite |
| Identity versus negative identity, FP32 and FP64 | 0 | NaN in all four components | Finite |

An additional tangent-space probe, differentiating through
`q = quat_from_rotvec([0,0,0])`, also returned three NaNs. This is not merely an
off-unit-sphere perturbation. Minimal reproduction with the existing environment:

```python
import torch
from supermimic.envs.mdp.rewards import _sq_quat_err
from supermimic.kinematics.rotations import quat_from_rotvec

x = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
q = quat_from_rotvec(x)
ref = torch.tensor([[1., 0., 0., 0.]], dtype=x.dtype)
print(torch.autograd.grad(_sq_quat_err(ref, q).sum(), x)[0])
# tensor([[nan, nan, nan]], dtype=torch.float64)
```

Probe environment: Torch `2.14.1+cu130`, mjlab `1.6.0`, CPU execution.
Installed `mjlab/utils/lab_api/math.py` SHA256:
`85e267ee202dc8dc2b340be07ec7abe9b10da9b82c3bf33e16889f422fcf9ca8`.

The proposed Box equation should use a finite-at-zero implementation of the same
shortest rotation-vector norm, with the existing local `rotvec_from_quat`
(`kinematics/rotations.py:55–65`) as a candidate. Its finite probe is not a complete
gradient validation. Required future checks include exact match, equivalent
quaternion signs, small rotations, nonidentity matches and behavior near pi.
Do not copy the numerical defect simply to reuse an identical function call.

The original robot reward also calls this helper. A formula-preserving numerical
repair needs separate review before declaring end-to-end BPTT gradients finite;
no shared helper, dependency or P4 file was changed here. The historical P4
robot reward regression checked values and composition, not this exact-match
orientation derivative, so its recorded passing results are preserved.

## 6. Review disposition and verification

This review resolves the missing **proposed formulation and placement**, not the
missing implementation or final reward configuration. Approval is still needed
for the four Box terms and their derived reference-velocity convention, deliberate
HOI next-frame scoring, and any formula-preserving quaternion numerical repair.
Final coefficients require measurements; none are selected by this document.
The original termination-reference convention and clip-end treatment must be
reconciled explicitly when P5 is authorized.

P4's physical target-validity issue, historical P3 PARTIAL result and M6.5
`training_ready=false` remain unresolved. Physical binary contact agreement,
independent left/right and any-hand/bilateral diagnostics, penetration and
post-release stability remain independent evaluation requirements. Measured
MuJoCo force/contact never enters this proposed differentiable reward, a new
prediction head or Actor/Critic observation.

Verification in this review was limited to source tracing, the eight-file
original-checkout comparison, reference-schema/hash inspection and the bounded
CPU quaternion probes above. No simulator rollout, training or optimizer update
was run; no P5 code or new test file was created. A before/after SHA256 inventory verifies
all 565 pre-existing protected files unchanged, including tracked sources and
reports, the accepted reference, asset manifest and user `.vscode/settings.json`.
The new document's complete diff and local source links were checked.

**Stop for review. P4 and the training code are unchanged.**
