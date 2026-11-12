# P5-B — joint imagination integration

Software integration result: **PASS** for the executed eager acceptance and
robot-only regression. Compiled validation is reported separately in
[P5-C](P5_C_REPORT.md). No training entry point was launched or enabled.

## Integration path and ownership

`hoi/imagination.py` supplies `HOIPolicyTrainer`, `HOIImagStep`,
`HOIRolloutContext`, `HOIPolicyReplayBuffer` and `HOIStartBatch`.
`HOIPolicyTrainer` only constructs components and supplies the robot Critic
projection; it inherits the original `PolicyTrainer.update`.

The exact path is:

```text
HOIPolicyReplayBuffer.sample_starts (original row selection/history algorithm)
  -> HOIStartBatch (clip, local/global frame, source hash, env/episode)
  -> PolicyTrainer.actor_loss (also called by its original update path)
  -> shared imagine recurrence + HOIRolloutContext
  -> original Policy(obs930, goal680) -> action29
  -> HOIImagStep._step
       JointWorldModel.step(physical84, action29, xi33)
         canonical86 + support18 + action29 + xi33 -> input166
         existing residual82 -> existing physical84 integration
       physical84 -> robot71 -> original FK/reward with safe HOI orientation
       four Box rewards + unchanged P4 proximity + original regularization
       return (state, scalar reward, terms13, termination, obs93, anchor7)
  -> shared policy_objective and backward through states/actions/history
  -> detached critic_batch -> shared gae_targets -> original Critic MSE loop
```

The reward insertion is immediately after the existing `wm.step` and before the
scalar reward return. There is one original imagination recurrence, TD(lambda)
recursion, Actor optimizer, Critic optimizer loop and training framework. No copy
of `Trainer.run`, Replay sampling, network architecture or robot reward was added.
Bounded tests invoke `actor_loss`, backward and detached Critic loss directly;
they do **not** invoke `update` or either optimizer's `step`.

The shared extensions are optional:

| File | Narrow extension | Robot-only default |
|---|---|---|
| `data/replay_buffer.py` | start eligibility/history/previous-action/envelope hooks | original sampling and RNG order |
| `imagination/rollout.py` | optional context, projection, active-row masks and dynamic terms | original current-frame scoring and clamp behavior |
| `imagination/objective.py` | optional lengths/bootstrap/continuation | original no-mask return and TD formula |
| `imagination/policy_trainer.py` | component hooks, reusable loss/batch construction, freeze restoration | original networks, update order, optimizer settings and minibatches |
| `envs/mdp/rewards.py`, `world_model/heads.py` | optional orientation-error callback | original quaternion function |

There is no force/contact measurement, Contact Decoder, Contact/Force Prediction
Head, BCE or force regression in this path. P2 force fields remain Replay sidecars
and are not consumed by the Actor reward or Joint WM input/loss.

## Reference identity and frame timing

`HOIReferenceRegistry` binds explicit replay IDs to explicit library IDs. It checks
source SHA, frame count, FPS/dt, native joint/body order, actual robot/body/goal
tensors and library global ranges. A local index is never guessed to be global.
The accepted full original NPZ is supported; compact/packed libraries require a
future provenance contract and are rejected. Synthetic distinct-source two-clip
tests verify mapping even when an incorrect index would still lie inside the library.

For transition `k`, Actor observes the current robot history and original
`goal_window(global(f+k),10)`. Only valid rows gather reference `f+k+1`, shared by
robot tracking, Box tracking, contact bool, termination pelvis features and next
Critic root error. Goal-window final-frame repetition remains the original
fixed-width behavior. It does not permit repeated terminal rewards.

| Start local frame | First reached reward frame | Contact |
|---:|---:|---|
| 258 | 259 | OFF |
| 259 | 260 | ON |
| 548 | 549 | ON |
| 549 | 550 | OFF |
| 838 | 839 | OFF; exactly one valid reward |
| 839 | none | rejected as a start |

The cache/registry gathers outside compilation and passes tensors into the step.
After a row ends, its state/history/frame stop advancing. No reward/FK/WM/reference
call evaluates that row as a padded transition. Diagnostics store evaluated local
and global indices, using `-1` for padding, and detached logging copies only.

Replay history retains original within-episode backfill and previous clean actions,
with additional clip/frame-continuity and transition-validity checks. A gap severs
all earlier history. A recorded physical failure caused by a different recorded
action is not assigned to a counterfactual imagined action. P2 lacks an always-known
remaining timeout and separate robot/object causes; optional start metadata accepts
these only when an authoritative caller supplies them. No episode age is guessed
across an overwritten ring.

## Dimensions and initialization

Physical84, canonical86, residual82, input166, support18, action29 and xi33 remain
unchanged. Every original robot consumer receives named `ROBOT=slice(0,71)`:
initial/reached/history analytic observations, FK, anchor, reward, termination,
root error, and `wm.spec.robot.canonical(...)` for the Critic. Joint canonical86
is not fed to the Critic.

Actor remains obs93×history10=930, goal680, latent32, action29. Critic remains
71+680+33+4=788. The pre-existing anchor option remains goal689/Critic797 and is
tested explicitly; it is not enabled by default. Strict original checkpoint loading
passed with no new keys/shapes. Checkpoint source:
`/home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt`, SHA256
`16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7`.
No optimizer/sampler state was loaded. Its training Git/dataset hashes are absent
from the source artifact and are not invented here.

Initialization metadata records the reward equation/version and explicit
coefficients, reference/derivative provenance, reached-frame scoring and zero natural
tail, dimensions, `optimizer_resume=false`, `sampler_resume=false`, and
`training_ready=false`. An original robot Critic is shape-compatible initialization,
not a calibrated HOI value function.

## Diagnostics and preserved limitations

All 13 additive terms, raw/weighted proximity, robot-positive/Box/regularization
aggregates, physical Box errors, ON/OFF counts, valid lengths and boundary reasons
are exposed using valid rows. Conditional means with no samples are omitted with
an explicit zero count. The rollout retains differentiable survival; training
metrics expose valid survival and terminal survival without changing the objective.

The Actor and Critic cannot directly distinguish Box-only changes when their
existing inputs are identical. Alias tests and direct zero Critic Box-gradient
tests document this. Existing termination features observe the robot pelvis;
they do not predict Box-only physical failures. Supplied object-failure mask tests
establish boundary handling only. These are unchanged design limitations requiring
P7 evaluation, not grounds to widen networks in P5.

The original robot-only matched-RNG regression compares the original checkout's
rollout, observations/noise/actions/values and gradients with the shared default
path exactly. Its legacy current-frame scoring and exact-match quaternion NaN
remain unchanged; the latter is explicitly tested and not relabeled as fixed.
