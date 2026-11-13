# P7.0.1 — Actor / Critic architecture review and design decision

**Review status: COMPLETE for approval. Architecture implementation: NOT AUTHORIZED.**

**PROPOSED final recommendation:** use an asymmetric dual-stream Actor with Robot
and Object histories and Robot and Object future goals (Actor B below). Use an
independent, privileged, reference-aware joint-state Critic (Critic B+ below).
Keep measured force and actual contact out of the main Actor/Critic interfaces.
Preserve the joint WM, differentiable imagination, Actor BPTT, original optimizers,
TD(lambda), and alternating collection/model/policy updates. This is not PPO.

The proposed Actor has raw observation **1080**, goal **830**, decoder input
**1058**, and action **29**. The proposed Critic has **1015** inputs and one value
output. These numbers describe a proposal, not the current networks. Current
Actor 930/goal 680 and Critic 788 remain unchanged. Approval must explicitly amend
the frozen HOI policy-interface clauses of the Master Plan before implementation.

Labels throughout this review:

- **VERIFIED** — established from the inspected source or identified existing
  evidence. Historical tests are identified as historical, not rerun results.
- **PROPOSED** — a concrete design or acceptance requirement, not implemented.
- **UNRESOLVED** — empirical performance, deployment, or authorization still open.

## 1. Authority, provenance and investigation scope

**VERIFIED.** Audit date: 2026-11-13. SMHOI branch `main`, HEAD
`36e90ecf2502121f2dcda2c445c16e78d176cc52`. The pre-existing untracked `.vscode/`
directory is outside this task. No applicable `AGENTS.md` was found. This review
adds only this document; no production source, network dimensions, assets,
references, solver parameters, training configurations, previous evidence or
Master Plan text was changed. No simulation, training, PPO, optimizer update,
commit or push was performed during P7.0.1.

Primary authority: the complete 3,152-line
[Master Plan](../SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md), SHA256
`68142af38edb0a9a443aac5815491aaae90dba696acc84d30fc54223c6e4a9cc`.
The complete document was read, including §§1.2, 4, 12 and 14–17. Reused evidence:
[P7.0 readiness](P7_TRAINING_READINESS.md),
[P7.0 specification](IMPLEMENTATION_SPEC.md),
[P5 specification](../p5/IMPLEMENTATION_SPEC.md), and P0
[architecture](../p0/ARCHITECTURE_AUDIT.md),
[file map](../p0/FILE_CHANGE_MAP.md),
[coverage](../p0/REQUIREMENT_COVERAGE.md),
[report](../p0/REPORT.md) and [closure](../p0/CLOSURE.md).
Historical P0 force-head proposals are superseded by the current Master Plan;
their presence does not authorize reinstating a force model.

The prior P7.0 minimal-current-Object/PPO proposal is preserved as a historical,
nonlaunchable proposal. It is not the final architecture decision for this
SuperMimic/BPTT investigation. Native reference-PD and independent PD carry
failures remain P6 open problems; fixing them is not a prerequisite for this
architecture decision or for investigating learned feedback control.

The requested [SuperMimic reproduction](https://github.com/CuboiLeo/SuperMimic-Reproduced)
was inspected locally at `/home/han/pyx/SuperMimic-Reproduced`, HEAD
`2ea2664bc40cc5d73dfff811a561073ae31463b9`, with no tracked modifications.
Public upstream `main` was independently resolved to
`3602aed6a397b3202ac052d688d47300f443a3ea`. All six requested code files have
identical Git blob hashes between that public tree and the local checkout:

| File under `src/supermimic/` | Verified Git blob SHA1 |
|---|---|
| `policy/nets.py` | `0b605334498b0ee4cf2c8330373e58f0da20669d` |
| `imagination/rollout.py` | `6316fa372f1859ee186308a7b1cb157e87ed35ce` |
| `imagination/policy_trainer.py` | `b4bdb04c82056c75239adc2c82314f45a2ff3de4` |
| `imagination/objective.py` | `e1e37b3ebed12dc603a4d8594cf57acbe4ac9d03` |
| `world_model/heads.py` | `2a20d27addf7f514051dee6f507e2b6392ea01ed` |
| `training/loop.py` | `b75a0d1996b9feba07204d62ac0e6d80cd77971e` |

The public reproduction now identifies itself as an independent implementation
written before official code was released. Its README points to the authors'
official repository. This audit follows the user's requested reproduction;
it does not silently substitute the official implementation or equate reproduction
choices with paper requirements. [Upstream provenance](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/3602aed6a397b3202ac052d688d47300f443a3ea/README.md).

## 2. Original SuperMimic implementation audit

### 2.1 Actor and goal interfaces — VERIFIED

The Actor is deterministic: `a = decoder(motion_encoder(goal), observation_history)`.
The Motion Encoder receives the **goal only**; proprioceptive history goes directly
to the Decoder. There is no action distribution, log probability, PPO ratio,
clipped surrogate or PPO optimizer in this architecture.
[Original nets](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/3602aed6a397b3202ac052d688d47300f443a3ea/src/supermimic/policy/nets.py#L19).

| Interface | Ordered content | Dimension |
|---|---|---:|
| One Robot observation | biased joint position relative to default 29; joint velocity 29; root angular velocity in body axes 3; projected gravity 3; previous **clean** action 29 | 93 |
| Robot history | 10 observations, oldest to newest, **term-major** flattening | 930 |
| One Robot goal frame | reference joint position relative to default 29; joint velocity 29; projected gravity 3; reference root angular velocity in its body axes 3; reference root linear velocity in its heading axes 3; reference root height 1 | 68 |
| Robot Future Goal | frames `f, f+1, ..., f+9`, **frame-major**, same-clip final-frame padding | 680 |
| Optional anchor goal | reference anchor position 3 and rotation 6 in actual Robot anchor frame, appended to goal | +9 |
| Motion latent | bounded by tanh | 32 |
| Decoder input | latent 32 followed by scaled history 930 | 962 |
| Action | normalized native joint setpoint command, linear network output | 29 |

“Future” includes the current reference frame. At 50 Hz, ten samples span
0.18 s between first and last; they are not `f+1..f+10`.
[Motion library](../../../../src/supermimic/envs/mdp/motion_library.py#L66),
[goal window](../../../../src/supermimic/envs/mdp/motion_library.py#L226),
[history ordering](../../../../src/supermimic/world_model/heads.py#L57).

Motion Encoder: `680 → 2048 → 1024 → 512 → 256 → 32`, SiLU hidden
activations, final tanh. Decoder: `962 → 2048 → 1024 → 512 → 256 → 29`,
SiLU hidden activations, linear output; final orthogonal initialization gain .01,
zero bias. Anchor mode changes encoder input to 689, not the history or action.
No learned running observation normalization is present.

Fixed divisors for Robot observations are `(1,5,2,1,1)` in the five term groups.
Goal divisors are `(1,5,1,2,1,1)` in the six goal groups. Uniform sensor-noise
half-widths are `.01 rad`, `.5 rad/s`, `.2 rad/s`, `.05` for joints, joint
velocities, root angular velocity and gravity; previous clean action is not
noised. Imagination reconstructs and re-noises history from clean state rather
than replaying the exact original observation-noise realization.
[Current matching nets](../../../../src/supermimic/policy/nets.py#L24),
[analytic observations](../../../../src/supermimic/world_model/heads.py#L30).

Physical execution can add exploration noise to the clean action before the
native scale/offset and actuator application. Replay stores clean and executed
actions separately. WM supervision uses executed action; observation history and
action-rate regularization use clean action. No reference joint position replaces
the Actor's output. Native PD actuation remains the control interface;
using it does not turn a feedback Actor into fixed-reference PD replay.
[Action implementation](../../../../src/supermimic/envs/actions.py).

### 2.2 Critic — VERIFIED

The Critic does **not** share the Actor's input interface. It receives privileged
canonical Robot state 71, scaled Robot goal 680, normalized episode context 33 and
root error 4: **788 → 256 → 128 → 1**, SiLU hidden activations and scalar linear
output. Anchor mode gives 797. The root errors are actual-minus-reference pelvis
position in **reference Robot heading** axes 3 and wrapped yaw error 1. They restore
tracking-drift information omitted by heading-invariant canonical state.
[Critic](../../../../src/supermimic/policy/nets.py#L85),
[root errors](../../../../src/supermimic/world_model/heads.py#L146).

`xi33 = [friction1, joint encoder bias29, torso COM offset3]`. Its offsets are
`[.75, 0×32]` and divisors `[.45, .01×29, .05×3]`. It is not Object mass,
Object contact force, support geometry, or a human-partner state. Root-error 4 is
unscaled in the current implementation. The Critic is a value estimator, not an
action-value network or a force estimator.

### 2.3 WM, imagination and update isolation — VERIFIED

Original Robot WM: canonical 71 + action 29 + xi33 =133 inputs; a shared
`2048,2048,2048` dynamics MLP produces residual 70. Velocity increments and pose
corrections are integrated into physical 71. Zero residual gives a constant-velocity
prior, not a static state at nonzero velocity. Reward/FK and observation heads are
analytic. A separate learned termination head uses height/tilt features
`[e_z,e_tilt,abs(e_z),abs(e_tilt)]`, MLP `4→64→64→1`; its BCE is **termination**
BCE, not Contact BCE.

`Trainer.run()` collects one physical vector step, then schedules WM updates and
Actor/Critic updates using the existing UTD mechanism. WM fitting autoregresses
over default eight-transition Replay windows without intermediate state detach.
It uses normalized physical-state loss plus termination BCE. Neither Actor nor
Critic is called while computing this supervised WM loss.
[Original Trainer](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/3602aed6a397b3202ac052d688d47300f443a3ea/src/supermimic/training/loop.py#L144),
[current shared Trainer](../../../../src/supermimic/training/loop.py#L181).

| Operation | Parameters updated | Gradient behavior |
|---|---|---|
| Simulator collection | None | `Trainer.act` is under `no_grad`; no differentiation through MuJoCo |
| WM fitting | Dynamics and learned termination | Replay tensors are data; Actor/Critic are outside the graph |
| Actor BPTT | Actor encoders and decoder | WM and Critic weights frozen; their **input Jacobians remain active** |
| Critic fitting | Critic only | Imagined features, rewards, termination probabilities and fixed targets detached; no graph to Actor/WM |

Original code sets WM/Critic `requires_grad=False` and later resets all flags to
true. Current SMHOI improves restoration with `frozen_parameters`, preserving
individual flags and restoring them in `finally`; this already exists and should
be reused. Actor/WM need not be explicitly frozen during Critic regression because
their graph has been detached and only Critic parameters enter that optimizer.
[Freeze context](../../../../src/supermimic/imagination/policy_trainer.py#L19).

### 2.4 Actor BPTT and terminal Critic bootstrap — VERIFIED

For an untruncated horizon H, the implemented objective is

\[
c_0=1,\qquad c_{t+1}=c_t(1-\hat d_{t+1}),
\]
\[
J=\mathbb E\left[\sum_{t=0}^{H-1}\gamma^t c_t r_{t+1}
  +\gamma^Hc_H V_\psi(\hat s_H,g_H,\xi,e_H)\right],
\qquad L_\pi=-J/H.
\]

The terminal value is **not detached**. Actor gradients pass through action→WM→
state→reward, state→observation→later action, state→termination→survival, and
terminal state→Critic features→frozen Critic. Anchor goals, when enabled, add
another state-dependent path; ordinary Robot goal 680 is reference-only.
Freezing Critic weights does not eliminate `∂V/∂s`. Merely checking that a value
appears in the objective is insufficient: its state derivative must be tested.
[Original objective](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/3602aed6a397b3202ac052d688d47300f443a3ea/src/supermimic/imagination/objective.py#L10),
[current terminal gather](../../../../src/supermimic/imagination/policy_trainer.py#L85).

Current HOI uses valid length L per sample, `gamma**L`, and the reached state at L.
Natural finite clip end and authoritative hard failure have zero tail; timeout
before clip end and artificial imagination horizon retain bootstrap. Coincident
natural end/failure takes precedence. Invalid padding contributes neither rewards
nor Critic regression samples. The Actor loss still divides by nominal H.
[Masked objective](../../../../src/supermimic/imagination/objective.py#L10).

### 2.5 Critic TD(lambda) — VERIFIED

\[
R_H=V_H,\qquad
R_t=r_{t+1}+\gamma(1-\hat d_{t+1})
\big[(1-\lambda)V_{t+1}+\lambda R_{t+1}\big].
\]

The same imagined rollout used for the Actor is detached for Critic fitting.
Targets are computed once, then held fixed over five shuffled epochs/four
minibatches by default. There is no EMA target Critic. The loss is MSE on
pre-transition rows, with the terminal value used for targets rather than an
extra terminal regression sample. Defaults include gamma .99, lambda .95,
Adam learning rate 2e-4 and Critic gradient norm clip 1. The MSE is not separately
weighted by survival. Current HOI masks invalid rows and retains the existing
TD(lambda) recurrence. Do not duplicate it for a wider Critic.
[Original policy trainer](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/3602aed6a397b3202ac052d688d47300f443a3ea/src/supermimic/imagination/policy_trainer.py#L97),
[current Critic batch](../../../../src/supermimic/imagination/policy_trainer.py#L103),
[TD targets](../../../../src/supermimic/imagination/objective.py#L34).

### 2.6 Paper requirements versus reproduction choices — VERIFIED distinction

The paper specifies alternating collection/WM/policy learning, deterministic
policy BPTT, a terminal value with fixed imagined TD(lambda) targets, and no EMA
target Critic. Its appendices give ten observation frames, ten reference frames,
the encoder/decoder widths above, a 32D tanh latent, and a 256/128 Critic.
These support the algorithmic structure; they do not prescribe SMHOI's Object
interface. [SuperMimic paper, §§3.1–3.3 and Appendices A–D](https://arxiv.org/pdf/2610.28480).

The exact physical 71/canonical 71 layout, 68D goal construction, heading and
normalization conventions, residual parameterization, SiLU/initialization,
privileged root-error 4 and optional anchor 9 are implementation decisions documented
in [the reproduction plan](../../../../docs/REPRODUCTION_PLAN.md), not universal
requirements of the method. The selected implementation's Robot reward scores
`s_(t+1)` against `ref_t` and advances the reference afterwards. Current HOI
explicitly scores the reached state against `ref_(t+1)`, as required by P5. Keep
that HOI convention and preserve original robot-only behavior through adapters.
The method's equations alone do not justify conflating the two indexing choices.

## 3. Current SMHOI physical and canonical data — VERIFIED

Let W denote environment-local positions expressed along world axes; E be batch
size; t be a physical control-step index; f_t be its clip-local reference frame.
All spatial lengths are metres, linear velocities m/s, joint angles radians and
angular velocities rad/s. Quaternions use wxyz and represent body-to-world
orientation. “Root” means the free-joint origin, not automatically the COM.

### 3.1 Exact layouts

| Physical 84 slice | Tensor field | Width / frame |
|---|---|---|
| `[0:3]` | Robot root position | 3, W |
| `[3:7]` | Robot root quaternion | 4, body→world |
| `[7:36]` | Robot joint positions | 29, native named joint order |
| `[36:39]` | Robot root-origin linear velocity | 3, world axes |
| `[39:42]` | Robot angular velocity | 3, Robot root body axes |
| `[42:71]` | Robot joint velocities | 29 |
| `[71:74]` | Object root position | 3, W |
| `[74:78]` | Object quaternion | 4, Object body→world |
| `[78:81]` | Object root-origin linear velocity | 3, world axes |
| `[81:84]` | Object angular velocity | 3, Object body axes |

Thus Robot physical 71 + Object physical 13 = physical 84. `JointStateIO` uses named
qpos/qvel addresses, subtracts each environment origin from both positions and
does not concatenate arbitrary global MuJoCo arrays. Box COM velocity, when
needed for diagnostics, is `v_root + omega_world × (R_obj * COM_offset)`.
[State access](../../../../src/supermimic/hoi/state.py#L99).

Let h(s) be the actual Robot root yaw-only orientation and `R_h` its matrix.
The unscaled Object canonical feature is

\[
\phi_o(s)=\left[
R_h^\top(p_o-p_r),\quad
\operatorname{r6}(R_h^\top R_o),\quad
R_h^\top v_o^W,\quad \omega_o^{body}
\right]\in\mathbb R^{15}.
\]

**The linear velocity is absolute Object root velocity expressed in heading axes.
It is not `v_o-v_r`, and not the derivative of the root-relative position.**
The latter would require Robot linear velocity and rotating-frame terms. The
Master §4.1 table's phrase “relative linear velocity” must be clarified to match
the verified implementation; no velocity conversion is silently proposed.
[JointStateSpec](../../../../src/supermimic/hoi/state_spec.py#L89).

`r6` is the first two columns of a rotation matrix flattened **row-major**:
`[R00,R01,R10,R11,R20,R21]`. It is not the alternative two-column concatenation
`[R00,R10,R20,R01,R11,R21]`. Quaternion sign changes give identical r6.
[Rotation helper](../../../../src/supermimic/kinematics/rotations.py#L78).

| Canonical tensor | Ordered content | Dimension |
|---|---|---:|
| Robot canonical | height 1, yaw-removed rotation 6, heading linear velocity 3, body angular velocity 3, joint position 29, joint velocity 29 | 71 |
| Object canonical | relative heading position 3, heading rotation 6, absolute heading linear velocity 3, Object-body angular velocity 3 | 15 |
| Joint canonical | Robot 71 followed by Object 15 | 86 |
| Support context | two support roots, each relative heading position 3 and rotation 6, manifest order | 18 |
| WM input | canonical 86 + support 18 + executed action 29 + normalized xi33 | 166 |
| WM residual | Robot 70 + Object 12; no force/contact component | 82 |

`JointStateSpec.canonical` already divides by fixed scales. Robot divisors are
`(1,1,1,2,1,5)` by its six groups; Object divisors are `(1,1,1,2)` by groups
`(3,6,3,3)`. Support positions/rotations currently have no additional scaling.
Support 18 is recomputed for each actual or imagined state and remains separate
from episode-constant xi33. It describes geometry, not measured support contact.
[Support encoding](../../../../src/supermimic/hoi/support_context.py#L64),
[Joint WM](../../../../src/supermimic/hoi/world_model.py#L35).

### 3.2 Replay, actions and histories

One `HOIReplayBuffer` row owns
`(s_t, a_clean_t, a_exec_t, s_(t+1), physical_interval_t, masks_t, clip, f_t)`.
Default WM windows contain `[9,N,84]` states and `[8,N,29]` executed actions.
The extra terminal state row has no outgoing transition. Transition masks
distinguish physical failure, timeout and natural clip end. Contiguity requires
matching next-state/current-state, xi, clip and frame progression; windows cannot
cross an episode, clip, reset, terminal boundary or overwritten ring interval.
[HOI Replay](../../../../src/supermimic/hoi/replay.py#L21).

Inherited `StartBatch.hist_states` is `[10,N,84]` for HOI, containing
`s_(t-9)..s_t`; `hist_actions` is `[10,N,29]`, containing the preceding **clean**
actions `a_(t-10)..a_(t-1)`. Robot history can be reconstructed from `[..., :71]`
and Object history from `[..., 71:84]`, with Robot root/heading from the same
historical state. History does not require another physical-state recorder.
[Start/history extraction](../../../../src/supermimic/data/replay_buffer.py#L162).

`HOIPolicyReplayBuffer` adds clip/local/global frame and hash identity. Its history
checks require consecutive reference frames, one clip, valid transitions and no
terminal rows. A gap severs all earlier history. Missing pre-episode history is
backfilled with the first valid state, and missing previous actions are zero.
The proposed Object history must apply **the same indices/backfill**, never
continue an Object history through a Robot reset.
[Policy Replay](../../../../src/supermimic/hoi/imagination.py#L40).

Storage sufficiency is distinct from integration: the P2 recorder installer still
constructs the base Replay adapter; the full training entry must install the
policy-capable adapter. Existing Replay records physical state, not camera
measurements or the exact historical noise of a future Object estimator. Such
estimator behavior needs its own observation contract, not a second physical 84
recorder. Current Replay reward entries are placeholders and do not supply a new
real-transition Critic objective.

## 4. Reference and future-goal availability — VERIFIED / PROPOSED

**VERIFIED.** Both original and candidate NPZs contain Object position `[840,3]`,
quaternion `[840,4]`, global Boolean `object_contact[840]`, Robot joints/body poses
and velocities, names and FPS. They do **not** contain Object velocity or measured
force. Both run at 50 Hz. Original and candidate intent bytes match; ON is 260–549
inclusive. `intent_ref` width 2 in Replay means **`[c_f,c_(f+1)]`**, not left/right.

| Quantity | Available source / exact semantics | Ready-made policy input? |
|---|---|---|
| Object position/quaternion | Validated immutable NPZ / `ReferenceClip` | Stored; Actor currently ignores |
| Object linear velocity | `RewardReferenceCache` centered position differences, one-sided endpoints; root-origin world axes | Derived cache, not an NPZ field |
| Object angular velocity | shortest world quaternion increment/log divided by sample span; world axes | Derived cache; body version also retained for audit |
| Binary intent | unchanged global Boolean at exact local frame | Reward gate only |
| Robot goal 680 | `MotionLibrary.goal_window` | Yes, current Actor/Critic |
| Object arbitrary-frame lookup | `HOIReferenceRegistry.gather(clip_id,local_frame)` | Yes for references/reward; no Object goal encoder yet |
| Object future window | explicit valid indices plus registry gathers | Feasible without NPZ changes; policy wrapper not implemented |

The derivative cache computes in FP64, normalizes quaternion **copies**, checks
shortest-log branch conditions and stores the requested dtype. Its angular output
is world angular velocity; canonical Object goals need conversion to the
**reference Object's body frame**. Derived reference twist is a commanded
reference quantity, never a substitute for measured simulator velocity.
[Derivative cache](../../../../src/supermimic/hoi/reward_reference.py#L24).

`ReferenceResetCache` separately constructs synchronized Robot 71/Object 13 reset
states. The future-goal implementation should reuse the checked reward-reference
cache rather than add a third finite-difference implementation.
`MotionLibrary.ref71` stores reference root angular velocity in **world** axes,
unlike physical Robot 71. Use `ref_qpos_qvel` when constructing physical-format
reference states; do not concatenate `ref71` as though it already had body omega.
[Reference conversion](../../../../src/supermimic/envs/mdp/motion_library.py#L236).

The registry verifies explicit Replay-ID→library-ID bindings, source SHA, length,
dt/FPS, joint/body names/order and Robot library tensors. A Robot reference from
the original NPZ must not be combined with Object data from another candidate
under the same identity. It rejects unknown IDs and out-of-range actual gathers.
[Registry](../../../../src/supermimic/hoi/reward_reference.py#L139).

**PROPOSED goal construction:** for current local frame f, form same-clip indices
`min(f+j,T_clip-1)` for j=0..9, gather synchronized Robot and Object references,
and construct the canonical Object goal defined in §5. Fixed-width **goal-only**
edge padding repeats the last valid reference, exactly like Robot goals; retain
the future validity metadata for diagnostics. Never clamp an invalid starting
frame or use padded goal rows as reward transitions. `ReferenceClip.future_window`
already distinguishes padding with a validity mask. Reward and TD validity remain
the existing finite-clip masks; no repeated final reward is introduced.

Current exposure is deliberately asymmetric: `MotionLibrary`, ordinary policy
observations and `ImagRollout.goals` expose Robot only. `ReferenceClip`, reset
cache, `HOIReferenceRegistry`, `HOIImagStep` reward input and joint WM already have
Robot–Object information. Access to a reference inside reward computation does
not mean that the Actor observes it.

## 5. Actor candidate comparison and precise input specification

All designs in this section are **PROPOSED**, with anchor mode disabled for the
preferred new schema. Keep the existing optional Robot anchor 9 only as an explicit
separate compatibility/ablation variant; it would add 9 to raw goal widths.

### 5.1 Common definitions

Robot history `O_r` preserves the raw 93/term-major 930 interface and its existing
normalization. Object observation uses the **unscaled** phi_o15 from §3; its
normalization is applied exactly once in the new Object branch. Existing
`object_canonical` returns scaled values, so an adapter must explicitly recover
raw units or share the same feature algebra with a raw-output function. Do not
divide an already normalized Object feature twice or change WM normalization.

For each history sample j, use that sample's actual Robot root/heading:

\[
O_o(t)=[\phi_o(s_{t-9}),\ldots,\phi_o(s_t)].
\]

Flatten Object history **by group then time**: position 30, rotation 60, linear
velocity 30, angular velocity 30 =150. It is a sequence of contemporaneous
Robot-relative observations, not a trajectory re-expressed in one fixed current
heading. Both Robot and Object histories end at s_t, before a_t.

For Object future goals use reference Robot root/heading at **each future frame**:

\[
G_o(f)=\operatorname{flatten}_{frame}
\left[\phi_o(s^{ref}_{f+j})\right]_{j=0}^{9}\in\mathbb R^{150},
\]

with the goal-only padding from §4. Reference Object angular velocity is converted
world→reference-Object-body before phi_o. These goals describe desired relative
formation, orientation and motion. They do not substitute reference Box pose for
the actual Box observation or WM state. Robot goal 680 remains unchanged.

Object observation and goal divisors are `(1 m, 1, 1 m/s, 2 rad/s)` for groups
`(3,6,3,3)`, matching the existing Object canonical scale meaning. These are fixed
initial interface scales, not claimed calibrated deployment normalizers. Preserve
Robot normalization separately; do not feed goal 830 to `_goal_scale_vector`,
which incorrectly interprets all features as repeated 68D Robot frames.

### 5.2 Exact dimensions and candidate networks

| Actor | Robot history | Object observation | Robot goal | Object goal | Raw observation / goal | Decoder input |
|---|---:|---:|---:|---:|---:|---:|
| Existing | 930 | absent | 680 | absent | 930 / 680 | 32+930 =962 |
| A: minimal feedback | 930 | current 15 | 680 | absent | 945 / 680 | 32+930+15 =977 |
| B: dual-stream tracking, **preferred** | 930 | history 10×15 =150 | 680 | 10×15 =150 | 1080 / 830 | 32+930+64+32 =1058 |
| C: adaptive history | 930 | history 10×15 =150 | 680 | absent | 1080 / 680 | 32+930+64 =1026 |

For A, append normalized current phi_o15 directly to the existing Decoder prefix.
For B and C, encode Object history using `150→128→64`, SiLU after the hidden 128
layer, linear 64 output. For B, separately encode Object goal using
`150→128→64→32`, SiLU hidden activations and final tanh. Preserve the original
Robot Motion Encoder `680→2048→1024→512→256→32`.

B concatenates `[Robot latent 32, scaled Robot history 930, Object history
embedding 64, Object goal latent 32]`, then uses the original Decoder hidden widths
`2048,1024,512,256` and linear action 29. There is one shared action decoder,
not separate competing Robot and Object controllers. All Actor branches are
updated by the same Actor optimizer/BPTT objective. No extra imitation loss,
attention mechanism, recurrent model, force head or latent-alignment objective
is required. Object and Robot encoders do not share weights.

The goal latent is now structured as Robot 32 plus Object 32; it must not be
misreported as an unchanged total 32 bottleneck. Object history 64 is an observation
embedding, not an additional physical state. Raw input dimensions and decoder
dimensions are different contracts.

### 5.3 Research comparison

| Question | A: current Object only | B: both histories and goals | C: both histories, no Object goal |
|---|---|---|---|
| React to independent Box perturbation at fixed Robot/reference? | Can, through current pose/twist | Can, through current and past Object observations | Can, through current and past Object observations |
| Same current Object state but different recent slip/motion? | Cannot distinguish using Object branch | Can distinguish finite history | Can distinguish finite history |
| Different desired Object motions with same Robot goal? | Must infer task from Robot goal | Explicit task conditioning | Must infer task from Robot goal/history |
| Learn Robot motion | Original Robot branch | Original Robot branch | Original Robot branch |
| Learn Object motion / coupled motion | Instantaneous kinematics only; coupling through decoder/WM | Temporal Object branch plus paired goals and shared decoder/WM | Temporal adaptation, but Object destination ambiguous |
| Adaptive recovery | Useful minimal baseline | Best information for task-conditioned correction, subject to reward tradeoff | Strong comparator for reactive/user-led deviations |
| Main limitation | No Object temporal evidence beyond supplied twist | May overfit reference; still limited to observed task/partner information | Object target/placement/release intention can be ambiguous |

The “A has no history” limitation must be stated carefully: phi_o includes linear
and angular velocity, so A is not pose-only and may be sufficient for a perfectly
observed Markov rigid-body state. History is justified by noisy/latent interaction,
estimation, slip trends and the history-based Robot controller—not by asserting
that current pose+twist is mathematically incapable of feedback control.

**Recommendation: B.** The research task still supplies a joint reference and
scores Object tracking throughout. Removing Object goals would hide task
information while preserving its reward consequences. B lets the policy learn
Robot motion, Object motion, their relative coordination and feedback corrections.
Choose smaller Object encoders because a single rigid Box has much less internal
structure than a 29-DoF humanoid. Equal temporal spans provide clean alignment;
equal network capacity or weight symmetry is neither necessary nor justified.

Object future conditioning is a target, not a hard action constraint. It may help
anticipate lift, rotation and placement; whether it actually improves these
outcomes is **UNRESOLVED**. It can also encourage overfitting or resisting a
partner's changed intent. B is therefore a preferred research design, not an
empirically established winner over C.

### 5.4 Adaptivity, frame alternatives and deployment

Reference-relative goals avoid injecting another absolute-world path command.
An independently moved Box changes actual phi_o while Robot observations and all
goals remain unchanged. This supplies the missing direct feedback channel. It
does not guarantee that random, zero-migrated or trained weights use that channel;
feature sensitivity and policy sensitivity require separate acceptance tests.

A common planar translation/yaw of the actual Robot and Box can leave all these
local Actor inputs unchanged while absolute reference errors change. Privileged
Critic inputs cannot overcome this Actor information alias. B should be described
as **adaptive relative coordination conditioned on a reference**, not complete
absolute-path observability. The unchanged tracking reward still penalizes some
deviations; changing the input frame does not relax the reward.

The main alternative is a **current-actual-root-anchored Object goal**, with each
future reference Object position expressed as
`R_h(actual_t)^T * (p_obj_ref(f+j)-p_robot(actual_t))`, orientation relative to the
same current heading, and velocity conventions stated separately. It remains 150D
and exposes absolute trajectory drift, but requires localization in the reference
world, makes the goal state-dependent in BPTT, and may resist sustained user-led
motion. Compare it as a controlled frame ablation; do not silently mix its
gradients/checkpoints with the preferred reference-only goal.

Deployment observability differs from simulation access. Robot encoders/IMU and
previous commands support the existing Robot history. Object phi_o requires a
time-synchronized Object pose estimate, Robot heading, and causal Object twist
estimation; absolute Object linear velocity in heading axes additionally requires
world-relative motion/ego-motion estimation. Relative pose alone cannot supply it.
No such deployment estimator is implemented here. Privileged simulator Object
state is an upper-bound observation interface, not proof of deployability.

For initial software comparisons, use explicit zero Object measurement noise,
preserve the existing Robot noise protocol, and label this a state-observed
simulation setting. Deployment experiments must calibrate Object pose/twist noise,
latency and dropout. Apply rotational noise on SO(3) before r6 conversion, not by
independently corrupting six matrix entries. Use the same causal observation
transform/noise law in physical collection and imagination. Never estimate actual
Object velocity using future physical frames; reference central differences are
permitted because the command trajectory is known in advance. Do not invent sensor
noise magnitudes or silently add validity dimensions in this review.

No current input describes human pose, intention or applied wrench. Recovery from
a Box perturbation is a valid adaptive-coordination test; sustained independent
human intent is a different task/command decision. B versus C cannot establish
human-collaboration capability without that later protocol.

## 6. Critic candidates and final privileged interface

### 6.1 Candidate comparison — PROPOSED

Evaluate Critic choices independently of Actor choices. A privileged Critic may
use noiseless physical state, known physical context and task errors unavailable
to the deployed Actor. Privilege does not authorize using an outcome before it
occurs or supplying an unmodelled measurement at an imagined future state.

All counts below use Robot goal 680 and no anchor 9. `xi33` and `support18` are
listed separately to avoid concealing an 18D change inside “environment context.”

| Critic | Ordered information, before optional additions | Total |
|---|---|---:|
| Existing | Robot canonical 71 + Robot goal 680 + xi33 + root error 4 | 788 |
| A0: joint-state | joint canonical 86 + Robot goal 680 + xi33 + root error 4 | 803 |
| A: joint-state with full existing physical context | A0 + support 18 | 821 |
| B4: unsigned-error comparison | A + existing `box_errors`4 | 825 |
| B12: signed log-error comparison | A + signed Object position 3/rotation-log 3/linear 3/angular 3 | 833 |
| B: Object-aware privileged | A + Object error 15 defined below | 836 |
| **B+: preferred** | B + Object future goal 150 + previous clean action 29 | **1015** |

A fixes the existing Critic's direct Box-state blindness. Support 18 removes an
additional ambiguity: equal heading-canonical Robot–Object configurations can
occupy different locations relative to fixed supports. It is already justified
for dynamics and is equally relevant to value near pickup/placement. A still
does not explicitly state which Object motion is desired.

B includes directional Object tracking information. Existing `box_errors` returns
**four squared scalar magnitudes**, not 12 or15 coordinates. Feeding those four
scalars can summarize error size but discards correction direction. B12 preserves
direction with a signed rotation log but has a pi branch cut. Prefer B's 15D
version with a continuous rotation residual 6 for broad orientation perturbations.
[Current error computation](../../../../src/supermimic/hoi/tracking_reward.py#L92).

B+ additionally conditions value on the same Object future command as the Actor
and on the previous clean action used in action-rate reward. These additions are
scientifically motivated, not required merely to make shapes fit. B without
Object goals can alias different intended future Object motions with identical
current error. Omitting previous action aliases different immediate smoothness
costs. Both pieces are available in Replay and throughout imagination without a
new physical recorder or prediction head.

### 6.2 Exact preferred Object error 15 — PROPOSED

At state index t, use the actual/imagined `s_t` and reference at **f_t**, not f_t+1.
Let `H_ref=R_h(q_robot_ref(f_t))`, and let `R_o` and `R_o_ref` be actual and
reference Object body-to-world rotations. Define actual-minus-reference errors:

\[
e_p=H_{ref}^{\top}(p_o-p_o^{ref})\quad(3),
\]
\[
\Delta R_h=H_{ref}^{\top}R_oR_{o,ref}^{\top}H_{ref},\qquad
e_R=\operatorname{r6}(\Delta R_h)-\operatorname{r6}(I)\quad(6),
\]
\[
e_v=H_{ref}^{\top}(v_o^W-v_o^{ref,W})\quad(3),\qquad
e_\omega=H_{ref}^{\top}(R_o\omega_o^{body}-\omega_o^{ref,W})\quad(3).
\]

Concatenate `[e_p,e_R,e_v,e_omega]`. Position/linear/angular errors use reference
Robot heading; orientation is a world-relative rotation expressed in that same
heading. Rotation residual 6 is dimensionless, sign-invariant and zero at agreement;
it is **not** a canonical rotation 6 to pass into `object_from_canonical`. Do not
subtract raw quaternions or subtract body-frame angular velocities from different
bodies directly. Fixed divisors are `(1 m,1,1 m/s,2 rad/s)` for these groups,
serialized independently of reward sigmas. No force normalization is introduced.

### 6.3 Preferred Critic architecture — PROPOSED

| Named block | Width | Information source / scaling |
|---|---:|---|
| Joint canonical state | 86 | Existing normalized JointStateSpec; do not normalize twice |
| Support context | 18 | Existing dynamic support representation; position divisor 1m, rotation divisor 1 |
| Object error | 15 | Equations above, same-index reference, explicit divisors |
| Previous clean action | 29 | `start.prev_action` at t=0; imagined `a_(t-1)` thereafter; dimensionless |
| **Physical/reference feature pack** | **148** | Four blocks above |
| Robot Future Goal | 680 | Original Robot goal scales/order |
| Object Future Goal | 150 | §5 reference-local frame-major goal scales/order |
| **Goal pack** | **830** | Robot and Object blocks remain separately named |
| Episode xi | 33 | Original normalization |
| Root error | 4 | Original reference-heading position/yaw features, original scaling |
| **Total** | **1015** | `148 + 830 + 33 + 4` |

Use `1015→256→128→1`, SiLU hidden activations and scalar linear output. Preserve
the original Critic optimizer, target calculation, value units and bootstrap masks.
Use a separate HOI Critic wrapper/feature schema; do not invoke the original goal
scale helper with 830, because it would incorrectly scale 12 Robot frames plus 14
extras. No Actor encoder weights are shared with the Critic: Critic regression
must not mutate Actor parameters, and freezing Critic must not freeze shared Actor
branches accidentally. A wider hidden Critic is an empirical capacity ablation,
not the default recommendation.

This Critic is more informative than the current one, but is still an approximate
value representation: the Actor depends on observation histories/noise whereas
the Critic does not receive the entire history. Previous action removes a known
reward alias, not all partial observability. State/reference goals can also omit
future intent changes beyond their window. Do not claim a fully Markov value
function. A history-conditioned Critic is a later ablation if residual value
aliasing is measured, with independent weights and explicit dimensions.

### 6.4 Feature availability, timing and gradients

| Feature | Time/type | Real Replay | Imagined states 0..L | Differentiability / leakage |
|---|---|---|---|---|
| Joint canonical 86 | Current state s_t | Reconstruct from physical 84 | Recompute from every s_t | Differentiable except coordinate singularities; no future outcome |
| Support 18 | Current state + static context | State and pinned support asset | Recompute each step | Differentiable in Robot pose; no sensor query |
| Robot/Object goals | Reference level at f_t..f_t+9 | Registry + row identity | Registry + context frame | Preferred goals constant w.r.t. policy; future command is not future measurement |
| Object error 15 | Current state + reference f_t | State + registry | State + registry | Differentiable in Object state; no detached diagnostics |
| Root error 4 | Current Robot + reference f_t | Reconstructible | Already constructed | Differentiable away from heading/wrap singularities |
| xi33 | Episode context | Stored | Constant from sampled start | Constant w.r.t. actions; privileged simulator parameter |
| Previous clean action 29 | Incoming action a_(t-1) | Existing action/history | Initial Replay action, then prior Actor output | Causal; imagined values must retain gradient |
| Reference intent 1, optional only | Reference level c_f | Existing label / registry | Exact lookup available | Exogenous nondifferentiable constant; not an actual-contact measurement |
| Measured interval force/contact | Transition [t,t+1], after a_t | Stored as row-t outcome | Not predicted | Leaks if supplied at s_t; unavailable in counterfactual future |

TD(lambda) algebra does not require Actor and Critic inputs to match. It requires
that `V_t` and `V_(t+1)` consistently condition on the intended state/task at those
times. B+ meets feature availability for both endpoints without changing that
recursion. In Actor optimization its terminal gradient becomes

\[
\frac{\partial V_L}{\partial F_L}
\frac{\partial F_L}{\partial \hat s_L}
\frac{\partial \hat s_L}{\partial a_{0:L-1}}
\frac{\partial a_{0:L-1}}{\partial\theta},
\]

plus the explicit path through previous action `a_(L-1)` and the existing survival
term. `F_L` includes Object state/error/support, all computed in Torch with a live
graph. Critic **weights** remain frozen. The function being bootstrapped changes
because the information improves; the discount, terminal selection and TD rules
do not. During Critic regression all these features and targets are detached as
before. No gradient through measured force is assumed.

## 7. Force/contact privileged-information investigation

### 7.1 Distinct physical and reference quantities — VERIFIED

Two filtered hand sensors include wrist pitch and yaw collision bodies against
the Box body; `reduce=netforce`, one slot, `secondary_policy=error`, history 4.
The per-transition captured arrays are chronological, even though native sensor
history is newest-first. The force sign is **hand on Box**, world axes.
[Sensing and aggregation](../../../../src/supermimic/hoi/contact.py#L16).

| Quantity per transition | Shape, excluding batch | Meaning / units |
|---|---|---|
| `intent_ref` | `[2]` | Global desired labels at start/end; **not** left/right |
| `force_raw_world` | `[4,2,2,3]` | Four solves × L/R × pitch/yaw primary bodies × xyz, N |
| `box_quat_solve` | `[4,4]` | Exact contemporaneous solve-time Object orientation, wxyz |
| `found_substeps` | `[4,2,2]` | Separately captured matched constraint counts |
| `force_mean_box` | `[2,3]` | Mean of four hand force vectors, each rotated at its own solve, N |
| `impulse_world` | `[2,3]` | `.005 * sum_k(F_world_k)`, N·s |
| `force_peak` | `[2]` | Largest norm of each hand's net force over four solves, N |
| P2 `occupancy` | `[2]` | Fraction above configured P2 diagnostic threshold |
| `found_final` | `[2]` | Fourth-solve counts, before final forward; not interval contact Boolean |
| P6 calibrated hands | `[2]` | Independent L/R force-bearing decisions recomputed from raw data |

For hand h, first sum simultaneous pitch/yaw vectors, then take the norm:

\[
F_{h,k}=\sum_bF_{h,k,b},\quad
o_h={1\over 4}\sum_{k=1}^4[\|F_{h,k}\|>\tau_h],\quad
c_h^{actual}=[o_h>\rho_h].
\]

P6 thresholds are L/R `.0743401046875112` / `.07375101817770177 N`, with strict
occupancy `>.25`, hence at least two of four solves. Calibration uses the midpoint
of the no-contact maximum and the fifth percentile of controlled positive loads,
with an independent held-out rotation; it does not fit against reference intent.
The preserved calibration explicitly excludes subthreshold and genuine one-solve
impacts from its guarantee. P2 `.001 N` occupancy is a different diagnostic and
must not be relabeled as calibrated contact.
[Calibration analysis](../../../../outputs/hoi/contact_supervision/contact_validation/scripts/analyze_p6.py#L15),
[calibration evidence](../../../../outputs/hoi/contact_supervision/contact_validation/calibration.json),
[actual-contact decision](../../../../src/supermimic/hoi/contact_validation.py#L10).

Constraint `found`, net force, calibrated force-bearing contact, collision-surface
distance and EEF-target proximity are different observations. Opposing local
forces can cancel in a net vector. A geometric proximity feature does not establish
load-bearing contact. Any-hand OR and bilateral AND remain distinct evaluations;
one global reference bit does not provide two annotated hand labels. The interval
measurement is compared with the exact **reached** reference label c_(f+1), without
claiming that this interval statistic is an instantaneous state sensor.

P7.0 retained 50 world transitions and raw physical interval data, sampled an
eight-transition `[9,2,84]` window, and separately recomputed calibrated contact.
Its Actor was untrained and it made zero optimizer updates. The measured 2 bilateral
intervals are not learned carrying evidence. These are preserved diagnostics,
not new P7.0.1 physical experiments.
[P7 sample](../../../../outputs/hoi/contact_supervision/p7_readiness/physical_final/sampled_physical_window.pt),
[P7 recomputation](../../../../outputs/hoi/contact_supervision/p7_readiness/analyze.py).

### 7.2 Critic C: explicit feature choices — PROPOSED, not selected

Use B=836 as the comparison base. The alternatives below are **individual
additions**, not an unspecified bundle; none is included in preferred B+1015.

| Added feature | Added width / total | Feasibility and decision |
|---|---:|---|
| Current reference intent c_f | +1 /837 | Available in both domains; could disambiguate gated reward, but requires a contact-observation amendment; optional ablation |
| Actual L/R calibrated contact | +2 /838 | Real interval only; no faithful imagined counterpart; reject for main Critic |
| L/R mean Box-local force | +6 /842 | Real interval vector, no future force model; reject |
| L/R calibrated occupancy | +2 /838 | Interval statistic; same availability problem; reject |
| L/R world impulse | +6 /842 | Interval statistic, N·s, not force; reject |
| L/R peak force | +2 /838 | Interval extreme, unavailable from state 84 alone; reject |
| L/R discrepancy `c_ref-c_actual_h` | +2 /838 | Requires unavailable actual contact; global bit is not new hand annotation; reject |
| Differentiable EEF-target distances | +2 /838 | Reconstructible geometry in both domains; not measured contact; optional, separately amended ablation |

For illustration only, adding reference intent 1, actual contact 2, force 6,
occupancy 2 and discrepancy 2 together would give 849, not the preferred Critic.
It would add redundant signals and an unresolved imagined-measurement interface.
At the preferred B+ baseline, adding just reference intent would give 1016 and
adding just geometry distances would give 1017. Exact schemas must identify the
selected subset and its timing.

Reference intent is legitimately known at an imagined state; it needs no prediction
head and need not be differentiable. It could condition a differentiable value
surface. Nevertheless, the current plan uses it as a reward condition and excludes
extra contact-related observations. Default B+ leaves that restriction intact.
If two otherwise identical state/goal inputs with different known intent cause
measurable value aliasing, test a **current-intent-only** Critic under a separate
amendment. Do not add future-contact sequences, inferred phases or invented hand
labels. Actual-contact/force conditioning poses a different, harder problem.

### 7.3 Can the Critic use measured force without a WM Force Head?

**VERIFIED limitation:** measured force is unavailable as a drop-in feature of the
current imagined Critic. That Critic must evaluate states throughout a
counterfactual rollout and provide the Actor's terminal bootstrap.

**PROPOSED alternative / architectural inference:** a separate, carefully causal
**real-data** value interface could use measured force without a WM Force Head.
No such estimator exists in the inspected implementation, and it would not solve
the main imagined Critic's missing-input problem.

The row-t force is produced after applying a_t over `[t,t+1]`. Giving it to V(s_t)
leaks that action's outcome. Giving it to V(s_(t+1)) is causal incoming-transition
information, but subsequent imagined actions do not come with measured forces.
Giving the previous real interval to V(s_t) is also causal at the starting state,
yet still leaves every later imagined input undefined. Freezing or detaching that
measurement does not fix the distribution mismatch. Copying the starting force,
copying a logged future force from another action, or filling future force with
zero is not a valid solution. Zero means a measured physical value, not “unknown.”

| Solution | What is legitimate | Limitations / cost / plan consistency |
|---|---|---|
| **State/reference/context Critic, preferred** | Reconstruct the same features from real or predicted physical 84 | No extra prediction model; lowest integration cost; needs only the explicit HOI policy-interface amendment; cannot directly observe solver loads |
| Geometry-derived information | Pure Torch FK/Box-target distances, relative motion, or explicitly audited geometric proxies | Available and differentiable, but no assertion of true force/contact. Hard geometric thresholds have poor derivatives. Requires observation amendment; collision/SDF models would require additional scope |
| Transition-aligned measured-contact interface | Separate real-data diagnostic value estimator or explicitly defined incoming-transition-conditioned estimator | Causality can be correct, but it cannot replace imagined V_H/TD(lambda) without a second consistent inference path, marginalization/distillation, or changed objective. Additional training algorithm/data/reward scope; not recommended here |
| Additional contact/force prediction model | Predict action-dependent interval measurements under imagined actions, with a stated history/state/action interface | New targets/loss, calibration, uncertainty, counterfactual and gradient validation. A “separate model” still changes research design. Contact BCE/force regression/heads are currently prohibited; explicit Master amendment required |

State 84 may not determine measured loads uniquely: previous solver/constraint
history, contact manifolds, future action, substep trajectories and unmodelled
partner forces matter. Geometry at the reached state alone cannot reconstruct a
four-solve force/impulse history. A force head that merely interpolates correlated
data would also not establish reliable Actor gradients. None is introduced here.

High force is not a value target or success metric. The selected Critic predicts
the existing task return; physical force/contact remains independent evidence for
contact, support, slip, lift and sustained load-bearing evaluation.

## 8. Imagination extension points and minimum implementation changes

### 8.1 Existing interfaces — VERIFIED

| Extension point | Current behavior | Required change for the proposal |
|---|---|---|
| `HOIPolicyReplayBuffer` | Full 84 histories, existing ring sampler, clip/frame/hash envelope | Reuse as-is for physical history; expose named Object observation construction, no second recorder |
| `HOIRolloutContext` | Reference/boundary/validity management | Add policy-observation and Object-goal construction interfaces while preserving masks |
| `project_state` | Joint physical 84→Robot 71 projection for old Robot consumers | Keep it; do not make Robot FK/analytic functions accept 84 |
| `HOIImagStep` | Joint 84 prediction/reward, but analytic output observation 93 is Robot-only | Emit/build joint policy observation through the HOI adapter; pure tensor features/no sensors |
| `step_reference` | Gathers reached f+1 for reward/termination | Preserve; Actor goal and Critic state features use their own current f |
| `record` | `@no_grad`, detached H reached-state diagnostic rows | Keep diagnostics separate; never use them as live Critic features |
| `ImagRollout.states` | Actually `[H+1,N,84]` on HOI path; annotation says 71 | Generalize/document state width, preserve all full states |
| `ImagRollout.goals` | Robot 680/689, H+1 rows | Preserve this meaning; add explicitly named Object-goal field or a versioned structured goal pack |
| `critic_features` | Callback receives only `ro.states`; HOI returns Robot canonical 71 | Add a rollout/context-aware input-builder hook for state/reference/previous-action features |
| `rollout_context_factory` | Creates finite-clip context from sampled start | Reuse to carry registry and reference identity into new feature construction |
| `HOIPolicyTrainer` | Enforces history 10/future 10/latent 32 and old Critic width | Replace only HOI-specific dimension/schema checks after approval |

Sources: [HOI adapters](../../../../src/supermimic/hoi/imagination.py),
[shared rollout](../../../../src/supermimic/imagination/rollout.py),
[Critic input hook](../../../../src/supermimic/imagination/policy_trainer.py#L72).

### 8.2 Minimal proposed data flow

1. Keep the existing state 84 Replay start and two aligned histories. Construct
   raw per-frame Robot 93/Object 15 features from each full state and preceding
   clean action. Flatten by the declared separate Robot/Object layouts. The
   Robot analytic observation still receives 71 only.
2. Add optional observation/noise/flatten/goal hooks to the existing shared
   `imagine` recurrence, with exact old defaults. HOI builds 1080 observation and
   passes the concatenated 830 Actor goal to its wrapper. Do not copy `imagine`
   or create another Actor rollout algorithm.
3. After each joint WM step, construct fresh Object features from predicted
   physical 84, append them to history together with the corresponding Robot
   observation, and retain the computation graph. Do not keep the initial Box
   observation frozen or use reference Box pose as the new observation.
4. Keep `ro.goals` Robot-only and add named Object goals `[H+1,N,150]` (or an
   explicit versioned equivalent). Provide previous-clean-action rows
   `[H+1,N,29]`: initial `start.prev_action`, followed by `ro.actions`. Padded
   rows are not samples; terminal feature uses the last **valid** action.
5. Widen the optional Critic input hook to receive rollout plus start/context,
   returning `(features148, goals830, root_error4)`; xi33 stays the existing
   separate argument. Construct Object errors using the registry at
   `ro.frames[t]` / corresponding local frame for all valid state rows 0..L.
6. Reuse `PolicyTrainer.actor_loss`, `critic_batch`, `update`, frozen-parameter
   context, distributed gradient handling, gradient clips, Adam optimizers,
   and `policy_objective`/`gae_targets` unchanged in their learning semantics.

A state-only `critic_features(states)` hook can implement A but cannot generally
implement reference-aware B+. `HOIRolloutContext.record` contains detached
`box_errors` with H rather than H+1 rows and lacks initial-state features. Reusing
those diagnostics would both misalign time and silently destroy Object→terminal
Critic→Actor gradients. Use a separate differentiable feature builder.

Three easy-to-miss shape hazards must be tested: original history flattening knows
only 93 Robot fields; Decoder scale is length 930; goal scaling assumes 68D Robot
frames. Merely appending tensors or passing larger integer dimensions to current
classes is insufficient and can silently drop or wrongly scale Object data.

The compiled step must receive explicit reference tensors and produce the same
features/gradients as eager execution. Registry/cache gathering stays outside
compiled numerical kernels as currently designed. Object history appends,
heading transforms, support features and Critic errors must not detach. Pure
reference-only goals may be precomputed, but actual-state-anchored goal ablations
must keep their state derivatives. P7.0 compiled timeouts remain incomplete
evidence; historical P5 passes do not validate these proposed extensions.

## 9. Checkpoint migration and strict compatibility — PROPOSED

Retain the original robot-only classes, CLI defaults and checkpoint schema.
Introduce a distinct HOI Actor/Critic schema with named blocks, history/future
lengths, flatten order, scales, coordinate conventions, reference-padding rules,
noise/estimator contract, encoder widths, and action-interface version.
Store separate original/candidate reference and actual Object/support identities;
do not silently promote `training_ready=false` artifacts.

### 9.1 Actor mapping

For matching original hidden widths, copy the Robot Motion Encoder 680→32
unchanged. Copy Decoder first-layer columns by name:

| New B decoder columns | Meaning | Initialization from original |
|---|---|---|
| `[0:32]` | Robot latent 32 | Copy old `[0:32]` |
| `[32:962]` | Robot history 930 | Copy old `[32:962]` |
| `[962:1026]` | Object-history embedding 64 | Zero new columns for exact-function initialization |
| `[1026:1058]` | Object-goal latent 32 | Zero new columns for exact-function initialization |

Copy compatible downstream weights/biases and Robot scaling exactly. Initialize
the separate Object encoders normally; they are initially disconnected by the
zero fusion columns. A uses 15 zero columns after 962; C uses 64. Validate identical
Robot-policy actions under equal inputs/RNG before any learning.

**Important limitation:** exact-preserving zero-column migration initially has
zero Object-to-action derivative and zero gradients into the new Object encoders
until decoder fusion weights become nonzero. Small nonzero fusion weights enable
immediate sensitivity but sacrifice exact action equality. These are different
initialization policies, not simultaneous guarantees. Prefer exact-preserving
migration for a controlled transfer test, with a separate nonzero-fusion fixture
for graph connectivity. A scratch policy can use normally initialized new
columns. Learned Object use must be measured after authorized learning; it is not
established by checkpoint migration alone.

### 9.2 Critic mapping

New input order is `[features148, goals830, xi33, root_error4]`:

| New input slice | Named content | Old Critic mapping |
|---|---|---|
| `[0:71]` | Robot canonical | old `[0:71]` |
| `[71:86]` | Object canonical | new, zero first-layer columns |
| `[86:104]` | Support 18 | new, zero columns |
| `[104:119]` | Object error 15 | new, zero columns |
| `[119:148]` | Previous clean action 29 | new, zero columns |
| `[148:828]` | Robot goal 680 | old `[71:751]` |
| `[828:978]` | Object goal 150 | new, zero columns |
| `[978:1011]` | xi33 | old `[751:784]` |
| `[1011:1015]` | Root error 4 | old `[784:788]` |

Copy matching hidden/output weights and bias. Validate functional equality at
initialization; larger physical/reward scope still invalidates an assertion that
robot-only values are calibrated HOI values. The preferred initial Critic policy
is fresh HOI calibration/learning, with weight transfer an explicit comparison.
Reject wrong widths, old anchor configuration, scale/order/frame/schema mismatch
and unrecognized provenance. Do not use `strict=False` or a contiguous-prefix copy
of 788 columns. Strictly load the complete resulting new-schema state dictionary.

Do not reuse old Actor/Critic optimizer moments, sampler returns, RNG/Replay state
as though this were an HOI resume. New parameter groups and changed return meaning
require explicit initialization. A true new-schema resume must restore all three
optimizers, observations/history, sampler, RNG and Replay/boundary provenance or
declare an episode-boundary/buffer-refill restart. Existing original `Trainer.save`
does not save Replay/RNG/UTD remainder. WM166→82 is unchanged by this proposal;
its existing migration qualifications/readiness gates remain separate.

The external robot checkpoint's prior strict-loading evidence does not prove
training-time provenance or HOI policy quality. No checkpoint was migrated,
rewritten or selected by this review.

## 10. Required implementation file map — PROPOSED ONLY

| File / area | Minimum intended work after approval | Reuse / protected behavior |
|---|---|---|
| New `hoi/policy_features.py` | Named raw Object 15, history flattening, Object goal 150 and differentiable Critic error 15/input packs | Reuse rotations, state layout, support and registry; no sensor import |
| New `hoi/policy.py` | HOI Actor/Object encoders and HOI Critic wrapper with explicit scales/schema | Reuse existing MLP and Robot encoder; no optimizer or TD code |
| `hoi/reward_reference.py` | Same-clip Object future gather / constant goal cache with explicit padding metadata | Reuse derivative cache and registry hashes; no NPZ change |
| `hoi/imagination.py` | New HOI observation/goal/Critic builders, dimension checks and metadata | Reuse context validity, step reference, full joint dynamics and reward |
| `imagination/rollout.py` | Optional observation/history/goal hooks and explicit Object goal/previous-action rollout fields | Same recurrence; defaults preserve Robot-only output and gradients |
| `imagination/policy_trainer.py` | Optional rollout-aware Critic input builder | Same Actor optimizer, Critic optimizer, TD targets and update order |
| `hoi/replay.py`, `data/replay_buffer.py` | No new physical storage required; only a demonstrated narrow metadata need would justify edits | Reuse existing history/index/boundary machinery |
| `hoi/checkpoint.py` or adjacent HOI policy checkpoint module | Named Actor/Critic migration and strict feature schema | No modification of original checkpoints or `sm-train` loading |
| Future HOI environment/component construction | Install the same real/imagination feature builders, aligned history resets and policy Replay adapter | Existing `Trainer.run()` remains the only training mainline |
| `training/loop.py` construction/save/load boundary, only if needed | Optional component and metadata hooks; remove no Robot defaults | No duplicated Trainer, UTD loop or algorithm |
| Future HOI evaluation adapter | Actual Actor features/actions and A–F physical metrics | Independent P6 sensing; never fixed-reference action substitution |
| New `tests/hoi/test_policy_features.py`, `test_hoi_policy_architecture.py` and existing policy/parity/migration suites | Acceptance tests in §13 | No tests or implementation added during this review |

No change is required to WM residual dimensions, physical state, reference arrays,
contact calibration, reward equations or optimizer math. `hoi/state.py`,
`hoi/state_spec.py`, `hoi/world_model.py`, contact modules and existing Robot
network defaults should be reused without semantic change. Source filenames for
new modules are proposals, not claims that they exist.

Training-framework reuse audit: one existing `Trainer.run`, one ring sampler,
one Joint WM fitter, one shared imagination recurrence, one Actor BPTT optimizer
and one Critic TD(lambda) implementation. New code should be feature construction,
network composition, schema/migration and narrow default-preserving hooks only.

## 11. Master Plan consistency and exact proposed amendments

**VERIFIED conflict:** the current Master does not already authorize an Object-aware
Actor or Critic merely because state 84 exists. §12 P7-B.5 explicitly keeps the
Critic Robot-based; P5/P7/§14/§17 freeze observation dimensions. Current
`HOIPolicyTrainer` enforces those restrictions. All A/B/C Actor candidates and all
joint-state Critic candidates therefore require a research-level interface
amendment. User authorization here covers investigation/proposal only.

### 11.1 Change classification

| Proposed change | Already permitted? / interface impact | Conflict and required disposition |
|---|---|---|
| Reconstruct Object history from Replay | Reading/reconstructing existing 84 is permitted; no new physical contract | Feeding it to Actor changes the frozen observation contract; needs M1/M2 below |
| Object Future Goal from registry | Data derivation feasible with unchanged NPZ | New 150D goal/encoder and total goal 830 require M1/M2 |
| Joint/support/Object-error Critic | Features available, not currently permitted Critic inputs | Replaces robot-only Critic and changes terminal gradient; requires M1/M2 |
| Previous clean action in Critic | Existing causal data, no new physical state | New 29D value conditioning; include explicitly in M1/M2 |
| Different Actor/Critic inputs | Original algorithm already supports asymmetry | Specific new HOI blocks/dimensions still require amendment |
| Observation/goal/Critic hook changes | Narrow hooks consistent with reuse principle | Implementation-level only **after** architecture approval; require Robot regression |
| Reference intent in Critic | Technically available, no predictor needed | Extra contact-related observation; separate M3, not preferred |
| Geometry distances in Critic | Analytic geometry exists for reward | Extra geometry/contact-related observation; separate M3, not preferred |
| Measured contact/force in Critic | Explicitly forbidden by §12 P7-B.3 and §1.2/§14 | Separate M4 and a complete imagined-feature solution; not recommended |
| Contact/force prediction model or loss | Explicitly forbidden | Separate M5; not recommended |
| Reward retargeting, human commands, force reward, solver/assets | Outside this architecture proposal | Remains unapproved; no implied permission |

### 11.2 M1 — add an explicit HOI interface contract to §4 (approval pending)

Proposed wording:

> For the separately versioned HOI policy interface, permit Robot observation
> history 10×93 and Object observation history 10×15, with Robot future goal 10×68
> and Object future goal 10×15. Use the reference-local Object frame, scales and
> ordering defined in the approved P7.0.1 review. The Actor has observation 1080,
> goal 830, Robot goal latent 32, Object history embedding 64, Object goal latent 32,
> decoder input 1058 and native action 29. The independent HOI Critic uses joint
> canonical 86, support 18, Object error 15, previous clean action 29, Robot goal 680,
> Object goal 150, xi33 and root error 4, totaling 1015. These are policy/value
> features, not new physical state. Preserve physical 84/canonical 86/residual 82,
> WM166→82, support 18/action 29/xi33, reference data, reward equations and native
> actuation. Measured force/contact and reference contact intent are not inputs
> to this approved Actor/Critic schema. Original robot-only interfaces remain
> unchanged.

Also clarify §4.1's Object canonical velocity wording to “absolute Object
root-origin linear velocity expressed in Robot heading axes; no Robot velocity
subtraction,” matching current `JointStateSpec`. This is a documentation
clarification, not a physical representation change.

### 11.3 M2 — narrowly supersede frozen HOI policy clauses (approval pending)

Apply an explicit exception in §1.3.4 P5/P7 rows, §10's no-added-observation
restriction (current line 1401), §12 P7-A.2/P7-B.5 and P7-B acceptance, §14,
§16 P7 instructions and §17.1 items 3/10. Cross-reference the prospective P7 HOI
exception wherever the linked legacy P5 specification freezes policy interfaces.
P5's dimension-preserving acceptance remains a historical result; do not rewrite
its reports, results or tests to pretend the new interface had already passed.

Proposed replacement for the relevant policy-interface clauses:

> Preserve original robot-only Actor/Critic dimensions, defaults and checkpoint
> behavior. For the separately approved HOI schema, use the P7.0.1 Actor/Critic
> feature contract and named checkpoint migration. Preserve the original
> alternating WM/Actor-Critic training mainline, BPTT objective, frozen-parameter
> input gradients, TD(lambda) recurrence and terminal/timeout/clip masks.
> The HOI Critic may use the approved reconstructible privileged Object,
> reference-error, support and previous-action features. Physical WM contracts
> and the prohibition on measured force/contact inputs remain unchanged.

This is a research-level change under §1.3.3/§15.3 because both policy information
and the terminal value gradient change. Interface widths are not altered merely
to fix numerical calibration. A future implementation specification must record
approval and replace the HOI adapter's old checks with the new schema; it must not
weaken original Robot-only regression or silently reinterpret P5 metadata.

### 11.4 M3–M5 — alternatives not included in the preferred approval

**M3, reference intent or geometry Critic ablation:** explicitly amend §1.2's
no-contact-observation rule, §4.5, P4.6/P5 interface exclusions, §14 and §17 to
permit only the named Critic feature and exact width/timing. For current intent,
wording must state: “one unchanged global c_ref at the Critic state's reference
frame; exogenous known condition; no measured contact, per-hand relabeling,
future-contact lookahead, classifier or WM input.” For geometry, state its exact
distance definition and that it is not actual contact. Default recommendation
does not request these exceptions.

**M4, measured-contact/force value interface:** additionally replace the explicit
§12 P7-B.3 prohibition for a named, causal real-data interface and document how
imagined V_H, TD targets and Actor gradients remain consistent. Specify whether
it is a separate diagnostic estimator, a marginalized value, or another algorithm;
list added optimizers/losses and missing evidence. Merely allowing “privileged
Critic inputs” is not a sufficient amendment. No such implementation is proposed.

**M5, contact/force predictor:** explicitly amend §§1.2/1.3.5/2.2, §4.1/4.5,
P3.5–P3.8, P4 exclusions, §12, §14 and §17. State input/history/action alignment,
output dimensions/units, new Contact BCE or force regression, normalization,
data coverage, counterfactual validation, checkpoint migration and any Actor
gradient effect. If auxiliary outputs are not integrated, preserve physical 84
explicitly; if they are integrated, a further state-contract change is required.
Calling it a separate auxiliary model does not avoid the amendment. This review
recommends neither a Force Head nor a Contact Head.

No amendment text has been applied to the Master. Approval of the architecture
would not authorize training, new assets, reference edits, solver changes, force
supervision or PPO.

## 12. Ablations and unresolved research decisions

All experiments below are **PROPOSED**, not executed. Use matched reference/asset
hashes, physical settings, reward coefficients, WM initialization/readiness,
source-episode splits, random seeds and bounded budgets. Record both compute and
parameter-count differences; add equal-capacity controls where branch removal
changes capacity. Keep Actor and Critic comparisons independent before testing
their interaction. Do not attribute effects to inputs while changing reward.

| Ablation | Research question / required observations |
|---|---|
| Existing Robot-only vs Actor A | Does independent Box feedback affect actions/recovery at all? |
| A vs C | Does Object history improve slip estimation/recovery beyond current pose+twist? |
| C vs B | Does explicit Object future intent improve coordinated lift/placement or reduce adaptivity? |
| Object history lengths 1/5/10, separate declared schemas | Is temporal memory useful under estimated velocity/noise, and how much? |
| B reference-local vs actual-root-anchored Object goals | Relative adaptivity versus absolute trajectory recovery/localization dependence |
| Correct vs temporally shuffled Object history | Does the controller use motion history rather than only duplicated current state? |
| Matched Robot goals, changed valid Object goals | Can the Actor represent distinct Object tasks without Robot-reference changes? |
| Critic 788 vs A821 vs B836 vs B+1015, same Actor | Contribution of Box state, support, directional error, future Object task and previous action |
| B+ minus support / minus error / minus previous action / minus Object goal | Isolate each privileged feature's value error and terminal-gradient effect |
| B+ with current intent 1, only after M3 | Does known intent resolve measured value aliasing at otherwise similar state/goals? |
| B+ with geometry 2, only after M3 | Does explicit geometry improve value beyond joint state/FK learnability? |

Counterfactual input tests must hold Robot observation and reference fixed while
changing Object position, orientation, linear velocity and angular velocity
independently. Also test identical current state with different valid histories,
and common Robot–Object drift. Input-only tests change tensors; they are not
post-reset physical teleportation. Later physical perturbation experiments require
their own approved protocol and must not become hidden assistance.

Report action Jacobians/sensitivity and effective joint commands, Robot/Object
tracking, contact establishment, persistence, support separation, unsupported
lift, slip, sustained carry, recovery time, release/placement and all failures.
Report Critic calibration, TD error by regime, terminal Object-gradient magnitude
and direction, finite differences, value/return scale and WM action-to-Box
prediction quality. High force and lower tracking error alone do not establish
physical carry. Keep the P7 A–F evaluation distinction: tracking; bilateral
contact; unsupported lift; sustained load-bearing carry; disturbance recovery;
release/placement. P6 fixed-PD baselines remain separate.

**UNRESOLVED before implementation/training decisions:**

1. Approval of M1/M2 and exact Object-feature/checkpoint schema. The preferred
   architecture is explicit; its empirical superiority is not established.
2. Whether the task ultimately prioritizes relative partner adaptation or exact
   global path recovery. B defaults to relative coordination; the goal-frame
   ablation must expose the global-drift limitation rather than hide it.
3. Object sensing/ego-motion estimation, causal twist, latency/dropout and
   calibrated noise. Simulator state access is not deployment validation.
4. Adequacy of the 10-frame Object history and compact encoder widths. These are
   defensible initial design choices, not optimized hyperparameters.
5. Residual Critic aliasing from policy history and reward intent. Measure before
   adding history/contact features; no new classifier/head is authorized.
6. Initialization provenance and value scale: scratch versus named weight transfer,
   zero-fusion transient, robot checkpoint limitations and new HOI optimizer state.
7. Existing P7 readiness blockers: actual HOI entry/reset/termination/checkpoint/
   evaluation wiring, candidate-aware provenance, reset-overlap disposition,
   actual-training precision guard and fail-fast monitoring, current compiled
   validation, and meaningful learned WM action-to-Box sensitivity. Object-only
   failure prediction is still absent from the Robot termination head. A wider
   Critic does not resolve these blockers.

Candidate reference `training_ready=false`, P6 failures and original production
assets retain their status. P7-A WM readiness is still required for useful Actor
BPTT through the Joint WM; no arbitrary requirement to make fixed-reference PD
carrying succeed is added.

## 13. Implementation acceptance tests — PROPOSED, NOT RUN HERE

These are acceptance obligations for a separately authorized implementation.
They do not claim that the proposed networks already exist or have passed tests.
Reuse the existing suites where they cover unchanged behavior.

| Test | Acceptance requirement |
|---|---|
| Named dimensions/order | Assert physical 84/canonical 86/WM166→82 unchanged; B observation 1080/goal 830/decoder 1058/action 29; Critic 1015; verify every named slice, not only totals |
| Original Robot regression | Original Policy/Critic/checkpoint outputs, history order, default goal 680/anchor 689, optimizer/TD behavior unchanged under equal inputs/RNG |
| Replay history | Reconstruct Robot/Object histories from the same sampled indices; reset backfill and prior clean action 0; no cross-episode/clip/gap/wrap contamination |
| History content | Different historical Object motions with equal current state produce different history features; correct chronological/group layout; do not require every random policy output to differ |
| Future reference | Original and candidate load under own hashes; f..f+9; multi-clip IDs; explicit goal padding; reject invalid starting/reward frames; no NPZ writes |
| Pose/twist frames | Translation/yaw equivariance with origins/reference/supports transformed consistently; root-versus-COM distinction; Object world linear and body angular conventions |
| Quaternion representation | q and−q identical; r6 layout correct; Critic error 6 zero at agreement and finite gradients through 180-degree Object errors; disclose heading singularity domain |
| Scaling/noise | Robot 930 scales exact; Object fields scaled once; Robot 680/Object 150 goal scales separately correct; structured pose noise, causal estimation, real/imagination convention equality |
| Box-only feedback | Fixed Robot/reference, perturbed Object → changed features and finite nonzero action derivative in a controlled nonzero-fusion fixture; separate zero-migration expectation |
| Object goal conditioning | Fixed observations/Robot goal, changed Object goal → changed B task features and controlled-fixture action; C remains Object-goal-independent by construction |
| Current versus reference Box | Changing actual Box changes features/geometry without changing reference; WM and observation never replace actual Box with reference pose |
| Real/imagination feature parity | Identical physical state, action history, xi and reference yield matching Actor/Critic features; dtype-appropriate declared tolerances and common noise seed |
| Critic Object/error sensitivity | Independent Object deviation or changed Object target changes correct named features; controlled Critic demonstrates both value and input-gradient dependence |
| Support alias test | Same canonical joint state at different fixed-support-relative locations changes support 18/value features; support remains separate from xi33 |
| Terminal Critic gradient | Isolate V_L contribution and verify Object/previous-action derivatives reach Actor through frozen WM/Critic; compare with finite differences and deliberate-detach negative control |
| Freeze/isolation | WM/Critic flags restored after success/exception; Actor loss updates only Actor; detached Critic loss reaches neither Actor nor WM; no shared trainable encoder weights |
| Reference timing | Critic(state_t) uses f_t; reward(state_(t+1)) uses f_t+1; all H+1 feature rows aligned; do not reuse H-row detached diagnostics |
| Boundary/TD recurrence | Natural end/failure zero tail; timeout/horizon bootstrap; coincident boundaries; no invalid reward/Critic rows; existing recurrence exact regression |
| Measurement exclusion | Replace Replay force/contact sidecars with arbitrary finite values: Actor input/action, imagined reward, main Critic features/value/targets and Actor gradients unchanged |
| Checkpoint migration | Named column mappings; exact-function initialization within declared numerical tolerance; explicit zero initial Object sensitivity; strict rejection of wrong shapes/scales/frames/provenance |
| New-schema resume | Roundtrip every feature scale/encoder/optimizer and required history/RNG/sampler state; distinguish exact resume from documented buffer refill |
| Eager/compiled parity | Completed comparisons of features, actions, states, rewards, value, TD targets and Actor gradients; a timeout is incomplete, never PASS |
| Learned use and physical evaluation | After later training authorization, demonstrate Object conditioning and useful WM sensitivity; evaluate actual Actor and independent A–F physical outcomes, keeping failures in denominators |

For algebraic checks, use the existing P5 FP32 tolerance `(atol,rtol)=(5e-6,5e-6)`
and FP64 `(1e-10,1e-8)` where applicable; preserve stricter existing exact-equality
contracts for hashes/indices/labels and declared migration tolerance where
floating-point accumulation order differs. Nonzero sensitivity in a synthetic
fixture demonstrates graph connectivity, not learned physical correctness. Real
learned Jacobian/recovery thresholds require predeclared data/physical criteria,
not arbitrary values chosen after results.

## 14. Audit verification, evidence limits and final decision

**VERIFIED during this task:** source-level inspection of all requested modules,
full Master Plan reading, public/local six-file blob comparison, original/candidate
reference SHA/schema/status checks, and CPU inspection of the preserved physical
Replay sample. No networks of new dimensions were instantiated. No simulator,
model fitting, backward experiment, new checkpoint, PPO, or training was run.

Reference/hash checks reconfirmed:

| Artifact | SHA256 |
|---|---|
| Original Part B reference | `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b` |
| `reference_candidate_v1` | `f38071e5347403f9eff993dd37884bf374e0c003400156dc4cb87d3d2b856cb0` |
| P6 calibrated-contact definition/evidence | `34a47e06454c64f71b456bb428110fceff65302f6c6517dc85a95e151d400266` |

The preserved Replay sample is a bundle with `metadata` and `window` keys. Under
`window`, states are `[9,2,84]`, next states `[8,2,84]`, clean/executed actions
`[8,2,29]`, raw forces `[8,2,4,2,2,3]`, intent `[8,2,2]`, and frame/clip/episode
IDs `[8,2]`. Metadata records the candidate Object separately from support-source
identity. This independently confirms data availability, not policy learning.

Substantive read-only commands included:

```bash
git status --short --branch
git rev-parse HEAD
git -C /home/han/pyx/SuperMimic-Reproduced rev-parse HEAD
git -C /home/han/pyx/SuperMimic-Reproduced status --short
git -C /home/han/pyx/SuperMimic-Reproduced hash-object \
  src/supermimic/policy/nets.py \
  src/supermimic/imagination/rollout.py \
  src/supermimic/imagination/policy_trainer.py \
  src/supermimic/imagination/objective.py \
  src/supermimic/world_model/heads.py \
  src/supermimic/training/loop.py
```

The public GitHub GETs were `commits/main` and
`git/trees/77e4e7ce2e70fc6645cb2e2f40f004ccf4911bc6?recursive=1` for the requested
reproduction. Paper HTML v1 returned 404; the actual 19-page arXiv v2 PDF was
successfully read. Neither a README nor local `PAPER_SPEC` was substituted for
the paper when identifying paper-level requirements.

CPU artifact inspection used
`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B` with
`numpy.load(..., allow_pickle=False)`, manifest JSON reads, SHA256 and
`torch.load(..., map_location='cpu', weights_only=True)`. To reproduce the central
saved-window dimension check without simulation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B - <<'PY'
import torch
p = 'outputs/hoi/contact_supervision/p7_readiness/physical_final/sampled_physical_window.pt'
b = torch.load(p, map_location='cpu', weights_only=True)['window']
for k in ('states', 'next_state', 'a_clean', 'a_exec', 'force_raw_world', 'intent_ref'):
    print(k, tuple(b[k].shape), b[k].dtype)
PY
```

P7.0's 66 component tests and bounded physical smoke are **reused historical
evidence**, not tests rerun here. The existing compiled timeouts remain unresolved.
The proposed implementation acceptance tests in §13 have not been executed.
Documentation checks and baseline preservation are the only completion checks for
this document-only task. **VERIFIED completion checks:** all 3,089 tracked files
match their pre-audit SHA256 inventory; the Git index is untouched and HEAD is
unchanged. All local document links resolve, fenced commands are balanced, and
the whitespace check reports no errors. The only new task file is this review;
the pre-existing untracked `.vscode/` directory remains outside its scope.

**Final design decision for approval — PROPOSED:**

- Actor B, with 10-frame Robot/Object histories, 10-frame Robot/Object goals,
  reference-local Object task encoding, compact separate Object encoders and one
  shared action decoder. Raw 1080/830, decoder 1058, action 29.
- Critic B+, with joint canonical 86, support 18, directional Object error 15,
  previous clean action 29, goals 830, xi33 and root error 4. Total 1015; independent
  weights; original value optimizer and TD(lambda).
- No measured force, actual contact, reference-intent observation, geometry-contact
  observation, Contact/Force Prediction Head, Contact BCE, force regression or
  force-based Actor reward in the preferred interface. Physical measurements stay
  independent diagnostics. Optional Critic intent/geometry branches need their
  own amendment and ablation.
- Explicit HOI-only policy/checkpoint schemas and M1/M2 amendments, preserving
  physical 84/canonical 86/WM166→82 and original `sm-train` behavior.

**Review acceptance: PASS as a code-grounded architecture investigation and
concrete recommendation. Empirical architecture superiority: UNRESOLVED.
Implementation/training readiness: not established by this review.** Stop here
for the user's architecture decision; do not implement, amend the Master, train,
commit or push as a consequence of completing this document.
