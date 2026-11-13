# P7.0 — RL training readiness and bounded pilot preparation

**Decision: PARTIAL. P7.1 training is NOT technically ready and is NOT authorized
by this audit.** The reusable physical and learning components support further
integration, but the checkout is not an executable robot–object learning system.
No optimizer update, long training, production-model change, default solver change,
commit, or push was performed.

Native PD reference replay and independent PD carrying remain **P6 open problems,
not prerequisites for testing learned-controller feasibility**. The present no-go
comes from missing training integration, missing Box feedback, incomplete lifecycle
support and unverified guard activation in a training process. It does not require
making a fixed reference succeed under native PD.

## Authority, checkout and scope

- Repository: `Ir1222/SMHOI`; local baseline/HEAD
  `a6b876f392f69709d9835eb0e4d664bfcfe75b8d`, branch `main`.
- Audit date: 2026-11-13, Asia/Shanghai. Pre-existing `.vscode/` is preserved.
- Current [Master Plan](../SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md): 3,152 lines,
  SHA256 `68142af38edb0a9a443aac5815491aaae90dba696acc84d30fc54223c6e4a9cc`.
  The complete plan was read; it was not rewritten. No standalone P7 implementation
  specification or HOI training executable existed at this baseline.
- Applicable implementation contracts: Master §§4, 12, 14–17; P0 architecture,
  dependency/contact, dataset and file-change audits; P2 recorder/window contract;
  P3 architecture-cleanup specification/report; P5 implementation specification,
  particularly §§9–11; P6 closure specification, candidate selection and C2/C4 reports.
  Historical force-head/file-map proposals are superseded by the current 166/82
  architecture and were not restored.
- [P7.0 implementation specification](IMPLEMENTATION_SPEC.md) records this audit's
  scope and predeclared checks. No applicable `AGENTS.md` was found.

The request explicitly asks for a **PPO proposal**, whereas Master §12 describes
alternating WM/BPTT and prohibits replacing it with an independent PPO loop.
This report therefore prepares a separately identified physical PPO feasibility
experiment using reusable components and the installed PPO implementation. It
does not rename BPTT as PPO, implement another algorithm loop, or change the Master
Plan. The mainline P7-A WM-readiness and P7-B alternating-training obligations remain
separate. Neither PD success nor WM predictive convergence is a prerequisite for a
separate model-free PPO feasibility test; adequate learned WM readiness remains
necessary before claiming useful BPTT-through-WM learning.

## Experimental inputs and preservation

Evidence root **E** below means
`outputs/hoi/contact_supervision/p7_readiness/`.

| Input | Exact identity / disposition |
|---|---|
| Original Part B `outputs/hoi/asset_audit/part_b/reference/reference.npz` | SHA256 `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b` |
| `outputs/hoi/contact_supervision/p6_closure/reference_candidate_v1/reference.npz` | SHA256 `f38071e5347403f9eff993dd37884bf374e0c003400156dc4cb87d3d2b856cb0`; unchanged `training_ready=false`, `P6_CLOSURE_CANDIDATE_NOT_PRODUCTION` |
| `outputs/hoi/contact_supervision/p6_closure/candidates/single_body.json` | SHA256 `6c3f1a9c601336cb264cd0f979ec4d039ed380e76df6ee1a0a33358d5a82e331`; `production_selected=false` |
| Production M6.5 asset manifest | SHA256 `1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`; unchanged |
| P6 `contact_validation/calibration.json` | SHA256 `34a47e06454c64f71b456bb428110fceff65302f6c6517dc85a95e151d400266` |
| Installed `mujoco_warp/_src/solver.py` | SHA256 `1bb26d51451a065b9e183acacfbd5e59d85df1eb46f17951aa3d1727db6853ff` |
| Opt-in precision fragment | SHA256 `41c06e49c60862d0d40e69c74cae8c0f2310d300516b096b5464607a4efe6dcd` |

The selected single-body, single-collider Box is 2 kg, 0.18 × 0.34 × 0.27 m;
COM `(0,0,0.125)` m relative to the original bottom-plate root. Its root-axis
inertia at COM is diagonal `(0.031416667,0.01755,0.024666667)` kg m². The manifest
stores the equivalent permuted principal moments with principal-axis quaternion
`(0.5,0.5,0.5,0.5)`; those numbers must not be read as root-axis moments directly.
It remains the P6 solid approximation with handles omitted and cavity filled.

Baseline Box contact parameters remain friction `(1,.005,.0001)`, condim 3,
solref `(.02,1)`, solimp `(.9,.95,.001,.5,2)`, margin/gap zero, priority 1.
Native robot/hand geometry, actuator gains, supports, dt=.005, decimation=4,
solver iterations=10 and line-search limit=20 were retained. No installed source
or global solver default was edited.

Both references load through `ReferenceClip` with archive and per-array validation.
Original binary intent remains byte-identical: 840 frames at 50 Hz, ON frames
260–549 inclusive, 290 ON/550 OFF. Derived reference twists exist only in memory.
Candidate robot-arm corrections retain their P6 provenance; no NPZ was regenerated.

The experimental constructor reuses P6 `closure_common.gpu_env("single_body")`.
It explicitly substitutes the candidate object **in memory**, retaining the
documented exception for the missing historical V3 planning file. The strict
production loader still fails on that missing file. This exception is visible in
`static_audit.json` and is not a production provenance approval.

## What exists and what is incomplete

Paths in this table are relative to `src/supermimic/` unless stated otherwise.

| Component | Verified implementation | Remaining work / consequence |
|---|---|---|
| Entrypoint and scheduler | `scripts/train.py`, `config.py`, `training/loop.py:Trainer`; collect → WM update → BPTT Actor/TD(lambda) Critic | Only `sm-train` is registered. Constructor hard-codes robot-only environment/library/WM/replay. No `smHOI-track`, PPO config or component factory. Supplying the candidate NPZ to `sm-train` does not create a Box. |
| Physical scene/reset | `hoi/scene.py`, `state.py`: native actuators, Box, fixed supports, independent sensors; named state84 I/O; synchronized reference pose/twist reset | Physical scene has no policy observation groups, rewards or task command. P7 must bind reference sampling, observations, termination, reward and recorder lifecycle. Sampled reset overlaps remain qualified below. |
| Reference sampling | Robot `MultiClipMotionCommand` supports return/uniform/start modes; HOI `ReferenceResetCache` and `HOIReferenceRegistry` validate frames/hashes | Existing robot reset writes robot state only. P2 `begin_episode` requires explicit binding after every reset; installing the recorder rejects command/event-driven scenes. A training reset adapter is missing. |
| Actor/Critic | `policy/nets.py`: Actor history930, robot goal680 (689 with anchor), latent32, action29; Critic788 (797 with anchor) | No direct current Box observation. Policy is deterministic, without PPO distribution/log-probability API. HOI adapter explicitly enforces old dimensions. |
| Joint state/observation | Physical84 and canonical86 exist; `hoi/imagination.py` projects robot71 for analytic observation and Critic | Joint WM state is not a joint Actor observation. A Box-only perturbation has exactly zero effect on the current Actor. Required Box feedback needs an explicit versioned extension. |
| Contact/intent | `hoi/contact.py`, `contact_validation.py`: filtered native L/R sensors, four solved samples and calibrated actual-contact decisions | Keep intent as desired label, measured contact as diagnostics, and geometry as proximity. P2 occupancy at .001 N is a diagnostic field, not the calibrated contact criterion. |
| Replay | `hoi/replay.py` extends original ring; transition-owned targets/reasons; eight-step windows; `HOIPolicyReplayBuffer` supplies history and registry metadata | P2 installer creates base HOI replay, not the policy adapter. Integrated construction/provenance is absent. Replay reward is a zero placeholder; it cannot be fed to PPO as task reward. PPO needs its own on-policy storage. |
| Joint WM | `hoi/world_model.py`, `state_spec.py`, `support_context.py`, `wm_trainer.py`; shared DynamicsMLP; original autoregressive fitter | Input166/output82, physical84/canonical86/support18/action29/xi33 pass. No force/contact head. Initialized dynamics have zero action-to-Box Jacobian; learned predictive readiness remains unestablished. |
| Imagination | `HOIImagStep`, `HOIRolloutContext`, `HOIPolicyTrainer`; shared rollout, frozen-parameter BPTT, TD(lambda), compiler hooks | Tested component adapters, not a training executable. Object-failure prediction is absent: termination head still sees robot height/tilt only. |
| Reward | `hoi/tracking_reward.py`, `rewards.py`, `reward_reference.py`: six positive robot terms, four Box terms, two original regularizers, reference-gated proximity | Physical training must evaluate captured pre-reset **reached** state against t+1 and apply dt once. Existing robot reward-manager timing is not interchangeable. Final reward/return scales are uncalibrated. |
| Termination | P2 distinguishes failure/timeout/clip-end; P5 masks implement zero clip/failure tail and timeout bootstrap | P1 scene has no physical-failure rule or timeout; P2 adds clip-end only. Robot command-dependent termination cannot simply be attached to it. Joint failure causes and terminal observations need wiring. |
| Checkpoints | Strict named WM migration and weights-only joint checkpoint; robot Trainer saves models/optimizers/sampler | No integrated HOI lifecycle. Original save omits Replay/RNG/UTD remainder; resume refills replay. Candidate Box identity needs distinct schema handling from support-manifest hash. Empty optimizer state roundtrip is not proof of trained resume. |
| Evaluation | `eval/benchmark.py` executes actual Policy but builds robot-only scene | No learned HOI Actor evaluation pipeline for support, lift, carry, disturbances or release. P6 PD scripts are diagnostics and cannot substitute. |
| Solver | Explicit hash-pinned guard, P1/P2 capacity checks | Guard works in the combined physical/Actor preflight. Default Trainer never enters it and uses `QuietSimulationCfg(warn_overflow=0)` without equivalent fail-fast monitoring. Training activation is **not verified**. |
| PPO reuse | Installed `rsl_rl.algorithms.PPO` 5.5.1 | Its stochastic `MLPModel`/TensorDict/distribution/storage interfaces are incompatible with current Policy/Critic without adapters. No PPO training was run. |

**Architecture assessment:** reusable components can support an implementation of
the requested experiment. The checkout **cannot support it as wired**. Robot
proprioception does provide closed-loop feedback, so the current Actor is not a
fixed trajectory player. However, it cannot respond directly to independent Box
motion at fixed robot observation and reference. That limitation must be addressed
or explicitly scoped as an observation-limited baseline; it cannot be hidden by
calling state84 an Actor observation.

## Executed smoke evidence

Final physical evidence: `E/physical_final/{result.json,transitions.jsonl,
substeps.json,guard_launches.json,sampled_physical_window.pt,smoke_weights.pt,source.py}`.
The actual `Policy` class has random initialization and test-only 64/64 trunks.
It is not a trained HOI policy. No action is replaced with a reference joint
position; reference poses are used only at reset and in goals/rewards.
The harness seeds Torch700 before construction; the reused physical-scene config
retains seed0. The exact construction/reset order is retained in each run's source
snapshot. This smoke configuration is distinct from the proposed pilot seed700.

| Requested check | Result and precise scope |
|---|---|
| 1. Reset and initial validity | **PARTIAL:** original and candidate synchronized state84 reset roundtrips pass atol=1e-6; two worlds retain local state/support separation. Forward-only inspection finds initial overlaps: frame0 foot–floor **12.236 mm**, candidate258 foot–floor **3.089 mm**, candidate548 **4.329 mm**, candidate827/838 **4.876 mm**. Original258 also has left wrist–Box overlap **9.160 mm**. Finite/usable reset is not a penetration-free validity pass. No pose was shifted to hide overlap. |
| 2. Original/candidate loading | **PASS:** both exact hashes, minimal schemas and immutable binary labels verified. Candidate status stays false/not-production. |
| 3. Actor/Critic/checkpoint compatibility | **PASS for existing dimensions; incomplete for proposed Box extension.** External robot-only checkpoint strictly loads Actor/Critic; synthetic observation perturbation changes its action by max .335151. This is compatibility only; no physical learned-policy evaluation or adoption. |
| 4. Robot–object observation consistency | **State/WM PASS; Actor requirement NOT SATISFIED.** Robot history930/goal680/Critic788/action29 confirmed. A .05 rad robot-joint perturbation changes random-Actor action by max 1.0716e-5; a .05 m Box-x perturbation changes it by **0**. |
| 5. Physical contacts/calibration | **PASS bounded reuse/integration.** Original P6 candidate calibration is reused, not refitted to intent. L/R thresholds .0743401046875/.0737510181777 N; strict occupancy>.25, i.e. at least two of four solves. Raw force/found/Box quaternion samples are retained. No-contact and isolated-side validity rely on the preserved P6 calibration, not on this policy's performance. |
| 6. Intent alignment | **PASS:** transition stores `[c_t,c_(t+1)]`; physical reward uses t+1. Segments cross 259→260, 549→550 and 838→839 without shifting/clamping valid rewards. Zero proximity outside intent is covered by component tests. |
| 7. Replay insertion/sampling | **PASS:** 50 world transitions, six explicitly reset source episodes, eight-step sampled states `[9,2,84]`; contiguous frame/episode/clip validation, terminal pre-reset capture; three natural clip-end transitions. |
| 8. WM shapes | **PASS:** canonical86 + support18 + action29 + xi33 =166; residual82 integrates physical84. Physical supervision only. |
| 9. Loss/backward | **PASS for finite software paths:** WM loss 200.790634, gradient norm162.169388; Actor loss−.0628953, gradient norm5.48412e-7; Critic MSE .0730644, gradient norm .695036. WM gradients remain absent during frozen-WM Actor backward. Initialized action-to-Box derivative is **zero**; these numbers do not establish useful learned dynamics or value learning. |
| 10. Save/load | **PASS weights only:** strict WM/Actor/Critic tensor roundtrip, all weights unchanged, zero optimizer updates. **Training resume NOT TESTED:** no integrated HOI checkpoint/optimizer/RNG/collector resume exists. |
| 11. Actual training-process guard | **NOT SATISFIED.** Installed hash matches and guarded kernel is captured/executed in the combined Actor/physical preflight. There is no HOI training process to validate; unmodified `sm-train` does not enable it. |
| 12. Overflow/NaN/Inf | **PASS in bounded preflight:** 100 substep callbacks ×2 worlds, all overflow bits zero; qpos/qvel/qacc/control/wrenches, contacts, rewards and gradients finite. Applied body/generalized wrenches zero. Action affine map checked each substep. On failure the harness raises immediately and saves state; warnings stay enabled. Production-training monitoring remains missing. |

The numerical loss/gradient values refer to the P5 **test-only** reward coefficients
and small initialized networks. All optimizer counts are **WM=0, Actor=0, Critic=0**.
They are not learning curves, fitted-model validation, or trained checkpoints.

Physical smoke trajectory summary (50 intervals): normalized action max magnitude
.00133615; within-episode action-change RMS .000111620; joint step-change RMS
.0444693 rad; max Box step displacement .0117185 m. L/R contact counts are 4/2,
any-hand 4, bilateral 2; reached-reference intent counts are 12 ON/38 OFF. Physical
reward range .105968–.157722, mean .135102. These include reference-initialized
states, gravity and initial settling. They do **not** establish contact learned by
the Actor, causal hand-driven Box motion, lift, or carry. Later-frame resets cannot
be counted as achieved lift/carry. Raw per-step records retain failures/conditions.

The 66-test component suite passed with one existing Torch deprecation warning.
It covers named migration, loss/backward, replay, reference alignment, boundary
semantics, original robot behavior and guard hash rejection/restoration. Synthetic
P5 coupled residual tests establish graph connectivity only. Existing P5 four-case
CPU/CUDA FP32/FP64 parity evidence remains historical evidence, unchanged.

**Current compiled validation is INCOMPLETE:** the broad CUDA FP32 parity case
hit its 300 s cap and the narrower one-signature step/gradient probe hit its 180 s
cap, both exit124 without a completed comparison. No backend fallback or increased
unbounded run was accepted. These are timeouts, not demonstrated numerical parity
failures. The preserved P5 evidence does not convert either rerun into PASS.
The initial PPO proposal uses eager execution; a later compiled BPTT launch needs
its own completed validation. Exact outcomes are in `E/execution_status.json`.

### Precision guard: evidence and limits

Environment: `.venv/bin/python` (Python3.12), Torch2.14.1+cu130, mjlab1.6.0,
MuJoCo3.16.0, MuJoCo-Warp3.16.0, Warp1.18.0, NumPy2.5.4; RTX4090D.
Installed mjlab `direct_url.json` records commit
`8bb2bc9e187797689ec82c78c35019afd2fd0376`, matching the repository pin.

The guard context encloses scene construction, graph capture/recapture, every
physical step, collection and loss checks. Launch instrumentation observes
`_smhoi_p6_linesearch_precision_kernel__locals__kernel` four times during capture,
alongside the unchanged fused-Jv helper. The captured graphs are launched 140 times
across setup/reset/stepping. These counts are graph-construction/launch evidence,
not the number of times the precision stopping condition fires. `warn_overflow=1`;
source files remain unchanged and the factory is restored on exit. Full physical
configuration and model object parameters are persisted in `result.json`.

The **actual existing Trainer** constructs only `robot`, with overflow printf
disabled and no guard reference/configuration. This was checked from installed
entrypoints, runtime config construction and actual Trainer source
(`E/static_audit.json`). Running that path cannot prove guarded HOI training and
would violate this task's monitoring requirement. It was not stepped. Before
P7.1, the real new entry must retain the same process-lifetime guard, reject an
unknown hash, attest the guarded captured kernel in that process and abort before
Replay insertion/optimizer stepping after any invalid solve. A successful solver
test or this component preflight is not sufficient evidence for that gate.

### Checkpoints, data and provenance limitations

`logs/hdmi_robot_only/ckpt_0012000.pt` remains absent in this repository. The external
checkpoint at `/home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt`
has SHA256 `16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7`.
Its Actor/Critic strict loading passed, but training Git/data attestation is missing.
It was not copied or silently selected as initialization. The P3 recorded FP32
multi-step WM migration tolerance failure remains unresolved; current small-network
migration tests do not erase it. Scratch initialization avoids relying on that
checkpoint but does not solve observation/training integration.

Existing P2/P3 data include controlled fixtures; the P3 report identifies 320
transitions/32 fixture episodes and historical auxiliary-model fitting. They are
not a validated final166/82 natural-HOI training corpus. This audit adds only
short untrained-Actor episodes and does not establish locomotion/lift/carry/release
coverage or held-out learned prediction. For WM experiments, split complete source
episodes before windows and report regime counts, initialization source, invalid
intervals and held-out robot/Box errors against initialized and constant-velocity
baselines. Do not require successful PD carrying to collect valid failed-policy
transitions; do not relabel fixture/teleport/external-force data as natural policy data.

P2 recorder metadata names the original asset manifest, while P6 substitutes the
candidate object. This harness retains that original **support/source** identity
and separately records candidate object hashes in metadata/provenance. The current
strict WM schema does not enforce that extra object field. A future full checkpoint
must validate support identity, actual Box profile, material parameters and reference
identity separately; saving the original manifest hash alone would misidentify
the trained physics.

## Proposed bounded PPO pilot

Machine-readable proposal: [ppo_pilot_v1.json](ppo_pilot_v1.json).
It is deliberately `PREPARED_NOT_LAUNCHABLE`, with launch/training authorization false.
There is no valid `smHOI-track --ppo` command in the checkout; no placeholder command
is presented as executable.

| Setting | Proposal |
|---|---|
| Purpose | Test exploration, effective action variation, reward/value updates and numerical stability; no full carrying optimization |
| Physics/assets | Pinned candidate/reference above; guard opt-in; unchanged native action map/contact parameters; no assistance, post-reset pose writes or external Box wrench |
| Budget | 4 worlds ×32 steps ×10 iterations =**1,280 training transitions**; 2 epochs ×2 minibatches/iteration =**40 optimizer minibatch updates max**; hard wall bound **600 s**, no automatic extension |
| Evaluation allowance | 2 deterministic 64-step episodes before and after =256 transitions; total physical cap **1,536**; same starts/seeds/assets; no P8/full-task claim |
| Initialization | Scratch Actor/Critic, fresh optimizers/sampler; configurable existing encoder/decoder topology with `(256,128)` hidden widths, latent32; explicitly new checkpoint schema |
| Observations | Proposed robot930 prefix plus current object15 in robot heading frame →945 Actor observation; goal680 unchanged. Critic proposal canonical86+goal680+xi33+error4=803. No actual contact/force input. This extension is **not implemented**; it changes the current dimension contract and must be reviewed as part of P7.1 integration. |
| PPO adapter | Reuse installed `rsl_rl.algorithms.PPO`; wrap existing encoder/decoder and Critic behind its model API. Diagonal Gaussian Actor, initial normalized std .1, learned log_std, exact sampled-action log probabilities. Env execution noise0, observation noise off, domain randomization off. |
| PPO parameters | Adam lr1e-4, fixed schedule, clip .2, gamma .99, lambda .95, entropy coefficient .001, value coefficient1, gradient norm cap1, FP32, no AMP; stop on approximate KL>.03 |
| Reset | Candidate starts `[0,240,258,548]` balanced, **conditional on reset-overlap disposition**; episode timeout64 steps. Restore both bodies once at reset, then bind metadata/history. Seeded contact/airborne states are labeled and excluded from achieved-contact/lift claims. |
| Rewards | Same robot+Box+intent-gated proximity equation, reached-state/t+1 alignment, dt once. Proposed provisional recipe: Box sigmas `(.2,.4,1,2)`, weights `(.7,.8,.9,1.1)`, proximity sigma .0423, beta .6. These are the exercised P5 numerical recipe, **not calibrated final defaults**. Review candidate component/gradient distributions before launch. |
| Storage/WM | PPO on-policy rollout storage for advantages/log probabilities; HOI replay retained for diagnostics/future dynamics data. WM optimizer updates0 in this separate PPO experiment; no claim of P7-A readiness or BPTT training. Preserve the alternating mainline separately. |
| Checkpoint | New, non-overwriting snapshots at iterations0/5/10; save Actor/Critic/distribution/optimizer/RNG/config/normalizers/identities/guard attestation/reset state. Verify strict reload and declare exact resume versus episode-boundary restart. |

The proposed Box15 branch can preserve the existing robot/latent ordering and use
named zero-initialized new input columns when migrating a compatible-width network.
That requires new scale buffers, replay/physical and imagined observation adapters,
updated dimension validation and named Critic mapping. Strict-loading old weights
into the wider network is not migration. Small pilot trunks are also incompatible
with the external checkpoint's larger widths; the proposed scratch run avoids
that ambiguity. No network/source change was made during P7.0.

PPO truncation needs explicit attention: installed `process_env_step` uses its
stored transition value for generic timeout correction. P7 must provide the
correct reached **terminal observation/value** and distinguish natural clip-end
from time limit; it must not bootstrap a finite clip from an automatically reset
observation or apply timeout correction twice. P5's tested boundary contract is
the semantic reference, not an automatically compatible PPO adapter.

Expected logs per rollout/update: sampled-action and mean/std/entropy distributions;
native scaled-target changes and saturation; robot root/joint/velocity errors;
Box pose/twist errors and displacement; independently measured L/R/OR/AND contacts,
occupancy, newly established contacts versus inherited contacts, longest continuous
contact and losses; support versus non-hand-body contacts; all 13 additive reward
terms, raw total returns, advantages and value targets; value MSE/explained variance,
PPO losses/ratios/clip fraction/KL; gradient norms and actual parameter deltas;
episode lengths/end causes, invalid-state/overflow counts, solver/penetration
diagnostics, throughput/memory and exact iteration/transition/update budgets.

Acceptance is finite, correctly aligned observations/actions/rewards; nontrivial
sampled **and physically applied** action variation; nonzero finite Actor/Critic
gradients and parameter deltas; valid checkpoint reload; and no ignored invalid
physics. A ten-iteration run need not improve returns monotonically, establish
contact, or carry. Zero contact is a reported exploration outcome. High reward,
high force or a tiny value loss alone is not physical success. Abort on nonfinite
states/losses/gradients, any solver overflow, metadata/guard mismatch, budget breach,
or the predeclared KL stop; retain the failed episode and pre-failure snapshot.

## Evaluation protocol: six separate claims

Every evaluation identifies the actual saved Actor hash, stochastic/mean mode,
reference/asset/solver hashes and seed. Use deterministic mean actions for paired
checkpoint comparisons and separately report stochastic exploration. Native-PD
baselines remain under `p6_closure/`; never pool them with Actor results. Report
all attempted episodes, failures and early terminations in denominators. Undefined
metrics are null with sample counts, not successful zeros.

| Claim | Required independent measurement and interpretation |
|---|---|
| **A. Reference tracking** | Robot root/body/joint/velocity errors and Box position/orientation/twist errors at exact reached frame; survival/end causes. Tracking-only improvements establish A only. |
| **B. Bilateral contact establishment** | Independently calibrated L/R contact, OR and AND; first newly established bilateral interval, duration/persistence and loss. Report intent ON/OFF confusion matrices, precision/recall/F1/balanced accuracy and OFF unintended contact. P4's bilateral task requirement covers ON260–549; it does not create two annotated intent labels. Seeded contact is separated from acquisition. |
| **C. Unsupported Box lift** | Start with measured support; establish loss of support/ground and other-body load paths, geometric clearance, vertical rise and duration, without external wrench/teleport. Retain P6 diagnostic threshold **≥3 cm rise for≥.2 s**, but identify airborne ballistic episodes separately; contact-assisted sustained support is required for a manipulation claim. A reset into the air or gravity-driven motion does not count. |
| **D. Sustained load-bearing carry** | Require C, continuous hand contact, no support/ground or unintended body support, transport and bounded slip/rotation. Keep P6's **≥3 cm lateral travel and≥.2 s bilateral unsupported interval** as a short diagnostic only. A sustained-task claim additionally needs a preregistered longer carry interval/path, load/acceleration consistency, slip limits and fall rejection; no such threshold/pass is established in P7.0. |
| **E. Carry under external disturbances** | Deferred independent experiment after D. Predeclare disturbance source/direction/magnitude/duration/timing, paired no-disturbance control and recovery horizon; measure maintained support, slip, tracking and fall rate. No perturbation is applied in this smoke or proposed pilot. Any future physical disturbance protocol must be separately authorized; it must not be disguised as root assistance or Box actuation. |
| **F. Release and placement** | Contact OFF plus independently verified intended-support contact, target placement error, hand detachment, low object twist and displacement over a declared post-release settling interval. Ground drop/fall is failure. Evaluate frame0 through839 and an additional settling interval for full-task claims; frame839 alone proves nothing. |

Contact force is used to define independently calibrated contact and to diagnose
load paths/impulses. **Higher contact force is never a success score.** Report
signed distances/penetration depth and duration, solver errors and non-hand
contacts alongside each claim. A ballistic throw cannot pass D merely by moving
the Box, and accurate tracking cannot establish C–F.

## Exact commands and outcomes

Run from `/home/han/pyx/SMHOI`. These commands use local `.venv` and do not install
or upgrade dependencies. The shell sandbox could not start (`bwrap` network-namespace
setup denied), so the read/test commands used approved tool escalation. No
automatic-review rejection occurred. The existing reference viewer process was
left alone.

```bash
# Actual default entry/config and pinned external Actor/Critic compatibility only.
.venv/bin/python outputs/hoi/contact_supervision/p7_readiness/static_audit.py
.venv/bin/python -m supermimic.scripts.train --help

# 66 passed; existing deprecation warning retained.
P5_EVIDENCE_DIR=outputs/hoi/contact_supervision/p7_readiness/component_evidence \
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/hoi/test_joint_architecture.py tests/hoi/test_checkpoint_migration.py \
  tests/hoi/test_force_replay.py tests/hoi/test_force_aggregation.py \
  tests/hoi/test_reward_reference.py tests/hoi/test_policy_adapter.py \
  tests/hoi/test_imagination_boundaries.py tests/hoi/test_robot_only_regression.py \
  tests/hoi/test_wm_reuse.py tests/hoi/test_solver_precision.py \
  --junitxml=outputs/hoi/contact_supervision/p7_readiness/components.xml

# Final physical/Actor/replay/backward/weight-roundtrip run; output must be NEW.
OMP_NUM_THREADS=1 timeout 240 .venv/bin/python scripts/hoi/p7_readiness.py \
  --output outputs/hoi/contact_supervision/p7_readiness/physical_final

# Offline reset overlap and trajectory analysis; no physical stepping.
.venv/bin/python outputs/hoi/contact_supervision/p7_readiness/analyze.py

# Broad CUDA FP32 parity attempt: hit 300 s cap (exit124), not PASS.
P5_EVIDENCE_DIR=outputs/hoi/contact_supervision/p7_readiness/compiled_evidence \
TORCHINDUCTOR_CACHE_DIR=/tmp/smhoi_p7_inductor OMP_NUM_THREADS=1 \
timeout 300 .venv/bin/python -m pytest -q \
  'tests/hoi/test_imagination_parity.py::test_inductor_parity[dtype0-cuda]' \
  --junitxml=outputs/hoi/contact_supervision/p7_readiness/compiled_cuda.xml

# Narrow one-signature probe also hit its180 s cap (exit124), not PASS.
TORCHINDUCTOR_CACHE_DIR=/tmp/smhoi_p7_inductor OMP_NUM_THREADS=1 \
timeout 180 .venv/bin/python outputs/hoi/contact_supervision/p7_readiness/compiled_step.py

.venv/bin/ruff check scripts/hoi/p7_readiness.py
.venv/bin/python -m py_compile scripts/hoi/p7_readiness.py
.venv/bin/python outputs/hoi/contact_supervision/p7_readiness/final_verify.py
git diff --check
```

Exact stdout/stderr is retained in E's matching `.log` files. To rerun physical
smoke, choose a new output directory; the script refuses overwrite. Initial
`physical_01` stopped after one vector step because the harness used a nonexistent
public action property; it was corrected to the installed `_processed_actions`
API. `physical_02` completed physical/backward checks but failed guard attestation
because the observer watched `wp.launch` without `wp.launch_tiled`; no activation
claim was accepted from that run. `physical_03` passed after the observer correction.
`physical_final` adds raw contact/replay retention and substep action-map checks.
All earlier failures and sources remain. The first compiled `-k 'float32 and cuda'`
selection deselected all four tests (exit5); the collected exact node above was
then used. None of these outcomes is concealed or relabeled as physical success.

## Remaining gates and P7.1 disposition

1. Integrate a candidate-aware HOI executable and reset/reference/observation/
   reward/termination/replay lifecycle. Preserve `sm-train` behavior and the native
   action interface; the new entry must make Actor-driven physical control explicit.
2. Resolve the measured Box-feedback gap with an explicit observation/checkpoint
   contract, or label the old contract as an observation-limited baseline. The
   required joint-feedback experiment cannot silently keep the current aliasing.
3. For the requested PPO path, implement/test stochastic model and on-policy
   storage adapters, correct timeout/clip/failure bootstrap and bounded logging.
4. Attest the guard and fail-fast overflow/NaN/Inf checks **inside that real training
   entry**, before any optimizer update and after graph capture/resume. Current
   component evidence is necessary but insufficient.
5. Validate candidate-aware checkpoints, trained optimizer/normalizer/RNG handling,
   deterministic Actor evaluation and bounded resume. Disposition of external
   checkpoint provenance/WM migration remains explicit; scratch is the proposed default.
6. Review reset overlap limits and the provisional reward/return scales for this
   experimental profile. Do not reinterpret `training_ready=false` or repair
   reference/production assets silently. Full natural carry is not required here.
7. For **WM/BPTT** only, establish source-episode coverage, held-out prediction and
   useful action-to-Box sensitivity before Actor updates through that WM. This is
   not imposed on the separate PPO pilot.

**P7.1 launch decision: NO.** P7.0 provides a reviewable bounded proposal and
component evidence, not authorization to execute it. Closing the listed software
and runtime gates permits a new bounded training decision without first resolving
the P6 fixed-PD carrying failures. No P7-A warm-up, P7-B, PPO updates or P8 began.

## Changed files and preservation

New files: this report, `IMPLEMENTATION_SPEC.md`, `ppo_pilot_v1.json`,
`scripts/hoi/p7_readiness.py`, and E's audit scripts, logs, JSON/XML records,
source snapshots and smoke-only weights/window artifacts. E's final inventory
enumerates generated files with hashes. No tracked baseline file was modified;
all historical reports, P6 failure evidence, accepted/candidate NPZs, production
assets, original checkpoints and installed solver source remain unchanged.
Preservation verification compares 3,031 pre-existing paths and records the result
in `E/preservation_after.json`. The Git index remains untouched; no commit/push.
