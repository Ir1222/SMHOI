# P0 architecture and tensor-contract audit

The current target implementation is robot-only. Every source file explicitly listed in P0.3 was read, together with configuration, action/context code, current HOI loaders/viewer construction, and dependency callback implementations. No training source file changed. The force-aware master plan takes precedence over V3's older prohibition on a Force Head; no phase classifier or six-phase branch is introduced.

## Verified dimensions

| Quantity | Current code | Force-aware master target | Finding |
|---|---:|---:|---|
| Robot physical state | 71 | 71 | matches |
| Object physical state | absent | 13 | new, explicitly indexed |
| Joint physical state | absent | 84 | not native global qpos/qvel concatenation |
| Robot canonical features | 71 | 71 | preserve order/scales |
| Object canonical features | absent | 15 | pending defined frame implementation |
| Joint canonical features | absent | 86 | target only |
| Action | 29 | 29 | preserve output/action affine map |
| Episode domain context | 33 | 33 | friction 1 + encoder bias 29 + torso COM offset 3 |
| Robot dynamics residual | 70 | 70 | preserve order/scales/integration |
| Object residual | absent | 12 | separate integration |
| Joint physical residual | absent | 82 | target only |
| Auxiliary force head | absent | 6 | separate, never integrated |
| Combined heads | absent | 88 | 82 physical + 6 auxiliary |
| Robot observation | 93 | preserve | joint pos29, joint vel29, omega3, gravity3, prior action29 |
| Observation history | 930 | preserve prefix | 10 samples, term-major |
| Robot future goal | 680 (689 with anchor option) | preserve prefix | 10×68, optional anchor pose9 |
| Latent / decoder input | 32 / 962 | preserve mapped columns | latent32 + obs930 |
| Current WM MLP input | 133 | 148 before separate support context | 71+29+33; target 86+29+33 |
| Current critic input | 788 (797 with anchor) | extend deliberately | canonical71 + goal680 + xi33 + root error4 |

The 84/86/82/88 target dimensions are consistent extensions, not currently tested HOI tensors. A joint scene has global nq=43, nv=41 but its global `[qpos|qvel]` ordering is not `[robot71|object13]`. `envs/state_io.py:16` asserts nq36/nv35 and will reject it. `heads.robot_bodies` slices `:36` and `36:` and also needs a robot-prefix adapter before receiving 84-D state.

`world_model/state_spec.py` robot layout: root pos[0:3], wxyz[3:7], q[7:36], v_root_world[36:39], omega_body[39:42], qdot[42:71]. Canonical order: z1, heading-relative rot6, heading-frame v3, body omega3, q29, qdot29. Residual order: dv_heading3, domega_body3, dqdot29, dp_heading3, dtheta_body3, dq29. Velocity is updated before position; orientation is right-multiplied by a body-frame exponential. Force must not be appended to this integration path.

## Loading, command and state acquisition

`config.py` and `training/loop.py:25–100` select the actual library and build the fixed robot-only components. `motion_library.py` loads the six body/joint fields, selects tracked bodies, and creates 68-D goals. Its reference 71-D vector stores world angular velocity, unlike simulator 71-D state which stores body angular velocity; `ref_qpos_qvel()` explicitly rotates it. The loader does not validate FPS and assumes 50 Hz in summaries. Part B's validated 50 Hz satisfies that assumption, but HOI loading must keep validation explicit.

`commands.py` writes robot state only on reset/resampling, advances frame after rewards, exposes current target frame, and clamps at clip end. `tracking_env_cfg.py` uses native G1, plane, 5 ms×4 steps, position actions, no force sensors/object/supports. `actions.py` preserves clean actions in observations while executed action adds Gaussian noise before scale/offset. `SimContext` samples episode-constant friction/bias and startup COM perturbations; no object or support context exists.

Current `hoi/` contains offline `reference.py`, `asset_manifest.py`, `reference_viewer.py`, `reference_browser.py`, viewer HTML/JS and initializer only. `ReferenceClip` validates the accepted artifact and supplies explicit future-window validity masks. `ReferenceScene` requires `display_only=True`, constructs a kinematic inspection model with no actuators, and uses pose assignment plus `mj_forward`; it cannot be used as the actuated P1 scene. Its accepted behavior and assets must remain intact.

## Recorder and replay

`data/recorder.py` captures reset state and context; nonterminal rows after step contain previous state plus clean/executed action, reward and previous frame. Terminal callbacks write the final transition and an extra terminal-state row before reset. Raw qpos/qvel are fresh there; derived kinematics lag a physics substep. Force must come from the interval history, and terminal histories must be captured before reset.

`ReplayBuffer` is `[env,time]` with circular write pointers, episode IDs, terminal flags and episode-start flags. `window_mask(k)` requires k+1 rows within one episode, forbids a terminal before the final target, and excludes write-pointer crossings. It does not return padded short-window validity masks. `sample_starts` backfills history only within episodes and zeros prior actions at starts. `xi` is taken once per window because it is episode-constant. Neither object force labels, contact history, sensor validity, asset ID, nor separate clip-completion/time-limit masks are stored.

**Verified existing discrepancy:** recorder stores `done_next=1` on the final transition row and also on the extra terminal-state row, but `sample_windows` reads `done` from successor rows. For rows s0(done0), s1(done1), s2(terminal,done1), the two-step sample yields `[1,1]` instead of transition labels `[0,1]`. [architecture_checks.json](architecture_checks.json) reproduces this without editing the implementation. P2's HOI replay adapter must establish one unambiguous alignment and test it; any shared correction requires a small separately reviewed change preserving the default path's intended behavior. The bug is recorded, not fixed in P0.

## WM, optimization and BPTT

`WorldModel` consists of DynamicsMLP (133→2048→2048→2048→70), an analytic integration spec, differentiable FK/reward/observation heads and learned termination head. The termination MLP consumes scaled height/tilt error plus their magnitudes (4→64→64→1). Dynamics final layer starts at zero, producing constant-velocity extrapolation rather than a static identity when dt is enabled.

`window_loss` starts from logged s0 and autoregressively feeds predictions through k steps without detaching. State loss sums the 70 normalized squared residual errors and averages over windows/steps; termination uses BCE. `WorldModelTrainer` gathers target references, fits executed actions over default eight-step windows with Adam, and optionally compiles one loss step. It has no force/object loss, per-step validity mask, warm-up validation or force gating.

`imagination/compiled.py::ImagStep` compiles one physical prediction, FK reward, termination and observation; references cross the compiled boundary as explicit tensors. `rollout.py::imagine` builds term-major history from replay, applies policy, rolls the learned state forward and computes fresh analytic observations. `PolicyTrainer.update` freezes WM/critic parameters but preserves gradients through their state/action inputs; J includes predicted survival and terminal critic value. Actor loss is −J/H. Critic targets use the existing TD(lambda) recursion and detached imagined batches. This optimizer structure should be reused, not copied into a second HOI trainer.

**Clip-end gap:** imagined frame indices are clamped with `minimum(frame+t,last)` while rollout continues for H steps, with no clip-valid reward/bootstrapping mask. Final-frame reference rewards can repeat. The current command also advances/flags clip-end after reward; P2/P5 must explicitly test final valid transition alignment rather than inheriting the clamp as completion semantics. Part B's window mask supplies an existing semantic precedent, not automatic runtime integration.

## Support-context identifiability

Robot canonical features discard global xy/yaw. A robot-relative object feature branch also discards a common robot/object xy translation and yaw rotation. With fixed support bodies in world coordinates, those equivalent canonical inputs can have different support contact outcomes. The existing 33-D context cannot resolve that ambiguity; it is unchanged throughout the episode. A synthetic translation probe confirmed robot canonical equality after a (2,−3) m translation.

Candidate representation for P3 review: transforms of both support roots relative to the current robot root/heading (position and orientation), together with immutable geometry/asset identity. This is time-varying physical context, recomputed from actual or imagined state, never appended silently to xi33. Its exact encoding/dimension is not frozen here. Required later identifiability/ablation: paired robot/object states with identical canonical features but different support-relative locations, then compare object-motion/force prediction with and without context. No experiment was run and no physical head added in P0.

Reference object velocity, if later needed, belongs in a separate cache: central position differences and quaternion-relative log-map angular velocity with explicit body/world order, one-sided endpoints. It is neither simulator velocity supervision nor a reason to rewrite the accepted NPZ.

## Trainer and checkpoint boundaries

`Trainer.run()` owns collect→WM fit→policy improvement, UTD scheduling, distributed gradients, logging and checkpoint cadence. Constructor currently instantiates all fixed classes directly and overwrites policy goal_dim from robot future_frames/anchor settings. It has no runtime factory. `PolicyTrainer.update` hard-calls module-level `imagine`, `ImagStep`, `_critic_inputs` and the current objective; optional hooks must default to the exact old functions. Merely subclassing WorldModel or registering a CLI will not connect a correct HOI run.

Checkpoint keys are iteration, num_updates, wm, policy, critic, three optimizer states, sampler bin_return/bin_memory and config. Replay, RNG and UTD remainder are not saved. Existing `load` is strict for model weights and restores optimizer/sampler state; resume refills replay. HOI initialization from a robot checkpoint is a different operation from HOI resume and must not reuse optimizer/replay blindly.

The discovered checkpoint passed strict loading of all three models under its actual config. Current shared source files are byte-identical to the external candidate checkout, but this does not prove training-time Git provenance. Key tensors:

| Weight / buffer | Shape | Later mapping obligation |
|---|---|---|
| wm.dynamics.mlp.net.0.weight | [2048,133] | map robot features[0:71], action[71:100], xi[100:133] by named fields; action/xi columns move when object features are inserted |
| wm.dynamics.mlp.net.6.weight | [70,2048] | preserve robot residual rows; initialize object rows separately |
| policy.encoders.motion.mlp.net.0.weight | [2048,680] | retain goal prefix and scales; zero new conditioning columns where equivalent |
| policy.encoders.motion.scale | [680] | retain per-term scale/order exactly |
| policy.decoder.mlp.net.0.weight | [2048,962] | latent32 + obs930; preserve history ordering/scales |
| policy.decoder.scale | [930] | retain term-major 10-frame normalization |
| policy.decoder.mlp.net.8.weight | [29,256] | action output remains 29 |
| critic.mlp.net.0.weight | [256,788] | map canonical, goal, xi and error fields explicitly |

For target features86, action and xi would begin at columns86 and115 respectively (before any separate support branch). Copying the first 133 columns as an undifferentiated prefix is wrong. No `strict=False` transfer, HOI migration, actor equivalence claim, or checkpoint write was performed. Numerical old-vs-migrated actor equivalence remains P7 work.
