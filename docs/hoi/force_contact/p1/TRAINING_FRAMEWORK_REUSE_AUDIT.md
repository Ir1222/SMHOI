# P1 Training Framework Reuse Audit

**PASS. No existing source or dependency file changed. No training algorithm was
copied, replaced or executed.** P1 adds physical construction, state access and raw
sensing under `src/supermimic/hoi/`.

| Original component | P1 reuse / preservation | Evidence |
|---|---|---|
| Native `get_g1_robot_cfg`, actuator configuration, `G1_ACTION_SCALE` | Reused directly; all 29 actuators and original gains/limits remain | Compiled actuator-array equality and action-config equality in `test_joint_scene.py` |
| `envs/actions.py:NoisyJointPositionActionCfg` | Reused; P1 calibration selects zero noise, original defaults unchanged | Original HDMI smoke test still measures execution noise |
| mjlab `ManagerBasedRlEnv.step/reset`, `Scene`, `Simulation`, `ContactSensor` | Reused directly; P1 does not subclass or duplicate their main step loop | Four native 5 ms callbacks, CUDA step/reset parity, partial-reset isolation |
| `hoi/asset_manifest.py:load_manifest/_add_geom` | Reused immutable M6.5 parameters and source verification | Seven compiled geoms, mass/COM/inertia, both supports, preservation hashes |
| `hoi/reference.py:ReferenceClip` | Reused validated read-only Part B reader | Hash matches committed reference manifest; reset cache is in memory only |
| `envs/state_io.py` | Reuses NQ/NV/state constants; original implementation and nq36/nv35 assertions preserved | Original StateIO works on robot scene and rejects joint scene; new named adapter tested with reversed entity order |
| `kinematics/rotations.py` | Reuses wxyz rotation/log/transform primitives | Synthetic twist, MuJoCo COM comparison and physical rotated-force checks |
| `world_model/state_spec.py` | Preserved; no second robot StateSpec | No modifications; original CPU state tests pass |
| `training/loop.py:Trainer.run`, scheduler/distributed/logging/checkpoint lifecycle | Preserved and reserved for later integration | Byte-identical; no new Trainer or training entry |
| `data/recorder.py`, `ReplayBuffer` | Preserved; no HOI Replay/schema/transition labels in P1 | Original environment smoke test samples eight-step Replay windows |
| WM fitting / `WorldModelTrainer` | Preserved; no second WM training loop | Byte-identical; no fitting or optimization launched |
| Actor BPTT, compiled imagination, Critic TD(lambda), optimizers | Preserved; no new optimization path | Byte-identical; existing applicable tests pass; LAFAN1-dependent tests explicitly skipped |
| `sm-train`, pyproject and default configuration | Preserved | Exact hashes unchanged; no `smHOI-track` registration in P1 |

Shared modifications and justifications: **none**. The per-world static support
collision-cache correction is entirely in the new HOI factory. It uses the pinned
dependency's existing model/data interfaces without editing installed code.

Duplicated functionality: **no training functionality**. The joint state adapter
gathers and scatters named addresses because the original StateIO's global 36/35
layout cannot represent the box. It reuses original robot layout constants and
rotation functions. The calibration script's manual single-substep calls and
fixed-robot fixture are diagnostics; control-interval timing is tested through the
unmodified native environment loop. No duplicate production stepping loop exists.

Regression result: complete suite **69 passed, 17 skipped**, one existing Torch
deprecation warning. Seven new P1 tests passed. The 17 skips require absent LAFAN1
NPZ/CSV inputs, with reasons listed in `tests_full_02.log`. The added real CUDA
robot-only HDMI test covers observation930/goal680, action29, state71, xi33, original
StateIO, execution noise and original eight-step Replay windows. It performs no
optimizer update. No checkpoint migration or policy-quality claim is made.

All 126 protected baseline files and all ten P0 dependency-source hashes match.
Accepted Part B, assets, historical audit FAILs, viewer, original checkpoints and
user-owned files remain intact. See [validation_summary.json](validation_summary.json).
