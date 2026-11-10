# P0 implementation file-change map

Reconciled against the complete 1,743-line master, SHA256 `7365f680cf975263e69672d75d585c62e83532115b9065c6e63929004bdbdc9b`, restored at `docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md`. Later-stage entries are audit findings, not implementation or authorization. Helper filenames remain proposals where the master allows equivalent organization.

## Existing code to reuse and narrow extension points

Paths below are relative to `src/supermimic/` unless stated otherwise.

| Existing implementation | Reuse / minimal change boundary |
|---|---|
| `hoi/asset_manifest.py:load_manifest`, `_add_geom` | Reuse validation and pure geometry parameters in P1. Preserve `build_audit_scene` and accepted assets; do not weaken physical gates. |
| `hoi/reference.py:ReferenceClip`, `ReferenceWindow` | Reuse accepted-array validation and finite-window semantics. New runtime adapter/cache for reset velocities and goals; no conversion or viewer changes. |
| `envs/tracking_env_cfg.py`, `actions.py`, `sim_context.py` | Reuse native robot/action configuration, .005×4 timing and xi33 through isolated HOI configuration. Preserve `sm-train` defaults. |
| `envs/state_io.py:StateIO` | Preserve nq36/nv35 assertions. A new named-address HOI adapter handles robot71/object13; no global concatenation. |
| `data/recorder.py`, `replay_buffer.py` | Reuse callbacks, ring storage and episode/window mechanisms. P2 adapter/sidecar fields; any shared terminal fix must be narrow and regression-tested. |
| `world_model/state_spec.py`, `dynamics.py`, `world_model.py`, `wm_trainer.py` | Reuse robot71/residual70, MLP and autoregressive fitting. P3 composition and optional loss/reference hooks if needed; no duplicate robot StateSpec or uncoupled second WM. |
| `kinematics/g1_fk.py`, `rotations.py`, `world_model/heads.py` | Reuse FK/transforms and original robot rewards/observations. HOI wrapper passes robot-prefix state to fixed robot slices. |
| `imagination/rollout.py`, `compiled.py`, `objective.py`, `policy_trainer.py` | P5 optional rollout/step/objective/critic-input hooks defaulting to old functions; preserve BPTT, optimizers and TD(lambda). |
| `policy/nets.py` | P7 named input/scaling maps and optional dimensions. Preserve history order and action29 output; zero new actor contributions for parity. |
| `training/loop.py:Trainer.__init__/save/load/run`, `training/schedule.py` | P7 optional component construction and checkpoint metadata hooks. Constructor currently hard-codes components/goal_dim. Keep `Trainer.run()` as the one training loop. |
| `scripts/train.py`, `config.py`, repository `pyproject.toml` | P7 separate `smHOI-track` registration/config; existing CLI defaults unchanged. |

## P1 — scene, joint state/reset and sensors

New: `hoi/scene.py`, `state.py`, scene subset of `config.py`, sensor/config access in `contact.py`; `scripts/hoi/calibrate_contacts.py`. Reuse native configuration, manifest helpers and reference reader. No shared Recorder, Replay, WM or Trainer changes.

Tests: `tests/hoi/test_joint_scene.py`, `test_contact_sensor.py`; appropriate original robot-only regression. Cover named robot/object addresses, consistent nonzero reset twist, origins/fixed supports, actual pitch+yaw colliders, `left_hand_box` and `right_hand_box`, two columns per hand, all seven secondary geoms, known/rotated loads, four-substep/transient/exclusion cases, capacity flags, reset and parallel isolation.

Dependencies: approved P0, mjlab 1.6.0/Warp semantics, unchanged M6.5. Risks: robot-only recorder assertions, final-forward cache, replaced history tensors, overflow and accidentally imported source assets/gains. No dependency patch is justified.

Outputs: `docs/hoi/force_contact/p1/REPORT.md`; `outputs/hoi/contact_supervision/sensor_calibration/`.

## P2 — interval labels and Replay

New/extend: `hoi/contact.py`, `replay.py`, thin `recorder.py`, runtime reference/command adapter as needed. Reuse original storage/sampling with schema/sidecar fields. A narrow shared terminal correction is conditional on evidence.

Tests: `tests/hoi/test_force_aggregation.py`, `test_force_replay.py`; original Replay regression. Verify N versus N·s, mean/peak/occupancy/raw samples, Box orientation sample times, executed-action alignment, terminal-before-reset copies, no stale buffers, eight-step windows, clip/episode boundaries and failure/timeout/completion masks. Short-window padding is optional, with explicit validity if used.

Dependencies: approved calibration. P2.1 prefers contemporaneous substep Box orientation; it also permits a documented interval anchor. Choose and persist the convention in P2, never silently use the post-step pose. Torque diagnostics must shift each primary column's centroid torque to a common point before summing; force6 is unaffected. Risks: existing `[1,1]` terminal-label error, no found history, vector cancellation and overwritten instantaneous data.

Outputs: `docs/hoi/force_contact/p2/REPORT.md`; `outputs/hoi/contact_supervision/replay_diagnostics/`.

## P3 — joint WM, Force Head and WM migration

New/extend: `hoi/world_model.py`, composed `state_spec.py`, thin `wm_trainer.py`, WM portion of `checkpoint.py`. Reuse original robot spec/MLP/fitting, adding default-preserving loss/reference hooks only if needed.

Tests: `tests/hoi/test_joint_world_model.py`, `test_force_head.py`, WM cases in `test_checkpoint_migration.py`; original WM regression. State84/canonical86, shared heads70+12+6, integrate82 only, constant-velocity quaternion prior, normalization, active/inactive weighting, episode-aware one/eight-step validation and readiness. Named WM migration belongs here under P3.9; full actor/critic integration remains P7.

Dependencies: approved Replay and simulator root-origin twist supervision. Support geometry relative to current robot heading remains a candidate; paired prediction/ablation in P3 must determine need/dimension, separately from xi33. Risks: leakage, zero-force dominance, reference-intent shortcuts, shifted action/xi columns and premature Actor force reward.

Outputs: `docs/hoi/force_contact/p3/REPORT.md`; `outputs/hoi/contact_supervision/force_model_validation/`, including support-context decision.

## P4 — differentiable interaction reward

New: `hoi/rewards.py`, optional pure `contact_geometry.py`; lookahead/optional smoothing in a separate runtime cache. Reuse FK, rotations, original robot reward and accepted bool intent.

Tests: `tests/hoi/test_soft_contact_rewards.py`, `test_release_reward.py`; all controlled cases in P4.11, finite gradients, bounded force saturation, independent hands, persistence without constant-force penalty, support-coupled release and raw/weighted metrics.

Dependencies: approved P3 and actual reference approach geometry for lookahead. Risks: copying HDMI's outside-intent baseline, SUGAR's bilateral equality, PhysHSI stages/unfiltered forces or using reference object pose as current state. No final coefficients chosen in P0.

Outputs: `docs/hoi/force_contact/p4/REPORT.md`; `outputs/hoi/contact_supervision/reward_validation/`.

## P5 — imagination and BPTT

New/extend: `hoi/imagination.py`, `observations.py`, optional compiled-step adapter. Narrow hooks in shared `imagination/policy_trainer.py` and, if necessary, `compiled.py`/`objective.py`, preserving defaults. These hooks belong in P5, not wholly in P7.

Tests: `tests/hoi/test_imagination_parity.py`, action→object/force gradient checks and original imagination regression. Compare states, force, confidence, persistence, every reward and action gradients in eager/compiled modes; recompute dynamic goals and enforce completion/truncation/survival/critic-bootstrap masks.

Dependencies: approved P4/readiness. Risks: freezing parameters accidentally detaches input gradients, stale goals and repeated final-frame reward. Diagnostic gradients are permitted; force-based Actor optimization stays gated pending P6.

Output: `docs/hoi/force_contact/p5/REPORT.md`; fresh documented parity/gradient evidence directory (the master does not prescribe a P5 output subdirectory).

## P6 — force-gradient fidelity

New: `scripts/hoi/validate_force_gradients.py`, optional focused HOI validation helper; `tests/hoi/test_force_gradient.py` for deterministic restoration/measurement mechanics. Reuse calibrated reset, P2 labels, P3 model and P4 rewards; no training-loop change.

Experiments: fixed-state/action perturbations for prescribed contact regimes at multiple epsilon values; left/right and per-action cosine/sign/magnitude comparisons, unstable-boundary diagnosis and explicit go/no-go.

Dependencies: approved P5. Risks: nonsmooth contact, drifting randomization/reset, inconsistent frame/interval definitions and mistaking low prediction error for useful gradients.

Outputs: `docs/hoi/force_contact/p6/REPORT.md`; `outputs/hoi/contact_supervision/gradient_fidelity/`.

## P7 — integrated entry, full migration and short training

New/extend: `hoi/train.py`, `config.py`, `checkpoint.py`, optional `runtime.py`; `pyproject.toml` CLI registration. Narrow `training/loop.py` construction/save/load hooks and policy dimension/scaling extensions as required. Reuse original `Trainer.run()`, optimizers and scheduler.

Tests: `tests/hoi/test_checkpoint_migration.py`, default/entry regression and save/resume/schema tests. Named actor/critic maps, zero-new-input actor parity, new optimizer/Replay state when incompatible, and metadata for schema/frame/sign/normalization/sensors/timing/rewards/readiness/reference/assets. Original save lacks Replay/RNG state: distinguish schema restoration and buffer refill from exact stochastic resume.

Dependencies: approved P6 decision and explicitly approved short experiments. Run only P7.6's staged experiments; record actual gradients/parameter deltas for intended modules, save/resume and NaN/Inf checks. Risks: incomplete source-checkpoint provenance and incompatible input scales/optimizer state. Never overwrite the original checkpoint or silently substitute the external candidate.

Outputs: `docs/hoi/force_contact/p7/REPORT.md`; `outputs/hoi/contact_supervision/training/`.

## P8 — full physical evaluation

New: `hoi/evaluate.py`, `scripts/hoi/evaluate_force_hoi.py`, focused metric/provenance tests. Reuse physical scene, statistics and frozen model/reference/asset loaders. No training.

Evaluation: user-selected checkpoint from frame 0 through all 840 samples, all contact/object/WM/task metrics in P8.3, documented post-release stability interval and every failed episode. Historical terminal-only settle is not achieved task completion.

Dependencies: P7 approval and explicit checkpoint/evaluation authorization. Risks: mistaking resets/reference replay for policy success, excluding failures, support inconsistency and success inferred solely from frame 839.

Outputs: `docs/hoi/force_contact/p8/REPORT.md`; `outputs/hoi/contact_supervision/evaluation/`.

## P0 changes and immutable boundaries

Original audit `579d1a8` and reconciliation `e895731` are preserved. Current changes restore the canonical master, close missing reference/frame checks and update documentation only; [CLOSURE.md](CLOSURE.md) lists every changed file.

Part B, asset manifests, M0–M6.5 evidence, source datasets, checkpoints, dependencies, unrelated experiments and the viewer remain untouched. The supplied complete `.txt` is copied byte-for-byte; its incomplete `.md` neighbor is preserved. No P1–P8 file above was created. No automatic commit or push.
