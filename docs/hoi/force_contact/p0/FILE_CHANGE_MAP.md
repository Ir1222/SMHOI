# P0 implementation file-change map

This is an audit map of required changes against the supplied master plan, not implementation or authorization for later stages. The available master file ends at P0.4, so new filenames below are provisional where the missing stage details could specify otherwise. Restore the full master before freezing the map. Existing public defaults and accepted artifacts remain unchanged.

## P0 actual changes

The original 21 documentation/evidence files under `docs/hoi/force_contact/p0/` were committed after approval as `579d1a8c661985c91926a365c7505488abe03bb8`; GitHub `main` was verified at that commit during reconciliation. No changes to `src/`, `tests/`, `scripts/`, pyproject, uv.lock, data, accepted outputs, dependencies or checkpoints. P0 code probes were temporary scripts under `/tmp`; a documentation transcript records their content.

The subsequent [closure review](CLOSURE.md) updates only `REPORT.md`, `VALIDATION.md` and this map, and adds `CLOSURE.md` and `reconciliation_checks.json`. These changes are uncommitted pending review. The canonical master destination is `docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md`; restoration awaits the complete text. No P1–P8 map entry below is frozen by this reconciliation.

## P1 physical prerequisites only, after explicit approval

| File | Change | Why / acceptance boundary |
|---|---|---|
| `src/supermimic/hoi/scene.py` (new) | Build mjlab entities from the verified M6.5 manifest: native actuated G1, one free seven-geom bread box, two fixed supports, ground | Part B's display scene is unactuated. Preserve all original native gains/colliders, box inertia/material parameters and original support transforms. Assert named joints/body/geom IDs and no assistance/object actuation. |
| `src/supermimic/hoi/config.py` (new, P1 subset) | Isolated scene/sensor validation config with 5 ms × 4 timing, manifest/reference paths and explicit capacity choices | Do not alter robot-only `make_env_cfg` defaults or expose training before its stage. Keep existing `training_ready=false` meaningful; physical calibration is its own explicit mode. |
| `src/supermimic/hoi/contact_sensors.py` (provisional new) | Independent left/right installed ContactSensor configs and validated output access | Each hand includes selected pitch+yaw bodies; secondary is the one box body; `secondary_policy=error`, netforce, one slot, four-history. Assert all seven geoms and both primary sets after compilation. No Force Head or Replay labels yet. |
| `src/supermimic/hoi/state.py` (new, P1 subset if needed) | Named robot/object reset and inspection functions | Avoid current global nq36/nv35 assumption. Reset only requested worlds; distinguish root/COM velocity and origins. Full 84-D Replay/state integration belongs to P2/P3. |
| `scripts/hoi/calibrate_contacts.py` (provisional new) | Deterministic physical sensor calibration harness writing a new P1 evidence directory | Exercise actual selected native hand bodies; known loads, sign/frame transforms, left-only/right-only/bilateral/no-contact/other-body cases; each box geom; transient and four-substep behavior; reset and multiworld isolation; compare raw contacts with netforce and check overflow. Missing master details must supply exact prescribed protocol/tolerances. |
| `tests/test_hoi_scene.py`, `tests/test_hoi_contact_sensors.py` (provisional new) | Focused scene/sensing tests and regression checks | P1 physical evidence must cover the user's enumerated cases. Synthetic P0 sensor bookkeeping and ball-plane probes are not substitutes. |
| P1 report/evidence path (not supplied in available master) | Report calibration results, differences and full diff | Stop for review, then separate P1 commit only after approval. |

No P1 changes to original Recorder, Replay, WM, imagination, policy optimization, or Trainer.run are justified by this audit. If using the original environment configuration for calibration, disable its robot-only recorder and replace/reset only the necessary HOI command/state path in isolated configuration; leaving ReplayRecorder active would immediately hit the nq/nv assertion. Preserve the original path for regression.

The existing manifest loader and pure geom-construction helpers can be reused without writing assets. Do not repurpose `ReferenceScene(display_only=True)` for physics or weaken its actuator/no-physics checks. Do not import the source carry checkout's phase policy, HDMI foot replacement, 2 ms stepping, or 35-D context.

## Later-stage map for review only; no P2+ work authorized

| Master stage | Primary HOI files / narrowly shared surface | Required change found by P0 |
|---|---|---|
| P2 | `hoi/state.py`, `hoi/replay.py`, provisional `hoi/recorder.py`, `hoi/contact_sensors.py` | Explicit robot71/object13 indexing; interval measurement alignment; terminal-before-reset capture; clean/executed actions; force/contact validity; asset hash; distinct failure/truncation/completion; padded short windows. Resolve the verified successor-row done misalignment in an HOI adapter or a separately reviewed minimal shared fix. |
| P3 | `hoi/world_model.py`, provisional `hoi/state_spec.py` and `hoi/wm_trainer.py` | Joint86 input features, physical82 residual, auxiliary force6, normalization and autoregressive masked losses; object velocity supervision from simulator; support-context identifiability and ablation. Reuse DynamicsMLP/WM fitting structure rather than rewriting the shared implementation. |
| P4 | `hoi/rewards.py`, provisional `hoi/contact_geometry.py` | Shared differentiable EEF/target geometry and force-aware reward functions, keeping desired, measured and predicted contact separate. Preserve original robot terms/scales. No phase one-hots or live sensors inside imagination. |
| P5 | `hoi/observations.py`, `hoi/imagination.py`, provisional `hoi/compiled.py` and runtime reference adapter | Current predicted object state drives goals/rewards each step; preserve robot prefixes; finite-clip validity and single final-transition reward; eager/compiled parity and action→object/force gradients. |
| P6 | Separate `scripts/hoi/` / HOI validation module and report | Force-gradient fidelity and go/no-go; do not activate force Actor optimization from prediction accuracy alone. |
| P7 | `hoi/train.py`, `hoi/config.py`, provisional `hoi/runtime.py`, `hoi/checkpoint.py`; `pyproject.toml`, `training/loop.py`, `imagination/policy_trainer.py`, possibly shared CLI bootstrap | Register `smHOI-track`; optional component/runtime factory in Trainer construction with exact default behavior; narrow default-no-op imagination/critic/step/objective hooks. Reuse Trainer.run and existing optimizers. Explicit named checkpoint maps and old actor numerical parity; separate init-from-tracking from full HOI resume. |
| P8 | `hoi/evaluate.py`, evaluation script/report | Full 840-frame frame-zero task, contact/force/slip and placement/release metrics, robot-only regression. No source-motion replay, terminal reset, or clip completion substituted for success. |

No unconditional changes to shared `StateSpec`, `StateIO`, Replay or WM dimensions are proposed. Their fixed slices must remain valid for robot-only training. A task factory supplies compatible HOI implementations; policy input scaling must use explicit field maps, because the current goal scaler repeats 68-D robot blocks and cannot interpret arbitrary appended HOI features safely.

## Immutable boundary

The accepted Part B NPZ, its committed manifests/reports, all M0–M6.5 records and both asset manifests remain immutable inputs. Existing checkpoints and dataset archives are never overwritten. Unrelated external checkout experiments remain untouched. P0 and P1 require separate reviewed commits; no automatic push. This map does not waive any historical physical FAIL, missing-plan requirement, or later-stage validation gate.
