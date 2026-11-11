# Revised P3 implementation report

**Overall P3 acceptance: PARTIAL.** The approved contact-model implementation is
present and its unit/regression checks pass. The authorized physical capture failed
after one of 32 transitions. Experimental execution stopped immediately: no retry,
bounded fit, or learned Contact checkpoint. Gate A and Gate B are not evaluated.

Baseline: `main` at `0a228cdcce01d896282c430d9ee77ac17a933163`. Implementation follows
the approved [revision specification](REVISION_SPEC.md) and
[audit](IMPLEMENTATION_AUDIT.md), without modifying either or the master plan.

## Acceptance by component

| Component | Result | Evidence / limit |
|---|---|---|
| Core implementation and unit/regression checks | PASS | Full pre-collection suite: **123 passed, 17 skipped**; final metric review changes: **8 passed**. Skips are existing missing LAFAN1 assets. |
| Actual old joint166/88 → contact167/83 migration | PASS | All named physical rows/columns preserved; FP32 physical prediction difference **0** at every horizon 1–8, for initial contacts 0, 1, 0.37; strict new checkpoint round trip **0**. |
| Original robot initialization | Qualified / PARTIAL | Deterministic 32-wide robot tests pass through eight steps. Real external 2048-wide robot passes one-step FP32 and every horizon in FP64; FP32 horizons 2–8 exceed the approved tolerance. See below. |
| Physical collection | FAIL | Reference reset succeeded; one transition saved, then unchanged P1 guard raised overflow 512. |
| Gate A: learned robot/object one/eight-step performance | NOT EVALUATED / PARTIAL | No complete approved train/validation dataset and no bounded fit. Migration parity is not learned prediction accuracy. |
| Gate B: learned reference-contact classification/events | NOT EVALUATED / PARTIAL | Only one OFF→OFF row captured; no onset/release or full eight-step window. |
| Overall P3 | **PARTIAL** | Physical capture blocker and external FP32 compatibility limitation remain. No claim of cross-clip generalization. |

“Implementation PASS” refers to the implemented contract and executed unit tests;
it does not certify universal FP32 checkpoint parity or predictive readiness.

## Implemented model and data flow

Physical state stays **84D = robot71 + box13**. The HOI batch adapter constructs
predictive state **85D** from physical state and existing `intent_ref [K,N,2]` pairs.
For an eight-step window, logged states are `[9,N,85]`, actions `[8,N,29]`, episode
xi `[N,33]`, and next-contact targets `[8,N,1]`. Binary labels must match the exact
canonical reference at each frame; adjacent pairs and state targets must agree.
The final contact bit comes from the last transition endpoint, not terminal padding.

One unchanged `DynamicsMLP` receives **167D** in the approved order:
canonical86, current contact1, support18, executed action29, normalized xi33. Its
SiLU trunk feeds robot residual70, object residual12, and one next-contact logit:
**83 outputs**. Only the first 82 outputs enter `JointStateSpec.integrate`. Sigmoid
probability is appended to the next physical state and fed back without detaching,
thresholding or teacher forcing. No frame index or future reference enters the MLP.
Default hidden widths remain 2048×3; the diagnostic initialization is 64×3. The
existing termination head remains separate.

Robot feature order, scaling and integration are unchanged. Box state uses
environment-local root position, `wxyz` quaternion, world root linear velocity and
body angular velocity. Canonical box15 uses robot-heading axes and root-relative
position, heading-relative rotation6, absolute object linear velocity in heading
axes, and object body angular velocity. The existing 12D residual integrates at
0.02 s with updated velocities and right-multiplied body rotation increment.
Support18 is recomputed from predicted physical84; xi33 remains separate.

Contact uses ordinary `BCEWithLogitsLoss`, independently of termination BCE. Physical
loss components retain their existing scales and reductions. The configured bounded
fit preserves migrated robot/object/termination coefficients
`0.016480818629139207 / 0.13455571473227598 / 1.4160031531976345`, with contact
coefficient 1. It uses the inherited original Adam/update/backward/allreduce/clipping
path. No force normalization, regression, balancing, or predictive force metrics
remain in that path. Physical sensors and their archive integrity checks remain.

The fit script is implemented for exactly 200 updates, CPU hidden64×3, batch32,
window8, one batch/update, learning rate 0.001 and seed 404, without tuning against
validation. **That bounded fit was not executed.** Unit tests exercise the original
optimizer on existing measured rows only to verify software behavior; they do not
establish a new trained Contact model. See the separate
[Training Framework Reuse Audit](TRAINING_FRAMEWORK_REUSE_AUDIT.md).

## Reference, capture, and stop condition

Canonical reference:
`/home/han/pyx/SMHOI/outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz`.
SHA256 `f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7`;
840 frames, 50 Hz, global boolean `object_contact`; onset 259→260, release 549→550.
The reference loader verified its manifest and field hashes. The reference, three
supporting JSON manifests, original 320-row archive and all historical outputs remain
unchanged. The old archive remains all `[1,1]` and is explicitly rejected by the new
bounded-fit preflight.

The collector attempted the approved schedule `(train onset 256, train release 546,
validation onset 255, validation release 545)`, eight transitions each, using one
native physical environment with separate resets and seeds 411–414. It called P1
`ReferenceResetCache`/`JointStateIO.reset_to_reference`, then the unchanged P2
recorder. No subsequent state overwrite or reference-derived physical supervision
was introduced.

Only train-onset seed 411 began. Reset and the full 256→257 transition passed; its
measured state/action/xi/sensors/masks and intent `[0,0]` are saved. During 257→258,
the third solve reached the solver iteration limit: the first two substeps had
been captured when `contact_capacity` raised `MuJoCo Warp overflow flags: [512]`.
Installed `mujoco_warp/_src/types.py:184` defines `ITERATIONS = 1 << 9`; this is not
evidence of contact-buffer saturation. Solver settings stayed 10 iterations and 20
line-search iterations; all physical-state and capacity guards stayed enabled.
No claim is made about the deeper cause of the solver limit.

The failure snapshot contains finite state84, overflow 512 and captured-substep
count 2. Only the first two substep entries are valid for the incomplete interval;
the remaining storage must not be interpreted as new samples. The one complete
row's sensor aggregation and reference intent were verified read-only. The archive
is marked `FAIL` and rejected by `load_archive`; it is not a training dataset.
No remaining episodes, retries, fitting or model-performance evaluation were run.

## Checkpoints and numerical limits

The old joint source SHA256 is
`6246b001cb14a4966a90f6a99815b2fb45172b82f9cf57a867e6cdd0ce95d56b`.
Migration preserves robot/object input 0:86, shifts support/action/xi columns by one,
zeros new contact input 86:87, preserves all named physical output rows 0:82,
discards force 82:88 and initializes new contact row 82:83 weight/bias to zero.
Initial contact probability is 0.5. Hidden layers, termination head and support
buffers are copied; incompatible metadata, shapes, missing/extra keys, nonfinite
values and altered support/reference contracts are rejected. New schema is
`hoi-joint-contact-wm-v2`; loading is strict. Optimizer state is not migrated.

`migration_02/migrated_contact_weights.pt` is **initialization only**, not a learned
Contact checkpoint. SHA256:
`8cfed1f83cc885ef3b29d5139c89d5a951f54bc16012ac831b921c81c85b6c36`.

The optional actual 2048-wide external robot checkpoint remains unchanged. Its
per-horizon FP32 maximum absolute differences were:
`[3.72529e-8, 7.62939e-6, 1.33514e-5, 1.33514e-5, 1.90735e-5, 1.62125e-5,
1.62125e-5, 2.28882e-5]`. With `atol=rtol=2e-6`, only horizon 1 passes. FP64
differences are 0 at all eight horizons, consistent with accumulation-order effects
from the changed matrix width rather than a named-weight mapping error. This is an
inference, not a universal FP32 guarantee. The first failed probe/log and the
subsequent precision diagnosis are both preserved; no tolerance was relaxed.

## Commits, files, and preservation

| Commit | Group | Executed evidence |
|---|---|---|
| `ff22661` | Model, adapter and labels | 17 targeted tests passed |
| `79560be` | Strict checkpoint migration | 11 tests passed; real joint parity 0; external precision limit recorded |
| `57633a9` | Metrics, bounded scripts and historical verifier | 123 passed/17 skipped; final 8 targeted checks passed; historical verifier PASS |
| `903ec78` | Failed capture and stop | One complete row, two partial substeps; failed archive rejected |

Changed source/test files (all repository-relative):

| File | Change |
|---|---|
| `src/supermimic/hoi/world_model.py` | Contact logit, state85, BCE and probability feedback |
| `src/supermimic/hoi/wm_trainer.py` | Canonical label checks, predictive batch adapter and physical online error |
| `src/supermimic/hoi/dataset.py` | Optional reference-aware archive validation and binary-label checks |
| `src/supermimic/hoi/validation.py` | All-row one-step and eight-step physical/contact metrics, baselines and two gates |
| `src/supermimic/hoi/checkpoint.py` | Named migration and strict contact schema |
| `scripts/hoi/collect_p3_diagnostic.py` | Exact 32-transition mode and preserved failure evidence |
| `scripts/hoi/fit_p3_diagnostic.py` | Approved bounded original-trainer fit and evaluation; not executed |
| `scripts/hoi/validate_wm_migration.py` | Actual joint and optional original robot checkpoint probes |
| `scripts/hoi/verify_p3_evidence.py` | Explicit legacy force-gate import/call only |
| `tests/hoi/test_force_head.py` | Retains two historical statistics tests; old Force Head test removed |
| `tests/hoi/test_wm_dataset_validation.py` | Label/diagnostic isolation, metrics, gates and historical evidence tests |
| `tests/hoi/test_checkpoint_migration.py` | Named columns/rows, strict negatives and one/eight-step parity |
| `tests/hoi/test_contact_intent_head.py` | New 85D feedback/BCE/dimension tests |
| `tests/hoi/test_contact_diagnostic.py` | New failure-preservation and invalid-fit-input tests |

New documents in this directory are `IMPLEMENTATION_REPORT.md`, `VALIDATION.md` and
`TRAINING_FRAMEWORK_REUSE_AUDIT.md`. Every new numerical/log/checkpoint artifact is
listed in [VALIDATION.md](VALIDATION.md); the final preservation manifest also lists
every file changed by this revision. No shared source, P1/P2 implementation, original
`sm-train` default, Actor, Critic, TD(lambda), imagination or dependency was modified.
User-owned master-plan changes and untracked files remain unstaged. No push/reset,
reference rewrite, or P4/P5/P6/P7 work occurred.

Further experimental progress requires explicit review of the recorded physical
failure. This implementation does not authorize changing solver settings, guards,
reference poses, targets, architecture or collection size to obtain a pass.
