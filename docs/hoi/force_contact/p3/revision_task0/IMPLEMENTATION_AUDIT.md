# Task 0 implementation audit

**Documentation-only audit, 2026-11-11.** Local and upstream main resolve to
[`9bf253c74c92ebce9857f7863c7298f99a3f9567`](https://github.com/Ir1222/SMHOI/commit/9bf253c74c92ebce9857f7863c7298f99a3f9567).
The [revision specification](REVISION_SPEC.md) is the proposed implementation
contract. No Python source, collected data, checkpoint, or earlier evidence is
changed by Task 0; no collection or optimization is executed.

## 1. Inspected baseline and evidence

Read the complete current master plan, P0 reports/audits/closure/validation, P1/P2
specifications/reports/validation/reuse audits, and P3 specification/report/support
evidence/validation/reuse audit. Relevant accepted baselines are P1 `a562d936`, P2
implementation `5730be4d`, and P2 final evidence `a991ee7`; current P3 commits run from
`29a48fd` through `9bf253c`. Original StateSpec, DynamicsMLP/WorldModel/loss loop,
trainer, Replay sampling, Actor/imagination, checkpoint interfaces, and current HOI
implementations were inspected. This revision uses current source, not proposed
interfaces from the master plan.

Initial user-owned changes, to preserve and exclude from staging:

```text
 M docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md
?? .vscode/
?? SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md
?? outputs/hoi/asset_audit/part_b/reference_refined/
```

The working master-plan SHA256 is
`5b0c26e3c3d8c721e5e1f76ffe194f6141994a9eba4894ce7f43d7266e9be82f`.
A pre-edit hash snapshot in `/tmp/smhoi_task0_before.json` covers 513 existing files,
including tracked files, visible untracked files, and ignored HOI evidence/reference
files. It is a local preservation check, not a new accepted experiment artifact.

## 2. Training Framework Reuse Audit

Paths below are repository-relative. These components can be reused unchanged:

| Component / files | Actual contract and reason for reuse |
|---|---|
| `src/supermimic/world_model/state_spec.py` | Original robot71/features71/residual70, ordering, normalization and integration. |
| `src/supermimic/hoi/state_spec.py` | Physical84/canonical86/residual82 and verified object geometry. Leave its legacy `FORCE_DIM=6` constant unused; it must not define the new output width. |
| `src/supermimic/hoi/support_context.py` | Support18, immutable asset geometry and root/heading transforms; keep xi33 separate. |
| `src/supermimic/world_model/dynamics.py` | Configurable input/output widths already support 167/83; one shared SiLU trunk. |
| `src/supermimic/world_model/world_model.py` | Existing auxiliary-target hook and autoregressive loss loop carry state85 and component83 without another shared-core change. Original robot defaults remain intact. |
| `src/supermimic/world_model/wm_trainer.py` | Original Adam, update/backward, allreduce and clipping inherited by HOI trainer. |
| `src/supermimic/hoi/replay.py`, `recorder.py`, `contact.py`, `state.py`, `reference.py`, `scene.py` | Existing physical84 storage, intent pair sidecar, episode-safe windows, native measurements, reference/reset IO. No P1/P2 edits. |
| Original Replay, Actor, Critic, TD(lambda), imagination and `Trainer.run()` | No integration into policy optimization and no lifecycle/default changes. |
| `scripts/hoi/p3_common.py` | Exact refined reference pin, complete transition snapshots, and non-overwriting archive writer. |
| `src/supermimic/hoi/force.py` | Retain as a dormant historical utility for old evidence/tests; remove it from the new predictive training path. |

Actual data flow is recorder → HOI Replay sidecars → archive validation → whole-
episode split → existing Replay window sampler → HOI trainer `loss_on` → inherited
`WorldModel.window_loss` → HOI `_loss_step` → inherited trainer `update`. The adapter
adds predictive contact to sampled batches; the simulator/Replay stay physical84.
The original physical-only `rollout_error` requires a HOI override/slicing adapter.

## 3. Exact prospective changes

These are **future implementation changes**, not files modified by Task 0.

| Existing file | Required change |
|---|---|
| `src/supermimic/hoi/world_model.py` | 167→83 shared model, 85D recurrent state, probability feedback, contact BCE/weights/metrics; remove force6 outputs and force-statistics constructor/config/loss dependence. |
| `src/supermimic/hoi/wm_trainer.py` | Validate/cache reference intent, assemble state85 from existing pairs, pass next-bit supervision; adapt online error. Keep inherited update method. |
| `src/supermimic/hoi/dataset.py` | Add binary and canonical-reference intent validation through an explicit reference-aware check; preserve hash/schema/raw-sensor checks and old archive API compatibility. No relabeling or new label source. |
| `src/supermimic/hoi/validation.py` | Replace active force evaluation, coverage and readiness with physical Gate A and global-intent Gate B. Evaluate all one-step rows and full eight-step windows; retain only the old readiness evaluator under explicit `legacy_force_readiness_gate` for historical verification. |
| `src/supermimic/hoi/checkpoint.py` | Named 166/88→167/83 migration, separate new schema, strict validation/loading, no force normalization in new checkpoints. Preserve explicit robot-only initialization compatibility. |
| `scripts/hoi/collect_p3_diagnostic.py` | Add an explicit contact-boundary schedule/mode with initial reference resets and split/seed metadata; retain the old collection mode for reproducibility. No change to P2 recording. |
| `scripts/hoi/fit_p3_diagnostic.py` | Consume approved balanced dataset/split; remove force fitting/weights/metrics, train through original update, save contact logits/probabilities/targets and new checkpoint schema. Reject an all-ON dataset as Gate B evidence. |
| `scripts/hoi/validate_wm_migration.py` | Exercise old joint checkpoint transfer and new strict round trip, retaining the separate robot-only probe; remove force-statistics fitting from the new model path. |
| `scripts/hoi/verify_p3_evidence.py` | Minimal compatibility import/call adjustment to the explicitly legacy force gate; never reinterpret or rewrite historical force JSON as new contact evidence. |
| `tests/hoi/test_force_head.py` | Retain the two isolated historical ForceStatistics tests; remove the old 166/88 model fixture/head test, replaced below. |
| `tests/hoi/test_wm_dataset_validation.py` | Replace force metric/gate tests; reverse the old assertion that corrupting `intent_ref` to 99 leaves loss unchanged. It must now be rejected. Preserve clean/executed action and episode/terminal checks. |
| `tests/hoi/test_checkpoint_migration.py` | Add old-joint transfer, new schema, contact initialization and strict negative tests; retain robot-only named migration coverage. |

Add `tests/hoi/test_contact_intent_head.py` for 167/83/85 contracts, BCE and probability
feedback. Keep existing `test_joint_world_model.py`, `test_support_context.py`,
`test_wm_reuse.py`, robot regression tests and P1/P2 sensor/Replay tests unchanged.
Historical support diagnostics and reports remain intact. A later implementation
report/reuse audit belongs beside this revision, with new numerical outputs under
`outputs/hoi/contact_supervision/contact_intent_model_validation/`; do not overwrite
`force_model_validation/`. No additional shared Python file is presently necessary.

## 4. Explicit checkpoint migration

The current `migrate_robot_state_dict` accepts original robot input133/output70.
It is **not** an old-joint migrator: its assumptions would lose object output rows.
The audited existing source is:

```text
outputs/hoi/contact_supervision/force_model_validation/migration_01/diagnostic_joint_weights.pt
SHA256 6246b001cb14a4966a90f6a99815b2fb45172b82f9cf57a867e6cdd0ce95d56b
schema hoi-joint-wm-v1; hidden (64,64,64); first weight [64,166]; last [88,64]
```

All slices below are zero-based, half-open, in `dynamics.mlp.net.0.weight`.
Copy named columns, not an overlapping prefix or positional zip:

| Named input | Old joint | New contact joint |
|---|---|---|
| `robot.height` | 0:1 | 0:1 |
| `robot.rotation6` | 1:7 | 1:7 |
| `robot.linear_heading` | 7:10 | 7:10 |
| `robot.angular_body` | 10:13 | 10:13 |
| `robot.joint_position` | 13:42 | 13:42 |
| `robot.joint_velocity` | 42:71 | 42:71 |
| `object.position_heading` | 71:74 | 71:74 |
| `object.rotation6_heading` | 74:80 | 74:80 |
| `object.linear_heading` | 80:83 | 80:83 |
| `object.angular_body` | 83:86 | 83:86 |
| `contact_intent.current` | absent | **86:87, initialize zero** |
| `support0.position_heading` | 86:89 | 87:90 |
| `support0.rotation6_heading` | 89:95 | 90:96 |
| `support1.position_heading` | 95:98 | 96:99 |
| `support1.rotation6_heading` | 98:104 | 99:105 |
| `action.executed` | 104:133 | 105:134 |
| `xi.friction` | 133:134 | 134:135 |
| `xi.encoder_bias` | 134:163 | 135:164 |
| `xi.torso_com_offset` | 163:166 | 164:167 |

For final linear weight/bias (`net.6` for three hidden layers), preserve named robot
rows: `dv_heading 0:3`, `dw_body 3:6`, `djoint_velocity 6:35`, `dp_heading 35:38`,
`dtheta_body 38:41`, `djoint_position 41:70`. Preserve object rows:
`dv_heading 70:73`, `dw_body 73:76`, `dp_heading 76:79`, `dtheta_body 79:82`.
Explicitly discard old force rows `82:88`. Initialize new contact row `82:83` weight
and bias to zero, hence probability 0.5. Copy first-layer bias, interior hidden
weights/biases, termination head, and verified support tensors unchanged.

Allowlist discarded legacy buffers: `force_statistics.scale_N`,
`counts_inactive_active`, `median_active_N`, `class_weights`, `force_on_N` (each with
the `force_statistics.` prefix). Validate the complete old schema/key set/shapes/
finite values and support/reference/scaling contracts before transfer; unexpected
keys or mismatched hidden widths fail. Any legacy ForceStatistics use is restricted
to validating the legacy source, never fitting it or constructing the new WM.
Do not interpret force weights/statistics/readiness as contact metadata.

Use new schema `hoi-joint-contact-wm-v2`: physical84, predictive85, canonical86,
contact1, support18, input167, residual82, output83, reference hash and global label
semantics, `t→t+1` alignment, probability feedback, threshold0.5, normalization/
integration/architecture contracts, and diagnostic provenance. Keep sensor metadata
as provenance only. Migration resets new acceptance to unvalidated and records
source hash/mappings/discards; it does not transfer optimizer state or claim resume.
Build a complete destination dictionary and load with `strict=True`; new-format
loads validate metadata and exact keys before strict loading. Never use `strict=False`.

Retain separate original-robot133→contact167 initialization: robot columns unchanged,
action `71:100→105:134`, xi `100:133→134:167` with the same named subfields; initialize
object/support/contact input columns and object/contact output rows to zero. Hidden
architecture must match; the existing external 2048-wide robot checkpoint cannot
silently initialize the 64-wide diagnostic model.

## 5. Required future tests

| Test scope | Concrete checks |
|---|---|
| Head/feedback | Input167, output83, state85; changing the contact output bias changes probability but not that step's physical84; BCE agrees with PyTorch at extreme logits; eight-step recurrence consumes predicted probability (including fractional values), not reference future bits. |
| Supervision isolation | Nonbinary/misaligned/reference-mismatched intent rejected; valid contact targets affect BCE; changing force/occupancy sidecars or `a_clean` in an otherwise validated training batch does not change the training loss. Raw archive integrity remains separately enforced. |
| Replay/metrics | Pair continuity, last endpoint, whole episodes/no reset crossing; all valid one-step rows; both classes, onset/release at the first/last possible window transition; absent-class/denominator handling; persistence/constant baselines and no double-counted unique events. Synthetic cases test arithmetic only. |
| Physical regression | Existing robot feature/loss/gradient parity, object integration/canonicalization, support covariance; physical prefix always84. Preserve existing float64 geometry tolerance `1e-9` and original robot regression tolerances. |
| Migration | Nonzero sentinel columns catch action/xi/support shifts; all 82 physical rows preserved. Old-joint/new physical predictions agree for contacts 0, 1 and 0.37 at one/eight steps (`atol=rtol=2e-6`); initial contact probability0.5; strict round-trip difference0; reject unknown/missing keys, invalid dimensions/scales/hash/support/NaN. |
| Framework | `JointWorldModelTrainer.update is WorldModelTrainer.update`; actual bounded fit records nonzero robot/object/contact parameter updates; original robot loss/gradient regression remains exact. No Actor training test or force-gradient experiment. |

## 6. Task 0 verification and limitations

Read-only checks used `.venv/bin/python` with `PYTHONDONTWRITEBYTECODE=1`, NumPy
`load(..., allow_pickle=False)`, `ReferenceClip`, `load_archive`, and
`torch.load(..., map_location='cpu', weights_only=True)`. The reference loader passed
all manifest/field checks. Existing collection loader passed archive metadata/hash
and saved raw-sensor reconstruction checks. Every saved intent pair matched the
refined reference at its recorded frame. Observed state/xi shapes were `[320,84]` /
`[320,33]`, with 32 episodes and 240/80 train/validation rows.

| Read-only result | Verified value |
|---|---|
| Reference class counts | OFF550, ON290; 839 valid pairs: `00=548`, `01=1`, `11=289`, `10=1` |
| Existing 320-row intent distribution | `[1,1]` exactly 320 times; no other pair |
| Proposed 32-row schedule arithmetic | Each 16-row split has `00=7`, `01=1`, `11=7`, `10=1`; this is label indexing, **not collected data** |
| Existing collection archive SHA256 | `663a332f61f0073345e7a6e0da9d851698b479f7c65be36b319400a790b0c4e1` |
| `reference_manifest.json` SHA256 | `5b375e66e52d7079363ddceae4abdc8b838f4e28f7fb5b1b28f09047fcd1a78e` |
| `reference_refined_manifest.json` SHA256 | `155c558be01b8318f4093fc595154cdc805bf730c38a38f78e3cf16c48fd016a` |
| `reference_manifest.before_schema_fix.json` SHA256 | `27efec2f4bb10e661382a8e83090dc1e459651bfc3f840e7af5fc74693f4cdbf` |

Repository inspection used `git branch --show-current`, `git rev-parse HEAD`,
`git status --short`, `git log --oneline -12`, `rg`, and source/report reads; upstream
main was checked using the GitHub connector without fetching or merging. Review
uses full diffs of these two new documents, link/mapping/count assertions, protected
hash comparison, and `git diff --cached --check` before explicitly staging/committing
only these documents. Documentation assertions passed: all 166 old input columns
map exactly once into 167 new columns with only column86 newly initialized; dimensions,
relative links, boundary counts and persistence baselines agree. All **513 protected
files are byte-for-byte unchanged**. Existing user changes remain unstaged.

Two preliminary read-only probes needed correction: ignored reference NPZ bytes
were added to the preservation snapshot, and FPS was read with `.item()` because
its shape is `(1,)`. Corrected checks passed. The normal execution sandbox failed
at startup (`bwrap: loopback: Failed RTM_NEWADDR`); read-only commands used the
available escalation mechanism. No automatic approval rejection occurred.

No pytest suite, simulator, fit, or model migration was executed in Task 0. The
historical P3 result of 102 passed / 17 skipped is prior evidence, not a new test
run. Task 0 establishes source/data compatibility and a reviewable revision;
revised P3 learned acceptance is **PARTIAL / not evaluated** until implementation,
approved boundary collection, and both validation gates are completed.
