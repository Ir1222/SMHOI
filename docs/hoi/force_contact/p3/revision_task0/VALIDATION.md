# Revised P3 validation record

Working directory `/home/han/pyx/SMHOI`; interpreter `.venv/bin/python`.
Outputs below are relative to
`outputs/hoi/contact_supervision/contact_intent_model_validation/`.
Core implementation/unit checks PASS; collection FAIL; Gate A/B NOT EVALUATED;
overall P3 PARTIAL. See [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md).

## Executed commands and results

The normal command sandbox failed before execution with
`bwrap: loopback: Failed RTM_NEWADDR`. Commands used the available escalation
mechanism; there was no automatic approval rejection. No experiment was enlarged.

Repository checks ran first: ancestor/repository `AGENTS.md` search (none found),
`git status --short`, `git branch --show-current`, `git log -5 --oneline`.
Approved Task 0 documents were read; preparation/design was not repeated or rewritten.
`baseline.json` pins HEAD/status, 515 protected existing files, reference SHA and
installed versions: torch 2.14.1+cu130, NumPy 2.5.4, MuJoCo/MuJoCo-Warp 3.16.0,
Warp 1.18.0. ReferenceClip validated the accepted 840-frame/50 Hz refined reference.

All pytest/probe/collector commands used this prefix:

```bash
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python
```

| Exact arguments after the prefix | Output log | Executed outcome |
|---|---|---|
| `-m pytest -q tests/hoi/test_contact_intent_head.py tests/hoi/test_wm_dataset_validation.py tests/hoi/test_force_head.py tests/hoi/test_joint_world_model.py tests/hoi/test_support_context.py tests/hoi/test_wm_reuse.py tests/hoi/test_robot_only_regression.py` | `group1_tests.log` | 17 passed, 12.25 s |
| `-m pytest -q tests/hoi/test_checkpoint_migration.py` | `migration_tests_01.log` | 11 passed, 3.49 s |
| `scripts/hoi/validate_wm_migration.py --robot-source /home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt --output outputs/hoi/contact_supervision/contact_intent_model_validation/migration_01` | `migration_probe_01.log` | Failed external robot FP32 multi-step assertion; earlier successful joint probe/initialization retained |
| Same migration command with `--output outputs/hoi/contact_supervision/contact_intent_model_validation/migration_02` | `migration_probe_02.log` | Joint FP32 parity 0; strict round trip 0; external robot FP32 PARTIAL / FP64 PASS, separately recorded |
| `-m pytest -q tests/hoi/test_wm_dataset_validation.py tests/hoi/test_contact_intent_head.py` | `metrics_tests_01.log` | 13 passed, 4.10 s |
| `scripts/hoi/verify_p3_evidence.py --output outputs/hoi/contact_supervision/contact_intent_model_validation/historical_force_integrity.json` | `historical_force_integrity.log` | PASS; old force readiness remains false with original semantics |
| `-m pytest -q -ra tests` | `tests_full_precollection_01.log` | **123 passed, 17 skipped**, 17.01 s |
| `-m pytest -q tests/hoi/test_wm_dataset_validation.py tests/hoi/test_contact_diagnostic.py` | `metrics_tests_02.log` | 8 passed, 3.74 s after final event-reporting/input-validation review fixes |
| `scripts/hoi/collect_p3_diagnostic.py --contact-boundaries --output outputs/hoi/contact_supervision/contact_intent_model_validation/collection_01` | `collection_01.log` | FAIL at frame 257→258, overflow 512; one complete transition retained; no retry |

Each command's stdout/stderr was redirected to its listed log; exit codes were
preserved. All pytest runs emitted one pre-existing Torch JIT deprecation warning.
The 17 skips were 4 analytic-head tests, 6 environment-smoke tests, 4 motion-conversion
tests and one each in imagination, motion library and world model, all requiring
missing LAFAN1 NPZ/CSV assets (some also require CUDA).

The full suite ran before collection. The only subsequent source review changes
added one-step event summaries and rejection of nonfinite event inputs; their eight
relevant tests passed before collection. No source change or further physical run
was made after the failed capture.

Scoped Ruff check/format checks passed for the 14 changed/new Python files. Full
group diffs were inspected before each commit; `git diff --cached --check` and
explicit staged-path inventories excluded user modifications. Final read-only
integrity verification compares 515 baseline hashes, verifies the exact 12 intended
existing-file changes, unchanged dependencies/reference/history, new artifact hashes
and documentation links. It does not recollect, retrain or reinterpret old results.

## Numerical checks and missing learned metrics

Executed deterministic assertions include exact original robot loss/gradient
parity; geometry/support tolerances inherited at 1e-9 in float64; physical integration
limited to 82 outputs; contact/BCE dimensions 167/83/85; fractional autoregressive
feedback with live gradients; BCE agreement including logits ±100; canonical label
alignment and pair continuity; force/occupancy loss independence; exact baseline
denominators/undefined precision; first/last-window event detection; all one-step
row coverage; whole-episode isolation; strict metadata/tensor corruption rejection.
These are implementation tests, not learned performance results.

Actual old joint checkpoint predictions: maximum absolute difference 0 at every
horizon 1–8 for each initial contact 0, 1, 0.37. New contact probability 0.5, physical
head weights unchanged, strict save/load difference 0. External 2048 robot FP32
eight-step max difference 2.288818359375e-5 exceeds the unchanged approved tolerance;
FP64 probe difference 0. The original robot checkpoint hash remains
`16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7`.

Capture evidence contains one measured row with `state/next_state [1,84]`, actions
`[1,29]`, xi `[1,33]`, intent `[1,2]=[[0,0]]`, raw physical samples `[1,4,2,2,3]`,
and original terminal masks. Its reference pair is 256→257. Failed interval 257→258
has two captured substeps and overflow 512; the installed enum identifies a solver
iteration limit. Saved complete-row sensor reconstruction and label verification
passed. The failed archive is refused as training input.

No bounded fit was executed. Robot/object learned NSE/RMSE, contact accuracy,
precision/recall/F1, onset/release detection and eight-step probability/BCE on the
approved holdout therefore have **no measured values**. Neither constant-velocity
nor frozen-model predictive performance was evaluated on a completed new holdout.
No always-OFF/ON/persistence numbers from synthetic tests are presented as learned
validation. The proposed balanced schedule's baseline 0.875 remains an arithmetic
expectation, not an observed dataset result. The implemented fit/evaluator will
record all these quantities and denominators if separately authorized collection
later succeeds; no such run is authorized by this report.

## Complete new artifact inventory

All paths in this list have the output prefix specified above:

```text
baseline.json
group1_tests.log
migration_tests_01.log
migration_probe_01.log
migration_probe_02.log
migration_01/migrated_contact_weights.pt
migration_02/migrated_contact_weights.pt
migration_02/results.json
metrics_tests_01.log
metrics_tests_02.log
tests_full_precollection_01.log
historical_force_integrity.json
historical_force_integrity.log
collection_01.log
collection_01/metadata.json
collection_01/transitions.npz
collection_01/failure_snapshot.npz
collection_failure_review.json
integrity_final.json
```

`collection_01/transitions.npz` SHA256:
`6e1a030578f4118e1f7965b805ad3a65983d8d137b69791415dbf575a3b0d051`.
Failure snapshot SHA256:
`2205a5222e935d43ab3bd542fe2614c96e226e52f5044c6ecd141a6112c4b6ae`.
The migration checkpoints are untrained-contact initialization artifacts; there is
no `diagnostic_contact_weights.pt` or new fit directory. All historical checkpoints,
P0/P1/P2/P3 evidence, ignored refined-reference bytes/manifests and pre-existing
user changes are preserved. Complete changed-file names/hashes are recorded in
`integrity_final.json`; its own path is listed without a self-referential hash.
