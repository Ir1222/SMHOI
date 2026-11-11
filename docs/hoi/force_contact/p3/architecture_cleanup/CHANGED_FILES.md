# P3 cleanup file inventory

Paths are relative to the repository root. Historical outputs/docs outside the new
cleanup directories are untouched; `.vscode/` is excluded from the commit.

## Existing source and tests

| Path | Change |
|---|---|
| `src/supermimic/hoi/world_model.py` | Physical84 → residual82 model; remove contact state/head/loss; preserve termination |
| `src/supermimic/hoi/wm_trainer.py` | Physical batch adapter and reference t+1; remove contact targets/cache |
| `src/supermimic/hoi/state_spec.py` | Remove unused force dimension constant; physical math unchanged |
| `src/supermimic/hoi/support_context.py` | A1 exact manifest binding and narrowly scoped missing-planning-document handling |
| `src/supermimic/hoi/checkpoint.py` | Final v3 strict schema, named robot/legacy-force migration, incompatible-source rejection |
| `src/supermimic/hoi/validation.py` | Remove classifier/event/readiness paths; retain deferred physical helpers |
| `scripts/hoi/collect_p3_diagnostic.py` | Retire collection entry before side effects; pin historical source |
| `scripts/hoi/fit_p3_diagnostic.py` | Retire fitting entry before side effects; pin historical source |
| `scripts/hoi/p3_common.py` | Explicit hash-bound archived-reference adapter for software probes only |
| `scripts/hoi/validate_wm_migration.py` | Final physical checkpoint and equivalence report; no fitting |
| `scripts/hoi/verify_p3_evidence.py` | Import isolated historical gate; no other logic change |
| `tests/hoi/test_checkpoint_migration.py` | Final mapping/schema/strict negative checks |
| `tests/hoi/test_contact_diagnostic.py` | Assert experiment entries are retired |
| `tests/hoi/test_contact_intent_head.py` | Remove obsolete classifier tests; baseline Git preserves source |
| `tests/hoi/test_wm_dataset_validation.py` | Archive/physical-loss isolation and saved-history checks; remove actual update/evaluation |

## New source and tests

| Path | Purpose |
|---|---|
| `src/supermimic/hoi/legacy_validation.py` | Verbatim historical force-gate definitions, isolated from active acceptance |
| `tests/hoi/test_joint_architecture.py` | Final shapes, synthetic gradients, rejection checks and support provenance negatives |

## New independent documentation

All under `docs/hoi/force_contact/p3/architecture_cleanup/`:

- `IMPLEMENTATION_SPEC.md`: approved scope, requirements, tests and A1.
- `REPORT.md`: partial acceptance, implementation, evidence and remaining limits.
- `VALIDATION.md`: commands, exclusions and reproducible preservation check.
- `CHANGED_FILES.md`: this inventory.

## New evidence

All under `outputs/hoi/contact_supervision/architecture_cleanup/`:

- `baseline.json`: immutable starting file/dependency hashes.
- `tests_01.log`: initial missing-provenance failures, retained verbatim.
- `tests_02.log`: final 28-pass core run.
- `regression_01.log`: 28-pass / 1-fail regression run, retained verbatim.
- `migration_01.log`: migration command stdout/stderr.
- `migration_01/results.json`: mappings, numerical errors and explicit PARTIAL status.
- `migration_01/migrated_physical_weights.pt`: new final-schema initialization artifact;
  explicitly staged despite the checkpoint ignore rule; no old checkpoint overwritten.
- `preservation.json`: 519 unchanged baseline files, 15 intentional changes, ten
  unchanged dependency sources and historical-gate AST equality.
- `static_checks.json`: lint/format/diff commands and results.

No other path belongs to this commit.
