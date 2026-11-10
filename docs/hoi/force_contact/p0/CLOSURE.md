# Final P0 closure report

**Stage:** P0 — repository, dataset, asset, dependency, architecture and reference-code audit. **Decision: PASS for review.** No P1 work was started. Current closure changes are uncommitted and have not been pushed.

## Source of truth and historical reconciliation

The complete 1,743-line `.txt` master was read in full and restored byte-for-byte at [the canonical path](../SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md). Both have SHA256 `7365f680cf975263e69672d75d585c62e83532115b9065c6e63929004bdbdc9b`. The incomplete user `.md` neighbor remains untouched. The previous missing-master blocker is resolved.

Current local branch/HEAD: `main` / `e89573171c86be7f40c8d120dc813cee0274a152`. Audit commit `579d1a8c661985c91926a365c7505488abe03bb8` and reconciliation commit `e895731` remain in history. Both were explicitly authorized; no amend/reset/rebase/duplicate commit occurred. The previous GitHub observation at `579d1a8` remains a dated observation in `reconciliation_checks.json`, not a claim about the current remote tip. No new remote-history claim is required for this local closure.

Original `artifact_inventory.json`, `preservation_final.json`, `repositories.json`, test logs and `reconciliation_checks.json` are historical evidence and remain byte-identical. Their earlier missing-plan / pre-commit fields are superseded by this report and `closure_checks.json`, not retroactively rewritten.

## Gaps closed without repeating completed audits

| Gap revealed by complete plan | Work completed | Evidence |
|---|---|---|
| P0.5 source implementation audit | Read actual HDMI, PhysHSI local fork and SUGAR mechanisms/config/label paths; classify reuse/adaptation/incompatibilities | `REFERENCE_IMPLEMENTATION_AUDIT.md`, `reference_implementation_checks.json` |
| Workhorse targets and deployment-error augmentation | Read paper IV-A–IV-D and official project; code unreleased | User explicitly approved paper/available-repository fallback; limitation retained |
| P0.4 force/torque frame specificity | Confirm world axes and separate force-weighted centroid torque per primary; document common-point wrench combination | Dependency audit addendum and coordinate JSON |
| Sections 4.3–4.4 numeric conventions not independently demonstrated | Five isolated origin/integration/velocity/vector/torque check groups | `coordinate_closure_checks.json`, exact probe in `CLOSURE_CHECKS.md` |
| P0.6 full stage map and prescribed output locations | Reconcile all P1–P8 boundaries, reuse, minimal hooks, new files, tests, dependencies and risks | `FILE_CHANGE_MAP.md` |
| Complete acceptance/coverage report | Map all P0 tasks and eight explicit acceptance criteria | `REQUIREMENT_COVERAGE.md` |

The additional secondary-policy, overflow, reset and instantaneous-overwrite bullets were already covered in the original dependency audit. No duplicate physics/history/sign probe was needed. Dataset schemas, accepted field comparisons, strict checkpoint loading, baseline tests, M0–M6.5 and Part B processing were not rerun. Hash checks are preservation/evidence-reuse checks, not dataset regeneration or physical experiments.

## Tensor, frame, unit and timing consequences

No runtime tensor changed. Baseline state71/canonical71/action29/xi33/residual70 remains verified. Target joint84/canonical86/physical82/force6 is mapped to later stages. Force6 is never integrated.

Existing force history is per primary body and newest-first. Two primary bodies per hand must be summed; the action spans four 5 ms substeps. Mean force is N, impulse is N·s, torque is N·m. World force is hand-on-box for hand-primary/box-secondary. Box components use inverse Box rotation. Torque rotation preserves its reference point; primary torques must first refer to a common point if summed. The preferred contemporaneous Box orientation and allowed explicit interval anchor are P2 decisions under the now-complete master.

Root translation is environment-local in stored robot state; root linear velocity is world-frame/root-origin, angular velocity body-local, and quaternion storage is wxyz. Existing offset-COM evidence is reused. New numeric checks verify right multiplication of body angular increments and signed quaternion-log reference velocity conventions. Synthetic convention checks do not implement joint scene resets or derive a reference velocity cache.

## Checks, numerical evidence and skipped work

| Executed check | Result |
|---|---|
| StateIO origins/read-write on two synthetic rows | PASS; position error `1.67e-16` m; selected-row write/clearing only |
| Existing StateSpec versus MuJoCo noncommuting quaternion integration | PASS; max component error `1.11e-16`; wrong left product differs `2.28e-3` |
| Synthetic reference pose finite differences, central/one-sided and sign flips | PASS; linear error `2.61e-15` m/s; angular error ≤`2.45e-14` rad/s |
| Box-local force/torque and inverse round trip | PASS; error ≤`8.89e-16` in vector units |
| Common-reference torque algebra | PASS; `5.56e-17` N·m error; naive torque sum differs `0.59439` N·m |
| Reference source hash/AST inspection | PASS; nine exact files recorded; no Isaac framework execution |
| Canonical master equality / Git ancestry / JSON / links / diff / preservation | PASS; detailed counts and hashes in `closure_checks.json` |

The coordinate command emitted one existing Torch `torch.jit.script` deprecation warning. No mandatory audit check failed. Existing seven non-training tests remain PASS from their original execution; they were not rerun after documentation-only changes. Source/dependency hashes were checked for evidence reuse. No P1 physical calibration, support-context prediction ablation, Force Head fit, Actor gradient fidelity, migration, training or full task evaluation was run; these require later-stage approval.

Exact substantive commands, inspected paths and the full temporary coordinate probe are recorded in [CLOSURE_CHECKS.md](CLOSURE_CHECKS.md). The full review patch is `/tmp/SMHOI_P0_FINAL_CLOSURE_REVIEW.patch`.

## Every new and modified file

New files:

- `docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md` — exact copy of supplied complete text.
- `docs/hoi/force_contact/p0/REFERENCE_IMPLEMENTATION_AUDIT.md` — P0.5 mechanisms and reuse decisions.
- `docs/hoi/force_contact/p0/reference_implementation_checks.json` — paths/hashes/revisions/symbol ranges and approved fallback.
- `docs/hoi/force_contact/p0/coordinate_closure_checks.json` — five missing numeric check groups.
- `docs/hoi/force_contact/p0/REQUIREMENT_COVERAGE.md` — complete P0 requirement matrix.
- `docs/hoi/force_contact/p0/CLOSURE_CHECKS.md` — commands and exact temporary probe.
- `docs/hoi/force_contact/p0/closure_checks.json` — current closure/preservation evidence.

Modified files:

- `docs/hoi/force_contact/p0/REPORT.md` — final P0 decision, source of truth and summary.
- `docs/hoi/force_contact/p0/CLOSURE.md` — this final closure report; prior review remains in commit `e895731`.
- `docs/hoi/force_contact/p0/FILE_CHANGE_MAP.md` — complete P1–P8 map.
- `docs/hoi/force_contact/p0/ARCHITECTURE_AUDIT.md` — complete-plan boundaries and new coordinate evidence.
- `docs/hoi/force_contact/p0/DEPENDENCY_CONTACT_AUDIT.md` — allowed Box-frame approaches and torque reference.
- `docs/hoi/force_contact/p0/VALIDATION.md` — current closure addendum, preserving original test matrix.

Git scope: six modified tracked documentation files and seven new documentation/evidence files. Index remains empty; four pre-existing untracked user files are excluded. No training source, tests, datasets, assets, checkpoints, dependencies or accepted Part B file changed.

## Limitations, approved exception and stop point

Workhorse code is unavailable; the user explicitly authorized paper/other-repository inspection. PhysHSI findings describe the actual local fork, with no claim of official-code parity. Neither source's algorithms/assets replace the master specification.

Historical M6.5 physical-hand-contact and support-validity FAILs, `training_ready=false`, target-local checkpoint absence and limited external checkpoint provenance remain recorded. An audit PASS establishes that these facts and required changes are understood; it does not resolve them experimentally or approve learning.

All eight P0 acceptance criteria are covered, with the explicit Workhorse fallback. **P0 may be marked PASS for review. Wait for user approval. No P1 was started, and no commit or push was made.**
