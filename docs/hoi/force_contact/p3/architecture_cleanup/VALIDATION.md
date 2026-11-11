# P3 cleanup validation record

Commands ran from `/home/han/pyx/SMHOI` using the existing `.venv`; dependencies were
not installed or changed. Logs are under
`outputs/hoi/contact_supervision/architecture_cleanup/`. Ordinary sandbox execution
failed at namespace setup (`bwrap: Failed RTM_NEWADDR`); the same bounded local
commands ran through the approved escalation mechanism. This is not a test result.

## Targeted software tests

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -p no:cacheprovider -q tests/hoi/test_joint_architecture.py tests/hoi/test_joint_world_model.py tests/hoi/test_support_context.py tests/hoi/test_checkpoint_migration.py tests/hoi/test_wm_dataset_validation.py tests/hoi/test_contact_diagnostic.py tests/hoi/test_wm_reuse.py
```

First execution: exit1, 12 passed / 15 failed (`tests_01.log`), because the V3
document or refined-reference manifest was missing. After documented A1 and one
additional negative provenance test: exit0, **28 passed** (`tests_02.log`).

Coverage: 166/82/84/86/18 contracts; constant-velocity prior; object frames,
quaternions and COM/root-origin conversion; support covariance; synthetic eight-step
backward to actions, initial states and both physical output blocks; independent
termination BCE; original robot loss/metrics/gradients; strict final checkpoints;
named mapping and source equivalence; incompatible/nonfinite/metadata rejection;
archive integrity and physical-loss independence from clean actions, force and
intent labels; retired experiment entry points. No optimizer step occurs. Replay
episode splitting here checks archive/window integrity, not a new contact-balanced
dataset or predictive experiment.

## Real checkpoint migration probe

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/hoi/validate_wm_migration.py --robot-source /home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt --output outputs/hoi/contact_supervision/architecture_cleanup/migration_01
```

Exit0; machine-readable acceptance is **PARTIAL**, not implicit PASS from exit0.
See `migration_01.log` and `migration_01/results.json`. The script is intentionally
non-overwriting: reruns must choose a new output directory. The legacy force
source, collection_02 archive and explicit refined reference are existing evidence.
The model is not updated; no predictive-performance metric is accepted.

## Additional regression

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -p no:cacheprovider -q tests/hoi/test_force_aggregation.py tests/hoi/test_force_replay.py tests/hoi/test_contact_sensor.py tests/test_world_model.py::test_state_spec_identity_and_errors tests/test_imagination.py::test_gae_targets_match_hand_computation tests/test_g1_fk.py
```

Exit1; **28 passed / 1 failed** (`regression_01.log`). The unchanged P1 sensor
configuration test fails in the unchanged asset provenance loader because V3 is
absent. The other tests check synthetic force aggregation/Replay/sensor adapters,
robot state identity, GAE arithmetic and CPU/CUDA FK/gradient consistency. FK tests
call `mj_forward` on random robot states for kinematic comparison; they perform no
physics rollout or data collection. No test in these selected runs was skipped.

## Explicitly not executed

- Contact-boundary capture, contact-balanced sampling, Contact Head fitting,
  binary event evaluation and contact readiness gates: skipped by the new design.
- Joint WM training, optimizer steps and learned predictive validation: deferred
  to separately authorized P7-A. Synthetic backward is not an update.
- `test_force_recorder.py`, `test_robot_only_regression.py`, physical joint-scene
  reset and environment/analytic smoke suites: would run physical environments or
  collection; outside this authorization. Original robot software checks above ran.
- `test_world_model.py::test_world_model_overfits_replay_windows` and imagination
  training smoke: would optimize models. The full pytest suite was not run.
- Historical `verify_p3_evidence.py`: it recomputes a ridge fit and also requires
  missing provenance files. Saved results, AST equivalence and file hashes were
  checked instead. No historical result was regenerated or overwritten.
- Dormant legacy force-loss normalization tests and unrelated asset/browser/data
  suites: outside the active architecture change; no PASS is claimed for them.

## Static checks and preservation

`static_checks.json` records the exact lint/format argv and exit codes. Both checks
cover the 14 surviving modified Python files and two new Python files. Final source
checks are read-only (`ruff check --no-cache`, `ruff format --check --no-cache`).
Earlier import sorting used `ruff check --fix`; no unmodified file was formatted.

`baseline.json` records the starting HEAD, 534 existing file hashes and ten
dependency source hashes. `preservation.json` was produced by comparing all those
hashes with the working tree, asserting that the changed set exactly equals the
15 existing source/test paths listed in CHANGED_FILES.md, comparing the legacy gate
AST to `git show 7251bf33f95fb4e2ee75f45a306f66fbadac85d1:src/supermimic/hoi/validation.py`,
and checking the external checkpoint SHA256. All assertions passed. This compact
read-only recheck reproduces the hash/AST acceptance:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import ast
import hashlib
import json
from pathlib import Path
import subprocess

out = Path('outputs/hoi/contact_supervision/architecture_cleanup')
b = json.loads((out / 'baseline.json').read_text())
p = json.loads((out / 'preservation.json').read_text())
def sha(path):
    path = Path(path)
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if path.is_file() else None
changed = {k: {'before': v, 'after': sha(k)} for k, v in b['protected_sha256'].items() if sha(k) != v}
assert changed == p['intentional_changes']
assert len(b['protected_sha256']) == 534 and len(changed) == 15
assert {k: sha(k) for k in b['dependency_sha256']} == b['dependency_sha256']
names = {'ReadinessCriteria', 'legacy_force_readiness_gate'}
def definitions(source):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}
old = subprocess.check_output(['git', 'show', b['head'] + ':src/supermimic/hoi/validation.py'], text=True)
assert definitions(old) == definitions(Path('src/supermimic/hoi/legacy_validation.py').read_text())
assert sha('/home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt') == p['external_robot_checkpoint_sha256']
print('PASS: preservation and historical gate semantics')
PY
```

Before committing, inspect `git diff`, then the complete `git diff --cached`, with
explicit staging of only the inventory paths. `git diff --check` passes for source.
The final staged check excludes only raw `*.log` evidence; default checking of those
logs reports 50 trailing-space lines in pytest tracebacks (47 initial / 3 regression),
retained exactly as emitted. No source/document warning is accepted.

The user authorized one commit. After committing, inspect `git show --stat --oneline
HEAD`, `git diff HEAD` and `git status --short --branch`; only the pre-existing
untracked `.vscode/` should remain. Do not push or execute the next stage.
