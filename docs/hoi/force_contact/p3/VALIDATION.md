# P3 validation commands and outcomes

Working directory: `/home/han/pyx/SMHOI`. Project interpreter `.venv/bin/python`.
All outputs below are relative to
`outputs/hoi/contact_supervision/force_model_validation/`. Runs use separate output
directories, created with `exist_ok=False`. Do not overwrite them to reproduce a run.
The normal command sandbox failed before process startup (`bwrap: loopback:
Failed RTM_NEWADDR`); execution used the available escalation mechanism. No approval
review rejection occurred. This is an execution-environment issue, not a test skip.

## Preparation and review

The complete master plan, mandatory P0/P1/P2 evidence and original source paths
listed in IMPLEMENTATION_SPEC were read before source edits. Initial checks included:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git log --oneline -12
find .. -name AGENTS.md
```

`baseline.json` records the exact current branch/HEAD/status, full protected-file
hash map, independently checked refined reference/field manifests, P1 reset cache,
P2 motion-frame compatibility, installed versions and ten dependency hashes.
`IMPLEMENTATION_SPEC.md` was committed first (`29a48fd`). No source code was changed
before that specification. No internet/dependency upgrade was needed.

## Targeted tests

The test entry points used during the reviewed groups are below; logs retain the
reported totals. Prefix physical tests with `MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1`.

```bash
.venv/bin/python -m pytest -q tests/hoi/test_joint_world_model.py tests/hoi/test_robot_only_regression.py
.venv/bin/python -m pytest -q tests/hoi/test_joint_world_model.py tests/hoi/test_support_context.py
.venv/bin/python -m pytest -q tests/hoi/test_joint_world_model.py tests/hoi/test_support_context.py tests/hoi/test_force_head.py tests/hoi/test_wm_reuse.py
.venv/bin/python -m pytest -q tests/hoi/test_wm_dataset_validation.py tests/hoi/test_force_replay.py
.venv/bin/python -m pytest -q tests/hoi/test_checkpoint_migration.py
```

| Log | Result | Meaning |
|---|---|---|
| `state_tests.log` | 4 passed | Robot prefix, frame invariance/round trip, MuJoCo integration/root-COM, residual derivatives, robot regression |
| `support_tests.log` | 4 passed | State tests and separate support-frame covariance |
| `model_tests_01.log` | 8 passed | Shared70/12/6 heads, force exclusion from state, robust balanced reversible loss, exact robot loss/gradient parity |
| `data_tests_01.log`, `data_tests_02.log` | 11 passed each | Complete archives/whole-episode split/original optimizer/metric gates plus P2 Replay regression; repeated after gate refinement |
| `checkpoint_tests_01.log` | 2 passed | Named input relocation, nonzero robot parity, mismatch rejection and strict round trip; final code also covered by full suite |

Numerical thresholds actually asserted: float64 invariance/roundtrip/integration
atol 1e-9; robot-prefix/default loss/metrics/gradient parity atol=rtol=0; force
normalization atol 2e-5 N/rtol 2e-6; legacy migration atol 2e-6/rtol 2e-6. Finite gradient
tests check implementation differentiation only, not simulator force-gradient fidelity.
Synthetic event arithmetic demonstrates correct metric code, not physical onsets.

## Simulator diagnostics and measured supervised fit

Core invocations (each output captured in the matching log):

```bash
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/diagnose_support_context.py --output outputs/hoi/contact_supervision/force_model_validation/support_01
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/diagnose_support_context.py --output outputs/hoi/contact_supervision/force_model_validation/support_02
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/collect_p3_diagnostic.py --output outputs/hoi/contact_supervision/force_model_validation/collection_01
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/collect_p3_diagnostic.py --output outputs/hoi/contact_supervision/force_model_validation/collection_02
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/fit_p3_diagnostic.py --source outputs/hoi/contact_supervision/force_model_validation/collection_02 --output outputs/hoi/contact_supervision/force_model_validation/fit_01
```

- Support 01: API error before simulator creation, missing mandatory manifest path.
  Fixed explicitly to DEFAULT_ASSET; output log retained.
- Support 02:64 physical measured transitions with raw substeps/state/action/xi.
  Initial modulo4 validation pairs were side-biased. Saved-data reanalysis uses
  the last two pairs per group 8, independently reproduced by the final verifier.
  No recollection or old metadata rewrite. Linear probe reduces object error but
  worsens force MAE; all values are in SUPPORT_CONTEXT.md and the numeric sidecar.
- Collection01: original16-step fixture attempt failed the additional final-forward
  assertion at episode0/step 11, leaving48 rows. P2 substep capacity checks passed;
  exact final-forward flag bit was not logged by that version. No claim of its cause.
- Collection02: narrowed to 10 steps per fixture, 320 rows, all checks passed. This
  bounds a diagnostic before the known failure, not a physical stability fix.
  Both captures and failures are retained; the loader refuses Collection01.
- Fit01:200 real CPU optimizer updates through original WM trainer, no policy
  optimization. Hidden64×3; seed304; batch 32/window 8; 24 train/8 validation episodes.
  No force/xi labels fabricated. Final metrics, train-only scales, gradients,
  parameter changes and predictions are saved in results.json/NPZ/PT.
- Force readiness remains false. No natural HOI training or lengthy run launched.

The final current scripts include the corrected API path, balanced pair split and
10-step bound. The earlier failed logs correspond to the earlier source versions;
rerunning current code does not reproduce those bugs. The fit results preserve its
original gate output; `readiness_review.json` contains the subsequently tightened
finite-horizon/coverage/threshold gate, independently re-evaluated without learning.

## Checkpoint probe

```bash
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/validate_wm_migration.py --source /home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt --data outputs/hoi/contact_supervision/force_model_validation/collection_02 --fit outputs/hoi/contact_supervision/force_model_validation/fit_01 --output outputs/hoi/contact_supervision/force_model_validation/migration_01
```

PASS: real 2048×3 robot prediction max difference1.9073486328125e-6, new force zero;
strict fitted diagnostic save/load difference0. Source SHA unchanged. The external
candidate has qualified provenance; target-local candidate remains missing.
`torch.load(weights_only=True)` uses an explicit PosixPath safe allowlist for the
legacy config. No optimizer/Actor/full-lifecycle migration or resume was attempted.

## Complete regression

Executed twice, with outputs `tests_full_01.log` and `tests_full_02.log`:

```bash
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -ra tests > outputs/hoi/contact_supervision/force_model_validation/tests_full_02.log 2>&1
```

First run:1 failed, 101 passed, 17 skipped. The historical M6 integrity test pinned the
pre-P3 shared WM source. Corrected only its source-version expectation: old hash
asserted, reviewed new hash pinned, all other historical hashes still enforced.
No historical evidence was changed. Final run: **102 passed, 17 skipped**, 17.07s,
one pre-existing Torch JIT deprecation warning. Exact skips:

- 4 `test_analytic_heads.py`: LAFAN1 NPZ not built.
- 6 `test_env_smoke.py`: LAFAN1 NPZ not built.
- 4 `test_motion_convert.py`: LAFAN1 G1 CSVs not downloaded.
- 1 each in imagination, motion-library and world-model tests: CUDA+LAFAN1 NPZ required;
  CUDA is available for P3 physics, but the reference training assets are missing.

## Final read-only evidence verification and diff checks

```bash
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/verify_p3_evidence.py --output outputs/hoi/contact_supervision/force_model_validation/integrity_final.json > outputs/hoi/contact_supervision/force_model_validation/integrity_02.log 2>&1
```

The first verifier run (`integrity_01.log`) stopped on an absent `archive_sha256`
key in the early support sidecar. The support archive's hash was checked against
its 7a2faa4 committed bytes and pinned explicitly, preserving the sidecar. Final
verification PASS:452 baseline files checked, 449 unchanged, exactly three intended
existing-file changes; ten dependency hashes and all recorded versions unchanged.
Reference/hash manifests, saved raw-force aggregation, failed-archive rejection,
240/80 rows, 72/24 windows, no episode overlap, nonzero updates, balanced support
comparison, final NOT_READY gate and unchanged external checkpoint all verified.
No new simulator or training run is part of this verification.

Scoped Ruff check/format checks passed for all 20 new P3 Python files. Existing
shared/HOI files retain their original formatting. Complete group/final diff review
uses `git diff`, explicit paths, `git diff --cached --check` and changed-file
inventories; binary artifacts are checked by hashes and actual load/array contracts.
The raw first full-suite failure log has two pytest-generated trailing-space lines;
only that file is excluded from whitespace checking:

```bash
git diff a991ee7 --check -- . ':(exclude)docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md' ':(exclude)outputs/hoi/contact_supervision/force_model_validation/tests_full_01.log'
git diff --cached --check -- . ':(exclude)outputs/hoi/contact_supervision/force_model_validation/tests_full_01.log'
git diff --cached --name-only
git diff --cached --stat
git status --short
```

REPORT.md provides the per-group commits and acceptance. Final evidence/report commit
is authorized by the user's explicit instruction; final hash/status are reported
in the response. No reset, clean, push or next-stage work.

The exact scoped style-check driver (output saved as `style_checks.log`) was:

```python
import subprocess
files = subprocess.check_output(
    ['git', 'diff', 'a991ee7', 'HEAD', '--diff-filter=A', '--name-only'], text=True
).splitlines()
files = [p for p in files if p.endswith('.py')] + ['scripts/hoi/verify_p3_evidence.py']
print('New P3 Python files:', len(files), flush=True)
print('\n'.join(files), flush=True)
subprocess.run(['.venv/bin/ruff', 'check', *files], check=True)
subprocess.run(['.venv/bin/ruff', 'format', '--check', *files], check=True)
```

This driver ran before the final evidence commit, when the verifier was untracked;
for later reproduction use the20 explicit paths in `style_checks.log` to avoid
counting that file twice.
