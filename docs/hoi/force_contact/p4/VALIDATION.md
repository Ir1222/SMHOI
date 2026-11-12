# P4 validation record

Working directory `/home/han/pyx/SMHOI`; baseline `main` / `f191f6d`.
Commands ran with the existing `.venv` (no package installation). The shell sandbox
could not initialize its loopback device, so command execution used reviewed
escalation. No automatic approval rejection occurred.

## Executed commands and outcomes

```bash
.venv/bin/python scripts/hoi/audit_proximity_reward.py --output outputs/hoi/contact_supervision/reward_validation/geometry_01 > outputs/hoi/contact_supervision/reward_validation/geometry_01.log 2>&1

.venv/bin/python -m pytest -q tests/hoi/test_geometric_contact_reward.py tests/hoi/test_contact_target_audit.py > outputs/hoi/contact_supervision/reward_validation/tests_01.log 2>&1

.venv/bin/python scripts/hoi/measure_proximity_scale.py --geometry outputs/hoi/contact_supervision/reward_validation/geometry_01/geometry.json --output outputs/hoi/contact_supervision/reward_validation/scales_01 > outputs/hoi/contact_supervision/reward_validation/scales_01.log 2>&1

.venv/bin/python -m pytest -q tests/hoi/test_geometric_contact_reward.py tests/hoi/test_contact_target_audit.py tests/test_g1_fk.py tests/hoi/test_joint_architecture.py tests/hoi/test_joint_world_model.py tests/hoi/test_force_aggregation.py tests/hoi/test_force_replay.py tests/hoi/test_robot_only_regression.py > outputs/hoi/contact_supervision/reward_validation/tests_02.log 2>&1

.venv/bin/ruff check src/supermimic/hoi/rewards.py scripts/hoi/audit_proximity_reward.py scripts/hoi/measure_proximity_scale.py tests/hoi/test_geometric_contact_reward.py tests/hoi/test_contact_target_audit.py
.venv/bin/ruff format --check src/supermimic/hoi/rewards.py scripts/hoi/audit_proximity_reward.py scripts/hoi/measure_proximity_scale.py tests/hoi/test_geometric_contact_reward.py tests/hoi/test_contact_target_audit.py
git diff --cached --check
git diff --cached --stat
git diff --cached
```

| Check | Result |
|---|---|
| Original Part B geometry audit, 840 frames | PASS; exact immutable archive/manifest identities |
| Initial P4 tests | 15 passed, 0 failed, 0 skipped; 4.35 s |
| Reference-state scale probes | PASS; three data-derived sigma candidates, zero optimizer steps |
| Final selected regression | 51 passed, 0 failed, 0 skipped; 6.18 s |
| Lint and formatting | PASS; five Python files |
| Existing files and dependencies | 548 baseline files and ten P0 dependency-source hashes unchanged |

The final suite includes the 15 new tests; do not sum initial and final counts.
The only pytest warning is the pre-existing TorchScript deprecation. Initial lint
found import ordering/unused-import issues (automatically fixed) and one `dict()`
style issue (manually corrected); no numerical test failed. Formatting-only edits
preceded the final 51-test run. The final diff check then identified CSV writer
CRLF endings as whitespace; the new CSV and its writer were changed to LF, with
all 840 parsed rows verified identical. This output-format-only correction does
not change the tested geometry/reward computations.

The original robot-only environment regression used the available HDMI robot clip,
four worlds and 16 control steps, and checked observation930/goal680/action29,
state71, context33 and eight-step Replay. No HOI physical rollout was performed.
Existing P3 tests perform synthetic backwards with optimizer stepping forbidden.

Not run: complete repository suite (unrelated missing LAFAN1 inputs), P1 strict
scene reconstruction (known missing historical V3 source), physical contact/force
calibration reruns, P3 checkpoint migration reruns, P5 compiled imagination,
P6 contact agreement, Actor/Critic/WM training or P8 full task evaluation. None is
reported as a passing P4 physical outcome. Prior failures are preserved by hash.

## Preservation reproduction

The initial baseline records hashes of all tracked files plus existing files under
`outputs`, `docs/hoi`, `data/assets`, `.vscode` and `logs` (excluding Python caches).
It was captured before the first P4 source edit. To repeat the final check:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from supermimic.hoi.reference import sha256
root = Path('outputs/hoi/contact_supervision/reward_validation')
baseline = json.loads((root / 'baseline.json').read_text())
for path, expected in baseline['files'].items():
    assert Path(path).is_file() and sha256(path) == expected, path
dependencies = json.loads(Path('docs/hoi/force_contact/p0/dependencies.json').read_text())
for path, expected in dependencies['inspected_file_sha256'].items():
    assert sha256(path) == expected, path
print(len(baseline['files']), 'baseline files and', len(dependencies['inspected_file_sha256']), 'dependency sources unchanged')
PY
```

Full new-source/test/docs diff is reviewed through the Git index. JSON/CSV/log
evidence is inspected for schemas, row counts, finite values, numeric conclusions,
hash provenance and agreement with these reports. No previous report or raw
failure evidence is rewritten.
