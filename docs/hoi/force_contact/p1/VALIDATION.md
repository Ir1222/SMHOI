# P1 validation commands and review record

Working directory: `/home/han/pyx/SMHOI`. Commands used the existing `.venv`; no
installation or dependency patch occurred. The sandbox launcher failed before
executing commands (`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`).
Execution then used tool-approved escalation. No automatic approval rejection
occurred. Warp compilation cache was `/tmp/smhoi_p1_warp_cache`; bytecode writing
was disabled for substantive Python runs.

## Baseline and implementation review

```bash
git status --short
git log -5 --oneline
git branch --show-current
git show --stat b5c604a
git diff -- docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md
```

Read the complete master in chunks, all mandatory P0 reports, supporting dependency
and coordinate evidence, existing manifest/reference/state/action/environment
code, and relevant installed Scene/Entity/ContactSensor/Simulation/step implementations.
The ten P0 dependency source hashes were compared before source edits. The baseline
snapshot hashes 126 protected source, asset, reference, historical evidence and
user-owned files. It records initial branch, HEAD, status and installed versions.

The first `git commit -m 'docs(hoi P1): specify physical scene and sensing implementation'`
failed because author identity was absent. After the user supplied their identity:

```bash
git config --get user.name
git config --get user.email
git var GIT_AUTHOR_IDENT
git var GIT_COMMITTER_IDENT
git add docs/hoi/force_contact/p1/IMPLEMENTATION_SPEC.md docs/hoi/force_contact/p1/baseline.json
git diff --cached --check
git diff --cached -- docs/hoi/force_contact/p1/IMPLEMENTATION_SPEC.md
git diff --cached --numstat
git commit -m 'docs(hoi P1): specify physical scene and sensing implementation'
```

Both identities verified `Ir1222 <Yiqian.Pan23@student.xjtlu.edu.cn>`;
result `f08b52b`. No other identity was configured or used.

## Executed tests

```bash
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/hoi/test_joint_scene.py tests/hoi/test_contact_sensor.py
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/hoi/test_joint_scene.py tests/hoi/test_contact_sensor.py > outputs/hoi/contact_supervision/sensor_calibration/tests_02.log 2>&1
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests > outputs/hoi/contact_supervision/sensor_calibration/tests_full_01.log 2>&1
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -ra tests > outputs/hoi/contact_supervision/sensor_calibration/tests_full_02.log 2>&1
```

First command: 5 passed / 1 failed (test fixture dtype mismatch). Second: 6 passed.
Both full suites: 69 passed / 17 skipped; final full run includes the actual static
geom cache correction, geom/body pose assertions and added robot-only regression.
One existing `torch.jit.script` deprecation warning. The final log enumerates every
skip; all require absent LAFAN1 NPZ/CSV inputs. No P1 CUDA test was skipped.

## Executed physical calibration

Each output directory was new; prior evidence was not overwritten.

```bash
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/calibrate_contacts.py --output outputs/hoi/contact_supervision/sensor_calibration/run_01 > outputs/hoi/contact_supervision/sensor_calibration/run_01.log 2>&1
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/calibrate_contacts.py --output outputs/hoi/contact_supervision/sensor_calibration/run_02 > outputs/hoi/contact_supervision/sensor_calibration/run_02.log 2>&1
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/calibrate_contacts.py --output outputs/hoi/contact_supervision/sensor_calibration/run_03 > outputs/hoi/contact_supervision/sensor_calibration/run_03.log 2>&1
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/calibrate_contacts.py --output outputs/hoi/contact_supervision/sensor_calibration/run_04 > outputs/hoi/contact_supervision/sensor_calibration/run_04.log 2>&1
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/calibrate_contacts.py --output outputs/hoi/contact_supervision/sensor_calibration/run_05 > outputs/hoi/contact_supervision/sensor_calibration/run_05.log 2>&1
```

Run_01–04 exit1; their causes/corrections are in `REPORT.md`. Run_05 exit0 and all
ten groups PASS. Overflow warnings in these logs belong to the **deliberately
undersized maxmatch1 negative test**. Normal samples explicitly recorded zero flags;
quiet logs were not used as evidence of capacity validity.

Targeted exploratory commands (before or between the retained runs):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/smhoi_p1_placements.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/smhoi_p1_bilateral.py
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/smhoi_p1_exclusion_probe.py > /tmp/smhoi_p1_exclusion_probe.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/smhoi_p1_coverage_probe.py
```

These located actual collider placements, reference frame260 bilateral contact,
the body-vs-static-geom pose discrepancy, and in-range wrist poses that expose the
pitch collider. They were diagnostic exploration, not acceptance evidence. Their
relevant operations are now reproducible in the committed calibration script.
An initial four-world inline smoke command exercised `make_physical_env`,
`JointStateIO.reset_to_reference` at frames `[0,300,500,839]`, COM velocity and one
native `env.step`; these checks are retained in `test_joint_scene.py`.

## Final static, integrity and diff checks

```bash
.venv/bin/ruff check src/supermimic/hoi/scene.py src/supermimic/hoi/state.py src/supermimic/hoi/contact.py scripts/hoi/calibrate_contacts.py tests/hoi
.venv/bin/ruff format --check src/supermimic/hoi/scene.py src/supermimic/hoi/state.py src/supermimic/hoi/contact.py scripts/hoi/calibrate_contacts.py tests/hoi
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/smhoi_p1_verify.py
git diff --cached -- src/supermimic/hoi/scene.py src/supermimic/hoi/state.py src/supermimic/hoi/contact.py
git diff --cached -- scripts/hoi/calibrate_contacts.py
git diff --cached -- tests/hoi
git diff --check
git diff --cached --check
```

Lint PASS; seven files formatted. `validation_summary.json` records source and
evidence SHA256, preservation comparison, all final numerical checks, author and
review HEAD. The integrity command checks every nested numeric value in final raw
evidence, all five JSONs, every target pair in all four worlds, all normal overflow
bits, and all protected/dependency hashes. Final physical evidence has 45 raw
samples and 112 target-pair/world checks. No rerun of completed P0 audits occurred.

The complete staged patch is reviewed in source/test/documentation chunks and raw
machine evidence is validated structurally and numerically. The existing master
typo correction is inspected but excluded. After staging only the listed P1 paths,
the authorized implementation commit uses:

```bash
git commit -m 'hoi: add joint scene and filtered hand-box sensing'
git log -2 --format='%h %an <%ae> %s'
git status --short
```

No push or next-stage command is part of this procedure.
