# P2 executed commands and review record

Working directory `/home/han/pyx/SMHOI`; existing `.venv`, no installation or
dependency edits. Sandbox launch failed with `bwrap: loopback: Failed RTM_NEWADDR`;
subsequent commands used tool-approved escalation. No automatic-review rejection.
No external messages, push, training or optimizer updates were requested/run.

## Preparation

```bash
pwd
git status --short
git rev-parse HEAD
git log -8 --oneline
git show --stat HEAD
git diff -- docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md
rg --files -g AGENTS.md /home/han /tmp
```

No applicable AGENTS.md was found. Read the complete master in line-range chunks,
all required P0/P1 Markdown and supporting JSONs; read actual P1 contact/scene/state,
original Replay/recorder/action/SimContext, installed sensor history, manager hooks
and environment stepping source. Dependency implementation was inspected locally;
no web source or changed dependency was needed.

```bash
.venv/bin/python /tmp/p2_baseline.py
```

This snapshot command hashes current tracked files and user-owned untracked files,
checks all 126 P1 protected hashes and ten P0 dependency hashes, records package
versions and Git state, and writes `baseline.json`. An additional inline Python
check compares every `implementation_sha256` and `evidence_sha256` entry in P1's
validation summary: all matched before edits. Final checks preserve all except the
intentional contact.py extension, with its six original function ASTs identical.

```bash
git add docs/hoi/force_contact/p2/IMPLEMENTATION_SPEC.md docs/hoi/force_contact/p2/baseline.json
git diff --cached --check
git commit -m 'docs(hoi P2): specify interval capture and replay adapter'
```

Result `bb6cb0c`. Commit identity remains the configured `Ir1222
<Yiqian.Pan23@student.xjtlu.edu.cn>`; no identity changes.

## Test and physical diagnostic commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/hoi/test_force_aggregation.py tests/hoi/test_force_replay.py > outputs/hoi/contact_supervision/replay_diagnostics/tests_cpu_01.log 2>&1
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/diagnose_force_replay.py --output outputs/hoi/contact_supervision/replay_diagnostics/run_01 > outputs/hoi/contact_supervision/replay_diagnostics/run_01.log 2>&1
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/diagnose_force_replay.py --output outputs/hoi/contact_supervision/replay_diagnostics/run_02 > outputs/hoi/contact_supervision/replay_diagnostics/run_02.log 2>&1
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -ra tests > outputs/hoi/contact_supervision/replay_diagnostics/tests_full_01.log 2>&1
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/hoi/diagnose_force_replay.py --output outputs/hoi/contact_supervision/replay_diagnostics/run_03 > outputs/hoi/contact_supervision/replay_diagnostics/run_03.log 2>&1
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -ra tests > outputs/hoi/contact_supervision/replay_diagnostics/tests_full_02.log 2>&1
```

CPU: 20 passed. Runs01/02/03 exit0. First full suite exit1 (one physical diagnostic
line-search rejection, 89 passes, 17 skips); second full suite exit0 (90 passes,
17 skips). Full-suite GPU test invokes the same diagnostic in a fresh pytest temp
directory and checks its summary. A1 in the specification records the fixture
correction. The failed summary was copied verbatim from
`/tmp/pytest-of-han/pytest-86/test_physical_interval_recorde0/physical/diagnostics.json`
to `full_suite_failure_01/diagnostics.json`. The failed log remains verbatim.

All skips enumerate missing LAFAN1 NPZ/CSV inputs; none is a P2 test. Existing Torch
`torch.jit.script` deprecation warning is retained. Warp cache is
`/tmp/smhoi_p2_warp_cache`; P1 tests retain their own cache. No benchmark is counted
as training or full-task evaluation.

## Lint, complete diff and preservation

```bash
.venv/bin/ruff check src/supermimic/hoi/contact.py src/supermimic/hoi/replay.py src/supermimic/hoi/recorder.py scripts/hoi/diagnose_force_replay.py tests/hoi/test_force_aggregation.py tests/hoi/test_force_replay.py tests/hoi/test_force_recorder.py
.venv/bin/ruff format --check src/supermimic/hoi/contact.py src/supermimic/hoi/replay.py src/supermimic/hoi/recorder.py scripts/hoi/diagnose_force_replay.py tests/hoi/test_force_aggregation.py tests/hoi/test_force_replay.py tests/hoi/test_force_recorder.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/verify_p2.py
git diff --cached -- src/supermimic/hoi/contact.py src/supermimic/hoi/replay.py src/supermimic/hoi/recorder.py
git diff --cached -- scripts/hoi/diagnose_force_replay.py tests/hoi/test_force_aggregation.py tests/hoi/test_force_replay.py tests/hoi/test_force_recorder.py docs/hoi/force_contact/p2/IMPLEMENTATION_SPEC.md
git diff --cached --check
git diff a562d93 HEAD --check -- . ':!outputs/hoi/contact_supervision/replay_diagnostics/tests_full_01.log'
```

Final lint and format pass on seven Python files. Early lint identified import
ordering and missing explicit subprocess check=False; corrected before the final
suite. Complete source/test/diagnostic patches and subsequent corrective diff were
read. Numerical machine artifacts were checked without dumping large arrays into
the transcript. The raw failed pytest log has nine trailing-space lines; the full
whitespace check reports them and the scoped check passes. Preserve this exception
instead of claiming a completely whitespace-clean raw transcript.

The exact final evidence verifier is included below. It writes the hash/result
summary; the final review adds commit/file inventory metadata only. It checks
all force formulas independently from archived data, found counts, reference intent,
state continuity, masks, raw overflow flags, P1 function preservation and source
hashes. It does not rerun completed P0 audits or mutate accepted data.

```python
import ast,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=json.loads(Path('docs/hoi/force_contact/p2/baseline.json').read_text())
changed=[p for p,s in base['protected_sha256'].items() if h(p)!=s]
assert changed==['src/supermimic/hoi/contact.py'],changed
assert all(h(p)==s for p,s in base['dependency_sha256'].items())
p1=json.loads(Path('docs/hoi/force_contact/p1/validation_summary.json').read_text())
assert all(h(p)==s for p,s in p1['evidence_sha256'].items())
assert all(h(p)==s for p,s in p1['implementation_sha256'].items() if p!='src/supermimic/hoi/contact.py')
old=ast.parse(subprocess.check_output(['git','show','a562d93:src/supermimic/hoi/contact.py'],text=True))
new=ast.parse(Path('src/supermimic/hoi/contact.py').read_text())
oldfunc={x.name:ast.dump(x) for x in old.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
newfunc={x.name:ast.dump(x) for x in new.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
assert all(newfunc[k]==v for k,v in oldfunc.items())
root=Path('outputs/hoi/contact_supervision/replay_diagnostics')
r=json.loads((root/'run_03/diagnostics.json').read_text()); assert r['acceptance']=='PASS'
d=np.load(root/'run_03/transitions.npz',allow_pickle=False)
assert len(d['frame'])==96
assert all(np.isfinite(d[k]).all() for k in d.files)
f=d['force_raw_world'].sum(3)
q=d['box_quat_solve']
rot=Rotation.from_quat(q.reshape(-1,4),scalar_first=True).as_matrix().reshape(96,4,3,3)
mean=np.einsum('nkji,nkhj->nkhi',rot,f).mean(1)
np.testing.assert_allclose(mean,d['force_mean_box'],atol=2e-5,rtol=2e-6)
np.testing.assert_allclose(f.sum(1)*.005,d['impulse_world'],atol=1e-7,rtol=2e-6)
np.testing.assert_allclose(np.linalg.norm(f,axis=-1).max(1),d['force_peak'],atol=1e-5,rtol=2e-6)
assert np.array_equal((np.linalg.norm(f,axis=-1)>.001).mean(1),d['occupancy'])
assert np.array_equal(d['found_substeps'][:,-1].sum(-1),d['found_final'])
assert np.array_equal(d['physical_failure']|d['timeout']|d['clip_end'],d['transition_terminal'])
with np.load('outputs/hoi/asset_audit/part_b/reference/reference.npz') as ref:
 assert np.array_equal(d['intent_ref'][:,0],ref['object_contact'][d['frame']])
 assert np.array_equal(d['intent_ref'][:,1],ref['object_contact'][d['frame']+1])
for e in range(4):
 for ep in np.unique(d['episode'][d['env']==e]):
  idx=np.flatnonzero((d['env']==e)&(d['episode']==ep))
  assert np.array_equal(d['state'][idx[1:]],d['next_state'][idx[:-1]])
  assert (np.diff(d['frame'][idx])==1).all()
  assert not d['transition_terminal'][idx[:-1]].any()
raw=json.loads((root/'run_03/raw_samples.json').read_text());assert len(raw)==104
assert all(not any(x['overflow']) for x in raw)
assert '90 passed, 17 skipped' in (root/'tests_full_02.log').read_text()
paths=[p for p in root.rglob('*') if p.is_file()]
code=[Path(p) for p in subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines() if p.endswith('.py')]
summary={'acceptance':'PASS','tests':{'passed':90,'skipped_existing_missing_LAFAN1':17,'failed':0,'new_P2':21},
'baseline_files_checked':len(base['protected_sha256']),'baseline_files_unchanged':len(base['protected_sha256'])-1,
'intentionally_extended':'src/supermimic/hoi/contact.py','original_contact_functions_AST_unchanged':list(oldfunc),
'P1_evidence_hashes_match':True,'dependency_files_unchanged':len(base['dependency_sha256']),
'shared_source_modifications':[], 'implementation_sha256':{str(p):h(p) for p in code},
'evidence_sha256':{str(p):h(p) for p in paths},'diagnostics':r,'review_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
Path('docs/hoi/force_contact/p2/validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS:',len(base['protected_sha256'])-1,'unchanged protected files;',len(oldfunc),'P1 functions unchanged;',len(paths),'evidence files reviewed')
```

## Authorized implementation and final evidence commits

After the complete source review, stage only P2 paths; implementation commit
`5730be4` used:

```bash
git commit -m 'hoi: add force aggregation and replay supervision'
```

The repository ignores all `*.npz`. Final review caught this and explicitly stages
only the five newly generated diagnostic archives with `git add -f`, alongside
P2 report/validation/reuse/file-inventory documents. No original dataset is staged.
The final commit uses `docs(hoi P2): report validation and preserve replay evidence`.
Final inspection compares the complete stage diff against `a562d93`, rechecks
protected/dependency hashes and the exact staged path set, then prints HEAD/status.
No push or P3 command is part of this procedure.
