# P0 complete-plan closing commands and probes

This record covers only work missing from the committed audit. Original commands/results remain in `VALIDATION.md` and `AUDIT_PROBES.md`. Temporary scripts live under `/tmp`; their exact bodies below are documentation, not installed training/test code. Use a fresh evidence directory for any future run; do not overwrite historical evidence.

## Specification, repository and source inspection

The following read commands were used from `/home/han/pyx/SMHOI`:

```bash
pwd
git branch --show-current
git rev-parse HEAD
git status --short --untracked-files=all
rg --files --hidden -g AGENTS.md -g '!**/.git/**' -g '!**/.venv/**' .
sed -n '1,450p' 'docs/hoi/force contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.txt'
sed -n '451,900p' 'docs/hoi/force contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.txt'
sed -n '901,1350p' 'docs/hoi/force contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.txt'
sed -n '1351,1800p' 'docs/hoi/force contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.txt'
```

Python read checks searched ancestor AGENTS paths and recorded plan bytes/lines/SHA256. The 385-path preservation snapshot is `/tmp/smhoi_p0_closure_baseline.json`, built from the original preservation list, current tracked files and all four pre-existing user files. Each path was hashed using `hashlib.file_digest(..., 'sha256')`; 76 original inventory inputs were also hashed. Source repository HEAD/branch/status/remotes were read with `git -C`. Only the already documented incomplete user plan differed from the original audit's artifact hashes. The canonical complete master was created with exclusive binary open (`'xb'`) and the `.txt` bytes; equality was asserted.

Reference sources were located with `rg --files`; relevant functions/data paths with `rg -n`; source bodies were read with `cat`/`sed`. Exact file hashes and AST symbol ranges are in `reference_implementation_checks.json`. The substantive reads were:

| Source file (prefixes in reference audit) | Inspected material |
|---|---|
| HDMI `command.py` | Sensor resolution 395–449; current target/EEF/force and label update 545–584 |
| HDMI `rewards.py` | `eef_contact_exp`, `_max`, `_all` implementations 367–490, object tracking declarations |
| HDMI bread-box task YAML | Entire file |
| PhysHSI `carrybox.py` | Initialization/step/tensor refresh 1–180; force acquisition 1055–1098; task/contact/hold/release reward bodies 1938–2137; collider index/use searches |
| PhysHSI `carrybox_config.py` | Entire file, including active scales, asset names, timing and PPO/AMP configuration |
| SUGAR `mdp/rewards.py` | Entire file |
| SUGAR G1 carry and base refiner configs | Entire files; carry config, inherited sensor/command/observation/reward/termination settings |
| SUGAR `mdp/commands.py` | Loader 43–135; contact-label data-flow searches; measured bilateral log construction 740–770 |
| Installed Warp `_src/sensor.py` | Netforce reduction/centroid/torque code 2470–2620 |
| Target StateIO, StateSpec, rotations | Entire files for missing coordinate probes |
| Target asset manifest, Trainer and PolicyTrainer | Reuse/constructor/save/load/fit/rollout hook boundaries; no code edits |

A filename search for Workhorse under accessible `/home/han/pyx` and `/tmp` returned no implementation. It reported permission-denied entries for unrelated private system-service directories; none was entered or modified. The official arXiv abstract, HTML v2 and project page were opened with the web tool. Primary URLs: `https://arxiv.org/abs/2610.09117`, `https://arxiv.org/html/2610.09117v2`, `https://hybridrobotics.github.io/workhorse/`. The official unreleased-code statement and user-approved fallback are recorded in the reference audit. No research implementation was inferred from paper text.

## New numeric check command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/smhoi_p0_closure_probe.py
```

All five check groups passed. This emitted one upstream Torch JIT deprecation warning. Numerical output: `coordinate_closure_checks.json`. No optimizer, Trainer, physical HOI rollout, dataset conversion, reference cache or checkpoint write was used.

```python
"""Read-only P0 coordinate probes; no scene rollout, optimizer or dataset write."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from supermimic.envs.state_io import StateIO
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
from supermimic.kinematics.rotations import (
    quat_mul, quat_conjugate, quat_from_rotvec, rotvec_from_quat,
    quat_apply, quat_apply_inverse,
)

OUT = Path('/home/han/pyx/SMHOI/docs/hoi/force_contact/p0/coordinate_closure_checks.json')
assert not OUT.exists(), 'Preserve earlier evidence'
torch.set_default_dtype(torch.float64)
checks = {}

def error(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

# Existing robot StateIO, with synthetic buffers and nonzero distinct env origins.
origins = torch.tensor([[2., -3., .1], [-4., 5., .2]])
qpos = torch.zeros(2, 36); qpos[:, 3] = 1
qpos[:, :3] = origins + torch.tensor([.2, .3, .9])
qvel = torch.arange(70).reshape(2, 35) / 100
warm = torch.ones(2, 35)
env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins),
    sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=qvel, qacc_warmstart=warm)))
io = StateIO(env); state = io.read()
local_error = error(state[:, :3], [[.2, .3, .9], [.2, .3, .9]])
velocity_error = error(state[:, 36:], qvel)
other_q, other_v = qpos[0].clone(), qvel[0].clone()
new = state[1:2].clone(); new[:, :3] = torch.tensor([.6, -.7, 1.2])
io.write(torch.tensor([1]), new, forward=False)
write_error = error(qpos[1, :3], new[0, :3] + origins[1])
isolation = bool(torch.equal(qpos[0], other_q) and torch.equal(qvel[0], other_v)
    and torch.all(warm[0] == 1) and torch.all(warm[1] == 0))
checks['robot_origin_read_write'] = {'position_max_error_m': local_error,
    'velocity_copy_max_error': velocity_error, 'write_origin_max_error_m': write_error,
    'selected_row_only': isolation, 'scope': 'Existing StateIO with synthetic buffers; not P1 physical reset.'}
assert max(local_error, velocity_error, write_error) < 1e-12 and isolation

# Noncommuting body angular increment: existing StateSpec versus MuJoCo integration.
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
q = torch.tensor([.8, .2, -.3, .4]); q /= q.norm()
v = torch.tensor([.3, -.2, .4]); w = torch.tensor([.7, -.5, .9]); dt = .02
s = torch.zeros(1, 71); s[0, :3] = torch.tensor([.2, .3, .4]); s[0, 3:7] = q
s[0, 36:39] = v; s[0, 39:42] = w
spec = StateSpec(StateSpecCfg(), 'cpu', dtype=torch.float64)
pred = spec.integrate(s, torch.zeros(1, 70))
qp = s[0, :7].numpy().copy(); qv = torch.cat([v, w]).numpy().copy()
mujoco.mj_integratePos(model, qp, qv, dt)
right = quat_mul(q, quat_from_rotvec(dt * w))
wrong_left = quat_mul(quat_from_rotvec(dt * w), q)
right_error = error(pred[0, 3:7], qp[3:7]); wrong_error = error(wrong_left, qp[3:7])
position_error = error(pred[0, :3], qp[:3])
checks['quaternion_integration'] = {'state_spec_vs_mujoco_quaternion_max_error': right_error,
    'right_product_max_error': error(right, qp[3:7]), 'wrong_left_product_max_error': wrong_error,
    'zero_residual_position_error_m': position_error, 'dt_s': dt,
    'convention': 'wxyz; q_next = q * exp(dt * omega_body)'}
assert right_error < 1e-12 and position_error < 1e-12 and wrong_error > 1e-3

# Reference finite-difference conventions, synthetic constant-twist samples only.
t = torch.arange(9) * dt
poses = s[0, :3] + t[:, None] * v
quats = quat_mul(q.expand(9, 4), quat_from_rotvec(t[:, None] * w))
quats[1::2] *= -1  # equivalent poses with alternating quaternion signs
linear, body, world = [], [], []
for i in range(9):
    lo, hi = max(0, i - 1), min(8, i + 1)
    span = (hi - lo) * dt
    linear.append((poses[hi] - poses[lo]) / span)
    body.append(rotvec_from_quat(quat_mul(quat_conjugate(quats[lo]), quats[hi])) / span)
    world.append(rotvec_from_quat(quat_mul(quats[hi], quat_conjugate(quats[lo]))) / span)
linear = torch.stack(linear); body = torch.stack(body); world = torch.stack(world)
ww = quat_apply(q, w)
linerr = error(linear, v.expand_as(linear)); berr = error(body, w.expand_as(body)); werr = error(world, ww.expand_as(world))
checks['reference_velocity_conventions'] = {'frames': 9, 'linear_max_error_m_s': linerr,
    'body_angular_max_error_rad_s': berr, 'world_angular_max_error_rad_s': werr,
    'endpoints_one_sided': True, 'interior_centered': True, 'quaternion_sign_flip_invariant': True,
    'scope': 'Synthetic convention probe only; no reference cache generated or accepted NPZ changed.'}
assert max(linerr, berr, werr) < 1e-11

# Independent MuJoCo rotation matrix verifies existing Torch vector transforms.
mat = np.empty(9); mujoco.mju_quat2Mat(mat, q.numpy()); mat = mat.reshape(3, 3)
fw = torch.tensor([3., -4., 5.]); tw = torch.tensor([.3, -.2, .7])
fb = quat_apply_inverse(q, fw); tb = quat_apply_inverse(q, tw)
ferr = error(fb, mat.T @ fw.numpy()); terr = error(tb, mat.T @ tw.numpy())
roundtrip = error(quat_apply(q, fb), fw)
q90 = quat_from_rotvec(torch.tensor([0., 0., np.pi / 2]))
axis_error = error(quat_apply_inverse(q90, torch.tensor([0., 1., 0.])), [1., 0., 0.])
checks['box_local_vectors'] = {'force_max_error_N': ferr, 'torque_max_error_Nm': terr,
    'force_roundtrip_max_error_N': roundtrip, 'yaw90_axis_error': axis_error,
    'force_norm_preserved': bool(torch.allclose(fb.norm(), fw.norm(), atol=1e-12)),
    'convention': 'R_box_world transpose times world vector; torque reference point unchanged by rotation.'}
assert max(ferr, terr, roundtrip, axis_error) < 1e-11

# Installed netforce torque is about a separate centroid for each primary column.
centers = np.array([[.2, 0., 0.], [0., .3, 0.]])
forces = np.array([[0., 2., 1.], [3., 0., -1.]])
torques = np.array([[.1, .2, .3], [-.1, .4, .2]])
origin = np.array([.01, .02, .03])
translated_sum = (torques + np.cross(centers - origin, forces)).sum(axis=0)
origin_sum_then_shift = (torques + np.cross(centers, forces)).sum(axis=0) - np.cross(origin, forces.sum(axis=0))
moment_error = error(translated_sum, origin_sum_then_shift)
naive_error = float(np.linalg.norm(translated_sum - torques.sum(axis=0)))
checks['multi_primary_torque_reference'] = {'common_origin_equivalence_error_Nm': moment_error,
    'naive_torque_sum_error_Nm': naive_error,
    'formula': 'T_O = sum_i(T_centroid_i + (centroid_i - O) cross F_i)',
    'scope': 'Algebraic diagnostic, not physical wrench calibration. Per-hand force summation needs no moment shift.'}
assert moment_error < 1e-12 and naive_error > .1
checks['all_pass'] = True
OUT.write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2))
```

## Closing preservation, scope and documentation command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/smhoi_p0_final_closure_checks.py
```

This validates integrity and documentation, then writes fresh closing evidence and the full review patch. The supplied master's bytes, including original whitespace, are retained exactly; they are not reformatted to satisfy whitespace style. Tracked changes and newly authored evidence are checked with `git diff --check` / `git diff --no-index --check`. No staging, commit or push occurs.

```python
"""Closing documentation/preservation validation; never edits source or assets."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess

R = Path('/home/han/pyx/SMHOI')
O = R / 'docs/hoi/force_contact/p0'
B = json.loads(Path('/tmp/smhoi_p0_closure_baseline.json').read_text())
MODIFIED = ['REPORT.md', 'CLOSURE.md', 'FILE_CHANGE_MAP.md', 'ARCHITECTURE_AUDIT.md', 'DEPENDENCY_CONTACT_AUDIT.md', 'VALIDATION.md']
NEW_P0 = ['REFERENCE_IMPLEMENTATION_AUDIT.md', 'reference_implementation_checks.json', 'coordinate_closure_checks.json', 'REQUIREMENT_COVERAGE.md', 'CLOSURE_CHECKS.md', 'closure_checks.json']
MASTER = 'docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md'
USER_FILES = ['.vscode/settings.json', 'SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md', 'docs/hoi/force contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md', 'docs/hoi/force contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.txt']

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=R, text=True).strip()

modified = {'docs/hoi/force_contact/p0/' + n for n in MODIFIED}
new = {'docs/hoi/force_contact/p0/' + n for n in NEW_P0} | {MASTER}
assert git('rev-parse', 'HEAD') == B['head'] == 'e89573171c86be7f40c8d120dc813cee0274a152'
for commit in ['579d1a8c661985c91926a365c7505488abe03bb8', B['head']]:
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=R, check=True)
assert set(git('diff', '--name-only').splitlines()) == modified
assert not git('diff', '--cached', '--name-only')

changed = [p for p,h in B['files'].items() if not (R/p).is_file() or sha(R/p) != h]
assert set(changed) == modified, changed
input_changes = [p for p,h in B['inventoried_inputs'].items() if sha(p) != h]
assert not input_changes, input_changes
source = json.loads((O/'source_inventory.json').read_text())
assert all(sha(R/p) == d['sha256'] for p,d in source.items())
deps = json.loads((O/'dependencies.json').read_text())['inspected_file_sha256']
assert all(sha(p) == h for p,h in deps.items())
refs = json.loads((O/'reference_implementation_checks.json').read_text())
assert all(sha(d['path']) == d['sha256'] for d in refs['files'])
for path, old in B['reference_repositories'].items():
    for key,args in [('head',['rev-parse','HEAD']), ('status',['status','--short'])]:
        assert subprocess.check_output(['git','-C',path,*args],text=True).strip() == old[key]
assert not refs['workhorse']['source_code_audit_claimed']
assert refs['workhorse']['user_approved_fallback'] == 'if workhorse code not released, just use paper and other papers and repo'
assert json.loads((O/'coordinate_closure_checks.json').read_text())['all_pass']

source_plan = R / USER_FILES[-1]
assert (R/MASTER).read_bytes() == source_plan.read_bytes()
assert sha(source_plan) == '7365f680cf975263e69672d75d585c62e83532115b9065c6e63929004bdbdc9b'
plan_text = source_plan.read_text()
assert len(plan_text.splitlines()) == 1743
assert all(re.search(r'^# \d+\. P'+str(i)+r' ', plan_text, re.M) for i in range(9))
ref_path = R/'outputs/hoi/asset_audit/part_b/reference/reference.npz'
manifest_rel = 'outputs/hoi/asset_audit/part_b/reference/reference_manifest.json'
committed_manifest = subprocess.check_output(['git','show','HEAD:'+manifest_rel],cwd=R)
assert committed_manifest == (R/manifest_rel).read_bytes()
assert sha(ref_path) == json.loads(committed_manifest)['sha256'] == 'bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b'

json_files = sorted(p for p in O.glob('*.json') if p.name != 'closure_checks.json')
for p in json_files:
    json.loads(p.read_text())
bad_links=[]
for name in MODIFIED + ['REFERENCE_IMPLEMENTATION_AUDIT.md','REQUIREMENT_COVERAGE.md','CLOSURE_CHECKS.md']:
    for target in re.findall(r'\]\(([^)]+)\)', (O/name).read_text()):
        if target.startswith(('http://','https://','#')):
            continue
        dest = O/target.split('#')[0]
        if dest != O/'closure_checks.json' and not dest.exists():
            bad_links.append({'file':name,'target':target})
assert not bad_links, bad_links
subprocess.run(['git','diff','--check'],cwd=R,check=True)

evidence = {
    'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'stage':'P0', 'decision':'PASS_FOR_REVIEW', 'branch':git('branch','--show-current'),
    'head':git('rev-parse','HEAD'), 'prior_audit_commit':'579d1a8c661985c91926a365c7505488abe03bb8',
    'history_preserved':True, 'initial_status':B['initial_status'],
    'master':{'source':str(source_plan),'canonical':MASTER,'sha256':sha(source_plan),'bytes':source_plan.stat().st_size,'lines':1743,'byte_identical':True},
    'workhorse_fallback':refs['workhorse'],
    'preservation':{'baseline_paths':len(B['files']),'changed_paths':changed,
        'all_changes_are_listed_documentation':True,'protected_paths_unchanged':len(B['files'])-len(changed),
        'existing_inventory_inputs_rehashed':len(B['inventoried_inputs']),'input_changes':input_changes,
        'source_files_unchanged':len(source),'dependency_files_unchanged':len(deps),
        'reference_source_files_unchanged':len(refs['files']),'reference_repository_heads_status_unchanged':True,
        'part_b_actual_and_committed_manifest_match':True,'part_b_sha256':sha(ref_path),'user_files_preserved':USER_FILES,
        'original_machine_evidence_preserved':True},
    'checks':{'coordinate_groups_passed':5,'reference_source_static_audit_complete':True,
        'original_tests_and_physics_probes_rerun':False,'existing_baseline_tests_reused':7,
        'json_files_parsed_including_this_output':len(json_files)+1,'local_links_valid':True,
        'tracked_whitespace_check_pass':True,'index_empty':True},
    'modified_files':sorted(modified),'new_files':sorted(new),
    'p1_started':False,'training_started':False,'committed':False,'pushed':False,
}
(O/'closure_checks.json').write_text(json.dumps(evidence,indent=2)+'\n')
json.loads((O/'closure_checks.json').read_text())
untracked = set(git('ls-files','--others','--exclude-standard').splitlines())
assert untracked == new | set(USER_FILES), untracked
patch = subprocess.check_output(['git','diff','--no-ext-diff','--binary','HEAD','--','docs/hoi/force_contact'],cwd=R)
for name in sorted(new):
    check = subprocess.run(['git','diff','--no-index','--check','/dev/null',name],cwd=R,capture_output=True,text=True)
    # Canonical master is the exact supplied text, with intentional Markdown hard breaks.
    if name != MASTER:
        assert check.returncode in (0,1) and not check.stdout and not check.stderr,(name,check.stdout,check.stderr)
    x = subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=R,capture_output=True)
    assert x.returncode == 1
    patch += x.stdout
Path('/tmp/SMHOI_P0_FINAL_CLOSURE_REVIEW.patch').write_bytes(patch)
print(json.dumps({'decision':evidence['decision'],'preservation':evidence['preservation'],'checks':evidence['checks'],
    'patch_bytes':len(patch),'patch_lines':len(patch.splitlines()),'status':git('status','--short','--untracked-files=all')},indent=2))
```

## Reused and skipped checks

The original seven non-training tests, strict checkpoint loading, native model/collider inventory, CPU/Warp/captured force sign, four-history/reset probe and offset-COM check are reused from committed evidence after hash verification. They were not executed again. No mandatory closing test failed. P1 calibration and P2–P8 tests/experiments were not run because their stages are not authorized. Current PASS is the P0 audit decision, pending user review.
