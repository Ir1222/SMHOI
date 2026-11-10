"""Capture immutable input hashes and repository state; never writes inputs."""
import hashlib, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/home/han/pyx/humanoid-amp-ppo')
HDMI=Path('/tmp/supermimic_hoi_audit_20261104/HDMI')
def sha(p):
    return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def git(p,*args):
    return subprocess.check_output(['git','-C',str(p),*args],text=True).strip()
def main():
    out=ROOT/'docs/hoi/asset_audit/baseline.json'
    if out.exists(): raise FileExistsError('Baseline is immutable; do not recapture after edits')
    files=list(Path('/home/han/pyx').glob('*.npz'))+[ROOT/'SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md']
    files+=list((SOURCE/'src/mjlab_g1/hoi').glob('*.py'))
    files+=list((SOURCE/'dataset/hoi/manifests').glob('hdmi*.json'))
    files+=[SOURCE/'dataset/external/hdmi/carry_and_place_bread_box-0829'/x for x in ('motion.npz','meta.json')]
    files+=list((HDMI/'active_adaptation/assets_mjcf/objects/bread_box').glob('*.xml'))
    files+=[HDMI/'cfg/task/G1/hdmi/carry_and_place_bread_box.yaml']
    for m in ('hdmi_usd_contact_assets.json','hdmi_usd_support_assets.json'):
        d=json.loads((SOURCE/'dataset/hoi/manifests'/m).read_text())
        files += [Path(v['path']) for v in d.values() if isinstance(v,dict) and 'path' in v]
    files+=list((ROOT/'.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1').rglob('*.xml'))
    files+=[ROOT/'.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py',ROOT/'src/supermimic/envs/tracking_env_cfg.py']
    data={'scope':'Part A M0-M6 only; user overrides conflicting Part D numbering; no learning pipeline',
      'cwd':str(ROOT),'agents_md':'No AGENTS.md found in source/target, or /, /home, /home/han, /home/han/pyx ancestors.',
      'initial_status_before_audit':{'target':'?? SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md','source':'?? MUJOCO_LOG.TXT\n?? docs/hoi/audits/hoi_research_freeze_20261107.md'},
      'repositories':{str(p):{k:git(p,*v) for k,v in {'root':('rev-parse','--show-toplevel'),'branch':('branch','--show-current'),'head':('rev-parse','HEAD'),'status_at_capture':('status','--short')}.items()} for p in (ROOT,SOURCE,HDMI)},
      'inputs':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(files))}}
    out.write_text(json.dumps(data,indent=2)+'\n')
    (out.parent/'baseline.md').write_text('# M0 — baseline\n\nTarget: '+str(ROOT)+' (the separate sibling SuperMimic-Reproduced is not the target).\n\nFull plan read. Scope follows user and Part A; Part D numbering conflicts. No training or reference generation.\n\nInitial target: `ba61053`, only untracked user plan. Initial source: `114f766`, two untracked user files recorded in baseline.json. No AGENTS.md found. All input paths, hashes, sizes, branches and full SHAs are in baseline.json. No reset/clean/stash/checkout. Inputs and source repository remain read-only.\n\nRuntime shell sandbox fails at bwrap loopback initialization; commands run through reviewed escalation. Only new audit files in SMHOI are staged.\n')
if __name__=='__main__': main()
