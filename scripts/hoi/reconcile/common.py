"""M6.5 isolated paths and read-only inputs; historical audit remains immutable."""
from pathlib import Path
import sys,json,hashlib,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/hoi'))
from asset_common import rot,vertices,SOURCE,HDMI
from audit_npz import load
from audit_contact import comparison_model,write_csv,stats,box_point
OUT=ROOT/'docs/hoi/asset_audit/m6_5'
HISTORY=ROOT/'docs/hoi/asset_audit'
NPZ=Path('/home/han/pyx/hdmi_bread_box_hoi_v1.npz')
MANIFEST=ROOT/'data/assets/hdmi_bread_box/asset_manifest.json'
def read(name):return json.loads((HISTORY/name).read_text())
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def baseline():
    p=OUT/'baseline.json'
    if p.exists():raise FileExistsError(p)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    files={str(ROOT/x):sha(ROOT/x) for x in tracked if x}
    for src in json.loads(MANIFEST.read_text())['sources']:files[src['path']]=sha(src['path'])
    for x in ['active_adaptation/envs/mdp/commands/hdmi/command.py','active_adaptation/envs/mdp/commands/hdmi/rewards.py','active_adaptation/assets_mjcf/objects/support/support0.xml','active_adaptation/assets_mjcf/objects/support/support1.xml']:
        files[str(HDMI/x)]=sha(HDMI/x)
    save('baseline.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_status':subprocess.check_output(['git','-C',str(SOURCE),'status','--short'],text=True),'target_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),'protected_sha256':files,'scope':'M6.5 only; offline geometry + short unassisted MuJoCo PD tests; no training'})
def verify():
    d=json.loads((OUT/'baseline.json').read_text());changed=[p for p,h in d['protected_sha256'].items() if sha(p)!=h]
    if changed:raise RuntimeError(changed)
    return len(d['protected_sha256'])
if __name__=='__main__':baseline()
