"""Fresh raw USD inspection; run with an existing Python environment providing pxr.

Uses the source's audited inspector without importing any simulator/training code.
"""
import json,importlib.util
from pathlib import Path
from baseline import ROOT,SOURCE,HDMI
from pxr import Usd,UsdPhysics
p=SOURCE/'src/mjlab_g1/scripts/hoi_inspect_hdmi_usd.py'
s=importlib.util.spec_from_file_location('inspect_usd',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
a=HDMI/'active_adaptation/assets'
paths={'robot':'g1/g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule.usd','object':'objects/bread_box/bread_box.usd','support0':'objects/support/support0.usd','support1':'objects/support/support1.usd'}
d={k:m.inspect(a/v,robot=k=='robot') for k,v in paths.items()}
out=ROOT/'docs/hoi/asset_audit'
(out/'usd_reread.json').write_text(json.dumps(d,indent=2)+'\n')
extra={}
for k,v in paths.items():
    stage=Usd.Stage.Open(str(a/v));records=[]
    for prim in stage.Traverse():
        if prim.GetTypeName()=='Material' or prim.HasAPI(UsdPhysics.CollisionAPI):
            attrs={attr.GetName():m.serial(attr.Get()) for attr in prim.GetAttributes() if attr.GetName().startswith(('physics:','physx'))}
            rels={rel.GetName():[str(x) for x in rel.GetTargets()] for rel in prim.GetRelationships() if 'material' in rel.GetName() or 'filter' in rel.GetName()}
            records.append({'path':str(prim.GetPath()),'type':prim.GetTypeName(),'attrs':attrs,'relationships':rels})
    extra[k]=records
(out/'usd_materials.json').write_text(json.dumps(extra,indent=2)+'\n')
