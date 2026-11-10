import sys,json,importlib.util
from pathlib import Path
p='/home/han/pyx/humanoid-amp-ppo/src/mjlab_g1/scripts/hoi_inspect_hdmi_usd.py'
s=importlib.util.spec_from_file_location('inspect_usd',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
a=Path('/tmp/supermimic_hoi_audit_20261104/HDMI/active_adaptation/assets')
d={k:m.inspect(a/v,robot=k=='robot') for k,v in {'robot':'g1/g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule.usd','object':'objects/bread_box/bread_box.usd','support0':'objects/support/support0.usd','support1':'objects/support/support1.usd'}.items()}
Path('/home/han/pyx/SMHOI/docs/hoi/asset_audit/usd_reread.json').write_text(json.dumps(d,indent=2)+'\n')
