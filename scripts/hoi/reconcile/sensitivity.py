"""Named half-step numerical check; original 50 Hz PD clock and gains retained."""
from dynamics import run_case
from common import *
verify();out=OUT/'dynamics_half_step';out.mkdir(exist_ok=False);ref=load(NPZ);results=[]
for profile in ('native','hdmi'):
 for frame in (400,530):
  case={'name':f'{profile}_f{frame}_dt2p5ms','profile':profile,'raise_m':0.,'robot':True,'start':frame,'seconds':1. if frame==400 else 2.,'dt':.0025}
  results.append(run_case(case,ref,out))
for dz in (0.,.01):
 case={'name':f'box_only_z{round(dz*1000)}_dt2p5ms','profile':'native','raise_m':dz,'robot':False,'start':839,'seconds':3.,'dt':.0025}
 results.append(run_case(case,ref,out))
(out/'summary.json').write_text(json.dumps(results,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x)+'\n');verify()
