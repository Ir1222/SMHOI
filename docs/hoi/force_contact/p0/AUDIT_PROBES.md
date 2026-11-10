# P0 temporary audit probe transcript

These are the exact temporary probe bodies used in this audit, preserved as documentation. They are not installed training/scene code and do not authorize P1. Each writes only new P0 evidence. For a new audit, select a fresh output directory and preserve the prior evidence rather than rerunning over it. The inventory/preservation operations and test command are described in VALIDATION.md.

## smhoi_p0_probe.py

```python
import json, hashlib, subprocess, dataclasses
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from mjlab.sensor.contact_sensor import ContactSensorCfg, ContactMatch
from supermimic.hoi.reference import ReferenceClip, ROBOT_FIELDS, array_info
from supermimic.hoi.asset_manifest import load_manifest
from supermimic.envs.tracking_env_cfg import make_env_cfg, SuperMimicEnvCfgArgs
from mjlab.asset_zoo.robots import get_g1_robot_cfg
R=Path('/home/han/pyx/SMHOI'); O=R/'docs/hoi/force_contact/p0'; results={}
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ref=ReferenceClip(R/'outputs/hoi/asset_audit/part_b/reference/reference.npz')
m=load_manifest(R/'data/assets/hdmi_bread_box/asset_manifest_m6_5.json')
committed=json.loads(subprocess.check_output(['git','show','HEAD:outputs/hoi/asset_audit/part_b/reference/reference_manifest.json'],cwd=R))
with np.load('/home/han/pyx/hdmi_bread_box_robot_only_50hz_supermimic_pyx.npz') as z:
 exact={k:array_info(z[k])==array_info(ref.arrays[k]) for k in ROBOT_FIELDS}
results['reference']={'actual_sha256':sha(ref.path),'committed_manifest_match':sha(ref.path)==committed['sha256'],'robot_exact':exact,'frames':ref.n_frames,'fps':ref.fps,'fields':list(ref.arrays),'intent_on_frames':[int(np.flatnonzero(ref.arrays['object_contact'])[0]),int(np.flatnonzero(ref.arrays['object_contact'])[-1])],'source_hash_matches':{k:sha(v['path'])==v['sha256'] for k,v in committed['sources'].items()}}
results['asset']={'sha256':sha(R/'data/assets/hdmi_bread_box/asset_manifest_m6_5.json'),'verified_all_manifest_sources':True,'training_ready':m['training_ready'],'status':m['status'],'selected_profile':m['robot']['selected_profile'],'object':m['object'],'supports':m['supports'],'contact':m['contact'],'gates':m['gates']}
cfg=make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[str(ref.path)],num_envs=2))
results['config']={'physics_dt':cfg.sim.mujoco.timestep,'decimation':cfg.decimation,'control_dt':cfg.sim.mujoco.timestep*cfg.decimation,'nconmax':cfg.sim.nconmax,'njmax':cfg.sim.njmax,'sensors':[s.name for s in cfg.scene.sensors],'entities':list(cfg.scene.entities)}
# Compile existing native robot configuration only, no HOI implementation or physics rollout.
rc=get_g1_robot_cfg(); spec=rc.spec_fn()
for c in rc.collisions:c.edit_spec(spec)
model=spec.compile(); hands=[]
for i in range(model.ngeom):
 name=model.geom(i).name
 if 'hand' in name or ('wrist' in name and (model.geom_contype[i] or model.geom_conaffinity[i])):
  bid=model.geom_bodyid[i]; hands.append({'geom':name,'body':model.body(bid).name,'type':int(model.geom_type[i]),'pos':model.geom_pos[i].tolist(),'quat':model.geom_quat[i].tolist(),'size':model.geom_size[i].tolist(),'contype':int(model.geom_contype[i]),'conaffinity':int(model.geom_conaffinity[i]),'condim':int(model.geom_condim[i]),'priority':int(model.geom_priority[i]),'friction':model.geom_friction[i].tolist()})
results['native_hand_colliders']=hands
# Installed ContactSensor history with synthetic sensordata (dependency bookkeeping probe only).
spec=mujoco.MjSpec.from_string('<mujoco><worldbody><body name="a"><freejoint/><geom type="sphere" size=".1"/></body><body name="b"><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
sensor=ContactSensorCfg(name='audit',primary=ContactMatch(mode='body',pattern='a'),secondary=ContactMatch(mode='body',pattern='b'),secondary_policy='error',reduce='netforce',fields=('found','force'),num_slots=1,history_length=4).build()
sensor.edit_spec(spec,{})
mm=spec.compile(); raw=torch.zeros(2,mm.nsensordata); sensor.initialize(mm,None,SimpleNamespace(sensordata=raw,time=torch.zeros(2)),'cpu')
fa=int(mm.sensor('audit_a_force').adr[0]); ca=int(mm.sensor('audit_a_found').adr[0]); samples=[1.,2.,3.,4.]
for v in samples:
 raw[:,fa:fa+3]=torch.tensor([v,0.,0.]);raw[:,ca]=1;sensor.update(.005)
h=sensor.data.force_history.clone(); raw[:,fa:fa+3]=99.
# Simulate raw buffer refresh without scene.update: cached live force stays as first read, history unchanged.
cache_value=sensor.data.force.clone(); before=sensor.data.force_history.clone(); sensor.reset(torch.tensor([0])); after=sensor.data.force_history.clone()
results['history_probe']={'shape':list(h.shape),'newest_first':h[0,0,:,0].tolist(),'four_samples_match':torch.equal(h[0,0,:,0],torch.tensor([4.,3.,2.,1.])),'cached_force_after_raw_refresh':cache_value.tolist(),'history_unchanged_by_raw_refresh':torch.equal(before,h),'partial_reset_zero':bool((after[0]==0).all()),'other_env_unchanged':torch.equal(after[1],h[1]),'found_history_exists':hasattr(sensor.data,'found_history'),'history_tensor_replaced_per_update':True,'synthetic_not_physical':True}
# Numeric free-joint reference-point/frame probe with nontrivial rotation and COM offset.
mm=mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0"/><worldbody><body name="free"><freejoint/><inertial pos=".1 -.05 -.2" mass="2" diaginertia=".03 .04 .05"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'); dd=mujoco.MjData(mm)
q=np.array([.8,.2,-.3,.4]);q/=np.linalg.norm(q);dd.qpos[:]=np.r_[[.2,.3,.4],q];dd.qvel[:]=[.3,-.2,.4,.7,-.5,.9];mujoco.mj_forward(mm,dd)
rot=dd.xmat[1].reshape(3,3);wb=dd.qvel[3:6];ww=rot@wb
vobj=np.zeros(6);mujoco.mj_objectVelocity(mm,dd,mujoco.mjtObj.mjOBJ_BODY,1,vobj,0)
# mj_objectVelocity(BODY) is centered at body COM; derive origin velocity.
vroot=vobj[3:]-np.cross(ww,dd.xipos[1]-dd.xpos[1]);eps=1e-7; qp=dd.qpos.copy();mujoco.mj_integratePos(mm,qp,dd.qvel,eps)
results['freejoint_probe']={'qvel':dd.qvel.tolist(),'body_velocity_world_ang_then_com_linear':vobj.tolist(),'omega_world_expected':ww.tolist(),'linear_root_from_com':vroot.tolist(),'root_linear_matches_qvel':bool(np.allclose(vroot,dd.qvel[:3],atol=1e-10)),'world_angular_matches_R_times_body':bool(np.allclose(vobj[:3],ww,atol=1e-10)),'root_position_derivative_matches_qvel':bool(np.allclose((qp[:3]-dd.qpos[:3])/eps,dd.qvel[:3],atol=1e-8)),'com_differs_from_root':bool(np.linalg.norm(vobj[3:]-dd.qvel[:3])>0.01)}
(O/'audit_checks.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({k:v for k,v in results.items() if k!='asset'},indent=2))
```

## smhoi_p0_checkpoint.py

```python
import json, hashlib, subprocess
from pathlib import Path
import torch
from supermimic.policy.nets import Policy, PolicyCfg, Critic, CriticCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from supermimic.world_model.state_spec import StateSpecCfg
R=Path('/home/han/pyx/SMHOI'); O=R/'docs/hoi/force_contact/p0'; p=Path('/home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt')
unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(p)
allowed={'pathlib.PosixPath','pathlib.Path'}
assert set(unsafe)<=allowed,unsafe
with torch.serialization.safe_globals([Path,type(p)]):ck=torch.load(p,map_location='cpu',weights_only=True)
config=ck['config']; out={'path':str(p),'unsafe_globals_allowlisted':unsafe,'keys':list(ck),'iteration':ck.get('iteration'),'num_updates':ck.get('num_updates'),'config':config,'tensor_shapes':{k:{n:list(v.shape) for n,v in ck[k].items()} for k in ('wm','policy','critic')},'strict_load':{}}
pcfg=PolicyCfg(**config['policy']); wcfg=dict(config['world_model']);wcfg['state_spec']=StateSpecCfg(**wcfg['state_spec'])
with torch.device('meta'):
 models={'policy':Policy(pcfg),'critic':Critic(CriticCfg(**config['critic']),pcfg=pcfg),'wm':WorldModel(WorldModelCfg(**wcfg),None,'meta')}
for k,model in models.items():
 try:
  r=model.load_state_dict(ck[k],strict=True,assign=True);out['strict_load'][k]={'pass':True,'missing':r.missing_keys,'unexpected':r.unexpected_keys}
 except Exception as e:out['strict_load'][k]={'pass':False,'error':str(e)}
# Check consumed checkpoint config paths, no Trainer initialization and no optimizer step.
for name in ('data','resume','run_name'):out['config_'+name]=config.get(name)
source=Path('/home/han/pyx/SuperMimic-Reproduced'); compared={}
for root in ['world_model','imagination','policy','data','training','envs']:
 for f in (R/'src/supermimic'/root).rglob('*.py'):
  rel=f.relative_to(R); other=source/rel
  if other.exists():compared[str(rel)]=f.read_bytes()==other.read_bytes()
out['current_vs_candidate_checkout_code_equal']=compared
out['provenance_limit']='Checkpoint has config/iteration/weights but no embedded Git SHA or dataset SHA; current external checkout equality is not proof of code at training time. No HOI migration performed.'
(O/'checkpoint_audit.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['tensor_shapes','config']},indent=2,default=str))
```

## smhoi_p0_force_probe.py

```python
import json
from pathlib import Path
import mujoco, numpy as np
import warp as wp
wp.config.kernel_cache_dir='/tmp/smhoi_p0_warp_cache'
import mujoco_warp as mjw
from mjlab.sensor.contact_sensor import ContactSensorCfg,ContactMatch
O=Path('/home/han/pyx/SMHOI/docs/hoi/force_contact/p0')
spec=mujoco.MjSpec.from_string('<mujoco><option timestep=".005"/><worldbody><body name="floor"><geom name="floor_geom" type="plane" size="1 1 .1"/></body><body name="ball" pos="0 0 .099"><freejoint/><geom name="ball_geom" type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
for name,prim,sec in [('ball_to_floor','ball','floor'),('floor_to_ball','floor','ball')]:
 s=ContactSensorCfg(name=name,primary=ContactMatch(mode='body',pattern=prim),secondary=ContactMatch(mode='body',pattern=sec),secondary_policy='error',fields=('found','force'),reduce='netforce',history_length=4,num_slots=1).build();s.edit_spec(spec,{})
mm=spec.compile();dd=mujoco.MjData(mm);mujoco.mj_forward(mm,dd)
cpu=dd.sensordata.copy()
with wp.ScopedDevice('cuda:0'):
 wm=mjw.put_model(mm);wd=mjw.put_data(mm,dd,nworld=2,nconmax=8,njmax=32)
 # Use existing CPU constraint solution to isolate Warp contact-sensor implementation.
 wd.sensordata.zero_();mjw.sensor_acc(wm,wd);wp.synchronize();gpu=wd.sensordata.numpy().copy()
 with wp.ScopedCapture() as cap:mjw.sensor_acc(wm,wd)
 wp.capture_launch(cap.graph);wp.synchronize();captured=wd.sensordata.numpy().copy()
forward=int(mm.sensor('ball_to_floor_ball_force').adr[0]);reverse=int(mm.sensor('floor_to_ball_floor_force').adr[0]);f=cpu[forward:forward+3];fr=cpu[reverse:reverse+3]
out={'purpose':'Dependency sign/netforce probe on an ephemeral ball-plane fixture. Not P1 HOI calibration.','cpu_sensor_values':cpu.tolist(),'warp_sensor_values':gpu.tolist(),'ball_primary_force':f.tolist(),'floor_primary_force':fr.tolist(),'force_is_primary_on_secondary':bool(f[2]<0 and fr[2]>0),'swapped_pair_negates':bool(np.allclose(f,-fr,atol=1e-8)),'cpu_warp_max_abs_difference':float(np.max(np.abs(gpu-cpu))),'cpu_warp_match':bool(np.allclose(gpu,cpu,rtol=5e-5,atol=5e-5)),'captured_eager_match':bool(np.allclose(captured,gpu,rtol=5e-5,atol=5e-5)),'scope_limit':'Tests sensor_acc using an existing constraint solution; does not validate full HOI stepping, reset, seven-geoms or transient contact.'}
(O/'force_sign_probe.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert all(out[k] for k in ['force_is_primary_on_secondary','swapped_pair_negates','cpu_warp_match','captured_eager_match'])
```

## smhoi_p0_arch_probe.py

```python
import json
from pathlib import Path
import torch
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.tracking_env_cfg import TRACKED_BODIES
from supermimic.world_model.state_spec import StateSpec,StateSpecCfg
O=Path('/home/han/pyx/SMHOI/docs/hoi/force_contact/p0'); b=ReplayBuffer(1,8,71,29,33,'cpu')
ids=torch.tensor([0]);a=torch.zeros(1,29);xi=torch.zeros(1,33)
# Exact row flag convention written by current ReplayRecorder for two transitions,
# second transition terminates: s0(row done=0), s1(row done=1), s2(terminal row done=1).
for i in range(3):
 s=torch.zeros(1,71);s[0,0]=i;s[0,3]=1
 b.add(ids,s,a,a,xi,torch.zeros(1),torch.tensor([float(i>0)]),torch.tensor([i]),torch.tensor([0]),torch.tensor([i==2]),torch.tensor([i==0]))
w=b.sample_windows(1,2)
# Existing goal window clamps. This only probes a loader, no reference writes.
lib=MotionLibrary(['/home/han/pyx/hdmi_bread_box_robot_only_50hz_supermimic_pyx.npz'],TRACKED_BODIES,torch.zeros(29),device='cpu')
last=torch.tensor([839]);g=lib.goal_window(last,3).reshape(1,3,68)
spec=StateSpec(StateSpecCfg(),'cpu');s=torch.zeros(2,71);s[:,3]=1;s[1,:2]=torch.tensor([2.,-3.])
out={'replay_terminal_label_alignment':{'actual_states_x':w.states[:,0,0].tolist(),'actual_done':w.done[:,0].tolist(),'correct_transition_done':[0.,1.],'matches_transition_contract':bool(torch.equal(w.done[:,0],torch.tensor([0.,1.]))),'explanation':'sample_windows reads successor-row done; recorder writes terminating done to both preterminal transition row and terminal state row, so previous transition is also labeled done.'},'reference_end':{'window_shape':list(g.shape),'all_goal_slots_at_839_equal':bool(torch.equal(g[:,0],g[:,1]) and torch.equal(g[:,1],g[:,2])),'valid_mask_returned':False,'imagine_frames_all':'torch.minimum(frame + arange(H+1), last); no finite-clip reward mask'},'support_context_identifiability':{'translated_state_canonical_equal':bool(torch.equal(spec.canonical(s)[0],spec.canonical(s)[1])),'translation_m':[2.,-3.],'interpretation':'A common xy translation of robot and object leaves robot canonical + robot-relative object features unchanged, but changes their relation to fixed supports. Episode-constant xi cannot encode this.'}}
(O/'architecture_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
```
