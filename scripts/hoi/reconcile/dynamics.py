"""Short original-PD feasibility probes. No learned policy, root aid or box actuation."""
import argparse,copy,re
import mujoco,numpy as np
from common import *
from supermimic.hoi.asset_manifest import build_audit_scene,apply_hand_profile,load_manifest

def build_model(profile='native',raise_m=0.,robot=True,dt=.005):
    manifest=load_manifest(MANIFEST);candidate=copy.deepcopy(manifest)
    for s in candidate['supports']:s['root_pos_world'][2]+=raise_m
    asset=build_audit_scene(candidate,[0,0,1],[1,0,0,0])
    pd=[]
    if robot:
        from mjlab.entity import Entity
        from mjlab.asset_zoo.robots import get_g1_robot_cfg,G1_ACTION_SCALE
        entity=Entity(get_g1_robot_cfg());spec=entity.spec;nominal=spec.compile()
        for i in range(nominal.nu):
            jid=nominal.actuator_trnid[i,0];name=nominal.joint(jid).name
            scales=[v for k,v in G1_ACTION_SCALE.items() if re.fullmatch(k,name)]
            assert len(scales)==1
            pd.append({'joint':name,'kp':float(nominal.actuator_gainprm[i,0]),'kd':float(-nominal.actuator_biasprm[i,2]),'effort_limit':nominal.actuator_forcerange[i].tolist(),'armature':float(nominal.dof_armature[nominal.jnt_dofadr[jid]]),'action_scale':float(scales[0]),'default_joint_pos':float(nominal.key_qpos[0,nominal.jnt_qposadr[jid]])})
        for key in list(spec.keys):spec.delete(key)
        if profile=='hdmi':apply_hand_profile(spec,manifest,'hdmi_hands_only_candidate')
        elif profile!='native':raise ValueError(profile)
        # Match deterministic native evaluation event (foot friction=1).
        for g in spec.geoms:
            if re.fullmatch(r'(left|right)_foot[1-7]_collision',g.name):g.friction[0]=1.
        spec.attach(asset,frame=spec.worldbody.add_frame(),prefix='',suffix='')
    else:spec=asset
    spec.worldbody.add_geom(name='ground',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.05],condim=3,priority=0,friction=[1,.005,.0001])
    spec.option.timestep=dt;spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations=10;spec.option.ls_iterations=20;spec.option.cone=mujoco.mjtCone.mjCONE_PYRAMIDAL
    m=spec.compile()
    assert m.neq==0 and m.nmocap==0 and m.nu==(29 if robot else 0)
    assert sum(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)==(2 if robot else 1)
    for a in range(m.nu):assert m.jnt_type[m.actuator_trnid[a,0]]==mujoco.mjtJoint.mjJNT_HINGE
    assert m.body_mass[m.body('bread_box').id]==2.
    return m,pd

def initialize(m,d,ref,frame,robot):
    """Only state-write location. All later changes are original PD ctrl + mj_step."""
    names=ref['body_names'].tolist()
    if robot:
        root=names.index('pelvis');j=m.joint('floating_base_joint').id;q=m.jnt_qposadr[j];v=m.jnt_dofadr[j]
        d.qpos[q:q+3]=ref['body_pos_w'][frame,root];d.qpos[q+3:q+7]=ref['body_quat_w'][frame,root]
        d.qvel[v:v+3]=ref['body_lin_vel_w'][frame,root];d.qvel[v+3:v+6]=rot(ref['body_quat_w'][frame,root]).inv().apply(ref['body_ang_vel_w'][frame,root])
        for i,name in enumerate(ref['joint_names']):
            j=m.joint(str(name)).id;d.qpos[m.jnt_qposadr[j]]=ref['joint_pos'][frame,i];d.qvel[m.jnt_dofadr[j]]=ref['joint_vel'][frame,i]
    j=m.joint('bread_box_root').id;q=m.jnt_qposadr[j];v=m.jnt_dofadr[j]
    d.qpos[q:q+3]=ref['box_pos_w'][frame];d.qpos[q+3:q+7]=ref['box_quat_w'][frame]
    d.qvel[v:v+3]=ref['box_lin_vel_w'][frame];d.qvel[v+3:v+6]=rot(ref['box_quat_w'][frame]).inv().apply(ref['box_ang_vel_w'][frame])
    mujoco.mj_forward(m,d)

def point_velocity(m,d,body,point):
    vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,body,vel,0)
    return vel[3:]+np.cross(vel[:3],point-d.xpos[body])

def set_original_pd(m,d,ref,frame,pd):
    names=ref['joint_names'].tolist()
    for i,entry in enumerate(pd):
        target=float(ref['joint_pos'][frame,names.index(entry['joint'])]);action=(target-entry['default_joint_pos'])/entry['action_scale']
        d.ctrl[i]=action*entry['action_scale']+entry['default_joint_pos']

def run_case(case,ref,output):
    name=case['name'];folder=output/name
    folder.mkdir(parents=True,exist_ok=False)
    m,pd=build_model(case['profile'],case['raise_m'],case['robot'],case.get('dt',.005));d=mujoco.MjData(m)
    initialize(m,d,ref,case['start'],case['robot']);box=m.body('bread_box').id
    start_pos=d.xpos[box].copy();start_q=d.xquat[box].copy();records=[];contacts=[];slip={'left':0.,'right':0.};fall=None
    geom_box={g for g in range(m.ngeom) if m.geom_bodyid[g]==box}
    for step in range(round(case['seconds']/m.opt.timestep)+1):
        f=min(case['start']+step//round(.02/m.opt.timestep),len(ref['joint_pos'])-1)
        if case['robot']:set_original_pd(m,d,ref,f,pd)
        assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
        mujoco.mj_forward(m,d)
        forces={k:0. for k in ('left','right','support0','support1','ground','other')};maxpen=0.;handpen=0.;speed={'left':[],'right':[]}
        for ci in range(d.ncon):
            c=d.contact[ci];g1,g2=int(c.geom1),int(c.geom2)
            if g1 not in geom_box and g2 not in geom_box:continue
            other=g2 if g1 in geom_box else g1;body=int(m.geom_bodyid[other]);bn=m.body(body).name;gn=m.geom(other).name
            cat=next((s for s in ('left','right') if bn.startswith(s+'_wrist')),None)
            if cat is None:cat=gn if gn in ('support0','support1','ground') else 'other'
            force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);normal=c.frame.reshape(3,3)[0]
            rel=point_velocity(m,d,body,c.pos)-point_velocity(m,d,box,c.pos);tan=rel-np.dot(rel,normal)*normal
            forces[cat]+=force[0];maxpen=max(maxpen,max(0,-c.dist))
            if cat in ('left','right'):
                handpen=max(handpen,max(0,-c.dist))
                if force[0]>1:speed[cat].append((float(force[0]),float(np.linalg.norm(tan))))
            contacts.append({'time_s':d.time,'reference_frame':f,'category':cat,'geom1':m.geom(g1).name,'geom2':m.geom(g2).name,'distance_m':c.dist,'normal_force_N':force[0],'tangent_force_N':np.linalg.norm(force[1:3]),'tangent1_force_N':force[1],'tangent2_force_N':force[2],**{f'normal_{a}':float(x) for a,x in zip('xyz',normal)},'relative_tangent_speed_m_s':np.linalg.norm(tan),**{f'contact_{a}':float(x) for a,x in zip('xyz',c.pos)}})
        for side in ('left','right'):
            if speed[side]:slip[side]+=sum(frc*v for frc,v in speed[side])/sum(frc for frc,v in speed[side])*m.opt.timestep*(step<round(case['seconds']/m.opt.timestep))
        bq=d.xquat[box].copy();bp=d.xpos[box].copy();bv=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,box,bv,0)
        row={'time_s':d.time,'reference_frame':f,'desired_contact':int(ref['object_contact'][f]),'box_error_m':float(np.linalg.norm(bp-ref['box_pos_w'][f])),'box_displacement_m':float(np.linalg.norm(bp-start_pos)),'box_rotation_error_rad':float((rot(bq).inv()*rot(ref['box_quat_w'][f])).magnitude()),'box_speed_m_s':float(np.linalg.norm(bv[3:])),'box_angular_speed_rad_s':float(np.linalg.norm(bv[:3])),'max_box_penetration_m':maxpen,'max_hand_box_penetration_m':handpen,'left_slip_path_m':slip['left'],'right_slip_path_m':slip['right'],**{f'box_{a}':float(v) for a,v in zip('xyz',bp)},**{f'box_q{a}':float(v) for a,v in zip('wxyz',bq)},**{f'{k}_force_N':float(v) for k,v in forces.items()}}
        if case['robot']:
            rp=d.xpos[m.body('pelvis').id];rq=d.xquat[m.body('pelvis').id];rn=ref['body_names'].tolist().index('pelvis')
            errs=[d.qpos[m.jnt_qposadr[m.joint(str(n)).id]]-ref['joint_pos'][f,i] for i,n in enumerate(ref['joint_names'])]
            row.update(root_z=float(rp[2]),root_error_m=float(np.linalg.norm(rp-ref['body_pos_w'][f,rn])),joint_rmse_rad=float(np.sqrt(np.mean(np.array(errs)**2))),max_actuator_force_Nm=float(abs(d.actuator_force).max()))
            if fall is None and rp[2]<.35:fall=float(d.time)
        records.append(row)
        if not np.isfinite(d.qpos).all() or np.linalg.norm(d.qvel)>1e4:raise RuntimeError('Unstable numerical state '+name)
        if step<round(case['seconds']/m.opt.timestep):mujoco.mj_step(m,d)
    write_csv(folder/'trajectory.csv',records)
    if contacts:write_csv(folder/'contacts.csv',contacts)
    tail=[r for r in records if r['time_s']>=case['seconds']-.5];active=[r for r in records if r['desired_contact']]
    end=records[-1];final_support='support1' if case['start']>=500 else 'support0'
    stable=bool(all(r[final_support+'_force_N']>1 for r in tail) and max(r['box_speed_m_s'] for r in tail)<.02 and max(r['box_angular_speed_rad_s'] for r in tail)<.1 and max(r['box_rotation_error_rad'] for r in tail)<.05 and end['box_error_m']<.02)
    summary={**case,'duration_actual_s':float(d.time),'initial_hand_penetration_m':records[0]['max_hand_box_penetration_m'],'max_hand_penetration_m':max(r['max_hand_box_penetration_m'] for r in records),'max_box_penetration_m':max(r['max_box_penetration_m'] for r in records),'box_position_rmse_m':float(np.sqrt(np.mean([r['box_error_m']**2 for r in records]))),'final_box_error_m':end['box_error_m'],'final_box_pos':bp,'final_box_speed_m_s':end['box_speed_m_s'],'root_fall_time_s':fall,'root_final_error_m':end.get('root_error_m'), 'joint_final_rmse_rad':end.get('joint_rmse_rad'),'bilateral_force_fraction_during_intent':float(np.mean([r['left_force_N']>1 and r['right_force_N']>1 for r in active])) if active else None,'force_peaks_N':{k:max(r[k+'_force_N'] for r in records) for k in forces},'slip_path_m':slip,'terminal_stable_last_half_second':stable,'tail_mean_support_force_N':float(np.mean([r[final_support+'_force_N'] for r in tail])),'no_assistance_verified':True,'nq':m.nq,'nv':m.nv,'nu':m.nu,'neq':m.neq,'nmocap':m.nmocap,'pd':pd}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='dynamics_v1');p.add_argument('--smoke',action='store_true');a=p.parse_args();verify()
    output=OUT/a.output;output.mkdir(exist_ok=False);ref=load(NPZ);cases=[]
    for profile in ('native','hdmi'):
        for dz in (0.,.004,.01):
            for start,seconds in [(250,1.2),(400,1.0),(530,2.0),(550,3.0),(839,3.0)]:
                cases.append({'name':f'{profile}_z{round(dz*1000)}_f{start}','profile':profile,'raise_m':dz,'robot':True,'start':start,'seconds':seconds})
    for dz in (0.,.004,.01):
        for f in (0,839):cases.append({'name':f'box_only_z{round(dz*1000)}_f{f}','profile':'native','raise_m':dz,'robot':False,'start':f,'seconds':3.0})
    if a.smoke:cases=cases[:1]+cases[-2:]
    results=[]
    for case in cases:
        r=run_case(case,ref,output);results.append(r);print(case['name'],'box final error',round(r['final_box_error_m'],4),'stable',r['terminal_stable_last_half_second'],'fall',r['root_fall_time_s'],flush=True)
    (output/'summary.json').write_text(json.dumps(results,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x)+'\n');verify()
if __name__=='__main__':main()
