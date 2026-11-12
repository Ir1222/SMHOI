"""P6 independent CPU dynamics of the unchanged nominal free Box and supports."""
from __future__ import annotations
import argparse
from pathlib import Path

from common import cpu_contacts, inputs, rot, save
import mujoco
import numpy as np
from supermimic.hoi.scene import _add_supports, _box_spec
from supermimic.hoi.contact_validation import penetration_summary
from asset_common import vertices


def model_for(manifest, origin=(0,0,0)):
    spec = mujoco.MjSpec()
    _add_supports(manifest, spec)
    for body in spec.bodies:
        if 'support' in body.name:
            body.pos = np.asarray(body.pos)+origin
    spec.attach(_box_spec(manifest), frame=spec.worldbody.add_frame(), prefix='box/')
    spec.worldbody.add_geom(name='ground', type=mujoco.mjtGeom.mjGEOM_PLANE,
                           size=[0,0,.05], friction=[1,.005,.0001], condim=3)
    spec.option.timestep=.005
    spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone=mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.iterations=10; spec.option.ls_iterations=20
    m=spec.compile()
    assert m.nq==7 and m.nv==6 and m.nu==0 and m.neq==0 and m.nmocap==0
    return m


def proposal(manifest):
    gs=manifest['object']['geoms']; mass=manifest['object']['inertia']['mass_kg']
    volumes=np.array([8*np.prod(g['size']) for g in gs]); masses=mass*volumes/volumes.sum()
    centers=np.array([g['pos'] for g in gs]); com=(masses[:,None]*centers).sum(0)/mass
    inertia=np.zeros((3,3))
    for g,m,c in zip(gs,masses,centers):
        h=np.array(g['size']); diag=m/3*np.array([h[1]**2+h[2]**2,h[0]**2+h[2]**2,h[0]**2+h[1]**2])
        r=rot(g['quat']).as_matrix(); offset=c-com
        inertia+=r@np.diag(diag)@r.T+m*(np.dot(offset,offset)*np.eye(3)-np.outer(offset,offset))
    points=np.concatenate([vertices(g) for g in gs])
    eigval,eigvec=np.linalg.eigh(inertia)
    if np.linalg.det(eigvec)<0: eigvec[:,0]*=-1
    return {'status':'PROPOSAL_ONLY_NOT_APPLIED',
            'assumption':'uniform density seven separately weighted cuboids; overlapping volumes counted per component; no material measurement',
            'original':manifest['object']['inertia'], 'envelope_min':points.min(0),'envelope_max':points.max(0),
            'component_mass_kg':dict(zip([g['name'] for g in gs],masses)),
            'proposal_com_root_m':com,'proposal_inertia_root_axes_at_com_kg_m2':inertia,
            'principal_inertia_kg_m2':eigval,'principal_axes_wxyz':rot_from_matrix(eigvec),
            'expected_effect':'moves COM inside geometry, changes gravitational/contact torque arms and angular acceleration; likely reduces artificial pendulum stability',
            'required_approval':'new versioned asset profile, material/mass assumptions, all P1 pair/sign/reset and P6 dynamics/solver/support/release regressions, separate reference comparison; original preserved'}


def rot_from_matrix(matrix):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(matrix).as_quat(scalar_first=True)


def run_case(case, manifest, output):
    folder=output/case['name']; folder.mkdir(exist_ok=False)
    origin=np.array(case.get('origin',[0,0,0]))
    m=model_for(manifest,origin); d=mujoco.MjData(m); bid=m.body('box/bread_box').id
    d.qpos[:3]=case['position']; d.qpos[3:7]=case['quaternion']; d.qvel[:]=0
    mujoco.mj_forward(m,d)
    initial=d.qpos.copy(); records=[]; contacts=[]
    dt=m.opt.timestep; steps=round(case['seconds']/dt)
    for k in range(steps):
        time=k*dt
        wrench=np.zeros(6)
        if case['kind']=='force':wrench[:3]=case['force']
        if case['kind']=='torque':wrench[3:]=case['torque']
        if case['kind']=='lift' and .5<=time<.75:wrench[2]=24.
        d.xfrc_applied[:]=0; d.xfrc_applied[bid]=wrench
        assert not np.any(d.qfrc_applied)
        # mj_step owns all state changes after this case's initial reset.
        mujoco.mj_step(m,d)
        new_contacts=cpu_contacts(m,d,k); contacts.extend(new_contacts)
        q=d.qpos[3:7].copy(); r=rot(q); omega=r.apply(d.qvel[3:6])
        com_velocity=d.qvel[:3]+np.cross(omega,r.apply(m.body_ipos[bid]))
        forces={cat:np.zeros(3) for cat in ('support','ground','left','right','robot_body')}
        for c in new_contacts:
            sign=1 if c['names'][1].startswith('box/') else -1
            forces[c['category']]+=sign*c['world_force_on_geom2_N']
        records.append({'sample':k,'time_end_s':float(d.time),'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),
                        'com_velocity_world':com_velocity,'omega_world':omega,'external_wrench':wrench,
                        'box_contact_force_world':forces,'solver_niter':d.solver_niter.copy(),
                        'warnings':{mujoco.mjtWarning(i).name:int(d.warning[i].number) for i in range(len(d.warning))},
                        'ncon':d.ncon,'orientation_error_rad':float((rot(initial[3:7]).inv()*r).magnitude())})
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            break
    tail=records[-min(100,len(records)):]
    speeds=[np.linalg.norm(x['com_velocity_world']) for x in tail]
    ang=[np.linalg.norm(x['omega_world']) for x in tail]
    support_z=[x['box_contact_force_world']['support'][2] for x in tail]
    ground_z=[x['box_contact_force_world']['ground'][2] for x in tail]
    analytical=None
    if case['kind'] in ('force','torque','freefall'):
        force=np.array(case.get('force',[0,0,0])); torque=np.array(case.get('torque',[0,0,0]))
        R=rot(initial[3:7]).as_matrix()@rot(m.body_iquat[bid]).as_matrix()
        I=R@np.diag(m.body_inertia[bid])@R.T
        expected_v=(force/m.body_mass[bid]+m.opt.gravity)*dt
        expected_omega=np.linalg.solve(I,torque)*dt
        analytical={'expected_first_com_velocity':expected_v,'expected_first_world_omega':expected_omega,
                    'first_velocity_error':float(np.linalg.norm(records[0]['com_velocity_world']-expected_v)),
                    'first_omega_error':float(np.linalg.norm(records[0]['omega_world']-expected_omega))}
    result={'case':case,'samples':len(records),'warnings':records[-1]['warnings'],
            'solver_iterations_max':max(int(x['solver_niter'].max()) for x in records),
            'initial_qpos':initial,'final_qpos':d.qpos.copy(),'final_qvel':d.qvel.copy(),
            'max_rise_m':max(x['qpos'][2]-initial[2] for x in records),
            'final_displacement_m':float(np.linalg.norm(d.qpos[:3]-initial[:3])),
            'tail_max_COM_speed_m_s':max(speeds),'tail_max_angular_speed_rad_s':max(ang),
            'tail_max_orientation_error_rad':max(x['orientation_error_rad'] for x in tail),
            'tail_mean_support_z_N':float(np.mean(support_z)),'tail_mean_ground_z_N':float(np.mean(ground_z)),
            'external_impulse_Ns':sum((r['external_wrench'][:3]*dt for r in records),np.zeros(3)),
            'analytical':analytical,'penetration':penetration_summary(contacts,len(records)),
            'stable_tail':bool(max(speeds)<.02 and max(ang)<.1 and max(x['orientation_error_rad'] for x in tail)<.05
                               and (min(support_z)>1 or min(ground_z)>1)),
            'no_state_reposition_after_reset':True}
    save(folder/'trajectory.json',records);save(folder/'contacts.json',contacts);save(folder/'summary.json',result)
    print(case['name'],'stable',result['stable_tail'],'max rise',result['max_rise_m'],flush=True)
    return result


def main(output):
    output.mkdir(parents=True,exist_ok=False); m,_,sources=inputs();save(output/'mass_proposal.json',proposal(m))
    cases=[]
    for kind,extra in [('freefall',{}),('force',{'force':[4.,0,0]}),('torque',{'torque':[.02,0,0]})]:
        for angle in (0.,.6):
            from scipy.spatial.transform import Rotation
            cases.append({'name':f'{kind}_rotation{angle}','kind':kind,'seconds':.1,'position':[0,0,3],
                          'quaternion':Rotation.from_rotvec([0,0,angle]).as_quat(scalar_first=True).tolist(),**extra})
    for i,s in enumerate(m['supports']):
        for kind,dz in [('settle',.04),('drop',.13),('lift',.03)]:
            cases.append({'name':f'support{i}_{kind}','kind':kind,'seconds':3.,
                          'position':(np.array(s['root_pos_world'])+[0,0,dz]).tolist(),'quaternion':s['root_quat_wxyz']})
    cases.append({'name':'floor_drop','kind':'drop','seconds':3.,'position':[0,0,.11],'quaternion':[1,0,0,0]})
    s=m['supports'][0];origin=np.array([4,-4,0])
    cases.append({'name':'support0_offset_settle','kind':'settle','seconds':3.,'origin':origin.tolist(),
                  'position':(np.array(s['root_pos_world'])+origin+[0,0,.04]).tolist(),'quaternion':s['root_quat_wxyz']})
    save(output/'config.json',{'cases':cases,'sources':sources,'dt':.005,'iterations':10,'ls_iterations':20,
                             'mode':'isolated free Box, unchanged mass/COM/inertia/geoms, explicit world COM wrench only in named tests'})
    results=[run_case(c,m,output) for c in cases];save(output/'results.json',results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
