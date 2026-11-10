"""All-frame geometric audit, not a dynamic rollout or measured contact test."""
import csv,itertools,json
from pathlib import Path
import mujoco,numpy as np
from asset_common import *
from audit_npz import load
from audit_robot import target_spec

def box_point(p,g):
    """Signed distance, closest boundary and outward normal of one oriented box."""
    R=rot(g['quat']); x=R.inv().apply(np.asarray(p)-g['pos']); h=np.array(g['size'])
    q=np.abs(x)-h; closest=np.clip(x,-h,h)
    if (q>0).any():
        sd=np.linalg.norm(np.maximum(q,0)); n=(x-closest)/sd
    else:
        axis=int(np.argmax(q)); sign=1 if x[axis]>=0 else -1
        closest[axis]=sign*h[axis]; sd=q[axis]; n=np.eye(3)[axis]*sign
    return float(sd),R.apply(closest)+g['pos'],R.apply(n)

def compound_point(p,geoms):
    """Exact exterior/on-boundary distance; min primitive SDF inside overlaps.

    Targets are on exposed walls, so their reported signed distance is exact.
    Interior overlap values for diagnostic EEF points are NOT union exit depths.
    """
    vals=[box_point(p,g) for g in geoms]; k=int(np.argmin([x[0] for x in vals]));return k,vals[k]

def point_primitive(p,g):
    if g['type']=='box': return box_point(p,g)[0]
    x=rot(g['quat']).inv().apply(np.asarray(p)-g['pos']);r,h=g['size'][:2]
    c=np.array([0,0,np.clip(x[2],-h,h)]);return float(np.linalg.norm(x-c)-r)

def comparison_model(box,hand):
    s=mujoco.MjSpec(); b=s.worldbody.add_body(name='box',mocap=True)
    for g in box:b.add_geom(name='box_'+g['name'],type=mujoco.mjtGeom.mjGEOM_BOX,pos=g['pos'],quat=g['quat'],size=g['size'])
    for body in sorted({g['body'] for g in hand}):
        b=s.worldbody.add_body(name=body,mocap=True)
        for g in hand:
            if g['body']!=body:continue
            b.add_geom(name=g['name'],type=getattr(mujoco.mjtGeom,'mjGEOM_'+g['type'].upper()),pos=g['pos'],quat=g['quat'],size=g['size'])
    m=s.compile();return m,mujoco.MjData(m)

def stats(x):
    x=np.asarray(x);return dict(zip(('min','p05','p50','p95','max'),map(float,np.percentile(x,[0,5,50,95,100]))))
def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    d=load(Path('/home/han/pyx/hdmi_bread_box_hoi_v1.npz'))
    box=json.loads((OUT/'box_audit.json').read_text()); robot=json.loads((OUT/'robot_audit.json').read_text())
    gs=box['geoms']; names=d['body_names'].tolist(); F=len(d['joint_pos']);mask=d['object_contact'].astype(bool)
    targets=np.array(box['task_contact']['contact_target_pos_offset']);offsets=np.array(box['task_contact']['contact_eef_pos_offset'])
    assert np.allclose(targets,d['contact_target_root'],atol=1e-8,rtol=0)
    assert np.allclose(offsets,d['contact_eef_offset'],atol=1e-8,rtol=0)
    br=rot(d['box_quat_w']); wr={s:rot(d['body_quat_w'][:,names.index(s+'_wrist_yaw_link')]) for s in ('left','right')}
    point_rows=[]; summary={'target_tolerance_m':1e-7,'penetration_reporting_tolerance_m':1e-4,'targets':[],'modes':{}}
    for si,side in enumerate(('left','right')):
        k,(sd,cp,n)=compound_point(targets[si],gs)
        summary['targets'].append({'side':side,'signed_distance_m':sd,'geom':gs[k]['name'],'normal_root':n.tolist(),'closest_root':cp.tolist(),'classification':'exposed side wall, below handle'})
        wp=d['body_pos_w'][:,names.index(side+'_wrist_yaw_link')];eef=wp+wr[side].apply(np.broadcast_to(offsets[si],(F,3)))
        tw=d['box_pos_w']+br.apply(np.broadcast_to(targets[si],(F,3)));normals=br.apply(np.broadcast_to(n,(F,3)))
        error=np.linalg.norm(tw-eef,axis=1);summary[side+'_eef_error_contact_m']=stats(error[mask])
        for f in range(F):
            row={'frame':f,'time_s':f/50,'side':side,'desired_contact':int(mask[f]),'target_surface_sdf_m':sd,'closest_box_geom':gs[k]['name'],'eef_error_m':error[f]}
            for key,val in [('target_world',tw[f]),('eef_world',eef[f]),('wrist_world',wp[f]),('surface_normal_world',normals[f])]:
                row.update({key+'_'+a:float(v) for a,v in zip('xyz',val)})
            row.update({f'wrist_quat_{a}':float(v) for a,v in zip('wxyz',d['body_quat_w'][f,names.index(side+'_wrist_yaw_link')])})
            point_rows.append(row)
    write_csv(OUT/'contact_points.csv',point_rows)
    modes={'native':[g for g in robot['native'] if g['contype'] and 'wrist' in g['body']],
           'hdmi_hand':[g for g in robot['hdmi_comparison_geoms'] if 'wrist' in g['body']]}
    pair_rows=[];frame_rows=[];models={}
    for mode,hand in modes.items():
        m,md=comparison_model(gs,hand);models[mode]=(m,md,hand)
        bg=[m.geom('box_'+g['name']).id for g in gs]
        for f in range(F):
            md.mocap_pos[m.body('box').mocapid]=d['box_pos_w'][f];md.mocap_quat[m.body('box').mocapid]=d['box_quat_w'][f]
            for body in {g['body'] for g in hand}:
                bid=m.body(body).mocapid;idx=names.index(body)
                md.mocap_pos[bid]=d['body_pos_w'][f,idx];md.mocap_quat[bid]=d['body_quat_w'][f,idx]
            mujoco.mj_forward(m,md)
            for si,side in enumerate(('left','right')):
                hs=[g for g in hand if g['body'].startswith(side)]
                dist=[]
                for g in hs:
                    hid=m.geom(g['name']).id
                    for bi,b in enumerate(bg):
                        line=np.zeros(6);v=mujoco.mj_geomDistance(m,md,hid,b,10,line)
                        dist.append((v,g['name'],gs[bi]['name']))
                        pair_rows.append({'frame':f,'mode':mode,'side':side,'hand_geom':g['name'],'box_geom':gs[bi]['name'],'signed_distance_m':v})
                closest=min(dist)
                ti=si*F+f;target=np.array([point_rows[ti]['target_world_'+a] for a in 'xyz'])
                eef=np.array([point_rows[ti]['eef_world_'+a] for a in 'xyz'])
                sdf=min(point_primitive(rot(d['body_quat_w'][f,names.index(g['body'])]).inv().apply(target-d['body_pos_w'][f,names.index(g['body'])]),g) for g in hs)
                # Counterfactual exact EEF alignment, current wrist orientation preserved.
                shift=target-eef
                for body in {g['body'] for g in hs}:md.mocap_pos[m.body(body).mocapid]+=shift
                mujoco.mj_forward(m,md)
                aligned=min(mujoco.mj_geomDistance(m,md,m.geom(g['name']).id,b,10,None) for g in hs for b in bg)
                for body in {g['body'] for g in hs}:md.mocap_pos[m.body(body).mocapid]-=shift
                mujoco.mj_forward(m,md)
                frame_rows.append({'frame':f,'time_s':f/50,'mode':mode,'side':side,'desired_contact':int(mask[f]),'min_hand_box_signed_distance_m':closest[0],'closest_hand_geom':closest[1],'closest_box_geom':closest[2],'target_in_hand_primitive_sdf_m':sdf,'exact_eef_alignment_min_pair_distance_m':aligned})
        ms={}
        for side in ('left','right'):
            rows=[x for x in frame_rows if x['mode']==mode and x['side']==side and x['desired_contact']]
            v=np.array([x['min_hand_box_signed_distance_m'] for x in rows])
            ms[side]={'min_pair_signed_distance_contact_m':stats(v),'penetrating_frames_over_0_1mm':int((v < -1e-4).sum()),'separated_frames_over_1mm':int((v>1e-3).sum()),'frames':len(v),'max_penetration_frame':rows[int(np.argmin(v))]['frame'],'aligned_eef_pair_distance_m':stats([x['exact_eef_alignment_min_pair_distance_m'] for x in rows])}
        summary['modes'][mode]=ms
    write_csv(OUT/'hand_box_pairs.csv',pair_rows);write_csv(OUT/'hand_box_frames.csv',frame_rows)
    # Actual target kinematics checked independently against preserved recorded poses.
    fm=target_spec().compile();fd=mujoco.MjData(fm);fk=[]
    for f in range(F):
        root=names.index('pelvis');fd.qpos[:3]=d['body_pos_w'][f,root];fd.qpos[3:7]=d['body_quat_w'][f,root]
        for j,name in enumerate(d['joint_names']):fd.qpos[fm.jnt_qposadr[fm.joint(str(name)).id]]=d['joint_pos'][f,j]
        mujoco.mj_forward(fm,fd)
        for side in ('left','right'):
            name=side+'_wrist_yaw_link';bid=fm.body(name).id;i=names.index(name)
            fk.append({'frame':f,'side':side,'position_error_m':float(np.linalg.norm(fd.xpos[bid]-d['body_pos_w'][f,i])),'rotation_error_rad':float((rot(fd.xquat[bid]).inv()*rot(d['body_quat_w'][f,i])).magnitude())})
    write_csv(OUT/'native_fk_check.csv',fk);summary['native_fk_vs_recorded']={k:stats([x[k] for x in fk]) for k in ('position_error_m','rotation_error_rad')}
    summary['method']='MuJoCo mj_geomDistance on primitive geoms at recorded body poses; negative = primitive overlap, positive = separation. Min pair penetration is not compound minimum translation or measured force. No dynamics stepped. Exact EEF alignment is a geometric translation probe, not a joint-reachability proof.'
    summary['representative_frames']={'before_contact':250,'first_contact':260,'early_sustained':280,'lift':298,'carry':400,'placement':530,'last_contact':549,'release':550,'final':839}
    save('contact_summary.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
