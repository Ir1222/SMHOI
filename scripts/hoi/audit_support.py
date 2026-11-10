"""Static support geometry and all-frame box/support consistency."""
from pathlib import Path
import json,itertools
import mujoco,numpy as np
from asset_common import *
from audit_npz import load
from audit_contact import comparison_model,write_csv,stats

def main():
    d=load(Path('/home/han/pyx/hdmi_bread_box_hoi_v1.npz'));usd=json.loads((OUT/'usd_reread.json').read_text())
    gs=json.loads((OUT/'box_audit.json').read_text())['geoms'];supports=[]
    source=selected_source(SOURCE/'src/mjlab_g1/hoi/env.py',{'support_spec'},{'mujoco':mujoco,'np':np})['support_spec']('/home/han/pyx/hdmi_bread_box_hoi_v1.npz').compile()
    for i in range(2):
        raw=usd[f'support{i}'];c=raw['colliders'][0];a=c['attributes'];h=(np.array(a['xformOp:scale'])*a['size']/2)
        pos=d['support_pos_w'][0,i];quat=d['support_quat_w'][0,i];R=rot(quat)
        assert np.array_equal(d['support_pos_w'][:,i],np.broadcast_to(pos,d['support_pos_w'][:,i].shape))
        assert np.array_equal(d['support_quat_w'][:,i],np.broadcast_to(quat,d['support_quat_w'][:,i].shape))
        supports.append({'name':f'support{i}','body':f'support{i}','type':'box','pos':a['xformOp:translate'],'quat':a['xformOp:orient'],'size':h.tolist(),
          'root_pos_world':pos.tolist(),'root_quat_wxyz':quat.tolist(),'static':True,'pose_exactly_constant':True,
          'top_center_world':(pos+R.apply(np.array(a['xformOp:translate'])+rot(a['xformOp:orient']).apply([0,0,h[2]]))).tolist(),
          'top_normal_world':(R*rot(a['xformOp:orient'])).apply([0,0,1]).tolist(),
          'source_builder_geom':geom(source,i),'usd_source':{'path':raw['path'],'sha256':raw['sha256']},
          'raw_mass_kg':raw['bodies'][0]['attributes']['physics:mass'],'task_mass_override_kg':.5,
          'static_origin':'objects.py SUPPORT0_CFG/SUPPORT1_CFG: kinematic_enabled=True. MuJoCo fixed support chosen because recorded poses are constant.',
          'friction_origin':'MuJoCo source support_spec fixed [1,.005,.0001], condim3 priority1; not a claim of PhysX material equivalence'})
    m,md=comparison_model(gs,supports);rows=[]
    for f in range(len(d['joint_pos'])):
        md.mocap_pos[m.body('box').mocapid]=d['box_pos_w'][f];md.mocap_quat[m.body('box').mocapid]=d['box_quat_w'][f]
        for s in supports:md.mocap_pos[m.body(s['name']).mocapid]=s['root_pos_world'];md.mocap_quat[m.body(s['name']).mocapid]=s['root_quat_wxyz']
        mujoco.mj_forward(m,md);vs=rot(d['box_quat_w'][f]).apply(np.concatenate([vertices(g) for g in gs]))+d['box_pos_w'][f]
        for s in supports:
            distances=[mujoco.mj_geomDistance(m,md,m.geom('box_'+g['name']).id,m.geom(s['name']).id,10,None) for g in gs]
            rows.append({'frame':f,'support':s['name'],'box_root_z':float(d['box_pos_w'][f,2]),'box_geometry_min_world_z':vs[:,2].min(),'support_top_z':s['top_center_world'][2],
              'lowest_box_distance_to_top_plane_m':float(((vs-s['top_center_world'])@np.array(s['top_normal_world'])).min()),'min_box_support_pair_distance_m':min(distances),'closest_box_geom':gs[int(np.argmin(distances))]['name']})
    write_csv(OUT/'support_frames.csv',rows)
    report={'supports':supports,'start':rows[:2],'end':rows[-2:],'all_frames':{s['name']:stats([r['min_box_support_pair_distance_m'] for r in rows if r['support']==s['name']]) for s in supports},
      'acceptance_tolerance_m':.001,'warning':'Plane distances alone ignore support XY footprint; use actual primitive pair distance to establish spatial separation/overlap. Negative means penetration, not measured force.'}
    save('support_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='supports'},indent=2))
if __name__=='__main__':main()
