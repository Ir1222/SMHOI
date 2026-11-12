"""Original construction, detailed limb clearance, root/COM and provenance audit."""
import gzip
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from closure_common import E,P6,active_pairs,cpu_model,inputs,reference_qpos,rot,save


def main():
    out=E/'audit_details';out.mkdir(exist_ok=False)
    manifest,clip,sources=inputs();inventory=json.loads((P6/'geometry_02/inventory.json').read_text())
    original={'object':manifest['object'],'supports':manifest['supports'],'contact':manifest['contact'],
              'compiled_original':inventory,'root_frame':'original bottom-plate centre; local Z-up; world wxyz; root distinct from COM',
              'manifest_sources':sources,'reference_sha256':clip.manifest['sha256']}
    save(out/'original_construction.json',original)
    summaries={}
    for name in ('original','single_body','smtrack_2kg'):
        m=cpu_model(name);d=mujoco.MjData(m);q=reference_qpos(m);obj,robot,hands,non=active_pairs(m)
        groups={
          'thigh':[g for g in robot if 'thigh' in m.geom(g).name],
          'knee_and_shin':[g for g in robot if 'shin' in m.geom(g).name or 'knee' in m.body(m.geom_bodyid[g]).name],
          'feet':[g for g in robot if 'foot' in m.geom(g).name],
          'torso_pelvis':[g for g in robot if any(s in m.geom(g).name for s in ('torso','pelvis'))],
          'nonhand_arms':[g for g in non if any(s in m.body(m.geom_bodyid[g]).name for s in ('shoulder','elbow','wrist'))]}
        rows=[]
        for k in range(840):
            d.qpos[:]=q[k];mujoco.mj_kinematics(m,d)
            values={}
            for group,geoms in groups.items():
                distances=[(mujoco.mj_geomDistance(m,d,g,b,10.,None),g,b) for g in geoms for b in obj]
                value,g,b=min(distances);values[group]={'distance_m':float(value),'pair':[m.geom(g).name,m.geom(b).name]}
            supports=[]
            for s in manifest['supports']:
                gid=m.geom(s['name']).id
                supports.append(min(mujoco.mj_geomDistance(m,d,g,gid,10.,None) for g in obj))
            rows.append({'frame':k,'intent':bool(clip.arrays['object_contact'][k]),'clearance':values,'support_min_pair_distance_m':supports})
        phases={'all':range(840),'pre_contact':range(260),'ON':range(260,550),'post_contact':range(550,840)}
        summary={phase:{group:min(rows[k]['clearance'][group]['distance_m'] for k in indices) for group in groups} for phase,indices in phases.items()}
        summary['geom_groups']={k:[m.geom(g).name for g in gs] for k,gs in groups.items()}
        summary['first_last_support_gaps_m']=[rows[0]['support_min_pair_distance_m'],rows[-1]['support_min_pair_distance_m']]
        summaries[name]=summary
        with gzip.open(out/(name+'_clearances.json.gz'),'wt') as f:json.dump(rows,f)
    a=clip.arrays;p=a['object_pos_w'];r=rot(a['object_quat_w']);interval=(r[1:]*r[:-1].inv()).as_rotvec()/.02
    summary={'object_root_min_world_m':p.min(0),'object_root_max_world_m':p.max(0),'first_root_pose':np.r_[p[0],a['object_quat_w'][0]],
             'last_root_pose':np.r_[p[-1],a['object_quat_w'][-1]],'max_root_speed_m_s':float(np.linalg.norm(np.gradient(p.astype(float),.02,axis=0),axis=1).max()),
             'max_object_angular_speed_rad_s':float(np.linalg.norm(interval,axis=1).max()),'limb_clearance':summaries}
    save(out/'geometry_details.json',summary)
    # Verify complete pinned upstream text blobs and give real content hashes.
    provenance=json.loads((E/'upstream/provenance.json').read_text())
    names={x['path'] for x in provenance['files']}
    for p in (E/'upstream').rglob('*'):
        if not p.is_file() or p.name=='provenance.json':continue
        name=str(p.relative_to(E/'upstream'))
        if name not in names:provenance['files'].append({'path':name,'url':f"https://github.com/shuayyy/humanoid-amp-ppo/blob/{provenance['commit']}/{name}"})
    for x in provenance['files']:
        data=(E/'upstream'/x['path']).read_bytes();x['snapshot_sha256']=hashlib.sha256(data).hexdigest()
        # apply_patch preserves an extra trailing blank line when source ended\n.
        # Git blob identity is checked against both literal and one-newline trim.
        hashes=[hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest() for d in (data,data[:-1])]
        x['git_blob_matches_snapshot_or_single_added_newline']=x.get('git_blob') in hashes if x.get('git_blob') else None
    save(out/'upstream_provenance.json',provenance)
    print(json.dumps(summary,default=lambda x:x.tolist(),indent=2))


if __name__=='__main__':main()
