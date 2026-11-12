"""Separate, non-production candidate reference with original intent/root/Box."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from closure_common import cpu_model, inputs, indices, rot, save
from supermimic.hoi.reference import array_info, deterministic_npz, validate_arrays


def export(source,out):
    out.mkdir(exist_ok=False);clip=inputs()[1];a={k:v.copy() for k,v in clip.arrays.items()}
    q=np.array(json.loads(source.read_text()));m=cpu_model('single_body');d=mujoco.MjData(m)
    _,_,names,jq,_=indices(m);order=[list(a['joint_names']).index(n) for n in names]
    arms=[i for i,n in enumerate(names) if any(s in n for s in ('shoulder','elbow','wrist'))]
    for i in arms:
        a['joint_pos'][:,order[i]]=q[:,jq[i]]
        a['joint_vel'][:,order[i]]=np.gradient(q[:,jq[i]],.02)
    bodies=[i for i,n in enumerate(a['body_names']) if any(s in n for s in ('shoulder','elbow','wrist'))]
    for k in range(840):
        d.qpos[:]=q[k];mujoco.mj_kinematics(m,d)
        for i in bodies:
            b=m.body('robot/'+a['body_names'][i]).id
            a['body_pos_w'][k,i]=d.xpos[b];a['body_quat_w'][k,i]=d.xquat[b]
    for i in bodies:
        a['body_lin_vel_w'][:,i]=np.gradient(a['body_pos_w'][:,i].astype(float),.02,axis=0)
        r=rot(a['body_quat_w'][:,i]);interval=(r[1:]*r[:-1].inv()).as_rotvec()/.02
        a['body_ang_vel_w'][0,i]=interval[0];a['body_ang_vel_w'][-1,i]=interval[-1]
        a['body_ang_vel_w'][1:-1,i]=(interval[:-1]+interval[1:])/2
    validate_arrays(a)
    for k in ('object_pos_w','object_quat_w','object_contact','joint_names','body_names','fps'):
        assert np.array_equal(a[k],clip.arrays[k])
    payload=deterministic_npz(a);(out/'reference.npz').write_bytes(payload)
    meta={'schema_version':1,'kind':'minimal_hoi_motion_reference','sha256':hashlib.sha256(payload).hexdigest(),
          'fields':{k:array_info(v) for k,v in a.items()},'training_ready':False,'status':'P6_CLOSURE_CANDIDATE_NOT_PRODUCTION',
          'source_reference_sha256':clip.manifest['sha256'],'source_qpos':str(source),'source_qpos_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'candidate_object':'single_body','changed_joints':[names[i] for i in arms],
          'changed_body_arrays':[str(a['body_names'][i]) for i in bodies],
          'velocity_method':'centred finite differences; world angular velocity from SO(3) interval log and centred average',
          'unchanged':'all root/leg/waist arrays; object pose; original global bool intent; frame count/fps/names',
          'semantics':'candidate native-contact motion, not original HDMI or learned performance'}
    save(out/'reference_manifest.json',meta);print(meta['sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.source,a.output)
