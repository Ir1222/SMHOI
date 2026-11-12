"""Independent native joint PD targets for bounded contact/lift/carry/release.

Kinematic planning only. Execution must use a free Box, free G1 and original
actuators; any planned object poses are targets, never imposed dynamic states.
"""
import argparse
import json
import shutil
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from closure_common import P6, cpu_model, inputs, indices, rot, save, dump_gz
from trajectory_ik import TrajectoryIK


def main(out):
    out.mkdir(exist_ok=False);shutil.copytree(Path(__file__).parent,out/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    m=cpu_model('single_body');d=mujoco.MjData(m);bq,bv,names,jq,jv=indices(m)
    sup=inputs()[0]['supports'][0];R=rot(sup['root_quat_wxyz'])
    q=m.key_qpos[0].copy();q[bq:bq+3]=np.array(sup['root_pos_world'])+[0,0,.03];q[bq+3:bq+7]=sup['root_quat_wxyz']
    q[:2]=(q[bq:bq+3]-R.apply([.35,0,0]))[:2];q[3:7]=sup['root_quat_wxyz']
    # Use the old reference-independent keyframe arm fixture only as an IK seed.
    seed=json.loads((P6/'geometry_02/independent_bilateral.json').read_text())['state84']
    arms=[i for i,n in enumerate(names) if any(s in n for s in ('shoulder','elbow','wrist'))]
    q[jq[arms]]=np.array(seed)[7+np.array(arms)]
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    ground=next(i for i in range(m.ngeom) if 'terrain' in m.geom(i).name)
    feet=[i for i in range(m.ngeom) if 'foot' in m.geom(i).name and m.geom_contype[i]]
    gap=min(mujoco.mj_geomDistance(m,d,g,ground,10.,None) for g in feet)
    q[2]+=.0002-gap
    N=250;qs=np.tile(q,(N,1));intent=np.arange(N)<175
    # .5s grasp,1s lift,1s lateral carry,1s lower,1.5s release/settle.
    def ease(t):return 3*t*t-2*t*t*t
    for k in range(N):
        z=.08*ease(np.clip((k-25)/50,0,1))-.08*ease(np.clip((k-125)/50,0,1))
        x=.06*ease(np.clip((k-75)/50,0,1))
        qs[k,bq:bq+3]+=R.apply([x,0,z])
        release=ease(np.clip((k-175)/40,0,1))
        qs[k,jq[arms]]=(1-release)*q[jq[arms]]+release*m.key_qpos[0,jq[arms]]
    ik=TrajectoryIK('single_body',qs,intent);lo=m.jnt_range[ik.jids,0]+1e-6;hi=m.jnt_range[ik.jids,1]-1e-6
    result=least_squares(ik.fun,np.clip(ik.ref,lo,hi).ravel(),jac=ik.jac,bounds=(np.tile(lo,N),np.tile(hi,N)),tr_solver='lsmr',tr_options={'maxiter':100},max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-5,verbose=2)
    qs[:,ik.qa]=result.x.reshape(N,14);rows,summary=ik.audit(qs)
    summary.update(optimizer_success=bool(result.success),message=result.message,reference_independent=True,
                   external_Box_force=False,planned_lift_m=.08,planned_horizontal_carry_m=.06,execution_not_yet_tested=True)
    save(out/'qpos.json',qs);save(out/'summary.json',summary);dump_gz(out/'frames.json.gz',rows)
    print(json.dumps(summary,default=lambda x:x.tolist(),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
