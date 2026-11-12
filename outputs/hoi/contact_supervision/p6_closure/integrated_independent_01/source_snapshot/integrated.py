"""Selected construction, original actuators, free roots: bounded C4 tests."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import warp as wp
from closure_common import E, P6, gpu_env, inputs, save
from run_p6 import run_intervals
from supermimic.hoi.contact_validation import actual_contact
from supermimic.hoi.reference import ReferenceClip
from supermimic.hoi.recorder import install_recording
from supermimic.hoi.solver_precision import bracket_precision_guard
from supermimic.hoi.state import JointStateIO, ReferenceResetCache


def main(args):
    out=args.output;out.mkdir(exist_ok=False);shutil.copytree(Path(__file__).parent,out/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    torch.manual_seed(606);torch.set_num_threads(1);wp.config.kernel_cache_dir='/tmp/smhoi_p6_warp_cache'
    with bracket_precision_guard() as profile:
        save(out/'solver_profile.json',profile);env,manifest=gpu_env('single_body');io=JointStateIO(env)
        term=env.action_manager.get_term('joint_pos');results={}
        try:
            if args.mode=='calibration':
                cal=json.loads((P6/'calibration.json').read_text());tau=cal['thresholds_N_LR']
                for roll in (0.,.31,.63):
                    for depth in (.0005,.001,.002):
                        name=f'roll{roll}_depth{depth}'
                        states=json.loads((P6/'calibration_02'/name/'initial.json').read_text())['state84']
                        result,rows,obs=run_intervals(env,io,states,1,out/name)
                        contacts=actual_contact(np.array(rows[0]['force_raw_world']),tau,[.25,.25]) if rows else None
                        result['P6_threshold_contact']=contacts
                        result['expected']=[[False,False],[True,False],[False,True],[True,True]]
                        result['definition_check']=bool(contacts is not None and np.array_equal(contacts['hands'],result['expected']))
                        results[name]=result
            elif args.mode=='independent':
                q=np.array(json.loads((E/'independent_controller_01/qpos.json').read_text()))
                rq=io.robot_q.cpu().numpy();bq=io.box_q.cpu().numpy()
                state=np.r_[q[0,rq],np.zeros(35),q[0,bq],np.zeros(6)]
                def action(k):
                    target=torch.tensor(q[k,rq[7:]],device=env.device,dtype=torch.float32)[None].expand(4,-1)
                    return (target-term._offset)/term._scale
                results['native_PD_contact_lift_carry_release']=run_intervals(env,io,[state.copy() for _ in range(4)],249,out/'native_PD_contact_lift_carry_release',action)[0]
            else:
                ref=inputs()[1] if args.mode=='original' else ReferenceClip(E/'reference_candidate_v1/reference.npz')
                cache=ReferenceResetCache(ref,io.joint_names,env.device)
                recorder=install_recording(env,force_on=.001,capacity=256,references=[str(ref.path)])
                for start,count in [(0,100),(250,125),(350,125),(475,125),(550,100),(740,99)]:
                    state=cache.states[start].cpu().numpy()
                    def action(k,start=start):
                        return (cache.states[start+k,7:36][None].expand(4,-1)-term._offset)/term._scale
                    name=f'reference_f{start}'
                    results[name]=run_intervals(env,io,[state.copy() for _ in range(4)],count,out/name,action,recorder,start)[0]
            save(out/'results.json',results)
        finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['calibration','independent','original','candidate'],required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
