"""Matched independent CPU physical tests; never natural hand carrying."""
import argparse
import shutil
from pathlib import Path
import mujoco
import numpy as np
from closure_common import E, candidate, manifest_for, rot, save
from box_dynamics import model_for, run_case


def main(out):
    out.mkdir(exist_ok=False);shutil.copytree(Path(__file__).parent,out/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    summaries=[]
    for name in ('original','p6_component_proposal','union_shell','single_body','single_envelope','smtrack_2kg','smtrack_1p5kg','grasp_pads'):
        folder=out/name;folder.mkdir();manifest=manifest_for(name);c=candidate(name)
        bottom=min(g['pos'][2]-g['size'][2] for g in c['geoms'])
        cases=[]
        for kind,extra in [('freefall',{}),('force',{'force':[4,0,0]}),('torque',{'torque':[.02,0,0]})]:
            for yaw in (0.,.6):
                cases.append(dict(name=f'{kind}_{yaw}',kind=kind,seconds=.1,position=[0,0,3],quaternion=rot_from_yaw(yaw),**extra))
        for support in (0,1):
            s=manifest['supports'][support]
            for kind,gap in [('settle',.01),('drop',.1),('lift',0.)]:
                cases.append(dict(name=f'support{support}_{kind}',kind=kind,seconds=3.,position=(np.array(s['root_pos_world'])+[0,0,s['geom']['size'][2]-bottom+gap]).tolist(),quaternion=s['root_quat_wxyz']))
        cases.append(dict(name='floor_drop',kind='drop',seconds=3.,position=[0,0,-bottom+.1],quaternion=[1,0,0,0]))
        model=model_for(manifest);mujoco.mj_saveLastXML(str(folder/'object_scene.xml'),model)
        for case in cases:
            result=run_case(case,manifest,folder)
            result['candidate']=name
            result['expected_weight_N']=c['inertia']['mass_kg']*9.81
            result['support_balance_relative_error']=abs(result['tail_mean_support_z_N']-result['expected_weight_N'])/result['expected_weight_N'] if case['name'].startswith('support') else None
            # Archive original P6 evaluator's fixed2kg convention explicitly; use
            # candidate-specific expected weight for controlled1.5kg comparison.
            summaries.append(result)
    save(out/'summary.json',summaries)


def rot_from_yaw(yaw):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec([0,0,yaw]).as_quat(scalar_first=True).tolist()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
