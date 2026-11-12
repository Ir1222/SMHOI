"""C1.7 / C2.1: matched native hands and original-floor material trials."""
import argparse
from contextlib import nullcontext
import json
import shutil
import traceback
from pathlib import Path
from unittest.mock import patch

from closure_common import E, P6, gpu_env, dump_gz, save
from contact_parameters import GROUPS, config_edit, decode, geom_config
import solver_diagnosis as diagnosis
from run_p6 import run_intervals
from supermimic.hoi.solver_precision import bracket_precision_guard
from supermimic.hoi.state import JointStateIO
import numpy as np
import torch
import warp as wp


def contact_snapshot(env):
    n=int(env.sim.wp_data.nacon.numpy()[0])
    c=env.sim.wp_data.contact
    return {k:getattr(c,k).numpy()[:n].copy() for k in
            ("geom","worldid","dist","pos","frame","dim","friction","solref","solimp","includemargin","efc_address","type")}


def main(out, groups=None):
    out.mkdir(exist_ok=False)
    shutil.copytree(Path(__file__).parent,out/"source_snapshot",ignore=shutil.ignore_patterns("__pycache__"))
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir="/tmp/smhoi_p6_warp_cache"
    summaries=[]
    for group in GROUPS:
        if groups and group not in groups:
            continue
        folder=out/group
        folder.mkdir()
        for guard in (False,True):
            run=folder/("guard" if guard else "original_solver")
            run.mkdir()
            def factory(name="single_body",num_envs=4,change=None):
                env,manifest=gpu_env(name,num_envs,config_edit(group,change))
                save(run/"compiled.json",{"object_geoms":[geom_config(env.sim.mj_model,i) for i in range(env.sim.mj_model.ngeom) if env.sim.mj_model.geom(i).name.startswith("box/")],
                     "warp_ls_iterations":env.sim.wp_model.opt.ls_iterations,
                     "warp_iterations":env.sim.wp_model.opt.iterations,
                     "ls_tolerance":env.sim.wp_model.opt.ls_tolerance.numpy(),
                     "solver":int(env.sim.wp_model.opt.solver),"cone":int(env.sim.wp_model.opt.cone)})
                return env,manifest
            old_snapshot=diagnosis.snapshot
            def snapshot(env):
                result=old_snapshot(env)
                result["contact_parameters"]=contact_snapshot(env)
                result["overflow_decode"]=[decode(v) for v in result["overflow"]]
                return result
            try:
                with bracket_precision_guard() if guard else nullcontext(), patch.object(diagnosis,"gpu_env",factory), patch.object(diagnosis,"snapshot",snapshot):
                    diagnosis.experiment("single_body","complete",run,1.0)
                result=json.loads((run/"complete/result.json").read_text())
            except NotImplementedError as error:
                result={"status":"NOT TESTED","stage":"GPU model conversion", "error":repr(error),"traceback":traceback.format_exc()}
                save(run/"unsupported.json",result)
            summaries.append({"group":group,"guard":guard,"floor":result})
        # Matched elevated .5s native PD hold; no external wrench or root fixation.
        with bracket_precision_guard():
            try:
                env,_=gpu_env("single_body",change=config_edit(group))
            except NotImplementedError as error:
                result={"status":"NOT TESTED","stage":"GPU model conversion","error":repr(error),"traceback":traceback.format_exc()}
                summaries.append({"group":group,"native_hold":result})
                save(out/"results.json",summaries)
                continue
            try:
                io=JointStateIO(env)
                states=np.array(json.loads((P6/"calibration_02/roll0.0_depth0.001/initial.json").read_text())["state84"])
                term=env.action_manager.get_term("joint_pos")
                hold=(torch.tensor(states[:,7:36],device=env.device,dtype=torch.float32)-term._offset)/term._scale
                records=[]
                native=env.sim.step
                def step():
                    native()
                    records.append(contact_snapshot(env))
                env.sim.step=step
                result,rows,obs=run_intervals(env,io,states,25,folder/"native_hold",lambda k:hold)
                dump_gz(folder/"native_contact_parameters.json.gz",records)
                summaries.append({"group":group,"native_hold":result})
            finally:
                env.close()
        save(out/"results.json",summaries)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--groups",nargs="+")
    a=p.parse_args()
    main(a.output,a.groups)
