"""Replay exact saved pre-failure GPU states for paired one-step CPU checks."""
import argparse
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
import warp as wp
from closure_common import E, gpu_env, save
from solver_diagnosis import snapshot
from supermimic.hoi.solver_precision import bracket_precision_guard


def run(mode,guard):
    source=json.loads((E/'solver_01'/mode/'first_failure.json').read_text());before=source['before']
    def change(cfg):
        if mode=='robot_floor':
            base=cfg.scene.spec_fn
            def edit(spec):
                base(spec)
                for g in spec.geoms:
                    if g.name.startswith('box/') or 'support' in g.name:g.contype=g.conaffinity=0
            cfg.scene.spec_fn=edit
    with bracket_precision_guard() if guard else nullcontext():
        env,manifest=gpu_env('original',change=change)
        try:
            for key in ('qpos','qvel','ctrl','qfrc_applied','xfrc_applied'):
                tensor=getattr(env.sim.data,key);tensor.copy_(torch.tensor(before[key],device=env.device,dtype=tensor.dtype))
            env.sim.forward()
            env.sim.data.qacc_warmstart.copy_(torch.tensor(before['qacc_warmstart'],device=env.device,dtype=torch.float32))
            env.sim.data.overflow.zero_()
            env.sim.step();env.scene.update(.005);after=snapshot(env)
            worlds=[]
            for w in range(4):
                m=copy.copy(env.sim.mj_model)
                for s in manifest['supports']:m.body_pos[m.body(s['name']).id]=np.array(s['root_pos_world'])+env.scene.env_origins[w].cpu().numpy()
                d=mujoco.MjData(m)
                for key in ('qpos','qvel','ctrl','qfrc_applied','xfrc_applied'):getattr(d,key)[:]=before[key][w]
                mujoco.mj_forward(m,d)
                M=np.empty((m.nv,m.nv));mujoco.mj_fullM(m,d,M);eigs=np.linalg.eigvalsh(M)
                contacts=[{'dist':c.dist,'pos':c.pos.copy(),'frame':c.frame.copy(),'geom1':int(c.geom1),'geom2':int(c.geom2),'names':[m.geom(c.geom1).name,m.geom(c.geom2).name]} for c in d.contact[:d.ncon]]
                J=d.efc_J.reshape(d.nefc,m.nv).copy()
                A=J@np.linalg.solve(M,J.T)+np.diag(d.efc_R)
                d.qacc_warmstart[:]=before['qacc_warmstart'][w]
                mujoco.mj_step(m,d)
                worlds.append({'world':w,'qpos_error':float(np.max(np.abs(after['qpos'][w]-d.qpos))),
                     'qvel_error':float(np.max(np.abs(after['qvel'][w]-d.qvel))),
                     'qacc_error':float(np.max(np.abs(after['qacc'][w]-d.qacc))),
                     'mass_matrix_condition':float(eigs[-1]/eigs[0]),'mass_matrix_min_eigenvalue':float(eigs[0]),
                     'regularized_constraint_condition':float(np.linalg.cond(A)) if len(A) else 0.,
                     'ncon':d.ncon,'nefc':d.nefc,'contacts':contacts,'max_ctrl':float(np.max(np.abs(d.ctrl))),
                     'max_actuator_force':float(np.max(np.abs(d.actuator_force))),
                     'warning_count':int(sum(w.number for w in d.warning))})
            return {'mode':mode,'precision_guard':guard,'same_pre_failure_state_and_control':True,
                    'overflow':after['overflow'],'solver_niter':after['solver_niter'],'worlds':worlds,
                    'acceptance_next_qpos_lt_1e_4_qvel_lt_1e_3':all(w['qpos_error']<1e-4 and w['qvel_error']<1e-3 for w in worlds)}
        finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);wp.config.kernel_cache_dir='/tmp/smhoi_p6_warp_cache'
    results=[]
    for mode in ('complete','robot_floor','reference0'):
        for guard in (False,True):
            result=run(mode,guard);save(a.output/f'{mode}_guard{guard}.json',result);results.append(result)
            print(mode,guard,result['overflow'],result['acceptance_next_qpos_lt_1e_4_qvel_lt_1e_3'],flush=True)
    save(a.output/'results.json',results)
