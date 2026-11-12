"""Bounded P6 native GPU tests; no policy, optimizer, or source-asset mutations."""
from __future__ import annotations

import argparse
import gzip
import json
import traceback
from pathlib import Path

from common import ROOT, cpu_contacts, inputs, pair_category, physical_env, rot, save
import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from scipy.optimize import least_squares
from calibrate_contacts import Placements, reset_states
from asset_common import geom
from supermimic.hoi.contact import SENSOR_NAMES, IntervalCapture, assert_contact_capacity
from supermimic.hoi.contact_validation import penetration_summary
from supermimic.hoi.recorder import install_recording
from supermimic.hoi.state import JointStateIO, ReferenceResetCache


def inventory(env, manifest, sources, output):
    model = env.sim.mj_model
    body = model.body('box/bread_box').id
    from supermimic.hoi.asset_manifest import load_manifest
    from supermimic.hoi.reference import DEFAULT_ASSET
    try:
        load_manifest(DEFAULT_ASSET)
        strict = 'PASS'
    except Exception as e:
        strict = repr(e)
    result = {'sources': sources, 'strict_production_loader': strict,
              'diagnostic_adapter': 'P6 spec A1; exact immutable manifest, missing V3 only',
              'nq_nv_nu': [model.nq, model.nv, model.nu],
              'mass': model.body_mass[body], 'com': model.body_ipos[body],
              'inertia': model.body_inertia[body], 'axes': model.body_iquat[body],
              'geoms': [geom(model, i) for i in range(model.ngeom) if model.geom_contype[i]],
              'support_roots_per_world': {s['name']: env.sim.model.body_pos[:, model.body(s['name']).id].cpu().numpy()
                                          for s in manifest['supports']},
              'origins': env.scene.env_origins.cpu().numpy(),
              'options': {k: np.asarray(getattr(model.opt, k)) for k in
                          ['timestep', 'iterations', 'ls_iterations', 'gravity', 'integrator', 'cone', 'tolerance']},
              'explicit_pairs': model.npair, 'excluded_body_pairs': model.nexclude}
    save(output/'inventory.json', result)


def bilateral(placements):
    """Independent native keyframe IK; box pose/target surface fixed, no reference."""
    model = placements.model
    state = placements.base.copy()
    state[71:78] = [.35, 0, 2.87, 1, 0, 0, 0]
    fit = []
    for side, sign in [('left', 1), ('right', -1)]:
        names = [n for n in placements.io.joint_names if n.startswith(side+'_')
                 and any(w in n for w in ['shoulder', 'elbow', 'wrist'])]
        indices = [7+placements.io.joint_names.index(n) for n in names]
        limits = np.array([model.jnt_range[model.joint('robot/'+n).id] for n in names])
        target = np.array([.35, sign*.2045, 3.0])
        gid = model.geom(f'robot/{side}_hand_collision').id
        initial = state[indices].copy()
        def residual(q):
            state[indices] = q
            placements.forward(state)
            axis = placements.data.geom_xmat[gid].reshape(3, 3)[:, 2]
            target[1] = sign*(.17+.035+model.geom_size[gid,1]*abs(axis[1])-.0005)
            pitch = model.geom(f'robot/{side}_wrist_collision').id
            clearance = [mujoco.mj_geomDistance(model, placements.data, pitch,
                         model.geom('box/'+name).id, 1., None)
                         for name in ('bottom','side1','side2','side3','side4','handle_left','handle_right')]
            return np.r_[placements.data.geom_xpos[gid]-target,
                         5*np.maximum(-np.array(clearance)-.0005, 0), .00001*(q-initial)]
        sol = least_squares(residual, initial, bounds=(limits[:, 0]+1e-5, limits[:, 1]-1e-5),
                            max_nfev=1500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        state[indices] = sol.x
        fit.append({'side': side, 'success': sol.success, 'residual': residual(sol.x), 'joints': names})
    placements.forward(state)
    if max(np.linalg.norm(f['residual'][:3]) for f in fit) > 1e-5:
        raise ValueError('Independent bilateral IK did not reach its surface targets')
    contacts = cpu_contacts(model, placements.data, 0)
    if not all(any(c['category'] == side for c in contacts) for side in ('left','right')):
        raise ValueError('Independent bilateral fixture lacks both hand pairs')
    if any(c['dist'] < -.001 for c in contacts):
        raise ValueError('Independent bilateral fixture has >1 mm initial overlap')
    return state, fit


def geometry(env, io, placements, cache, output):
    model = env.sim.mj_model
    bi, fits = bilateral(placements)
    save(output/'independent_bilateral.json', {'state84': bi, 'fit': fits,
                                            'contacts': cpu_contacts(model, placements.data, 0)})
    scans = []
    for side in ('left', 'right'):
        for part in ('wrist', 'hand'):
            state = placements.contact(side, part)
            for shift in np.linspace(.023, -.002, 51):
                s = state.copy(); s[71] += shift
                placements.forward(s)
                contacts = cpu_contacts(model, placements.data, 0)
                scans.append({'side': side, 'part': part, 'box_x_shift_m': shift,
                              'state84': s, 'contacts': contacts})
    save(output/'approach_scans.json', scans)
    # Whole-motion geometry uses original reference poses, never integrated states.
    rows, contacts = [], []
    box_geoms = [model.geom('box/'+g['name']).id for g in inputs()[0]['object']['geoms']]
    manifest, clip, _ = inputs()
    for frame, state in enumerate(cache.states.cpu().numpy()):
        placements.forward(state)
        data = placements.data
        eefs, targets, pairs = [], [], []
        for side_i, side in enumerate(('left', 'right')):
            bid = model.body(f'robot/{side}_wrist_yaw_link').id
            eefs.append(data.xpos[bid]+data.xmat[bid].reshape(3, 3)@np.array([.05, 0, 0]))
            targets.append(state[71:74]+rot(state[74:78]).apply(manifest['contact']['target_root_m'][side_i]))
            values = []
            for part in ('wrist', 'hand'):
                gid = model.geom(f'robot/{side}_{part}_collision').id
                for other in box_geoms:
                    values.append((mujoco.mj_geomDistance(model, data, gid, other, 10., None), gid, other))
            pairs.append(min(values))
        rows.append({'frame': frame, 'contact_ref': bool(clip.arrays['object_contact'][frame]),
                     'eef': eefs, 'targets': targets, 'distance_m': np.linalg.norm(np.array(eefs)-targets, axis=-1),
                     'min_pair_distance_LR_m': [p[0] for p in pairs], 'nearest_pairs': pairs})
        contacts.extend(cpu_contacts(model, data, frame))
    save(output/'reference_geometry.json', rows)
    save(output/'reference_geometry_contacts.json', contacts)
    return bi


class Observer:
    """Capture solved contacts BEFORE the original fail-closed capacity callback."""
    def __init__(self, env, io):
        self.env, self.io = env, io
        self.samples, self.contacts = [], []
        self.raw_error = 0.

    def sample(self):
        env, model = self.env, self.env.sim.mj_model
        n = int(env.sim.data.nacon[0]); c = env.sim.wp_data.contact
        with wp.ScopedDevice(env.sim.wp_device):
            ids = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32)
            forces = wp.zeros(n, dtype=wp.spatial_vector)
            if n:
                mjwarp.contact_force(env.sim.wp_model, env.sim.wp_data, ids, True, forces)
            force = forces.numpy()
        geoms, worlds, types = c.geom.numpy()[:n], c.worldid.numpy()[:n], c.type.numpy()[:n]
        distances, pos, frames = c.dist.numpy()[:n], c.pos.numpy()[:n], c.frame.numpy()[:n]
        direct = np.zeros((env.num_envs, 2, 2, 3))
        sample = len(self.samples)
        for i, (g1, g2) in enumerate(geoms):
            if min(g1, g2) < 0:
                continue
            cat = pair_category(model, g1, g2)
            if cat is None:
                continue
            row = {'sample': sample, 'env': int(worlds[i]), 'category': cat,
                   'dist': float(distances[i]), 'pos': pos[i].copy(), 'frame': frames[i].copy(),
                   'geom1': int(g1), 'geom2': int(g2), 'type': int(types[i]),
                   'names': [model.geom(int(g1)).name, model.geom(int(g2)).name],
                   'world_force_on_geom2_N': force[i, :3].copy()}
            self.contacts.append(row)
            if cat in ('left', 'right') and int(types[i]) & 1:
                hand_g = int(g2 if model.geom(int(g1)).name.startswith('box/') else g1)
                col = int('yaw' in model.body(int(model.geom_bodyid[hand_g])).name)
                sign = -1 if hand_g == g2 else 1
                direct[int(worlds[i]), int(cat == 'right'), col] += sign*force[i, :3]
        raw = np.stack([env.scene[n].data.force.cpu().numpy() for n in SENSOR_NAMES], 1)
        overflow = env.sim.data.overflow[:].cpu().numpy().copy()
        self.samples.append({'state84': self.io.read().cpu().numpy(), 'solve_state84': self.solve_state, 'raw_world_N': raw,
                             'solve_quat': env.sim.data.xquat[:, self.io.box_body_id].cpu().numpy().copy(),
                             'found': np.stack([env.scene[n].data.found.cpu().numpy() for n in SENSOR_NAMES], 1),
                             'overflow': overflow, 'nacon': n,
                             'nefc': env.sim.data.nefc[:].cpu().numpy().copy(),
                             'solver_niter': env.sim.wp_data.solver_niter.numpy().copy()})
        if not overflow.any():
            self.raw_error = max(self.raw_error, float(np.max(np.abs(raw-direct))))

    def write(self, folder):
        save(folder/'samples.json', self.samples)
        with gzip.open(folder/'contacts.json.gz', 'wt') as f:
            json.dump(self.contacts, f, default=lambda x: x.tolist(), allow_nan=False)
        summaries = {}
        for e in range(self.env.num_envs):
            summaries[str(e)] = penetration_summary([c for c in self.contacts if c['env'] == e], len(self.samples))
        save(folder/'penetration.json', summaries)


def run_intervals(env, io, states, count, folder, action_fn=None, recorder=None, frames_start=None):
    folder.mkdir(exist_ok=False)
    save(folder/'initial.json', {'state84': states, 'policy_steps_requested': count,
                                 'reference_start': frames_start, 'external_forces': 'zero'})
    reset_states(env, io, states)
    ids = torch.arange(env.num_envs, device=env.device)
    if recorder is not None:
        recorder.begin_episode(ids, torch.full((env.num_envs,), frames_start, device=env.device), torch.zeros_like(ids))
    capture = IntervalCapture(env, io, force_on=.001) if recorder is None else None
    observer = Observer(env, io)
    native_step = env.sim.step
    def solve():
        observer.solve_state = io.read().cpu().numpy()
        native_step()
    env.sim.step = solve
    original = env.metrics_manager.compute_substep
    def observe():
        observer.sample()
        original()
        if capture is not None:
            capture.sample()
    env.metrics_manager.compute_substep = observe
    rows, result = [], {'status': 'RUNNING'}
    try:
        for k in range(count):
            before = io.read().clone()
            action = action_fn(k) if action_fn else torch.zeros(env.num_envs, 29, device=env.device)
            ptr = recorder.buffer.ptr.clone() if recorder else None
            env.step(action)
            assert_contact_capacity(env)  # Also catches final-forward warnings.
            if recorder:
                labels = {key: value[ids, ptr].cpu().numpy().copy() for key, value in recorder.buffer.fields.items()}
            else:
                labels = {key: value.cpu().numpy() for key, value in capture.consume(ids).items()}
                labels['next_state'] = io.read().cpu().numpy()
            row = {'state': before.cpu().numpy(), 'action': action.cpu().numpy(), **labels}
            rows.append(row)
        result['status'] = 'PASS'
    except Exception as error:
        result.update(status='FAIL', error=repr(error), traceback=traceback.format_exc(),
                      overflow_at_failure=env.sim.data.overflow[:].cpu().numpy())
    finally:
        env.metrics_manager.compute_substep = original
        env.sim.step = native_step
        observer.write(folder)
        save(folder/'intervals.json', rows)
        result.update(completed_intervals=len(rows), solved_vector_substeps=len(observer.samples),
                      raw_sensor_max_error_N=observer.raw_error)
        save(folder/'result.json', result)
    print(folder.name, result['status'], len(rows), flush=True)
    return result, rows, observer


def main(output, mode):
    output.mkdir(parents=True, exist_ok=False)
    import shutil
    snapshot = output/'source_snapshot'; snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, snapshot/source.name)
    shutil.copy2(ROOT/'src/supermimic/hoi/contact_validation.py', snapshot/'contact_validation.py')
    torch.manual_seed(606)
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir = '/tmp/smhoi_p6_warp_cache'
    env, manifest, sources = physical_env()
    try:
        io = JointStateIO(env)
        placements = Placements(env, io)
        cache = ReferenceResetCache(inputs()[1], io.joint_names, env.device)
        inventory(env, manifest, sources, output)
        if mode == 'geometry':
            geometry(env, io, placements, cache, output)
        elif mode == 'calibration':
            bi, fit = bilateral(placements)
            save(output/'bilateral_fixture.json', {'state84': bi, 'fit': fit})
            results = {}
            for roll in (0., .31, .63):
                for depth in (.00025, .0005, .001, .002, .003):
                    states = [placements.base.copy()]
                    for side in ('left', 'right'):
                        s = placements.contact(side, roll=roll); s[71] += .003-depth
                        states.append(s)
                    states.append(bi)
                    name = f'roll{roll}_depth{depth}'
                    results[name] = run_intervals(env, io, states, 1, output/name)[0]
            save(output/'results.json', results)
        elif mode == 'dynamics':
            results = {}
            bi, fit = bilateral(placements)
            states = [placements.base.copy(), placements.contact('left'), placements.contact('right'), bi]
            action = env.action_manager.get_term('joint_pos')
            targets = torch.tensor(np.array(states)[:, 7:36], device=env.device, dtype=torch.float32)
            hold = (targets-action._offset)/action._scale
            results['independent_hand_release'] = run_intervals(env, io, states, 50, output/'independent_hand_release', lambda k: hold)[0]
            states = []
            for e in range(4):
                s = placements.base.copy(); s[:3] = [5., 5., 3.]
                sup = manifest['supports'][e%2]
                s[71:74] = np.array(sup['root_pos_world'])+[0,0,.04]
                s[74:78] = sup['root_quat_wxyz']
                states.append(s)
            results['original_supports_parallel'] = run_intervals(env, io, states, 150, output/'original_supports_parallel')[0]
            # Reproduce the historical floor fixture without relabeling it as independent.
            base = cache.states[260].cpu().numpy().copy(); base[36:71] = 0; base[78:84] = 0
            for elevation in (0., 2.):
                states = [placements.base.copy(), placements.contact('left', roll=.63), placements.contact('right'), base.copy()]
                states[-1][2] += elevation; states[-1][73] += elevation
                def prescribed(k):
                    return torch.sin(torch.arange(29, device=env.device)[None,:]*.2+k*.31).expand(4,-1)*.05
                name = f'historical_fixture_elevation{elevation}'
                results[name] = run_intervals(env, io, states, 24, output/name, prescribed)[0]
            save(output/'results.json', results)
        elif mode == 'replay':
            recorder = install_recording(env, force_on=.001, capacity=128, references=[str(inputs()[1].path)])
            action = env.action_manager.get_term('joint_pos')
            results = {}
            for start, count in [(0,50),(250,60),(400,50),(530,60),(550,60),(800,39)]:
                state = cache.states[start].cpu().numpy()
                def prescribed(k, start=start):
                    target = cache.states[start+k, 7:36][None,:].expand(4,-1)
                    return (target-action._offset)/action._scale
                name = f'reference_f{start}'
                results[name] = run_intervals(env, io, [state.copy() for _ in range(4)], count, output/name,
                                             prescribed, recorder, start)[0]
            save(output/'results.json', results)
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['geometry','calibration','dynamics','replay'], required=True)
    a = p.parse_args(); main(a.output, a.mode)
