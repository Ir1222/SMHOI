"""Recompute P6 measurements from immutable run outputs, never from reference relabels."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from common import ROOT, inputs, save
from supermimic.hoi.contact_validation import actual_contact, agreement, penetration_summary


def calibrate(root):
    training=[[],[]]; null=[[],[]]; held=[[],[]]; rows=[]
    for folder in sorted((root/'calibration_02').glob('roll*')):
        samples=json.loads((folder/'samples.json').read_text())
        if len(samples)!=4 or any(np.any(s['overflow']) for s in samples):
            raise ValueError('Incomplete/invalid calibration')
        raw=np.stack([s['raw_world_N'] for s in samples],axis=1)
        norms=np.linalg.norm(raw.sum(3),axis=-1)
        holdout='roll0.31_' in folder.name
        for h in range(2):
            (held if holdout else training)[h].append(norms[h+1,0,h])
            null[h].extend(norms[0,:,h]);null[h].extend(norms[2-h,:,h])
        rows.append((folder.name,raw,holdout))
    nmax=np.array([max(x) for x in null]);p05=np.array([np.quantile(x,.05) for x in training])
    if np.any(nmax>=p05) or np.any(nmax>1e-6):raise ValueError('No reliable force separation')
    tau=.5*(nmax+p05);rho=np.array([.25,.25]);results=[]
    for name,raw,heldout in rows:
        result=actual_contact(raw,tau,rho)
        results.append({'name':name,'held_out':heldout,'occupancy':result['occupancy'],
                        'actual_hands':result['hands'],'first_force_N':result['magnitude_N'][:,0]})
    previous=json.loads((ROOT/'outputs/hoi/contact_supervision/sensor_calibration/run_05/calibration.json').read_text())
    transients={}
    for key in ('substep1_only','substep4_only','contact_ends_before_final'):
        samples=previous['scenarios'][key]['samples']
        raw=np.stack([x['force_columns_world_N'] for x in samples],axis=1)
        r=actual_contact(raw,tau,rho)
        transients[key]={'original_pattern':previous['scenarios'][key]['pattern'],
                         'occupancy':r['occupancy'],'actual_hands':r['hands']}
    result={'thresholds_N_LR':tau,'occupancy_cutoffs_LR':rho,'strict_comparison':'>',
            'definition':'norm of simultaneous pitch+yaw net force > tau at >=2/4 substeps; no reference used',
            'null_max_N_LR':nmax,'train_positive_p05_N_LR':p05,
            'training_first_solve_N_LR':training,'heldout_first_solve_N_LR':held,
            'heldout_first_solve_recall_LR':[float((np.array(v)>t).mean()) for v,t in zip(held,tau)],
            'training_rolls_rad':[0.,.63],'heldout_roll_rad':.31,'depths_m':[.00025,.0005,.001,.002,.003],
            'fixtures':results,'reused_P1_transients_at_new_definition':transients,
            'scope':'deterministic native controlled loads; excludes subthreshold/genuine one-substep impacts; net cancellation remains possible'}
    save(root/'calibration.json',result)
    return result


def geometry(root):
    rows=json.loads((root/'geometry_02/reference_geometry.json').read_text())
    on=np.array([r['contact_ref'] for r in rows]);d=np.array([r['distance_m'] for r in rows]);pairs=np.array([r['min_pair_distance_LR_m'] for r in rows])
    def stats(x):return {'min':x.min(0),'median':np.median(x,axis=0),'max':x.max(0),'rms':np.sqrt((x*x).mean(0))}
    result={'frames':len(rows),'on':int(on.sum()),'off':int((~on).sum()),'eef_target_m_ON':stats(d[on]),
            'eef_target_m_OFF':stats(d[~on]),'pair_distance_m_ON':stats(pairs[on]),
            'overlap_gt_1mm_ON_LR':(pairs[on]<-.001).sum(0),'gap_gt_2mm_ON_LR':(pairs[on]>.002).sum(0),
            'both_in_M6_band_ON':int(((pairs[on]>=-.001)&(pairs[on]<=.002)).all(1).sum()),
            'frame_worst_overlap_LR':pairs.argmin(0),'worst_overlap_m_LR':-pairs.min(0)}
    scans=json.loads((root/'geometry_02/approach_scans.json').read_text());approach={}
    for side in ('left','right'):
        for part in ('wrist','hand'):
            rs=[r for r in scans if r['side']==side and r['part']==part]
            touching=[r for r in rs if any(c['category']==side for c in r['contacts'])]
            first=touching[0] if touching else None
            approach[side+'_'+part]={'placements':len(rs),'initial_hand_contacts':sum(c['category'] in ('left','right') for c in rs[0]['contacts']),
                                    'first_shift_m':first['box_x_shift_m'] if first else None,
                                    'first_min_dist_m':min(c['dist'] for c in first['contacts'] if c['category']==side) if first else None,
                                    'opposite_hand_contact_placements':sum(any(c['category'] not in (side,'support','ground','robot_body') for c in r['contacts']) for r in rs)}
    result['approach']=approach;save(root/'geometry_summary.json',result)
    return result


def replay(root,cal):
    raws=[];starts=[];ends=[];episodes=[];valid=[];meta=[];case_results={}
    labels=inputs()[1].arrays['object_contact']
    for ci,folder in enumerate(sorted((root/'replay_01').glob('reference_f*'))):
        frame=int(folder.name.split('_f')[1]);rows=json.loads((folder/'intervals.json').read_text());status=json.loads((folder/'result.json').read_text())
        case_results[folder.name]=status
        for k,row in enumerate(rows):
            for e in range(4):
                raw=np.array(row['force_raw_world'][e]);raws.append(raw)
                starts.append(frame+k);ends.append(frame+k+1);episodes.append(ci*4+e);valid.append(True)
                meta.append({'case':folder.name,'world':e,'frame_start':frame+k,'frame_end':frame+k+1,
                             'state':row['state'][e],'next_state':row['next_state'][e]})
    if not raws:raise ValueError('No valid replay intervals')
    raw=np.array(raws);start=np.array(starts);end=np.array(ends);ep=np.array(episodes);mask=np.array(valid)
    tau=cal['thresholds_N_LR'];rho=cal['occupancy_cutoffs_LR']
    result=agreement(raw,labels,start,end,ep,mask,tau,rho)
    result.update(cases=case_results,task_required_aggregate='bilateral AND during ON260..549, P4.1 explicit manifest evidence',
                  failed_partial_substeps_excluded='Solver-invalid/incomplete final interval retained in per-case samples; all earlier complete rows retained, including bad tracking',
                  unique_reference_end_frames=len(np.unique(end)),thresholds_N_LR=tau,cutoffs_LR=rho)
    result['sensitivity']={}
    for multiplier in (.5,1.,2.):
        for cutoff in (0.,.25,.5,.75):
            metrics=agreement(raw,labels,start,end,ep,mask,np.array(tau)*multiplier,[cutoff]*2)
            result['sensitivity'][f'tau_x{multiplier}_rho{cutoff}']={a:metrics[a] for a in ('any','bilateral')}
    save(root/'replay_summary.json',result)
    # Independent P4 FK geometry at integrated end states, no force in reward path.
    import torch
    from audit_robot import target_spec
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.hoi.rewards import ProximityReward
    m,clip,_=inputs();fk=G1Kinematics(target_spec().compile(),dtype=torch.float64)
    proxy=ProximityReward(fk,tuple(m['contact']['wrist_bodies']),m['contact']['target_root_m'],m['contact']['eef_offset_wrist_m'],sigma_pos=1.)
    # Named runtime order verified separately against FK; never assume index coincidence.
    source_names = clip.arrays['joint_names'].tolist()
    order = [source_names.index(n) for n in fk.joint_names[1:]]
    for folder in (root/'replay_01').glob('reference_f*'):
        f = int(folder.name.split('_f')[1])
        initial = np.array(json.loads((folder/'initial.json').read_text())['state84'])
        np.testing.assert_allclose(initial[:,7:36], np.broadcast_to(clip.arrays['joint_pos'][f,order], (4,29)), atol=1e-7, rtol=0)
    states=torch.tensor([r['next_state'] for r in meta],dtype=torch.float64)
    eef,target=proxy.geometry(states);distance=(eef-target).norm(dim=-1).numpy()
    actual=actual_contact(raw,tau,rho)
    for i,row in enumerate(meta):
        row.update(contact_ref=bool(labels[end[i]]),actual_LR=actual['hands'][i],
                   actual_any=bool(actual['any'][i]),actual_bilateral=bool(actual['bilateral'][i]),
                   occupancy_LR=actual['occupancy'][i],force_peak_N_LR=actual['magnitude_N'][i].max(0),
                   eef_target_distance_LR_m=distance[i],eef_world=eef[i].numpy(),targets_world=target[i].numpy(),
                   validity=True)
    save(root/'replay_rows.json',meta)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--part',choices=['calibration','geometry','replay','all'],default='all');a=p.parse_args()
    if a.part in ('calibration','all'):calibrate(a.root)
    if a.part in ('geometry','all'):geometry(a.root)
    if a.part in ('replay','all'):replay(a.root,json.loads((a.root/'calibration.json').read_text()))
