"""M6.5 full-trajectory counterfactuals. Never changes source motion or assets."""
import copy,itertools
import mujoco,numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import lsq_linear,minimize
from common import *

def profiles():
    r=read('robot_audit.json')
    native=[g for g in r['native'] if g['contype'] and 'wrist' in g['body']]
    hdmi=[g for g in r['hdmi_comparison_geoms'] if 'wrist' in g['body']]
    thin=copy.deepcopy(native)
    for g in thin:g['size'][0]=.03
    slim=copy.deepcopy(hdmi)
    for g in slim:g['size'][1]*=.8
    return {'native':native,'hdmi':hdmi,'native_r30_diagnostic':thin,'hdmi_y80_diagnostic':slim}

def place(m,md,d,f,hand):
    md.mocap_pos[m.body('box').mocapid]=d['box_pos_w'][f];md.mocap_quat[m.body('box').mocapid]=d['box_quat_w'][f]
    for body in {g['body'] for g in hand}:
        i=d['body_names'].tolist().index(body)
        md.mocap_pos[m.body(body).mocapid]=d['body_pos_w'][f,i];md.mocap_quat[m.body(body).mocapid]=d['body_quat_w'][f,i]
    mujoco.mj_forward(m,md)

def pair_distance(m,md,hand,box,side):
    vals=[(mujoco.mj_geomDistance(m,md,m.geom(g['name']).id,m.geom('box_'+b['name']).id,10,None),g['name'],b['name']) for g in hand if g['body'].startswith(side) for b in box]
    return min(vals)

def semantic_analysis(d):
    mask=d['object_contact'].astype(bool);names=d['body_names'].tolist();B=rot(d['box_quat_w']);rows=[];summary={}
    for si,side in enumerate(('left','right')):
        i=names.index(side+'_wrist_yaw_link');W=rot(d['body_quat_w'][:,i]);target=d['contact_target_root'][si].astype(float)
        p=d['body_pos_w'][:,i];bp=d['box_pos_w'];R=(B.inv()*W).as_matrix();r=B.inv().apply(p-bp)
        required=W.inv().apply(bp+B.apply(np.broadcast_to(target,p.shape))-p)
        fit=required[mask].mean(0)
        eef=r+np.einsum('fij,j->fi',R,[.05,0,0]);mean_target=eef[mask].mean(0);mean_target[1]=target[1];mean_target[0]=np.clip(mean_target[0],-.09,.09);mean_target[2]=np.clip(mean_target[2],0,.26)
        candidates={'official':(np.array([.05,0,0]),target),'wrist_origin':(np.zeros(3),target),'offset_x80':(np.array([.08,0,0]),target),'offset_x100':(np.array([.1,0,0]),target),'fit_offset_unconstrained':(fit,target),'fit_wall_target':(np.array([.05,0,0]),mean_target)}
        # Fit offset(3) and surface-constrained target x,z(2), training first half only.
        active=np.flatnonzero(mask);train=active[:len(active)//2];test=active[len(active)//2:]
        A=np.concatenate([R[train],np.broadcast_to(-np.eye(3)[:,[0,2]],(len(train),3,2))],axis=2).reshape(-1,5)
        rhs=(np.array([0,target[1],0])-r[train]).reshape(-1)
        fit2=lsq_linear(A,rhs,bounds=([0,-.04,-.05,-.08,.02],[.18,.04,.05,.08,.24])).x
        candidates['bounded_fit_train_half']=(fit2[:3],np.array([fit2[3],target[1],fit2[4]]))
        s={'required_time_varying_offset_m':{'min':required[mask].min(0),'max':required[mask].max(0),'std':required[mask].std(0)},'candidates':{}}
        for key,(off,t) in candidates.items():
            err=r+np.einsum('fij,j->fi',R,off)-t;norm=np.linalg.norm(err,axis=1)
            s['candidates'][key]={'offset':off,'target':t,'contact_error_m':stats(norm[mask]),'rmse_m':float(np.sqrt(np.mean(norm[mask]**2))),'heldout_second_half_rmse_m':float(np.sqrt(np.mean(norm[test]**2)))}
            for f in range(len(p)):
                rows.append({'frame':f,'side':side,'candidate':key,'desired_contact':int(mask[f]),'error_m':norm[f],'wall_normal_error_m':err[f,1]*(1 if si==0 else -1),'wall_tangential_error_m':np.linalg.norm(err[f,[0,2]])})
        summary[side]=s
    write_csv(OUT/'eef_candidates.csv',rows);save('eef_candidates.json',summary)

def hand_analysis(d):
    box=read('box_audit.json')['geoms'];rows=[];summary={};mask=d['object_contact'].astype(bool)
    for mode,hand in profiles().items():
        m,md=comparison_model(box,hand)
        for f in range(len(mask)):
            place(m,md,d,f,hand)
            for side in ('left','right'):
                v,a,b=pair_distance(m,md,hand,box,side)
                rows.append({'frame':f,'profile':mode,'side':side,'desired_contact':int(mask[f]),'distance_m':v,'hand_geom':a,'box_geom':b})
        summary[mode]={}
        for side in ('left','right'):
            rr=[r for r in rows if r['profile']==mode and r['side']==side and r['desired_contact']];v=np.array([r['distance_m'] for r in rr])
            summary[mode][side]={'distance_m':stats(v),'penetration_over_1mm':int((v<-.001).sum()),'separation_over_2mm':int((v>.002).sum()),'near_contact_without_deep_overlap':int(((v>=-.001)&(v<=.002)).sum()),'worst_pair':min(rr,key=lambda r:r['distance_m'])}
    write_csv(OUT/'hand_candidates.csv',rows);save('hand_candidates.json',summary)
    # Isolate orientation versus scale, without claiming these are authorized trajectories.
    ablations=[]
    for profile in ('native','hdmi'):
        hand=profiles()[profile]
        for variant in ('baseline','box_scale90','box_scale110','frozen_wrist_relative_orientation'):
            bg=copy.deepcopy(box)
            if 'scale' in variant:
                scale=.9 if variant.endswith('90') else 1.1
                for g in bg:g['pos']=(np.array(g['pos'])*scale).tolist();g['size']=(np.array(g['size'])*scale).tolist()
            m,md=comparison_model(bg,hand);names=d['body_names'].tolist()
            for f in np.flatnonzero(mask):
                place(m,md,d,f,hand)
                if variant=='frozen_wrist_relative_orientation':
                    for body in {g['body'] for g in hand}:
                        i=names.index(body);q=(rot(d['box_quat_w'][f])*rot(d['box_quat_w'][260]).inv()*rot(d['body_quat_w'][260,i])).as_quat()[[3,0,1,2]]
                        md.mocap_quat[m.body(body).mocapid]=q
                    mujoco.mj_forward(m,md)
                for side in ('left','right'):
                    v,a,b=pair_distance(m,md,hand,bg,side);ablations.append({'frame':int(f),'profile':profile,'ablation':variant,'side':side,'distance_m':v,'hand_geom':a,'box_geom':b})
    write_csv(OUT/'hand_ablations.csv',ablations)

def clip_polygon(subject,clip):
    """Sutherland–Hodgman, convex CCW polygons, in actual world XY."""
    p=np.asarray(subject)
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    for a,b in zip(clip,np.roll(clip,-1,axis=0)):
        if len(p)==0:break
        result=[]
        for u,v in zip(p,np.roll(p,-1,axis=0)):
            du=cross(b-a,u-a);dv=cross(b-a,v-a)
            if du>=-1e-12:result.append(u)
            if (du>=0)!=(dv>=0):result.append(u+(v-u)*du/(du-dv))
        p=np.array(result)
    return p

def area(p):return 0. if len(p)<3 else float(abs(np.dot(p[:,0],np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1)))/2)
def hull_xy(v):
    xy=np.unique(v[:,:2],axis=0);return xy[ConvexHull(xy).vertices]

def support_analysis(d):
    manifest=json.loads(MANIFEST.read_text());box=manifest['object']['geoms'];hands=[]
    for s in manifest['supports']:
        g=copy.deepcopy(s['geom']);g['body']=s['name'];hands.append(g)
    m,md=comparison_model(box,hands);rows=[];polygons=[];F=len(d['joint_pos'])
    for f in range(F):
        bv=rot(d['box_quat_w'][f]).apply(vertices(box[0]))+d['box_pos_w'][f];poly=hull_xy(bv)
        for s in manifest['supports']:
            sv=rot(s['root_quat_wxyz']).apply(vertices(s['geom']))+s['root_pos_world'];sp=hull_xy(sv);inter=clip_polygon(poly,sp)
            com=d['box_pos_w'][f]+rot(d['box_quat_w'][f]).apply(manifest['object']['inertia']['com_pos_root_m']);local=rot(s['root_quat_wxyz']).inv().apply(com-np.array(s['root_pos_world']))
            polygons.append({'frame':f,'support':s['name'],'overlap_area_m2':area(inter),'bottom_projection_area_m2':area(poly),'com_xy_inside_support':bool(np.all(abs(local[:2])<=np.array(s['geom']['size'])[:2])),'com_xy_edge_margin_m':float(np.min(np.array(s['geom']['size'])[:2]-abs(local[:2])))})
    def evaluate(dz,record=False):
        minima={s['name']:(10,None) for s in manifest['supports']}
        for f in range(F):
            md.mocap_pos[m.body('box').mocapid]=d['box_pos_w'][f];md.mocap_quat[m.body('box').mocapid]=d['box_quat_w'][f]
            for s in manifest['supports']:
                md.mocap_pos[m.body(s['name']).mocapid]=np.array(s['root_pos_world'])+[0,0,dz];md.mocap_quat[m.body(s['name']).mocapid]=s['root_quat_wxyz']
            mujoco.mj_forward(m,md)
            for si,s in enumerate(manifest['supports']):
                ds=[mujoco.mj_geomDistance(m,md,m.geom('box_'+b['name']).id,m.geom(s['name']).id,10,None) for b in box];v=min(ds)
                if v<minima[s['name']][0]:minima[s['name']]=(v,f)
                if record:rows.append({'frame':f,'support':s['name'],'height_correction_m':dz,'distance_m':v,'closest_box_geom':box[np.argmin(ds)]['name'],**{k:v for k,v in polygons[2*f+si].items() if k not in ('frame','support')}})
        return minima
    summary={}
    for dz in [0,.004,.007,.01]:
        minima=evaluate(dz,True);summary[str(dz)]={}
        for s in manifest['supports']:
            rr=[r for r in rows if r['height_correction_m']==dz and r['support']==s['name']];v=np.array([r['distance_m'] for r in rr]);summary[str(dz)][s['name']]={'min_distance_m':minima[s['name']][0],'min_frame':minima[s['name']][1],'penetration_frames_over_1mm':int((v<-.001).sum()),'penetration_frames_over_0_1mm':int((v<-.0001).sum()),'initial_distance_m':float(v[0]),'final_distance_m':float(v[-1])}
    # Exact all-frame bisection of maximum common static raise for <=1 mm penetration.
    lo,hi=0.,.012
    for _ in range(12):
        mid=(lo+hi)/2
        if min(v[0] for v in evaluate(mid).values())>=-.001:lo=mid
        else:hi=mid
    save('support_candidates.json',{'candidates':summary,'max_common_raise_for_leq_1mm_overlap_m':[lo,hi],'needed_raise_for_initial_final_gap_leq_1mm_m':[.009000022,.011000022],'footprint_at_endpoints':[polygons[0],polygons[-1]],'static_height_only_feasible':lo>=.009000022})
    write_csv(OUT/'support_candidates.csv',rows);write_csv(OUT/'support_footprints.csv',polygons)

def pose_search(d):
    """Bounded, multi-start local feasibility search with exact fixed robot poses.

    Not an IK solver or proof of global infeasibility; no derived trajectory saved.
    """
    box=read('box_audit.json')['geoms'];rows=[]
    for mode in ('native','hdmi'):
        hand=profiles()[mode];m,md=comparison_model(box,hand);bm=m.body('box').mocapid
        for f in [260,280,298,330,370,400,436,438,446,466,505,530,549]:
            place(m,md,d,f,hand);p0=d['box_pos_w'][f].copy();r0=rot(d['box_quat_w'][f])
            def distances(x):
                from scipy.spatial.transform import Rotation
                md.mocap_pos[bm]=p0+x[:3];md.mocap_quat[bm]=(Rotation.from_rotvec(x[3:])*r0).as_quat()[[3,0,1,2]];mujoco.mj_forward(m,md)
                return np.array([pair_distance(m,md,hand,box,s)[0] for s in ('left','right')])
            def objective(x):return float(np.sum((x[:3]/.03)**2)+np.sum((x[3:]/np.deg2rad(5))**2))
            constraints={'type':'ineq','fun':lambda x:np.r_[distances(x)+.001,.002-distances(x)]}
            trials=[]
            for y in (0.,-.02,.02):
                x0=np.array([0,y,0,0,0,0]);res=minimize(objective,x0,method='SLSQP',bounds=[(-.03,.03)]*3+[(-np.deg2rad(5),np.deg2rad(5))]*3,constraints=constraints,options={'maxiter':100,'ftol':1e-9,'eps':1e-5})
                v=distances(res.x);feasible=bool(np.all(v>=-.00101)&np.all(v<=.00201));trials.append((not feasible,objective(res.x),res,v))
                if feasible:break
            _,_,res,v=min(trials,key=lambda t:t[:2]);feasible=bool(np.all(v>=-.00101)&np.all(v<=.00201))
            rows.append({'frame':f,'profile':mode,'feasible_local_search':feasible,'translation_norm_m':np.linalg.norm(res.x[:3]),'rotation_norm_rad':np.linalg.norm(res.x[3:]),'left_distance_m':v[0],'right_distance_m':v[1],**{k:float(val) for k,val in zip(['dx','dy','dz','rx','ry','rz'],res.x)}})
    write_csv(OUT/'box_pose_feasibility.csv',rows)

def main():
    verify();d=load(NPZ);semantic_analysis(d);print('EEF candidates complete',flush=True);hand_analysis(d);print('Hand geometry complete',flush=True);support_analysis(d);print('Support candidates complete',flush=True);pose_search(d);verify();print('M6.5 geometry complete')
if __name__=='__main__':main()
