"""Standalone side-by-side M6.5 figures from frozen CSV/JSON evidence."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd,numpy as np
from common import *
from geometry import hull_xy,clip_polygon
from visualize_audit import draw
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'native':'#216b89','hdmi':'#c16642','native_r30_diagnostic':'#78a5b8','hdmi_y80_diagnostic':'#d9a082'}
def export(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=160);fig.savefig(OUT/(name+'.pdf'),metadata={'CreationDate':None});plt.close(fig)
def main():
    e=pd.read_csv(OUT/'eef_candidates.csv');fig,axs=plt.subplots(1,2,figsize=(13,5))
    labels=['official','wrist_origin','offset_x80','offset_x100','fit_offset_unconstrained','fit_wall_target','bounded_fit_train_half']
    for ax,side in zip(axs,['left','right']):
        z=e[(e.side==side)&(e.desired_contact==1)];val=[np.sqrt(np.mean(z[z.candidate==k].error_m**2))*1000 for k in labels]
        ax.barh(np.arange(len(labels)),val,color=['#216b89']+['#9bacb1']*3+['#c16642']*3);ax.set_yticks(np.arange(len(labels)),[s.replace('_',' ') for s in labels]);ax.invert_yaxis();ax.set(title=side+' EEF-target RMS',xlabel='RMS error (mm)')
        for i,v in enumerate(val):ax.text(v+1,i,f'{v:.1f}',va='center')
        ax.set_xlim(0,90)
    fig.suptitle('Point-definition changes reduce a label error; none changes physical collision');export(fig,'eef_comparison')
    h=pd.read_csv(OUT/'hand_candidates.csv');fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for ax,side in zip(axs,['left','right']):
        for mode,c in COLORS.items():
            z=h[(h.side==side)&(h.profile==mode)&(h.desired_contact==1)];ax.plot(z.frame/50,z.distance_m*1000,label=mode,color=c,lw=1.3)
        ax.axhspan(-1,2,color='green',alpha=.12,label='near-contact tolerance [-1,+2] mm');ax.axhline(0,c='black',lw=.6);ax.set(title=side,ylabel='Closest pair distance (mm)');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2)
    axs[-1].set_xlabel('Clip time (s)');fig.suptitle('Penetration / separation trade-off across the entire desired-contact interval');export(fig,'hand_geometry_comparison')
    # Shape/marker comparison in wrist yaw frame, at actual worst geometry frame 446.
    ref=load(NPZ);robot=read('robot_audit.json');names=ref['body_names'].tolist();f=446;fig=plt.figure(figsize=(12,6))
    for c,mode in enumerate(('native','hdmi')):
        ax=fig.add_subplot(1,2,c+1,projection='3d');side='right';i=names.index(side+'_wrist_yaw_link');W=rot(ref['body_quat_w'][f,i]);p=ref['body_pos_w'][f,i]
        hand=[g for g in (robot['native'] if mode=='native' else robot['hdmi_comparison_geoms']) if g.get('contype') and g['body'].startswith(side+'_wrist')]
        for g in hand:
            j=names.index(g['body']);draw(ax,g,W.inv().apply(ref['body_pos_w'][f,j]-p),W.inv()*rot(ref['body_quat_w'][f,j]),COLORS[mode],.25)
        # Box surfaces in wrist coordinates prove orientation-dependent intersection.
        bp=W.inv().apply(ref['box_pos_w'][f]-p);B=W.inv()*rot(ref['box_quat_w'][f])
        detail=[g for g in read('box_audit.json')['geoms'] if g['name'] in ('side4','handle_right')]
        cloud=np.concatenate([B.apply(vertices(g))+bp for g in detail]+[np.array([[-.07,-.06,-.06],[.21,.06,.06]])])
        for g in detail:draw(ax,g,bp,B,'#b9a36b',.08)
        ax.scatter(.05,0,0,c='purple',s=60,label='official internal EEF')
        target=W.inv().apply(ref['box_pos_w'][f]+rot(ref['box_quat_w'][f]).apply([0,-.17,.14])-p);ax.scatter(*target,c='red',s=50,label='box surface target')
        fit=json.loads((OUT/'eef_candidates.json').read_text())['right']['candidates']['fit_offset_unconstrained']['offset'];ax.scatter(*fit,c='black',marker='x',s=50,label='fitted constant offset')
        ax.set(xlim=(-.07,.22),ylim=(-.12,.12),zlim=(-.13,.13),title=f'{mode}: right wrist coordinates, frame 446',xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)');lo=cloud.min(0)-.02;hi=cloud.max(0)+.02;ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),title=f'{mode}: side4 + right handle, frame 446');ax.set_box_aspect(hi-lo);ax.view_init(25,-60);ax.legend(fontsize=8)
    export(fig,'collision_surface_snapshots')
    s=pd.read_csv(OUT/'support_candidates.csv');fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for ax,support in zip(axs,['support0','support1']):
        for dz,c in zip([0,.004,.007,.01],['#216b89','#518b68','#b59139','#be4e4e']):
            z=s[(s.support==support)&(s.height_correction_m==dz)].copy();z.loc[z.overlap_area_m2<1e-9,'distance_m']=np.nan
            ax.plot(z.frame/50,z.distance_m*1000,label=f'raise {dz*1000:.0f} mm',color=c)
        ax.set(title=support,ylabel='Signed pair distance (mm)',ylim=(-8,35));ax.axhspan(-1,1,color='grey',alpha=.15);ax.axhline(0,c='black',lw=.6);ax.grid(alpha=.2);ax.legend(ncol=4)
    axs[-1].set_xlabel('Clip time (s); curves shown where actual bottom/support footprints overlap');fig.suptitle('Full-motion support-height comparison | no hidden 10 mm correction');export(fig,'support_height_comparison')
    manifest=json.loads(MANIFEST.read_text());fig,axs=plt.subplots(1,4,figsize=(14,4))
    for ax,f,si in zip(axs,[0,253,544,839],[0,0,1,1]):
        support=manifest['supports'][si];center=np.array(support['root_pos_world'])
        box=manifest['object']['geoms'][0];bv=rot(ref['box_quat_w'][f]).apply(vertices(box))+ref['box_pos_w'][f];poly=hull_xy(bv)-center[:2]
        sv=rot(support['root_quat_wxyz']).apply(vertices(support['geom']))+center;sp=hull_xy(sv)-center[:2];inter=clip_polygon(poly,sp)
        ax.add_patch(Polygon(poly,fc='#c8ac77',ec='#b39560',alpha=.3,label='actual bottom projection'));ax.add_patch(Polygon(sp,fc='none',ec='#216b89',lw=2,label='support'));ax.add_patch(Polygon(inter,fc='#518b68',alpha=.4,label='intersection'))
        com=ref['box_pos_w'][f]+rot(ref['box_quat_w'][f]).apply([0,0,-.2]);ax.scatter(*(com[:2]-center[:2]),c='black',s=25,label='COM projection')
        ax.set(xlim=(-.23,.23),ylim=(-.23,.23),aspect='equal',title=f'frame {f} / support{si}',xlabel='relative world X (m)',ylabel='relative world Y (m)')
    axs[-1].legend(fontsize=7,loc='upper right');export(fig,'support_footprints')
    fig,axs=plt.subplots(4,2,figsize=(14,12),sharex='col')
    for col,f in enumerate([400,530]):
        for mode,c in [('native','#216b89'),('hdmi','#c16642')]:
            z=pd.read_csv(OUT/'dynamics_v1'/f'{mode}_z0_f{f}'/'trajectory.csv')
            axs[0,col].plot(z.time_s,z.box_z,label=mode,color=c);axs[1,col].plot(z.time_s,z.box_error_m,label=mode,color=c)
            axs[2,col].plot(z.time_s,z.left_force_N,label=mode+' left',color=c);axs[2,col].plot(z.time_s,z.right_force_N,label=mode+' right',color=c,ls='--')
            axs[3,col].plot(z.time_s,z.root_z,label=mode+' pelvis Z',color=c)
        axs[0,col].plot(z.time_s,ref['box_pos_w'][z.reference_frame.astype(int),2],c='black',ls=':',label='reference box root');axs[0,col].set_title(f'Unassisted PD from frame {f}')
        axs[3,col].set_xlabel('Seconds since reset')
        for ax in axs[:,col]:ax.legend(fontsize=8);ax.grid(alpha=.2)
    for row,label in enumerate(['Box root world Z (m)','Object position error (m)','Actual summed normal force (N)','Free robot root height (m)']):axs[row,0].set_ylabel(label)
    fig.suptitle('Free box + free robot | original PD, no root assistance, no attachment');export(fig,'dynamics_comparison')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for dz,c in [(0,'#216b89'),(4,'#518b68'),(10,'#be4e4e')]:
        z=pd.read_csv(OUT/'dynamics_v1'/f'box_only_z{dz}_f839'/'trajectory.csv');axs[0].plot(z.time_s,(z.box_z-z.box_z.iloc[0])*1000,c=c,label=f'support +{dz} mm');axs[1].plot(z.time_s,z.support1_force_N,c=c,label=f'support +{dz} mm')
    axs[1].axhline(19.62,c='black',ls=':',label='2 kg × 9.81');axs[0].set(title='Terminal box settles under gravity',ylabel='Box root drop (mm)');axs[1].set(title='Actual support normal force',ylabel='Normal force (N)')
    for ax in axs:ax.set_xlabel('Seconds after reset');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Conditional terminal stability is not successful pickup/carry/release');export(fig,'terminal_stability')
if __name__=='__main__':main()
