"""Export standalone scientific audit plots from measured geometry and CSV."""
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from asset_common import *
from audit_npz import load
FACES=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
def draw(ax,g,pos,rotation,color,alpha=.2):
    if g['type']=='box':
        local=vertices(g);v=rotation.apply(local)+pos
        ax.add_collection3d(Poly3DCollection(v[FACES],facecolors=color,edgecolors=color,alpha=alpha,linewidths=.45))
    elif g['type']=='capsule':
        # Exact capsule boundary: cylinder and both hemispheres in geom Z axis.
        theta=np.linspace(0,2*np.pi,20);phi=np.linspace(0,np.pi,24)
        r,h=g['size'][:2];x=r*np.sin(phi)[:,None]*np.cos(theta);y=r*np.sin(phi)[:,None]*np.sin(theta)
        z=np.broadcast_to(r*np.cos(phi)[:,None]+np.where(np.cos(phi)[:,None]>=0,h,-h),x.shape)
        v=rotation.apply(rot(g['quat']).apply(np.stack([x,y,z],-1).reshape(-1,3))+g['pos'])+pos
        v=v.reshape(*x.shape,3);ax.plot_surface(*[v[:,:,i] for i in range(3)],color=color,alpha=alpha,linewidth=0)

def main():
    d=load(Path('/home/han/pyx/hdmi_bread_box_hoi_v1.npz'));names=d['body_names'].tolist()
    box=json.loads((OUT/'box_audit.json').read_text())['geoms'];r=json.loads((OUT/'robot_audit.json').read_text())
    modes={'Native capsules':[g for g in r['native'] if g['contype'] and 'wrist' in g['body']], 'HDMI hand boxes':[g for g in r['hdmi_comparison_geoms'] if 'wrist' in g['body']]}
    points=pd.read_csv(OUT/'contact_points.csv');dist=pd.read_csv(OUT/'hand_box_frames.csv')
    frames=[250,260,298,400,446,530,550,839]
    for part in range(2):
        fig=plt.figure(figsize=(16,9));fig.suptitle('Recorded reference geometry in box-root frame | target red, EEF purple, wrist axes RGB\nSolid collisions only; kinematic snapshots, no simulation',fontsize=14)
        for row,(mode,hand) in enumerate(modes.items()):
            for col,f in enumerate(frames[part*4:part*4+4]):
                ax=fig.add_subplot(2,4,row*4+col+1,projection='3d');br=rot(d['box_quat_w'][f]);bp=d['box_pos_w'][f]
                for g in box:draw(ax,g,np.zeros(3),rot([1,0,0,0]),'#b47b35',.13)
                for g in hand:
                    idx=names.index(g['body']);p=br.inv().apply(d['body_pos_w'][f,idx]-bp);rr=br.inv()*rot(d['body_quat_w'][f,idx]);draw(ax,g,p,rr,'#26778d' if g['body'].startswith('left') else '#ce6170',.28)
                for side in ('left','right'):
                    z=points[(points.frame==f)&(points.side==side)].iloc[0]
                    t=br.inv().apply(z[[f'target_world_{a}' for a in 'xyz']].to_numpy(float)-bp)
                    e=br.inv().apply(z[[f'eef_world_{a}' for a in 'xyz']].to_numpy(float)-bp)
                    ax.scatter(*t,c='red',s=28);ax.scatter(*e,c='purple',s=22);ax.plot(*np.stack([t,e]).T,c='purple',lw=1)
                    i=names.index(side+'_wrist_yaw_link');p=br.inv().apply(d['body_pos_w'][f,i]-bp);R=(br.inv()*rot(d['body_quat_w'][f,i])).as_matrix()
                    for a,c in enumerate(('red','green','blue')):ax.quiver(*p,*(R[:,a]*.045),color=c,arrow_length_ratio=.2)
                ax.set_title(f'{mode} | f={f} ({f/50:.2f}s)',fontsize=10)
                ax.set(xlim=(-.27,.27),ylim=(-.32,.32),zlim=(-.08,.4),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)');ax.set_box_aspect((.54,.64,.48));ax.view_init(22,35)
                if f == 839:
                    allp=br.inv().apply(d['body_pos_w'][f,[names.index(s+'_wrist_yaw_link') for s in ('left','right')]]-bp)
                    lo=np.minimum(allp.min(0)-.18,[-.27,-.32,-.08]);hi=np.maximum(allp.max(0)+.18,[.27,.32,.4])
                    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect(hi-lo)
        fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/f'contact_snapshots_{part+1}.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for side,c in [('left','#26778d'),('right','#ce6170')]:
        p=points[points.side==side];axs[0].plot(p.time_s,p.eef_error_m*1000,label=side,color=c)
        for mode,ls in [('native','-'),('hdmi_hand','--')]:
            z=dist[(dist.side==side)&(dist['mode']==mode)];axs[1].plot(z.time_s,z.min_hand_box_signed_distance_m*1000,label=f'{side}, {mode}',color=c,ls=ls)
    axs[2].plot(np.arange(len(d['joint_pos']))/50,d['box_pos_w'][:,2],c='#b47b35',label='box root Z')
    axs[0].set_ylabel('EEF target error (mm)');axs[1].set_ylabel('Min pair distance (mm)\nnegative = overlap');axs[2].set_ylabel('World Z (m)');axs[2].set_xlabel('Time (s)')
    for ax in axs:ax.axvspan(5.2,11,color='grey',alpha=.12,label='desired contact');ax.grid(alpha=.2);ax.legend(fontsize=9,loc='upper right')
    axs[1].axhline(0,c='black',lw=.8);fig.suptitle('All 840 frames | desired contact is an intent label, not physical contact');fig.tight_layout();fig.savefig(OUT/'contact_timeseries.png',dpi=170);fig.savefig(OUT/'contact_timeseries.pdf');plt.close(fig)
if __name__=='__main__':main()
