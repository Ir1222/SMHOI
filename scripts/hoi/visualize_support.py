import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from asset_common import OUT
s=json.loads((OUT/'support_audit.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(11,4.5))
for ax,row,title in zip(axs,[s['start'][0],s['end'][1]],['Initial frame 0 / support0','Final frame 839 / support1']):
    top=row['support_top_z'];bottom=row['box_geometry_min_world_z'];root=row['box_root_z']
    ax.add_patch(Rectangle((-.1,top),.2,-.04,color='#64798c',alpha=.8))
    ax.add_patch(Rectangle((-.09,bottom),.18,.02,color='#b47b35',alpha=.6))
    ax.axhline(root,ls=':',color='#b47b35',label='Box root Z')
    ax.annotate('',xy=(0,bottom),xytext=(0,top),arrowprops={'arrowstyle':'<->','color':'#ae263b'})
    ax.text(.01,top+.003,'10.00 mm gap',color='#ae263b',fontsize=11)
    ax.text(-.09,bottom+.012,'Bottom collision plate');ax.text(-.09,top-.025,'Support collision')
    ax.set(xlim=(-.14,.14),ylim=(top-.05,top+.05),title=title,xlabel='Schematic horizontal section (m)',ylabel='World Z (m)');ax.legend(loc='upper left');ax.grid(alpha=.15)
fig.suptitle('Actual bottom plate versus support top | original recorded heights preserved')
fig.tight_layout();fig.savefig(OUT/'support_gaps.png',dpi=180);fig.savefig(OUT/'support_gaps.pdf')
