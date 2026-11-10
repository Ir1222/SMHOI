"""Measure internal hand shape overlaps/gaps at representative recorded poses."""
from pathlib import Path
import itertools,json
import mujoco,numpy as np
from asset_common import *
from audit_npz import load
from audit_contact import comparison_model,write_csv
r=json.loads((OUT/'robot_audit.json').read_text());d=load(Path('/home/han/pyx/hdmi_bread_box_hoi_v1.npz'));names=d['body_names'].tolist()
box=json.loads((OUT/'box_audit.json').read_text())['geoms'];rows=[]
for mode,hand in [('native',[g for g in r['native'] if g['contype'] and 'wrist' in g['body']]),('hdmi_hand',[g for g in r['hdmi_comparison_geoms'] if 'wrist' in g['body']])]:
 m,md=comparison_model(box,hand)
 for f in [0,250,260,280,298,400,446,530,549,550,839]:
  for body in {g['body'] for g in hand}:
   idx=names.index(body);md.mocap_pos[m.body(body).mocapid]=d['body_pos_w'][f,idx];md.mocap_quat[m.body(body).mocapid]=d['body_quat_w'][f,idx]
  mujoco.mj_forward(m,md)
  for side in ('left','right'):
   for a,b in itertools.combinations([g for g in hand if g['body'].startswith(side)],2):
    rows.append({'frame':f,'mode':mode,'side':side,'geom_a':a['name'],'geom_b':b['name'],'signed_distance_m':mujoco.mj_geomDistance(m,md,m.geom(a['name']).id,m.geom(b['name']).id,10,None)})
write_csv(OUT/'hand_internal_pairs.csv',rows)
print({mode:[min(x['signed_distance_m'] for x in rows if x['mode']==mode),max(x['signed_distance_m'] for x in rows if x['mode']==mode)] for mode in ('native','hdmi_hand')})
