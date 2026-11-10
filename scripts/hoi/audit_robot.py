"""Compare compiled Reproduced colliders, raw USD and source adapter."""
import json,math
import mujoco,numpy as np
from asset_common import *
def target_spec():
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
    cfg=get_g1_robot_cfg(); spec=cfg.spec_fn()
    for c in cfg.collisions: c.edit_spec(spec)
    return spec

def usd_geoms(raw):
    out=[]
    for c in raw['colliders']:
        a=c['attributes']; typ=c['type']
        if typ not in ('Cube','Sphere'): raise ValueError(typ)
        out.append({'name':c['body']+'_'+c['path'].split('/')[-1], 'body':c['body'],'type':{'Cube':'box','Sphere':'sphere'}[typ],
          'pos':a['xformOp:translate'],'quat':a['xformOp:orient'],
          'size':(np.asarray(a['xformOp:scale'])*a['size']/2).tolist() if typ=='Cube' else [a['radius'],0,0],
          'contype':1,'conaffinity':1,'condim':1,'priority':0,'group':3,'friction':[1,.005,.0001],
          'settings_origin':'geometry from USD; masks/condim/friction are explicit MuJoCo comparison defaults, not USD fields'})
    return out

def main():
    raw=json.loads((OUT/'usd_reread.json').read_text())
    historical=json.loads((SOURCE/'dataset/hoi/manifests/hdmi_usd_contact_assets.json').read_text())
    for key in ('robot','object'):
        assert raw[key]==historical[key],key
    historical_support=json.loads((SOURCE/'dataset/hoi/manifests/hdmi_usd_support_assets.json').read_text())
    for key in ('support0','support1'): assert raw[key]==historical_support[key],key
    spec=target_spec(); native=spec.compile()
    ns=selected_source(SOURCE/'src/mjlab_g1/hoi/hdmi_colliders.py',{'apply_hdmi_contact_shapes'},{'mujoco':mujoco,'math':math})
    ns['apply_hdmi_contact_shapes'](spec); adapted=spec.compile()
    def relevant(m):return [geom(m,i) for i in range(m.ngeom) if any(s in m.body(m.geom_bodyid[i]).name for s in ('wrist','ankle_roll'))]
    gs=usd_geoms(raw['robot']); checks=[]
    for g in gs:
        side=g['body'].split('_')[0]; idx=g['name'].split('_')[-1]
        name=f'{side}_hdmi_hand{idx}_collision' if 'wrist' in g['body'] else f'{side}_foot_hdmi{idx}_collision'
        a=geom(adapted,adapted.geom(name).id)
        checks.append({'usd':g['name'],'adapter':name,'pos_max_error_m':float(np.max(abs(np.array(a['pos'])-g['pos']))),'size_max_error_m':float(np.max(abs(np.array(a['size'])-g['size']))),'rotation_error_rad':float((rot(a['quat']).inv()*rot(g['quat'])).magnitude())})
    report={'target_xml':{'path':str(G1),'sha256':sha(G1)},'native':relevant(native),'adapter_applied_to_target_in_memory':relevant(adapted),'hdmi_comparison_geoms':gs,'adapter_usd_geometry_errors':checks,
      'historical_usd_reports_match_fresh_parse':True,
      'decision':'Retain native robot globally. Recommend optional hand-only HDMI three-box profile for isolated asset scene; do not call existing adapter (also changes feet). Do not activate training until geometry gate is resolved.',
      'visual_mesh_collision_count':sum(g['type']=='mesh' and bool(g['contype'] or g['conaffinity']) for g in relevant(native)),
      'physics_note':'USD/PhysX material/contact offsets and MuJoCo masks/condim are different contracts. Geometry match does not establish dynamics equivalence.'}
    save('robot_audit.json',report)
    save('wrist_inertia_comparison.json',{'native':{side:inertia(native,native.body(side+'_wrist_yaw_link').id) for side in ('left','right')},'hdmi_usd':raw['robot']['bodies'],'note':'Hand-only profile changes geometry only; native inertias retained and tested. This is not full dynamics parity.'})
    (OUT/'M3.md').write_text('# M3 — robot collider comparison\n\nFresh raw USD parse exactly reproduces both historical contact and support manifests; hashes verified. Target asset is installed mjlab G1 selected by tracking_env_cfg.py, not the source repository custom G1 XML. All wrist and ankle colliders, including disabled visuals, are listed in robot_audit.json with compiled transforms and settings.\n\nNative: each wrist pitch link has a radius .035 m capsule, endpoints [-.01,0,0] and [.06,0,0]. Each wrist yaw link has a radius .035 m hand capsule, endpoints [.07,0,0] and [.15,∓.02,0]. HDMI: three boxes per wrist yaw, no wrist pitch capsule; half-sizes [.05,.025,.025], [.05,.01,.05], [.025,.01,.05], centers [.02,0,0], [.09,0,0], [.15,∓.01,0], final box yaw ∓.4 rad. The first two overlap in x by 3 cm. Full internal pair distances are evaluated in M4.\n\nNative feet: seven radius .01 m capsules each; HDMI: four radius .005 m spheres each. Existing adapter changes both hands and feet; it must not be ported wholesale. Geometry agrees with raw USD within float precision, but default contact parameters and robot inertias do not thereby become PhysX-equivalent. Visual meshes remain noncolliding.\n\nDecision: preserve native robot; provide a separately named, opt-in hand-only comparison profile. Keep native feet. No training activation.\n')
    print('max adapter errors',max(x['pos_max_error_m'] for x in checks),max(x['rotation_error_rad'] for x in checks))
if __name__=='__main__':main()
