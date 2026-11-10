"""Offline asset extraction helpers; no environment or training imports."""
import ast,json,math
from collections import namedtuple
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from baseline import ROOT,SOURCE,HDMI,sha
OUT=ROOT/'docs/hoi/asset_audit'
XML=HDMI/'active_adaptation/assets_mjcf/objects/bread_box/bread_box.xml'
G1=ROOT/'.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
def save(name,data): (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def rot(q): return Rotation.from_quat(np.asarray(q)[...,[1,2,3,0]])
def selected_source(path,names,ns):
    """Execute only named, inspected AST definitions, without importing source training code."""
    tree=ast.parse(path.read_text()); nodes=[]
    for n in tree.body:
        name=n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
        if name in names: nodes.append(n)
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

def source_box():
    ns={'mujoco':mujoco,'ObjectBox':namedtuple('ObjectBox','name pos_root half_extents')}
    selected_source(SOURCE/'src/mjlab_g1/hoi/hdmi.py',{'HDMI_BREAD_BOX_PARTS'},ns)
    return selected_source(SOURCE/'src/mjlab_g1/hoi/env.py',{'box_spec'},ns)['box_spec']().compile()
def geom(m,i):
    r={'name':m.geom(i).name,'body':m.body(m.geom_bodyid[i]).name,'type':mujoco.mjtGeom(int(m.geom_type[i])).name.removeprefix('mjGEOM_').lower()}
    for key in ('pos','quat','size','friction','solref','solimp','contype','conaffinity','condim','priority','group','margin','gap','solmix'):
        r[key]=getattr(m,'geom_'+key)[i].tolist()
    r['mesh']=m.mesh(int(m.geom_dataid[i])).name if r['type']=='mesh' else None
    return r

def inertia(m,b):
    return {'mass_kg':float(m.body_mass[b]),'com_pos_root_m':m.body_ipos[b].tolist(),'principal_axes_wxyz':m.body_iquat[b].tolist(),'diagonal_kg_m2':m.body_inertia[b].tolist()}
def vertices(g):
    import itertools
    return rot(g['quat']).apply(np.array(list(itertools.product((-1,1),repeat=3)))*g['size'])+g['pos']
def main_box():
    import xml.etree.ElementTree as ET,yaml
    m=mujoco.MjModel.from_xml_path(str(XML)); other=source_box()
    gs=[geom(m,i) for i in range(m.ngeom)]; sg=[geom(other,i) for i in range(other.ngeom)]
    differences=[]
    for a,b in zip(gs,sg,strict=True):
        diff={k:{'xml':a[k],'source_builder':b[k]} for k in a if a[k]!=b[k]}
        differences.append({'geom':a['name'],'differences':diff})
    vs=np.concatenate([vertices(g) for g in gs]); main=np.concatenate([vertices(g) for g in gs[:5]])
    task=yaml.safe_load((HDMI/'cfg/task/G1/hdmi/carry_and_place_bread_box.yaml').read_text())
    d={'source':{'path':str(XML),'sha256':sha(XML)},'mujoco_version':mujoco.__version__,'geoms':gs,'inertia':inertia(m,1),
      'root_initial_pose_xml':{'pos':m.body_pos[1].tolist(),'quat':m.body_quat[1].tolist()},
      'aabb':{'min':vs.min(0).tolist(),'max':vs.max(0).tolist(),'dimensions':np.ptp(vs,axis=0).tolist(),'center':((vs.max(0)+vs.min(0))/2).tolist()},
      'main_dimensions':np.ptp(main,axis=0).tolist(),'source_builder_differences':differences,'source_builder_inertia':inertia(other,1),
      'inertia_exact_equal':inertia(m,1)==inertia(other,1),'restitution':None,'restitution_note':'not specified in MJCF; solref/solimp are not a PhysX restitution coefficient',
      'task_randomization':task['randomization'],'task_contact':{k:v for k,v in task['command'].items() if k.startswith('contact_')},
      'com_warning':'COM is 0.19 m below lowest collision point; cannot describe positive mass confined to these seven solids. Preserve official model for parity; never silently recenter.',
      'obb_error':'source diagnostic OBB spans z=[0,.26], missing bottom [-.01,0]; correct union center .125 and half-height .135'}
    assert len(gs)==7 and d['inertia_exact_equal']
    save('box_audit.json',d)
    (OUT/'M2.md').write_text('# M2 — official bread box\n\nCompiled local XML and actual source box_spec() (isolated AST, no environment launch). Seven boxes and inertia match exactly. Name/body names differ; source raises priority 0 → 1 on every geom. XML implicit MuJoCo defaults: friction [1, .005, .0001], condim 3, masks 1/1, priority 0; source uses same friction and condim, priority 1. No direct restitution coefficient in MJCF.\n\nMain dimensions: .18 × .34 × .27 m; including handles: .18 × .38 × .27 m. AABB [-.09,-.19,-.01] to [.09,.19,.26]; center [0,0,.125]. Source diagnostic OBB omits bottom 1 cm. It is never used as collision geometry.\n\nMass 2 kg; COM [0,0,-.2] m; inertia [.0334,.02167,.04167] kg m² at COM, identity principal axes. COM is outside the collision envelope (19 cm below it). Preserve this unusual official choice explicitly; do not invent corrected inertia. YAML randomization (mass 1.2–1.8, scale .9–1.1, friction/restitution ranges) is distinct from nominal raw asset. Full compiled data and differences: box_audit.json.\n')
    print(d['aabb'])
if __name__=='__main__':main_box()
