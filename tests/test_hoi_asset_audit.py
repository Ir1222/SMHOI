"""Focused offline physical-contract checks. Passing tests do not pass failed gates."""
import copy,json,sys
from pathlib import Path
import mujoco,numpy as np,pytest
from scipy.spatial.transform import Rotation
from supermimic.hoi.asset_manifest import load_manifest,validate_manifest,build_audit_scene,apply_hand_profile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/hoi'))
from asset_common import XML,geom,inertia
from audit_robot import target_spec
from audit_contact import box_point,comparison_model
from audit_npz import load,ROBOT_FIELDS

@pytest.fixture(scope='module')
def manifest():return load_manifest(ROOT/'data/assets/hdmi_bread_box/asset_manifest.json')

def test_inputs_and_robot_arrays_preserved(manifest):
    a=load(Path('/home/han/pyx/hdmi_bread_box_robot_only_50hz_supermimic_pyx.npz'))
    b=load(Path('/home/han/pyx/hdmi_bread_box_hoi_v1.npz'))
    for k in ROBOT_FIELDS:np.testing.assert_array_equal(a[k],b[k])
    for k in ROBOT_FIELDS[:6]:assert a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes()

def test_compiled_box_xml_parity(manifest):
    model=build_audit_scene(manifest,[0,0,1],[1,0,0,0]).compile()
    xml=mujoco.MjModel.from_xml_path(str(XML))
    assert model.nq==7 and model.nv==6
    assert inertia(model,model.body('bread_box').id)==inertia(xml,1)
    for i in range(7):
        a=geom(xml,i);b=geom(model,model.geom(a['name']).id)
        for k in ('type','pos','quat','size','friction','condim','contype','conaffinity','solref','solimp'):
            assert a[k]==b[k],(a['name'],k)
        assert a['priority']==0 and b['priority']==1

def test_target_exposed_surface_and_world_equivariance(manifest):
    R=Rotation.from_rotvec([.31,-.2,1.4]);p=np.array([2,-1,.7])
    for target,idx in zip(manifest['contact']['target_root_m'],[3,4]):
        g=manifest['object']['geoms'][idx];sd,cp,n=box_point(target,g)
        assert abs(sd)<1e-7
        # Independent transformed primitive oracle must agree on outward side.
        wg=copy.deepcopy(g);wg['pos']=(p+R.apply(g['pos'])).tolist()
        q=R.as_quat();wg['quat']=q[[3,0,1,2]].tolist()
        world=p+R.apply(target)
        assert abs(box_point(world,wg)[0])<1e-7
        assert box_point(world+R.apply(n)*.001,wg)[0]==pytest.approx(.001)
        assert box_point(world-R.apply(n)*.001,wg)[0]==pytest.approx(-.001)

def test_eef_rotation_convention():
    # 90 deg around Z must send +X offset to +Y, never +X world.
    q=np.array([np.sqrt(.5),0,0,np.sqrt(.5)])
    from asset_common import rot
    np.testing.assert_allclose(rot(q).apply([.05,0,0])+[1,2,3],[1,2.05,3],atol=1e-15)

def test_manifest_hand_profile_does_not_change_feet_or_inertias(manifest):
    spec=target_spec();before=spec.compile()
    apply_hand_profile(spec,manifest,'hdmi_hands_only_candidate');after=spec.compile()
    for i in range(before.ngeom):
        g=geom(before,i)
        if 'foot' in g['name']:assert g==geom(after,after.geom(g['name']).id)
    np.testing.assert_array_equal(before.body_mass,after.body_mass)
    np.testing.assert_array_equal(before.body_ipos,after.body_ipos)
    np.testing.assert_array_equal(before.body_inertia,after.body_inertia)
    for n in manifest['robot']['profiles']['hdmi_hands_only_candidate']['disable_geoms']:
        assert after.geom_contype[after.geom(n).id]==0
    for g in manifest['robot']['profiles']['hdmi_hands_only_candidate']['geoms']:
        for k in ('pos','quat','size'):np.testing.assert_allclose(geom(after,after.geom(g['name']).id)[k],g[k],atol=1e-7)

def test_scene_origin_and_reset_preserve_measured_gap(manifest):
    data=load(Path('/home/han/pyx/hdmi_bread_box_hoi_v1.npz'))
    for frame,support in [(0,'support0'),(839,'support1')]:
        distances=[]
        for origin in ([0,0,0],[3,-4,.5]):
            m=build_audit_scene(manifest,data['box_pos_w'][frame],data['box_quat_w'][frame],origin).compile()
            d=mujoco.MjData(m);mujoco.mj_resetData(m,d);mujoco.mj_forward(m,d)
            np.testing.assert_allclose(d.xpos[m.body('bread_box').id],data['box_pos_w'][frame]+origin)
            distances.append(mujoco.mj_geomDistance(m,d,m.geom('bottom').id,m.geom(support).id,10,None))
        np.testing.assert_allclose(distances,[.010000022,.010000022],atol=1e-8)

def test_primitive_overlap_oracle():
    # Independent simple cubes: 0.1m separation, exact touch, 0.1m penetration.
    g={'name':'a','body':'a','type':'box','pos':[0,0,0],'quat':[1,0,0,0],'size':[.2,.2,.2]}
    h={**g,'name':'b','body':'b'};m,d=comparison_model([g],[h])
    for x,want in [(.5,.1),(.4,0),(.3,-.1)]:
        d.mocap_pos[m.body('b').mocapid]=[x,0,0];mujoco.mj_forward(m,d)
        assert mujoco.mj_geomDistance(m,d,0,1,10,None)==pytest.approx(want,abs=1e-7)

@pytest.mark.parametrize('mutation',['size','quat','approval'])
def test_invalid_manifest_rejected(manifest,mutation):
    m=copy.deepcopy(manifest)
    if mutation=='size':m['object']['geoms'][0]['size'][0]=-1
    elif mutation=='quat':m['object']['geoms'][0]['quat']=[0,0,0,0]
    else:m['training_ready']=True
    with pytest.raises(ValueError):validate_manifest(m,verify_sources=False)

def test_geometry_gate_remains_blocked(manifest):
    assert not manifest['training_ready']
    assert manifest['gates']['physical_hand_contact']['status']=='FAIL'
    assert manifest['gates']['scene_support_validity']['status']=='FAIL'

def test_xml_matches_independently_parsed_usd(manifest):
    from asset_common import vertices
    raw=json.loads((ROOT/'docs/hoi/asset_audit/usd_reread.json').read_text())['object']
    for g,u in zip(manifest['object']['geoms'],raw['colliders'],strict=True):
        v=vertices(g)
        np.testing.assert_allclose(np.stack([v.min(0),v.max(0)]),u['body_relative_vertex_bounds'],atol=2e-8,rtol=0)
    inertia=raw['bodies'][0]['attributes']
    np.testing.assert_allclose(manifest['object']['inertia']['com_pos_root_m'],inertia['physics:centerOfMass'],atol=1e-8,rtol=0)
    np.testing.assert_allclose(manifest['object']['inertia']['diagonal_kg_m2'],inertia['physics:diagonalInertia'],atol=1e-8,rtol=0)
