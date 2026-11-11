"""M6.5 independent sanity checks; a passing test does not approve M7."""
import sys,json
from pathlib import Path
import mujoco,numpy as np,pandas as pd,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/hoi/reconcile'))
from common import OUT,NPZ,load,sha
from geometry import clip_polygon,area
from dynamics import build_model,initialize,set_original_pd

def test_frozen_m6_and_references_unchanged():
    # The M6.5 evidence stays immutable. P3 necessarily extends this one shared
    # interface; pin its reviewed replacement hash instead of exempting the file.
    # Exact old/current default loss and gradient parity is tested separately in
    # tests/hoi/test_wm_reuse.py. All other historical hashes remain mandatory.
    expected=json.loads((OUT/'baseline.json').read_text())['protected_sha256']
    source=str(ROOT/'src/supermimic/world_model/world_model.py')
    assert expected[source]=='24a6157e285584fd1f8433360cc93b2c0b5fe68395fa46ccc4358109ce0e7ba2'
    expected[source]='d1e75e0ef550c8516701c736dff081ce681f8a0f114dd604f7a0386a2631d40a'
    assert len(expected)>100
    assert {p:sha(p) for p in expected}==expected

def test_footprint_intersection_analytic():
    a=np.array([[-.09,-.17],[.09,-.17],[.09,.17],[-.09,.17]])
    b=np.array([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
    assert area(clip_polygon(a,b))==pytest.approx(.036)
    assert area(clip_polygon(a+1,b))==0
    assert area(clip_polygon(a,b))==pytest.approx(area(clip_polygon(b,a)))

@pytest.mark.parametrize('profile',['native','hdmi'])
def test_original_actuator_parameters_and_no_assistance_paths(profile):
    from mjlab.entity import Entity
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    expected=Entity(get_g1_robot_cfg()).spec.compile()
    m,pd=build_model(profile)
    assert m.nu==29 and m.neq==0 and m.nmocap==0
    np.testing.assert_array_equal(m.actuator_gainprm,expected.actuator_gainprm)
    np.testing.assert_array_equal(m.actuator_biasprm,expected.actuator_biasprm)
    np.testing.assert_array_equal(m.actuator_forcerange,expected.actuator_forcerange)
    for i in range(m.nu):
        j=m.actuator_trnid[i,0];assert m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE
        assert pd[i]['joint']==m.joint(j).name
        assert m.dof_armature[m.jnt_dofadr[j]]==expected.dof_armature[expected.jnt_dofadr[expected.joint(m.joint(j).name).id]]
    d=mujoco.MjData(m);ref=load(NPZ);initialize(m,d,ref,260,True);set_original_pd(m,d,ref,260,pd)
    for i,p in enumerate(pd):assert d.ctrl[i]==pytest.approx(ref['joint_pos'][260,ref['joint_names'].tolist().index(p['joint'])],abs=1e-15)
    q=d.qpos.copy();mujoco.mj_step(m,d)
    assert not np.array_equal(q,d.qpos)
    assert not d.xfrc_applied.any() and not d.qfrc_applied.any()

def test_full_840_frame_support_sweep_and_footprints():
    s=pd.read_csv(OUT/'support_candidates.csv')
    for (_, _),x in s.groupby(['support','height_correction_m']):assert x.frame.tolist()==list(range(840))
    x=s[(s.support=='support1')&(s.height_correction_m==.01)]
    assert x.loc[x.distance_m.idxmin(),'frame']==544
    assert x.distance_m.min()==pytest.approx(-.00569365263,abs=1e-8)
    assert x[x.frame==839].overlap_area_m2.iloc[0]>0.035
    assert x[x.frame==839].distance_m.iloc[0]==pytest.approx(0,abs=1e-7)

def test_dynamics_command_hold_and_terminal_force_balance():
    for directory in ['dynamics_v1','dynamics_half_step']:
        results=json.loads((OUT/directory/'summary.json').read_text())
        for r in results:
            x=pd.read_csv(OUT/directory/r['name']/'trajectory.csv');dt=r.get('dt',.005)
            expected=np.minimum(r['start']+np.arange(len(x))//round(.02/dt),839)
            np.testing.assert_array_equal(x.reference_frame,expected)
            assert r['no_assistance_verified'] and r['neq']==0 and r['nmocap']==0
            if not r['robot']:
                assert r['terminal_stable_last_half_second']
                assert r['tail_mean_support_force_N']==pytest.approx(2*9.81,abs=.01)
                assert r['final_box_speed_m_s']<1e-5
    r=json.loads((OUT/'dynamics_v1/summary.json').read_text())
    assert len(r)==36

def test_old_measurement_reproduced_without_rewriting_it():
    old=pd.read_csv(ROOT/'docs/hoi/asset_audit/hand_box_frames.csv')
    new=pd.read_csv(OUT/'hand_candidates.csv')
    for a,b in [('native','native'),('hdmi_hand','hdmi')]:
        x=old[old['mode']==a].sort_values(['frame','side'])
        y=new[new.profile==b].sort_values(['frame','side'])
        np.testing.assert_allclose(x.min_hand_box_signed_distance_m,y.distance_m,atol=1e-12,rtol=0)
