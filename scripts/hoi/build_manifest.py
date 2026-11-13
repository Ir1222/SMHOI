"""Generate only static physical configuration and gate decisions, never motion."""
import copy,json
from asset_common import *
from supermimic.hoi.provenance import effective_sources, research_authority
def main():
    box=json.loads((OUT/'box_audit.json').read_text());robot=json.loads((OUT/'robot_audit.json').read_text());sup=json.loads((OUT/'support_audit.json').read_text());contact=json.loads((OUT/'contact_summary.json').read_text())
    baseline=json.loads((OUT/'baseline.json').read_text())
    files={p:v['sha256'] for p,v in baseline['inputs'].items()}
    for p in (SOURCE/'src/mjlab_g1/scripts/hoi_inspect_hdmi_usd.py',HDMI/'active_adaptation/assets/objects.py',HDMI/'active_adaptation/envs/mdp/commands/hdmi/randomizations.py'):
        files[str(p)]=sha(p)
    gs=copy.deepcopy(box['geoms'])
    for g in gs:g['priority']=1
    supports=[]
    for s in sup['supports']:
        g=copy.deepcopy(s['source_builder_geom']);g['pos']=[0,0,0];g['quat']=[1,0,0,0]
        supports.append({'name':s['name'],'fixed':True,'root_pos_world':s['root_pos_world'],'root_quat_wxyz':s['root_quat_wxyz'],'geom':g,'source':s['usd_source']})
    gates={
      'asset_parity':{'status':'PASS_WITH_DOCUMENTED_DEVIATION','detail':'Seven XML boxes, root and inertia exact. priority 1 rather than XML default 0, inherited from source env.py. Nominal XML 2 kg, not randomized HDMI runtime.'},
      'robot_collider_parity':{'status':'MEASURED_DECISION_RECORDED','detail':'Native retained; optional hand-only HDMI profile for comparison. Not dynamically equivalent; feet unchanged.'},
      'contact_target_surface':{'status':'PASS','tolerance_m':1e-7},
      'eef_convention':{'status':'PASS','detail':'Official wrist yaw +5 cm local X; internal marker, not collision surface.'},
      'physical_hand_contact':{'status':'FAIL','detail':'Both profiles have cm-scale overlap and separation in desired-contact frames; zero EEF error also induces penetration.'},
      'reference_integrity':{'status':'PASS','detail':'All six numeric arrays byte-identical, all nine shared fields value-identical; input archive hashes unchanged.'},
      'scene_support_validity':{'status':'FAIL','tolerance_m':.001,'measured_start_end_gap_m':.010000021904706948},
      'coordinate_consistency':{'status':'PASS_POSE_CONVENTION','detail':'wxyz, Z-up, meters; target wrist FK matches within 8.2e-8 m. Existing hoi_v1 velocities have pose_fd_link_world_v1 provenance; raw-source velocity origin unresolved.'}}
    manifest={'schema_version':1,'asset_id':'hdmi_bread_box_nominal_audit_v1','status':'AUDITED_BLOCKED_CONTACT_AND_SUPPORT','training_ready':False,
      'units':{'length':'m','mass':'kg','inertia':'kg*m^2','angle':'rad','quaternion':'wxyz','world_up':'Z'},
      'sources':effective_sources([{'path':p,'sha256':h} for p,h in sorted(files.items())]),
      'research_authority':research_authority(),
      'object':{'name':'bread_box','dynamic':True,'joint':'free','scale':1.0,'inertia':box['inertia'],'geoms':gs,
        'diagnostic_aabb_only':box['aabb'],'restitution':None,'contact_parameter_semantics':'MuJoCo friction sliding/torsional/rolling; defaults solref/solimp retained. No direct restitution mapping.',
        'official_runtime_distinctions':{'raw_xml_usd_mass_kg':2.,'objects_py_spawn_mass_kg':3.,'task_randomization':box['task_randomization']},'com_warning':box['com_warning']},
      'supports':supports,
      'robot':{'native_asset':robot['target_xml'],'selected_profile':'native_unchanged','feet_policy':'native_unchanged','inertia_policy':'retain_native_robot_inertias; raw HDMI wrist mass and inertia differ',
        'profiles':{'native_unchanged':{'description':'No mutations'},'hdmi_hands_only_candidate':{'description':'Optional audit comparison; not training-approved. Geometry from raw USD, no foot changes.',
          'disable_geoms':['left_wrist_collision','left_hand_collision','right_wrist_collision','right_hand_collision'],
          'geoms':[g for g in robot['hdmi_comparison_geoms'] if 'wrist' in g['body']]} }},
      'contact':{'target_root_m':box['task_contact']['contact_target_pos_offset'],'wrist_bodies':['left_wrist_yaw_link','right_wrist_yaw_link'],'eef_offset_wrist_m':box['task_contact']['contact_eef_pos_offset'],
        'eef_formula':'p_eef_world = p_wrist_world + R(q_wrist_wxyz) @ offset_wrist',
        'target_formula':'p_target_world = p_box_root_world + R(q_box_wxyz) @ target_root',
        'box_com_formula':'p_com_world = p_box_root_world + R(q_box_wxyz) @ com_root',
        'velocity_formula':'v_com_world = v_root_world + omega_world cross (R(q_box) @ com_root); MuJoCo free-joint angular qvel is body-local',
        'sensor_contract':{'hand_primary_bodies':['left_wrist_pitch_link','left_wrist_yaw_link','right_wrist_pitch_link','right_wrist_yaw_link'],'secondary_body':'bread_box','secondary_geoms':[g['name'] for g in gs],'semantics':'future physical contacts must filter actual enabled geoms; object_contact is desired intent, not force or bilateral contact'},
        'distance_semantics':contact['method']},
      'deliberate_deviations':[{'field':'object.geoms.*.priority','official_xml':0,'selected':1,'reason':'Match source env.py object-vs-robot contact priority; explicit MuJoCo adaptation'},
        {'field':'supports','reason':'Fixed MuJoCo bodies for constant recorded kinematic supports; USD size rounded to authored decimal values within 2e-9 m'},
        {'field':'robot.feet','reason':'Keep native seven capsules; do not import unrelated HDMI four-sphere feet'},
        {'field':'randomization','reason':'Nominal static audit asset only; task material/mass/scale randomization not activated'}],
      'gates':gates,'unresolved':['Reference hand-box penetration/separation','10 mm original support gap','Official COM outside collision envelope; preserve, do not silently repair','Physical material/solver parity across PhysX and MuJoCo not established']}
    path=ROOT/'data/assets/hdmi_bread_box/asset_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    save('gates.json',{'training_ready':False,'gates':gates});print(path)
if __name__=='__main__':main()
