"""Publish a versioned NO-GO decision without modifying the M6 manifest."""
import copy
from common import *
from supermimic.hoi.asset_manifest import validate_manifest

def main():
    count=verify();old=json.loads(MANIFEST.read_text());new=copy.deepcopy(old)
    new['asset_id']='hdmi_bread_box_m6_5_review_v1'
    new['status']='M6_5_NO_GO_M7';new['training_ready']=False
    new['parent_manifest']={'path':str(MANIFEST),'sha256':sha(MANIFEST)}
    new['selection_note']='Unchanged nominal/native baseline selected for review, not a certified learning configuration. No tested minimal candidate reconciles physical hand contact over the preserved reference.'
    new['contact']['marker_semantics']='Official fixed EEF and target retained as soft semantic markers. Marker equality is not a physical contact constraint.'
    new['contact']['physical_validation']={'signed_pair_distance_band_m':[-.001,.002],'per_hand_actual_normal_force_threshold_N':1.,'bilateral_force_required':True,'geometry_distance_is_not_force':True,'closest_surface_points_are_state_dependent':True,'slip':'force-weighted contact-point relative tangential speed integral; separate from object displacement'}
    new['gates']['physical_hand_contact']={'status':'FAIL','detail':'840-frame geometry, four hand profiles, semantic fits, bounded pose probes and 42 short physical tests do not validate continuous interaction.'}
    new['gates']['scene_support_validity']={'status':'FAIL','detail':'Strict reference geometry remains inconsistent: 10 mm endpoint gaps; +10 mm correction causes 2.854/5.694 mm overlap. Original support poses retained.'}
    new['gates']['terminal_support_stability']={'status':'CONDITIONAL_PASS','detail':'Original nominal box at original terminal pose settles by about 10.108 mm and supports 19.62 N over final .5 s of 3 s probe. This is a reset-only settle, not achieved delivery or original 1 mm reference gate pass.'}
    new['review_required_before_M7']=True
    new['reconciliation_evidence']=[{'path':str(p),'sha256':sha(p)} for p in [OUT/'eef_candidates.json',OUT/'hand_candidates.json',OUT/'support_candidates.json',OUT/'box_pose_feasibility.csv',OUT/'dynamics_v1/summary.json',OUT/'dynamics_half_step/summary.json']]
    validate_manifest(new)
    path=ROOT/'data/assets/hdmi_bread_box/asset_manifest_m6_5.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(new,indent=2)+'\n')
    extra=[HDMI/'cfg/task/base/hdmi-base.yaml',ROOT/'.venv/lib/python3.12/site-packages/mjlab/actuator/builtin_actuator.py',ROOT/'.venv/lib/python3.12/site-packages/mjlab/entity/entity.py',ROOT/'.venv/lib/python3.12/site-packages/mjlab/utils/spec.py',ROOT/'.venv/lib/python3.12/site-packages/mjlab/sim/sim.py']
    save('additional_sources.json',[{'path':str(p),'sha256':sha(p)} for p in extra])
    save('integrity_final.json',{'all_preexisting_files_unchanged':True,'protected_file_count':count,'source_git_status':subprocess.check_output(['git','-C',str(SOURCE),'status','--short'],text=True),'original_manifest_sha256':sha(MANIFEST),'original_hoi_npz_sha256':sha(NPZ),'new_training_entry_point':False,'training_started':False,'M7_recommendation':'NO-GO','scope_complete':'M6.5 report, then pause for review'})
if __name__=='__main__':main()
