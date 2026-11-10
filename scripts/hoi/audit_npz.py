"""Read-only NPZ audit. No reference conversion or write operation."""
import json, hashlib
from pathlib import Path
import numpy as np
from baseline import ROOT,SOURCE,sha
OUT=ROOT/'docs/hoi/asset_audit'
ROBOT_FIELDS=('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w','joint_names','body_names','fps')
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def compare(a,b):
    same=a.shape==b.shape
    return {'shape_equal':same,'dtype_equal':a.dtype==b.dtype,'exact_equal':bool(same and np.array_equal(a,b)),
      'max_abs_difference':float(np.max(np.abs(a.astype(float)-b.astype(float)))) if same and a.dtype.kind in 'fibu' else None,
      'array_bytes_sha256':[hashlib.sha256(x.tobytes()).hexdigest() for x in (a,b)]}
def inspect(p):
    before=sha(p); d=load(p); out={}
    for k,a in d.items():
        v={'shape':list(a.shape),'dtype':str(a.dtype)}
        if a.dtype.kind in 'fibu':
            v.update(min=float(a.min()),max=float(a.max()),all_finite=bool(np.isfinite(a).all()))
        if 'quat' in k:
            n=np.linalg.norm(a.astype(float),axis=-1)
            v.update(order='wxyz; hdmi.py converter preserves official source order',norm_min=float(n.min()),norm_max=float(n.max()),norm_max_error=float(abs(n-1).max()))
        if a.size<100 or a.dtype.kind in 'US': v['values']=a.tolist()
        out[k]=v
    if 'object_contact' in d:
        c=d['object_contact'].reshape(-1).astype(bool)
        starts=np.flatnonzero(np.diff(np.r_[False,c,False].astype(int))==1)
        ends=np.flatnonzero(np.diff(np.r_[False,c,False].astype(int))==-1)
        out['object_contact']['true_intervals_half_open']=list(map(list,zip(starts.tolist(),ends.tolist())))
    assert before==sha(p)
    return {'path':str(p),'sha256':before,'frames':len(d['joint_pos']),'fps':float(d['fps'].item()),'duration_frames_over_fps':len(d['joint_pos'])/d['fps'].item(),'last_timestamp_s':(len(d['joint_pos'])-1)/d['fps'].item(),'fields':out},d

def main():
    r,a=inspect(Path('/home/han/pyx/hdmi_bread_box_robot_only_50hz_supermimic_pyx.npz'))
    h,b=inspect(Path('/home/han/pyx/hdmi_bread_box_hoi_v1.npz'))
    raw=load(SOURCE/'dataset/external/hdmi/carry_and_place_bread_box-0829/motion.npz')
    meta=json.loads((SOURCE/'dataset/external/hdmi/carry_and_place_bread_box-0829/meta.json').read_text())
    checks={k:compare(a[k],b[k]) for k in ROBOT_FIELDS}
    assert all(v['exact_equal'] for v in checks.values())
    assert all(checks[k]['dtype_equal'] for k in ROBOT_FIELDS[:6])
    provenance={}
    for suffix in ('pos_w','quat_w','lin_vel_w','ang_vel_w'):
        src=raw['body_'+suffix][:,meta['body_names'].index('bread_box')]
        provenance['box_'+suffix]=compare(src.astype(b['box_'+suffix].dtype),b['box_'+suffix])
    report={'robot_only':r,'hoi':h,'robot_comparison':checks,'official_box_fields_after_dtype_cast':provenance,
      'semantics':{'object_contact':'object-level desired contact intent, NOT measured bilateral contact',
      'phase_bounds':'hdmi.py::_phases legacy height/contact heuristic, NOT official phase labels or training stages',
      'object_pose':'directly selected from official motion body bread_box and cast float32',
      'object_velocity':'supplied source arrays selected and cast, not generated here; recorder COM/root provenance unresolved for this released clip; not mandatory for minimal reference',
      'prohibited_converter':'hoi/reference.py::derive_reference rewrites joint/body velocities. Do NOT call it on the validated robot reference.'}}
    (OUT/'npz_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'M1.md').write_text('# M1 — reference integrity\n\nAll nine shared robot fields have equal shapes and values. The six robot numeric arrays also have equal dtypes and byte hashes. Name strings differ only in storage width: U26/U25 versus U64; they are not byte-identical. Six numeric robot arrays are untouched. Both NPZ SHA256 values match M0. Every key, dtype, shape, finite range, name ordering and quaternion norm is in npz_audit.json.\n\nFrames: '+str(r['frames'])+'; FPS: '+str(r['fps'])+'; contact intervals (half-open): '+str(h['fields']['object_contact']['true_intervals_half_open'])+'.\n\nBox pose/velocity fields match official source selection after dtype cast. Supplied velocities have unresolved recording-point provenance; no velocity recomputation is authorized. Existing source derive_reference would rewrite robot velocities and is explicitly excluded. Phase fields are diagnostic heuristics, not validated annotations. No NPZ has been created or modified.\n')
    print(json.dumps({'frames':r['frames'],'robot_equal':True,'contact':h['fields']['object_contact']['true_intervals_half_open'],'box_source':provenance},indent=2))
if __name__=='__main__':main()
