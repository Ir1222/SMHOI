"""Part B offline motion contract. No simulator, environment, or training integration."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

ROBOT_NUMERIC = (
    'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
    'body_lin_vel_w', 'body_ang_vel_w',
)
ROBOT_FIELDS = ROBOT_NUMERIC + ('joint_names', 'body_names', 'fps')
OBJECT_SOURCE = {
    'object_pos_w': 'box_pos_w', 'object_quat_w': 'box_quat_w',
    'object_contact': 'object_contact',
}
FIELDS = ROBOT_FIELDS + tuple(OBJECT_SOURCE)
TIME_FIELDS = ROBOT_NUMERIC + tuple(OBJECT_SOURCE)
REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROBOT = Path('/home/han/pyx/hdmi_bread_box_robot_only_50hz_supermimic_pyx.npz')
DEFAULT_HOI = Path('/home/han/pyx/hdmi_bread_box_hoi_v1.npz')
DEFAULT_ASSET = REPO / 'data/assets/hdmi_bread_box/asset_manifest_m6_5.json'
DEFAULT_OUTPUT = REPO / 'outputs/hoi/asset_audit/part_b/reference'


def sha256(path: str | Path) -> str:
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def array_info(a: np.ndarray) -> dict:
    return {'shape': list(a.shape), 'dtype': a.dtype.str,
            'bytes_sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def validate_arrays(a: dict) -> dict:
    if set(a) != set(FIELDS):
        raise ValueError(f'Expected minimal fields {FIELDS}; got {tuple(a)}')
    n = len(a['joint_pos'])
    shapes = {'joint_pos': (n, 29), 'joint_vel': (n, 29),
              'body_pos_w': (n, 30, 3), 'body_quat_w': (n, 30, 4),
              'body_lin_vel_w': (n, 30, 3), 'body_ang_vel_w': (n, 30, 3),
              'joint_names': (29,), 'body_names': (30,), 'fps': (1,),
              'object_pos_w': (n, 3), 'object_quat_w': (n, 4),
              'object_contact': (n,)}
    for k, shape in shapes.items():
        if a[k].shape != shape:
            raise ValueError(f'{k}: invalid shape {a[k].shape}, expected {shape}')
        if k.endswith('_names'):
            if a[k].dtype.kind != 'U' or len(set(a[k].tolist())) != len(a[k]):
                raise ValueError(f'{k}: expected unique Unicode names')
        elif not np.isfinite(a[k]).all():
            raise ValueError(f'{k}: nonfinite values')
    for k in ROBOT_NUMERIC + ('object_pos_w', 'object_quat_w'):
        if a[k].dtype.kind != 'f':
            raise ValueError(f'{k}: expected floating point')
    if n != 840 or a['fps'][0] != 50:
        raise ValueError('This HDMI contract requires 840 frames at 50 Hz')
    if a['object_contact'].dtype != np.dtype(bool):
        raise ValueError('Desired contact must retain source bool dtype')
    if 'pelvis' not in a['body_names']:
        raise ValueError('Missing pelvis root')
    errors = {}
    for k in ('body_quat_w', 'object_quat_w'):
        errors[k] = float(np.max(np.abs(np.linalg.norm(a[k].astype(float), axis=-1) - 1)))
        if errors[k] > 1e-5:
            raise ValueError(f'{k}: quaternion norm error {errors[k]} exceeds 1e-5')
    return {'frame_count': n, 'fps': 50, 'last_sample_time_s': (n - 1) / 50,
            'sample_coverage_s': n / 50, 'quaternion_order': 'wxyz',
            'max_quaternion_norm_error': errors,
            'quaternions_rewritten': False}


def deterministic_npz(arrays: dict) -> bytes:
    """Fixed member order, timestamp and compression; never normalizes array values."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for k in FIELDS:
            data = io.BytesIO()
            np.lib.format.write_array(data, arrays[k], allow_pickle=False)
            info = zipfile.ZipInfo(k + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            z.writestr(info, data.getvalue(), compresslevel=9)
    return out.getvalue()


def convert(robot: Path, hoi: Path, asset: Path, output: Path) -> dict:
    """Create a new bundle; sources and existing output directories are never written."""
    from supermimic.hoi.asset_manifest import load_manifest

    robot, hoi, asset, output = map(lambda p: Path(p).resolve(), (robot, hoi, asset, output))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    inputs = {role: {'path': str(p), 'sha256': sha256(p)}
              for role, p in [('robot', robot), ('hoi', hoi), ('asset', asset)]}
    physical = load_manifest(asset)  # Validate provenance; does not approve training.
    shared = {}
    with np.load(robot, allow_pickle=False) as r, np.load(hoi, allow_pickle=False) as h:
        for k in ROBOT_FIELDS:
            x, y = r[k], h[k]
            equal = x.shape == y.shape and np.array_equal(x, y)
            same_bytes = x.dtype == y.dtype and x.tobytes() == y.tobytes()
            shared[k] = {'value_equal': equal, 'shape_equal': x.shape == y.shape,
                         'dtype_equal': x.dtype == y.dtype, 'bytes_equal': same_bytes,
                         'robot_dtype': x.dtype.str, 'hoi_dtype': y.dtype.str,
                         'max_abs_difference': (float(np.max(np.abs(x - y)))
                                                if equal and x.dtype.kind not in 'US' else None)}
            if not equal or (k in ROBOT_NUMERIC and not same_bytes):
                raise ValueError(f'Shared robot field mismatch: {k}')
        arrays = {k: r[k].copy() for k in ROBOT_FIELDS}
        # Deliberately no reads of HOI support, phase, velocity or contact-geometry arrays.
        arrays.update({k: h[source].copy() for k, source in OBJECT_SOURCE.items()})
    checks = validate_arrays(arrays)
    payload = deterministic_npz(arrays)
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        for k in FIELDS:
            if array_info(z[k]) != array_info(arrays[k]):
                raise ValueError(f'Serialized array changed: {k}')
    for item in inputs.values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Input changed during conversion')
    manifest = {
        'schema_version': 1, 'kind': 'minimal_hoi_motion_reference',
        'artifact': 'reference.npz', 'sha256': hashlib.sha256(payload).hexdigest(),
        'sources': inputs, 'fields': {k: array_info(arrays[k]) for k in FIELDS},
        'field_sources': {**{k: f'robot:{k}' for k in ROBOT_FIELDS},
                          **{k: f'hoi:{v}' for k, v in OBJECT_SOURCE.items()}},
        'motion': checks, 'units': {'position': 'm', 'time': 's', 'joint_angle': 'rad',
                                   'quaternion': 'wxyz', 'world_up': 'Z'},
        'object_contact_semantics': 'desired intent, not measured contact or force',
        'contact_geometry': {'authority': 'sources.asset',
                             'target': 'contact.target_root_m',
                             'eef': 'contact.eef_offset_wrist_m',
                             'constant_targets_in_npz': False},
        'training_ready': False, 'asset_status': physical['status'],
        'conversion': 'Exact canonical robot copies; rename object pose; copy desired contact. '
                      'No normalization, interpolation, retargeting or velocity recomputation.',
    }
    profile = {
        'schema_version': 1, 'name': 'part_b_hdmi_reference_inspection',
        'mode': 'kinematic_display_only', 'reference': 'reference.npz',
        'reference_manifest': 'reference_manifest.json',
        'reference_sha256': manifest['sha256'], 'asset_manifest': inputs['asset'],
        'training_ready': False, 'asset_training_ready': physical['training_ready'],
        'simulation_operation': 'pose_assignment_then_mj_forward',
        'reference_samples': 840, 'fps': 50,
        'excluded': ['physics stepping', 'PD', 'training', 'root assistance',
                     'object actuation', 'reference trajectory modification'],
    }
    validation = {'status': 'PASS', 'shared_robot_fields': shared, 'motion': checks,
                  'roundtrip_all_fields_exact': True, 'inputs_unchanged': True,
                  'deterministic_serialization': payload == deterministic_npz(arrays),
                  'physical_gate_approval': False}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'reference.npz').write_bytes(payload)
    for name, value in [('reference_manifest.json', manifest),
                        ('validation.json', validation), ('experiment_profile.json', profile)]:
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    rows = '\n'.join(f'| `{k}` | {v["dtype"]} | {v["shape"]} |'
                     for k, v in manifest['fields'].items())
    (output / 'VALIDATION.md').write_text(
        '# B1 reference conversion\n\nPASS: shared robot fields verified before merge; '
        'six numeric robot arrays and all canonical metadata preserved exactly. '
        'HOI names have wider Unicode dtype but identical values/order; canonical widths retained.\n\n'
        f'SHA256: `{manifest["sha256"]}`. 840 frames, 50 Hz; final timestamp 16.78 s; '
        'sample coverage 16.80 s. Quaternion wxyz norms pass 1e-5 without normalization.\n\n'
        '| Field | dtype | Shape |\n|---|---|---|\n' + rows + '\n\n'
        'Object pose and desired contact retain exact source values/dtypes. Fixed targets, EEF '
        'offsets, supports and seven-geom assets are resolved from the hashed M6.5 manifest, '
        'so no static geometry is duplicated in this 12-field motion artifact. No phases or '
        'object velocities included. Desired contact is not measured contact.\n\n'
        'See validation.json for shared-field comparisons and reference_manifest.json for '
        'per-array byte hashes and provenance. Automated test results are recorded alongside '
        'this bundle. Existing output directories are refused. Deterministic ZIP timestamps, '
        'member order and compression are fixed; regenerate with the same Python/NumPy/zlib '
        'environment for the same archive hash. Physical gates remain FAIL; no training approval.\n')
    return manifest


@dataclass(frozen=True)
class ReferenceWindow:
    indices: np.ndarray
    valid: np.ndarray
    arrays: dict


class ReferenceClip:
    """Read-only access to the processed artifact, with explicit terminal-window masks."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.manifest = json.loads((self.path.parent / 'reference_manifest.json').read_text())
        if self.manifest['kind'] != 'minimal_hoi_motion_reference':
            raise ValueError('Not a processed Part B reference manifest')
        if sha256(self.path) != self.manifest['sha256']:
            raise ValueError('Processed reference hash mismatch')
        with np.load(self.path, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        validate_arrays(arrays)
        for k, a in arrays.items():
            if array_info(a) != self.manifest['fields'][k]:
                raise ValueError(f'Array manifest mismatch: {k}')
            a.flags.writeable = False
        self.arrays = MappingProxyType(arrays)
        self.n_frames = len(arrays['joint_pos'])
        self.fps = float(arrays['fps'][0])

    def _check_index(self, index: int) -> None:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.n_frames:
            raise IndexError(f'Frame outside [0, {self.n_frames - 1}]: {index}')

    def frame(self, index: int) -> dict:
        self._check_index(index)
        return {k: self.arrays[k][index] for k in TIME_FIELDS}

    def future_window(self, start: int, offsets) -> ReferenceWindow:
        """Edge-pad for inspection; valid=False padding must never represent transitions."""
        self._check_index(start)
        offsets = np.asarray(offsets)
        if offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or np.any(offsets < 0):
            raise ValueError('Offsets must be a 1D nonnegative integer array')
        # Avoid integer overflow even for huge unsigned offsets.
        valid = offsets <= self.n_frames - 1 - start
        indices = start + np.minimum(offsets, self.n_frames - 1 - start).astype(np.int64)
        return ReferenceWindow(indices, valid, {k: self.arrays[k][indices] for k in TIME_FIELDS})
