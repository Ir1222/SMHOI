"""P6-only bounded diagnostic adapter. Production provenance gates are untouched."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'scripts/hoi'))
from supermimic.hoi.reference import DEFAULT_ASSET, DEFAULT_OUTPUT, ReferenceClip, sha256
from supermimic.hoi.asset_manifest import load_manifest
from supermimic.hoi import scene

ASSET_SHA = '1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1'
REF_SHA = 'bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b'
MISSING_SHA = '168465ae0a2e17153902a337963574942551909680d0d8dd36f7ffb22102b05c'


def save(path, value):
    def convert(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n')


def inputs():
    if sha256(DEFAULT_ASSET) != ASSET_SHA:
        raise ValueError('Changed accepted asset')
    m = load_manifest(DEFAULT_ASSET, verify_sources=False)
    sources = []
    for s in m['sources']:
        p = Path(s['path'])
        actual = sha256(p) if p.is_file() else None
        if actual != s['sha256'] and not (
            actual is None and p == ROOT/'SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md'
            and s['sha256'] == MISSING_SHA
        ):
            raise ValueError(f'Unapproved missing/changed source {p}')
        sources.append({**s, 'actual_sha256': actual})
    clip = ReferenceClip(DEFAULT_OUTPUT/'reference.npz')
    if sha256(clip.path) != REF_SHA:
        raise ValueError('Changed original Part B')
    return m, clip, sources


def physical_env(num_envs=4, noise=0.):
    m, _, sources = inputs()
    def diagnostic_load(path):
        if Path(path).resolve() != DEFAULT_ASSET.resolve() or sha256(path) != ASSET_SHA:
            raise ValueError('Diagnostic adapter only accepts pinned original manifest')
        return m
    with patch.object(scene, 'load_manifest', diagnostic_load):
        env = scene.make_physical_env(scene.PhysicalSceneCfg(num_envs=num_envs, action_noise_std=noise))
    return env, m, sources


def rot(q):
    return Rotation.from_quat(q, scalar_first=True)


def pair_category(model, g1, g2, box_prefix='box/'):
    names = [model.geom(int(g)).name for g in (g1, g2)]
    box = [n.startswith(box_prefix) for n in names]
    if sum(box) != 1:
        return None
    other = int(g2 if box[0] else g1)
    name = model.geom(other).name
    body = model.body(int(model.geom_bodyid[other])).name
    for side in ('left', 'right'):
        if body in (f'robot/{side}_wrist_pitch_link', f'robot/{side}_wrist_yaw_link'):
            return side
    if 'support' in name:
        return 'support'
    if 'terrain' in name or name == 'ground':
        return 'ground'
    return 'robot_body'


def cpu_contacts(model, data, sample):
    import mujoco
    result = []
    for i in range(data.ncon):
        c = data.contact[i]
        category = pair_category(model, c.geom1, c.geom2)
        if category is None:
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f)
        result.append({'sample': sample, 'category': category, 'dist': float(c.dist),
                       'pos': c.pos.copy(), 'frame': c.frame.reshape(3, 3).copy(),
                       'geom1': int(c.geom1), 'geom2': int(c.geom2),
                       'names': [model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name],
                       'local_force_N': f[:3], 'world_force_on_geom2_N': c.frame.reshape(3, 3).T@f[:3]})
    return result
