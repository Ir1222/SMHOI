"""Real-source Part B conversion checks; no training or original data writes."""
import json

import numpy as np
import pytest

from supermimic.hoi.reference import (
    DEFAULT_ASSET, DEFAULT_HOI, DEFAULT_ROBOT, FIELDS, OBJECT_SOURCE,
    ROBOT_FIELDS, ReferenceClip, array_info, convert, sha256, validate_arrays,
)


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp('b1') / 'reference'
    convert(DEFAULT_ROBOT, DEFAULT_HOI, DEFAULT_ASSET, out)
    return out


@pytest.mark.parametrize('field', ROBOT_FIELDS)
def test_canonical_exact(bundle, field):
    with np.load(DEFAULT_ROBOT, allow_pickle=False) as r, np.load(bundle / 'reference.npz') as z:
        assert array_info(z[field]) == array_info(r[field])
        assert np.array_equal(z[field], r[field])


@pytest.mark.parametrize('field', OBJECT_SOURCE)
def test_hoi_exact(bundle, field):
    with np.load(DEFAULT_HOI, allow_pickle=False) as h, np.load(bundle / 'reference.npz') as z:
        assert array_info(z[field]) == array_info(h[OBJECT_SOURCE[field]])


def test_contract(bundle):
    clip = ReferenceClip(bundle / 'reference.npz')
    assert set(clip.arrays) == set(FIELDS)
    assert clip.n_frames == 840 and clip.fps == 50
    assert np.flatnonzero(clip.arrays['object_contact']).tolist() == list(range(260, 550))
    assert all(v < 1e-5 for v in validate_arrays(clip.arrays)['max_quaternion_norm_error'].values())
    report = json.loads((bundle / 'validation.json').read_text())
    assert all(v['value_equal'] for v in report['shared_robot_fields'].values())
    assert not report['shared_robot_fields']['joint_names']['dtype_equal']
    assert not json.loads((bundle / 'experiment_profile.json').read_text())['training_ready']


def test_deterministic_and_immutable(bundle, tmp_path):
    paths = [DEFAULT_ROBOT, DEFAULT_HOI, DEFAULT_ASSET]
    before = [sha256(p) for p in paths]
    convert(*paths, tmp_path / 'second')
    for name in ['reference.npz', 'reference_manifest.json', 'validation.json',
                 'VALIDATION.md', 'experiment_profile.json']:
        assert (bundle / name).read_bytes() == (tmp_path / 'second' / name).read_bytes()
    assert before == [sha256(p) for p in paths]
    with pytest.raises(FileExistsError):
        convert(*paths, bundle)


def test_indexing_and_future_window(bundle):
    c = ReferenceClip(bundle / 'reference.npz')
    for i in [0, 260, 549, 550, 839]:
        assert np.array_equal(c.frame(i)['joint_pos'], c.arrays['joint_pos'][i])
    for i in [-1, 840, 1.5]:
        with pytest.raises(IndexError):
            c.frame(i)
    w = c.future_window(838, [0, 1, 2, 20])
    assert w.indices.tolist() == [838, 839, 839, 839]
    assert w.valid.tolist() == [True, True, False, False]
    assert np.array_equal(w.arrays['object_pos_w'], c.arrays['object_pos_w'][w.indices])
    assert c.future_window(839, np.array([2**64-1], dtype=np.uint64)).indices[0] == 839
    for offsets in [[-1], [0.5], [[0, 1]]]:
        with pytest.raises(ValueError):
            c.future_window(0, offsets)
    with pytest.raises(ValueError):
        c.arrays['joint_pos'][0, 0] = 1


@pytest.mark.parametrize('field', ['joint_pos', 'joint_names', 'fps'])
def test_shared_mismatch_rejected(tmp_path, field):
    with np.load(DEFAULT_HOI) as z:
        a = {k: z[k] for k in z.files}
    a[field].flat[0] = 'wrong_joint' if field == 'joint_names' else 123
    source = tmp_path / 'altered_fixture.npz'
    np.savez(source, **a)
    with pytest.raises(ValueError, match='Shared robot field mismatch'):
        convert(DEFAULT_ROBOT, source, DEFAULT_ASSET, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('failure', ['quaternion', 'nonfinite', 'frames', 'fps'])
def test_invalid_contract_rejected(bundle, failure):
    c = ReferenceClip(bundle / 'reference.npz')
    a = {k: v.copy() for k, v in c.arrays.items()}
    if failure == 'quaternion':
        a['object_quat_w'][0] = 0
    elif failure == 'nonfinite':
        a['object_pos_w'][0, 0] = np.nan
    elif failure == 'frames':
        a['joint_pos'] = a['joint_pos'][:-1]
    else:
        a['fps'][0] = 60
    with pytest.raises(ValueError):
        validate_arrays(a)
