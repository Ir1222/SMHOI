"""Display contract, independent transformations and frame-clock tests."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from supermimic.hoi.reference import DEFAULT_ASSET, DEFAULT_OUTPUT, ReferenceClip
from supermimic.hoi.reference_viewer import (
    KEY_COMMANDS,
    Playback,
    ReferenceScene,
    inspect_all_frames,
    points_world,
)


@pytest.fixture(scope='module')
def scene():
    return ReferenceScene(ReferenceClip(DEFAULT_OUTPUT / 'reference.npz'), display_only=True)


def test_display_only_and_asset_hash(tmp_path):
    c = ReferenceClip(DEFAULT_OUTPUT / 'reference.npz')
    with pytest.raises(ValueError, match='display_only'):
        ReferenceScene(c)
    altered = tmp_path / 'asset.json'
    altered.write_text(DEFAULT_ASSET.read_text() + '\n')
    with pytest.raises(ValueError, match='Asset manifest differs'):
        ReferenceScene(c, altered, display_only=True)


def test_static_and_compound_geometry(scene):
    m = scene.model
    assert not scene.asset['training_ready']
    assert m.nu == m.neq == m.nmocap == 0
    assert m.nq == 43 and m.nv == 41
    box = m.body(scene.asset['object']['name']).id
    assert m.body_geomnum[box] == 7
    for g in scene.asset['object']['geoms']:
        gid = m.geom(g['name']).id
        np.testing.assert_array_equal(m.geom_size[gid], g['size'])
        np.testing.assert_array_equal(m.geom_pos[gid], g['pos'])
        np.testing.assert_array_equal(m.geom_quat[gid], g['quat'])
    for s in scene.asset['supports']:
        assert m.body(s['name']).jntnum == 0
        gid = m.geom(s['geom']['name']).id
        np.testing.assert_array_equal(m.geom_size[gid], s['geom']['size'])


def test_points_world_known_rotation():
    q = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    np.testing.assert_allclose(points_world([1, 2, 3], q, [[.05, 0, 0], [0, .17, .14]]),
                               [[1, 2.05, 3], [.83, 2, 3.14]], atol=1e-15)


def test_no_stepping_and_all_frame_transforms(scene, monkeypatch):
    import mujoco
    def forbidden(*args, **kwargs):
        raise AssertionError('Physics stepping is forbidden')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    monkeypatch.setattr(mujoco, 'mj_step1', forbidden)
    monkeypatch.setattr(mujoco, 'mj_step2', forbidden)
    rows, report = inspect_all_frames(scene)
    assert len(rows) == 840 and report['status'] == 'PASS'
    assert sum(r['desired_contact'] for r in rows) == 290
    assert [r['time_s'] for r in rows] == [i/50 for i in range(840)]
    assert scene.data.time == 839/50
    assert not scene.data.qfrc_applied.any() and not scene.data.xfrc_applied.any()


def test_pose_assignment_and_reference_markers(scene):
    for i in [0, 260, 298, 400, 530, 550, 839]:
        scene.set_frame(i)
        f = scene.clip.frame(i)
        np.testing.assert_array_equal(scene.data.qpos[scene.joint_adrs], f['joint_pos'])
        np.testing.assert_array_equal(scene.data.qpos[scene.root_adr:scene.root_adr+3],
                                      f['body_pos_w'][scene.root_index])
        b = scene.box_adr
        np.testing.assert_array_equal(scene.data.qpos[b:b+7],
                                      np.r_[f['object_pos_w'], f['object_quat_w']])
        # Independent scalar rotation matrices, not the viewer helper.
        for side, body_index in enumerate(scene.wrist_indices):
            mat = Rotation.from_quat(f['body_quat_w'][body_index], scalar_first=True).as_matrix()
            expected = f['body_pos_w'][body_index] + mat @ np.array(
                scene.asset['contact']['eef_offset_wrist_m'][side])
            np.testing.assert_allclose(scene.eef_reference[side], expected, atol=1e-12)
        import mujoco
        overlay = mujoco.MjvScene(scene.model, maxgeom=10)
        overlay.ngeom = 0
        scene.add_lines(overlay)
        for side in range(2):
            np.testing.assert_allclose(overlay.geoms[2*side+1].pos, scene.eef_reference[side], atol=1e-7)
        assert ('ON' if f['object_contact'] else 'OFF') in scene.status_text()


def test_clock_controls_and_no_terminal_wrap():
    p = Playback(840, 50)
    p.advance(1)
    assert p.frame == 0
    p.command('toggle')
    for _ in range(60):
        p.advance(1/60)
    assert p.frame == 50
    p.command('faster')
    p.advance(.5)
    assert p.frame == 100 and p.speed == 2
    p.command('slower')
    p.command('previous')
    p.advance(100)
    assert p.frame == 99 and p.paused
    p.command('next')
    assert p.frame == 100
    p.command('toggle')
    p.advance(100)
    assert p.frame == 839 and p.paused
    p.command('next')
    assert p.frame == 839
    p.command('reset')
    p.command('previous')
    assert p.frame == 0 and p.paused
    assert KEY_COMMANDS[262] == 'next' and KEY_COMMANDS[263] == 'previous'
    for _ in range(10):
        p.command('slower')
    assert p.speed == .125
    for _ in range(10):
        p.command('faster')
    assert p.speed == 8
    for bad in [-1, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            p.advance(bad)


@pytest.mark.parametrize('speed', [.125, .5, 1, 2, 8])
def test_exact_full_clip_clock(speed):
    p = Playback(840, 50, paused=False, speed=speed)
    p.advance(16.78 / speed)
    assert p.frame == 839 and p.paused


def test_viewer_never_parses_source_npz(monkeypatch):
    original_load = np.load
    expected = (DEFAULT_OUTPUT / 'reference.npz').resolve()
    paths = []
    def guarded_load(path, *args, **kwargs):
        from pathlib import Path
        assert Path(path).resolve() == expected
        paths.append(Path(path))
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded_load)
    c = ReferenceClip(expected)
    s = ReferenceScene(c, display_only=True)
    s.set_frame(839)
    assert len(paths) == 1
