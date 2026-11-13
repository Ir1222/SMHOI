from dataclasses import replace

import pytest
import torch
from p5_fixtures import fixture, start_batch
from test_policy_adapter import fill_replay

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.hoi.imagination import HOIPolicyReplayBuffer
from supermimic.hoi.policy_features import (
    CRITIC_SLICES,
    DIMENSIONS,
    OBJECT_HISTORY_SLICES,
    HOIObservationAdapter,
    critic_features,
    feature_schema,
    flatten_object_history,
    flatten_robot_history,
    object_future_goal,
    object_observation,
    object_reference_error,
    object_scale,
)
from supermimic.hoi.reference import ReferenceClip, sha256
from supermimic.hoi.reward_reference import ClipBinding, HOIReferenceRegistry
from supermimic.kinematics.rotations import (
    heading_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
)


def test_named_shapes_and_group_orders():
    h = torch.arange(2 * 10 * 15).reshape(2, 10, 15)
    flat = flatten_object_history(h)
    assert flat.shape == (2, 150)
    offset = 0
    for name, width in (
        ("position", 3),
        ("rotation6", 6),
        ("linear_heading", 3),
        ("angular_body", 3),
    ):
        assert torch.equal(
            flat[:, OBJECT_HISTORY_SLICES[name]], h[:, :, offset : offset + width].flatten(1)
        )
        offset += width
    robot = torch.arange(2 * 10 * 93).reshape(2, 10, 93)
    out = flatten_robot_history(robot)
    assert out.shape == (2, 930)
    assert torch.equal(out[:, :290], robot[:, :, :29].flatten(1))
    assert torch.equal(out[:, 640:], robot[:, :, 64:].flatten(1))
    assert DIMENSIONS["decoder"] == 1058 and DIMENSIONS["critic"] == 1015
    assert [s.stop - s.start for s in CRITIC_SLICES.values()] == [86, 18, 15, 29, 680, 150, 33, 4]
    assert feature_schema()["excluded"] == [
        "measured_force",
        "actual_contact",
        "reference_intent",
        "contact_geometry",
    ]
    for bad in (torch.zeros(2, 9, 15), torch.zeros(10, 2, 15)):
        with pytest.raises(ValueError):
            flatten_object_history(bad)


def test_actual_heading_absolute_velocity_and_normalization_once():
    fx = fixture()
    s = start_batch(fx, (300,)).state.repeat(10, 1)
    s[:, :3] = 0
    s[:, 3:7] = quat_from_rotvec(
        torch.stack((torch.zeros(10), torch.zeros(10), torch.arange(10) * 0.1), -1).double()
    )
    s[:, 71:74] = torch.tensor([1.0, 2.0, 3.0])
    s[:, 78:81] = torch.tensor([3.0, 4.0, 5.0])
    s[:, 81:84] = torch.tensor([2.0, 4.0, 6.0])
    raw = object_observation(s, fx.wm.spec)
    h = heading_quat(s[:, 3:7])
    torch.testing.assert_close(raw[:, :3], quat_apply_inverse(h, s[:, 71:74]))
    torch.testing.assert_close(raw[:, 9:12], quat_apply_inverse(h, s[:, 78:81]))
    torch.testing.assert_close(raw[:, 12:], s[:, 81:84])
    torch.testing.assert_close(raw * object_scale().double(), fx.wm.spec.object_canonical(s))
    changed = s.clone()
    changed[:, 36:39] += 100
    assert torch.equal(raw, object_observation(changed, fx.wm.spec))
    history = flatten_object_history(raw[None])
    assert not torch.equal(history[:, :3], history[:, 27:30])
    # Re-expressing every frame with the last heading is observably wrong.
    wrong = quat_apply_inverse(h[-1:].expand_as(h), s[:, 71:74])
    assert not torch.allclose(raw[:, :3], wrong)


@pytest.mark.parametrize("angle", [0.0, 0.1, 1.7, 3.141592653589793, -3.141592653589793])
def test_quaternion_sign_errors_and_finite_orientation_gradients(angle):
    fx = fixture()
    st = start_batch(fx, (300,))
    ref = fx.registry.gather(st.clip_id, st.local_frame)
    s = st.state.clone()
    s[:, 71:74], s[:, 74:78], s[:, 78:81] = (
        ref.box.position,
        ref.box.quaternion,
        ref.box.linear_world,
    )
    s[:, 81:84] = quat_apply_inverse(quat_normalize(ref.box.quaternion), ref.box.angular_world)
    torch.testing.assert_close(
        object_reference_error(s, ref), torch.zeros(1, 15).double(), atol=1e-10, rtol=1e-8
    )
    rotation = quat_from_rotvec(s.new_tensor([[angle, 0.07, -0.03]]))
    s[:, 74:78] = quat_mul(rotation, s[:, 74:78])
    s.requires_grad_()
    flipped = s.detach().clone()
    flipped[:, 3:7] *= -1
    flipped[:, 74:78] *= -1
    torch.testing.assert_close(
        object_observation(s, fx.wm.spec), object_observation(flipped, fx.wm.spec)
    )
    error = object_reference_error(s, ref)
    torch.testing.assert_close(error, object_reference_error(flipped, ref))
    # Independent matrix expression verifies world-relative, reference-heading error6.
    h = matrix_from_quat(heading_quat(quat_normalize(ref.robot.root_quat_w)))
    ro, rr = (
        matrix_from_quat(quat_normalize(s[:, 74:78])),
        matrix_from_quat(quat_normalize(ref.box.quaternion)),
    )
    matrix = h.transpose(-1, -2) @ ro @ rr.transpose(-1, -2) @ h
    expected = (matrix - torch.eye(3).double())[:, :, :2].reshape(1, 6)
    torch.testing.assert_close(error[:, 3:9], expected, atol=1e-10, rtol=1e-8)
    g = torch.autograd.grad(error.square().sum() + object_observation(s, fx.wm.spec).sum(), s)[0]
    assert torch.isfinite(g).all() and g[:, 71:].norm() > 0


def test_rigid_yaw_translation_equivariance_and_world_body_omega():
    fx = fixture()
    st = start_batch(fx, (300,))
    s = st.state.clone()
    yaw = quat_from_rotvec(s.new_tensor([[0.0, 0.0, 0.7]]))
    transformed = s.clone()
    shift = s.new_tensor([[1.2, -0.4, 0.0]])
    for p in (slice(0, 3), slice(71, 74)):
        transformed[:, p] = quat_apply(yaw, s[:, p]) + shift
    for q in (slice(3, 7), slice(74, 78)):
        transformed[:, q] = quat_mul(yaw, s[:, q])
    for v in (slice(36, 39), slice(78, 81)):
        transformed[:, v] = quat_apply(yaw, s[:, v])
    torch.testing.assert_close(
        object_observation(s, fx.wm.spec),
        object_observation(transformed, fx.wm.spec),
        atol=1e-10,
        rtol=1e-8,
    )
    ref = fx.registry.gather(st.clip_id, st.local_frame)
    s[:, 81:84] = s.new_tensor([[0.3, -0.4, 0.5]])
    h = heading_quat(quat_normalize(ref.robot.root_quat_w))
    expected = quat_apply_inverse(
        h, quat_apply(quat_normalize(s[:, 74:78]), s[:, 81:84]) - ref.box.angular_world
    )
    torch.testing.assert_close(object_reference_error(s, ref)[:, 12:], expected)


def test_future_goals_multiclip_hashes_padding_and_no_npz_writes():
    fx = fixture()
    original = fx.clip
    candidate = ReferenceClip(
        "outputs/hoi/contact_supervision/p6_closure/reference_candidate_v1/reference.npz"
    )
    paths = [original.path, candidate.path]
    before = [sha256(p) for p in paths]
    assert before == [
        "bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b",
        "f38071e5347403f9eff993dd37884bf374e0c003400156dc4cb87d3d2b856cb0",
    ]
    lib = MotionLibrary(paths, fx.lib.body_names, fx.q0, device="cpu", dtype=torch.float64)
    registry = HOIReferenceRegistry(
        lib,
        [ClipBinding(7, 0, original), ClipBinding(42, 1, candidate)],
        library_sources=paths,
        dt=0.02,
        joint_names=fx.joint_names,
    )
    clips, frames = torch.tensor([7, 42, 42]), torch.tensor([300, 838, 839])
    goal = object_future_goal(registry, clips, frames, fx.wm.spec)
    assert goal.values.shape == (3, 150)
    assert goal.valid.sum(1).tolist() == [10, 2, 1]
    assert goal.local_frames[0].tolist() == list(range(300, 310))
    assert goal.local_frames[1].tolist() == [838] + [839] * 9
    assert goal.global_frames[1, 0] == 840 + 838
    for i in range(3):
        ref = registry.gather(clips[i : i + 1].expand(10), goal.local_frames[i])
        h = heading_quat(quat_normalize(ref.robot.root_quat_w))
        expected_pos = quat_apply_inverse(h, ref.box.position - ref.robot.root_pos_w)
        expected_omega = quat_apply_inverse(
            quat_normalize(ref.box.quaternion), ref.box.angular_world
        )
        torch.testing.assert_close(goal.values[i].reshape(10, 15)[:, :3], expected_pos)
        torch.testing.assert_close(goal.values[i].reshape(10, 15)[:, 12:], expected_omega)
    with pytest.raises(IndexError):
        object_future_goal(registry, clips[:1], torch.tensor([840]), fx.wm.spec)
    with pytest.raises(ValueError, match="Unknown"):
        object_future_goal(registry, torch.tensor([99]), frames[:1], fx.wm.spec)
    with pytest.raises(ValueError, match="SHA"):
        HOIReferenceRegistry(
            lib,
            [ClipBinding(7, 0, candidate), ClipBinding(42, 1, original)],
            library_sources=paths,
            dt=0.02,
            joint_names=fx.joint_names,
        )
    assert before == [sha256(p) for p in paths]
    assert candidate.manifest["training_ready"] is False


def test_replay_shared_rows_backfill_clip_episode_and_noise_parity():
    fx = fixture(torch.float32)
    buffer = HOIPolicyReplayBuffer(2, 32, "cpu", force_on=0.001, registry=fx.registry)
    fill_replay(buffer, fx)
    # Give every stored Object row an identifiable value without a second recorder.
    buffer.s[:, :, 71] = torch.arange(buffer.T)[None]
    st = buffer._gather_starts(torch.tensor([0]), torch.tensor([2]), 10)
    adapter = HOIObservationAdapter(fx.wm.spec, fx.q0)
    raw = adapter.history(st.hist_states, st.hist_actions, st.xi)
    assert raw.shape == (1, 1080)
    expected = torch.stack(
        [adapter.observation(s, a, st.xi) for s, a in zip(st.hist_states, st.hist_actions)]
    )
    assert torch.equal(raw, adapter.flatten_history(expected))
    assert st.hist_states[:8, 0, 71].tolist() == [0.0] * 8
    assert st.hist_states[-2:, 0, 71].tolist() == [1.0, 2.0]
    assert not st.hist_actions[:8].any()
    for field in (buffer.clip, buffer.ep_id):
        saved = field.clone()
        field[0, 1] += 100
        separated = buffer._gather_starts(torch.tensor([0]), torch.tensor([2]), 10)
        assert torch.equal(
            separated.hist_states, separated.state[None].expand_as(separated.hist_states)
        )
        assert not separated.hist_actions.any()
        field.copy_(saved)
    # Same inputs/RNG: collection-facing history and per-step imagination transform.
    torch.manual_seed(71)
    physical = adapter.history(st.hist_states, st.hist_actions, st.xi, noise=True)
    torch.manual_seed(71)
    imagined = adapter.flatten_history(
        torch.stack(
            [
                adapter.add_noise(adapter.observation(s, a, st.xi), True)
                for s, a in zip(st.hist_states, st.hist_actions)
            ]
        )
    )
    assert torch.equal(physical, imagined)
    assert torch.equal(physical[:, 930:], raw[:, 930:])
    assert not torch.equal(physical[:, :640], raw[:, :640])


def test_privileged_features_are_causal_live_and_independent_of_measurements():
    fx = fixture()
    st = start_batch(fx, (300,))
    ref = fx.registry.gather(st.clip_id, st.local_frame)
    state = st.state.clone().requires_grad_()
    feat = critic_features(state, st.prev_action, ref, fx.wm.spec, fx.wm.support)
    assert feat.shape == (1, 148)
    torch.testing.assert_close(feat[:, :86], fx.wm.spec.canonical(state))
    torch.testing.assert_close(feat[:, 104:119], object_reference_error(state, ref))
    changed_intent = replace(ref, box=replace(ref.box, contact=~ref.box.contact))
    assert torch.equal(
        feat, critic_features(state, st.prev_action, changed_intent, fx.wm.spec, fx.wm.support)
    )
    moved = st.state.clone()
    moved[:, :3] += moved.new_tensor([1.0, 0.0, 0.0])
    moved[:, 71:74] += moved.new_tensor([1.0, 0.0, 0.0])
    other = critic_features(moved, st.prev_action + 1, ref, fx.wm.spec, fx.wm.support)
    torch.testing.assert_close(other[:, :86], feat[:, :86])
    assert not torch.equal(other[:, 86:104], feat[:, 86:104])
    assert torch.equal(other[:, 119:], st.prev_action + 1)
    grad = torch.autograd.grad(feat.square().sum(), state)[0]
    assert torch.isfinite(grad).all() and grad[:, 71:].norm() > 0


@pytest.mark.parametrize("name", ["in_scale", "xi_offset", "xi_scale", "object_in_scale"])
def test_versioned_adapter_rejects_changed_canonical_or_context_normalization(name):
    fx = fixture()
    owner = fx.wm.spec if name == "object_in_scale" else fx.wm.spec.robot
    setattr(owner, name, getattr(owner, name) + 0.125)
    with pytest.raises(ValueError, match="scale mismatch"):
        HOIObservationAdapter(fx.wm.spec, fx.q0)
