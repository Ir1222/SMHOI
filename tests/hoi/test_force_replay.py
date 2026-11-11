import pytest
import torch

from supermimic.hoi.contact import aggregate_interval
from supermimic.hoi.replay import HOIReplayBuffer


def put(b, t, *, end=None, clip=0, ids=None):
    ids = torch.arange(b.E) if ids is None else ids
    n = len(ids)
    raw = torch.full((n, 4, 2, 2, 3), float(t))
    q = torch.zeros(n, 4, 4)
    q[..., 0] = 1
    labels = aggregate_interval(raw, q, torch.ones(n, 4, 2, 2), force_on=1e-3)
    masks = {
        name: torch.full((n,), end == name, dtype=torch.bool)
        for name in ["physical_failure", "timeout", "clip_end"]
    }
    b.append(
        ids,
        state=torch.full((n, 84), float(t)),
        next_state=torch.full((n, 84), float(t + 1)),
        action_clean=torch.full((n, 29), float(t)),
        action_exec=torch.full((n, 29), t + 0.5),
        xi=torch.zeros(n, 33),
        frame=torch.full((n,), t, dtype=torch.long),
        clip=torch.full((n,), clip, dtype=torch.long),
        labels=labels,
        intent_ref=torch.zeros(n, 2),
        **masks,
    )


@pytest.mark.parametrize("end", ["physical_failure", "timeout", "clip_end"])
def test_transition_done_not_successor_and_all_fields(end):
    b = HOIReplayBuffer(2, 32, "cpu", force_on=1e-3)
    for t in range(8):
        put(b, t, end=end if t == 7 else None)
    w = b.sample_windows(128)
    assert w["states"].shape == (9, 128, 84)
    assert w["transition_terminal"][:-1].sum() == 0
    assert w["transition_terminal"][-1].all()
    assert w[end][-1].all()
    assert w["transition_valid"].all()
    assert (w["a_exec"][..., 0] == w["frame"] + 0.5).all()
    assert (w["force_mean_box"][..., 0, 0] == 2 * w["frame"]).all()
    assert torch.equal(w["states"][1:], w["next_state"])
    assert (w["episode_id"] == 0).all()
    two = b.sample_windows(256, 2)
    last = two["frame"][0] == 6
    assert last.any()
    assert two["transition_terminal"][:, last][:, 0].tolist() == [False, True]
    assert b.size == 18
    assert not b.fields["transition_valid"][:, 8].any()


def test_wrap_reset_and_clip_boundaries():
    b = HOIReplayBuffer(2, 13, "cpu", force_on=1e-3)
    for t in range(18):
        put(b, t)
    w = b.sample_windows(100, 8)
    assert (w["frame"][1:] - w["frame"][:-1] == 1).all()
    assert w["frame"].min() >= 5 and w["frame"].max() <= 16
    b.end_episode(torch.tensor([0]))
    for t in range(9):
        put(b, t, clip=1, ids=torch.tensor([0]))
    w = b.sample_windows(300, 8)
    assert (w["clip"] == w["clip"][0]).all()
    assert (w["env_id"] == w["env_id"][0]).all()
    assert (w["episode_id"] == w["episode_id"][0]).all()
    assert set(w["clip"][0].tolist()) == {0, 1}


@pytest.mark.parametrize("field", ["clip", "frame", "valid", "terminal"])
def test_masks_fail_closed_on_invalid_rows(field):
    b = HOIReplayBuffer(1, 20, "cpu", force_on=1e-3)
    for t in range(9):
        put(b, t)
    assert b.window_mask().sum() == 1
    if field == "clip":
        b.clip[0, 4] = 1
    if field == "frame":
        b.frame[0, 4] += 1
    if field == "valid":
        b.fields["transition_valid"][0, 4] = False
    if field == "terminal":
        b.fields["transition_terminal"][0, 4] = True
    assert b.window_mask().sum() == 0
    with pytest.raises(RuntimeError):
        b.sample_windows(1)


def test_discontinuity_rejected_before_mutation_and_short_episode():
    b = HOIReplayBuffer(1, 20, "cpu", force_on=1e-3)
    put(b, 0)
    with pytest.raises(ValueError):
        put(b, 2)
    assert b.size == 1 and b.ptr.item() == 1
    put(b, 1, end="timeout")
    assert not b.window_mask().any()
    w = b.sample_windows(1, 2)
    assert w["timeout"][:, 0].tolist() == [False, True]
    # Retained samples cannot alias reused ring storage.
    before = w["force_raw_world"].clone()
    b.fields["force_raw_world"].zero_()
    assert torch.equal(before, w["force_raw_world"])
