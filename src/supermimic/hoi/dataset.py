"""Complete measured archives -> episode partitions -> original P2 Replay storage."""

import json
from pathlib import Path

import numpy as np
import torch

from supermimic.hoi.contact import aggregate_interval
from supermimic.hoi.reference import DEFAULT_ASSET, sha256
from supermimic.hoi.replay import FORCE_SHAPES, HOIReplayBuffer


def load_archive(directory, *, reference_hashes, references=None):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("acceptance") != "PASS":
        raise ValueError("Failed or incomplete collection cannot be training evidence")
    path = directory / "transitions.npz"
    if metadata.get("archive_sha256") != sha256(path):
        raise ValueError("Archive content hash missing or mismatched")
    contract = metadata["replay"]
    expected = {
        "schema": "hoi-transition-v1",
        "force_sign": "hand_on_box",
        "force_frame": "contemporaneous_box_at_substep_solve",
        "force_unit": "N",
        "physics_dt_s": 0.005,
        "decimation": 4,
        "control_dt_s": 0.020,
        "asset_manifest_sha256": sha256(DEFAULT_ASSET),
        "hand_order": ["left", "right"],
    }
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("Archive physical schema/frame/timing/asset mismatch")
    if [r["sha256"] for r in contract["references"]] != list(reference_hashes):
        raise ValueError("Archive reference identity mismatch; aliases do not create new motion")
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    n = len(data["state"])
    shapes = {
        "state": (84,),
        "next_state": (84,),
        "action_clean": (29,),
        "action_exec": (29,),
        "xi": (33,),
        "intent_ref": (2,),
        **FORCE_SHAPES,
    }
    masks = ("physical_failure", "timeout", "clip_end", "transition_terminal", "transition_valid")
    identities = ("env", "episode", "clip", "frame")
    for name, shape in {**shapes, **{k: () for k in masks + identities}}.items():
        if name not in data or data[name].shape != (n, *shape) or not np.isfinite(data[name]).all():
            raise ValueError(f"Missing or invalid measured archive field: {name}")
    if (
        not n
        or any(data[k].dtype != bool for k in masks)
        or any(data[k].dtype.kind not in "iu" for k in identities)
    ):
        raise ValueError("Invalid archive count/mask/identity dtype")
    if not data["transition_valid"].all() or any((data[k] < 0).any() for k in identities):
        raise ValueError("Invalid archive transitions/identity")
    if data["clip"].max() >= len(reference_hashes):
        raise ValueError("Unknown archive clip")
    if not np.isin(data["intent_ref"], [0, 1]).all():
        raise ValueError("Reference contact intent must be binary")
    if references is not None:
        validate_reference_intent(data, metadata, references)
    if not np.array_equal(
        data["transition_terminal"], data["physical_failure"] | data["timeout"] | data["clip_end"]
    ):
        raise ValueError("Terminal reason mismatch")
    rebuilt = aggregate_interval(
        torch.tensor(data["force_raw_world"]),
        torch.tensor(data["box_quat_solve"]),
        torch.tensor(data["found_substeps"]),
        force_on=contract["force_on_N"],
    )
    for name in FORCE_SHAPES:
        if not np.allclose(rebuilt[name].numpy(), data[name], atol=2e-5, rtol=2e-6):
            raise ValueError(f"Saved physical label does not match raw substeps: {name}")
    return data, metadata


def validate_reference_intent(data, metadata, references):
    """Verify the existing global pair source; never infer or repair labels."""
    if [c.manifest["sha256"] for c in references] != [
        r["sha256"] for r in metadata["replay"]["references"]
    ]:
        raise ValueError("Contact reference identity mismatch")
    pairs = data["intent_ref"]
    if pairs.shape != (len(data["frame"]), 2) or not np.isin(pairs, [0, 1]).all():
        raise ValueError("Reference contact intent must be binary pairs")
    for clip_id in np.unique(data["clip"]):
        if not 0 <= clip_id < len(references):
            raise ValueError("Unknown contact clip")
        selected = data["clip"] == clip_id
        frame = data["frame"][selected]
        intent = references[int(clip_id)].arrays["object_contact"]
        if ((frame < 0) | (frame + 1 >= len(intent))).any():
            raise ValueError("Contact frame outside reference; no clamping")
        expected = np.stack((intent[frame], intent[frame + 1]), -1)
        if not np.array_equal(pairs[selected], expected):
            raise ValueError("Contact intent does not match canonical reference")


def episode_keys(data, metadata):
    refs = metadata["replay"]["references"]
    return [
        (refs[int(c)]["sha256"], int(e), int(ep))
        for c, e, ep in zip(data["clip"], data["env"], data["episode"])
    ]


def partition_episodes(data, metadata, validation_keys):
    """Explicit whole episodes; caller records whether source clips are also held out."""
    keys = episode_keys(data, metadata)
    validation_keys = set(validation_keys)
    if not validation_keys or not validation_keys <= set(keys):
        raise ValueError("Validation episode identities are absent or empty")
    valid = np.array([k in validation_keys for k in keys])
    if valid.all():
        raise ValueError("No independent training episodes remain")
    return np.flatnonzero(~valid), np.flatnonzero(valid)


def replay_from_archive(data, metadata, indices, *, device="cpu"):
    """Populate existing Replay. Never synthesize xi, actions, forces or next states."""
    keys = episode_keys(data, metadata)
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate archive row selection")
    selected_keys = sorted({keys[i] for i in indices})
    groups = [np.array([i for i in indices if keys[i] == key]) for key in selected_keys]
    if any(
        set(group) != {i for i, key in enumerate(keys) if key == selected_keys[e]}
        for e, group in enumerate(groups)
    ):
        raise ValueError("Partition must include whole source episodes")
    if not groups:
        raise ValueError("Empty dataset partition")
    # Retain original per-episode order; append validates continuity and xi.
    capacity = max(9, max(map(len, groups)) + 2)
    replay = HOIReplayBuffer(
        len(groups), capacity, device, force_on=metadata["replay"]["force_on_N"]
    )
    replay.metadata.update(metadata["replay"])
    replay.metadata["archive_episode_keys"] = selected_keys
    for e, group in enumerate(groups):
        rows = group[np.argsort(data["frame"][group], kind="stable")]
        if data["transition_terminal"][rows[:-1]].any():
            raise ValueError("Terminal transition inside archive episode")
        ids = torch.tensor([e], device=device)

        def value(name, row):
            return torch.as_tensor(data[name][row : row + 1], device=device)

        for row in rows:
            replay.append(
                ids,
                state=value("state", row),
                next_state=value("next_state", row),
                action_clean=value("action_clean", row),
                action_exec=value("action_exec", row),
                xi=value("xi", row),
                frame=value("frame", row).long(),
                clip=value("clip", row).long(),
                labels={k: value(k, row) for k in FORCE_SHAPES},
                intent_ref=value("intent_ref", row),
                physical_failure=value("physical_failure", row),
                timeout=value("timeout", row),
                clip_end=value("clip_end", row),
            )
        last = int(rows[-1])
        if not data["transition_terminal"][last]:
            # External collection stop: target-only row, no fabricated done/force.
            z = torch.zeros(1, 29, device=device)
            b = torch.zeros(1, dtype=torch.bool, device=device)
            replay.add(
                ids,
                value("next_state", last),
                z,
                z,
                value("xi", last),
                torch.zeros(1, device=device),
                torch.zeros(1, device=device),
                value("frame", last).long() + 1,
                value("clip", last).long(),
                ~b,
                b,
            )
            replay.end_episode(ids)
    return replay
