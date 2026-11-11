"""Bounded P3 diagnostics: immutable reference and complete measured Replay snapshots."""

import json
from pathlib import Path

import numpy as np

from supermimic.hoi.reference import REPO, ReferenceClip, sha256

REFERENCE = REPO / "outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz"
REFERENCE_SHA256 = "f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7"


def reference():
    clip = ReferenceClip(REFERENCE)
    if clip.manifest["sha256"] != REFERENCE_SHA256:
        raise ValueError("P3 requires the explicitly accepted refined reference hash")
    return clip


def snapshot(buffer, ids, pointers):
    rows = []
    for e, t in zip(ids.tolist(), pointers.tolist()):
        row = {k: v[e, t].cpu().numpy().copy() for k, v in buffer.fields.items()}
        for name, field in [
            ("state", "s"),
            ("action_clean", "a"),
            ("action_exec", "a_exec"),
            ("xi", "xi"),
            ("frame", "frame"),
            ("clip", "clip"),
            ("episode", "ep_id"),
        ]:
            row[name] = getattr(buffer, field)[e, t].cpu().numpy().copy()
        row["env"] = np.asarray(e)
        rows.append(row)
    return rows


def save_rows(output, rows, metadata):
    if rows:
        np.savez_compressed(
            Path(output) / "transitions.npz", **{k: np.stack([r[k] for r in rows]) for k in rows[0]}
        )
        metadata["archive_sha256"] = sha256(Path(output) / "transitions.npz")
    (Path(output) / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
