"""Read-only preflight inventory. No collection, fitting, asset or reference writes."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from supermimic.hoi.contact_validation import actual_contact
from supermimic.hoi.provenance import research_authority
from supermimic.hoi.reference import DEFAULT_ASSET, DEFAULT_OUTPUT, REPO, sha256
from supermimic.hoi.training_scene import CALIBRATION, SINGLE_BODY, profile_manifest


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def distribution(values):
    x = np.asarray(values).reshape(-1)
    x = x[np.isfinite(x)]
    return (
        dict(
            zip(
                ("min", "p25", "median", "p75", "max"),
                np.percentile(x, [0, 25, 50, 75, 100]).tolist(),
            )
        )
        if len(x)
        else None
    )


def summary(data):
    state = np.asarray(data["state"])
    reached = np.asarray(data["next_state"])
    n = len(state)
    finite = np.isfinite(state).all(-1) & np.isfinite(reached).all(-1)
    action_key = next((k for k in ("action_exec", "action", "action_clean") if k in data), None)
    if action_key:
        finite &= np.isfinite(data[action_key]).all(-1)
    valid = np.asarray(data.get("transition_valid", np.ones(n, dtype=bool))).astype(bool)
    result = {
        "rows": n,
        "finite_state_action_rows": int(finite.sum()),
        "declared_valid_rows": int(valid.sum()) if "transition_valid" in data else None,
        "validity_scope": "saved mask plus finite state/action; no geometric acceptance inferred"
        if "transition_valid" in data
        else "no saved validity mask; finite values do not establish physical admission",
        "invalid_rows": int((~finite | ~valid).sum()),
        "fields": sorted(data),
        "action_field": action_key,
        "missing_archive_fields": sorted(
            {"xi", "action_exec", "action_clean", "frame", "episode", "clip", "env"} - set(data)
        ),
        "box_linear_speed_m_s": distribution(np.linalg.norm(reached[:, 78:81], axis=-1)),
        "box_angular_speed_rad_s": distribution(np.linalg.norm(reached[:, 81:84], axis=-1)),
        "robot_root_speed_m_s": distribution(np.linalg.norm(reached[:, 36:39], axis=-1)),
        "box_step_displacement_m": distribution(
            np.linalg.norm(reached[:, 71:74] - state[:, 71:74], axis=-1)
        ),
        "action_distribution": distribution(data[action_key]) if action_key else None,
    }
    for k in ("physical_failure", "timeout", "clip_end", "transition_terminal"):
        result[k] = int(np.asarray(data[k]).sum()) if k in data else None
    if "force_raw_world" in data:
        calibration = read(CALIBRATION)
        raw = np.asarray(data["force_raw_world"])
        force_valid = np.isfinite(raw).all(axis=(1, 2, 3, 4)) & finite & valid
        contact = actual_contact(
            raw[force_valid], calibration["thresholds_N_LR"], calibration["occupancy_cutoffs_LR"]
        )
        left, right = contact["hands"].T
        result["measured_contact_rows"] = {
            "none": int((~left & ~right).sum()),
            "left_only": int((left & ~right).sum()),
            "right_only": int((~left & right).sum()),
            "bilateral": int((left & right).sum()),
        }
        result["force_peak_N"] = float(contact["magnitude_N"].max()) if force_valid.any() else None
    if "intent_ref" in data:
        result["reference_on_reached"] = int(np.asarray(data["intent_ref"])[:, 1].sum())
    if all(k in data for k in ("env", "episode", "clip", "frame")):
        groups = np.stack([data[k] for k in ("env", "episode", "clip")], axis=1)
        episodes = []
        terminal = np.asarray(data.get("transition_terminal", np.zeros(n, dtype=bool)))
        for group in np.unique(groups, axis=0):
            idx = np.flatnonzero((groups == group).all(-1))
            idx = idx[np.argsort(data["frame"][idx], kind="stable")]
            frames = data["frame"][idx]
            jumps = np.diff(frames) != 1
            windows = 0
            for j in range(max(len(idx) - 7, 0)):
                w = idx[j : j + 8]
                windows += int(
                    (valid[w] & finite[w]).all()
                    and not terminal[w[:-1]].any()
                    and np.all(np.diff(data["frame"][w]) == 1)
                )
            link_error = (
                float(np.abs(state[idx[1:]] - reached[idx[:-1]]).max()) if len(idx) > 1 else 0.0
            )
            episodes.append(
                {
                    "env_episode_clip": group.tolist(),
                    "rows": len(idx),
                    "first_frame": int(frames[0]),
                    "last_frame": int(frames[-1]),
                    "frame_discontinuities": int(jumps.sum()),
                    "state_link_max_abs_error": link_error,
                    "ends_with_terminal": bool(terminal[idx[-1]]),
                    "eight_transition_windows": windows,
                }
            )
        result["source_episodes"] = episodes
    # Nonzero displacement/contact does not establish hand-supported lifting or carrying.
    result["verified_lift_carry_regimes"] = (
        "not established by these fields; consult independent support/load-path evidence"
    )
    return result


def audit_npz(path):
    metadata = read(path.parent / "metadata.json", {})
    diagnostic = read(path.parent / "diagnostics.json", {})
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    record = {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "keys": sorted(data),
        "historical_status": metadata.get("acceptance", diagnostic.get("acceptance")),
        "declared_natural_HOI_training_data": metadata.get("natural_HOI_training_data"),
        "declared_kind": metadata.get("kind", diagnostic.get("fixture")),
    }
    if "state" in data and "next_state" in data:
        record["transition_summary"] = summary(data)
        record["classification"] = "controlled diagnostic archive; not admitted as natural P7 data"
    else:
        record["classification"] = (
            "reference, prediction or partial diagnostic; not a complete measured transition archive"
        )
    record["metadata"] = metadata
    return record


def audit_intervals(path):
    rows = read(path)
    initial = read(path.parent / "initial.json", {})
    result = read(path.parent / "result.json", {})
    record = {
        "path": str(path),
        "sha256": sha256(path),
        "historical_result": result,
        "reference_start": initial.get("reference_start"),
        "external_forces": initial.get("external_forces"),
        "classification": "controller-driven reference replay"
        if path.parent.name.startswith("reference_f")
        else "controlled physical diagnostic",
    }
    if "integrated_original_02" in str(path) or "integrated_candidate_02" in str(path):
        record["physical_profile_evidence"] = {
            "box": "single_body (both arms of C4 comparison)",
            "box_sha256": sha256(SINGLE_BODY),
            "reference": "original"
            if "integrated_original_02" in str(path)
            else "reference_candidate_v1",
            "source": "docs/hoi/force_contact/p6/closure/C4_INTEGRATED_VALIDATION.md",
        }
    if not rows:
        return record
    data = {k: np.concatenate([np.asarray(row[k]) for row in rows]) for k in rows[0]}
    if "state" not in data or "next_state" not in data:
        record["keys"] = list(data)
        return record
    worlds = len(rows[0]["state"])
    # Explicit local source identity for audit only, not an exported/training archive.
    data["env"] = np.tile(np.arange(worlds), len(rows))
    data["episode"] = np.zeros(len(rows) * worlds, dtype=np.int64)
    data["clip"] = np.zeros(len(rows) * worlds, dtype=np.int64)
    start = initial.get("reference_start")
    data["frame"] = np.repeat(np.arange(len(rows)) + (0 if start is None else start), worlds)
    record["audit_identity_origin"] = "file SHA256 + world; no inferred xi/executed action; " + (
        "unbound diagnostic ordinal, NOT a reference frame"
        if start is None
        else "local sequence index + saved reference_start"
    )
    record["transition_summary"] = summary(data)
    samples = path.parent / "samples.json"
    if samples.exists():
        record["samples_sha256"] = sha256(samples)
        records = read(samples)
        record["substep_nonzero_overflow_world_samples"] = sum(
            int(np.count_nonzero(s["overflow"])) for s in records
        )
    record["training_admission"] = (
        "pending complete archive/provenance conversion and profile approval; reference disagreement alone is not rejection"
    )
    return record


def audit_jsonl(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    record = {
        "path": str(path),
        "sha256": sha256(path),
        "vector_steps": len(rows),
        "classification": "bounded P7 readiness policy diagnostic; not an admitted dataset",
    }
    if not rows:
        record["classification"] = "empty failed-run archive; retained"
        return record
    vector_keys = [k for k, v in rows[0].items() if np.asarray(v).ndim > 0]
    data = {k: np.concatenate([np.asarray(row[k]) for row in rows]) for k in vector_keys}
    record["non_vector_step_summaries_not_per_world"] = {
        k: [row[k] for row in rows] for k in rows[0] if k not in vector_keys
    }
    if "intent" in data:
        data["intent_ref"] = data.pop("intent")
    # Raw files do not persist source-episode/clip ids. Count reset discontinuities
    # for inspection without pretending these are complete training metadata.
    frames = np.asarray([row["frame"] for row in rows])
    record["frame_discontinuities_per_world"] = (np.diff(frames, axis=0) != 1).sum(0).tolist()
    record["transition_summary"] = summary(data)
    record["training_admission"] = (
        "pending complete archive/provenance conversion; experimental reference remains unapproved"
    )
    return record


def audit_checkpoint(path):
    record = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (pickle.UnpicklingError, RuntimeError, OSError, ValueError, TypeError) as error:
        record["classification"] = "not safely readable as tensor-only physical Replay"
        record["read_error"] = str(error).splitlines()[0]
        return record
    record["schema"] = payload.get("schema", payload.get("metadata", {}).get("schema"))
    if payload.get("schema") != "hoi-training-lifecycle-v1":
        record["classification"] = (
            "weights, sampled window or Robot checkpoint; not full HOI Replay"
        )
        record["keys"] = sorted(payload)
        return record
    record["identity"] = payload["identity"]
    record["optimizer_steps"] = payload["optimizer_steps"]
    replay, fields = payload["runtime"]["replay"], payload["runtime"]["replay_fields"]
    mask = replay["valid"] & fields["transition_valid"]
    record["occupied_rows"] = int(replay["valid"].sum())
    record["rejected_occupied_rows"] = int((replay["valid"] & ~fields["transition_valid"]).sum())
    env, row = mask.nonzero(as_tuple=True)
    data = {k: v[env, row].numpy() for k, v in fields.items()}
    data.update(
        {
            k: replay[v][env, row].numpy()
            for k, v in {
                "state": "s",
                "action_clean": "a",
                "action_exec": "a_exec",
                "frame": "frame",
                "clip": "clip",
                "episode": "ep_id",
                "xi": "xi",
            }.items()
        }
    )
    data["env"] = env.numpy()
    record["classification"] = "bounded integration physical Replay; not a readiness dataset"
    if len(env):
        record["transition_summary"] = summary(data)
        record["transition_content_sha256"] = hashlib.sha256(
            b"".join(data[k].tobytes() for k in sorted(data))
        ).hexdigest()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = REPO / "outputs/hoi"
    # Read repository archives and every existing temporary checkpoint from the
    # accepted integration runs/tests. Symlink aliases are deduplicated by path.
    npz = [audit_npz(p) for p in sorted(outputs.rglob("*.npz"))]
    intervals = [
        audit_intervals(p)
        for p in sorted((outputs / "contact_supervision").rglob("intervals.json"))
        if "p7_preflight" not in p.parts  # Inventory reports are not raw physical intervals.
    ]
    jsonl = [audit_jsonl(p) for p in sorted(outputs.rglob("transitions.jsonl"))]
    checkpoints = set(outputs.rglob("*.pt"))
    for root in list(Path("/tmp").glob("smhoi*")) + [Path("/tmp/pytest-of-han")]:
        if root.is_dir():
            checkpoints.update(p.resolve() for p in root.rglob("*.pt"))
    checkpoints = [audit_checkpoint(p) for p in sorted(checkpoints)]
    profiles = {}
    for name in ("p6-original", "p6-single-body"):
        manifest = profile_manifest(name)
        profiles[name] = {
            "inertia": manifest["object"]["inertia"],
            "geoms": manifest["object"]["geoms"],
            "supports": manifest["supports"],
            "contact": manifest["contact"],
            "training_ready": manifest["training_ready"],
        }
    for name, value in (
        ("archives.json", npz),
        ("intervals.json", intervals),
        ("policy_jsonl.json", jsonl),
        ("checkpoints.json", checkpoints),
        ("profiles.json", profiles),
    ):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "research_authority": research_authority(),
                "asset_sha256": sha256(DEFAULT_ASSET),
                "single_body_sha256": sha256(SINGLE_BODY),
                "original_reference_sha256": sha256(DEFAULT_OUTPUT / "reference.npz"),
                "archive_count": len(npz),
                "interval_file_count": len(intervals),
                "policy_jsonl_file_count": len(jsonl),
                "checkpoint_count": len(checkpoints),
                "scope": "repository outputs/hoi plus surviving /tmp/smhoi* and /tmp/pytest-of-han checkpoints; no external dataset assumed",
                "decision": "INSUFFICIENT natural HOI coverage; no train/validation partition or training admission performed",
                "original_reference_modified": False,
                "optimizer_updates": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
