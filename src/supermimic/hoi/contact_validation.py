"""P6 offline physical measurements; never imported by learning/reward paths."""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def actual_contact(raw_world, thresholds, occupancy_cutoffs):
    """Chronological [N,4,2,2,3] primary forces → independent hand decisions.

    Strict inequalities follow Master §11. No smoothing, reference labels, or
    absolute sum of opposing local loads. Cutoffs .25 require >=2 of 4 solves.
    """
    raw = np.asarray(raw_world)
    tau, rho = np.asarray(thresholds), np.asarray(occupancy_cutoffs)
    if raw.ndim != 5 or raw.shape[1:] != (4, 2, 2, 3):
        raise ValueError("Expected [N,4,2,2,3] raw force")
    if tau.shape != (2,) or rho.shape != (2,):
        raise ValueError("Independent L/R thresholds and cutoffs required")
    if (
        not np.isfinite(raw).all()
        or not np.isfinite(tau).all()
        or not np.isfinite(rho).all()
        or np.any(tau <= 0)
        or np.any(rho < 0)
        or np.any(rho >= 1)
    ):
        raise ValueError("Invalid physical contact measurement/definition")
    magnitude = np.linalg.norm(raw.sum(axis=3), axis=-1)
    occupancy = (magnitude > tau).mean(axis=1)
    hands = occupancy > rho
    return {
        "magnitude_N": magnitude,
        "occupancy": occupancy,
        "hands": hands,
        "any": hands.any(axis=1),
        "bilateral": hands.all(axis=1),
    }


def confusion(reference, measured):
    reference, measured = np.asarray(reference), np.asarray(measured)
    if reference.dtype != bool or measured.dtype != bool or reference.shape != measured.shape:
        raise ValueError("Expected equally shaped bool labels")
    tn = int((~reference & ~measured).sum())
    fp = int((~reference & measured).sum())
    fn = int((reference & ~measured).sum())
    tp = int((reference & measured).sum())

    def ratio(a, b):
        return a / b if b else None

    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {
        "matrix_ref_rows_sim_columns": [[tn, fp], [fn, tp]],
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "count": tn + fp + fn + tp,
        "accuracy": ratio(tp + tn, tn + fp + fn + tp),
        "precision": ratio(tp, tp + fp),
        "recall": recall,
        "F1": ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None,
    }


def agreement(raw, labels, frames_start, frames_end, episodes, valid, thresholds, cutoffs):
    """Compare interval force against exact reached frame t+1, one shared mask.

    Episodes identify distinct source/world episodes. Tracking failures do not
    remove otherwise finite, aligned, solver-valid measurements.
    """
    labels = np.asarray(labels)
    start, end, ep, mask = map(np.asarray, (frames_start, frames_end, episodes, valid))
    n = len(raw)
    if labels.dtype != bool or labels.ndim != 1 or mask.dtype != bool:
        raise ValueError("Bool reference and explicit bool validity required")
    if any(x.shape != (n,) for x in (start, end, ep, mask)):
        raise ValueError("Metadata lengths differ")
    if start.dtype.kind not in "iu" or end.dtype.kind not in "iu":
        raise ValueError("Integer frame indices required")
    if np.any(end != start + 1) or np.any(start < 0) or np.any(end >= len(labels)):
        raise ValueError("Misaligned or out-of-clip frame; never clamp")
    contact = actual_contact(raw, thresholds, cutoffs)
    ref = labels[end]
    result = {
        "total_rows": n,
        "valid_rows": int(mask.sum()),
        "invalid_rows": np.flatnonzero(~mask).tolist(),
        "alignment": "interval [t,t+1] compared to unchanged object_contact[t+1]",
        "any": confusion(ref[mask], contact["any"][mask]),
        "bilateral": confusion(ref[mask], contact["bilateral"][mask]),
    }
    for name, values in (
        ("left", contact["hands"][:, 0]),
        ("right", contact["hands"][:, 1]),
        ("any", contact["any"]),
        ("bilateral", contact["bilateral"]),
    ):
        record = result.setdefault(name, {})
        record["contact_ratio"] = float(values[mask].mean()) if mask.any() else None
        for phase, select in (("ON", mask & ref), ("OFF", mask & ~ref)):
            count = int(select.sum())
            record[phase] = {
                "frames": count,
                "duration_s": count * 0.02,
                "contact_duration_s": int(values[select].sum()) * 0.02,
                "contact_ratio": float(values[select].mean()) if count else None,
            }
        onsets = losses = adj = 0
        for episode in np.unique(ep):
            indices = np.flatnonzero(ep == episode)
            # Require recorded ordering, do not silently repair a scrambled episode.
            if np.any(np.diff(start[indices]) <= 0):
                raise ValueError("Duplicate/nonmonotonic episode frames")
            for i, j in pairwise(indices):
                if mask[i] and mask[j] and start[j] == end[i]:
                    adj += 1
                    onsets += int(not values[i] and values[j])
                    losses += int(values[i] and not values[j])
        record.update(
            onsets=onsets,
            losses=losses,
            adjacency_duration_s=adj * 0.02,
            onset_Hz=onsets / (adj * 0.02) if adj else None,
            loss_Hz=losses / (adj * 0.02) if adj else None,
        )
    for hand, name in enumerate(("left", "right")):
        for phase, select in (("ON", mask & ref), ("OFF", mask & ~ref)):
            occ = contact["occupancy"][select, hand]
            result[name][phase].update(
                mean_occupancy=float(occ.mean()) if len(occ) else None,
                force_bearing_substep_duration_s=float(occ.sum() * 0.02),
            )
    return result


def penetration_summary(contacts, sample_count, dt=0.005):
    """One time sample per solver evaluation; empty samples stay in denominator.

    Mean depth is over contact records including zero depth, not time-weighted.
    Duration counts unique solver samples, not manifold point multiplicity.
    """
    if sample_count < 0 or dt <= 0:
        raise ValueError("Invalid sampling interval/count")
    result = {}
    categories = ["left", "right", "hand_box", "support", "ground", "robot_body", "all_box"]
    for category in categories:
        selected = [
            c
            for c in contacts
            if category == "all_box"
            or c["category"] == category
            or (category == "hand_box" and c["category"] in ("left", "right"))
        ]
        distances = np.array([c["dist"] for c in selected], dtype=float)
        samples = np.array([c["sample"] for c in selected], dtype=int)
        if (
            not np.isfinite(distances).all()
            or np.any(samples < 0)
            or np.any(samples >= sample_count)
        ):
            raise ValueError("Invalid contact sample/distance")
        depths = np.maximum(-distances, 0)
        record = {
            "contact_records": len(selected),
            "contact_samples": len(np.unique(samples)),
            "min_signed_distance_m": float(distances.min()) if len(distances) else None,
            "max_penetration_m": float(depths.max()) if len(depths) else 0.0,
            "mean_depth_during_contact_m": float(depths.mean()) if len(depths) else None,
            "duration_s": sample_count * dt,
        }
        for limit in (0.0, 0.0001, 0.001, 0.005):
            active = np.zeros(sample_count, dtype=bool)
            active[samples[depths > limit]] = True
            padded = np.r_[False, active, False].astype(int)
            starts, ends = np.where(np.diff(padded) == 1)[0], np.where(np.diff(padded) == -1)[0]
            record[str(limit)] = {
                "samples": int(active.sum()),
                "duration_s": float(active.sum() * dt),
                "frequency_fraction": float(active.mean()) if sample_count else None,
                "episodes": len(starts),
                "max_contiguous_s": float((ends - starts).max() * dt) if len(starts) else 0.0,
            }
        result[category] = record
    return result
