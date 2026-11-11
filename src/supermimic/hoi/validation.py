"""Physical-unit prediction metrics and a fail-closed evidence-based readiness gate."""

import math
from dataclasses import dataclass, field

import torch


def _mean_or_none(values):
    return float(values.mean()) if values.numel() else None


def force_metrics(prediction, target, occupancy, stats):
    """Per-hand calibration is thresholded mean prediction versus measured occupancy.

    They need not coincide when substep net forces cancel; report that distinction.
    Missing directional/event strata return None, never a successful zero error.
    """
    output = {}
    for h, hand in enumerate(("left", "right")):
        p, t = prediction[..., h, :].reshape(-1, 3), target[..., h, :].reshape(-1, 3)
        active = occupancy[..., h].reshape(-1) > 0
        pn, tn = p.norm(dim=-1), t.norm(dim=-1)
        predicted_active = pn > stats.force_on_N
        high = tn > stats.median_active_N[h]
        output[hand] = {}
        for name, selected in {
            "all": torch.ones_like(active),
            "active": active,
            "inactive": ~active,
            "low": active & ~high,
            "high": active & high,
        }.items():
            directional = selected & (tn > stats.force_on_N)
            cos = (p * t).sum(-1) / (pn * tn).clamp_min(1e-12)
            output[hand][name] = {
                "count": int(selected.sum()),
                "mae_N": _mean_or_none((p - t).abs().mean(-1)[selected]),
                "magnitude_mae_N": _mean_or_none((pn - tn).abs()[selected]),
                "direction_cosine": _mean_or_none(cos[directional]),
                "direction_count": int(directional.sum()),
            }
        output[hand]["calibration"] = {
            "actual_occupancy_active_rate": float(active.float().mean()),
            "predicted_mean_active_rate": float(predicted_active.float().mean()),
            "active_recall": _mean_or_none(predicted_active[active].float()),
            "inactive_false_positive_rate": _mean_or_none(predicted_active[~active].float()),
        }
    return output


def transition_events(predictions, occupancy, stats):
    """Within-window adjacency only; overlapping windows may repeat physical events."""
    p = predictions.norm(dim=-1) > stats.force_on_N
    t = occupancy > 0
    result = {}
    for h, hand in enumerate(("left", "right")):
        result[hand] = {}
        for name, truth, predicted in (
            ("onset", ~t[:-1, :, h] & t[1:, :, h], ~p[:-1, :, h] & p[1:, :, h]),
            ("loss", t[:-1, :, h] & ~t[1:, :, h], p[:-1, :, h] & ~p[1:, :, h]),
        ):
            result[hand][name] = {
                "true_count_with_window_repeats": int(truth.sum()),
                "predicted_count_with_window_repeats": int(predicted.sum()),
                "recall": _mean_or_none(predicted[truth].float()),
                "false_event_rate": _mean_or_none(predicted[~truth].float()),
            }
    return result


def state_metrics(wm, prediction, target, occupancy):
    robot = wm.spec.robot.error_vector(prediction[..., :71], target[..., :71]).square().mean(-1)
    obj = wm.spec.object_error_vector(prediction, target)
    physical = obj * wm.spec.object_err_scale
    result = {}
    active = occupancy.any(-1)
    for name, selected in {
        "all": torch.ones_like(active, dtype=torch.bool),
        "contact": active,
        "noncontact": ~active,
    }.items():
        result[name] = {
            "count": int(selected.sum()),
            "robot_nse": _mean_or_none(robot[selected]),
            "object_nse": _mean_or_none(obj.square().mean(-1)[selected]),
        }
        for i, key in enumerate(
            ("position_RMSE_m", "rotation_RMSE_rad", "linear_RMSE_m_s", "angular_RMSE_rad_s")
        ):
            e = physical[..., i * 3 : i * 3 + 3][selected]
            result[name][key] = float(e.square().sum(-1).mean().sqrt()) if e.numel() else None
    return result


@torch.no_grad()
def evaluate_windows(wm, replay, *, k=8):
    e, t = replay.window_mask(k).nonzero(as_tuple=True)
    if not len(e):
        raise ValueError("No valid episode-contained evaluation windows")
    batch = replay._gather_windows(e, t, k)
    state = batch["states"][0]
    predictions = []
    forces = []
    for step in range(k):
        state, force = wm.predict(state, batch["a_exec"][step], batch["xi"])
        predictions.append(state)
        forces.append(force)
    predictions, forces = torch.stack(predictions), torch.stack(forces)
    if not torch.isfinite(predictions).all() or not torch.isfinite(forces).all():
        raise ValueError("Nonfinite learned predictions")
    result = {
        "eligible_windows": len(e),
        "horizon": k,
        "evaluation_semantics": "each eligible start exactly once; horizons evaluated separately; event adjacency can repeat across windows",
    }
    for step in (0, k - 1):
        result[f"horizon_{step + 1}"] = {
            "force": force_metrics(
                forces[step],
                batch["force_mean_box"][step],
                batch["occupancy"][step],
                wm.force_statistics,
            ),
            "state": state_metrics(
                wm, predictions[step], batch["states"][step + 1], batch["occupancy"][step] > 0
            ),
        }
    result["events"] = transition_events(forces, batch["occupancy"], wm.force_statistics)
    return result, {
        "state_prediction": predictions.cpu(),
        "force_prediction_N": forces.cpu(),
        "target_state": batch["states"][1:].cpu(),
        "target_force_N": batch["force_mean_box"].cpu(),
        "occupancy": batch["occupancy"].cpu(),
        "start_env": e.cpu(),
        "start_row": t.cpu(),
    }


def unique_coverage(data, indices, stats):
    """Dataset rows, not sampled windows: unique class/event counts for readiness."""
    selected = torch.as_tensor(indices)
    force = torch.as_tensor(data["force_mean_box"])[selected]
    occ = torch.as_tensor(data["occupancy"])[selected] > 0
    counts = {}
    for h, hand in enumerate(("left", "right")):
        high = force[:, h].norm(dim=-1) > stats.median_active_N[h].cpu()
        counts.update(
            {
                f"{hand}.active": int(occ[:, h].sum()),
                f"{hand}.inactive": int((~occ[:, h]).sum()),
                f"{hand}.low": int((occ[:, h] & ~high).sum()),
                f"{hand}.high": int((occ[:, h] & high).sum()),
                f"{hand}.onset": 0,
                f"{hand}.loss": 0,
            }
        )
    # Adjacent physical rows must share env/episode/clip and consecutive frames.
    lookup = {
        (
            int(data["env"][i]),
            int(data["episode"][i]),
            int(data["clip"][i]),
            int(data["frame"][i]),
        ): i
        for i in indices
    }
    for key, i in lookup.items():
        j = lookup.get((*key[:3], key[3] + 1))
        if j is None or data["transition_terminal"][i]:
            continue
        a, b = data["occupancy"][i] > 0, data["occupancy"][j] > 0
        for h, hand in enumerate(("left", "right")):
            counts[f"{hand}.onset"] += int(not a[h] and b[h])
            counts[f"{hand}.loss"] += int(a[h] and not b[h])
    return counts


@dataclass
class ReadinessCriteria:
    # No research accuracy thresholds are invented by default.
    maxima: dict[str, float] = field(default_factory=dict)
    minima: dict[str, float] = field(default_factory=dict)
    minimum_counts: dict[str, int] = field(default_factory=dict)


def readiness_gate(metrics, *, coverage, provenance, trained_updates, criteria=None):
    reasons = []
    if trained_updates <= 0:
        reasons.append("No learned update evidence")
    if not provenance.get("natural_HOI_training_data", False):
        reasons.append("Controlled diagnostics do not establish natural HOI generalization")
    for key in ("episode_disjoint", "reference_verified", "simulator_labels_verified"):
        if not provenance.get(key, False):
            reasons.append(f"Missing verified provenance: {key}")
    expected_counts = {
        f"{hand}.{kind}"
        for hand in ("left", "right")
        for kind in ("active", "inactive", "low", "high", "onset", "loss")
    }
    for name in sorted(expected_counts):
        if coverage.get(name, 0) <= 0:
            reasons.append(f"Missing validation coverage: {name}")
    required_max = {
        f"horizon_{k}.force.{h}.all.{metric}"
        for k in (1, 8)
        for h in ("left", "right")
        for metric in ("mae_N", "magnitude_mae_N")
    }
    required_max |= {
        f"horizon_{k}.state.all.{metric}"
        for k in (1, 8)
        for metric in (
            "robot_nse",
            "object_nse",
            "position_RMSE_m",
            "rotation_RMSE_rad",
            "linear_RMSE_m_s",
            "angular_RMSE_rad_s",
        )
    }
    required_max |= {
        f"horizon_{k}.force.{h}.calibration.inactive_false_positive_rate"
        for k in (1, 8)
        for h in ("left", "right")
    }
    required_min = {
        f"horizon_{k}.force.{h}.all.direction_cosine" for k in (1, 8) for h in ("left", "right")
    }
    required_min |= {
        f"horizon_{k}.force.{h}.calibration.active_recall"
        for k in (1, 8)
        for h in ("left", "right")
    }
    required_min |= {
        f"events.{h}.{event}.recall" for h in ("left", "right") for event in ("onset", "loss")
    }
    for path in sorted(required_max | required_min):
        value = metrics
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            reasons.append(f"Missing finite validation metric: {path}")
    if (
        criteria is None
        or not required_max <= criteria.maxima.keys()
        or not required_min <= criteria.minima.keys()
        or not expected_counts <= criteria.minimum_counts.keys()
    ):
        reasons.append("Validation thresholds/count requirements have not been fully specified")
    if criteria is not None:
        for direction, limits in [("max", criteria.maxima), ("min", criteria.minima)]:
            for path, limit in limits.items():
                value = metrics
                for part in path.split("."):
                    value = value.get(part) if isinstance(value, dict) else None
                if (
                    value is None
                    or not math.isfinite(value)
                    or not math.isfinite(limit)
                    or (value > limit if direction == "max" else value < limit)
                ):
                    reasons.append(f"Unmet {direction} criterion: {path}")
        for name, minimum in criteria.minimum_counts.items():
            if minimum < 1 or coverage.get(name, 0) < minimum:
                reasons.append(f"Insufficient unique validation count: {name}")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "actor_force_optimization_enabled": False,
        "scope": "Predictive readiness only; P6 gradient fidelity remains separately required",
    }
