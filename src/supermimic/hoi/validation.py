"""Physical dynamics and global reference-intent diagnostics (no force prediction)."""

import math
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from supermimic.hoi.world_model import augment_states


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def binary_metrics(probability, target, logits=None):
    """Global intent metrics with explicit denominators and exact deterministic baselines."""
    if probability.shape != target.shape or (logits is not None and logits.shape != target.shape):
        raise ValueError("Misaligned binary metrics")
    p, t = probability.double().flatten(), target.double().flatten()
    if not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not ((t == 0) | (t == 1)).all():
        raise ValueError("Invalid intent probability/target")
    predicted, positive = p >= 0.5, t.bool()
    tp, tn = int((predicted & positive).sum()), int((~predicted & ~positive).sum())
    fp, fn = int((predicted & ~positive).sum()), int((~predicted & positive).sum())
    n = t.numel()
    if logits is not None:
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite contact logits")
        loss = F.binary_cross_entropy_with_logits(logits.double().flatten(), t, reduction="none")
    else:
        loss = -torch.xlogy(t, p) - torch.xlogy(1 - t, 1 - p)
    infinite = int((~torch.isfinite(loss)).sum())
    return {
        "count": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _ratio(tp + tn, n),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "denominators": {
            "accuracy": n,
            "precision": tp + fp,
            "recall": tp + fn,
            "f1": 2 * tp + fp + fn,
        },
        "target_on": tp + fn,
        "target_off": tn + fp,
        "mean_probability": float(p.mean()) if n else None,
        "brier": float((p - t).square().mean()) if n else None,
        "bce": float(loss.mean()) if n and not infinite else None,
        "bce_infinite_count": infinite,
        "bce_status": "infinite" if infinite else ("finite" if n else "undefined_empty"),
    }


def intent_events(probabilities, pairs, identities=None):
    """Include c0; never join windows/resets. Timing uses nearest same-kind event."""
    if probabilities.ndim != 2 or pairs.shape != (*probabilities.shape, 2):
        raise ValueError("Events require [K,N] probabilities and [K,N,2] pairs")
    if (
        not len(probabilities)
        or not torch.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
        or not ((pairs == 0) | (pairs == 1)).all()
    ):
        raise ValueError("Invalid event probabilities or binary reference pairs")
    if not torch.equal(pairs[:-1, :, 1], pairs[1:, :, 0]):
        raise ValueError("Discontinuous event targets")
    initial = pairs[0, :, 0].bool()[None]
    truth = torch.cat((initial, pairs[..., 1].bool()), 0)
    predicted = torch.cat((initial, probabilities >= 0.5), 0)
    result = {}
    for name, a, b in (
        ("onset", ~truth[:-1] & truth[1:], ~predicted[:-1] & predicted[1:]),
        ("release", truth[:-1] & ~truth[1:], predicted[:-1] & ~predicted[1:]),
    ):
        count, detected, hits = int(a.sum()), int(b.sum()), int((a & b).sum())
        timing = []
        for window in range(probabilities.shape[1]):
            candidates = b[:, window].nonzero().flatten()
            for step in a[:, window].nonzero().flatten().tolist():
                nearest = (
                    int(candidates[(candidates - step).abs().argmin()])
                    if candidates.numel()
                    else None
                )
                timing.append(
                    {
                        "window": window,
                        "true_step": step + 1,
                        "nearest_predicted_step": nearest + 1 if nearest is not None else None,
                        "error_steps": nearest - step if nearest is not None else None,
                    }
                )
        errors = [x["error_steps"] for x in timing if x["error_steps"] is not None]
        record = {
            "true_count": count,
            "predicted_count": detected,
            "exact_hits": hits,
            "misses": count - hits,
            "false_events": detected - hits,
            "precision": _ratio(hits, detected),
            "recall": _ratio(hits, count),
            "denominators": {"precision": detected, "recall": count, "timing": len(errors)},
            "mean_signed_error_steps": sum(errors) / len(errors) if errors else None,
            "mean_absolute_error_steps": sum(map(abs, errors)) / len(errors) if errors else None,
            "timing": timing,
            "counts_include_overlapping_window_occurrences": True,
        }
        if identities is not None:

            def unique(mask):
                return len({identities[j][i] for j, i in mask.nonzero().tolist()})

            record.update(
                unique_true_count=unique(a),
                unique_predicted_count=unique(b),
                unique_exact_hit_count=unique(a & b),
                unique_false_event_count=unique(b & ~a),
            )
        result[name] = record
    return result


def state_metrics(wm, prediction, target, intent):
    prediction, target = prediction[..., :84], target[..., :84]
    robot = wm.spec.robot.error_vector(prediction[..., :71], target[..., :71]).square().mean(-1)
    obj = wm.spec.object_error_vector(prediction, target)
    physical = obj * wm.spec.object_err_scale
    result = {}
    for name, selected in {
        "all": torch.ones_like(intent, dtype=torch.bool),
        "reference_intent_on": intent.bool(),
        "reference_intent_off": ~intent.bool(),
    }.items():
        count = int(selected.sum())
        record = {
            "count": count,
            "robot_nse": float(robot[selected].mean()) if count else None,
            "object_nse": float(obj[selected].square().mean()) if count else None,
        }
        for i, key in enumerate(
            ("position_RMSE_m", "rotation_RMSE_rad", "linear_RMSE_m_s", "angular_RMSE_rad_s")
        ):
            e = physical[..., i * 3 : i * 3 + 3][selected]
            record[key] = float(e.square().sum(-1).mean().sqrt()) if count else None
        result[name] = record
    return result


def _identities(replay, batch):
    result = []
    for j in range(len(batch["frame"])):
        row = []
        for i in range(batch["frame"].shape[1]):
            e = int(batch["env_id"][j, i])
            if "archive_episode_keys" in replay.metadata:
                key = tuple(replay.metadata["archive_episode_keys"][e])
            else:
                c = int(batch["clip"][j, i])
                key = (
                    replay.metadata["references"][c]["sha256"],
                    e,
                    int(batch["episode_id"][j, i]),
                )
            row.append((*key, int(batch["frame"][j, i])))
        result.append(row)
    return result


@torch.no_grad()
def evaluate_windows(wm, replay, *, k=8, constant_velocity=False):
    """All one-step transitions and every complete AR window; no teacher forcing."""

    def capture(horizon):
        e, t = replay.window_mask(horizon).nonzero(as_tuple=True)
        if not len(e):
            raise ValueError(f"No valid episode-contained {horizon}-step windows")
        batch = replay._gather_windows(e, t, horizon)
        state = augment_states(batch["states"], batch["intent_ref"])[0]
        predictions, probabilities, logits = [], [], []
        for action in batch["a_exec"]:
            if constant_velocity:
                physical = wm.spec.integrate(state[:, :84], torch.zeros_like(state[:, :82]))
                state = torch.cat((physical, state[:, 84:85]), -1)
                logit = None
            else:
                state, logit = wm.predict(state, action, batch["xi"])
                logits.append(logit[:, 0])
            predictions.append(state[:, :84])
            probabilities.append(state[:, 84])
        pred, prob = torch.stack(predictions), torch.stack(probabilities)
        if not torch.isfinite(pred).all() or not torch.isfinite(prob).all():
            raise ValueError("Nonfinite learned predictions")
        return batch, pred, prob, torch.stack(logits) if logits else None

    one, ps1, pc1, l1 = capture(1)
    batch, ps, pc, logits = capture(k)
    targets = batch["intent_ref"][..., 1]

    def summary(b, physical, prob, logit, step):
        return {
            "state": state_metrics(
                wm, physical[step], b["states"][step + 1], b["intent_ref"][step, :, 1]
            ),
            "contact": binary_metrics(
                prob[step], b["intent_ref"][step, :, 1], logit[step] if logit is not None else None
            ),
        }

    identities = _identities(replay, batch)
    one_identities = _identities(replay, one)
    baselines = {}
    for name in ("always_off", "always_on", "persistence"):
        if name == "persistence":
            p1 = one["intent_ref"][0, :, 0]
            p8 = batch["intent_ref"][0, :, 0][None].expand_as(targets)
        else:
            value = float(name == "always_on")
            p1 = torch.full_like(pc1[0], value)
            p8 = torch.full_like(pc, value)
        baselines[name] = {
            "one_step": binary_metrics(p1, one["intent_ref"][0, :, 1]),
            "one_step_events": intent_events(p1[None], one["intent_ref"], one_identities),
            "horizons": {str(j + 1): binary_metrics(p8[j], targets[j]) for j in range(k)},
            "sequence": binary_metrics(p8, targets),
            "events": intent_events(p8, batch["intent_ref"], identities),
        }
    result = {
        "one_step_rows": int(pc1.numel()),
        "eligible_windows": pc.shape[1],
        "horizon": k,
        "scope": "one-step: all valid rows; AR: every full episode-contained window",
        "physical_baseline": "constant_velocity_zero_residual" if constant_velocity else "model",
        "horizon_1": summary(one, ps1, pc1, l1, 0),
        "one_step_events": intent_events(pc1, one["intent_ref"], one_identities),
        f"horizon_{k}": summary(batch, ps, pc, logits, k - 1),
        "autoregressive": {
            "horizons": {str(j + 1): summary(batch, ps, pc, logits, j) for j in range(k)},
            "sequence_contact": binary_metrics(pc, targets, logits),
            "events": intent_events(pc, batch["intent_ref"], identities),
        },
        "contact_baselines": baselines,
    }
    arrays = {
        "one_step_state_prediction": ps1[0],
        "one_step_target_state": one["states"][1],
        "one_step_contact_probability": pc1[0],
        "one_step_contact_target": one["intent_ref"][0, :, 1],
        "one_step_frame": one["frame"][0],
        "one_step_env": one["env_id"][0],
        "ar_state_prediction": ps,
        "ar_target_state": batch["states"][1:],
        "ar_contact_probability": pc,
        "ar_contact_target": targets,
        "ar_initial_contact": batch["intent_ref"][0, :, 0],
        "ar_frame": batch["frame"],
        "ar_env": batch["env_id"],
        "ar_episode": batch["episode_id"],
        "ar_clip": batch["clip"],
    }
    if logits is not None:
        arrays.update(one_step_contact_logit=l1[0], ar_contact_logit=logits)
    return result, {name: value.cpu() for name, value in arrays.items()}


def unique_coverage(data, indices, metadata):
    refs = metadata["replay"]["references"]
    keys = {
        (
            refs[int(data["clip"][i])]["sha256"],
            int(data["env"][i]),
            int(data["episode"][i]),
            int(data["frame"][i]),
        )
        for i in indices
    }
    if len(keys) != len(indices):
        raise ValueError("Duplicate source transition identity")
    pairs = torch.as_tensor(data["intent_ref"][indices])
    counts = {
        f"pair_{a}{b}": int(((pairs[:, 0] == a) & (pairs[:, 1] == b)).sum())
        for a, b in ((0, 0), (0, 1), (1, 1), (1, 0))
    }
    return {
        **counts,
        "rows": len(indices),
        "target_on": int((pairs[:, 1] == 1).sum()),
        "target_off": int((pairs[:, 1] == 0).sum()),
        "onset": counts["pair_01"],
        "release": counts["pair_10"],
    }


def readiness_gate(
    metrics, *, coverage, provenance, trained_updates, baseline=None, parameter_updates=None
):
    """Approved bounded diagnostic criteria, not a natural-HOI or Actor readiness claim."""
    missing = []
    if trained_updates <= 0:
        missing.append("No learned update evidence")
    for name in ("episode_disjoint", "reference_verified", "simulator_labels_verified"):
        if not provenance.get(name, False):
            missing.append(f"Missing verified provenance: {name}")
    a_missing, b_missing, a_failed, b_failed = list(missing), list(missing), [], []
    for key in ("target_on", "target_off", "onset", "release"):
        if coverage.get(key, 0) <= 0:
            b_missing.append(f"Missing validation coverage: {key}")
    updates = parameter_updates or {}
    for name in ("robot_head", "object_head", "contact_head"):
        group_missing, group_failed = (
            (b_missing, b_failed) if name == "contact_head" else (a_missing, a_failed)
        )
        value = updates.get(name)
        if value is None or not math.isfinite(value):
            group_missing.append(f"Missing parameter-update evidence: {name}")
        elif value <= 0:
            group_failed.append(f"No parameter update: {name}")

    def get(obj, *path):
        for key in path:
            obj = obj.get(key) if isinstance(obj, dict) else None
        return obj

    comparisons = {}
    for horizon in (1, 8):
        key = f"horizon_{horizon}"
        for body in ("robot", "object"):
            value, initial = [
                get(x, key, "state", "all", f"{body}_nse") for x in (metrics, baseline)
            ]
            label = f"{key}.{body}_nse"
            if (
                value is None
                or initial is None
                or not math.isfinite(value)
                or not math.isfinite(initial)
            ):
                a_missing.append(f"Missing finite physical comparison: {label}")
            else:
                limit = initial + 2e-6 + 2e-6 * abs(initial)
                comparisons[label] = {"learned": value, "frozen": initial, "maximum": limit}
                if value > limit:
                    a_failed.append(f"Worse than frozen initialized model: {label}")
    for metric in ("accuracy", "f1"):
        value = get(metrics, "horizon_1", "contact", metric)
        persist = get(metrics, "contact_baselines", "persistence", "one_step", metric)
        if (
            value is None
            or persist is None
            or not math.isfinite(value)
            or not math.isfinite(persist)
        ):
            b_missing.append(f"Missing finite contact metric/baseline: {metric}")
        elif value <= persist:
            b_failed.append(f"One-step {metric} does not exceed persistence")
    for name in ("onset", "release"):
        event = get(metrics, "autoregressive", "events", name)
        if not event or event.get("true_count", 0) <= 0:
            b_missing.append(f"No evaluated {name}")
        elif event["exact_hits"] != event["true_count"] or event["false_events"] != 0:
            b_failed.append(f"Inexact/missed/false autoregressive {name}")
    endpoint = get(metrics, "horizon_8", "contact")
    if not endpoint or not endpoint.get("target_on") or not endpoint.get("target_off"):
        b_missing.append("Eighth-step endpoints do not cover both classes")
    elif endpoint["accuracy"] != 1.0:
        b_failed.append("Not all eighth-step endpoints correct")

    def gate(absent, failed):
        return {
            "status": "PARTIAL" if absent else ("FAIL" if failed else "PASS"),
            "missing": absent,
            "failed": failed,
        }

    a, b = gate(a_missing, a_failed), gate(b_missing, b_failed)
    status = (
        "PARTIAL"
        if "PARTIAL" in (a["status"], b["status"])
        else ("FAIL" if "FAIL" in (a["status"], b["status"]) else "PASS")
    )
    return {
        "gate_a": {**a, "comparisons": comparisons},
        "gate_b": b,
        "overall": status,
        "ready": status == "PASS",
        "scope": "bounded single-reference diagnostic only",
        "cross_clip_generalization": False,
        "actor_force_optimization_enabled": False,
    }


# Historical force readiness semantics retained verbatim for evidence inspection only.
@dataclass
class ReadinessCriteria:
    # No research accuracy thresholds are invented by default.
    maxima: dict[str, float] = field(default_factory=dict)
    minima: dict[str, float] = field(default_factory=dict)
    minimum_counts: dict[str, int] = field(default_factory=dict)


def legacy_force_readiness_gate(metrics, *, coverage, provenance, trained_updates, criteria=None):
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
