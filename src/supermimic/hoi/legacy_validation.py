"""Historical force-gate interpretation only; never an active WM acceptance gate."""

import math
from dataclasses import dataclass, field


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
