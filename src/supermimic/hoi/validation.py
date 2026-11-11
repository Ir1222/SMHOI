"""Physical metric helpers for deferred P7-A; no classifier or readiness gate.

These utilities do not establish predictive readiness. P3 cleanup executes only
synthetic contract checks and source-to-migrated-model equivalence probes.
"""

import torch


def state_metrics(wm, prediction, target, intent):
    if prediction.shape != target.shape or prediction.shape[-1] != 84:
        raise ValueError("Physical metrics require matching state84 tensors")
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


@torch.no_grad()
def evaluate_windows(wm, replay, *, k=8, constant_velocity=False):
    """Physical-only evaluation interface; actual learned validation belongs to P7-A."""
    result = {"scope": "physical prediction only; no readiness decision", "horizon": k}
    arrays = {}
    for horizon in sorted({1, k}):
        e, t = replay.window_mask(horizon).nonzero(as_tuple=True)
        if not len(e):
            raise ValueError(f"No valid episode-contained {horizon}-step windows")
        batch = replay._gather_windows(e, t, horizon)
        state = batch["states"][0]
        predictions = []
        for action in batch["a_exec"]:
            state = (
                wm.spec.integrate(state, state.new_zeros(*state.shape[:-1], 82))
                if constant_velocity
                else wm.step(state, action, batch["xi"])
            )
            if state.shape[-1] != 84 or not torch.isfinite(state).all():
                raise ValueError("Invalid physical prediction")
            predictions.append(state)
        result[f"horizon_{horizon}"] = state_metrics(
            wm, state, batch["states"][-1], batch["intent_ref"][-1, :, 1]
        )
        result[f"windows_{horizon}"] = len(e)
        arrays[f"state_prediction_{horizon}"] = torch.stack(predictions).cpu()
    return result, arrays
