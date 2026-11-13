"""Whole-system HOI snapshots. No Robot checkpoint format or loader is changed."""

import copy
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from supermimic.hoi.contact import SENSOR_NAMES
from supermimic.hoi.policy_features import feature_schema
from supermimic.hoi.training_safety import finite

SCHEMA = "hoi-training-lifecycle-v1"


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    return copy.deepcopy(value)


def tensor_attributes(obj):
    return {k: v for k, v in vars(obj).items() if isinstance(v, torch.Tensor)}


def warp_arrays(obj, prefix=""):
    """All native workspace arrays, including warmstarts/contacts/solver scratch.

    Restoring in place preserves captured pointers. Derived scratch is deliberately
    retained too: selecting only qpos/qvel cannot promise exact physical resume.
    """
    import warp as wp

    result = {}
    for name, value in vars(obj).items():
        key = prefix + name
        if isinstance(value, wp.array):
            result[key] = wp.to_torch(value)
        elif hasattr(value, "__dict__") and type(value).__module__.startswith("mujoco_warp"):
            result.update(warp_arrays(value, key + "."))
    return result


def runtime_tensors(trainer):
    env, recorder = trainer.env.raw, trainer.recorder
    result = {
        "simulator": warp_arrays(env.sim.wp_data),
        "replay": tensor_attributes(trainer.buffer),
        "replay_fields": trainer.buffer.fields,
        "recorder": tensor_attributes(recorder),
        "capture": tensor_attributes(recorder.capture),
        "sampler": tensor_attributes(trainer.command),
        "sampler_metrics": trainer.command.metrics,
        "environment": tensor_attributes(env),
        "actions": tensor_attributes(env.action_manager),
        "action_term": tensor_attributes(env.action_manager.get_term("joint_pos")),
        "termination": tensor_attributes(env.termination_manager),
        "termination_terms": env.termination_manager._term_dones,
        "metrics": tensor_attributes(env.metrics_manager),
        "metric_sums": env.metrics_manager._episode_sums,
        "metric_max": env.metrics_manager._episode_max,
        "metric_accum": {str(i): v for i, v in enumerate(env.metrics_manager._substep_accum)},
        "last_transition": recorder.last,
        "trainer": {k: getattr(trainer, k) for k in ("ep_reward_sum", "ep_len")},
    }
    for name in SENSOR_NAMES:
        result["sensor/" + name] = env.scene[name]._history_state
    for name in ("robot", "box"):
        result["entity/" + name] = tensor_attributes(env.scene[name].data)
    return result


def rng_state():
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": (
            np_state[0],
            torch.tensor(np_state[1].astype(np.int64)),
            np_state[2],
            np_state[3],
            np_state[4],
        ),
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].cpu().numpy().astype(np.uint32), n[2], n[3], n[4]))


def config_contract(cfg):
    result = cpu_tree(asdict(cfg))
    for name in ("log_root", "run_name", "resume", "approval_record", "initial_wm"):
        result.pop(name, None)
    # Exact learning schedule and budget are part of resume, not silently reset.
    return result


def model_states(trainer):
    return {k: getattr(trainer, k).state_dict() for k in ("wm", "policy", "critic")}


def optimizers(trainer):
    return {
        "wm": trainer.wm_trainer.opt,
        "actor": trainer.policy_trainer.opt,
        "critic": trainer.policy_trainer.critic_opt,
    }


def snapshot(trainer):
    env = trainer.env.raw
    if trainer.recorder.capture.count.any() or not trainer.recorder.ready.all():
        raise RuntimeError("Checkpoint must be between complete bound control intervals")
    trainer.guard.attest(env, "checkpoint")
    finite("checkpoint weights", model_states(trainer))
    return {
        "schema": SCHEMA,
        "features": feature_schema(),
        "identity": trainer.identity,
        "config": config_contract(trainer.cfg),
        "lifecycle": trainer.cfg.execution,
        "models": cpu_tree(model_states(trainer)),
        "optimizers": cpu_tree({k: v.state_dict() for k, v in optimizers(trainer).items()}),
        "runtime": cpu_tree(runtime_tensors(trainer)),
        "replay_metadata": copy.deepcopy(trainer.buffer.metadata),
        "last_terms": cpu_tree(trainer.recorder.last_terms),
        "counters": {
            "iteration": trainer.iteration,
            "start_iteration": trainer.start_iteration,
            "num_updates": trainer.num_updates,
            "utd_acc": trainer._utd_acc,
            "total_rows": trainer.buffer.total_rows,
            "sim_step_counter": env._sim_step_counter,
            "common_step_counter": env.common_step_counter,
            "metric_substep_count": env.metrics_manager._substep_count,
        },
        "optimizer_steps": dict(trainer.learning_safety.steps),
        "rng": rng_state(),
        "guard_evidence": trainer.guard.evidence(),
    }


def validate_tensors(saved, expected, *, where):
    if set(saved) != set(expected):
        raise ValueError(f"Checkpoint keys differ at {where}")
    for key, target in expected.items():
        value = saved[key]
        if isinstance(target, dict):
            validate_tensors(value, target, where=where + "/" + key)
        elif (
            not isinstance(value, torch.Tensor)
            or value.shape != target.shape
            or value.dtype != target.dtype
        ):
            raise ValueError(f"Checkpoint tensor differs at {where}/{key}")


def validate_optimizer(saved, optimizer, label):
    expected = optimizer.state_dict()
    if set(saved) != {"state", "param_groups"} or len(saved["param_groups"]) != len(
        expected["param_groups"]
    ):
        raise ValueError(f"Invalid {label} optimizer schema")
    valid_indices = set()
    for group, schema, target in zip(
        saved["param_groups"], expected["param_groups"], optimizer.param_groups
    ):
        if set(group) != set(schema) or group["params"] != schema["params"]:
            raise ValueError(f"Invalid {label} optimizer parameter groups")
        finite(label + " optimizer hyperparameters", group)
        for key in schema.keys() - {"params", "lr"}:
            if group[key] != schema[key]:
                raise ValueError(f"Changed {label} optimizer {key}")
        for index, parameter in zip(group["params"], target["params"]):
            valid_indices.add(index)
            state = saved["state"].get(index, {})
            if state and set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(f"Invalid {label} Adam state")
            if state and (
                not isinstance(state["step"], torch.Tensor)
                or state["step"].numel() != 1
                or state["step"].item() < 0
            ):
                raise ValueError(f"Invalid {label} Adam step")
            for name in ("exp_avg", "exp_avg_sq"):
                if state and (
                    state[name].shape != parameter.shape or state[name].dtype != parameter.dtype
                ):
                    raise ValueError(f"Invalid {label} Adam moments")
    if set(saved["state"]) - valid_indices:
        raise ValueError(f"Unknown {label} optimizer parameter")
    finite(label + " checkpoint optimizer", list(saved["state"].values()))


def restore(trainer, checkpoint):
    expected_header = {
        "schema": SCHEMA,
        "features": feature_schema(),
        "identity": trainer.identity,
        "config": config_contract(trainer.cfg),
        "lifecycle": trainer.cfg.execution,
        "replay_metadata": trainer.buffer.metadata,
    }
    for name, value in expected_header.items():
        if checkpoint.get(name) != value:
            raise ValueError(f"Incompatible HOI checkpoint {name}")
    current = runtime_tensors(trainer)
    validate_tensors(checkpoint["models"], model_states(trainer), where="models")
    validate_tensors(checkpoint["runtime"], current, where="runtime")
    finite("checkpoint models", checkpoint["models"])
    for key in ("wm", "policy", "critic"):
        for name, tensor in getattr(trainer, key).named_buffers():
            if not torch.equal(checkpoint["models"][key][name], tensor.cpu()):
                raise ValueError(f"Changed fixed model buffer {key}/{name}")
    for key, optimizer in optimizers(trainer).items():
        validate_optimizer(checkpoint["optimizers"][key], optimizer, key)
    counters = checkpoint["counters"]
    if not 0 <= counters["iteration"] <= trainer.cfg.schedule.total_iterations:
        raise ValueError("Invalid checkpoint iteration")
    for key in (
        "iteration",
        "start_iteration",
        "num_updates",
        "total_rows",
        "sim_step_counter",
        "common_step_counter",
        "metric_substep_count",
    ):
        if not isinstance(counters[key], int) or counters[key] < 0:
            raise ValueError(f"Invalid checkpoint counter {key}")
    if not 0 <= counters["utd_acc"] < 1 or counters["start_iteration"] > counters["iteration"]:
        raise ValueError("Invalid checkpoint schedule accumulator")
    if checkpoint["optimizer_steps"].keys() != optimizers(trainer).keys() or any(
        not isinstance(n, int) or n < 0 for n in checkpoint["optimizer_steps"].values()
    ):
        raise ValueError("Invalid optimizer counters")
    # Validate RNG formats using isolated generators; leave global RNG untouched.
    rng = checkpoint["rng"]
    random.Random().setstate(rng["python"])
    torch.Generator().set_state(rng["torch"])
    if len(rng["cuda"]) != torch.cuda.device_count():
        raise ValueError("CUDA RNG device count differs")
    for i, state in enumerate(rng["cuda"]):
        torch.Generator(device=f"cuda:{i}").set_state(state)
    n = rng["numpy"]
    np.random.RandomState().set_state((n[0], n[1].numpy().astype(np.uint32), n[2], n[3], n[4]))
    finite(
        "checkpoint runtime",
        {k: v for k, v in checkpoint["runtime"].items() if k not in ("simulator", "metric_max")},
        checkpoint["last_terms"],
    )
    native = checkpoint["runtime"]["simulator"]
    finite(
        "checkpoint physical state",
        *(native[k] for k in ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl")),
    )
    if native["overflow"].any():
        raise ValueError("Checkpoint contains a failed physical solve")
    if (
        not checkpoint["runtime"]["recorder"]["ready"].all()
        or checkpoint["runtime"]["capture"]["count"].any()
    ):
        raise ValueError("Checkpoint contains an incomplete control interval")
    # Validation precedes all model/state mutation. Capture under the active guard,
    # then copy the complete saved native workspace back into the captured arrays.
    trainer.env.raw.sim.create_graph()
    trainer.guard.attest(trainer.env.raw, "resume_after_recapture")
    for key, model in ((k, getattr(trainer, k)) for k in ("wm", "policy", "critic")):
        model.load_state_dict(checkpoint["models"][key], strict=True)
    for key, optimizer in optimizers(trainer).items():
        optimizer.load_state_dict(checkpoint["optimizers"][key])
    with torch.no_grad():
        for group, tensors in current.items():
            for key, target in tensors.items():
                target.copy_(checkpoint["runtime"][group][key].to(target.device))
    trainer.recorder.last_terms = {
        k: v.to(trainer.device) for k, v in checkpoint["last_terms"].items()
    }
    trainer.buffer.total_rows = counters["total_rows"]
    for key in ("iteration", "start_iteration", "num_updates"):
        setattr(trainer, key, counters[key])
    trainer._utd_acc = counters["utd_acc"]
    env = trainer.env.raw
    env._sim_step_counter = counters["sim_step_counter"]
    env.common_step_counter = counters["common_step_counter"]
    env.metrics_manager._substep_count = counters["metric_substep_count"]
    for name in SENSOR_NAMES:
        env.scene[name]._invalidate_cache()
    trainer.learning_safety.steps.update(checkpoint["optimizer_steps"])
    trainer.env.resume_pending = True
    restore_rng(checkpoint["rng"])  # Last: construction/capture must not consume resumed RNG.


def save_exclusive(trainer, path):
    path = Path(path)
    payload = snapshot(trainer)
    stream = path.open("xb")
    try:
        with stream:
            torch.save(payload, stream)
    except BaseException:
        path.unlink()  # Only the incomplete file exclusively created above.
        raise


def load(trainer, path):
    restore(trainer, torch.load(path, map_location="cpu", weights_only=True))
