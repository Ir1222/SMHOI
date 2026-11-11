"""Named robot/legacy-joint migration and strict reference-contact WM checkpoints.

Initialization only: no Actor/Critic migration, optimizer reuse or stochastic resume.
"""

import json
from dataclasses import asdict

import torch

from supermimic.hoi.state_spec import (
    OBJECT_RESIDUALS,
    ROBOT_FEATURES,
    ROBOT_RESIDUALS,
    ObjectSpecCfg,
)
from supermimic.hoi.world_model import JointWorldModelCfg, LossWeights
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def input_columns(*, joint, support=False, contact=False):
    columns = {f"robot.{k}": v for k, v in ROBOT_FEATURES.items()}
    offset = 71
    if joint:
        for name, width in [
            ("position_heading", 3),
            ("rotation6_heading", 6),
            ("linear_heading", 3),
            ("angular_body", 3),
        ]:
            columns[f"object.{name}"] = slice(offset, offset + width)
            offset += width
        if contact:
            columns["contact_intent.current"] = slice(offset, offset + 1)
            offset += 1
        if support:
            for support_id in range(2):
                for name, width in [("position_heading", 3), ("rotation6_heading", 6)]:
                    columns[f"support{support_id}.{name}"] = slice(offset, offset + width)
                    offset += width
    columns["action.executed"] = slice(offset, offset + 29)
    offset += 29
    for name, width in [("friction", 1), ("encoder_bias", 29), ("torso_com_offset", 3)]:
        columns[f"xi.{name}"] = slice(offset, offset + width)
        offset += width
    return columns


def _canonical(value):
    # JSON intentionally treats tuple/list configs identically, never tensors.
    return json.dumps(value, sort_keys=True, allow_nan=False)


def robot_config(mapping):
    mapping = dict(mapping)
    state = dict(mapping.pop("state_spec"))
    for key, value in state.items():
        if isinstance(value, list):
            state[key] = tuple(value)
    for key in ("hidden", "term_hidden"):
        mapping[key] = tuple(mapping[key])
    return WorldModelCfg(**mapping, state_spec=StateSpecCfg(**state))


def contact_config(mapping, *, legacy=False):
    """Explicit config translation; force coefficients never become contact weights."""
    mapping = dict(mapping)
    if legacy:
        if mapping.pop("force_loss") not in ("huber", "l1") or not 0 < mapping.pop(
            "force_delta"
        ) < float("inf"):
            raise ValueError("Invalid legacy force config")
        weights = mapping["weights"]
        if set(weights) != {"robot", "object", "force"}:
            raise ValueError("Invalid legacy loss weights")
        mapping["weights"] = {
            "robot": weights["robot"],
            "object": weights["object"],
            "contact": 1.0,
        }
    obj = ObjectSpecCfg(**{k: tuple(v) for k, v in mapping.pop("object_spec").items()})
    weights = LossWeights(**mapping.pop("weights"))
    support = mapping.pop("support_context")
    robot = robot_config(mapping)
    return JointWorldModelCfg(
        **{k: v for k, v in asdict(robot).items() if k != "state_spec"},
        state_spec=robot.state_spec,
        object_spec=obj,
        weights=weights,
        support_context=support,
    )


def migrate_robot_state_dict(source_dict, source_cfg, target):
    """Validate the complete source strictly before named, shape-checked transfer."""
    if (
        not source_cfg.hidden
        or tuple(source_cfg.hidden) != tuple(target.cfg.hidden)
        or tuple(source_cfg.term_hidden) != tuple(target.cfg.term_hidden)
        or _canonical(asdict(source_cfg.state_spec)) != _canonical(asdict(target.cfg.state_spec))
        or source_cfg.height_threshold != target.cfg.height_threshold
        or source_cfg.tilt_threshold != target.cfg.tilt_threshold
    ):
        raise ValueError(
            "Incompatible robot trunk, normalization, integration or termination contract"
        )
    original = WorldModel(source_cfg, target.heads, "cpu")
    original.load_state_dict(source_dict, strict=True)
    if any(not torch.isfinite(t).all() for t in source_dict.values()):
        raise ValueError("Nonfinite source checkpoint")
    destination = {k: v.clone() for k, v in target.state_dict().items()}
    old_columns = input_columns(joint=False)
    new_columns = input_columns(joint=True, support=True, contact=True)
    first = "dynamics.mlp.net.0.weight"
    final_index = 2 * len(source_cfg.hidden)
    last = f"dynamics.mlp.net.{final_index}.weight"
    last_bias = f"dynamics.mlp.net.{final_index}.bias"
    destination[first].zero_()
    for name, source_slice in old_columns.items():
        destination[first][:, new_columns[name]] = source_dict[first][:, source_slice]
    destination[last].zero_()
    destination[last_bias].zero_()
    for name, rows in ROBOT_RESIDUALS.items():
        destination[last][rows] = source_dict[last][rows]
        destination[last_bias][rows] = source_dict[last_bias][rows]
    for key, value in source_dict.items():
        if key in (first, last, last_bias):
            continue
        if key not in destination or destination[key].shape != value.shape:
            raise ValueError(f"Unmapped/incompatible source tensor: {key}")
        destination[key].copy_(value)
    target.load_state_dict(destination, strict=True)
    return {
        "source": "robot71/action29/xi33/residual70",
        "input_mapping": {
            k: {
                "source": [v.start, v.stop],
                "destination": [new_columns[k].start, new_columns[k].stop],
            }
            for k, v in old_columns.items()
        },
        "robot_output_mapping": {k: [v.start, v.stop] for k, v in ROBOT_RESIDUALS.items()},
        "new_input_columns": "zero",
        "new_object_and_contact_heads": "zero (contact probability 0.5)",
        "optimizer": "not migrated; initialization, not resume",
    }


def _replay_contract(metadata):
    keys = (
        "schema",
        "state",
        "force_frame",
        "force_sign",
        "hand_order",
        "force_unit",
        "force_on_N",
        "physics_dt_s",
        "decimation",
        "control_dt_s",
        "asset_manifest_sha256",
        "sensors",
        "references",
    )
    if not all(k in metadata for k in keys):
        raise ValueError("Complete physical Replay metadata required")
    contract = {k: metadata[k] for k in keys}
    # Relocation of the exact bytes is allowed; reference identity is the SHA, not path.
    contract["references"] = [r["sha256"] for r in metadata["references"]]
    if (
        contract["schema"] != "hoi-transition-v1"
        or contract["force_frame"] != "contemporaneous_box_at_substep_solve"
        or contract["force_sign"] != "hand_on_box"
        or contract["force_unit"] != "N"
        or contract["hand_order"] != ["left", "right"]
        or contract["control_dt_s"] != 0.02
        or contract["physics_dt_s"] != 0.005
        or contract["decimation"] != 4
    ):
        raise ValueError("Incompatible physical force/state contract")
    return contract


CONTACT_CONTRACT = {
    "source": "canonical_reference.object_contact",
    "meaning": "global_desired_reference_contact_intent",
    "pair": "intent_ref=[contact_t,contact_t+1]",
    "feedback": "sigmoid_probability_without_teacher_forcing",
    "threshold": 0.5,
    "frame_index_input": False,
}
DIMENSIONS = {
    "physical_state": 84,
    "predictive_state": 85,
    "canonical": 86,
    "contact": 1,
    "support": 18,
    "input": 167,
    "physical_residual": 82,
    "output": 83,
}


def joint_checkpoint(wm, *, replay_metadata, readiness, provenance):
    contract = _replay_contract(replay_metadata)
    if wm.spec.dt != contract["control_dt_s"]:
        raise ValueError("State integration timestep differs from measurements")
    if wm.support.asset_sha256 != contract["asset_manifest_sha256"]:
        raise ValueError("Support geometry asset mismatch")
    if replay_metadata.get("intent_ref") != "global[start,end]":
        raise ValueError("Global reference intent Replay contract required")
    metadata = {
        "schema": "hoi-joint-contact-wm-v2",
        "dimensions": dict(DIMENSIONS),
        "contact": dict(CONTACT_CONTRACT),
        "config": asdict(wm.cfg),
        "replay": contract,  # Physical sensors remain provenance, never targets.
        "support_schema": wm.support.schema,
        "readiness": readiness,
        "provenance": provenance,
        "actor_force_optimization_enabled": False,
        "lifecycle": "WM weights only; no optimizer/Replay/RNG resume",
    }
    return {
        "metadata": metadata,
        "wm": {k: v.detach().cpu().clone() for k, v in wm.state_dict().items()},
    }


def _check_tensors(weights, expected):
    if weights.keys() != expected.keys():
        raise ValueError("Checkpoint tensor keys must match exactly")
    for key, value in weights.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[key].shape
            or value.dtype != expected[key].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Checkpoint tensor shape/dtype/value mismatch: {key}")


def _check_support(weights, target):
    for key in ("support.positions", "support.quaternions"):
        if not torch.equal(weights[key].cpu(), target.state_dict()[key].cpu()):
            raise ValueError("Support tensor disagrees with immutable asset")


def load_joint_weights(wm, checkpoint, *, expected_replay_metadata):
    if set(checkpoint) != {"metadata", "wm"}:
        raise ValueError("Checkpoint payload keys must match exactly")
    metadata = checkpoint["metadata"]
    expected = joint_checkpoint(
        wm, replay_metadata=expected_replay_metadata, readiness={}, provenance={}
    )["metadata"]
    if metadata.keys() != expected.keys():
        raise ValueError("Checkpoint metadata keys must match exactly")
    for key in expected.keys() - {"readiness", "provenance"}:
        if _canonical(metadata[key]) != _canonical(expected[key]):
            raise ValueError(f"Checkpoint metadata mismatch: {key}")
    _canonical(metadata)  # Reject nonfinite/non-JSON metadata too.
    weights = checkpoint["wm"]
    _check_tensors(weights, wm.state_dict())
    _check_support(weights, wm)
    wm.load_state_dict(weights, strict=True)
    return metadata


def validate_legacy_joint(checkpoint, target, *, expected_replay_metadata):
    """Validate an old 166/88 payload completely, only for historical migration."""
    from supermimic.hoi.force import ForceStatistics

    if set(checkpoint) != {"metadata", "wm"}:
        raise ValueError("Legacy payload keys must match exactly")
    meta, weights = checkpoint["metadata"], checkpoint["wm"]
    required = {
        "schema",
        "dimensions",
        "config",
        "replay",
        "force_normalization",
        "force_active_mass",
        "support_schema",
        "readiness",
        "provenance",
        "actor_force_optimization_enabled",
        "lifecycle",
    }
    if set(meta) != required or meta["schema"] != "hoi-joint-wm-v1":
        raise ValueError("Incompatible legacy schema/metadata keys")
    if meta["dimensions"] != {
        "state": 84,
        "canonical": 86,
        "support": 18,
        "physical_residual": 82,
        "force": 6,
    }:
        raise ValueError("Incompatible legacy dimensions")
    cfg = contact_config(meta["config"], legacy=True)
    # Preserve physical/termination coefficients as well as integration and architecture.
    if _canonical(asdict(cfg)) != _canonical(asdict(target.cfg)):
        raise ValueError("Incompatible legacy config/normalization/architecture")
    if (
        _canonical(meta["replay"]) != _canonical(_replay_contract(expected_replay_metadata))
        or target.support.asset_sha256 != meta["replay"]["asset_manifest_sha256"]
    ):
        raise ValueError("Legacy replay/reference/asset mismatch")
    if (
        meta["support_schema"] != target.support.schema
        or meta["actor_force_optimization_enabled"] is not False
    ):
        raise ValueError("Incompatible legacy support/Actor contract")
    _canonical(meta)
    # No fitting: reconstruct and validate only the persisted historical statistics.
    names = {"scale_N", "counts_inactive_active", "median_active_N", "class_weights", "force_on_N"}
    if (
        set(meta["force_normalization"]) != names
        or not {f"force_statistics.{k}" for k in names} <= weights.keys()
    ):
        raise ValueError("Incomplete legacy normalization keys")
    stats = ForceStatistics(
        weights["force_statistics.scale_N"],
        weights["force_statistics.counts_inactive_active"],
        weights["force_statistics.median_active_N"],
        force_on_N=meta["replay"]["force_on_N"],
        active_mass=meta["force_active_mass"],
    )
    expected = {k: v.cpu() for k, v in target.state_dict().items()}
    final = 2 * len(target.cfg.hidden)
    first, last, bias = (
        "dynamics.mlp.net.0.weight",
        f"dynamics.mlp.net.{final}.weight",
        f"dynamics.mlp.net.{final}.bias",
    )
    expected[first] = torch.empty(target.cfg.hidden[0], 166)
    expected[last] = torch.empty(88, target.cfg.hidden[-1])
    expected[bias] = torch.empty(88)
    expected.update({f"force_statistics.{k}": v for k, v in stats.state_dict().items()})
    _check_tensors(weights, expected)
    for key, value in stats.state_dict().items():
        stored = weights[f"force_statistics.{key}"].cpu()
        if not torch.equal(stored, value.cpu()) or not torch.equal(
            stored, torch.as_tensor(meta["force_normalization"][key], dtype=stored.dtype)
        ):
            raise ValueError("Inconsistent legacy force normalization")
    _check_support(weights, target)
    return first, last, bias


def migrate_joint_weights(checkpoint, target, *, expected_replay_metadata):
    first, last, bias = validate_legacy_joint(
        checkpoint, target, expected_replay_metadata=expected_replay_metadata
    )
    source = checkpoint["wm"]
    destination = {k: v.clone() for k, v in target.state_dict().items()}
    old_columns = input_columns(joint=True, support=True)
    new_columns = input_columns(joint=True, support=True, contact=True)
    destination[first].zero_()
    for name, old_slice in old_columns.items():
        destination[first][:, new_columns[name]] = source[first][:, old_slice]
    output_rows = {f"robot.{k}": v for k, v in ROBOT_RESIDUALS.items()}
    output_rows.update(
        {f"object.{k}": slice(v.start + 70, v.stop + 70) for k, v in OBJECT_RESIDUALS.items()}
    )
    destination[last].zero_()
    destination[bias].zero_()
    for rows in output_rows.values():
        destination[last][rows] = source[last][rows]
        destination[bias][rows] = source[bias][rows]
    for key in destination.keys() - {first, last, bias}:
        destination[key].copy_(source[key])
    target.load_state_dict(destination, strict=True)
    return {
        "source_schema": "hoi-joint-wm-v1",
        "destination_schema": "hoi-joint-contact-wm-v2",
        "input_mapping": {
            k: {
                "source": [v.start, v.stop],
                "destination": [new_columns[k].start, new_columns[k].stop],
            }
            for k, v in old_columns.items()
        },
        "output_mapping": {k: [v.start, v.stop] for k, v in output_rows.items()},
        "new_contact_input_column": [86, 87],
        "new_contact_input_initialization": "zero",
        "discarded_output_rows": [82, 88],
        "new_contact_output_row": [82, 83],
        "new_contact_output_initialization": "weight/bias zero; probability 0.5",
        "discarded_tensors": sorted(k for k in source if k.startswith("force_statistics.")),
        "readiness": "unvalidated",
        "optimizer": "not migrated; initialization, not resume",
    }


@torch.no_grad()
def legacy_joint_step(checkpoint, target, state, action, xi):
    """Read-only physical prediction probe after validate_legacy_joint; no second WM."""
    from torch.nn import functional as F

    weights = checkpoint["wm"]
    x = torch.cat(
        (target.spec.canonical(state), target.support(state), action, target.spec.normalize_xi(xi)),
        -1,
    )
    for layer in range(len(target.cfg.hidden) + 1):
        prefix = f"dynamics.mlp.net.{2 * layer}"
        x = F.linear(x, weights[prefix + ".weight"], weights[prefix + ".bias"])
        if layer < len(target.cfg.hidden):
            x = F.silu(x)
    return target.spec.integrate(state, x[..., :82])
