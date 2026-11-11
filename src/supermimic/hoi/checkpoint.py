"""Explicit named robot-WM migration and strict HOI weight/contract loading.

Initialization only: no Actor/Critic migration, optimizer reuse or stochastic resume.
"""

import json
from dataclasses import asdict

import torch

from supermimic.hoi.force import ForceStatistics
from supermimic.hoi.state_spec import ROBOT_FEATURES, ROBOT_RESIDUALS
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def input_columns(*, joint, support=False):
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
    new_columns = input_columns(joint=True, support=target.support is not None)
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
        "new_object_and_force_heads": "zero",
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


def joint_checkpoint(wm, *, replay_metadata, readiness, provenance):
    contract = _replay_contract(replay_metadata)
    if wm.spec.dt != contract["control_dt_s"]:
        raise ValueError("State integration timestep differs from measurements")
    if wm.support is not None and wm.support.asset_sha256 != contract["asset_manifest_sha256"]:
        raise ValueError("Support geometry asset mismatch")
    if not abs(float(wm.force_statistics.force_on_N) - contract["force_on_N"]) < 1e-8:
        raise ValueError("Force threshold differs from supervision")
    metadata = {
        "schema": "hoi-joint-wm-v1",
        "dimensions": {
            "state": 84,
            "canonical": 86,
            "support": 18 if wm.support else 0,
            "physical_residual": 82,
            "force": 6,
        },
        "config": asdict(wm.cfg),
        "replay": contract,
        "force_normalization": {
            k: v.detach().cpu().tolist() for k, v in wm.force_statistics.state_dict().items()
        },
        "force_active_mass": wm.force_statistics.active_mass,
        "support_schema": wm.support.schema if wm.support else None,
        "readiness": readiness,
        "provenance": provenance,
        "actor_force_optimization_enabled": False,
        "lifecycle": "WM weights only; no optimizer/Replay/RNG resume",
    }
    return {
        "metadata": metadata,
        "wm": {k: v.detach().cpu().clone() for k, v in wm.state_dict().items()},
    }


def load_joint_weights(wm, checkpoint, *, expected_replay_metadata):
    metadata = checkpoint["metadata"]
    expected = joint_checkpoint(
        wm, replay_metadata=expected_replay_metadata, readiness={}, provenance={}
    )["metadata"]
    for key in ("schema", "dimensions", "config", "replay", "support_schema", "force_active_mass"):
        if _canonical(metadata.get(key)) != _canonical(expected[key]):
            raise ValueError(f"Checkpoint metadata mismatch: {key}")
    if metadata.get("actor_force_optimization_enabled") is not False:
        raise ValueError("P3 weights cannot enable Actor force optimization")
    weights = checkpoint["wm"]
    current = wm.state_dict()
    if weights.keys() != current.keys():
        raise ValueError("Checkpoint tensor keys must match exactly")
    for key, value in weights.items():
        if value.shape != current[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint tensor shape/value mismatch: {key}")
    if set(metadata["force_normalization"]) != set(wm.force_statistics.state_dict()):
        raise ValueError("Incomplete force normalization metadata")
    verified_stats = ForceStatistics(
        weights["force_statistics.scale_N"],
        weights["force_statistics.counts_inactive_active"],
        weights["force_statistics.median_active_N"],
        force_on_N=metadata["replay"]["force_on_N"],
        active_mass=metadata["force_active_mass"],
    )
    for key, value in verified_stats.state_dict().items():
        if not torch.equal(value.cpu(), weights[f"force_statistics.{key}"].cpu()):
            raise ValueError("Inconsistent force normalization/counts/class weights")
    for key, value in metadata["force_normalization"].items():
        if not torch.equal(
            weights[f"force_statistics.{key}"].cpu(),
            torch.as_tensor(value, dtype=weights[f"force_statistics.{key}"].dtype),
        ):
            raise ValueError("Force normalization metadata/tensor disagreement")
    # Check fixed geometry against the constructor, preventing stale/manipulated
    # support tensors from silently overriding the hashed manifest contract.
    for key in ("support.positions", "support.quaternions"):
        if key in current and not torch.equal(weights[key].cpu(), current[key].cpu()):
            raise ValueError("Support tensor disagrees with immutable asset")
    wm.load_state_dict(weights, strict=True)
    return metadata
