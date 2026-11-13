"""Strict, named Actor/Critic weight migration; never a full training resume.

The original sm-train loader and joint WM migration are deliberately separate.
No optimizer moments, Replay, sampler or RNG state is imported here.
"""

import json
from dataclasses import asdict
from itertools import pairwise

import torch

from supermimic.hoi.checkpoint import _check_tensors
from supermimic.hoi.policy import HOIActor, HOICritic
from supermimic.hoi.policy_features import (
    CRITIC_SLICES,
    DECODER_SLICES,
    SCHEMA,
    object_scale,
)
from supermimic.policy.nets import (
    Critic,
    CriticCfg,
    Policy,
    PolicyCfg,
    _goal_scale_vector,
    _obs_scale_vector,
)

CHECKPOINT_SCHEMA = "hoi-policy-weights-v1"
LIFECYCLE = "weights-only initialization; no optimizer/Replay/sampler/RNG resume"
OLD_CRITIC_SLICES = {
    "robot_canonical": slice(0, 71),
    "robot_goal": slice(71, 751),
    "xi": slice(751, 784),
    "root_error": slice(784, 788),
}


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _digest(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Explicit lowercase artifact SHA256 required")


def _source(provenance):
    if set(provenance) != {"checkpoint_sha256", "provenance_status"}:
        raise ValueError("Explicit source hash and provenance qualification required")
    _digest(provenance["checkpoint_sha256"])
    if provenance["provenance_status"] not in {
        "verified",
        "unverified-training-provenance",
        "synthetic-test",
    }:
        raise ValueError("Unrecognized source provenance status")


def _identity(identity):
    if set(identity) != {"references", "object_asset_sha256", "support_asset_sha256"}:
        raise ValueError("Separate reference/Object/support identity required")
    for key in ("object_asset_sha256", "support_asset_sha256"):
        _digest(identity[key])
    if not isinstance(identity["references"], list) or not identity["references"]:
        raise ValueError("Nonempty explicit reference identity required")
    for ref in identity["references"]:
        for key in ("source_sha256", "manifest_sha256"):
            _digest(ref.get(key))
        if type(ref.get("training_ready")) is not bool or ref.get("role") not in {
            "original",
            "experimental",
        }:
            raise ValueError("Reference readiness/role must be preserved explicitly")
    _json(identity)


def _buffers(module, expected):
    actual = dict(module.named_buffers())
    if actual.keys() != expected.keys():
        raise ValueError("Unexpected policy normalization buffers")
    for name, value in expected.items():
        if not torch.equal(actual[name], value.to(actual[name])):
            raise ValueError(f"Policy scale/schema mismatch: {name}")


def _actor_buffers():
    cfg = PolicyCfg()
    return {
        "robot_encoder.scale": _goal_scale_vector(cfg, 680),
        "robot_history_scale": _obs_scale_vector(cfg),
        "object_history_scale": object_scale(history=True),
        "object_goal_scale": object_scale(future=True),
    }


def _critic_buffers():
    return {
        "robot_goal_scale": _goal_scale_vector(PolicyCfg(), 680),
        "object_goal_scale": object_scale(future=True),
        "object_error_scale": object_scale(),
    }


def _layers(mlp, dimensions):
    layers = [m for m in mlp.net if isinstance(m, torch.nn.Linear)]
    if [(m.in_features, m.out_features) for m in layers] != list(pairwise(dimensions)):
        raise ValueError("Incompatible HOI network dimensions")


def _validate(actor=None, critic=None):
    if actor is not None:
        if type(actor) is not HOIActor or actor.schema != SCHEMA:
            raise ValueError("Dedicated Actor B schema required")
        _layers(actor.robot_encoder.mlp, [680, 2048, 1024, 512, 256, 32])
        _layers(actor.object_history_encoder, [150, 128, 64])
        _layers(actor.object_goal_encoder, [150, 128, 64, 32])
        _layers(actor.decoder, [1058, 2048, 1024, 512, 256, 29])
        _buffers(actor, _actor_buffers())
    if critic is not None:
        if type(critic) is not HOICritic or critic.schema != SCHEMA:
            raise ValueError("Dedicated Critic B+ schema required")
        _layers(critic.mlp, [1015, 256, 128, 1])
        _buffers(critic, _critic_buffers())
    if (
        actor is not None
        and critic is not None
        and {id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}
    ):
        raise ValueError("Actor/Critic parameters must be independent")


def _source_weights(weights, original):
    _check_tensors(weights, original.state_dict())
    for key, buffer in original.named_buffers():
        if not torch.equal(weights[key].cpu(), buffer.cpu()):
            raise ValueError(f"Source normalization mismatch: {key}")
    original.load_state_dict(weights, strict=True)


def migrate_robot_actor(weights, source_cfg, target, *, provenance):
    """Exact-function initialization, with ZERO initial Object sensitivity.

    New Object encoders cannot receive a gradient until their fusion columns
    become nonzero. A nonzero-fusion test fixture is a separate initialization.
    """
    _source(provenance)
    _validate(actor=target)
    if _json(asdict(source_cfg)) != _json(asdict(PolicyCfg())):
        raise ValueError("Only the exact Robot680/history930/no-anchor contract can migrate")
    original = Policy(source_cfg).to(dtype=next(target.parameters()).dtype)
    _source_weights(weights, original)
    destination = {k: v.clone() for k, v in target.state_dict().items()}
    for name in original.encoders["motion"].state_dict():
        destination[f"robot_encoder.{name}"].copy_(weights[f"encoders.motion.{name}"])
    first = "decoder.net.0.weight"
    destination[first].zero_()
    old_first = weights["decoder.mlp.net.0.weight"]
    mapping = {"robot_latent": slice(0, 32), "robot_history": slice(32, 962)}
    for name, source_slice in mapping.items():
        destination[first][:, DECODER_SLICES[name]].copy_(old_first[:, source_slice])
    for name in original.decoder.mlp.state_dict():
        if name != "net.0.weight":
            destination[f"decoder.{name}"].copy_(weights[f"decoder.mlp.{name}"])
    destination["robot_history_scale"].copy_(weights["decoder.scale"])
    target.load_state_dict(destination, strict=True)
    return {
        "source": dict(provenance),
        "destination_schema": SCHEMA,
        "new_fusion_columns": [962, 1058],
        "new_fusion_initialization": "zero",
        "initial_object_sensitivity": "zero",
        "initial_object_encoder_gradient": "zero",
        "lifecycle": LIFECYCLE,
    }


def migrate_robot_critic(weights, source_cfg, source_policy_cfg, target, *, provenance):
    """Named 788→1015 mapping; transferred values are not calibrated HOI values."""
    _source(provenance)
    _validate(critic=target)
    if _json(asdict(source_cfg)) != _json(asdict(CriticCfg())) or _json(
        asdict(source_policy_cfg)
    ) != _json(asdict(PolicyCfg())):
        raise ValueError("Exact original Critic788/no-anchor contract required")
    original = Critic(source_cfg, pcfg=source_policy_cfg).to(dtype=next(target.parameters()).dtype)
    _source_weights(weights, original)
    destination = {k: v.clone() for k, v in target.state_dict().items()}
    first = "mlp.net.0.weight"
    destination[first].zero_()
    mapping = {**CRITIC_SLICES, "robot_canonical": slice(0, 71)}
    for name, source_slice in OLD_CRITIC_SLICES.items():
        destination[first][:, mapping[name]].copy_(weights[first][:, source_slice])
    for key in weights:
        if key.startswith("mlp.") and key != first:
            destination[key].copy_(weights[key])
    target.load_state_dict(destination, strict=True)
    return {
        "source": dict(provenance),
        "destination_schema": SCHEMA,
        "mapping": {
            name: {
                "source": [s.start, s.stop],
                "destination": [mapping[name].start, mapping[name].stop],
            }
            for name, s in OLD_CRITIC_SLICES.items()
        },
        "hoi_value_calibrated": False,
        "lifecycle": LIFECYCLE,
    }


def policy_checkpoint(actor, critic, *, identity, provenance):
    from supermimic.hoi.provenance import research_authority

    _validate(actor, critic)
    _identity(identity)
    _json(provenance)
    result = {
        "metadata": {
            "schema": CHECKPOINT_SCHEMA,
            "research_authority": research_authority(),
            "actor": actor.metadata(),
            "critic": critic.metadata(),
            "identity": json.loads(_json(identity)),
            "provenance": json.loads(_json(provenance)),
            "lifecycle": LIFECYCLE,
            "training_ready": False,
            "hoi_value_calibrated": False,
        },
        "actor": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
        "critic": {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()},
    }
    for key, module in (("actor", actor), ("critic", critic)):
        _check_tensors(result[key], module.state_dict())
    return result


def load_policy_weights(actor, critic, checkpoint, *, expected_identity):
    """Validate both networks and immutable buffers before mutating either."""
    from supermimic.hoi.provenance import research_authority

    _validate(actor, critic)
    _identity(expected_identity)
    if set(checkpoint) != {"metadata", "actor", "critic"}:
        raise ValueError("Weights-only payload required; optimizer/resume state not accepted")
    metadata = checkpoint["metadata"]
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "research_authority": research_authority(),
        "actor": actor.metadata(),
        "critic": critic.metadata(),
        "identity": expected_identity,
        "lifecycle": LIFECYCLE,
        "training_ready": False,
        "hoi_value_calibrated": False,
    }
    if set(metadata) != set(expected) | {"provenance"}:
        raise ValueError("Policy checkpoint metadata keys mismatch")
    _json(metadata)
    for key, value in expected.items():
        if _json(metadata[key]) != _json(value):
            raise ValueError(f"Policy schema/identity mismatch: {key}")
    for key, module, buffers in (
        ("actor", actor, _actor_buffers()),
        ("critic", critic, _critic_buffers()),
    ):
        _check_tensors(checkpoint[key], module.state_dict())
        for name, value in buffers.items():
            stored = checkpoint[key][name]
            if not torch.equal(stored, value.to(stored)):
                raise ValueError(f"Checkpoint normalization mismatch: {name}")
    actor.load_state_dict(checkpoint["actor"], strict=True)
    critic.load_state_dict(checkpoint["critic"], strict=True)
    return metadata
