"""Explicit pinned profiles; production provenance and candidate status are retained."""

import copy
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from supermimic.hoi.asset_manifest import load_manifest
from supermimic.hoi.reference import DEFAULT_ASSET, sha256

ROOT = Path(__file__).resolve().parents[3]
ASSET_SHA = "1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1"
SINGLE_BODY = ROOT / "outputs/hoi/contact_supervision/p6_closure/candidates/single_body.json"
SINGLE_SHA = "6c3f1a9c601336cb264cd0f979ec4d039ed380e76df6ee1a0a33358d5a82e331"
CALIBRATION = ROOT / "outputs/hoi/contact_supervision/contact_validation/calibration.json"
CALIBRATION_SHA = "34a47e06454c64f71b456bb428110fceff65302f6c6517dc85a95e151d400266"


def profile_manifest(profile):
    if sha256(DEFAULT_ASSET) != ASSET_SHA or sha256(CALIBRATION) != CALIBRATION_SHA:
        raise ValueError("Changed physical manifest or contact calibration")
    if profile == "production":
        manifest = load_manifest(DEFAULT_ASSET)
        if manifest.get("training_ready") is not True:
            raise ValueError(
                "Production M6.5 training_ready is not true; use an explicit diagnostic profile"
            )
    elif profile in ("p6-original", "p6-single-body"):
        manifest = load_manifest(DEFAULT_ASSET)
        if profile == "p6-single-body":
            if sha256(SINGLE_BODY) != SINGLE_SHA:
                raise ValueError("Changed P6 candidate")
            candidate = json.loads(SINGLE_BODY.read_text())
            manifest = copy.deepcopy(manifest)
            for key in ("geoms", "inertia"):
                manifest["object"][key] = candidate[key]
    else:
        raise ValueError("Unknown physical profile")
    return manifest


def digest_json(value):
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def physical_identity(profile, manifest, reference, guard):
    return {
        "profile": profile,
        "research_authority": manifest["research_authority"],
        "production_manifest_sha256": ASSET_SHA,
        "box_sha256": digest_json(manifest["object"]),
        "box_candidate_sha256": SINGLE_SHA if profile == "p6-single-body" else None,
        "support_sha256": digest_json(manifest["supports"]),
        "contact_targets_sha256": digest_json(manifest["contact"]),
        "calibration_sha256": CALIBRATION_SHA,
        "reference_sha256": reference.manifest["sha256"],
        "reference_manifest_sha256": sha256(reference.path.parent / "reference_manifest.json"),
        "reference_training_ready": reference.manifest.get("training_ready", False),
        "training_ready": False,
        "solver": guard.identity,
        "code_sha256": digest_json(
            {
                str(p.relative_to(ROOT)): sha256(p)
                for p in sorted((ROOT / "src/supermimic").rglob("*.py"))
            }
        ),
        "robot_manifest_sha256": digest_json(manifest["robot"]),
        "physics_dt": 0.005,
        "decimation": 4,
        "iterations": 10,
        "ls_iterations": 20,
        "dependencies": {
            n: version(n)
            for n in (
                "torch",
                "numpy",
                "mjlab",
                "mujoco",
                "mujoco-warp",
                "warp-lang",
            )
        },
    }


def compiled_model_identity(env):
    """Fingerprint compiled native Robot, Box, support, actuator and solver data."""
    model = env.sim.mj_model
    arrays = {}
    for name in dir(model):
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            arrays[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
    options = {}
    for name in dir(model.opt):
        value = getattr(model.opt, name)
        if isinstance(value, (int, float)):
            options[name] = value
        elif isinstance(value, np.ndarray):
            options[name] = value.tolist()
    # Also covers per-world support origins expanded in the native Warp model.
    from supermimic.hoi.training_checkpoint import warp_arrays

    origins = {
        key: env.sim.model.__getattr__(key).cpu().tolist() for key in ("body_pos", "geom_pos")
    }
    return {
        "compiled_arrays_sha256": digest_json(arrays),
        "options": options,
        "world_origins_sha256": digest_json(origins),
        "native_workspace_schema": {
            k: [list(v.shape), str(v.dtype)] for k, v in warp_arrays(env.sim.wp_data).items()
        },
    }
