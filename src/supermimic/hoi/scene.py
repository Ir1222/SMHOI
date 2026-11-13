"""P1 physical scene construction using original native G1 actions and mjlab stepping."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mujoco
import torch
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.hoi.asset_manifest import _add_geom, load_manifest
from supermimic.hoi.contact import HAND_BODIES, capacity_metric, hand_box_sensor_cfgs
from supermimic.hoi.reference import DEFAULT_ASSET


@dataclass(frozen=True)
class PhysicalSceneCfg:
    asset_manifest: Path = DEFAULT_ASSET
    num_envs: int = 4
    env_spacing: float = 4.0
    action_noise_std: float = 0.0  # deterministic calibration, same original action map
    nconmax: int = 256
    njmax: int = 1024
    contact_sensor_maxmatch: int = 64
    check_capacity: bool = True


def _box_spec(manifest: dict) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    obj = manifest["object"]
    body = spec.worldbody.add_body(name=obj["name"])
    body.add_freejoint(name="bread_box_root")
    inertia = obj["inertia"]
    body.explicitinertial = True
    body.mass = inertia["mass_kg"]
    body.ipos = inertia["com_pos_root_m"]
    body.iquat = inertia["principal_axes_wxyz"]
    body.inertia = inertia["diagonal_kg_m2"]
    for geom in obj["geoms"]:
        _add_geom(body, geom)
    return spec


def _add_supports(manifest: dict, spec: mujoco.MjSpec) -> None:
    # Add directly: fixed non-articulated Entity specs are automatically wrapped
    # as mocap by mjlab. These supports must remain genuinely static bodies.
    for support in manifest["supports"]:
        body = spec.worldbody.add_body(
            name=support["name"],
            pos=support["root_pos_world"],
            quat=support["root_quat_wxyz"],
        )
        _add_geom(body, support["geom"])


def make_physical_env_cfg(
    cfg: PhysicalSceneCfg | None = None, *, manifest: dict | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build only physics/actions/sensors: no tracking command, reward or Replay."""
    cfg = cfg or PhysicalSceneCfg()
    manifest = load_manifest(cfg.asset_manifest) if manifest is None else manifest
    if manifest["robot"]["selected_profile"] != "native_unchanged":
        raise ValueError("P1 requires the accepted M6.5 native_unchanged robot profile")
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=cfg.num_envs,
            env_spacing=cfg.env_spacing,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_g1_robot_cfg(),
                "box": EntityCfg(
                    spec_fn=partial(_box_spec, manifest),
                    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
                ),
            },
            spec_fn=partial(_add_supports, manifest),
            sensors=hand_box_sensor_cfgs(manifest["object"]["name"]),
        ),
        actions={
            "joint_pos": NoisyJointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=G1_ACTION_SCALE,
                use_default_offset=True,
                noise_std=cfg.action_noise_std,
            )
        },
        events={},
        metrics={"contact_capacity": MetricsTermCfg(func=capacity_metric, per_substep=True)}
        if cfg.check_capacity
        else {},
        sim=SimulationCfg(
            nconmax=cfg.nconmax,
            njmax=cfg.njmax,
            contact_sensor_maxmatch=cfg.contact_sensor_maxmatch,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=0.0,
        seed=0,
    )


def validate_scene(model: mujoco.MjModel, manifest: dict) -> None:
    """Reject wrong scene topology or accidentally disabled native hand filters."""
    if (model.nq, model.nv, model.nu) != (43, 41, 29):
        raise ValueError(f"Unexpected joint scene dimensions: {model.nq, model.nv, model.nu}")
    box_id = model.body(f"box/{manifest['object']['name']}").id
    box_geoms = {model.geom(i).name for i in range(model.ngeom) if model.geom_bodyid[i] == box_id}
    if box_geoms != {f"box/{g['name']}" for g in manifest["object"]["geoms"]}:
        raise ValueError("Box body must cover all seven manifest geoms")
    for side, bodies in HAND_BODIES.items():
        for body_name, geom_name in zip(
            bodies, (f"{side}_wrist_collision", f"{side}_hand_collision")
        ):
            geom = model.geom(f"robot/{geom_name}").id
            if (
                model.geom_bodyid[geom] != model.body(f"robot/{body_name}").id
                or model.geom_contype[geom] != 1
                or model.geom_conaffinity[geom] != 1
            ):
                raise ValueError(f"Invalid native hand collider: {geom_name}")
    for support in manifest["supports"]:
        body = model.body(support["name"]).id
        if model.body_dofnum[body] != 0 or model.body_mocapid[body] != -1:
            raise ValueError("Supports must be fixed, non-mocap bodies")


def make_physical_env(
    cfg: PhysicalSceneCfg | None = None,
    device: str = "cuda:0",
    *, manifest: dict | None = None, configure=None,
) -> ManagerBasedRlEnv:
    """Reuse ManagerBasedRlEnv unchanged; caller drives calibration/reset only."""
    from mjlab.utils.torch import configure_torch_backends

    cfg = cfg or PhysicalSceneCfg()
    configure_torch_backends()
    env_cfg = make_physical_env_cfg(cfg, manifest=manifest)
    if configure is not None:
        configure(env_cfg)
    env = ManagerBasedRlEnv(env_cfg, device=device)
    manifest = load_manifest(cfg.asset_manifest) if manifest is None else manifest
    validate_scene(env.sim.mj_model, manifest)
    # Static support poses are per-world constants. Expansion recaptures graphs;
    # following in-place writes preserve their pointers. Never mutate source assets.
    env.sim.expand_model_fields(("body_pos",))
    for support in manifest["supports"]:
        bid = env.sim.mj_model.body(support["name"]).id
        pos = torch.tensor(support["root_pos_world"], device=device, dtype=torch.float32)
        env.sim.model.body_pos[:, bid] = pos + env.scene.env_origins
        # Warp skips static geom transforms in every forward/step. Translate
        # their cached world poses once too; reset preserves these arrays.
        # In-place writes preserve the addresses captured by CUDA graphs.
        gid = env.sim.mj_model.geom(support["geom"]["name"]).id
        env.sim.data.geom_xpos[:, gid] += env.scene.env_origins
    env.sim.forward()
    return env
