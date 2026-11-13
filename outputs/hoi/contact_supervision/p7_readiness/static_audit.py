"""P7.0 reproducible read-only entrypoint and external checkpoint inspection."""

import inspect
import json
from importlib import metadata
from pathlib import Path, PosixPath
from types import SimpleNamespace

import torch

from supermimic.config import TrainConfig
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.hoi.asset_manifest import load_manifest
from supermimic.hoi.reference import DEFAULT_ASSET, sha256
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.training.loop import Trainer

root = Path(__file__).resolve().parents[4]
out = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.manual_seed(700)
cfg = make_env_cfg(
    SuperMimicEnvCfgArgs(
        npz_files=[
            str(
                root
                / "outputs/hoi/contact_supervision/p6_closure/reference_candidate_v1/reference.npz"
            )
        ],
        num_envs=2,
    )
)
opt = SimpleNamespace()
cfg.sim.apply_wp_opt(opt)
report = {
    "console_scripts": {e.name: e.value for e in metadata.distribution("supermimic").entry_points},
    "training_env_entities": list(cfg.scene.entities),
    "training_observation_groups": list(cfg.observations),
    "training_warn_overflow": opt.warn_overflow,
    "train_config_fields": list(TrainConfig.__dataclass_fields__),
    "guard_referenced_by_trainer": "bracket_precision_guard" in inspect.getsource(Trainer),
    "target_checkpoint_exists": (root / "logs/hdmi_robot_only/ckpt_0012000.pt").is_file(),
    "production_scene_source_check": None,
    "optimizer_updates": 0,
}
try:
    load_manifest(DEFAULT_ASSET)
    report["production_scene_source_check"] = "PASS"
except (FileNotFoundError, ValueError) as error:
    report["production_scene_source_check"] = str(error)

p = Path("/home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt")
assert sha256(p) == "16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7"
with torch.serialization.safe_globals([PosixPath]):
    ck = torch.load(p, map_location="cpu", weights_only=True)
pcfg = PolicyCfg(**ck["config"]["policy"])
policy = Policy(pcfg)
critic = Critic(CriticCfg(**ck["config"]["critic"]), pcfg=pcfg)
policy.load_state_dict(ck["policy"], strict=True)
critic.load_state_dict(ck["critic"], strict=True)
obs = torch.randn(2, 930) * 0.1
goal = torch.randn(2, pcfg.goal_dim) * 0.1
with torch.no_grad():
    a = policy(obs, goal)
    b = policy(obs + 0.01, goal)
    v = critic(torch.zeros(2, 71), goal, torch.zeros(2, 33), torch.zeros(2, 4))
assert torch.isfinite(a).all() and torch.isfinite(v).all()
assert not torch.equal(a, b)
report["external_checkpoint"] = {
    "path": str(p),
    "sha256": sha256(p),
    "strict_policy_and_critic_load": True,
    "obs_width": 930,
    "goal_width": pcfg.goal_dim,
    "critic_input_width": critic.mlp.net[0].in_features,
    "action_shape": list(a.shape),
    "value_shape": list(v.shape),
    "synthetic_obs_perturbation_action_max_change": float((b - a).abs().max()),
    "adopted_as_pilot_initialization": False,
    "provenance": "External robot-only candidate; training Git/data hashes absent. Compatibility only.",
    "physical_policy_evaluation": "NOT RUN",
}
print(json.dumps(report, indent=2))
(out / "static_audit.json").write_text(json.dumps(report, indent=2) + "\n")
