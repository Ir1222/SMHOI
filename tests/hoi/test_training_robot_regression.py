"""Exercise original Trainer construction/run and its historical checkpoint schema."""

import copy

import pytest
import torch

from supermimic.config import TrainConfig
from supermimic.hoi.reference import DEFAULT_ROBOT
from supermimic.policy.nets import Critic, Policy
from supermimic.training.loop import Trainer
from supermimic.world_model.world_model import WorldModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Original sm-train requires CUDA")
def test_original_trainer_and_checkpoint(tmp_path):
    cfg = TrainConfig(log_root=tmp_path, run_name="robot")
    cfg.data.npz_dir = DEFAULT_ROBOT.parent
    cfg.data.clips = [DEFAULT_ROBOT.stem]
    cfg.env.num_envs = 2
    cfg.env.replay_capacity = 32
    cfg.env.obs_noise = cfg.env.reset_noise = cfg.env.domain_randomization = False
    cfg.env.sampling_mode = "start"
    cfg.wm_trainer.compile = cfg.policy_trainer.compile = None
    cfg.schedule.total_iterations = cfg.schedule.log_interval = cfg.schedule.checkpoint_interval = 2
    cfg.schedule.updates_per_env_step = 0
    trainer = Trainer(cfg)
    try:
        assert type(trainer.wm) is WorldModel
        assert type(trainer.policy) is Policy and type(trainer.critic) is Critic
        trainer.run()
        assert trainer.iteration == 2 and trainer.num_updates == 0
        path = trainer.log_dir / "ckpt_latest.pt"
        saved = torch.load(path, weights_only=False)
        assert set(saved) == {
            "iteration",
            "num_updates",
            "wm",
            "policy",
            "critic",
            "wm_opt",
            "policy_opt",
            "critic_opt",
            "bin_return",
            "bin_memory",
            "config",
        }
        expected = {k: v.detach().clone() for k, v in trainer.policy.state_dict().items()}
    finally:
        trainer.env.close()
        trainer.writer.close()
    cfg2 = copy.deepcopy(cfg)
    cfg2.run_name = "robot_resume"
    cfg2.resume = path
    trainer = Trainer(cfg2)
    try:
        assert trainer.iteration == trainer.start_iteration == 2
        for key, value in trainer.policy.state_dict().items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
        obs, _ = trainer.env.reset()
        assert obs["policy"].shape == (2, 930) and obs["goal"].shape == (2, 680)
        assert trainer.env.sim.mj_model.nq == 36
    finally:
        trainer.env.close()
        trainer.writer.close()
