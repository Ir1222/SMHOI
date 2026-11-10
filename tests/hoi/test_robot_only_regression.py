"""Exercise the unchanged SuperMimic environment using the available HDMI clip."""

import pytest
import torch

from supermimic.hoi.reference import DEFAULT_ROBOT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Original environment needs CUDA")
def test_original_robot_state_actions_context_and_replay():
    import warp as wp

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.state_io import StateIO

    wp.config.kernel_cache_dir = "/tmp/smhoi_p1_warp_cache"
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[str(DEFAULT_ROBOT)],
            num_envs=4,
            replay_capacity=64,
            obs_noise=False,
            reset_noise=False,
            sampling_mode="start",
            seed=0,
        ),
        "cuda:0",
    )
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (4, 930) and obs["goal"].shape == (4, 680)
        assert env.action_manager.total_action_dim == 29
        assert (env.sim.mj_model.nq, env.sim.mj_model.nv) == (36, 35)
        io = StateIO(env)
        initial = io.read()
        assert initial.shape == (4, 71)
        io.write(torch.arange(4, device=env.device), initial)
        torch.testing.assert_close(io.read(), initial, atol=2e-6, rtol=0)
        for _ in range(16):
            obs, reward, _, _, _ = env.step(torch.zeros(4, 29, device=env.device))
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        recorder = env.recorder_manager.get_term("replay")
        assert recorder.ctx.read().shape == (4, 33)
        windows = recorder.buffer.sample_windows(8, 8)
        assert windows.states.shape == (9, 8, 71)
        assert windows.a_exec.shape == (8, 8, 29) and windows.xi.shape == (8, 33)
        assert torch.isfinite(windows.states).all()
        assert (windows.a_exec - windows.a_clean).std() > 0.01
    finally:
        env.close()
