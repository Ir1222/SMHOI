"""No learned physical updates: scheduling spies and <=16-step zero-update paths."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import tyro

from supermimic.config import TrainConfig
from supermimic.hoi import training_checkpoint as checkpoint
from supermimic.hoi.play import PlayConfig, load_play_source
from supermimic.hoi.play import execute as play
from supermimic.hoi.reference import ReferenceClip
from supermimic.hoi.startup import StartupProgress
from supermimic.hoi.training import HOITrainer
from supermimic.hoi.training_config import REFINED_SHA256, HOITrainConfig
from supermimic.hoi.training_safety import SolverGuard
from supermimic.hoi.training_scene import profile_manifest, validate_reference_binding
from supermimic.training.loop import Trainer


def test_default_scratch_and_reference_binding():
    cfg = tyro.cli(HOITrainConfig, args=[])
    cfg.validate()
    assert cfg.resume is cfg.initial_wm is None
    assert cfg.execution == "alternating" and cfg.env.sampling_mode == "return"
    assert cfg.effective_sampler_g_max == pytest.approx(18)
    clip = ReferenceClip(cfg.reference)
    validate_reference_binding(cfg.profile, clip)
    assert clip.manifest["sha256"] == REFINED_SHA256
    assert not clip.manifest["training_ready"]
    manifest = profile_manifest(cfg.profile)
    assert len(manifest["object"]["geoms"]) == 1 and not manifest["training_ready"]
    old = ReferenceClip(cfg.reference.parent.parent / "reference/reference.npz")
    with pytest.raises(ValueError, match="pinned refined"):
        validate_reference_binding(cfg.profile, old)


def test_solver_iteration_budget_validation():
    cfg = HOITrainConfig()
    assert cfg.solver_iterations == 50
    cfg.solver_iterations = 0
    with pytest.raises(ValueError, match="Solver iteration"):
        cfg.validate()


def test_explicit_resume_only(tmp_path):
    cfg = HOITrainConfig(resume=tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="Explicit resume"):
        cfg.validate()
    cfg.resume = None
    cfg.initial_wm = tmp_path / "old.pt"
    with pytest.raises(ValueError, match="starts from scratch"):
        cfg.validate()
    cfg.initial_wm = None
    cfg.validate()
    with pytest.raises(FileNotFoundError):
        load_play_source(PlayConfig(ckpt=tmp_path / "missing.pt"))


def mock_loop(cfg, monkeypatch, *, hoi):
    """Run the REAL Trainer.run using simulated calls, with no models/physics/optimizers."""
    events = []
    t = object.__new__(HOITrainer if hoi else Trainer)
    t.cfg = cfg
    t.progress = StartupProgress()
    t.iteration = t.start_iteration = t.num_updates = 0
    t._utd_acc = 0.0
    t.is_main = True
    t.world = 1
    t.max_step_reward = 0.18
    t.ep_reward_sum = torch.zeros(1)
    t.ep_len = torch.zeros(1)
    t.writer = Mock()
    t.save = lambda *a: events.append("save")
    t.act = lambda obs: events.append("actor_interaction") or torch.zeros(1, 29)

    def step(action):
        events.append("collect")
        return (
            {},
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )

    t.env = SimpleNamespace(reset=lambda: ({}, {}), step=step)

    def wm_update(buffer):
        events.append("wm")
        return {"loss": 1.0}

    def policy_update(buffer):
        events.extend(("actor_bptt", "critic_td_lambda"))
        return {"loss": 1.0}

    t.wm_trainer = SimpleNamespace(set_lr=lambda _: None, update=wm_update)
    t.policy_trainer = SimpleNamespace(set_lr=lambda _: None, update=policy_update)
    t.buffer = SimpleNamespace()
    if hoi:
        t.log_dir = Path(".")
        t.device = "cpu"
        t.learning_safety = SimpleNamespace(steps={}, gradient_norms={})
        monkeypatch.setattr("supermimic.hoi.training_metrics.append_json", Mock())
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 0)
        t.updates_ready = lambda: t.progress.ready(cfg, t.iteration, 64, 64)
        t.write_startup_status = lambda: None
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    Trainer.run(t)
    return t, events


def test_original_loop_stage_order_and_robot_defaults(monkeypatch):
    cfg = HOITrainConfig()
    cfg.schedule.total_iterations = 5
    cfg.schedule.warmup_iterations = 2
    cfg.schedule.updates_per_env_step = 1
    cfg.startup.wm_only_updates = 2
    t, events = mock_loop(cfg, monkeypatch, hoi=True)
    assert HOITrainer.run is Trainer.run
    assert [e for e in events if e not in ("actor_interaction", "collect", "save")] == [
        "wm",
        "wm",
        "wm",
        "actor_bptt",
        "critic_td_lambda",
        "wm",
        "actor_bptt",
        "critic_td_lambda",
    ]
    assert t.progress.state_dict() == {
        "phase": "alternating",
        "wm_updates": 4,
        "alternating_updates": 2,
        "waiting_since": None,
    }
    robot = TrainConfig()
    assert (robot.env.num_envs, robot.policy.goal_dim, robot.schedule.updates_per_env_step) == (
        4096,
        680,
        0.4,
    )
    robot.schedule.total_iterations = 5
    robot.schedule.warmup_iterations = 2
    robot.schedule.updates_per_env_step = 1
    t, events = mock_loop(robot, monkeypatch, hoi=False)
    assert t.num_updates == 4
    assert events.count("wm") == events.count("actor_bptt") == events.count("critic_td_lambda") == 4


@pytest.mark.parametrize("updates", [0, 1, 3, 5])
def test_phase_resume_without_restarting_bootstrap(updates):
    cfg = HOITrainConfig()
    cfg.startup.wm_only_updates = 3
    cfg.schedule.warmup_iterations = 0
    progress = StartupProgress()
    for _ in range(updates):
        assert progress.ready(cfg, 20, 40, 40)
        progress.completed(cfg)
    restored = StartupProgress.from_state(progress.state_dict(), cfg, updates)
    assert restored == progress
    assert restored.ready(cfg, 20, 40, 40)
    assert (restored.phase == "alternating") == (updates >= 3)
    bad = restored.state_dict()
    bad["wm_updates"] += 1
    with pytest.raises(ValueError):
        StartupProgress.from_state(bad, cfg, updates)


def test_missing_windows_reports_concrete_reason():
    cfg = HOITrainConfig()
    p = StartupProgress()
    assert not p.ready(cfg, 1, 0, 8)
    with pytest.raises(RuntimeError, match="windows8=0/32.*policy_starts=8/32"):
        p.ready(cfg, cfg.startup.max_collect_iterations, 0, 8)


def no_optimizer_steps(monkeypatch):
    def prohibited(*args, **kwargs):
        pytest.fail("Physical smoke must never call an optimizer")

    monkeypatch.setattr(torch.optim.Adam, "step", prohibited)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native HOI")
def test_actual_scratch_constructor_never_loads_weights(tmp_path, monkeypatch):
    no_optimizer_steps(monkeypatch)

    def prohibited_load(*args, **kwargs):
        pytest.fail("Scratch must not read any checkpoint")

    monkeypatch.setattr(torch, "load", prohibited_load)
    cfg = HOITrainConfig(log_root=tmp_path, run_name="scratch_construction", check_only=True)
    cfg.env.num_envs = 2
    with SolverGuard() as guard:
        t = HOITrainer(cfg, guard)
        try:
            assert t.iteration == t.start_iteration == t.num_updates == t.buffer.total_rows == 0
            assert not t.buffer.valid.any()
            assert all(not opt.state for opt in checkpoint.optimizers(t).values())
            assert not t.wm.dynamics.mlp.net[-1].weight.any()
            obs, _ = t.env.reset()
            assert obs["policy"].shape == (2, 1080) and obs["goal"].shape == (2, 830)
            assert t.buffer.total_rows == 0
            assert t.recorder.cache.states.shape == (840, 84)
            assert t.identity["reference_sha256"] == REFINED_SHA256
            assert t.reset_geometry["status"] == "UNRESOLVED_RESET_OVERLAP"
            assert t.reset_geometry["max_penetration_m"] > 0.04
            assert t.recorder.registry.hashes(t.recorder.clip) == (REFINED_SHA256, REFINED_SHA256)
            torch.testing.assert_close(
                t.recorder.io.read(), t.recorder.cache.frames(t.recorder.frame), atol=1e-6, rtol=0
            )
            assert not list(t.log_dir.glob("*.pt"))
        finally:
            t.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native HOI")
def test_zero_update_smoke_resume_and_policy_play(tmp_path, monkeypatch):
    no_optimizer_steps(monkeypatch)
    cfg = HOITrainConfig(log_root=tmp_path, run_name="untrained_test", execution="integration")
    cfg.env.num_envs = 2
    cfg.env.sampling_mode = "start"
    cfg.env.episode_length_s = 0.24
    cfg.env.replay_capacity = 64
    cfg.reset_frames = (0, 830)
    cfg.schedule.total_iterations = 16
    cfg.schedule.updates_per_env_step = 0
    cfg.schedule.log_interval = cfg.schedule.checkpoint_interval = 8
    cfg.policy_trainer.batch_size = cfg.wm_trainer.batch_size = 4
    traces = []
    with SolverGuard() as guard:
        t = HOITrainer(cfg, guard)
        try:
            step = t.env.step

            def record(action):
                result = step(action)
                traces.append(checkpoint.cpu_tree(t.recorder.last))
                return result

            t.env.step = record
            t.run()
            assert t.num_updates == 0 and t.learning_safety.steps == {
                "wm": 0,
                "actor": 0,
                "critic": 0,
            }
            assert guard.checked_world_substeps == 16 * 2 * 4
            mask = t.buffer.window_mask(8)
            assert mask.any()
            from supermimic.hoi.training_metrics import prediction_report

            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
            prediction = prediction_report(t, n=4)
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
            # Scratch WM has zero residual: the reported baseline must match it.
            assert prediction["errors"]["learned"] == prediction["errors"]["zero_residual_prior"]
            batch = t.buffer.sample_windows(4, 8)
            assert (batch["episode_id"] == batch["episode_id"][0]).all()
            assert torch.equal(batch["states"][1:], batch["next_state"])
            assert not batch["transition_terminal"][:-1].any()
            final = checkpoint.snapshot(t)
            assert final["artifact_kind"] == "untrained/test artifact"
        finally:
            t.close()
    resume = copy.deepcopy(cfg)
    resume.run_name = "resumed_untrained_test"
    resume.resume = tmp_path / cfg.run_name / "hoi_0000008.pt"
    with SolverGuard() as guard:
        t = HOITrainer(resume, guard)
        try:
            assert t.iteration == 8 and t.env.resume_pending
            assert t.progress.phase == "collect"
            saved = torch.load(resume.resume, weights_only=True)
            torch.testing.assert_close(
                t.recorder.state.cpu(), saved["runtime"]["recorder"]["state"], atol=0, rtol=0
            )
            t.run()
            assert t.iteration == 16 and t.num_updates == 0
            torch.testing.assert_close(
                t.recorder.last["next_state"].cpu(), traces[-1]["next_state"], atol=1e-4, rtol=0
            )
            bad = copy.deepcopy(final)
            bad["startup"]["phase"] = "alternating"
            with pytest.raises(ValueError, match="phase"):
                checkpoint.restore(t, bad)
            with pytest.raises(FileExistsError):
                checkpoint.save_exclusive(t, resume.resume)
        finally:
            t.close()
    source = tmp_path / cfg.run_name / "hoi_0000016.pt"
    with pytest.raises(ValueError, match="no trained Actor"):
        load_play_source(PlayConfig(ckpt=source))
    result = play(
        PlayConfig(
            ckpt=source,
            output_dir=tmp_path / "tests/policy_play_untrained",
            max_steps=8,
            allow_untrained_test_artifact=True,
        )
    )
    assert result["mode"] == "physical_actor_policy_play" and result["steps"] <= 8
    assert result["optimizer_steps_during_play"] == {"wm": 0, "actor": 0, "critic": 0}
    assert result["records"][0]["frame"] == 1
    assert result["task_success"] is None
    (tmp_path / "smoke_summary.json").write_text(
        json.dumps(
            {
                "artifact_kind": "untrained/test artifact",
                "scratch_control_steps": 16,
                "resume_control_steps": 8,
                "policy_play_control_steps": result["steps"],
                "physical_data_optimizer_updates": 0,
                "source": str(source),
            },
            indent=2,
        )
    )
