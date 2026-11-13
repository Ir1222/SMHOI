"""P7.0.4 bounded lifecycle checks; no physical-data optimizer updates."""

import ast
import copy
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from p5_fixtures import TEST_CFG

from supermimic.config import TrainConfig
from supermimic.hoi import training_checkpoint as checkpoint
from supermimic.hoi.training import HOITrainer
from supermimic.hoi.training_config import HOITrainConfig
from supermimic.hoi.training_safety import SolverGuard, finite
from supermimic.training.loop import Trainer

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "d3c3db10bce01b6da4bb5c18d759787680e14afc"


def evidence(name, value):
    directory = os.environ.get("P704_EVIDENCE_DIR")
    if directory:
        path = Path(directory) / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def config(tmp_path, name="continuous"):
    cfg = HOITrainConfig(
        log_root=tmp_path,
        run_name=name,
        profile="p6-original",
        reference=ROOT / "outputs/hoi/asset_audit/part_b/reference/reference.npz",
        execution="integration",
        reward=TEST_CFG,
        sampler_g_max=18.2,
        reset_frames=(250, 830),
    )
    cfg.env.action_noise_std = 0.03
    cfg.env.num_envs = 2
    cfg.env.sampling_mode = "start"
    cfg.schedule.total_iterations = 16
    cfg.schedule.updates_per_env_step = 0
    cfg.env.obs_noise = cfg.policy_trainer.obs_noise = True
    cfg.env.episode_length_s = 0.24
    cfg.schedule.checkpoint_interval = 8
    cfg.policy_trainer.horizon = 2
    cfg.policy_trainer.batch_size = 2
    cfg.policy_trainer.critic_minibatches = 2
    cfg.wm_trainer.batch_size = 2
    return cfg


def test_shared_run_and_robot_checkpoint_methods_are_identical():
    original = subprocess.check_output(
        ["git", "show", f"{BASELINE}:src/supermimic/training/loop.py"], cwd=ROOT
    ).decode()
    current = (ROOT / "src/supermimic/training/loop.py").read_text()

    def methods(source):
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
        }

    a, b = methods(original), methods(current)
    for name in ("act", "save", "load"):
        assert a[name] == b[name]
    assert HOITrainer.run is Trainer.run
    robot = TrainConfig()
    assert robot.env.num_envs == 4096 and robot.policy.goal_dim == 680
    assert robot.schedule.updates_per_env_step == 0.4


@pytest.mark.parametrize("change", ["updates", "budget", "worlds", "approval", "reward", "noise"])
def test_scope_and_required_config_fail_closed(tmp_path, change):
    cfg = config(tmp_path)
    if change == "updates":
        cfg.schedule.updates_per_env_step = 1
    elif change == "budget":
        cfg.schedule.total_iterations = 65
    elif change == "worlds":
        cfg.env.num_envs = 5
    elif change == "approval":
        cfg.initial_wm = tmp_path / "retired_external_wm.pt"
    elif change == "reward":
        cfg.reward = None
    else:
        cfg.policy_trainer.obs_noise = False
    with pytest.raises(ValueError):
        cfg.validate()


def test_nonfinite_guard_checks_all_worlds():
    for world in range(4):
        state = torch.ones(4, 84)
        state[world, -1] = float("nan")
        with pytest.raises(FloatingPointError):
            finite("state", state)


def traced_steps(trainer, target):
    original = trainer.env.step

    def step(action):
        result = original(action)
        r = trainer.recorder
        torch.testing.assert_close(r.history_states[-1], r.io.read(), atol=1e-6, rtol=0)
        assert torch.equal(r.state, r.history_states[-1])
        ids = torch.arange(trainer.env.num_envs, device=trainer.device)
        row = r.last["row"]
        target.append(
            checkpoint.cpu_tree(
                {
                    "transition": r.last,
                    "observations": result[0],
                    "reward": result[1],
                    "terminated": result[2],
                    "timeout": result[3],
                    "masks": {k: v[ids, row] for k, v in trainer.buffer.fields.items()},
                    "terms": r.last_terms,
                }
            )
        )
        return result

    trainer.env.step = step


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def differences(a, b, path=""):
    result = {}
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape and a.dtype == b.dtype
        if a.is_floating_point():
            mask = torch.isfinite(a) & torch.isfinite(b)
            assert torch.equal(torch.isfinite(a), torch.isfinite(b))
            result[path] = float((a[mask] - b[mask]).abs().max()) if mask.any() else 0.0
        else:
            assert_tree_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            result.update(differences(a[k], b[k], path + "/" + str(k)))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            result.update(differences(x, y, path + "/" + str(i)))
    else:
        assert a == b
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Physical CUDA integration required")
def test_physical_collection_masks_backwards_and_resume(tmp_path):
    cfg = config(tmp_path)
    traces = []
    with SolverGuard() as guard:
        trainer = HOITrainer(cfg, guard)
        try:
            traced_steps(trainer, traces)
            trainer.run()  # The original lifecycle, with zero update budget.
            assert trainer.iteration == 16 and trainer.num_updates == 0
            assert trainer.learning_safety.steps == {"wm": 0, "actor": 0, "critic": 0}
            assert guard.checked_world_substeps == 16 * 4 * 2
            first = traces[0]
            assert first["transition"]["frame"].tolist() == [250, 830]
            assert first["observations"]["policy"].shape == (2, 1080)
            assert first["observations"]["goal"].shape == (2, 830)
            noise = torch.stack(
                [t["transition"]["executed_action"] - t["transition"]["action"] for t in traces]
            )
            assert noise.std() > 0.01
            clip_count = timeout_count = 0
            for row in traces:
                masks, tr = row["masks"], row["transition"]
                reference = trainer.recorder.registry.gather(
                    torch.zeros(2, dtype=torch.long, device=trainer.device),
                    tr["frame"].to(trainer.device) + 1,
                )
                direct = trainer.recorder.reward(
                    tr["next_state"].to(trainer.device),
                    tr["action"].to(trainer.device),
                    tr["previous_action"].to(trainer.device),
                    reference.robot,
                    reference.box,
                )["total"].cpu()
                torch.testing.assert_close(row["reward"], direct, atol=5e-6, rtol=5e-6)
                assert torch.equal(masks["next_state"], tr["next_state"])
                assert masks["transition_valid"].all()
                expected = ~(masks["physical_failure"] | masks["clip_end"])
                assert torch.equal(masks["bootstrap"], expected)
                assert torch.equal(masks["continuation"], expected)
                torch.testing.assert_close(row["reward"], tr["reward"], atol=0, rtol=0)
                torch.testing.assert_close(
                    masks["terminal_value"], tr["value"] * expected, atol=0, rtol=0
                )
                total = sum(row["terms"][k] for k in trainer.policy_trainer.step_fn.term_names)
                torch.testing.assert_close(total, row["reward"], atol=5e-6, rtol=5e-6)
                clip_count += int(masks["clip_end"].sum())
                timeout_count += int(masks["timeout"].sum())
            assert clip_count >= 1 and timeout_count >= 1
            final = checkpoint.snapshot(trainer)
            # The saved RNG must be compared before backward-only probes consume it.
            batch = trainer.buffer.sample_windows(4, 8)
            loss, _ = trainer.wm_trainer.loss_on(batch)
            loss.backward()
            wm_gn = sum(
                p.grad.square().sum() for p in trainer.wm.parameters() if p.grad is not None
            ).sqrt()
            finite("WM backward", wm_gn)
            trainer.wm.zero_grad(set_to_none=True)
            from supermimic.imagination.policy_trainer import frozen_parameters

            start = trainer.buffer.sample_starts(2, 10)
            assert (start.remaining_steps > 0).all()
            with frozen_parameters(trainer.wm, trainer.critic):
                aloss, _, rollout, inputs, xi = trainer.policy_trainer.actor_loss(start)
                aloss.backward()
            actor_gn = sum(
                p.grad.square().sum() for p in trainer.policy.parameters() if p.grad is not None
            ).sqrt()
            assert all(p.grad is None for p in trainer.wm.parameters())
            X, Y, _, targets = trainer.policy_trainer.critic_batch(rollout, xi, inputs)
            closs = (trainer.critic(*X) - Y).square().mean()
            closs.backward()
            finite("Actor/Critic backward", aloss, actor_gn, closs, targets)
            stats = {
                "transitions": 32,
                "clip_ends": clip_count,
                "timeouts": timeout_count,
                "executed_noise_std": float(noise.std()),
                "wm_loss": float(loss.detach()),
                "wm_gradient_norm": float(wm_gn),
                "actor_loss": float(aloss.detach()),
                "actor_gradient_norm": float(actor_gn),
                "critic_loss": float(closs.detach()),
                "guard": guard.evidence(),
                "optimizer_steps": trainer.learning_safety.steps,
            }
            from supermimic.hoi.training_diagnostics import retained_contact_report

            stats["physical_diagnostics"] = retained_contact_report(trainer)
        finally:
            trainer.close()
    resumed = []
    cfg2 = copy.deepcopy(cfg)
    cfg2.run_name = "resumed"
    cfg2.resume = tmp_path / "continuous/hoi_0000008.pt"
    with SolverGuard() as guard:
        trainer = HOITrainer(cfg2, guard)
        try:
            assert trainer.iteration == 8 and trainer.env.resume_pending
            saved = torch.load(cfg2.resume, weights_only=True)
            before = checkpoint.snapshot(trainer)
            for key in ("models", "optimizers", "runtime", "counters", "rng", "optimizer_steps"):
                assert_tree_equal(saved[key], before[key])
            traced_steps(trainer, resumed)
            trainer.run()
            drift = differences(traces[8:], resumed)
            # Native GPU constraint reductions need not be bit-deterministic.
            # Record every difference; exact restoration is asserted above.
            physical = {
                k: v
                for k, v in drift.items()
                if "/transition/" in k or "/observations/" in k or k.endswith("/reward")
            }
            assert max(physical.values()) < 1e-4
            after = checkpoint.snapshot(trainer)
            for key in ("models", "optimizers", "counters", "rng", "optimizer_steps"):
                assert_tree_equal(final[key], after[key])
            assert any(a["reason"] == "resume_after_recapture" for a in guard.attestations)
            stats["resume"] = {
                "restore": "bit-exact",
                "continued_steps": 8,
                "trajectory_bit_exact": not any(drift.values()),
                "physical_max_absolute_difference": max(physical.values()),
                "trajectory_absolute_differences": drift,
                "guard": guard.evidence(),
            }
            evidence("physical_resume", stats)
            # Reject corruption before changing models, physical state or RNG.
            baseline = checkpoint.snapshot(trainer)
            for kind in ("identity", "runtime", "model", "optimizer", "rng", "overflow"):
                bad = copy.deepcopy(baseline)
                if kind == "identity":
                    bad["identity"]["reference_sha256"] = "0" * 64
                elif kind == "runtime":
                    bad["runtime"]["recorder"]["state"] = torch.zeros(1)
                elif kind == "model":
                    next(iter(bad["models"]["policy"].values())).flatten()[0] = float("nan")
                elif kind == "optimizer":
                    bad["optimizers"]["actor"]["state"][9999] = {}
                elif kind == "rng":
                    bad["rng"]["torch"] = torch.zeros(4, dtype=torch.uint8)
                else:
                    bad["runtime"]["simulator"]["overflow"][1] = 1
                with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
                    checkpoint.restore(trainer, bad)
                after_bad = checkpoint.snapshot(trainer)
                for key in ("models", "optimizers", "runtime", "counters", "rng"):
                    assert_tree_equal(baseline[key], after_bad[key])
            path = trainer.log_dir / "exclusive.pt"
            checkpoint.save_exclusive(trainer, path)
            digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            with pytest.raises(FileExistsError):
                checkpoint.save_exclusive(trainer, path)
            assert __import__("hashlib").sha256(path.read_bytes()).hexdigest() == digest
            evidence(
                "checkpoint_rejections",
                {"cases": 6, "atomic_rejection": True, "no_overwrite": True},
            )

            # Isolated serialization probe, after all physical comparisons:
            # artificial constant gradients only; no collected state/reward target.
            def synthetic_adam_step():
                for optimizer in checkpoint.optimizers(trainer).values():
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        for parameter in group["params"]:
                            parameter.grad = torch.full_like(parameter, 1e-5)
                    optimizer.step()

            synthetic_adam_step()
            path = trainer.log_dir / "synthetic_populated_optimizer_only.pt"
            checkpoint.save_exclusive(trainer, path)
            synthetic_adam_step()
            expected = checkpoint.snapshot(trainer)
            checkpoint.load(trainer, path)
            synthetic_adam_step()
            actual = checkpoint.snapshot(trainer)
            for key in ("models", "optimizers", "runtime", "rng", "counters", "optimizer_steps"):
                assert_tree_equal(expected[key], actual[key])
            evidence(
                "full_populated_checkpoint",
                {
                    "restore_and_next_Adam_step": "bit-exact",
                    "models": ["JointWM", "ActorB", "CriticB+"],
                    "artificial_gradient": 1e-5,
                    "effective_steps_per_model": 2,
                    "replayed_steps_per_model": 1,
                    "physical_data_used": False,
                    "no_subsequent_physical_execution": True,
                },
            )
        finally:
            trainer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Physical CUDA integration required")
def test_partial_reset_failures_and_all_world_safety(tmp_path):
    cfg = config(tmp_path, "safety")
    cfg.env.obs_noise = cfg.policy_trainer.obs_noise = False
    with SolverGuard() as guard:
        trainer = HOITrainer(cfg, guard)
        try:
            env, r = trainer.env, trainer.recorder
            env.reset()
            env.step(torch.zeros(2, 29, device=trainer.device))
            other = checkpoint.cpu_tree(
                {
                    "state": r.io.read()[1],
                    "frame": r.frame[1],
                    "history": r.history_observations[:, 1],
                    "episode": r.buffer.ep_counter[1],
                }
            )
            env.reset(env_ids=torch.tensor([0], device=trainer.device))
            assert_tree_equal(
                other,
                checkpoint.cpu_tree(
                    {
                        "state": r.io.read()[1],
                        "frame": r.frame[1],
                        "history": r.history_observations[:, 1],
                        "episode": r.buffer.ep_counter[1],
                    }
                ),
            )
            assert (r.history_states[:, 0] == r.state[0]).all()
            assert not r.history_actions[:, 0].any()
            assert (r.history_observations[:, 0] == r.history_observations[0, 0]).all()
            env.raw.action_manager.process_action(torch.zeros(2, 29, device=trainer.device))
            # Fault injection is a monitor unit probe, never a physics step with altered state.
            for world in range(2):
                count = trainer.buffer.total_rows
                action = torch.zeros(2, 29, device=trainer.device)
                action[world, 0] = float("nan")
                with pytest.raises(FloatingPointError):
                    env.step(action)
                assert trainer.buffer.total_rows == count
                native = env.raw.sim.data
                native.overflow[world] = 1
                with pytest.raises(RuntimeError, match="overflow"):
                    guard.monitor(env.raw)
                native.overflow[world] = 0
                import warp as wp

                # Explicit native tensor view, independent of mjlab's TorchArray proxy.
                velocity = wp.to_torch(env.raw.sim.wp_data.qvel)
                original = velocity[world, 0].clone()
                velocity[world, 0] = float("nan")
                with pytest.raises(FloatingPointError):
                    guard.monitor(env.raw)
                velocity[world, 0] = original
            from mujoco_warp._src import solver

            factory = solver._linesearch_iterative_kernel
            try:
                solver._linesearch_iterative_kernel = None
                with pytest.raises(RuntimeError, match="guard"):
                    guard.attest(env.raw, "deliberate_guard_loss")
            finally:
                solver._linesearch_iterative_kernel = factory
            evidence(
                "safety",
                {
                    "worlds_checked": 2,
                    "action_nan_rejected": 2,
                    "state_nan_rejected": 2,
                    "overflow_rejected": 2,
                    "guard_loss_rejected": True,
                    "partial_reset_isolated": True,
                },
            )
        finally:
            trainer.close()
    cfg = config(tmp_path, "terminal_overlap")
    cfg.reset_frames = (838, 250)
    cfg.env.episode_length_s = 0.02
    cfg.box_min_height = 10.0  # Explicit test-only failure threshold; no state injection.
    with SolverGuard() as guard:
        trainer = HOITrainer(cfg, guard)
        try:
            trainer.env.reset()
            _, _, terminated, timeout, _ = trainer.env.step(
                torch.zeros(2, 29, device=trainer.device)
            )
            assert terminated.all() and timeout.all()
            ids = torch.arange(2, device=trainer.device)
            f = {k: v[ids, trainer.recorder.last["row"]] for k, v in trainer.buffer.fields.items()}
            assert f["physical_failure"].all() and f["object_failure"].all()
            assert f["clip_end"].tolist() == [True, False]
            assert not f["bootstrap"].any() and not f["continuation"].any()
            assert not f["terminal_value"].any()
            assert not torch.equal(f["next_state"], trainer.recorder.io.read())
            assert not trainer.buffer.window_mask(8).any()
            evidence(
                "terminal_overlap",
                {
                    "physical_failures": 2,
                    "timeouts": 2,
                    "clip_ends": 1,
                    "bootstrap_count": 0,
                    "pre_reset_state_preserved": True,
                },
            )
        finally:
            trainer.close()


def test_populated_adam_resume_and_learning_safety():
    from types import SimpleNamespace

    from supermimic.hoi.training_safety import LearningSafety

    torch.manual_seed(1704)
    models = [torch.nn.Linear(3, 2) for _ in range(3)]
    opts = [torch.optim.Adam(m.parameters(), lr=0.001) for m in models]
    wm = SimpleNamespace(wm=models[0], opt=opts[0], loss_on=lambda: torch.tensor(1.0))
    policy = SimpleNamespace(
        policy=models[1],
        critic=models[2],
        opt=opts[1],
        critic_opt=opts[2],
        actor_loss=lambda: torch.tensor(1.0),
        critic_batch=lambda: torch.tensor(1.0),
    )
    safety = LearningSafety(wm, policy)

    def step():
        for model, optimizer in zip(models, opts):
            optimizer.zero_grad(set_to_none=True)
            model(torch.tensor([[0.2, -0.4, 0.7]])).square().sum().backward()
            optimizer.step()

    step()
    saved_m = checkpoint.cpu_tree([m.state_dict() for m in models])
    saved_o = checkpoint.cpu_tree([o.state_dict() for o in opts])
    step()
    expected_m = checkpoint.cpu_tree([m.state_dict() for m in models])
    expected_o = checkpoint.cpu_tree([o.state_dict() for o in opts])
    for i, (m, o) in enumerate(zip(models, opts)):
        checkpoint.validate_optimizer(saved_o[i], o, str(i))
        m.load_state_dict(saved_m[i])
        o.load_state_dict(copy.deepcopy(saved_o[i]))
    step()
    assert_tree_equal(expected_m, checkpoint.cpu_tree([m.state_dict() for m in models]))
    assert_tree_equal(expected_o, checkpoint.cpu_tree([o.state_dict() for o in opts]))
    count = dict(safety.steps)
    for model, optimizer in zip(models, opts):
        optimizer.zero_grad(set_to_none=True)
        with pytest.raises(FloatingPointError, match="gradient"):
            (model(torch.ones(1, 3)) * float("nan")).sum().backward()
        optimizer.zero_grad(set_to_none=True)
        next(model.parameters()).grad = torch.full_like(next(model.parameters()), float("inf"))
        with pytest.raises(FloatingPointError):
            optimizer.step()
    with pytest.raises(FloatingPointError):
        wm.numerics_check("loss", torch.tensor(float("nan")))
    assert safety.steps == count
    evidence(
        "synthetic_optimizer_resume",
        {
            "models": 3,
            "populated_Adam_restore_and_next_step": "bit-exact",
            "effective_steps": 2,
            "replayed_steps": 1,
            "physical_training_data_used": False,
            "nonfinite_gradient_rejected_for_all_models": True,
        },
    )


def test_inherited_update_interfaces_with_one_synthetic_batch():
    from types import SimpleNamespace

    from p7_policy_fixtures import fixture, start_batch

    from supermimic.hoi.training_safety import LearningSafety
    from supermimic.hoi.wm_trainer import JointWorldModelTrainer
    from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer

    fx = fixture(dtype=torch.float32, coupled=False, horizon=1)
    st = start_batch(fx, (300,))
    trainer = JointWorldModelTrainer(
        fx.wm, [fx.clip], WMTrainerCfg(batch_size=1, batches_per_update=1, compile=None)
    )
    assert JointWorldModelTrainer.update is WorldModelTrainer.update
    fx.trainer.cfg.critic_epochs = fx.trainer.cfg.critic_minibatches = 1
    safety = LearningSafety(trainer, fx.trainer)
    states = st.state[None].expand(9, -1, -1).clone()
    batch = {
        "states": states,
        "next_state": states[1:],
        "frame": torch.arange(300, 308)[:, None],
        "clip": torch.zeros(8, 1, dtype=torch.long),
        "env_id": torch.zeros(8, 1, dtype=torch.long),
        "episode_id": torch.zeros(8, 1, dtype=torch.long),
        "transition_valid": torch.ones(8, 1, dtype=torch.bool),
        "transition_terminal": torch.zeros(8, 1, dtype=torch.bool),
        "physical_failure": torch.zeros(8, 1, dtype=torch.bool),
        "a_exec": torch.zeros(8, 1, 29),
        "xi": st.xi,
    }
    source = SimpleNamespace(sample_windows=lambda n, k: batch, sample_starts=lambda n, h: st)
    wm_metrics = trainer.update(source)
    policy_metrics = fx.trainer.update(source)
    assert safety.steps == {"wm": 1, "actor": 1, "critic": 1}
    finite("synthetic update metrics", wm_metrics, policy_metrics)
    evidence(
        "inherited_synthetic_updates",
        {
            "wm": wm_metrics,
            "policy": policy_metrics,
            "optimizer_steps": safety.steps,
            "scope": "one artificial batch, no physical Replay or training run",
        },
    )
