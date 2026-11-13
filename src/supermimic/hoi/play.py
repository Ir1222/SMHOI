"""Physical Actor playback in the training Robot–Box scene (never reference stepping)."""

import json
import time
import traceback
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path

import torch
import tyro

from supermimic.hoi import training_checkpoint
from supermimic.hoi.policy_features import feature_schema
from supermimic.hoi.reference import sha256
from supermimic.hoi.training import HOITrainer
from supermimic.hoi.training_config import HOITrainConfig
from supermimic.hoi.training_safety import SolverGuard, finite
from supermimic.hoi.training_scene import CALIBRATION


@dataclass
class PlayConfig:
    ckpt: Path
    output_dir: Path = Path("logs/hoi_policy_play")
    device: str = "cuda:0"
    max_steps: int = 839
    headless: bool = True
    # Only for bounded interface tests; never describe this artifact as a trained policy.
    allow_untrained_test_artifact: bool = False


def _restore_config(default, saved):
    known = {f.name for f in fields(default)}
    if set(saved) - known:
        raise ValueError(f"Unknown checkpoint config fields: {set(saved) - known}")
    values = {}
    for key, value in saved.items():
        baseline = getattr(default, key)
        if is_dataclass(baseline):
            value = _restore_config(baseline, value)
        elif isinstance(baseline, Path) or key in (
            "resume",
            "initial_wm",
            "npz_dir",
            "manifest",
            "library",
        ):
            value = Path(value) if value is not None else None
        elif isinstance(baseline, tuple):
            value = tuple(value)
        values[key] = value
    return replace(default, **values)


def load_play_source(cfg):
    if not cfg.ckpt.is_file():
        raise FileNotFoundError(f"Explicit HOI policy checkpoint does not exist: {cfg.ckpt}")
    if not 1 <= cfg.max_steps <= 839:
        raise ValueError("Frame-0 policy play supports 1..839 control steps")
    saved = torch.load(cfg.ckpt, map_location="cpu", weights_only=True)
    if (
        saved.get("schema") != training_checkpoint.SCHEMA
        or saved.get("features") != feature_schema()
    ):
        raise ValueError(
            "Expected this project's complete HOI checkpoint, not a robot-only/migration checkpoint"
        )
    untrained = saved.get("optimizer_steps", {}).get("actor", 0) == 0
    if saved.get("artifact_kind") == "untrained/test artifact" or untrained:
        if not cfg.allow_untrained_test_artifact:
            raise ValueError(
                "Checkpoint has no trained Actor; only explicit bounded test-artifact playback is allowed"
            )
        if cfg.max_steps > 16 or "tests" not in cfg.output_dir.resolve().parts:
            raise ValueError(
                "Untrained test playback requires <=16 steps and an output directory under tests/"
            )
    train = _restore_config(HOITrainConfig(), saved["config"])
    train.resume = train.initial_wm = None  # Playback never restores optimizer/runtime or trains.
    train.check_only = True
    train.log_root, train.run_name = cfg.output_dir.parent, cfg.output_dir.name
    train.device = cfg.device
    train.env.num_envs = 1
    train.env.sampling_mode = "start"
    train.env.action_noise_std = 0.0
    train.env.obs_noise = train.policy_trainer.obs_noise = False
    train.env.episode_length_s = 16.8
    train.reset_frames = (0,)
    return train, saved


def validate_play_identity(saved, actual):
    # Different world count is intentional for evaluation. All physical/reference,
    # algorithm/code and dependency identities stay strict.
    expected = saved.get("identity", {})
    for key in actual.keys() - {"native_model"}:
        if expected.get(key) != actual[key]:
            raise ValueError(f"Incompatible policy play identity: {key}")


def execute(cfg):
    train, saved = load_play_source(cfg)
    from supermimic.hoi.contact_validation import actual_contact

    calibration = json.loads(CALIBRATION.read_text())
    records = []
    result = {
        "mode": "physical_actor_policy_play",
        "checkpoint": str(cfg.ckpt.resolve()),
        "checkpoint_sha256": sha256(cfg.ckpt),
        "artifact_kind": saved["artifact_kind"],
        "actor_optimizer_steps_in_source": saved["optimizer_steps"]["actor"],
        "start_frame": 0,
        "end_reason": "step_limit",
        "task_success": None,
        "optimizer_updates_during_play": 0,
    }
    with SolverGuard() as guard:
        trainer = HOITrainer(train, guard)
        viewer = None
        try:
            validate_play_identity(saved, trainer.identity)
            finite("policy checkpoint", saved["models"]["policy"], saved["models"]["critic"])
            finite("WM evaluation checkpoint", saved["models"]["wm"])
            trainer.wm.load_state_dict(saved["models"]["wm"], strict=True)
            trainer.policy.load_state_dict(saved["models"]["policy"], strict=True)
            trainer.policy.eval()
            # Critic values are diagnostic only; keep timeout/value diagnostics meaningful.
            trainer.critic.load_state_dict(saved["models"]["critic"], strict=True)
            obs, _ = trainer.env.reset()
            if not cfg.headless:
                import mujoco
                import mujoco.viewer

                model = trainer.env.raw.sim.mj_model
                display = mujoco.MjData(model)
                viewer = mujoco.viewer.launch_passive(model, display)
            for _ in range(cfg.max_steps):
                started = time.monotonic()
                with torch.no_grad():
                    action = trainer.act(obs)
                    obs, reward, term, timeout, _ = trainer.env.step(action)
                recorder = trainer.recorder
                row = int(recorder.last["row"][0])
                masks = {
                    k: bool(trainer.buffer.fields[k][0, row])
                    for k in ("robot_failure", "object_failure", "clip_end", "timeout")
                }
                measured = actual_contact(
                    trainer.buffer.fields["force_raw_world"][0, row][None].cpu().numpy(),
                    calibration["thresholds_N_LR"],
                    calibration["occupancy_cutoffs_LR"],
                )
                desired = bool(trainer.buffer.fields["intent_ref"][0, row, 1])
                from supermimic.hoi.training_metrics import reached_geometry_report

                geometry = reached_geometry_report(trainer)
                records.append(
                    {
                        "max_penetration_m": geometry["max_penetration_m"],
                        "worst_contact_pair": geometry["worlds"][0]["worst_pair"],
                        "frame": int(recorder.last["frame"][0]) + 1,
                        "reward": float(reward[0]),
                        "action_l2": float(action.norm()),
                        "box_height_m": float(recorder.last["next_state"][0, 73]),
                        "desired_contact": desired,
                        "actual_contact_LR": measured["hands"][0].tolist(),
                        "contact_missing_diagnostic": desired
                        and not bool(measured["bilateral"][0]),
                        "box_vertical_velocity_m_s": float(recorder.last["next_state"][0, 80]),
                        "box_position_error_m": float(
                            (
                                recorder.last["next_state"][0, 71:74]
                                - recorder.registry.gather(
                                    recorder.clip, recorder.last["frame"] + 1
                                ).box.position[0]
                            ).norm()
                        ),
                        **masks,
                    }
                )
                if viewer is not None:
                    # Copy actual reached state to a display-only CPU buffer. Never write
                    # reference qpos to the simulation or generate a reference-PD action.
                    import mujoco

                    reached = recorder.last["next_state"][0].cpu().numpy()
                    origin = recorder.io.origins[0].cpu().numpy()
                    display.qpos[recorder.io.robot_q.cpu().numpy()] = reached[:36]
                    display.qpos[recorder.io.box_q.cpu().numpy()] = reached[71:78]
                    display.qpos[recorder.io.robot_q[:3].cpu().numpy()] += origin
                    display.qpos[recorder.io.box_q[:3].cpu().numpy()] += origin
                    mujoco.mj_forward(model, display)
                    viewer.sync()
                    if not viewer.is_running():
                        result["end_reason"] = "viewer_closed"
                        break
                    time.sleep(max(0, 0.02 - (time.monotonic() - started)))
                if term[0] or timeout[0]:
                    result["end_reason"] = "+".join(k for k, v in masks.items() if v)
                    break
            if trainer.buffer.window_mask(8).any():
                from supermimic.hoi.training_metrics import prediction_report

                result["wm_prediction"] = prediction_report(trainer)
                result["wm_prediction"]["scope"] = (
                    "Frozen trained WM versus zero-residual prior on this new frame-0 Actor "
                    "play trajectory; no fitting on these samples, single episode only"
                )
        except BaseException as exc:
            result["end_reason"] = (
                "invalid_numerics"
                if isinstance(exc, FloatingPointError)
                else "invalid_solve"
                if any(word in str(exc).lower() for word in ("overflow", "solver", "guard"))
                else "execution_error"
            )
            result["error"] = traceback.format_exc()
            raise
        finally:
            result.update(
                steps=len(records),
                records=records,
                guard=guard.evidence(),
                reset_geometry_status=trainer.reset_geometry["status"],
                optimizer_steps_during_play=trainer.learning_safety.steps,
            )
            (trainer.log_dir / "policy_play.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            if viewer is not None:
                viewer.close()
            trainer.close()
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "guard")}, indent=2))
    return result


def main():
    execute(tyro.cli(PlayConfig))


if __name__ == "__main__":
    main()
