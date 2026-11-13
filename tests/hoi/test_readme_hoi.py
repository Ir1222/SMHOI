"""Parse the published commands and construct the scratch route without training."""

import copy
import re
import shlex
from pathlib import Path

import pytest
import torch
import tyro

from supermimic.hoi.play import PlayConfig
from supermimic.hoi.train import execute
from supermimic.hoi.training_config import HOITrainConfig
from supermimic.training.loop import Trainer

ROOT = Path(__file__).resolve().parents[2]


def readme_commands():
    blocks = re.findall(r"```bash\n(.*?)```", (ROOT / "README_HOI.md").read_text(), re.DOTALL)
    return [shlex.split(b.replace("\\\n", " ")) for b in blocks if "-m supermimic.hoi." in b]


def training_configs():
    return [
        tyro.cli(HOITrainConfig, args=c[c.index("supermimic.hoi.train") + 1 :])
        for c in readme_commands()
        if "supermimic.hoi.train" in c
    ]


def test_readme_commands_parse_and_resume_contract():
    scratch, resume, check = training_configs()
    scratch.validate()
    check.validate()
    assert scratch.resume is scratch.initial_wm is None
    assert scratch.env.num_envs == 64 and scratch.schedule.total_iterations == 320000
    assert resume.resume == Path("logs/hoi_refined_scratch_v1/hoi_0004000.pt")
    from supermimic.hoi.training_checkpoint import config_contract

    assert config_contract(scratch) == config_contract(resume)
    shell = (ROOT / "configs/hoi/refined_full.sh").read_text().split("exec env", 1)[1]
    command = [word for word in shlex.split(shell.replace("\\\n", " ")) if word != "$@"]
    scripted = tyro.cli(HOITrainConfig, args=command[command.index("supermimic.hoi.train") + 1 :])
    assert config_contract(scripted) == config_contract(scratch)
    for command in readme_commands():
        if "supermimic.hoi.play" in command:
            cfg = tyro.cli(PlayConfig, args=command[command.index("supermimic.hoi.play") + 1 :])
            assert cfg.max_steps == 839 and not cfg.allow_untrained_test_artifact


def test_reference_viewer_binds_same_refined_single_body(monkeypatch):
    import mujoco

    from supermimic.hoi.reference import ReferenceClip
    from supermimic.hoi.reference_viewer import ReferenceScene
    from supermimic.hoi.training_config import REFINED_SHA256

    def forbidden(*args, **kwargs):
        pytest.fail("Reference viewer must never step physical simulation")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    cfg = HOITrainConfig()
    scene = ReferenceScene(ReferenceClip(cfg.reference), display_only=True, profile=cfg.profile)
    assert scene.clip.manifest["sha256"] == REFINED_SHA256
    assert len(scene.asset["object"]["geoms"]) == 1
    for frame in (0, 406, 460, 839):
        scene.set_frame(frame)
        assert scene.frame == frame


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native construction needs CUDA")
def test_readme_scratch_constructs_without_run_or_weights(tmp_path, monkeypatch):
    cfg = copy.deepcopy(training_configs()[0])
    # Same CLI/config path and networks; physical validation is bounded to two worlds.
    cfg.env.num_envs = 2
    cfg.log_root, cfg.run_name = tmp_path, "readme_construct_only"
    cfg.check_only = True

    def forbidden(*args, **kwargs):
        pytest.fail("README construction check must not run/train/load old weights")

    monkeypatch.setattr(Trainer, "run", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    execute(cfg)
    out = tmp_path / cfg.run_name
    assert (out / "config.json").is_file() and (out / "reset_geometry.json").is_file()
    assert not list(out.glob("*.pt"))
