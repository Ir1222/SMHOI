"""P7 software-only fixtures: full approved networks, synthetic coupled dynamics."""

import json
import os
from pathlib import Path

import torch
from p5_fixtures import fixture as p5_fixture
from p5_fixtures import start_batch  # noqa: F401

from supermimic.hoi.imagination import HOIPolicyTrainer
from supermimic.hoi.policy import HOIActor, HOICritic


def fixture(dtype=torch.float64, device="cpu", *, horizon=3, coupled=True):
    fx = p5_fixture(dtype, device, horizon=horizon, coupled=coupled)
    fx.policy = HOIActor().to(device=device, dtype=dtype)
    fx.critic = HOICritic().to(device=device, dtype=dtype)
    fx.trainer = HOIPolicyTrainer(
        fx.wm, fx.policy, fx.critic, fx.registry, fx.reward, fx.trainer.cfg
    )
    return fx


def controlled_fusion(actor):
    """Explicit small nonzero columns, separate from exact-preserving migration."""
    with torch.no_grad():
        w = actor.decoder.net[0].weight[:, 962:]
        w.copy_(
            0.01
            * torch.sin(
                torch.arange(w.numel(), device=w.device, dtype=w.dtype).reshape_as(w) * 0.13
            )
        )


def save_evidence(name, value):
    directory = os.environ.get("P7_POLICY_EVIDENCE_DIR")
    if directory:
        p = Path(directory) / f"{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
