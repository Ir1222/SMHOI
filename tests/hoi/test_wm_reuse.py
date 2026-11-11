"""Compare default shared WM loss and gradients to the accepted P2 source, not a reimplementation."""

import copy
import subprocess
import sys
import types

import torch

from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def test_original_window_loss_and_gradients_unchanged():
    source = subprocess.check_output(
        [
            "git",
            "show",
            "a991ee776b8da4f5f278dbc48947785e7011b5e8:src/supermimic/world_model/world_model.py",
        ],
        text=True,
    )
    module = types.ModuleType("_p3_accepted_robot_wm")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "<accepted-P2-world-model>", "exec"), module.__dict__)  # noqa: S102 -- pinned local accepted source
        torch.manual_seed(19)
        cfg = WorldModelCfg(hidden=(16, 16))
        current = WorldModel(cfg, AnalyticHeads, "cpu")
        with torch.no_grad():
            current.dynamics.mlp.net[-1].weight.normal_(0, 0.01)
        old = module.WorldModel(copy.deepcopy(cfg), AnalyticHeads, "cpu")
        old.load_state_dict(current.state_dict(), strict=True)
        state = torch.randn(9, 4, 71) * 0.1
        state[:, :, 3] = 1
        state[:, :, 3:7] /= state[:, :, 3:7].norm(dim=-1, keepdim=True)
        xi = torch.zeros(4, 33)
        xi[:, 0] = 1
        actions = torch.randn(8, 4, 29) * 0.1
        refs = [
            ReferenceFrame(None, None, None, None, state[i, :4, :3], state[i, :4, 3:7])
            for i in range(8)
        ]
        values = []
        for wm in (old, current):
            loss, metrics = wm.window_loss(state, actions, xi, torch.zeros(8, 4), refs)
            loss.backward()
            values.append((loss, metrics, [p.grad for p in wm.parameters()]))
        torch.testing.assert_close(values[0][0], values[1][0], atol=0, rtol=0)
        assert values[0][1].keys() == values[1][1].keys()
        for key in values[0][1]:
            torch.testing.assert_close(values[0][1][key], values[1][1][key], atol=0, rtol=0)
        for a, b in zip(values[0][2], values[1][2]):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    finally:
        del sys.modules[module.__name__]
