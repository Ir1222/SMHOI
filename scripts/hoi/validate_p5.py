"""Reproduce bounded P5 validation with an enforced no-optimizer/no-physics guard.

Run from the repository root with the existing .venv. Output directories must be
new, so earlier failures/evidence cannot be overwritten. No training is invoked.
"""

import argparse
import contextlib
import importlib.metadata
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import pytest
import torch

SUITES = {
    "eager": [
        f"tests/hoi/{name}.py"
        for name in (
            "test_box_tracking_reward",
            "test_reward_reference",
            "test_quaternion_reward_numerics",
            "test_imagination_boundaries",
            "test_policy_adapter",
            "test_p5_validation",
        )
    ],
    "compiled": ["tests/hoi/test_imagination_parity.py"],
    "regression": [
        f"tests/hoi/{name}.py"
        for name in (
            "test_force_replay",
            "test_force_aggregation",
            "test_geometric_contact_reward",
            "test_contact_target_audit",
            "test_joint_world_model",
            "test_joint_architecture",
            "test_support_context",
            "test_wm_reuse",
            "test_wm_dataset_validation",
            "test_checkpoint_migration",
        )
    ]
    + [
        "tests/test_g1_fk.py",
        "tests/test_world_model.py::test_state_spec_identity_and_errors",
        "tests/test_imagination.py::test_gae_targets_match_hand_computation",
        "tests/hoi/test_joint_scene.py::test_reference_twist_endpoints_rotation_and_sign_flips",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["P5_EVIDENCE_DIR"] = str(args.output / "measurements")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/smhoi_p5_inductor")
    torch.set_num_threads(1)
    attempts = []

    def forbidden(*unused_args, **unused_kwargs):
        attempts.append("optimizer update or physical stepping attempted")
        raise AssertionError("P5 validation forbids optimizer updates and physical stepping")

    # Test-only guard. Production modules are never patched by the HOI adapters.
    originals = {
        "adam": torch.optim.Adam.step,
        "sgd": torch.optim.SGD.step,
        "mj_step": mujoco.mj_step,
    }
    torch.optim.Adam.step = forbidden
    torch.optim.SGD.step = forbidden
    mujoco.mj_step = forbidden
    started = time.monotonic()
    command = ["-x", "-q", "-ra", *SUITES[args.suite], f"--junitxml={args.output / 'tests.xml'}"]
    try:
        with (
            (args.output / "pytest.log").open("w") as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            code = int(pytest.main(command))
    finally:
        torch.optim.Adam.step = originals["adam"]
        torch.optim.SGD.step = originals["sgd"]
        mujoco.mj_step = originals["mj_step"]
    cases = ET.parse(args.output / "tests.xml").findall(".//testcase")
    counts = {"total": len(cases), "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    details = []
    for case in cases:
        status = next(
            (
                s
                for tag, s in (("failure", "failed"), ("error", "errors"), ("skipped", "skipped"))
                if case.find(tag) is not None
            ),
            "passed",
        )
        counts[status] += 1
        details.append({"test": f"{case.get('classname')}::{case.get('name')}", "status": status})
    summary = {
        "suite": args.suite,
        "status": "PASS"
        if code == 0 and not counts["skipped"]
        else "PARTIAL"
        if code == 0
        else "FAIL",
        "pytest_arguments": command,
        "exit_code": code,
        "counts": counts,
        "tests": details,
        "seconds": time.monotonic() - started,
        "forbidden_action_attempts": attempts,
        "optimizer_steps": 0,
        "physical_rollouts": 0,
        "versions": {x: importlib.metadata.version(x) for x in ("torch", "numpy", "mjlab")},
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("tests", "pytest_arguments")}, indent=2
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
