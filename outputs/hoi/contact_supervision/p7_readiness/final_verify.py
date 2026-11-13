"""Verify P7-only file scope, preserved baseline and final evidence consistency."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[4]
out = Path(__file__).parent


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def write(name, value):
    (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


before = json.loads((out / "preservation_before.json").read_text())
changes = []
for relative, digest in before.items():
    path = root / relative
    actual = sha(path) if path.is_file() else None
    if actual != digest:
        changes.append({"path": relative, "before": digest, "after": actual})
assert not changes, changes
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == (
    "a6b876f392f69709d9835eb0e4d664bfcfe75b8d"
)
assert not subprocess.check_output(["git", "diff", "--name-only"], cwd=root)
assert not subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=root)
write(
    "preservation_after.json",
    {
        "status": "PASS",
        "paths_checked": len(before),
        "changed": changes,
        "tracked_diff_empty": True,
        "index_diff_empty": True,
    },
)

result = json.loads((out / "physical_final/result.json").read_text())
assert result["preflight_status"] == "PASS"
assert result["p7_1_ready"] is False
assert result["checkpoint"]["optimizer_updates"] == 0
assert result["feedback_probe"]["box_x_0p05m_action_max_change"] == 0
assert result["guard_restored_dependency_unchanged"]
analysis = json.loads((out / "analysis.json").read_text())
assert analysis["contact_recomputed_from_retained_raw_substeps"]
pilot = json.loads((root / "docs/hoi/force_contact/p7/ppo_pilot_v1.json").read_text())
assert not pilot["launch_enabled"] and not pilot["training_authorized"]
budget = pilot["budget"]
assert (
    budget["training_transitions_max"]
    == budget["num_envs"] * budget["rollout_steps"] * budget["iterations"]
)
assert (
    budget["total_transitions_max"]
    == budget["training_transitions_max"] + budget["evaluation_transitions_max"]
)
assert budget["optimizer_minibatch_updates_max"] == (
    budget["iterations"] * pilot["ppo"]["num_learning_epochs"] * pilot["ppo"]["num_mini_batches"]
)

# Actual relevant local source/dependency identities; no downloads or changes.
sources = list((root / "src/supermimic").rglob("*.py"))
sources += [
    root / p
    for p in (
        ".venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py",
        ".venv/lib/python3.12/site-packages/mjlab/envs/manager_based_rl_env.py",
        ".venv/lib/python3.12/site-packages/mjlab/sensor/contact_sensor.py",
        ".venv/lib/python3.12/site-packages/mjlab/sim/sim.py",
        ".venv/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py",
        ".venv/lib/python3.12/site-packages/mjlab-1.6.0.dist-info/direct_url.json",
        "docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md",
    )
]
sources += list((root / "outputs/hoi/contact_supervision/p6_closure/scripts").glob("*.py"))
write("source_inventory.json", {str(p.relative_to(root)): sha(p) for p in sources})
generated = list(out.rglob("*")) + list((root / "docs/hoi/force_contact/p7").glob("*"))
generated.append(root / "scripts/hoi/p7_readiness.py")
inventory = {
    str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha(p)}
    for p in sorted(generated)
    if p.is_file() and "__pycache__" not in p.parts and p.name != "file_inventory.json"
}
write("file_inventory.json", inventory)
print(
    json.dumps(
        {
            "preserved_paths": len(before),
            "evidence_files": len(inventory),
            "status": "PASS",
            "training_authorized": False,
        },
        indent=2,
    )
)
