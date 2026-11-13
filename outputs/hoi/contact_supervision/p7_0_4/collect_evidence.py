"""Summarize bounded CLI checkpoints without adding model binaries to Git."""

import hashlib
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUNS = Path("/tmp/smhoi_p704_final")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    records = {}
    for name in ("continuous", "resumed", "candidate_smoke"):
        directory = RUNS / name
        exported = OUT / "entry_evidence" / name
        exported.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.json"):
            shutil.copy2(path, exported / path.name)
        checkpoints = []
        for path in sorted(directory.glob("hoi_*.pt")):
            data = torch.load(path, map_location="cpu", weights_only=True)
            checkpoints.append({
                "path": str(path), "bytes": path.stat().st_size, "sha256": sha(path),
                "schema": data["schema"], "iteration": data["counters"]["iteration"],
                "optimizer_steps": data["optimizer_steps"],
                "replay_valid_transitions": int(data["runtime"]["replay_fields"]["transition_valid"].sum()),
                "native_workspace_arrays": len(data["runtime"]["simulator"]),
                "runtime_groups": list(data["runtime"]),
            })
        records[name] = checkpoints
    write("entry_checkpoints.json", records)

    baseline = json.loads((OUT / "baseline.json").read_text())
    changed, missing = [], []
    for name, digest in baseline["files"].items():
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
        elif sha(path) != digest:
            changed.append(name)
    allowed = ["pyproject.toml", "src/supermimic/hoi/recorder.py", "src/supermimic/hoi/scene.py",
               "src/supermimic/imagination/policy_trainer.py", "src/supermimic/training/loop.py",
               "src/supermimic/world_model/wm_trainer.py"]
    assert sorted(changed) == sorted(allowed) and not missing
    candidate = ROOT / "outputs/hoi/contact_supervision/p6_closure/reference_candidate_v1"
    manifest = json.loads((candidate / "reference_manifest.json").read_text())
    assert manifest["training_ready"] is False
    assert sha(candidate / "reference.npz") == "f38071e5347403f9eff993dd37884bf374e0c003400156dc4cb87d3d2b856cb0"
    write("preservation.json", {"baseline": baseline["head"], "checked_paths": len(baseline["files"]),
        "allowed_changed_paths": changed, "missing": missing, "unintended_changes": [],
        "candidate_reference_training_ready": False, "candidate_reference_sha256": sha(candidate / "reference.npz")})


if __name__ == "__main__":
    main()
