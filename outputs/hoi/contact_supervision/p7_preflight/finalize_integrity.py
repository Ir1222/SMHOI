"""Verify this preflight against the starting inventory; never alter old evidence."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    baseline = json.loads((OUT / "baseline.json").read_text())
    allowed = {
        "scripts/hoi/audit_proximity_reward.py", "scripts/hoi/baseline.py",
        "scripts/hoi/build_manifest.py", "scripts/hoi/reconcile/common.py",
        "scripts/hoi/run_audit.py", "scripts/hoi/verify_p3_evidence.py",
        "src/supermimic/hoi/asset_manifest.py", "src/supermimic/hoi/checkpoint.py",
        "src/supermimic/hoi/policy_checkpoint.py", "src/supermimic/hoi/recorder.py",
        "src/supermimic/hoi/support_context.py", "src/supermimic/hoi/training_checkpoint.py",
        "src/supermimic/hoi/training_scene.py", "tests/hoi/test_joint_architecture.py",
        "tests/hoi/test_checkpoint_migration.py", "tests/hoi/test_hoi_policy_checkpoint.py",
        "tests/test_hoi_reconciliation.py",
    }
    changes, unexpected, unchanged = {}, {}, 0
    for name, before in baseline["files"].items():
        path = Path(name)
        after = digest(path) if path.is_file() else None
        if before != after:
            record = {"before": before, "after": after}
            changes[name] = record
            relative = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else name
            if relative not in allowed:
                unexpected[name] = record
        else:
            unchanged += 1
    write("preservation.json", {
        "baseline_commit": baseline["head"], "checked": len(baseline["files"]),
        "unchanged": unchanged, "intentional_source_changes": changes,
        "unexpected_changes": unexpected,
        "master_assets_references_calibration_historical_outputs_checkpoints_and_user_files_preserved": not unexpected,
    })
    assert not unexpected, unexpected
    existing = subprocess.check_output(["git", "diff", "--name-only", baseline["head"]], cwd=ROOT, text=True).splitlines()
    new = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    paths = sorted(set(existing + [p for p in new if not p.startswith(".vscode/")]))
    write("changes.json", [{"path": p, "sha256": digest(ROOT / p), "bytes": (ROOT / p).stat().st_size} for p in paths])
    write("files.json", [{"path": str(p.relative_to(ROOT)), "sha256": digest(p), "bytes": p.stat().st_size}
                         for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "files.json"])
    print(json.dumps({"checked": len(baseline["files"]), "unchanged": unchanged,
                      "intentional_changes": len(changes), "unexpected": len(unexpected)}))


if __name__ == "__main__":
    main()
