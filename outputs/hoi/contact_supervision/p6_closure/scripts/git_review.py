"""Archive complete new-file inventory and reviewable source diff; never stage."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[5]
E = ROOT / "outputs/hoi/contact_supervision/p6_closure"
D = ROOT / "docs/hoi/force_contact/p6/closure"


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for data in iter(lambda: f.read(2**20), b""):
            h.update(data)
    return h.hexdigest()


def main():
    # Refresh provenance for every fetched source/asset, retaining earlier indices.
    tree = {r["path"]: r for r in json.loads((E / "upstream_tree.json").read_text())["tree"]}
    upstream = []
    for p in sorted((E / "upstream").rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = str(p.relative_to(E / "upstream"))
        data = p.read_bytes()
        row = {"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
        if rel in tree:
            row.update(
                repository="shuayyy/humanoid-amp-ppo",
                commit="b5087b7c8b3a123135b094b9cece3b2cefef26bc",
                git_blob=tree[rel]["sha"],
            )
            row["git_blob_match"] = any(
                hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()
                == tree[rel]["sha"]
                for b in [data, data[:-1] if data.endswith(b"\n") else data]
            )
            assert row["git_blob_match"], rel
        elif rel.startswith("mjlab_v1_4_"):
            row.update(
                repository="mujocolab/mjlab",
                commit="4360096b49c1bcc478668c11ce7d458db747e49d",
                provenance="Pinned source fetch, SHA256; no dependency replacement",
            )
        else:
            row["provenance"] = "Local source index"
        upstream.append(row)
    (E / "upstream_final_provenance.json").write_text(json.dumps(upstream, indent=2) + "\n")
    source = [
        ROOT / "src/supermimic/hoi/solver_precision.py",
        ROOT / "tests/hoi/test_solver_precision.py",
        ROOT / "tests/hoi/test_p6_object_candidates.py",
    ]
    files = source + sorted(D.glob("*.md")) + sorted((E / "scripts").glob("*.py"))
    patches = []
    for p in files:
        r = subprocess.run(
            ["git", "diff", "--no-index", "--", "/dev/null", str(p.relative_to(ROOT))],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert r.returncode in (0, 1), r.stderr
        patches.append(r.stdout)
    (E / "source.diff").write_text("".join(patches))
    tracked = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True)
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True
    )
    assert not tracked.strip() and not staged.strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    review = {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status": status,
        "tracked_diff": tracked,
        "staged_diff": staged,
        "new_source_and_report_paths": [str(p.relative_to(ROOT)) for p in files],
        "preexisting_user_changes": ".vscode/ preserved by baseline SHA inventory",
        "commit_or_push_performed": False,
    }
    (E / "git_review.json").write_text(json.dumps(review, indent=2) + "\n")
    paths = source + sorted(
        p
        for folder in (D, E)
        for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != "file_inventory.json"
    )
    rows = [
        {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in paths
    ]
    inventory = {
        "scope": "All new closure source/tests/docs/evidence, excluding caches and this self-referential inventory",
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(r["bytes"] for r in rows),
    }
    (E / "file_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(status, end="")
    print(
        "New artifact files",
        len(rows),
        "bytes",
        inventory["total_bytes"],
        "source/report diff files",
        len(files),
    )


if __name__ == "__main__":
    main()
