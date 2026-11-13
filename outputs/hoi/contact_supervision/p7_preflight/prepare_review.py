"""Record concrete review inputs from pinned evidence; never approve a profile."""

import json
from pathlib import Path

import torch

from supermimic.hoi.provenance import research_authority
from supermimic.hoi.reference import sha256
from supermimic.hoi.training_scene import SINGLE_BODY, profile_manifest

OUT = Path(__file__).resolve().parent
RUNS = Path("/tmp/smhoi_p7_preflight")


def save(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    manifest = profile_manifest("p6-single-body")
    identity = json.loads((RUNS / "authority_smoke/identity.json").read_text())
    save(
        "physical_profile_proposal_v1.json",
        {
            "profile_id": "p7-single-body-original-ref-v1-proposed",
            "status": "PENDING_USER_RESEARCH_REVIEW",
            "approved": False,
            "training_ready": False,
            "production_selected": False,
            "scope": "candidate for bounded physical collection only; not P7-A/B approval",
            "research_authority": research_authority(),
            "implementation_basis": "p6-single-body diagnostic profile, unchanged",
            "candidate": {"path": str(SINGLE_BODY), "sha256": sha256(SINGLE_BODY)},
            "identity": {
                k: identity[k]
                for k in (
                    "production_manifest_sha256",
                    "box_sha256",
                    "support_sha256",
                    "contact_targets_sha256",
                    "calibration_sha256",
                    "reference_sha256",
                    "reference_manifest_sha256",
                    "robot_manifest_sha256",
                    "solver",
                    "physics_dt",
                    "decimation",
                    "iterations",
                    "ls_iterations",
                    "dependencies",
                )
            },
            "object": manifest["object"],
            "supports": manifest["supports"],
            "contact_targets": manifest["contact"],
            "limitations_requiring_review": [
                "homogeneous solid approximation; cavity filled and handles omitted",
                "original HDMI reference and labels unchanged; native manipulation failed",
                "frame-0 foot-floor reset overlap 0.01223564506031003 m",
                "original reference contact-frame resets have up to 0.035 m hand overlap; excluded from proposal",
                "bounded precision-guard evidence, not universal solver reliability",
                "no verified natural hand-supported lift/carry dataset",
                "no new acceptance threshold for geometric overlap selected",
            ],
            "reference_candidate_v1": {"promoted": False, "training_ready": False},
            "collection_proposal": {
                "id": "p7-native-pd-frame0-collection-v1-proposed",
                "approved": False,
                "controller": "existing P6 native joint-position reference controller; free roots",
                "reset_frames": [0],
                "num_envs": 4,
                "seeds": [0, 1],
                "max_vector_steps_per_seed": 839,
                "max_world_transitions": 6712,
                "max_wall_seconds_total": 600,
                "action_noise_std": 0.03,
                "reset_noise": False,
                "domain_randomization": False,
                "optimizer_steps": 0,
                "source_partition_proposal": "all seed-0 source episodes train; all seed-1 source episodes held out; identical trajectories deduplicated/grouped before windows",
                "reset_policy": "normal per-world reset to original frame 0 only; preserve reached terminal state first",
                "failure_policy": "abort on any invalid solve or nonfinite; retain failed interval separately; never target it",
                "implementation": "approval-gated collection mode on original Trainer.run and accepted recorder; not implemented or launched in preflight",
                "insufficient_coverage_policy": "stop and report missing regimes; no automatic extension, reference edits, external Box wrench or teleportation",
            },
        },
    )
    records = {}
    payloads = {}
    for name in ("authority_smoke", "authority_resume"):
        root = RUNS / name
        values = {p.name: json.loads(p.read_text()) for p in root.glob("*.json")}
        checkpoints = []
        for path in sorted(root.glob("hoi_*.pt")):
            p = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            checkpoints.append(
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "schema": p["schema"],
                    "iteration": p["counters"]["iteration"],
                    "optimizer_steps": p["optimizer_steps"],
                    "replay_rows": int(p["runtime"]["replay"]["valid"].sum()),
                }
            )
            if p["counters"]["iteration"] == 4:
                payloads[name] = p
        assert values["updates.json"] == {"wm": 0, "actor": 0, "critic": 0}
        records[name] = {"json_evidence": values, "checkpoints": checkpoints}
    a, b = payloads["authority_smoke"], payloads["authority_resume"]
    discrete = ("ep_id", "frame", "clip", "valid")
    comparison = {
        "scope": "four-step smoke; exact restore is separately unit-tested; not long-run equivalence",
        "final_physical_state_max_abs": float(
            (a["runtime"]["recorder"]["state"] - b["runtime"]["recorder"]["state"]).abs().max()
        ),
        "all_weights_identical": all(
            torch.equal(v, b["models"][m][k]) for m, w in a["models"].items() for k, v in w.items()
        ),
        "replay_discrete_identical": all(
            torch.equal(a["runtime"]["replay"][k], b["runtime"]["replay"][k]) for k in discrete
        ),
        "checkpoint_identity_identical": a["identity"] == b["identity"],
        "optimizer_steps": a["optimizer_steps"],
    }
    save("entry_evidence.json", {"runs": records, "comparison": comparison})
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
