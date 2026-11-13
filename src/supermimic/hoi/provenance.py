"""Current research authority and explicit retirement of the incorrect V3 upload.

Historical manifests stay byte-identical. Their obsolete planning-source record
is interpreted through this named metadata migration, never opened or required.
All other source hashes remain strict. This grants no physical/training approval.
"""

from pathlib import Path

from supermimic.hoi.reference import REPO, sha256

MASTER_RELATIVE = "docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md"
MASTER_SHA256 = "f8e829911dfd1ac6808e4c3e2b472abd55c2bb5ddeab2167f9e311b40afa5aea"
MIGRATION = "retire-incorrect-v3-use-current-master-v1"
RETIRED_NAME = "SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md"
RETIRED_SHA256 = "168465ae0a2e17153902a337963574942551909680d0d8dd36f7ffb22102b05c"


def research_authority():
    if sha256(REPO / MASTER_RELATIVE) != MASTER_SHA256:
        raise ValueError("Current Master Plan SHA256 changed; explicit provenance review required")
    return {
        "path": MASTER_RELATIVE,
        "sha256": MASTER_SHA256,
        "migration": MIGRATION,
        "authority": "sole research and implementation plan; explicit user decision",
    }


def effective_sources(sources):
    """Replace only the exact retired planning record; never waive a physical source."""
    authority = research_authority()
    current = {"path": str(REPO / MASTER_RELATIVE), "sha256": authority["sha256"]}
    result = []
    for source in sources:
        if Path(source["path"]).name == RETIRED_NAME:
            if source["sha256"] != RETIRED_SHA256:
                raise ValueError("Unrecognized retired planning-source identity")
            continue
        if Path(source["path"]).resolve() == (REPO / MASTER_RELATIVE).resolve():
            if source["sha256"] != MASTER_SHA256:
                raise ValueError("Incorrect current Master Plan source identity")
            continue
        result.append(dict(source))
    return result + [current]


def verify_sources(sources):
    for source in effective_sources(sources):
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("Provenance source changed: " + source["path"])
