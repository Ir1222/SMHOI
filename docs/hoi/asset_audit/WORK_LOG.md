# M0–M6 work log — 2026-11-10

Scope follows user instructions and Part A of the fully read V3 plan. Part D milestone numbering conflicts and does not authorize learning work. Initial untracked files remain outside all audit commits.

| Task | Commit | Record |
|---|---|---|
| M0 | b7f8135 | Baseline, both repositories, input paths and immutable hashes |
| M1 | bf64a62 | Every NPZ field, six numeric arrays byte-identical; name dtype width difference |
| M1 provenance refinement | c319284 | Existing derived velocity archive hash and both heuristic phase sources |
| M2 | 09f5edf | Seven XML geoms, AABB, explicit inertia, source builder priority difference |
| M3 | e79844f | Fresh USD parse; native/HDMI hand and foot comparison |
| M4 | 96ae701 | All-frame CSV, primitive distances, FK check, visualizations; contact gate FAIL |
| M5 | ea84c7b | Actual supports, all-frame distances, initial/final 10 mm gap; support gate FAIL |
| M3–M4 refinement | 873d12c | Hand union connectivity, USD material provenance, released-hand plot framing |
| M6 | commit titled `audit(M6): publish blocked physical asset manifest and verified audit report` | Independent manifest, offline builder, explicit candidate hand-only profile, tests, report and stop |

Implementation corrections during verification: shared string metadata was initially over-constrained to dtype equality; inspection isolated the width-only difference while confirming numeric byte equality. Historical object velocity provenance was corrected after the raw comparison and SHA-matched derivation manifest; no reference was changed. A missing Path import and NumPy JSON scalar conversion were fixed before publishing the corresponding stage. Native/HDMI hand comparison was extended to record inertia differences and material-default limitations. All failures are recorded rather than hidden by altered poses or rewards.

M6 is an audit completion, not a physical gate pass. No new HOI training entry point, WM, replay, reward integration, checkpoint conversion or M7–M9 execution. No long training or rollout started. Source repository and user data/checkpoints/records untouched. The final hand/contact and scene-support discrepancies require an explicit next-stage decision; work pauses here.
