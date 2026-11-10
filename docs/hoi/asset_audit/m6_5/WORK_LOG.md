# M6.5 work log

| Stage | Commit | Result |
|---|---|---|
| Baseline and scope | b248726 | Read required documents, freeze historical files and data, define geometry/PD protocol |
| A–B reconciliation | 3da865c | Full 840-frame surface/footprint/height sweep, EEF fits, shape/orientation ablations, bounded fixed-robot box-pose search; document candidates before dynamics |
| C physical tests | 5608302 | 36 original-PD/free-box primary cases, 6 half-step cases, excluded smoke preserved, actual forces/slip/trajectory logs, figures, focused tests |
| D review decision | commit titled `audit(M6.5 D): publish NO-GO reconciliation report and immutable review manifest` | Versioned manifest, report, combined regression, final integrity, pause for review |

Executed tooling uses the current SMHOI .venv and installed CPU MuJoCo. No packages installed, no source checkout switched, no original reference regenerated, no trained policy/checkpoint loaded, no root assist, no box actuation, no weld. No old report, old manifest or shared training module changed. Existing untracked user plan and source files remain outside these commits.

Source-level finding: official contact error is a soft reward (0.3 m scale), with separate measured contact forces. Source offsets and coordinate transforms were already correct. Best constant EEF fit still leaves ~28 mm RMS; physical intersections remain. Static support-height constraints are incompatible at the original strict tolerances. Conditional terminal gravity settling is stable, but short coupled interaction probes fail. These distinct results are not merged into a false gate pass.

Development correction: preliminary smoke frame indexing rounded rather than holding the 50 Hz reference command. Smoke outputs remain under dynamics_smoke with explicit exclusion. dynamics_v1 and dynamics_half_step use integer 4/8-step holds and are independently tested. No previous experiment was overwritten.

Reproduction details:

- `geometry.py` generates only the M6.5 diagnostic CSV/JSON; it never writes NPZ. To preserve this completed run, use another output location/copy before re-running its generated diagnostics.
- `dynamics.py --output <new_name>` refuses an existing experiment directory. Each case initializes reference state once, then only writes original PD controls and calls MuJoCo stepping.
- `sensitivity.py` was run once into dynamics_half_step and refuses reuse.
- `plots.py` reads those CSVs to export seven PNG/PDF figure pairs; all are new M6.5 artifacts.
- `decision.py` creates a new asset_manifest_m6_5.json and refuses to overwrite it; source manifest is frozen.
- `pytest tests/test_hoi_reconciliation.py tests/test_hoi_asset_audit.py -q` checks new logic and M6 regression. No broad training/evaluation launch.

See baseline.json / integrity_final.json for hash protection and additional_sources.json for the newly read reward, Entity, actuator and simulator implementation sources. Artifact SHA256 inventory is artifact_hashes.json. Current physical decision is NO-GO, not a proof that every future controller or broader object/contact reconciliation is impossible. Stop after M6.5 and await review.
