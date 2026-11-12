# P6 implementation and validation report

**P6 = PARTIAL; PHYSICAL_VALIDITY = PARTIAL; REFERENCE_COMPATIBILITY = FAIL.**
The complete authorized P6 implementation, bounded experiments, analysis and
reports are delivered for review. Validation did not establish full physical
training readiness. **P7 prerequisites are not satisfied.** No P7/P8, WM fitting,
Actor/Critic updates, automatic commit or push occurred.

Detailed decisions: [physical validation](PHYSICAL_VALIDATION_REPORT.md),
[reference compatibility](REFERENCE_COMPATIBILITY_REPORT.md), and the concrete
[implementation specification](IMPLEMENTATION_SPEC.md).

## Implementation and main evidence

Baseline: `main` / `e4f574185597d23570fad462224f76f741b36538`.
Master Plan SHA256: `68142af38edb0a9a443aac5815491aaae90dba696acc84d30fc54223c6e4a9cc`.
The sole pre-existing visible user entry was untracked `.vscode/`; it is preserved.
No applicable repository/ancestor AGENTS.md was found. Current dependency/source
implementations and the complete Master Plan/P0–P5/M6.5 reports were inspected.

All evidence is under
`outputs/hoi/contact_supervision/contact_validation/` (E).
`baseline.json` inventories 770 existing paths, including ignored accepted/refined
NPZs, historical reports, checkpoints, assets, external physical sources and ten
installed dependency files. `preservation_final.json` records the final comparison.

Reused unchanged P1 scene/sensors/state/reset and P2 IntervalCapture,
aggregate_interval, recorder and HOIReplayBuffer. Reused P4 geometry/FK and P5
next-state/reference semantics. No replacement sensor, Replay or training loop.
P1/P2 passing suites were not rerun; new physics regimes and the newly defined
contact metric supplied the specific new coverage.

New code provides independent calibrated binary contact, common-mask confusion
metrics, episode-aware onset/loss and penetration-duration summaries; bounded GPU
geometry/calibration/solver/reference replay; isolated CPU force/torque/lift/release;
and reproducible saved-evidence analysis. All measured force remains diagnostic.
Physical84/canonical86/residual82, WM166/82, support18/action29/xi33 and original
Actor/Critic observation contracts are unchanged.

| New experiment / check | Executed result |
|---|---|
| Native approach scans | 204 FK placements; correct side/first-contact isolation |
| Independent bilateral IK | <1.4e-9 m residual; ≤0.5 mm initial overlap after fixture correction |
| Force calibration | 60 world-transitions, 15 initial configurations; zero null force; held-out per-hand 5/5 detection |
| Binary contact definition | τL=0.07434010469 N, τR=0.07375101818 N; strict occupancy>.25, at least2/4 solves |
| Robot-body/Box and support/Box filtering | Loaded excluded contacts; hand sensors exactly zero; corrected torso load2.709 N |
| Isolated Box dynamics | 14 CPU cases; analytic force/torque error≤1.29e-8 m/s and1.75e-18 rad/s |
| Original supports, external lift/release | 19.62 N balance; lift83.6245 mm; stable final0.5 s; no CPU warnings |
| Original supports in four GPU worlds, distant robot clear of floor | 3 s complete; settle10.1079 mm; force19.620002–19.620003 N |
| Coupled original floor-impact cases | FAIL, LS1024 at0.700/0.835 s; failures and partial samples retained |
| Original-reference geometry | All840 frames; worst native overlap37.223/44.995 mm; no ON frame with both hands in [−1,+2] mm band |
| Bounded original-PD replay | Six start windows ×4 worlds; 1,124 valid transitions,244 distinct reached frames; two solver-stopped windows |
| OR reference agreement | TN624/FP20/FN351/TP129; accuracy66.99%, F1 41.02%, balanced accuracy61.88% |
| AND reference agreement | TN632/FP12/FN428/TP52; accuracy60.85%, F1 19.12%, balanced accuracy54.48% |
| Independent left/right contact ratio | 10.68% /8.27%; full ON/OFF occupancy/duration and event metrics reported |
| Saved force arithmetic/alignment | 2,616 world-transitions checked; max mean error7.62e-6 N, impulse1.48e-7 N·s |
| Final focused software tests | 10 passed,0 failed,0 skipped; `tests_final.log`/XML |
| Final evidence verifier | 19 checks PASS; failed physical/reference gates retained separately |

Durations across replay worlds are pooled world-seconds, not a continuous full-task
trajectory. Replay applies original joint PD commands; no learned policy or Box
teleport occurs after reset. External Box lifting is explicitly forced and separate
from hand release/freefall. Successful native-hand lifting/carrying is NOT TESTED.

## Remaining failures and proposals

- **COM/inertia:** nominal2 kg COM is190 mm below the collision envelope. Its
  numerical response is correct for the declared parameters, but the physical mass
  distribution is not reconciled as a real bread box. An unapplied uniform-density
  component proposal gives COMz=.107491082 m and root-axis COM inertia
  diag(.043111707,.023643455,.037979707) kg·m², with assumptions and required
  regression tests in the physical report and `box_dynamics_01/mass_proposal.json`.
- **Solver/floor:** LS1024 occurs without contact-buffer saturation. Paired controls
  associate one reproducible failure with foot/floor impact; raising only the
  distant robot avoids that impact and leaves Box/support behavior stable. This
  does not validate the original floor scenario. Historical P2 LS1024 and P3
  ITERATIONS512 failures remain historical FAILs; no solver setting was changed.
- **Reference:** EEF markers are internal to capsules; reference pose geometry,
  support gap, contact timing and unassisted tracking mismatch remain. Bilateral
  requirement follows P4.1 only during ON260–549. The original label and NPZ remain
  intact. Full mismatch classification is in the reference report.
- **Provenance:** missing historical V3 text still blocks the strict production
  loader. A pinned P6-only adapter permits these authorized diagnostics; it does
  not change the loader or `training_ready=false` / `M6_5_NO_GO_M7`.
- **Training prerequisites:** learned Joint-WM validation remains deferred; prior
  P3 FP32 migration limitations remain. No natural carrying demonstration or
  approved disposition of the physical blockers was produced.

Next review should disposition the nominal mass model and the original-floor
solver limitations, then decide whether to authorize any versioned asset/solver
experiment or bounded controller demonstration. Reference changes require their
own approval/provenance. No such research-level correction has been applied.

## Failure preservation and corrections

`calibration_01`/`geometry_01` retain the initial invalid bilateral IK; the
position-only `ik_fixture_probe` also retains its pitch overlap. The corrected
fixture adds residual and actual-collider guards; `calibration_02`/`geometry_02`
are final. `filters_01`/`verification_01` preserve the insufficient torso-load
failure; `filters_02`/`verification_02` pass without lowering the criterion.
Physical warning cases in `dynamics_01`, `isolation_01`, `replay_01` remain FAIL.

Initial diagnostic runners returned process success when the experiment loop
completed even if a case JSON said FAIL. The final runner exits nonzero on any
failed case. Reports always use the saved case status and raw flags, never that
old process exit code. Lint findings and corrective logs are retained. The existing
TorchScript deprecation warning appears; no package was installed or changed.
The shell sandbox failed before execution (`bwrap` loopback permission); the
approved external-shell fallback was used. No automatic approval rejection blocked
this work.

## Files, exact commands and review

New code:

- `src/supermimic/hoi/contact_validation.py`.
- `tests/hoi/test_physical_contact_agreement.py`.
- `tests/hoi/test_collision_diagnostics.py`.
- E/scripts: `common.py`, `run_p6.py`, `box_dynamics.py`, `analyze_p6.py`,
  `physical_analysis.py`, `verify_evidence.py`, `export_commands.py`,
  `ik_fixture_probe.py`.
- The four requested P6 Markdown documents in this directory.
- E baseline/configuration, per-run initial states, interval samples, contacts,
  compiled models, exact source snapshots, analyses, failures and test logs.

**Every new artifact**, size and hash is enumerated in E/`files.json` (generated
cache files and the inventory itself are excluded); final Git scope/status is in
E/`review.json`. The inventory explicitly identifies its own excluded path. There are no
changes to pre-existing tracked source, assets, NPZs, checkpoints or dependencies.
`source_review.patch` contains the complete new source/test/documentation diff;
machine evidence is checked structurally and numerically, with hashes in the file
inventory. Nothing was staged, committed or pushed.

E/`commands.json` records **every exact shell command**, including inspection,
failed sandbox attempts, file-writing commands, physical runs, corrections,
analysis, tests and final review, exported from this task's own tool-call transcript.
Main executable commands (run from repository root, project Python):

```bash
.venv/bin/python outputs/hoi/contact_supervision/contact_validation/scripts/run_p6.py --mode calibration --output outputs/hoi/contact_supervision/contact_validation/calibration_02
.venv/bin/python outputs/hoi/contact_supervision/contact_validation/scripts/run_p6.py --mode geometry --output outputs/hoi/contact_supervision/contact_validation/geometry_02
.venv/bin/python outputs/hoi/contact_supervision/contact_validation/scripts/box_dynamics.py --output outputs/hoi/contact_supervision/contact_validation/box_dynamics_01
.venv/bin/python outputs/hoi/contact_supervision/contact_validation/scripts/run_p6.py --mode dynamics --output outputs/hoi/contact_supervision/contact_validation/dynamics_01
.venv/bin/python outputs/hoi/contact_supervision/contact_validation/scripts/run_p6.py --mode isolation --output outputs/hoi/contact_supervision/contact_validation/isolation_01
.venv/bin/python outputs/hoi/contact_supervision/contact_validation/scripts/run_p6.py --mode filters --output outputs/hoi/contact_supervision/contact_validation/filters_02
.venv/bin/python outputs/hoi/contact_supervision/contact_validation/scripts/run_p6.py --mode replay --output outputs/hoi/contact_supervision/contact_validation/replay_01
.venv/bin/python outputs/hoi/contact_supervision/contact_validation/scripts/physical_analysis.py --root outputs/hoi/contact_supervision/contact_validation --output outputs/hoi/contact_supervision/contact_validation/analysis_02
.venv/bin/python outputs/hoi/contact_supervision/contact_validation/scripts/verify_evidence.py --root outputs/hoi/contact_supervision/contact_validation --output outputs/hoi/contact_supervision/contact_validation/verification_02.json
.venv/bin/python -m pytest tests/hoi/test_physical_contact_agreement.py tests/hoi/test_collision_diagnostics.py -q
```

Existing output directories intentionally cannot be rerun in place. Use new paths;
executed versions are stored in run snapshots, and the exact original commands
include redirection/analysis options in the ledger. Never invoke old M0–M6.5
aggregate generation to reproduce P6.

## Training framework reuse audit and stop

No shared training file changed. Native mjlab stepping, original G1 actuators,
P1/P2 sensors/reset/recording/Replay, accepted reference loading and P4 geometry
were composed. No Trainer, WM learner, Actor BPTT, Critic, optimizer or scheduler
was copied or run. No contact/force prediction head, BCE, regression loss, force
reward or differentiable measured-force path was introduced. Original robot-only
behavior is protected by unchanged source hashes; previously passed suites were
not repeated without an affected interface.

**Stop for review after P6. No P7 prerequisites or full physical readiness are
claimed, and no subsequent stage is started.**
