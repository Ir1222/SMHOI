# P3 architecture cleanup specification

Authorized scope: final 166D input / 82D output joint dynamics, software regression,
named checkpoint compatibility, independent report, and one reviewed commit. The
user approved this scope and explicitly requested a commit on 2026-11-11. No push,
P4 implementation, simulator collection, optimizer update, or predictive training /
validation is authorized. Predictive validation is deferred to P7-A; historical P3
remains PARTIAL.

## Baseline and precedence

Branch main, HEAD `7251bf33f95fb4e2ee75f45a306f66fbadac85d1`, matching GitHub main at
the preceding read-only review. Master Plan SHA256:
`500a59dd17644284f012e1844c3f64e98287d7d84e0bda9f2ee0f4272733d75d`.
Read the complete Master Plan and P0 audits/closure/file map, P1/P2 reports and P3
original/revision specifications/reports. P1 `a562d936`, P2 implementation `5730be4d`
and report `a991ee7` are ancestors. Existing untracked `.vscode/` is user-owned.

Master §§1/2/4/8/14/17 and the user's latest decisions supersede the duplicate old
force-era P5–P8 chapters and residual Force Head target wording in §7. Do not edit
the Master Plan or historical reports. Original Part B reference `bbbb40a4…` and
historical refined reference `f6ba446c…` remain distinct identities: no switching or
relabeling. Migration probes use the historical source's explicit refined identity.

## Traceability and changes

| Requirement | P0 / actual evidence | Implementation decision | Validation |
|---|---|---|---|
| §8.1–4 physical84/canonical86/residual82/support18 | P0 architecture and coordinate closure; current JointStateSpec and accepted support_02 evidence | Preserve StateSpec composition, frames, scaling, integration, SupportContext and xi33; remove unused force dimension declaration only | Existing canonical/round-trip/quaternion/COM/support tests |
| §8.5 shared 166/82 dynamics | Current model has contact column86 and row82, state85 | DynamicsMLP features104 + action29 + xi33, output82; predict/step return physical84 tensor; no contact augmentation/logit/probability/BCE | Dimensions, rejection of85, synthetic eight-step loss/backward, no optimizer |
| §8.6 reference intent outside WM | P2 intent_ref stores global[start,end]; dataset validates canonical labels | Keep archive/reference/Replay unchanged; trainer validates physical windows only, no contact cache/target; preserve reference t+1 for termination | Label/force/clean-action independence, reference lookup and window checks |
| §8.7 explicit initialization and strict loading | P0 named action/xi mapping; legacy strict force schema; contact schema v2 | New physical-only v3 schema; robot133/70 and strict force166/88 migration; reject contact167/83; preserve term head and physical coefficients; no optimizer or readiness transfer | Sentinel maps, real source parity 1–8, strict negatives/round trip |
| §8.8 skip diagnostic, defer prediction | Failed capture one row/flag512; old Gate B and fitting scripts remain | Retire contact collector/fitter before imports/side effects; remove classifier metrics/gates from active validation; retain isolated legacy force gate | Retired entry tests; no fitting or simulator subprocess |
| §§1.3.6/14 framework reuse | Original window_loss auxiliary hook, inherited trainer update | Use empty auxiliary-target mapping only to activate named physical component metrics; no shared source changes or duplicate loops | Exact original robot loss/metrics/gradients; inherited update identity |
| §§15 protection/report | Historical PARTIAL, M6.5 training_ready=false | Separate cleanup docs/output; hash protected files; preserve dormant force utilities and historical Git sources | Full diff, immutable hashes, skipped-test ledger |

## Final contracts

Physical state: robot71 + object13; canonical: robot71 + object15. Object physical
pose is environment-local root position and wxyz quaternion; twist is root-origin
world linear velocity and body angular velocity. Canonical object linear velocity
is absolute velocity rotated into robot heading, with no robot velocity subtraction.
Residual order remains robot70 + object[dv_heading3,domega_body3,dp_heading3,
dtheta_body3]. Velocity updates precede position, quaternion increments multiply on
the right, dt=.020 s; zero residual preserves constant root-origin twist.

Input named blocks: robot0:71, object71:86, support86:104, action104:133, xi133:166.
Support transforms are recomputed each step; xi33 stays episode-constant. Output
robot0:70/object70:82 only. TerminationHead is separate and retains original BCE
against physical_failure, not timeout/clip_end. Replay remains physical84 with
diagnostic force/contact fields and exact reference binary pairs.

Robot initialization maps action71:100→104:133 and xi100:133→133:166; new
object/support input columns and object output rows are zero. Historical strict
force checkpoint copies named inputs and physical82 rows; explicitly discards
force82:88/statistics. Raw old fit snapshots lack strict metadata and are rejected.
Transitional contact checkpoints are rejected, not converted. Metadata records all
dimensions including action29/xi33, schemas, timing, reference/asset provenance and
P7-A deferred validation. Strict=False is forbidden.

## Software-only verification and risks

Use existing float64 geometry/integration atol1e-9; FP32 mapping predictions
atol=rtol=2e-6; strict round-trip exact. External 2048-wide robot checkpoint has a
historical FP32 multi-step mismatch: report new measurements without relaxing the
criterion; FP64 is a diagnostic, not a substitute PASS. Source lacks training Git /
dataset hashes, so parity establishes architecture compatibility only.

Run selected CPU contract tests, synthetic forward/backward, archive integrity,
negative checkpoint tests, and original robot regression. Never run optimizer.step,
trainer.update, training smoke tests, physical recorder tests, boundary collection,
contact fitting, or learned prediction evaluation. Keep those exclusions explicit.
The legacy evidence verifier recomputes a ridge fit; do not execute it here. Preserve
its saved evidence and historical force-gate interpretation with read-only checks.

P1/P2 sensing and ten dependency hashes are unchanged. M6.5 physical hand/support
FAILs, training_ready=false, solver flag512 failure and historical P3 PARTIAL remain.
No asset/solver/reference changes are justified by architecture cleanup.

Outputs: `outputs/hoi/contact_supervision/architecture_cleanup/`; documentation in
this directory. Record exact commands, tests, failures, skips, numerical evidence,
every changed file and full Git review in REPORT.md before the authorized commit.

## A1 — unavailable historical provenance documents (implementation adjustment)

First targeted run: 12 passed / 15 failed. The checkout lacks root
`SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md` and all three refined-reference JSON manifests;
these were historically untracked, and no exact copies were found in Git or local
ignored files. Do not regenerate them or claim their historical hashes were checked.

SupportContext now binds the exact accepted M6.5 manifest SHA256 before structural
validation. All available source files still require their declared hashes; the
only permitted missing source is the historical V3 planning document with its
exact known name/hash. This does not authorize alternate assets or alter support
math. Physical scene/asset loader validation remains unchanged. Test corrupted
manifest and unexpected missing/changed source rejection. This is a runtime
provenance adaptation, not resolution of M6.5 physical failures.

The migration-only reference adapter checks the exact accepted refined archive
SHA256 and existing array schema directly, without claiming missing sidecar
validation. Original ReferenceClip and historical p3_common.reference() remain
unchanged. No arrays or manifest files are generated. Use this explicit adapter
only for software probes; future physical/training entry prerequisites remain open.
