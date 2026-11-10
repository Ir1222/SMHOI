# Final P0 requirement coverage matrix

Specification: [complete canonical master](../SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md), 1,743 lines, SHA256 `7365f680cf975263e69672d75d585c62e83532115b9065c6e63929004bdbdc9b`. Supplied source: `docs/hoi/force contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.txt`; canonical copy is byte-identical. Original evidence is preserved in commits `579d1a8` and `e895731`.

PASS below means the **P0 audit obligation** is satisfied. It does not mean the later implementation exists, the historical physical gates passed, or the user approved P1. Workhorse paper-only inspection follows the user's explicit fallback instruction after the official site confirmed code is unreleased.

## P0 tasks and cross-stage contracts

| Requirement / master location | Existing evidence reused | Missing work completed in closure | Status |
|---|---|---|---|
| 1.1 / 16: read the whole specification, one stage only | Earlier audit read available fragment/V3 | Read all 1,743 lines including P1–P8, boundaries and acceptance; restored canonical copy | PASS |
| P0.1: AGENTS.md / V3 | `REPORT.md`, `repositories.json`, preservation baseline | Applicable AGENTS search still finds none; V3 hash unchanged; no unnecessary reread | PASS |
| P0.1: current branch, HEAD, recent commits/status | Original snapshots identify `8458784`; reconciliation identifies `579d1a8` | Current `main` at `e895731`; both earlier commits preserved, ancestry checked | PASS |
| P0.1: preserve user changes | Original and reconciliation snapshots | Four untracked user files, including new complete `.txt`, preserved; snapshot/final hash check | PASS |
| P0.1: safely observable processes | `processes.txt`: external TensorBoard, no observed training/evaluation | Reused observation; no process stopped or new training launched | PASS |
| P0.1: checkpoints / relevant outputs | `checkpoint_files.json`, dataset/asset audit, historical output hashes | No reload/evaluation needed; integrity hashes preserved | PASS |
| P0.2 / 3.1: actual source checkouts | `repositories.json`, fork/official distinctions | New reference-code inventory records actual HDMI/PhysHSI/SUGAR HEAD/status | PASS |
| 3.2: locate exact original HDMI motion | `artifact_inventory.json`, `DATASET_ASSET_AUDIT.md` | None; exact official-checkout path/hash already found | PASS |
| 3.3: both exact `/home/han/pyx` NPZs | Inventory keys/shapes/dtypes/FPS/size/SHA | None; preserved without regeneration | PASS |
| 3.4: seven source dataset/manifest paths, including ignored files | Inventory and provenance links | None; all explicitly inventoried | PASS |
| 3.5: HDMI XML/task/command/rewards | Asset inventory and original source inspection | P0.5 adds mechanism-level command/reward/YAML comparison | PASS |
| 3.6: Part B actual archive versus committed manifest | `audit_checks.json`, preservation final | Reused exact SHA and nine-field byte comparison; archive hash unchanged | PASS |
| 3.7: discover M6.5 manifest / diagnostics / historical FAILs | Actual `data/assets/hdmi_bread_box/asset_manifest_m6_5.json`, hash list and audit records | Preserve exact discovered paths and FAILs; no rerun | PASS |
| 3.7: candidate checkpoint existence/provenance/architecture | Target path MISSING; separately inventoried external candidate; strict load PASS, provenance qualified | No substitution/copy/migration; absence is explicitly reported as permitted | PASS |
| 3.8: every named artifact status, size/hash, schema, provenance, role, consumer | `artifact_inventory.json` (78 records), dataset report | Complete plan/canonical path now separately recorded; original inventory remains a historical snapshot | PASS |
| P0.2: actual current entry-point dataset loading | Dataset and architecture reports, candidate config | None; `sm-train` only, no current `smHOI-track` | PASS |
| P0.3: Trainer loop and checkpoint machinery | `ARCHITECTURE_AUDIT.md`, `checkpoint_audit.json` | Narrow factory/save/load boundaries rechecked for complete map | PASS |
| P0.3: StateIO, environment, command, observations/rewards/library | Architecture/source inventory | Added isolated origin read/write check; preserved global nq/nv assertions | PASS |
| P0.3: recorder/replay timing, windows, terminal alignment | `architecture_checks.json`, dependency callback trace | None; existing verified discrepancy mapped to P2.5 | PASS |
| P0.3: StateSpec/dynamics/heads/WM/fitting | Architecture report, shape maps, original tests | Added noncommuting integration check versus MuJoCo | PASS |
| P0.3: rollout/compiled/objective/policy trainer | Architecture report | P5 shared hook stage clarified; no code changed | PASS |
| P0.3: policy nets and existing HOI modules | Architecture report and source hashes | Map preserves fixed robot slices and display-only viewer | PASS |
| P0.4: installed/locked mjlab, MuJoCo Warp and exact pin | `dependencies.json` | Existing installed-file hashes checked for evidence reuse | PASS |
| P0.4: update/history, scene, stepping, recorder and CUDA order | Dependency report, history probe | None; actual path is `mjlab/scene/scene.py`, not guessed `mjlab/scene.py` | PASS |
| P0.4: netforce aggregation and secondary ambiguity | Dependency report and force-sign probe | None; one box body covers seven geoms; aggregate both primary columns | PASS |
| Newly visible P0.4: capacity/overflow and selected reset | Existing report already traced match limit, flags and row resets | No physical rerun; actual scene capacity/isolation remains P1 acceptance | PASS |
| Newly visible P0.4: force/torque frames | Force sign/world-frame trace existed | Torque centroid reference made explicit; common-point algebra and Box rotation checked | PASS |
| Newly visible P0.4: post-step forward overwrites instantaneous data | Existing update/cache/callback trace | None; history remains interval source | PASS |
| P0.5: HDMI actual implementation | Source file inventory existed | Detailed geometry, force/intent/reward mechanisms and compatibility report | PASS |
| P0.5: PhysHSI exact carrybox/config | Repository recorded, detailed audit absent | Actual local fork inspected; unfiltered contact, approach/hold/stage gates and xyzw differences recorded | PASS |
| P0.5: SUGAR rewards and relevant G1 carry refiner | Repository recorded, detailed audit absent | Exact rewards/config/inheritance/loader inspected; bilateral labels, three-sample max and object-relative tracking recorded | PASS |
| P0.5: Workhorse task targets/deployment-error augmentation | No completed mechanism audit | Paper IV-A–IV-D and official project inspected; unavailable code explicitly qualified under user-approved fallback | PASS with approved fallback |
| P0.5: reusable/adaptable/incompatible mechanisms | Prior audit warned against wholesale source port | Source-specific reuse matrix and framework/asset/reward differences written | PASS |
| P0.6: five mandated reports | All exist in original commit | Updated primary reports, validation, closure and map; new reference report and coverage matrix | PASS |
| P0.6: P1–P8 map: reuse, minimal changes, new files, tests, dependencies/risks | Earlier provisional map | Reconciled every stage and output destination; joint state in P1, WM migration P3, hooks P5, integrated migration P7 | PASS |
| 4.1: actual dimensions versus candidate targets | Robot71/canonical71/action29/xi33/residual70 verified | Joint84/canonical86/physical82/force6/combined88 remain planned contracts; no tensor changes | PASS audit |
| 4.1 / P3.4: support-context identifiability | Translation-equivalent canonical probe, candidate relative transforms | Full master assigns prediction ablation/final dimension to P3; retain candidate separately from xi33 | PASS audit; P3 experiment deferred |
| 4.2: physics/control time | `.005 × 4 = .020` verified | No change; interval mean N versus impulse N·s carried into map | PASS |
| 4.3: root/world/body/COM frames | Existing offset-COM numeric probe and source trace | StateIO origin test, quaternion right-product check, Box vector transform and torque point audit | PASS audit; physical HOI calibration P1 |
| 4.4: reference velocity frame/sign | Method documented, no explicit numeric probe | Synthetic central/one-sided pose differences and quaternion log tested, including sign-flipped equivalent quaternions; no cache/NPZ rewrite | PASS convention check |
| 4.5: intent/actual/predicted signals | Existing architecture/contact reports | Source audit explicitly rejects bilateral-source label substitution and reference intent as physical-force input | PASS |
| 2.2 / 14 / 15.4: preserve boundaries and immutable inputs | Prior preservation evidence | Closing preservation/diff checks verify documentation-only changes | PASS |
| 15.1: commands, tests, numbers, files, Git, limits and decision | Original validation/probe transcript | `CLOSURE.md`, `CLOSURE_CHECKS.md`, new JSONs and full diff | PASS |
| 15.2 / 16: stop for review; no next stage or automatic commit | Two prior explicitly approved commits | Current closure uncommitted, no push and P1 unstarted | PASS |

## Exact P0 acceptance checklist

| Acceptance criterion from master lines 412–420 | Result / evidence |
|---|---|
| Actual checkout, branch, HEAD and dependency versions recorded | PASS — current closure snapshot plus original version/pin audit |
| All listed dataset/asset/checkpoint paths marked | PASS — exact inventory; missing target checkpoint is explicit, no hidden substitute |
| Part B integrity verified against committed manifest | PASS — original nine-field/hash checks reused, actual archive and manifest hashes unchanged |
| Actual tensor shapes and integration semantics traced | PASS — architecture/checkpoint maps and added integration/coordinate probes |
| Sensor update/history/recording order understood | PASS — installed source trace and existing history/force probes; new torque clarification |
| Source implementation audit complete | PASS — HDMI/PhysHSI/SUGAR code plus user-approved Workhorse paper fallback |
| P1–P8 file-change map grounded in actual code | PASS — revised reuse/hooks/new-files/tests/dependencies/risks for every stage |
| No training source/data/assets/checkpoints/Part B modifications | PASS — closing integrity checks and documentation-only diff |

**Overall P0: PASS for review.** P1 physical calibration and all P2–P8 implementation/experiments remain unexecuted. Historical asset FAILs, limited checkpoint provenance and Workhorse's unavailable code remain visible limitations; none has been recast as physical or learning success.
