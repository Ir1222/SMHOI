# Part C P0 audit report

**Status: PASS for user review. P1 has not started.** The complete master has been read and restored, its P0 requirements are covered, and only the missing read-only checks were added. Workhorse paper-only inspection follows the user's explicit approval because its code is unreleased. This is audit acceptance, not physical-task or training readiness.

## Authoritative specification and Git history

Supplied complete source: `docs/hoi/force contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.txt` (62,075 bytes, 1,743 lines), SHA256 `7365f680cf975263e69672d75d585c62e83532115b9065c6e63929004bdbdc9b`. It was copied byte-for-byte to the previously requested canonical path `docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md`. Both original user files in the space-named folder are preserved.

Current branch is `main`, HEAD `e89573171c86be7f40c8d120dc813cee0274a152`. Original P0 audit `579d1a8c661985c91926a365c7505488abe03bb8` and reconciliation `e895731` are preserved; this closing documentation has not been committed or pushed. The older PARTIAL decisions were correct for the truncated input available then. Their JSON/log evidence remains unchanged as historical snapshots, including pre-commit fields and the formerly missing canonical-plan path.

The [final coverage matrix](REQUIREMENT_COVERAGE.md) maps each P0 task, cross-stage audit contract and all eight P0 acceptance criteria to evidence. The [closure report](CLOSURE.md) contains the complete file list, commands/checks, numerical results, limitations and review gate.

## Existing evidence reused

- Repository/user-change/process/checkpoint discovery and the entire V3 read are recorded in the original audit. No applicable AGENTS.md was found; V3 remains hash-identical.
- All named data/assets/checkpoints have explicit status/size/hash/provenance/consumer records. Eight NPZ schemas were inspected previously; no conversion or schema audit was repeated. See [DATASET_ASSET_AUDIT.md](DATASET_ASSET_AUDIT.md) and [artifact_inventory.json](artifact_inventory.json).
- Accepted Part B SHA256 remains `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`, matching its committed manifest. The original nine robot-field byte comparisons all passed. Its 12 fields, 840 samples, 50 Hz and global intent ON at frames 260–549 remain intact; object velocity/force/per-hand intent were not fabricated.
- M6.5 manifest remains `data/assets/hdmi_bread_box/asset_manifest_m6_5.json`, hash `1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`. Native robot, seven box geoms, 2 kg mass, offset COM/inertia and original supports are preserved. Historical hand-contact/support FAILs and `training_ready=false` remain unchanged.
- Target `logs/hdmi_robot_only/ckpt_0012000.pt` is explicitly MISSING. The separately identified external candidate has hash `16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7`; its prior strict robot policy/critic/WM load passed. No embedded training Git/data SHA attests its provenance. It was not copied, migrated, overwritten or evaluated for policy quality.
- Installed/locked mjlab is 1.6.0 at `8bb2bc9e187797689ec82c78c35019afd2fd0376`; MuJoCo and MuJoCo Warp 3.16.0, Warp 1.18.0 and Torch 2.14.1+cu130. Existing source/dependency hashes support reuse of the original tests and sensor probes.

## Missing work completed

P0.5 was absent from the truncated plan. The new [reference implementation audit](REFERENCE_IMPLEMENTATION_AUDIT.md) inspects actual HDMI command/reward code, local PhysHSI carry-box/config code and SUGAR reward/carry-refiner/loader code. It records exact revisions/hashes and separates reusable mathematics, adaptations and incompatible mechanisms. Workhorse's targets and deployment-error augmentation were inspected in its paper and official description under the approved fallback; no source-code verification or force-supervision claim is made.

The newly visible P0.4 secondary-policy, capacity, reset and final-forward requirements were already covered. The closing review made torque coordinates/reference points explicit and added five isolated coordinate check groups. Existing tests, checkpoint loads, CPU/Warp force probe and historical experiments were not repeated.

[FILE_CHANGE_MAP.md](FILE_CHANGE_MAP.md) now covers reuse, minimal shared interfaces, new files, tests, dependencies, risks and report/output destinations for every P1–P8 stage. Joint state/reset belongs in P1; WM migration tests begin in P3; narrow imagination hooks belong in P5; integrated actor/critic migration and entry wiring remain P7.

## Preserved contracts and findings

Robot state/canonical71, action29, xi33 and residual70 remain unchanged. Joint state84/canonical86 and heads70+12+6 are planned extensions: only physical82 is integrated; force6 remains auxiliary. No HOI model/state/Replay component was implemented in P0.

Both wrist-pitch and wrist-yaw collision parents are required per hand. A single bread-box secondary body covers seven geoms. Netforce reports primary-on-secondary in world axes; hand-primary therefore gives force on the box. History is newest-first, sampled once per 5 ms substep. Final `forward()` refreshes instantaneous data without a fifth history entry; terminal interval history must be copied before reset. There is no built-in found history.

Torque columns are about their own force-weighted contact centroids. Combining torque diagnostics requires translation to a common reference point; force vectors sum directly. Box-local components use inverse Box rotation. P2.1 prefers contemporaneous substep orientations and permits an explicitly documented interval anchor; P0 does not silently select post-step orientation.

The original replay successor-row done discrepancy and repeated final-frame imagination rewards remain recorded for P2/P5, without code fixes. Support-relative context remains a time-varying candidate separate from xi33; P3.4 explicitly requires paired prediction/ablation before freezing it. See [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md) and [DEPENDENCY_CONTACT_AUDIT.md](DEPENDENCY_CONTACT_AUDIT.md).

## Validation and stop point

The five new isolated check groups passed: origins/read-write, quaternion integration, synthetic reference velocities, Box-local vectors and torque-reference algebra. Quaternion error versus MuJoCo was `1.11e-16`; reference angular-velocity error was at most `2.45e-14` rad/s. The original seven-test PASS and prior sensor/history/checkpoint evidence were reused. These checks do not replace P1 physical calibration.

Closing preservation checks verify source, datasets, assets, checkpoints, accepted reference, original machine evidence and user files. Only the documentation paths listed in [CLOSURE.md](CLOSURE.md) differ. Exact commands and probe code are in [CLOSURE_CHECKS.md](CLOSURE_CHECKS.md); results are in [closure_checks.json](closure_checks.json).

**P0 can legitimately be marked PASS for review under the approved Workhorse fallback. Stop here.** No P1/P2+ work, training, commit or push is authorized by this result. Wait for the user's explicit review approval and separate instructions.
