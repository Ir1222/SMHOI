# Part C P0 audit report

**Status: PARTIAL — available P0 audit completed; complete-plan acceptance cannot be certified.** No P1 implementation, training, commit or push was performed. All P0 additions are documentation under this directory. The user must review the report/diff and explicitly approve before any stage commit or P1 work.

## Source of truth and scope limitation

The requested `docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md` does not exist. The discovered file is `docs/hoi/force contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md` (17,892 bytes, SHA256 `b0fd536b3fefe03bfcdbe10b80506bc615eef4bb79c67f9c2abf21df87908745`). It has 361 newline characters and ends at P0.4's `reduce="netforce" semantics;` bullet. It contains no detailed P1–P8 sections or the promised stage-report paths. The entire available file and 665-line V3 plan were read. No missing requirements were invented. The force-aware master overrides the older V3 Force Head restriction.

The user was asked to locate/restore the complete master while independent audit work continued. This report uses `docs/hoi/force_contact/p0/` as a provisional documentation destination; it does not rename or alter the user's master. Restore the remainder and reconcile its additional P0 outputs/checks before marking P0 PASS or implementing P1.

## Repository and preservation

Target `/home/han/pyx/SMHOI`: branch `main`, HEAD `845878416d544ae2bef94f4a43d2ce059bc8c4ed` (Part B acceptance). Initial tracked/index diff was empty. Existing untracked user files were `.vscode/settings.json`, `SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md`, and the master under `docs/hoi/force contact/`; all retained byte-for-byte. No applicable AGENTS.md was found in the target or ancestors.

| Local checkout | Branch / HEAD | State and role |
|---|---|---|
| SMHOI | main / 845878416d544ae2bef94f4a43d2ce059bc8c4ed | authoritative target |
| humanoid-amp-ppo | master / 114f766405efd3d90cbf5470a4b1c2d3e1da8f27 | remote is Ir1222/supermimic_carry; two existing untracked files preserved |
| SuperMimic-Reproduced | feature/supermimic-hoi / 2ea2664bc40cc5d73dfff811a561073ae31463b9 | local Ir1222/supermimic_pyx fork, not a clean CuboiLeo checkout; existing untracked VS Code settings |
| HDMI | main / 32282f6dcf26cae70b814d585ceb12cc38aa1b60 | discovered under `/tmp/supermimic_hoi_audit_20261104/HDMI`; official remote; clean |
| SUGAR | main / 01fe1234b57188412ec38775150f008f133e0ad4 | discovered alongside HDMI; official remote; clean |
| PhysHSI-carrybox | pose_amp_vel / c75efd8ce8ded98031281b5a9b00ed71390dbccd | local Ir1222 fork; untracked `torch` and `yaw_left.csv`; not asserted identical to official InternRobotics HEAD |

Repository records include an additional nearby humanoid-carrybox checkout for provenance discovery. Official remote HEADs were not fetched and no papers/reference algorithms were newly implemented. These local snapshots establish availability, not upstream parity. An external TensorBoard process for hdmi_robot_only was observed; no running training/evaluation job was found in the process snapshot. Nothing was stopped.

Final integrity checks rehashed **361 protected existing files** and **76 existing inventoried inputs**, including external datasets/assets and the candidate checkpoint: no changes. The committed Part B reference manifest/acceptance report and M6.5 manifest/hash list still match HEAD. Existing tracked/index diff remains empty.

## Verified data, assets and checkpoint

All specifically named NPZs and manifests in the source checkout exist. The original HDMI motion was located at its exact relative path in the discovered official checkout; its hash matches the source checkout copy. Eight NPZ archives were inspected for keys/shapes/dtypes/FPS and numeric finiteness. Every inventoried path has a status, size/hash when present, role and current consumer in the [dataset/asset report](DATASET_ASSET_AUDIT.md) and [machine inventory](artifact_inventory.json).

Accepted Part B archive SHA256 exactly matches both committed manifest and the user plan: `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`. All nine canonical robot fields match shape, dtype and bytes. The 12-field reference remains 840 samples at 50 Hz, with one global desired-contact signal ON at frames 260–549. Object velocities, actual hand forces and per-hand intent labels are absent; none was fabricated.

The selected M6.5 profile is `native_unchanged`; seven box collision geoms, mass 2 kg, COM `(0,0,-.2)`, inertia `(.0334,.02167,.04167)`, original static supports and native hand/wrist/foot collision remain unchanged. The actual manifest is `data/assets/hdmi_bread_box/asset_manifest_m6_5.json`, hash `1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`. Every declared source hash validated. Historical hand-contact/support FAILs and `training_ready=false` remain visible; the authorized P0 audit does not redo those experiments or turn them into a success claim.

Target `logs/hdmi_robot_only/ckpt_0012000.pt` is **MISSING**. A separately identified candidate in `/home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/` exists, SHA256 `16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7`. Policy, critic and WM strict loading all pass (iteration 12000, 4784 updates); current configured training dataset is byte-identical to the canonical robot NPZ. No training-time Git/data SHA is embedded, so provenance is supported but not fully attested. It was not copied, migrated, evaluated as a successful policy, or overwritten.

## Dependency/contact findings

Installed/locked mjlab is 1.6.0 at exactly `8bb2bc9e187797689ec82c78c35019afd2fd0376`; MuJoCo/Warp 3.16.0, Warp 1.18.0 and Torch 2.14.1+cu130 agree with the lock. The [dependency/contact audit](DEPENDENCY_CONTACT_AUDIT.md) traces matching, solver force extraction, history, environment/recorder order and CUDA capture against actual installed files.

Key implementation implications:

- Each selected hand comprises **wrist-pitch and wrist-yaw collision bodies**. Two independent side sensors must include both; yaw-only source sensors would omit native wrist contacts.
- A single box-body secondary includes all seven geoms. `secondary_policy="any"` removes filtering; `"first"` can hide ambiguous matches. Use explicit names and `"error"`.
- Netforce is a world-frame vector sum of force **primary exerts on secondary**. Hand primary→box secondary therefore reports force on box. A small CPU/Warp/captured-sensor probe verified sign and parity; it is not P1 physical calibration.
- Four history entries are recorded once per 5 ms substep, newest-first. The post-loop `forward()` refreshes instantaneous contact data without adding history. Capture terminal history before reset; reacquire buffers after each interval. There is no built-in found history.
- Actual root qvel translation is root-origin/world velocity; angular qvel is body-local. Offset-COM numeric checks verified the required conversion. Interval box-local force frame anchoring remains a P2 contract question for the missing master remainder.
- Match/contact/constraint capacities and overflow flags need explicit P1 checks. Existing robot-only defaults suppress overflow printing and cannot certify a compound HOI scene.

## Architecture findings and change map

The [architecture report](ARCHITECTURE_AUDIT.md) traces all requested implementations. Robot state71, canonical71, action29, xi33 and residual70 match the baseline. Joint state84, canonical86, physical residual82 and auxiliary force6 (combined88) are target extensions, not current features.

Current StateIO asserts nq36/nv35, so an object cannot simply be added while retaining the recorder. Current HOI code is offline reference/asset inspection; `smHOI-track` is not registered. `Trainer.run()` can be preserved with a narrow optional construction/runtime factory and later policy-trainer hooks, defaulting to the existing robot-only path.

Two concrete gaps were reproduced: successor-row replay termination labels duplicate a terminal flag onto the previous transition (`[1,1]` instead of `[0,1]`), and imagined rollouts repeat clamped final-frame references without a valid-transition mask. Neither was fixed in P0. Support-relative context also needs explicit time-varying representation: robot canonical features erase global xy/yaw needed to locate fixed supports; it cannot silently become part of xi33.

[FILE_CHANGE_MAP.md](FILE_CHANGE_MAP.md) identifies P1 scene/config/sensor/reset/calibration files and separately maps later-stage extensions. P1 can remain isolated from WM/replay/imagination/Trainer implementation. No P2+ files were created. New filenames and exact report destination remain provisional pending the complete master.

## Checks and review gate

[VALIDATION.md](VALIDATION.md) contains the complete check matrix and exact executed regression command. **Seven non-training baseline tests passed.** Artifact/source hashes, installed history/reset bookkeeping, offset-COM frames, strict checkpoint shapes and isolated CPU/Warp sensor sign/capture checks passed within their documented scope. Physical task success, full HOI scene calibration, force-gradient fidelity and checkpoint migration were not claimed.

All new P0 documents and raw evidence are in this directory. `git diff HEAD --` and staged diff are empty because these are new untracked files; the separate review patch includes their complete contents. No staging, commit or push occurred. Existing user changes are excluded from the proposed P0 commit.

**Stop after P0 for review.** Restore the complete master, review this report and the full documentation diff, and explicitly approve any P0 commit/P1 continuation. Approval of a report alone must not be inferred from elapsed time. P0 and P1 will remain separate commits after their respective reviews; P2 or later is outside the current authorization.
