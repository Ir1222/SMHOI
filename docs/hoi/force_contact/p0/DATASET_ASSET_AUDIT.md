# P0 dataset and asset inventory

Audit date: 2026-11-10. Read-only inventory of the exact Section 3 inputs plus discovered provenance files. Full absolute paths, complete SHA256, NPZ field schemas/array hashes, and manifest hash declarations are in [artifact_inventory.json](artifact_inventory.json). No conversion, rendering, M0–M6.5 rerun, or Part B regeneration was performed.

## Exact requested artifacts

`S` = `/home/han/pyx/humanoid-amp-ppo`; `H` = `/tmp/supermimic_hoi_audit_20261104/HDMI`; `R` = `/home/han/pyx/SMHOI`. These are discovered paths, not assumed aliases.

| Path | Status | Bytes | SHA256 | Role / current consumer |
|---|---|---:|---|---|
| `R/docs/hoi/force_contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md` | MISSING | — | `—` | user planning input; P0 read only |
| `R/docs/hoi/force contact/SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md` | EXISTS | 17892 | `b0fd536b3fefe03bfcdbe10b80506bc615eef4bb79c67f9c2abf21df87908745` | user planning input; P0 read only |
| `R/SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md` | EXISTS | 27439 | `168465ae0a2e17153902a337963574942551909680d0d8dd36f7ffb22102b05c` | user planning input; P0 read only |
| `/home/han/pyx/hdmi_bread_box_robot_only_50hz_supermimic_pyx.npz` | EXISTS | 882198 | `8da85c075132ac9a887d435c6c3e92abc9907e3e0cb15d854e69864b2fdf9527` | canonical robot-only reference; Part B offline loader default; sm-train only if selected via data.npz_dir/data.clips; candidate external hdmi.npz is byte-identical |
| `/home/han/pyx/hdmi_bread_box_hoi_v1.npz` | EXISTS | 897802 | `d89da14d49cc5f86398f3a86ad6037e777f5588972eaf7441917833da4b04dbe` | derived HOI source/reference; source hoi/env.py DEFAULT_MOTION -> HOIReference; target Part B default source; no target HOI training entry |
| `S/dataset/tracking/dual_arm1_box.npz` | EXISTS | 962502 | `6396ebd4a905427b47ceed7f3b5c1d72ac3e1650295ee8a461977e55ef749548` | Lift derived tracking dataset; source tasks/smtrack/env_cfgs.py lift default; not target HDMI scene |
| `S/dataset/hoi/generated/hdmi_bread_box_pcp_v2.npz` | EXISTS | 905919 | `5f33bdde5c005130a26dc9450bd71d98436182b07eaca2b63517663c75ae1aca` | derived HDMI intermediate; source hoi/reference.py derivation and hoi_hdmi_viewer.py / hoi_reference_audit.py defaults; not target training default |
| `S/dataset/hoi/generated/hdmi_bread_box_hoi_v1.npz` | EXISTS | 897802 | `d89da14d49cc5f86398f3a86ad6037e777f5588972eaf7441917833da4b04dbe` | derived HOI source/reference; source hoi/env.py DEFAULT_MOTION -> HOIReference; target Part B default source; no target HOI training entry |
| `S/dataset/hoi/manifests/hdmi_bread_box_hoi_v1.json` | EXISTS | 1419 | `3d2dafff9ab0bf0dbd29b36f128ca404042c38dcbf2f39d2ed6e292743b942a4` | source conversion/asset manifest; source HOI conversion, collider/export/audit tools; target M6.5 provenance verification only |
| `S/dataset/hoi/manifests/hdmi_bread_box_pcp_v2.json` | EXISTS | 2249 | `1a18fac661fa19e8aa5da8f747340524a88e99b58520e2ddbaa7fb831d01fba8` | source conversion/asset manifest; source HOI conversion, collider/export/audit tools; target M6.5 provenance verification only |
| `S/dataset/hoi/manifests/hdmi_usd_contact_assets.json` | EXISTS | 27941 | `ad7776c36fc12e9c2da1fda4f80e650d2fb8ce4ca88ce5c7be08bea76edefe06` | source conversion/asset manifest; source HOI conversion, collider/export/audit tools; target M6.5 provenance verification only |
| `S/dataset/hoi/manifests/hdmi_usd_support_assets.json` | EXISTS | 3674 | `5b5fdece8de408e7f5cd867ae4aa4d3e3dd09523908b51b9d98856cda68d735d` | source conversion/asset manifest; source HOI conversion, collider/export/audit tools; target M6.5 provenance verification only |
| `S/dataset/external/hdmi/carry_and_place_bread_box-0829/motion.npz` | EXISTS | 1666169 | `ff35558659a4b7fca63df6fc686cfd518b40bbc2cf4e3b5063e69e0ea73d7344` | original HDMI source motion; HDMI task YAML data_path -> HDMI command motion loading; source hdmi.py conversion (not rerun) |
| `S/dataset/external/hdmi/carry_and_place_bread_box-0829/meta.json` | EXISTS | 1945 | `28f02a2f4fe3ccd5c8582f301e0b557d5df0ef5ae80bb762e8b1419dba2d2f4b` | source metadata; source motion metadata/provenance inspection |
| `H/data/motion/data_for_sim/carry_and_place_bread_box-0829/motion.npz` | EXISTS | 1666169 | `ff35558659a4b7fca63df6fc686cfd518b40bbc2cf4e3b5063e69e0ea73d7344` | original HDMI source motion; HDMI task YAML data_path -> HDMI command motion loading; source hdmi.py conversion (not rerun) |
| `H/active_adaptation/assets_mjcf/objects/bread_box/bread_box.xml` | EXISTS | 1559 | `b91114480649634e49cd9f16e577109e01860c6d7607176a5b4e3f02a38087c6` | source physical asset; XML: simulator/spec source; USD: historical extraction/provenance; target selected configuration is hashed M6.5 manifest |
| `H/cfg/task/G1/hdmi/carry_and_place_bread_box.yaml` | EXISTS | 1448 | `a8ff4820b8e51c6a7dce5fbb92257f7d3331ce26d93085c578510bbcc4546561` | HDMI task configuration; official HDMI task configuration; not target runtime defaults |
| `H/active_adaptation/envs/mdp/commands/hdmi/command.py` | EXISTS | 33751 | `05c483224f052f9e26436ef93c697fbfb9e108a1c9bd1c48c234c9b6f85f4eba` | source implementation; inspection and manifest source hashing; source HOI files are not imported by target Trainer |
| `H/active_adaptation/envs/mdp/commands/hdmi/rewards.py` | EXISTS | 24011 | `28d738424974400ba979060e2534aa2f50524bb3eb09102d10db4f57e7ecc804` | source implementation; inspection and manifest source hashing; source HOI files are not imported by target Trainer |
| `R/outputs/hoi/asset_audit/part_b/reference/reference.npz` | EXISTS | 888300 | `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b` | accepted reference; scripts/hoi/replay_reference.py -> ReferenceClip; no current training entry consumes HOI fields |
| `R/outputs/hoi/asset_audit/part_b/reference/reference_manifest.json` | EXISTS | 4436 | `129e1181fda2528250d2cd64fa214c2de4d7998ed37698c17302cc4f58128efb` | accepted reference manifest; ReferenceClip hash/schema validation; reviewed provenance |
| `R/outputs/hoi/asset_audit/part_b/DATA_ACCEPTANCE.md` | EXISTS | 9004 | `17972d7e7b3c6bbae395418eee9055dabcf199bf56ad84510a37b6a4d19d20fe` | accepted Part B report; review/provenance only |
| `R/data/assets/hdmi_bread_box/asset_manifest_m6_5.json` | EXISTS | 33590 | `1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1` | physical asset manifest; hoi.asset_manifest.load_manifest / offline audit scene and Part B viewer; no runtime HOI scene |
| `R/data/assets/hdmi_bread_box/asset_manifest.json` | EXISTS | 31046 | `8cee449a0af34365e28b3265479436fb72a6371f0a065b0c87f2d199e4c400fb` | physical asset manifest; hoi.asset_manifest.load_manifest / offline audit scene and Part B viewer; no runtime HOI scene |
| `R/docs/hoi/asset_audit/m6_5/artifact_hashes.json` | EXISTS | 24431 | `b26171ca9e8573bc6124059442759278df00d4cdb04aec42ec186bb0584ea5e9` | historical reconciliation evidence; review/provenance only; no replay or training loader |
| `R/logs/hdmi_robot_only/ckpt_0012000.pt` | MISSING | — | `—` | candidate robot-only checkpoint; target Trainer.load only with explicit --resume; no target HOI migration loader exists |
| `/home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt` | EXISTS | 215360015 | `16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7` | candidate robot-only checkpoint; target Trainer.load only with explicit --resume; no target HOI migration loader exists |

## Every NPZ schema

All numeric arrays checked finite; no pickle was enabled. Field names and dtypes below are read from disk. The two copies of raw HDMI motion have identical archive hashes, as do the two HOI-v1 copies. Raw HDMI has 31 bodies including object/support entries; the converted robot references have 30 robot bodies after documented completion of two waist links.


### `/home/han/pyx/hdmi_bread_box_robot_only_50hz_supermimic_pyx.npz`

FPS: `[50]`.

| Key | Shape | dtype |
|---|---|---|
| `fps` | `[1]` | `int64` |
| `joint_pos` | `[840, 29]` | `float32` |
| `joint_vel` | `[840, 29]` | `float32` |
| `body_pos_w` | `[840, 30, 3]` | `float32` |
| `body_quat_w` | `[840, 30, 4]` | `float32` |
| `body_lin_vel_w` | `[840, 30, 3]` | `float32` |
| `body_ang_vel_w` | `[840, 30, 3]` | `float32` |
| `body_names` | `[30]` | `<U25` |
| `joint_names` | `[29]` | `<U26` |

### `/home/han/pyx/hdmi_bread_box_hoi_v1.npz`

FPS: `[50]`.

| Key | Shape | dtype |
|---|---|---|
| `joint_pos` | `[840, 29]` | `float32` |
| `joint_vel` | `[840, 29]` | `float32` |
| `body_pos_w` | `[840, 30, 3]` | `float32` |
| `body_quat_w` | `[840, 30, 4]` | `float32` |
| `body_lin_vel_w` | `[840, 30, 3]` | `float32` |
| `body_ang_vel_w` | `[840, 30, 3]` | `float32` |
| `box_pos_w` | `[840, 3]` | `float32` |
| `box_quat_w` | `[840, 4]` | `float32` |
| `box_lin_vel_w` | `[840, 3]` | `float32` |
| `box_ang_vel_w` | `[840, 3]` | `float32` |
| `object_contact` | `[840]` | `bool` |
| `contact_target_root` | `[2, 3]` | `float32` |
| `support_pos_w` | `[840, 2, 3]` | `float32` |
| `support_quat_w` | `[840, 2, 4]` | `float32` |
| `phase_bounds` | `[5, 2]` | `int64` |
| `phase_names` | `[5]` | `<U16` |
| `body_names` | `[30]` | `<U64` |
| `schema_version` | `[1]` | `int64` |
| `contact_eef_offset` | `[3]` | `float32` |
| `contact_target_provenance` | `[]` | `<U52` |
| `contact_label_semantics` | `[]` | `<U60` |
| `phase_bounds_semantics` | `[]` | `<U77` |
| `joint_names` | `[29]` | `<U64` |
| `fps` | `[1]` | `int64` |
| `hoi_phase_bounds` | `[6, 2]` | `int64` |
| `hoi_phase_names` | `[6]` | `<U9` |
| `hoi_velocity_contract` | `[]` | `<U21` |
| `hoi_source_sha256` | `[]` | `<U64` |

### `/home/han/pyx/humanoid-amp-ppo/dataset/tracking/dual_arm1_box.npz`

FPS: `[50]`.

| Key | Shape | dtype |
|---|---|---|
| `fps` | `[1]` | `int64` |
| `joint_pos` | `[520, 29]` | `float32` |
| `joint_vel` | `[520, 29]` | `float32` |
| `body_pos_w` | `[520, 30, 3]` | `float32` |
| `body_quat_w` | `[520, 30, 4]` | `float32` |
| `body_lin_vel_w` | `[520, 30, 3]` | `float32` |
| `body_ang_vel_w` | `[520, 30, 3]` | `float32` |
| `box_pos_w` | `[520, 3]` | `float32` |
| `box_quat_w` | `[520, 4]` | `float32` |
| `box_lin_vel_w` | `[520, 3]` | `float32` |
| `box_ang_vel_w` | `[520, 3]` | `float32` |
| `grasp_frame` | `[]` | `int64` |
| `box_half_extents` | `[3]` | `float32` |
| `box_forward_offset` | `[]` | `float32` |

### `/home/han/pyx/humanoid-amp-ppo/dataset/hoi/generated/hdmi_bread_box_pcp_v2.npz`

FPS: `[50]`.

| Key | Shape | dtype |
|---|---|---|
| `joint_pos` | `[840, 29]` | `float32` |
| `joint_vel` | `[840, 29]` | `float32` |
| `body_pos_w` | `[840, 30, 3]` | `float32` |
| `body_quat_w` | `[840, 30, 4]` | `float32` |
| `body_lin_vel_w` | `[840, 30, 3]` | `float32` |
| `body_ang_vel_w` | `[840, 30, 3]` | `float32` |
| `box_pos_w` | `[840, 3]` | `float32` |
| `box_quat_w` | `[840, 4]` | `float32` |
| `box_lin_vel_w` | `[840, 3]` | `float32` |
| `box_ang_vel_w` | `[840, 3]` | `float32` |
| `object_contact` | `[840]` | `bool` |
| `contact_target_root` | `[2, 3]` | `float32` |
| `support_pos_w` | `[840, 2, 3]` | `float32` |
| `support_quat_w` | `[840, 2, 4]` | `float32` |
| `phase_bounds` | `[5, 2]` | `int64` |
| `phase_names` | `[5]` | `<U16` |
| `body_names` | `[30]` | `<U64` |
| `schema_version` | `[1]` | `int64` |
| `contact_eef_offset` | `[3]` | `float32` |
| `contact_target_provenance` | `[]` | `<U52` |
| `contact_label_semantics` | `[]` | `<U60` |
| `phase_bounds_semantics` | `[]` | `<U77` |
| `joint_names` | `[29]` | `<U64` |
| `fps` | `[1]` | `int64` |

### `/home/han/pyx/humanoid-amp-ppo/dataset/hoi/generated/hdmi_bread_box_hoi_v1.npz`

FPS: `[50]`.

| Key | Shape | dtype |
|---|---|---|
| `joint_pos` | `[840, 29]` | `float32` |
| `joint_vel` | `[840, 29]` | `float32` |
| `body_pos_w` | `[840, 30, 3]` | `float32` |
| `body_quat_w` | `[840, 30, 4]` | `float32` |
| `body_lin_vel_w` | `[840, 30, 3]` | `float32` |
| `body_ang_vel_w` | `[840, 30, 3]` | `float32` |
| `box_pos_w` | `[840, 3]` | `float32` |
| `box_quat_w` | `[840, 4]` | `float32` |
| `box_lin_vel_w` | `[840, 3]` | `float32` |
| `box_ang_vel_w` | `[840, 3]` | `float32` |
| `object_contact` | `[840]` | `bool` |
| `contact_target_root` | `[2, 3]` | `float32` |
| `support_pos_w` | `[840, 2, 3]` | `float32` |
| `support_quat_w` | `[840, 2, 4]` | `float32` |
| `phase_bounds` | `[5, 2]` | `int64` |
| `phase_names` | `[5]` | `<U16` |
| `body_names` | `[30]` | `<U64` |
| `schema_version` | `[1]` | `int64` |
| `contact_eef_offset` | `[3]` | `float32` |
| `contact_target_provenance` | `[]` | `<U52` |
| `contact_label_semantics` | `[]` | `<U60` |
| `phase_bounds_semantics` | `[]` | `<U77` |
| `joint_names` | `[29]` | `<U64` |
| `fps` | `[1]` | `int64` |
| `hoi_phase_bounds` | `[6, 2]` | `int64` |
| `hoi_phase_names` | `[6]` | `<U9` |
| `hoi_velocity_contract` | `[]` | `<U21` |
| `hoi_source_sha256` | `[]` | `<U64` |

### `/home/han/pyx/humanoid-amp-ppo/dataset/external/hdmi/carry_and_place_bread_box-0829/motion.npz`

FPS: `{'npz_field': 'ABSENT', 'metadata_value': 50, 'metadata_path': '/home/han/pyx/humanoid-amp-ppo/dataset/external/hdmi/carry_and_place_bread_box-0829/meta.json'}`.

| Key | Shape | dtype |
|---|---|---|
| `body_pos_w` | `[840, 31, 3]` | `float64` |
| `body_quat_w` | `[840, 31, 4]` | `float64` |
| `joint_pos` | `[840, 29]` | `float64` |
| `body_lin_vel_w` | `[840, 31, 3]` | `float64` |
| `body_ang_vel_w` | `[840, 31, 3]` | `float64` |
| `joint_vel` | `[840, 29]` | `float64` |
| `object_contact` | `[840, 1]` | `bool` |

### `/tmp/supermimic_hoi_audit_20261104/HDMI/data/motion/data_for_sim/carry_and_place_bread_box-0829/motion.npz`

FPS: `{'npz_field': 'ABSENT', 'metadata_value': 50, 'metadata_path': '/home/han/pyx/humanoid-amp-ppo/dataset/external/hdmi/carry_and_place_bread_box-0829/meta.json'}`.

| Key | Shape | dtype |
|---|---|---|
| `body_pos_w` | `[840, 31, 3]` | `float64` |
| `body_quat_w` | `[840, 31, 4]` | `float64` |
| `joint_pos` | `[840, 29]` | `float64` |
| `body_lin_vel_w` | `[840, 31, 3]` | `float64` |
| `body_ang_vel_w` | `[840, 31, 3]` | `float64` |
| `joint_vel` | `[840, 29]` | `float64` |
| `object_contact` | `[840, 1]` | `bool` |

### `/home/han/pyx/SMHOI/outputs/hoi/asset_audit/part_b/reference/reference.npz`

FPS: `[50]`.

| Key | Shape | dtype |
|---|---|---|
| `joint_pos` | `[840, 29]` | `float32` |
| `joint_vel` | `[840, 29]` | `float32` |
| `body_pos_w` | `[840, 30, 3]` | `float32` |
| `body_quat_w` | `[840, 30, 4]` | `float32` |
| `body_lin_vel_w` | `[840, 30, 3]` | `float32` |
| `body_ang_vel_w` | `[840, 30, 3]` | `float32` |
| `joint_names` | `[29]` | `<U26` |
| `body_names` | `[30]` | `<U25` |
| `fps` | `[1]` | `int64` |
| `object_pos_w` | `[840, 3]` | `float32` |
| `object_quat_w` | `[840, 4]` | `float32` |
| `object_contact` | `[840]` | `bool` |

## Provenance and current entry points

The source HOI manifest links PCP-v2 (`5f33bdde…`) to HOI-v1 (`d89da14d…`) and hashes its derivation code. PCP-v2 links raw motion (`ff355586…`), metadata (`28f02a2f…`) and native G1 XML (`febdcbe…`). HOI-v1 contains derived link-origin velocities and both older five-interval and newer six-interval phase metadata. Those phase fields are historical source data, not permitted Part C inputs. Raw HDMI velocity reference-point provenance remains unresolved; the accepted reference uses the already accepted canonical robot arrays.

The contact/support USD manifests identify extracted source geometry. Their complete hashes are inventoried; current M6.5 `load_manifest(..., verify_sources=True)` successfully verified every source it declares. Nothing was regenerated from USD.

Only `sm-train = supermimic.scripts.train:main` is registered. It reads `data.library` if supplied (packed `.pt`, compact mode); otherwise it discovers `*.npz` under `data.npz_dir`, optionally filtered by `data.clips` and a CSV `data.manifest`. Default `data/npz/lafan1` is absent in this target checkout. It has no hard-coded HDMI selection and ignores HOI fields in an NPZ. `smHOI-track` does not exist. The actual external candidate run config uses `data/npz/HDMI_RoboOnly/hdmi.npz`, hash-identical to the canonical robot NPZ; no training process was observed.

Source `hoi/env.py` uses HOI-v1, 2 ms × 10 stepping, a different dependency baseline (mjlab v1.4.0 lock/source), HDMI hand/foot adaptations, 35-D context and phase-based logic. It is an inspection reference, not a drop-in target implementation. Target remains 5 ms × 4, native M6.5 hands/feet and 33-D episode context.

## Accepted Part B and physical profile

The committed reference manifest, actual file and stated expected SHA256 all agree: `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`. All nine robot fields match the canonical robot archive by shape, dtype and byte hash. The accepted NPZ has exactly 12 fields, 840 samples, 50 Hz, last sample 16.78 s, coverage 16.80 s. Global desired-contact intent is ON at 260–549 inclusive. It has neither object velocity nor measured force nor per-hand labels. No derived cache was created.

The exact committed M6.5 manifest is `data/assets/hdmi_bread_box/asset_manifest_m6_5.json`, not a guessed file under outputs. Its hash is `1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`. The reconciliation hash list is `docs/hoi/asset_audit/m6_5/artifact_hashes.json`; Part B artifacts live under the specified outputs root. Historical outputs/records are preserved.

Selected profile: `native_unchanged`; nominal box mass 2 kg; COM `(0,0,-0.2)` m; principal inertia `(0.0334,0.02167,0.04167)` kg m². All seven boxes share one free body. Collision friction `(1,.005,.0001)`, `condim=3`, priority 1, masks 1/1. Priority 1 is a documented deviation from XML priority 0. The COM below the collision envelope and non-equivalence to PhysX remain explicit.

| Box geom | Root-local position (m) | Half extents (m) |
|---|---|---|
| bottom | `[0.0, 0.0, 0.0]` | `[0.09, 0.17, 0.01]` |
| side1 | `[-0.08, 0.0, 0.13]` | `[0.01, 0.17, 0.13]` |
| side2 | `[0.08, 0.0, 0.13]` | `[0.01, 0.17, 0.13]` |
| side3 | `[0.0, 0.16, 0.13]` | `[0.09, 0.01, 0.13]` |
| side4 | `[0.0, -0.16, 0.13]` | `[0.09, 0.01, 0.13]` |
| handle_left | `[0.0, 0.17, 0.21]` | `[0.03, 0.02, 0.01]` |
| handle_right | `[0.0, -0.17, 0.21]` | `[0.03, 0.02, 0.01]` |

Two supports remain static, half extents `(0.1,0.1,0.02)` m. Root positions are `(-1.4667341709,-0.1597542167,0.6605498791)` and `(1.3813182116,-0.2843948603,0.6251227856)` m, both wxyz quaternion `(0.0290281866,0,0,0.9995785952)`. Targets `(0,±.17,.14)` are box-root-local; EEF offsets `(.05,0,0)` are wrist-yaw-local. See [audit_checks.json](audit_checks.json) for exact numeric records and current native collider bodies.

The historical gates remain `physical_hand_contact=FAIL`, `scene_support_validity=FAIL`, `training_ready=false`, `status=M6_5_NO_GO_M7`. Prior measured native worst overlaps were 37.22/44.99 mm; endpoint support gaps were 10 mm. These are cited historical results, not rerun P0 experiments. A P1 sensor calibration pass must not be reported as successful full-motion interaction or approval for learning.

## Checkpoint location and limits

`R/logs/hdmi_robot_only/ckpt_0012000.pt` is MISSING. A separately discovered candidate at `/home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt` EXISTS (215360015 bytes; SHA256 `16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7`). It is not silently substituted for the missing target path. Strict robot-only policy/critic/WM loading passed. Iteration 12000; 4784 updates. Config and actual current dataset support its provenance, but it contains no embedded training Git SHA or dataset hash, and no policy-quality evaluation was performed. See [checkpoint_audit.json](checkpoint_audit.json) and [checkpoint_files.json](checkpoint_files.json).
