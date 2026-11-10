# Part B — data acceptance

**B1, B2 and B3 complete for motion-data preparation and kinematic inspection.** The processed reference is ready as a data artifact for a separately authorized implementation stage. **M7 / HOI learning remains NO-GO under the unchanged M6.5 physical manifest.** This report does not replace that physical decision.

## Artifact and contracts

- [Processed NPZ](reference/reference.npz), SHA256: `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`.
- [Reviewed asset manifest](../../../../data/assets/hdmi_bread_box/asset_manifest_m6_5.json), SHA256: `1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`; `training_ready=false`.
- 840 samples at 50 Hz; final sample timestamp **16.78 s**, complete sample coverage/video duration **16.80 s**. Quaternions are wxyz; world is Z-up, positions in metres.
- Twelve fields: `joint_pos`, `joint_vel` [840,29]; `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w` [840,30,3]; `body_quat_w` [840,30,4]; `joint_names` [29], `body_names` [30], `fps` [1]; `object_pos_w` [840,3], `object_quat_w` [840,4], `object_contact` [840]. Numeric motion fields retain float32; contact bool; fps int64; canonical names U26/U25.
- All robot fields come from robot-only, including exact dtype and six numeric-array byte hashes. Shared HOI robot values/order were checked before merging; its wider Unicode storage is documented. Only object pose/contact are imported from HOI; no retargeting, interpolation, smoothing, normalization or velocity recalculation.
- Constant target/EEF/support geometry stays in the existing hashed asset manifest; the optional `contact_target_root` NPZ field is omitted. There are no object velocities, phase boundaries or redundant support arrays. Contact is desired intent only, ON at frames 260–549.

[Reference provenance and array hashes](reference/reference_manifest.json), [B1 validation](reference/VALIDATION.md), [display-only experiment profile](reference/experiment_profile.json).

## Viewer and numerical acceptance

The viewer loads the processed NPZ once, validates its hash and the asset hash/provenance, maps 29 joint names, assigns the recorded pelvis and object root poses, then calls `mj_forward()`. Native G1 colliders, seven box geoms and two fixed support transforms come from the reviewed configuration. No actuators, welds, mocap or physics stepping exist in this viewer. Zeroed display velocity buffers do not replace any saved velocity reference. EEF overlays use recorded wrist poses; target sites use the box-local manifest points. Floating support slabs represent the actual fixed collision bodies; decorative table legs are not invented.

All **840** frames checked ([per-frame CSV](inspection/frames.csv), [metrics](inspection/inspection.json)):

| Check | Maximum error |
|---|---:|
| All 30 robot link positions vs saved reference | 8.292e-8 m |
| All robot link orientations vs saved reference | 8.294e-8 rad |
| Object position / orientation | 0 m / 2.220e-16 rad |
| Box-local target → world site | 3.153e-16 m |
| EEF from rendered robot FK vs recorded wrist reference | 8.119e-8 m |
| Fixed support positions / orientations | 0 m / 0 rad |

Named joint qpos assignments are exact. Quaternion unit norms are checked at 1e-5 without rewriting source samples. No indexing wrap or conversion-induced discontinuity was found. Adjacent source maxima: pelvis translation **18.872 mm**, object translation **27.273 mm**, pelvis rotation **0.05091 rad**, object rotation **0.04708 rad** per 20 ms sample. The largest joint change is **0.317981 rad**, left ankle pitch, **647→648** (next sample also changes rapidly). This is present in the canonical source and preserved; acceptance does not assert a smooth or physically feasible motion.

The exported [full video](inspection/reference_50hz.mp4) decodes to **840 frames at exactly 50/1 fps**, 1200×720 ([ffprobe evidence](video_validation.json)). Playback uses elapsed time without interpolating poses; pauses at frame 839. Single-step clamps at endpoints, reset returns to zero, speeds range 0.125–8×. Future windows edge-pad with explicit invalid masks; no training integration exists.

## Visual inspection

Reviewed [overview montage](inspection/overview_montage.png) and corresponding close-ups. These are inspection bookmarks, not phase annotations or motion fields.

| Frame | Time (s) | Intent | EEF-target L / R (mm) | Visual observation |
|---|---:|---|---:|---|
| 0 | 0.00 | OFF | 1344.82 / 1328.74 | Box aligned above support0; original gap visible; robot approaching. |
| 260 | 5.20 | ON | 46.94 / 40.61 | First intent ON; hands near walls; EEF and target markers remain distinct. |
| 280 | 5.60 | ON | 40.93 / 37.91 | Box lifted above support0; target transforms follow the box. |
| 298 | 5.96 | ON | 72.59 / 63.59 | Clear lift; marker errors grow, as already recorded in the source. |
| 400 | 8.00 | ON | 48.26 / 43.58 | Carry between fixed supports; native visual hands intersect walls. |
| 530 | 10.60 | ON | 11.98 / 14.51 | Box approaches support1; small marker error does not certify contact. |
| 549 | 10.98 | ON | 11.47 / 18.21 | Last intent ON sample; hands remain beside the box. |
| 550 | 11.00 | OFF | 10.97 / 19.09 | Intent OFF; hands have not instantly detached. This is a label transition. |
| 839 | 16.78 | OFF | 1433.26 / 1425.79 | Robot away; box above support1; original endpoint gap retained. |

Robot/object alignment and marker axes agree with numerical transforms. Both supports stay fixed. Near-side markers and offset lines are visible; far-side/inside-hand markers can be occluded by real rendered geometry. Camera and collider-view controls allow alternate inspection. The known hand/wall inconsistencies and approximately 10 mm endpoint support gaps are retained, not corrected by this display.

## Interaction verification and limitations

The native passive viewer implementation supports Space, Left/Right, +/−, R/Backspace, camera modes 1/2/3 and C collider view. Its actual window launch is **blocked by this session's display server** (no GLX framebuffer configurations; software-GLX and EGL-window probes also failed). Logs are preserved. Native desktop interaction is not claimed as validated.

The **local browser fallback works with EGL offscreen rendering**. Actual HTTP requests rendered frames 0/260/400/550/839, cameras and collider view, with correct intent/time/distance payloads. Node tests exercised the real JS button, keyboard and slider handlers, playback timing, endpoints, render loop and error states. No installed desktop browser was available for a manual browser-window test. Slow rendering may skip displayed samples in real-time playback; pause/single-step accesses every frame and all recorded samples remain unchanged. This is a display performance limitation, not resampling.

**55 focused tests passed**: 36 new Part B tests plus 19 unchanged M6/M6.5 regressions ([log](final_tests.txt)); one upstream Torch deprecation warning. Ruff and git whitespace checks passed. All **305** pre-existing tracked files, the user's plan, both source NPZs, historical asset manifests/reports and source repository status are unchanged ([integrity result](preservation_final.json)). Checkpoints and training files were not written. No long training, environment integration, WM/Actor/Critic/replay changes or new training entry point was introduced.

## Reproduce and inspect

Run from `/home/han/pyx/SMHOI` using the existing virtualenv. A new output directory is required; regeneration refuses to overwrite experiments.

```bash
# Regenerate the identical bundle in a fresh directory.
.venv/bin/python scripts/hoi/process_reference.py \
  --output-dir outputs/hoi/asset_audit/part_b/reference_regenerated

# Interactive viewer recommended for this machine; open the printed localhost URL.
MUJOCO_GL=egl .venv/bin/python scripts/hoi/replay_reference.py \
  --display-only --browser

# Inspect the regenerated artifact instead, if desired.
MUJOCO_GL=egl .venv/bin/python scripts/hoi/replay_reference.py \
  --display-only --browser \
  --reference outputs/hoi/asset_audit/part_b/reference_regenerated/reference.npz

# Native GUI on a machine/session with a working GLX context.
MUJOCO_GL=glfw .venv/bin/python scripts/hoi/replay_reference.py --display-only

# Reproduce all-frame checks, selected snapshots and full 50 Hz video.
MUJOCO_GL=egl .venv/bin/python scripts/hoi/replay_reference.py \
  --display-only --headless --video \
  --output-dir outputs/hoi/asset_audit/part_b/inspection_regenerated

.venv/bin/python -m pytest tests/test_hoi_reference.py \
  tests/test_hoi_reference_viewer.py tests/test_hoi_reference_browser.py \
  tests/test_hoi_asset_audit.py tests/test_hoi_reconciliation.py -q
```

Exact new/changed files are listed in [FILES.md](FILES.md); incremental decisions and commit groups are in [WORK_LOG.md](WORK_LOG.md). Stop after Part B and await review. Physical acceptance and training authorization remain outstanding.
