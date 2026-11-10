# Part B execution record

## B1 — exact motion reference conversion

Read the complete unchanged V3 plan, previous audit reports and M6.5 review manifest. No applicable AGENTS.md found in either repository or parent directories. Source repository remains read-only with its two pre-existing untracked files. Target starts at 43a7761; the user's untracked implementation plan is preserved and excluded from commits. `preservation_baseline.json` records every pre-existing tracked file hash plus plan hash and source status.

The plan's only explicit output root is `outputs/hoi/asset_audit`; it does not name a separate Part B directory. This implementation uses its isolated `part_b/` child. `reference/` contains the NPZ, reference manifest, validation JSON/Markdown and experiment profile. Fixed target/EEF/support data stay in the hashed M6.5 asset manifest; the optional constant target NPZ field is omitted to keep one geometry authority.

Commands executed:

```bash
.venv/bin/python scripts/hoi/process_reference.py
.venv/bin/python -m pytest tests/test_hoi_reference.py -q
```

Result: **22 passed in 0.61 s**. B1 is accepted before B2 starts. Tests cover all nine canonical robot fields including dtype/bytes, object pose/contact identity, shared-field mismatch rejection, 840 frames/50 Hz, quaternion validity, deterministic outputs, immutable inputs, invalid data rejection, frame bounds and explicit terminal future-window masks. The six robot numeric arrays are unchanged; shared Unicode width differences are documented. No quaternions or velocities are rewritten.

Processed SHA256: `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`.
Asset SHA256: `1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`.

The profile is display-only with training_ready=false. B1 tests do not approve physical contact or training.

B1 lint follow-up: initial lint found import formatting, a non-executable shebang and an unnecessary map expression. Corrected without changing conversion behavior. Ruff passes; the 22 B1 tests pass again. Recorded in a separate commit, preserving the initial implementation commit.

## B2 — processed-reference viewer

Inspected all six required files: `scripts/play.py`, `scripts/media/render_rollout.py`, motion library, replay buffer and both audit visualizers. Existing policy playback steps an environment; the new viewer instead follows the rendering script's direct qpos → mj_forward pattern, with named joints and separate static assets. No shared runtime files changed.

Implemented `scripts/hoi/replay_reference.py` and isolated `reference_viewer.py`. The explicit `--display-only` flag is required. Both the processed NPZ and reviewed asset hashes are checked; asset source provenance is validated; training_ready=false is retained and shown. Seven compound box geoms and both fixed supports use the existing manifest builder; selected native hand geometry is retained. Large target spheres follow box-local sites; small EEF spheres are world overlays calculated from canonical recorded wrist poses and manifest offsets. The independent FK/marker comparison checks all 840 frames.

Interactive keys: Space pause/resume; Left/Right single-frame (pauses); +/- or brackets speed (0.125–8x); R/Backspace reset to zero; 1 overview, 2 robot follow, 3 contact close-up; C native collider view. Camera mouse controls remain available. Playback holds sampled poses without interpolation, uses a monotonic clock, excludes GUI startup delay, and pauses at 839 without wrapping. Fractional frame time survives render cadence and speed changes. Terminal future windows have explicit padding masks.

B2 development checks caught installed MuJoCo API details (light type, attachment naming and static-site caching). Corrected before acceptance; EEF reference overlays avoid static-site caches. No historical output was touched. **34 combined B1/B2 tests passed**, one upstream Torch deprecation warning; Ruff passed. Tests actively forbid mj_step/mj_step1/mj_step2, verify all-frame FK/target/EEF/support transforms, reject non-display loading or asset mismatch, guard against parsing original source NPZ in the viewer, and exercise keyboard clock semantics at all supported speed scales. Full render/GUI acceptance follows in B3.

## B3-discovered display limitation and B2 fallback correction

The native GUI smoke run cannot create a window on this session's DISPLAY=:1: GLX reports no framebuffer configuration. Software-GLX and explicit EGL-window probes also fail. All failure logs are retained, including MuJoCo's generated error log moved into this output directory. This is distinct from the successful EGL offscreen rendering path.

Added `--browser`, a loopback-only HTTP viewer rendering the same processed-reference scene through EGL. It needs no new dependencies and no external services. Buttons/keyboard/slider support pause, ±frame, 0.125–8x speed and reset; overview/follow/contact cameras and native collider view are available. Slow browsers skip only displayed samples according to elapsed playback time; source data and timestamps never change. Frame status reports the actually rendered frame, with desired contact and both distances. The blocked asset status remains visible.

Two fallback tests passed: actual HTTP requests rendered frames 0/260/400/550/839 and verified status, contact and JPEG dimensions; Node exercised the real JS button/keyboard/slider handlers, frame-clock endpoints, render loop and failure handling. Desktop-browser visual interaction is not claimed; native GUI remains environment-limited. The fallback renders successfully in this environment. The headless all-frame inspection and video remain valid and unchanged.
