# Part B complete file inventory

All paths are relative to the repository. All are newly introduced by Part B; no pre-existing tracked file was changed. The user plan remains untracked and unchanged. Generated NPZ is explicitly tracked despite the global *.npz ignore rule. Historical M6/M6.5 files are not included in this change set.

## Implementation

- `scripts/hoi/process_reference.py`
- `scripts/hoi/replay_reference.py`
- `src/supermimic/hoi/reference.py`
- `src/supermimic/hoi/reference_browser.py`
- `src/supermimic/hoi/reference_viewer.py`
- `src/supermimic/hoi/viewer/index.html`
- `src/supermimic/hoi/viewer/player.mjs`

## Tests

- `tests/hoi_browser_controls.mjs`
- `tests/test_hoi_reference.py`
- `tests/test_hoi_reference_browser.py`
- `tests/test_hoi_reference_viewer.py`

## Artifacts and evidence

- `outputs/hoi/asset_audit/part_b/DATA_ACCEPTANCE.md`
- `outputs/hoi/asset_audit/part_b/FILES.md`
- `outputs/hoi/asset_audit/part_b/WORK_LOG.md`
- `outputs/hoi/asset_audit/part_b/b2_tests.txt`
- `outputs/hoi/asset_audit/part_b/browser_tests.txt`
- `outputs/hoi/asset_audit/part_b/final_tests.txt`
- `outputs/hoi/asset_audit/part_b/gui_egl_probe.txt`
- `outputs/hoi/asset_audit/part_b/gui_mujoco_log.txt`
- `outputs/hoi/asset_audit/part_b/gui_smoke.txt`
- `outputs/hoi/asset_audit/part_b/gui_software_retry.txt`
- `outputs/hoi/asset_audit/part_b/headless_run.txt`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0000_contact.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0000_overview.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0260_contact.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0260_overview.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0280_contact.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0280_overview.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0298_contact.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0298_overview.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0400_contact.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0400_overview.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0530_contact.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0530_overview.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0549_contact.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0549_overview.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0550_contact.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0550_overview.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0839_contact.png`
- `outputs/hoi/asset_audit/part_b/inspection/frame_0839_overview.png`
- `outputs/hoi/asset_audit/part_b/inspection/frames.csv`
- `outputs/hoi/asset_audit/part_b/inspection/inspection.json`
- `outputs/hoi/asset_audit/part_b/inspection/overview_montage.png`
- `outputs/hoi/asset_audit/part_b/inspection/reference_50hz.mp4`
- `outputs/hoi/asset_audit/part_b/lint.txt`
- `outputs/hoi/asset_audit/part_b/preservation_baseline.json`
- `outputs/hoi/asset_audit/part_b/preservation_final.json`
- `outputs/hoi/asset_audit/part_b/reference/VALIDATION.md`
- `outputs/hoi/asset_audit/part_b/reference/experiment_profile.json`
- `outputs/hoi/asset_audit/part_b/reference/reference.npz`
- `outputs/hoi/asset_audit/part_b/reference/reference_manifest.json`
- `outputs/hoi/asset_audit/part_b/reference/validation.json`
- `outputs/hoi/asset_audit/part_b/video_validation.json`

The final B3 commit also updates the Part B work log; earlier Part B commits are preserved. To verify the cumulative file set: `git diff --name-status 43a7761 HEAD`.
