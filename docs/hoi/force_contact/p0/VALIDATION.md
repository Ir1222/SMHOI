# P0 validation record

Date: 2026-11-10. Only read-only checks and temporary dependency probes were run. P1 physical calibration, training, data conversion, reference visualization and historical audit experiments were not run.

## Results

| Check | Result | Evidence / limit |
|---|---|---|
| Branch/HEAD/status/recent history | PASS | `main`, `845878416d544ae2bef94f4a43d2ce059bc8c4ed`; original user untracked files preserved |
| Applicable AGENTS.md | ABSENT | Checked `/`, `/home`, `/home/han`, `/home/han/pyx`, target root and recursive target paths; none found |
| Complete master plan available | PARTIAL | Requested `force_contact` path absent; available `force contact` file ends at P0.4. Its hash is `b0fd536b3fefe03bfcdbe10b80506bc615eef4bb79c67f9c2abf21df87908745` |
| Whole available master + V3 read | PASS | 361 newline characters in master (last line has no newline); V3 665 lines. Missing remainder cannot be read or inferred |
| Exact Section 3 data/asset inventory | PASS for accessible listed inputs | 78 inventory records; 76 exist; missing plan path and target-local checkpoint reported explicitly; 8 NPZ archives inspected |
| Part B SHA256 and committed manifest | PASS | Exact stated archive hash, committed manifest bytes and 12-field schema |
| Canonical robot fields | PASS | All nine fields match shape, dtype and byte hash; no normalization or recomputation |
| M6.5 source and manifest integrity | PASS | Existing loader validated all declared source hashes; committed selected manifest and acceptance records unchanged |
| Historical physical gates | FAIL retained | Hand-contact and strict support validity remain FAIL; no claim of PhysX equivalence or learning readiness |
| Dependency pin/version comparison | PASS | pyproject/uv.lock/installed direct_url and package metadata agree |
| Current time/dimension contracts | PASS for robot baseline | .005×4=.020 s, 71 state/canonical, 70 residual, action29, xi33; target HOI dimensions not implemented |
| Native collider/body identification | PASS inspection | Enabled pitch-wrist and yaw-hand capsules on each side; full numeric records captured |
| Installed four-history bookkeeping | PASS synthetic probe | `[4,3,2,1]` newest-first; selected reset zeroed only its row; no `found_history` |
| Force sign / CPU-Warp / captured sensor parity | PASS isolated probe | Max CPU-Warp error `3.052e-8` N; pair swap negates; primary exerts force on secondary. Not full-scene P1 validation |
| Root-vs-COM and angular frame | PASS isolated numeric probe | Translational qvel matches root-origin world velocity; angular qvel body-local, world angular=R×local |
| Existing Replay termination alignment | DISCREPANCY VERIFIED | `[1,1]` labels produced where transition contract requires `[0,1]`; recorded for P2, untouched |
| Clip-valid imagination masks | GAP VERIFIED | Current clamp repeats final reference; no finite-clip reward mask |
| Support-context identifiability | GAP VERIFIED | Robot canonical loses translation/yaw needed to locate fixed supports; exact encoding and ablation deferred to P3 |
| Candidate checkpoint architecture | PASS strict robot-only load | All policy/critic/WM keys/shapes match; no missing/unexpected keys; candidate external file remains unchanged |
| Candidate training provenance | PARTIAL | Config-selected dataset currently byte-identical to canonical source; no embedded training Git/dataset SHA or policy-quality evaluation |
| Existing non-training baseline tests | PASS | 7 passed, one upstream Torch deprecation warning |
| Preservation | PASS | 361 protected files and 76 existing inventory records rehashed, no changes; selected committed manifests/reports byte-identical |
| Git scope / complete P0 diff | documentation only | No tracked/index diff; review patch includes every new P0 documentation file; pre-existing untracked files excluded from proposed changes |

Because the source of truth is incomplete, this record cannot certify that every requirement in the missing P0 remainder has been met. Overall P0 status is **PARTIAL**, not an inferred PASS. No extra skill or historical-record approval flow is imposed: the user's explicit stage-review rule and missing master are the reasons to stop.

## Executed regression command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_world_model.py::test_state_spec_identity_and_errors \
  tests/test_imagination.py::test_gae_targets_match_hand_computation \
  tests/test_g1_fk.py -q
```

[Complete output](baseline_tests.txt): `7 passed, 1 warning in 3.42s`. The WM-overfit and Trainer smoke tests were deliberately not selected because they train/save; no LAFAN1 dataset was generated or substituted. No tests were added to repository source.

The temporary `/tmp` probes performed artifact/reference validation, native model compilation, synthetic installed ContactSensor history, free-joint kinematics, strict checkpoint load using `weights_only=True` with only `pathlib.PosixPath` allowlisted, and a small Warp `sensor_acc` force-sign fixture. Their exact probe code is recorded in [AUDIT_PROBES.md](AUDIT_PROBES.md); numerical results are separate JSON files. They do not instantiate Trainer or optimize a model.

## Evidence files

- [repositories.json](repositories.json): actual local repository HEADs, branches, remotes, status and recent commits; initial target status separately identified.
- [artifact_inventory.json](artifact_inventory.json): every exact path/status/size/hash, full NPZ schemas/array hashes, manifest provenance, role and consumer.
- [dependencies.json](dependencies.json): installed/locked versions and inspected dependency source hashes.
- [audit_checks.json](audit_checks.json): accepted artifact, native collider, history, timing and velocity checks.
- [architecture_checks.json](architecture_checks.json): replay discrepancy, clamp behavior and support identifiability.
- [force_sign_probe.json](force_sign_probe.json), [force_probe.log](force_probe.log): isolated CPU/Warp sensor result.
- [checkpoint_audit.json](checkpoint_audit.json), [checkpoint_files.json](checkpoint_files.json): actual candidate config, model tensor maps and existing run files.
- [preservation_baseline.json](preservation_baseline.json), [preservation_final.json](preservation_final.json): before/after integrity results.
- [source_inventory.json](source_inventory.json): hashes/line counts of all 46 Python source files; AST parsing is an integrity/syntax check, not a claim that every unrelated module was reviewed in depth.
- [processes.txt](processes.txt): observation only. Existing external TensorBoard was left running; no training/evaluation process was observed.

Initial sandbox reads failed before execution (`bwrap: loopback: Failed RTM_NEWADDR`). Read/documentation commands subsequently ran with sandbox escalation approved by the tool. No approval rejection occurred. Warp probe compilation cache was confined to `/tmp/smhoi_p0_warp_cache` and bytecode generation was disabled.
