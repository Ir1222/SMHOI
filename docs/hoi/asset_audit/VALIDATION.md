# Validation and reproduction

- Offline asset tests: **12 passed**, one upstream Torch deprecation warning. [Raw output](asset_tests.txt).
- Existing native CPU FK regression: **4 passed**, CUDA case deliberately deselected, one upstream warning. [Raw output](native_fk_tests.txt).
- `compileall` on new audit scripts and `src/supermimic/hoi`: passed.
- `git diff --check`: passed.
- All 53 M0 input hashes unchanged; source git status unchanged. [Integrity evidence](integrity_final.json).
- No shared training/env/WM/imagination files, pyproject.toml or uv.lock changed from baseline.
- Test success verifies code and declared measurements; physical hand-contact and support gates remain **FAIL**. No dynamics rollout, force validation, training, state packing, WM tests or actor migration was performed. These are outside the user-authorized audit phase.

Run from `/home/han/pyx/SMHOI` with its existing virtualenv:

```bash
.venv/bin/python -m pytest tests/test_hoi_asset_audit.py -q
.venv/bin/python -m pytest tests/test_g1_fk.py -q -k 'not cuda'
.venv/bin/python scripts/hoi/run_audit.py
```

The aggregate command runs the already-implemented stages sequentially, verifies M0 hashes before/after, updates only this task's generated audit outputs and manifest, and runs the asset tests. It does not recapture the original M0 baseline. Individual commands actually run during the audit were:

```bash
.venv/bin/python scripts/hoi/audit_npz.py
.venv/bin/python scripts/hoi/asset_common.py
/home/han/XJINproject/magicZ1/Z1/soma_retargeter_autoconfig/.venv/bin/python scripts/hoi/read_usd.py
.venv/bin/python scripts/hoi/audit_robot.py
.venv/bin/python scripts/hoi/audit_contact.py
.venv/bin/python scripts/hoi/audit_internal.py
.venv/bin/python scripts/hoi/audit_support.py
.venv/bin/python scripts/hoi/visualize_audit.py
.venv/bin/python scripts/hoi/visualize_support.py
.venv/bin/python scripts/hoi/build_manifest.py
```

USD was parsed using existing `pxr` 0.26.3 in the named environment; no packages installed. The aggregate command accepts `--usd-python` for another existing compatible environment. Paths identify this local checkout and original assets; reproduction with strict source verification requires those original files. The static manifest contains all selected construction values independently of the time-series NPZ.

Schema/physical checks cover source SHA256, six-array byte preservation, all nine field value equality, seven compound geoms, XML/USD bounds, explicit mass/COM/principal inertia, only documented priority differences, target signed-distance sign under a nontrivial rotation, EEF axis convention, hand-only candidate preserving foot geoms and all robot inertias, fixed supports/free box, reset/world-origin consistency, known overlap/separation analytic cases, and rejection of invalid dimensions/quaternions or approval with failed gates.

The regular `sm-train` runtime was not launched: no shared path changed and the authorized phase excludes training. Original CPU FK tests supply a relevant local regression check, not a claim of end-to-end policy evaluation.
