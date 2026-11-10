# B1 reference conversion

PASS: shared robot fields verified before merge; six numeric robot arrays and all canonical metadata preserved exactly. HOI names have wider Unicode dtype but identical values/order; canonical widths retained.

SHA256: `bbbb40a49ccdd47f474bf43f75260a91361eac2ab3f3d9e54574d206996ad61b`. 840 frames, 50 Hz; final timestamp 16.78 s; sample coverage 16.80 s. Quaternion wxyz norms pass 1e-5 without normalization.

| Field | dtype | Shape |
|---|---|---|
| `joint_pos` | <f4 | [840, 29] |
| `joint_vel` | <f4 | [840, 29] |
| `body_pos_w` | <f4 | [840, 30, 3] |
| `body_quat_w` | <f4 | [840, 30, 4] |
| `body_lin_vel_w` | <f4 | [840, 30, 3] |
| `body_ang_vel_w` | <f4 | [840, 30, 3] |
| `joint_names` | <U26 | [29] |
| `body_names` | <U25 | [30] |
| `fps` | <i8 | [1] |
| `object_pos_w` | <f4 | [840, 3] |
| `object_quat_w` | <f4 | [840, 4] |
| `object_contact` | |b1 | [840] |

Object pose and desired contact retain exact source values/dtypes. Fixed targets, EEF offsets, supports and seven-geom assets are resolved from the hashed M6.5 manifest, so no static geometry is duplicated in this 12-field motion artifact. No phases or object velocities included. Desired contact is not measured contact.

See validation.json for shared-field comparisons and reference_manifest.json for per-array byte hashes and provenance. Automated test results are recorded alongside this bundle. Existing output directories are refused. Deterministic ZIP timestamps, member order and compression are fixed; regenerate with the same Python/NumPy/zlib environment for the same archive hash. Physical gates remain FAIL; no training approval.
