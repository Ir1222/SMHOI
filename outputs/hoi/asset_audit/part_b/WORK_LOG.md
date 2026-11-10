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
