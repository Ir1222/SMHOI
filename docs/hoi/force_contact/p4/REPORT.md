# P4 report — 2026-11-12

**Overall acceptance: PARTIAL. Reward software/tests and geometric conventions:
PASS.** P4 implementation and bounded checks are complete; physical target validity
and the missing existing object-tracking reward prevent unconditional acceptance.
No P5/P6 work, HOI physical rollout, Actor/WM training or optimizer update occurred.

## Implementation and scope

Baseline: `main` / `f191f6d` (approved plan consistency audit). Prior P3 architecture
commit: `bab5132`, historically PARTIAL. The user explicitly requested P4 and a
commit after changes. [IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md) was written
before source edits and reconciled with final evidence. The Master Plan is unchanged.

`src/supermimic/hoi/rewards.py` provides exact global reference-frame lookup,
existing Torch FK plus current Box-root target transforms, the specified
`c_ref * mean(exp(-distance / sigma_pos))`, and additive composition of supplied
robot/object/regularization terms. It imports no sensor/Replay/physical scene path.
There is no force factor, contact classifier, OFF repulsion, temporal lookahead,
new state or observation dimension. Configuration is explicit; no final targets,
scale or weights are silently activated in training.

Physical84/canonical86/residual82, WM166/82, support18/action29/xi33 and original
Actor/Critic contracts remain unchanged. Positions are meters in shared
environment-local world axes; quaternions are wxyz. Geometry uses robot qpos36 and
current Box pose71:78. Gradients reach both hand joint coordinates and Box
translation/orientation. Lookup uses the evaluated frame (t+1 for next-state reward)
and rejects invalid/terminal padding indices. The raw proximity has no implicit dt;
the original robot reward is dt-scaled, documented for later coefficient selection.

## Audit and numerical results

See [CONTACT_TARGET_AUDIT.md](CONTACT_TARGET_AUDIT.md) and
[VALIDATION.md](VALIDATION.md) for conventions, exact commands and test scope.
Evidence root: `outputs/hoi/contact_supervision/reward_validation/`.

| Check | Result |
|---|---|
| All 840 original-reference frames | 290 ON (260–549), 550 OFF; unchanged bool label |
| Torch EEF versus MuJoCo FK | max 1.7766e-15 m |
| Native wrist FK versus recorded reference | max 8.1811e-8 m; rotation max 8.2935e-8 rad |
| Distances versus historical contact CSV | max difference 6.5724e-8 m |
| Target surface correspondence | both 8.6736e-18 m from exposed side3/side4 walls |
| EEF versus native yaw capsule | marker is 15 mm inside each capsule |
| Final selected tests | **51 passed, 0 failed, 0 skipped** |
| New P4 tests within that suite | **15 passed** |
| Lint/format | PASS, all five new Python files |

Tests cover gate boundaries, ON positivity/monotonicity, exact OFF-zero reward and
gradients, named hand ordering, rotated/current Box transforms, batch independence,
input preservation, joint/Box finite differences and gradcheck, zero-distance finite
subgradients, sensor independence, original robot tracking equations, additive
object-term preservation, and existing FK/state/WM/contact aggregation/Replay/robot
environment regressions. No physical contact success is inferred from these tests.

| Candidate sigma (m), pooled ON quantile | Mean raw ON proximity | Mean ON joint-gradient norm |
|---|---:|---:|
| 0.0252139293, p25 | 0.24483194 | 3.45414102 |
| 0.0423003221, p50 | 0.40023489 | 3.22905152 |
| 0.0628937157, p75 | 0.52512410 | 2.79129543 |

All 550 OFF rewards and state gradients are exactly zero for every candidate.
Original positive robot tracking terms average 0.0999999033 per ON reference state.
`scales_01/scales.json` records Box-pose gradient norms, controlled ±5 mm and ±.02 rad
geometry responses, and nine illustrative coefficients for 10%/25%/50% of that
measured robot scale. These are reference-state algebraic measurements, not policy
experiments. No final sigma or reward coefficients were selected.

## Acceptance limitations and decisions for review

1. **Physical target validity remains unresolved.** Both original slots are
   justified semantic proximity candidates with verified ordering, transforms and
   surface location. However, EEF markers are internal to native capsules; placing
   an internal point on the wall cannot establish nonpenetrating hand contact.
   Historical 37.22/44.99 mm native overlap and support failures remain intact.
   No target set/offset/asset was changed. Any physical target revision requires a
   separate evidence-backed approval; current software does not approve training.
2. **P4.5 assumes an existing object-tracking reward that is absent in this checkout.**
   Repository Python-source inspection found only the original robot tracking
   implementation. The additive composition is implemented and tested with supplied
   terms, but an actual object reward equation, its scale and final combined
   objective remain pending a research-level decision. No arbitrary replacement or
   object-weight value was invented. Original robot/object dynamics are preserved.
3. **Bilateral requirement is explicit task evidence, not an achievement.** M6.5's
   `contact.physical_validation.bilateral_force_required=true` establishes the task
   requirement for ON frames 260–549. It does not create per-hand reference labels.
   Independent hand and any-hand/bilateral physical diagnostics remain required;
   P4 does not implement their aggregation or use them in the reward.
4. **Historical prerequisites remain PARTIAL.** Missing historical V3 provenance
   still blocks strict physical-scene loading; P3 FP32 multi-step migration remains
   a recorded failure. `training_ready=false` / `M6_5_NO_GO_M7` are unchanged. The
   original accepted Part B archive with valid sidecars is used only in this new
   audit; no existing refined-reference pipeline was redirected.

Thus P4 binary gating, differentiable equation, geometry conventions, software
gradients, dimension preservation and robot-only regressions PASS. Full physical
target acceptance and preservation of a real object-tracking reward cannot be
marked PASS. No criterion was weakened to hide these gaps.

## Training Framework Reuse Audit

Reused `G1Kinematics`, quaternion primitives, `ReferenceClip`, `ReferenceResetCache`,
`AnalyticHeads`/`tracking_reward_terms`, native model construction and existing
asset-audit point/primitive helpers. No shared source files changed. No Trainer,
ReplayBuffer, WM learner, Actor BPTT or Critic TD(lambda) implementation was copied.
P1/P2 sensors/aggregation/recorders remain byte-identical; their data remains
independent physical evidence. The original robot-only environment regression
passes, including observation930/goal680/action29/state71/xi33 and Replay contracts.

## Files, preservation and diff review

All changes are additions:

- `src/supermimic/hoi/rewards.py`.
- `scripts/hoi/audit_proximity_reward.py`, `scripts/hoi/measure_proximity_scale.py`.
- `tests/hoi/test_geometric_contact_reward.py`, `tests/hoi/test_contact_target_audit.py`.
- `docs/hoi/force_contact/p4/IMPLEMENTATION_SPEC.md`, `CONTACT_TARGET_AUDIT.md`,
  `VALIDATION.md`, `REPORT.md`.
- Evidence under `outputs/hoi/contact_supervision/reward_validation/`:
  `baseline.json`, `geometry_01.log`, `geometry_01/geometry.json`,
  `geometry_01/distances.csv`, `scales_01.log`, `scales_01/scales.json`,
  `tests_01.log`, `tests_02.log`, `preservation.json`, `static_checks.log`.

All **548** pre-existing files in the preservation inventory and **10** audited
dependency-source hashes are unchanged. This includes accepted original/refined
NPZs, assets, historical evidence/checkpoints, the Master Plan and user
`.vscode/settings.json`. No prior checkpoint was loaded for migration or written.

The complete added source/test/documentation diff and machine-readable evidence
were reviewed. Only these P4 paths are staged for the explicitly requested commit;
the unrelated untracked `.vscode/` remains untouched. Commit hash/final status are
reported in the accompanying response. No push and no next-stage authorization.

**Stop for explicit review of this PARTIAL P4 result.**
