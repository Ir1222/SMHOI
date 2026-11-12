# P4 implementation specification

Scope: P4 only, authorized by the user's request following plan audit `f191f6d`.
The user also authorizes a commit after changes. No P5/P6 implementation, training,
HOI physical rollout, asset repair, checkpoint migration or dependency update.

Baseline: `main` at `f191f6d`; Master Plan SHA256
`268047aa4e431866bf708c6da2b271ef48d44dfec561458d09e11f76df563a41`.
Previous implementation: P3 architecture cleanup `bab5132`, reported PARTIAL
(software contracts PASS; FP32 multi-step migration and missing scene provenance
remain unresolved). Current P4 authorization does not turn these findings into PASS.
Only pre-existing user change: untracked `.vscode/`. The initial 548-file hash
inventory is `outputs/hoi/contact_supervision/reward_validation/baseline.json`.
Python environment: repository `.venv`, Torch 2.14.1+cu130, MuJoCo 3.16.0,
mjlab 1.6.0, NumPy 2.5.4. No installed source changes.

## Traceability and decisions before source implementation

| Master requirement | P0 / prior evidence and actual code | Concrete decision and files | Verification |
|---|---|---|---|
| P4.1 targets, EEF, FK, collision correspondence | P0 DATASET_ASSET_AUDIT accepted profile; REFERENCE_IMPLEMENTATION_AUDIT HDMI transforms; `asset_manifest.py`, `g1_fk.py`, `contact_points.csv` | Read exact M6.5 manifest and original Part B with `ReferenceClip`; offline audit script and CONTACT_TARGET_AUDIT.md; no target edits | All 840 geometric frames; named native joint/body order; independent MuJoCo FK; seven box surfaces and native capsules; compare existing CSV |
| P4.1 bilateral requirement | Manifest distinguishes global intent from contact; historical physical-validation bilateral requirement is not per-hand annotation | Report task evidence and applicable scope separately from the two-target reward set; do not implement physical aggregation in P4 | Audit source YAML, manifest, CSV and original labels; retain any-hand/bilateral distinction for P6/P8 |
| P4.2 exact reference gate | P0 accepted original hash `bbbb40…61b`, ON 260–549; P2 t/t+1 labels; `ReferenceClip` rejects invalid indices | Small Torch bool-label lookup with explicit evaluated frame indices; no clamping, shifting, interpolation or derived labels | Boundary frames 259/260/549/550, shuffled batches, invalid and terminal indices |
| P4.3 geometry and state | P0 ARCHITECTURE_AUDIT robot qpos36 prefix; P1 `state.py`; current `G1Kinematics` | New `hoi/rewards.py` reuses FK and quaternion transforms, slices qpos36 and Box root pose71:78 from state84 | Transform oracle, rotated Box, common-frame invariance, current-state perturbations, batch independence |
| P4.4 proximity | HDMI `eef_contact_exp` position factor inspected locally; source force factor/outside-intent baseline are excluded by current plan | Exact bool gate × mean exp(-Euclidean distance / explicit positive sigma); no default target set/scale/weights | Positive/monotonic ON, exact OFF zero and zero gradients, finite zero-distance subgradient, finite differences/gradcheck |
| P4.5 preserve tracking | Original `tracking_reward_terms` and `AnalyticHeads.reward` exist; search of target Python source finds no object-tracking reward | Preserve original modules; additive combiner accepts already computed robot/object/regularization terms and explicit coefficients. No new object reward equation or integration | Robot formula regression and additive gradient/term preservation tests; missing object reward remains an acceptance limitation |
| P4.6 independent physical diagnostics | P0 DEPENDENCY_CONTACT_AUDIT; P1 calibration and P2 action-aligned measurements | No imports of scene/contact/recorder/Replay or sensor inputs in reward module; only offline geometric diagnostics use MuJoCo | Sensor/physics functions patched to fail while reward and gradients run; P1/P2 software regressions; preserved hashes |
| P4.7 and deliverables | Reuse existing test conventions, original robot FK/reward/state/WM tests | New `test_geometric_contact_reward.py`, `test_contact_target_audit.py`, audit CSV/JSON, REPORT.md | Required unit tests, numerical audit, static checks, complete diff and preservation verification |

All four P0 audit categories apply: architecture fixes dimensions/reuse, dependency
audit separates real sensing from reward, dataset/assets fixes immutable inputs,
reference-code audit fixes transforms and excludes source force/baseline shaping.
Historical force-head/lookahead recommendations are superseded by the current plan.

## Contracts

- State `[...,84]`: robot71 followed by object13. Robot qpos is `:36`; object
  position `71:74`, wxyz quaternion `74:78`. Both use the same environment-local
  world axes. Canonical86/residual82/WM166/support18/action29/xi33 are untouched.
- FK body indices and constant offsets/targets are explicit, identically ordered,
  nonempty. Current proposed set is left/right wrist yaw, offsets `(0.05,0,0)` m,
  Box-root targets `(0,±0.17,0.14)` m; no physical-contact equality claim.
- Geometry outputs `[...,Nc,3]`, distances and per-target contributions `[...,Nc]`,
  raw proximity `[...]` in [0,1]. Sigma has units m; no implicit dt weighting.
  Caller coefficients must express the same units as precomputed tracking terms
  (original robot reward includes dt=0.02). No final coefficient is selected here.
- Reference labels are bool `[frames]`; lookup indices are long with the reward's
  batch shape. State/frame alignment is caller-explicit; next-state rewards use
  t+1. Invalid frames raise instead of padding; no rollout/mask integration in P4.
- Finite floating inputs and unit quaternions are required. Rotation normalization
  uses existing Torch conventions without writing to input state/reference.
- Euclidean norm is not classically differentiable exactly at zero; Torch's finite
  zero subgradient is tested. Do not change the equation to a squared/smoothed norm.

## Bounded validation criteria declared before experiments

- FP64 Torch FK versus MuJoCo: max position error <= 1e-10 m; recorded float32
  reference versus recomputed FK <= 2e-6 m. Rotation comparison is sign invariant.
- Target surface distance <= 1e-7 m (existing manifest criterion). Geometric
  capsule overlap is reported, never converted to measured contact or force.
- Analytic transforms/reward match independent numeric formulas at FP64 1e-12;
  gradcheck uses eps=1e-6, atol=1e-5, rtol=1e-3 away from zero; central differences
  must agree at those tolerances. No action-to-WM/Actor BPTT test in P4.
- Candidate sigma values: ON-distance pooled p25/p50/p75, not a copied HDMI
  default. Measure raw proximity, gradient norms and illustrative coefficients
  matching 10%, 25%, 50% of observed positive robot tracking scale; these are
  bounded algebraic probes, not adopted training hyperparameters or sweeps.
- Original robot reward closed-form regression, existing FK and selected state,
  architecture, contact aggregation and Replay regressions. Any unavailable or
  failed prerequisite is reported, never silently skipped or repaired here.

## Adjustments and unresolved risks

A1 (implementation-level): use original accepted Part B `reference/reference.npz`
and its complete manifest under current §3.6, not P3's historically refined NPZ
whose sidecars are absent. Both files remain byte-identical; no reference switch in
any existing pipeline. The new standalone audit names/hashes its actual input.

A2 (implementation-level): offline inspection may structurally read the exact
hash-pinned manifest and report missing source files without invoking a physical
environment. This is not relaxation of `load_manifest` or the strict scene loader;
no model/assets/physical gates change. All available manifest sources are checked.

R1: M6.5 physical contact/support FAIL, internal EEF marker and unreachable or
penetrating exact-target poses may prevent physical acceptance. Audit, retain
semantic candidates explicitly, and request approval before any target-set change.
No IK retargeting or asset modification is authorized.

R2: no current object-tracking reward implementation was found. The P4 combination
formula can be implemented/tested with supplied terms, but actual object-tracking
preservation/integration and final weight selection cannot be claimed. A new object
reward equation is a research-level decision, pending review; independent proximity
work proceeds. No nonexistent object implementation is invented.

R3: P3 partial migration/scene provenance findings remain unchanged; no training
readiness claim. Full physical binary-contact agreement and post-release/penetration
evaluation remain P6/P8; P4 uses geometric inspection only, with no physics stepping.

## Final reconciliation

Implemented the planned standalone reward, additive combiner, audit and scale
scripts, two test modules and reports. No shared training-core files changed.
All 15 new P4 cases pass; final selected suite is 51 passed / 0 failed / 0 skipped.
The original robot-only smoke test performs 16 control steps in four worlds;
this is preserved-behavior regression, not HOI contact validation or training.

Geometry thresholds passed: Torch/MuJoCo EEF max 1.7766e-15 m, recorded wrist max
8.1811e-8 m, historical CSV distance difference 6.5724e-8 m, target surface
distance 8.6736e-18 m. Tests cover FP32/FP64 point rewards and FP64 state gradients;
existing FK regression includes CPU/GPU. New reward CUDA/compiled rollout parity
is not claimed; integration/compiled parity remains P5.

P4.1 establishes the bilateral task requirement from the manifest's explicit
physical-validation flag, limited to global ON frames 260–549. It does not establish
actual bilateral achievement or per-hand source labels. Both targets remain
explicit semantic candidates; each EEF is 15 mm inside its native yaw capsule.
Physical attainability/target acceptance remains R1, not a passed contact test.

Three sigma candidates and nine illustrative coefficient calculations completed;
OFF rewards/gradients are exactly zero. No final sigma/weight was selected and no
object reward was invented. R2 remains unresolved. Final acceptance is PARTIAL;
software and geometric convention checks PASS. No research-level amendment was
implemented and no later stage was started.
