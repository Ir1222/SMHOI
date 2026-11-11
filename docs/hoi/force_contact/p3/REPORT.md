# P3 — Joint robot–object world model and Force Head

**Acceptance: PARTIAL.** P3.1–P3.9 implementation, deterministic regression,
physical support-identifiability evidence, bounded supervised learning and explicit
checkpoint migration are delivered. Natural HOI prediction readiness is **not**
established. No P4 rewards, P5 imagination changes, P6 force-gradient experiments,
Actor force optimization or P7 integrated training were started.

The master plan remains unchanged by this work. This report records implementation
and evidence, not a new plan. The user's explicit instruction to commit each reviewed
group governs these commits; no push occurred. Review is required before further work.

## Baseline, reference and protection

Branch `main`; initial HEAD `a991ee776b8da4f5f278dbc48947785e7011b5e8`.
Accepted P1 `a562d936541ac7a73c6fa70fcf727449e426f28a`, P2 implementation
`5730be4d0c07862c7e91da75803ea4626384f7c4`, P2 report `a991ee7`.
Preparation read the complete master and mandatory P0/P1/P2 reports,
specifications, validation and framework audits, then inspected actual installed
code, original training/checkpoint paths and Replay tensors. The implementation
specification was committed before source changes.

Canonical motion, used explicitly by every P3 motion-dependent diagnostic:

- Path: `/home/han/pyx/SMHOI/outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz`
- Filename: `reference_refined.npz`; **840 frames, 50 Hz**.
- SHA256: `f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7`.
- Loader manifest: `reference_manifest.json`; refinement provenance and independent
  field hashes: `reference_refined_manifest.json`; historical schema evidence:
  `reference_manifest.before_schema_fix.json`. Exact hashes are in `baseline.json`.
- P1 named-state reset cache: `[840,84]`. Object pose/contact fields are unchanged
  from the older motion. Changed robot pose frames431–489 and velocity frames430–490
  do not intersect P2 run_03 transition endpoints. P2 identities remain compatible,
  but its original capture provenance is never relabeled as refined.
- No fallback, regeneration or modification of reference/evidence. Original P1/P2
  regression fixtures still test their legacy reference explicitly. Calibration
  `Placements` uses native model keyframes, not a hidden old reference load.

`baseline.json` and `integrity_final.json` check 452 protected files:449 unchanged
and exactly three intended source/test changes, plus ten installed
sensor/simulation dependency hashes. Versions: Python 3.12, Torch 2.14.1+cu130,
mjlab 1.6.0 (pinned commit 8bb2bc9e187797689ec82c78c35019afd2fd0376),
MuJoCo/MuJoCo Warp 3.16.0, Warp 1.18.0, NumPy 2.5.4. No dependency edits.
M6.5 asset SHA256 is
`1e8e1def25b262778195b56b0ea9622480ed2cede9cd3d3ae98c5679bb8288a1`.
Historical physical FAILs and `training_ready=false` remain unresolved.

Pre-existing user changes remain unstaged and intact: master-plan typo correction,
`.vscode/`, `SMHOI_TRACK_IMPLEMENTATION_PLAN_V3.md`, and the refined-reference directory
(including its ignored NPZ). The initial master working-copy hash is
`5b0c26e3c3d8c721e5e1f76ffe194f6141994a9eba4894ce7f43d7266e9be82f`.

## Implemented contracts and data flow

| Contract | Exact behavior |
|---|---|
| Robot physical/canonical | Original StateSpec 71, original ordering/scales/integration unchanged |
| Object physical 13 | Position 3, wxyz 4, root-origin world linear velocity 3, body angular velocity 3 |
| Joint physical 84 | Robot 71 followed by object 13; positions are environment-local |
| Object canonical 15 | Full robot-root-relative position in heading axes 3, heading-relative orientation 6, absolute object linear velocity in heading axes 3, body angular velocity 3 |
| Joint canonical 86 | Original robot canonical 71 followed by object 15 |
| Support 18 | Two manifest-order supports, each heading-relative root position 3 and orientation 6; recomputed per predicted state, separate from xi 33 |
| Dynamics input 166 | `[robot71, object15, support18, executed_action29, normalized_xi33]` |
| Shared output 88 | Robot residual 70, object residual 12, auxiliary force 6; only 82 integrated |
| Force 6 | `[left_xyz,right_xyz]`, hand-on-box N, mean over next 20 ms in each substep's contemporaneous Box frame |
| Training window | States `[9,N,84]`, actions `[8,N,29]`, xi `[N,33]`, force `[8,N,2,3]`, occupancy `[8,N,2]` |

All rotations use wxyz. Six-dimensional rotations are the row-flattened first two
matrix columns, matching original code. No moving-frame robot velocity subtraction.
Object residual order is `[dv_heading,domega_body,dp_heading,dtheta_body]`, each 3.
After velocity updates, `p_next=p+dt*v_next+R_heading*dp` and
`q_next=q*exp(dt*omega_body_next+dtheta_body)`, with right multiplication and
normalization. Zero residual is a constant **root-origin twist** prior. For COM offset
`r_body`, `v_COM=v_root+omega_world × R_object*r_body`; this is not a constant-COM prior.
Eight repeated integrations match MuJoCo `mj_integratePos` within 1e-9 in float64.
Force output cannot enter integration; width 88 residuals are rejected.

One original `DynamicsMLP` uses SiLU and default 2048×3 hidden layers. Its final
linear 88 is three sets of rows on one shared trunk. The bounded fit uses 64×3,
explicitly recorded as a diagnostic. Original termination head 4→64→64→1 remains,
with physical-failure labels and explicit reference frame `t+1`; timeout/clip end
are not failure labels. No intent/contact classifier or independent WM is added.

Data flow: unchanged P2 sensors/recorder → measured Replay → hashed archive with
all xi 33 fields → whole-episode partition → original P2 Replay append/gather →
HOI batch/reference adapter → **original** `WorldModel.window_loss` autoregression →
**original** `WorldModelTrainer.update` Adam/allreduce/gradient-clip/step. Intent and
clean action are not physical-model inputs. Tests alter them without changing loss.
The original `Trainer.run()` lifecycle, Actor, Critic and TD(lambda) are untouched.
See [framework reuse audit](TRAINING_FRAMEWORK_REUSE_AUDIT.md).

## P3.4 support decision

[SUPPORT_CONTEXT.md](SUPPORT_CONTEXT.md) records the experiment before the encoding
was frozen. There are 64 measured transitions in 32 corresponding pairs, including
16 no-hand, 8 left-hand and 8 right-hand pairs. A common 1 m translation preserves
canonical/action/xi inputs to 2.38419e-7 but changes object linear velocity by up to
.263008 m/s and measured hand-force components by 18.2043 N left/18.0121 N right.
The corresponding ambiguity is physical, not merely an algebraic demonstration.

With 24 training and 8 held-out whole pairs, a fixed ridge diagnostic changes object
normalized MSE from .00486310774 without context to .000124138570 with candidate18.
Force MAE **worsens**, from 1.25843/1.24432 N to 1.97066/1.93464 N. Thus support transforms
are required to identify object dynamics, but this experiment does not establish
force improvement. It is a linear probe, not an ablation of a fully trained joint WM.
The selected encoding is allowed by P3.4 and never extends episode xi 33.
No research-level semantics were amended.

## Data sufficiency and actual learned evidence

Evidence categories are kept separate:

| Evidence | Scope and use |
|---|---|
| Deterministic unit tests | Numerical contracts, gradients and guards; synthetic inputs are not physical validation |
| P2 run_03 | 96 controlled measured transitions; 10 left/8 right active, no onsets; serialized archive lacks xi 33; not used to fit the model |
| P3 support_02 | 64 measured transitions for paired identifiability only |
| P3 collection_01 | Failed post-forward assertion after 48 rows; retained and rejected by training loader |
| P3 collection_02 | 320 measured transitions, 32 fixture episodes, 4 worlds×8 resets×10 steps; all substep/final-forward guards passed |
| P3 fit_01 | Actual 200 inherited optimizer updates on 240 rows/24 episodes; 80 rows/8 entire episodes held out |

Collection02 SHA256:
`663a332f61f0073345e7a6e0da9d851698b479f7c65be36b319400a790b0c4e1`.
Support 02 SHA256:
`c34ec7f9b9635ca28d1f5c7745cf379062e4ee1bbd808ceedc6b37cc5cd8915d`.
The latter independently matches the archive committed in 7a2faa4; its early sidecar
predates archive-hash recording and was preserved unchanged.

The fixture set uses native no-contact/left/right placements and a refined-frame260
bilateral placement elevated 2 m, with small root/roll/action perturbations, native PD
actuation and action noise std.03. These are genuine simulator measurements but
**not natural pick–carry–place–release episodes**. There is one reference identity,
so no cross-clip generalization claim. Episode IDs 0–5 train, 6–7 validate for each
world. Canonical reference SHA/env/episode keys are disjoint. All eligible 72 training
and 24 validation windows are evaluated exactly once per horizon; overlapping-window
events are labeled as repeats, with unique row/event counts reported separately.

Force stays in N in Replay. Train-only active RMS component scales are
left 5.021431923 N/right 4.181304932 N, zero offset; normalization is reversible.
The uniform-window training distribution contains 576 target occurrences per hand:
inactive/active `[516,60]` left and `[534,42]` right. Active mass=.5 gives class weights
`[.558139563,4.80000019]` and `[.539325833,6.857142925]`. Active means measured
occupancy>0. Training active median magnitudes 4.974197865 N/.844837308 N define low/high.
Huber delta 1 is dimensionless; L1 is also configurable.

Initial unweighted window components robot 60.676597595/object 7.431865692/
force .169895172/termination.706213117 were measured before selecting diagnostic
weights as their reciprocals: `.01648081863`, `.13455571473`, `5.88598244157`,
`1/.706213117`. These are diagnostic coefficients, not frozen research weights.
Object nominal input/output/error scales remain documented in IMPLEMENTATION_SPEC.

Seed 304; CPU, one Torch thread; hidden 64×3, batch 32, window 8, Adam lr.001,
weight decay 0, grad clip 1, compile disabled. The sole update algorithm is inherited.
First/last sampled training loss 4.429178→.05230978. Parameter delta L2:
trunk 6.406816923, robot head 1.404190660, object head .525353372, force head .678402841.
Final head gradient L2=.335414499/.116658308/.246266440. These demonstrate learning,
not task readiness. Termination data contain no physical failures.

Held-out results (force MAE averages absolute xyz components; state RMSE is vector
norm RMSE, rotation uses the relative rotation vector):

| Metric | Prior, step 1 | Learned, step 1 | Prior, step 8 | Learned, step 8 |
|---|---:|---:|---:|---:|
| Robot normalized squared error | .1318641 | .01042528 | 2.0049002 | .05184336 |
| Object normalized squared error | .04829761 | .00301312 | 1.6872989 | .00632538 |
| Object position RMSE m | .003421947 | .000673716 | .140656888 | .005248508 |
| Object orientation RMSE rad | .00810154 | .00220054 | .09816890 | .01767405 |
| Object linear velocity RMSE m/s | .2591251 | .03981071 | 1.653717 | .05605857 |
| Object angular velocity RMSE rad/s | .5474858 | .17073980 | .65651345 | .14518382 |
| Left force MAE N | 1.53272879 | .17756991 | 0 | .13904218 |
| Right force MAE N | 1.03977907 | .32809272 | 0 | .14704025 |

| Step 1 force metric | Left | Right |
|---|---:|---:|
| Active MAE N | .31514105 | .44448563 |
| Inactive MAE N | .03999880 | .26989627 |
| Low-force MAE N | .15136530 | .22050963 |
| High-force MAE N | .47891676 | .57887119 |
| Magnitude MAE N, all | .31166092 | .62502766 |
| Direction cosine, defined true-force rows | .81215525 | .84533572 |
| Active recall at .001 N | 1 | 1 |
| Inactive false-positive rate at .001 N | **1** | **1** |
| Contact-loss recall | **0** | **0** |

Step 1 object NSE contact/noncontact=.004511076/.0000172086 (16/8 window starts).
Robot NSE contact/noncontact=.012913605/.005448645. Step 8 has **zero active true
hand forces**, so learned inactive force errors are worse than the zero prior;
active, direction and contact-state metrics are null, not successful zeros.
Unique validation left/right active counts 12/8, inactive 68/72, low 6/3, high 6/5,
onsets 0/0, losses 4/4. There is no onset-performance estimate. All per-hand/bin,
contact/noncontact and repeated-event details are in `fit_01/results.json`.
The calibration compares mean predicted force to measured substep occupancy;
these can differ when measured net forces cancel. That caveat does not excuse the
observed all-inactive false positives.

## Checkpoint migration

Named input slices preserve robot 71 and relocate executed action 71:100→104:133,
xi 100:133→133:166. Named robot residual rows remain 0:70. Original hidden/termination
weights are copied strictly; new object/support input columns and object/force output
rows are intentionally zero. Source configs/scales/keys must match. No first-N
blind copy, `strict=False`, Actor migration or optimizer transfer.

Read-only real source: `/home/han/pyx/SuperMimic-Reproduced/logs/hdmi_robot_only/ckpt_0012000.pt`,
SHA256 `16d2b0ca9138fe43336ad87c44d88c9c726e62e684c87919de81a4de1b618bc7`.
This is the external P0 candidate, not the missing target-local checkpoint. It lacks
embedded training Git/data hashes; the test establishes architecture compatibility,
not selection as an HOI policy. Default 2048×3 migration max robot prediction error
**1.9073486328125e-6**, within atol/rtol 2e-6. New force output is zero.
The fitted diagnostic checkpoint strict save/load prediction difference is **0**.
Its hash is `6246b001cb14a4966a90f6a99815b2fb45172b82f9cf57a867e6cdd0ce95d56b`.

New metadata includes physical/reference/asset contracts, normalization, support
schema, dimensions/config, provenance and NOT_READY gate. Loader rejects missing
keys, incompatible frames/scales, altered support tensors and normalization mismatch.
This is a WM-weights artifact, not an integrated optimizer/Replay/RNG resume bundle.

## Validation, failures and acceptance

[VALIDATION.md](VALIDATION.md) contains commands and test outcomes. Complete suite:
**102 passed, 17 skipped**, one Torch deprecation warning. Skips require unavailable
LAFAN1 NPZ/CSVs (three messages also mention CUDA); available CUDA physical P3
collection did run. Exact original/current robot WM loss, metrics and all gradients
match with zero tolerance. Robot StateSpec/default configuration/training loop hashes
are unchanged. Object numerical and loss-gradient checks are not P6 experiments.

Failures are retained: support 01 API-path error before simulation; initial support 02
side-biased holdout corrected by saved-data reanalysis; collection01 post-forward
assertion (exact bit not recorded); first full suite historical source-hash mismatch;
first final-verifier run assumed a support-sidecar hash that did not exist. No failed
capture is training evidence. Collection was narrowed from 16 to 10 steps, not made
physically stable by that change. The historical integrity test now pins the reviewed
WM source hash while keeping every other historical hash and all original evidence.

| P3 acceptance item | Result |
|---|---|
| Robot-only state semantics and WM regression | PASS |
| Object integration, zero-residual prior and frames | PASS |
| Support decision based on paired physical evidence | PASS, limited linear-probe force generalization |
| Shared trunk, correct 70/12/6 heads and force exclusion from state | PASS |
| Reversible force normalization, robust configurable balanced loss | PASS |
| Episode-aware validation and actual nontrivial left/right predictions | PASS for controlled diagnostics |
| Required natural-data/coverage/calibration evidence | PARTIAL; not established |
| Named checkpoint migration and strict compatibility | PASS with external-source provenance qualification |
| Overall P3 | **PARTIAL; not accepted as complete** |

`readiness_review.json` is the final gate (the earlier fit-sidecar gate is retained).
It is closed because data are controlled, both onsets are absent, active step 8
metrics are undefined and complete predictive thresholds/count requirements have not
been specified. No arbitrary thresholds are asserted. Force reward use is disabled
regardless of this predictive gate; P6 remains a separate future requirement.

Missing natural transition coverage is an **additional experimental requirement**,
not an implementation prerequisite now that complete controlled xi/force archives
exercise the original training path. Proposed next data procedure, requiring explicit
approval before lengthy work: first diagnose post-forward physical instability;
collect independent seeded episodes from explicit refined reset frames spanning
approach/onset, sustained left/right/bilateral contact, placement/loss/release and
support regimes; use original native actuator/action mapping and record the exact
controller/checkpoint; never actuate or teleport the object after reset. Save all
four solved substeps, xi, actions, states, terminal reasons and failures. Partition
whole source episodes before windows, reserve validation before fitting, inspect
unique strata counts and force scale distributions before selecting an approved
budget and readiness thresholds. One reference can support episode-held-out evidence,
but cannot establish cross-clip generalization. No such long collection was launched.

## Git review and deliverable inventory

Implementation commits, each reviewed/tested and staged by explicit path:

| Commit | Group and evidence |
|---|---|
| `29a48fd` | Specification and verified immutable baseline before source edits |
| `4b87669` | Joint state/integration; 4 targeted tests |
| `7a2faa4` | Support experiment/context; 4 tests and 64 measured transitions |
| `de5c5b9` | Shared joint dynamics/force loss/trainer adapter; 8 tests |
| `a375bf5` | Episode-aware supervised diagnostics; 11 tests, 320 valid measurements, 200 updates |
| `98645f2` | Named migration/strict checkpoint and full regression; 2 migration tests, 102 total passes |

Report preparation parent HEAD is `98645f2`. The final documentation/evidence commit
hash is reported to the user after committing, avoiding a self-referential hash.
[CHANGED_FILES.md](CHANGED_FILES.md) lists **every** P3 new/modified file;
`files.json` records per-file diff counts and hashes for the reviewed artifacts.
The complete runtime/test/script diff and all structured evidence were inspected;
NPZ/PT contents were validated by shape, provenance, raw-label recomputation and
strict loading. Verbatim pytest failure output contains two trailing-space lines;
whitespace checks exempt only that log, not source files. Final status must retain
only the four pre-existing user entries listed above. No reset, clean or push.

P3 stops here for explicit review. No next stage has started.
