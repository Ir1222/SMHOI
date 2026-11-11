# P3 implementation specification

Scope: Master §8 P3.1–P3.9 only, under §§1.3, 1.3.4, 1.3.6, 2, 14, 15.
This is an implementation contract, not a replacement plan. The current user
explicitly accepts P1/P2, authorizes P3 and commits after each reviewed group.
No push, Actor/reward/imagination change, P4–P8 work or lengthy experiment is authorized.

## Verified baseline and immutable inputs

`main`, initial HEAD `a991ee776b8da4f5f278dbc48947785e7011b5e8`; accepted implementation
commits `a562d93` and `5730be4`. Master working-copy SHA256
`5b0c26e3c3d8c721e5e1f76ffe194f6141994a9eba4894ce7f43d7266e9be82f`.
Read the entire master, P0 coverage/report/closure/file map and all four audits,
P1/P2 specifications/reports/validation/reuse audits and supporting hash/numeric
records. Accepted evidence and ten installed dependency source hashes match.
`baseline.json` records protected files, versions, identities and exact reference.
No applicable AGENTS.md found. Preserve the user's master typo fix, `.vscode/`,
untracked V3 and all refined-reference files, including ignored NPZ.

Canonical motion is exclusively
`/home/han/pyx/SMHOI/outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz`,
840 frames, 50 Hz, SHA256
`f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7`.
`reference_manifest.json` is the compatible loader manifest;
`reference_refined_manifest.json` independently records output/array hashes and IK
provenance; `reference_manifest.before_schema_fix.json` is historical. All archive,
field hashes, names, shapes, finite values and quaternion norms passed the actual
ReferenceClip loader and independent refinement manifest checks. P1
ReferenceResetCache yields [840,84] in native named joint order. No file is rewritten.
The existing legacy loader defaults stay unchanged; all P3 tools pass the exact
refined path and require its hash explicitly, with no fallback.

Compared read-only with the old reference, actual changed robot pose frames are
431–489 and velocity frames 430–490; object pose/intent and names/FPS are identical.
All P2 run_03 frame/start+end metadata are outside those changed frames. Thus P2
identity/schema semantics are compatible, but the old capture hash remains old
provenance and is never relabeled. Refinement is kinematic, not physical feasibility.

Installed: Python3.12, Torch2.14.1+cu130, mjlab1.6.0 pinned to
8bb2bc9e187797689ec82c78c35019afd2fd0376, MuJoCo/MuJoCo Warp3.16.0,
Warp1.18.0, NumPy2.5.4. Reuse local implementations; no dependency change.

## Actual reusable architecture and data flow

Original StateSpec: physical/canonical71; root position3,wxyz4,joint29,world
root-origin linear3,body angular3,joint velocity29. Canonical order is height1,
heading-relative row-flattened first two rotation columns6, heading linear3,
body angular3,joint29,joint velocity29. Keep all scales and robot integration.
DynamicsMLP concatenates [canonical71,action29,normalized xi33], then
133→2048→2048→2048→70, SiLU, zero final affine. Robot residual order is
[dv_heading3,domega_body3,djoint_velocity29,dp_heading3,dtheta_body3,djoint29].
TerminationHead is existing 4→64→64→1 over reference height/tilt errors and magnitudes.

WorldModel.window_loss already performs undetached autoregression over eight
steps, sums normalized errors and BCE. WorldModelTrainer.update owns Adam,
zero_grad/backward/allreduce/clip/step and minibatch aggregation. Trainer.run owns
collection/UTD/scheduling/checkpoint/logging; it stays unchanged. P2 returns a dict,
not WindowBatch: [9,N,84] states, [8,N,29] executed actions, [N,33] xi,
[8,N,2,3] physical forces, transition-indexed masks, frame/clip/env/episode IDs.
HOI trainer adapter overrides only loss_on/online_error; it inherits the entire
original update/optimizer lifecycle. References for physical-failure termination
are gathered at explicit next frame t+1; timeout/clip-end are not failures.

## Requirement traceability and concrete decisions

| Requirement and unchanged acceptance | P0 finding / verified source | Decision and adjustment class | Files/interfaces | Validation |
|---|---|---|---|---|
| P3.1 robot preserved; physical84/canonical86 | Architecture “Verified dimensions”; P1 state.py named addresses; P0 coordinate JSON | Compose original StateSpec; named object slices13 and feature15. No duplicate robot spec. Implementation adaptation. | new hoi/state_spec.py | Exact robot prefix equality; wrong widths rejected; P1 state/address tests reused. |
| P3.2 explicit frames, invariance and round trip | P0 “Support-context identifiability”, root/world/body evidence | Object p relative to full robot root, inverse robot heading; orientation heading^-1*object; absolute root-origin object linear velocity rotated into heading (no robot velocity subtraction, matching robot convention); body angular unchanged. SI scaling explicit. | state_spec.py | Common XY/yaw invariance, nonidentity rotations and feature→physical reconstruction given robot state, quaternion sign invariance. |
| P3.3 12 residual/CV/root-COM | P0 coordinate_closure_checks; original right-product integration; P1 COM adapter | [dv_heading3,domega_body3,dp_heading3,dtheta_body3]. Update velocities then p+=dt*v_new+heading*dp; q=q*exp(dt*omega_new+dtheta). Constant root-origin twist prior, not constant COM velocity. No quaternion component regression. | state_spec.py | Compare MuJoCo mj_integratePos and offset-COM relation; zero/nonzero residual, eight repeated steps, finite derivatives. |
| P3.4 paired identifiability before encoding freeze | P0 candidate; xi33 episode constant; P1 fixed supports verified | First run bounded physical paired states under common XY translation with equal canonical86/action/xi and different support overlap. Use P2 measured one-step outcomes. Diagnostic ridge comparison with/without candidate support transforms, pair-held-out, same target/scaling. Candidate: two supports × (heading-relative pos3+rot6d6)=18; fixed geometry bound to asset hash, recompute each prediction. Do not freeze if measured evidence fails. Explicitly allowed option, no xi extension. | diagnostic script, later hoi/support_context.py only after evidence | Canonical equality ≤2e-6; outcome difference beyond numerical noise (position1e-5m or velocity1e-4m/s); report all object and force errors including null force differences. Compare held-out prediction errors, no inference of task-level validity. |
| P3.5 one joint shared trunk, 70+12+6 | Original DynamicsMLP supports feat_dim/out_dim; existing window loop | Reuse same DynamicsMLP with combined output88, split named heads; integrate physical82 only. Default hidden2048×3; reduced width permitted only labeled diagnostic runs. Single final affine is exactly three linear heads on same hidden vector. | hoi/world_model.py; minimal shared world_model.py interface | Shapes, coupling, zero prior, eight-step loss and finite gradients; original robot regression. |
| P3.6 measured left/right20ms force | P2 contact.py/Replay frame and sign metadata | Force6 predicts mean contemporaneous Box-local hand-on-box N, left then right. Inputs canonical joint/action_exec/xi/justified context only. Intent excluded from model signature. | world_model.py, hoi/wm_trainer.py | Compare saved P2 labels to raw sums; verify no clean-action or intent shortcut; force excluded from physical integration. |
| P3.7 reversible normalization/robust loss/imbalance | P2 run_03 10/96 left,8/96 right active, no onsets | Train-only per-hand isotropic active RMS component scale, zero offset, positive numerical floor; store scale/threshold/counts. Configurable Huber beta, per-class mass and component weights. Per-hand inverse empirical class frequency balances active/inactive. Active means measured occupancy>0 (can have canceled mean). Measure initial component losses before selecting diagnostic λ as inverse observed scale; no final research coefficients asserted. | hoi/force.py; model buffers/checkpoint | Encode/decode, units, absent class and nonfinite rejection, duplicated inactive samples do not change class balance after refit; per-hand low/high bins determined from training active median. |
| P3.8 episode/clip-aware one/eight-step metrics/readiness | P0 leakage risk; P2 full-window mask | Split by reference identity then episode; aliases of same reference are one clip. One source clip cannot demonstrate cross-clip generalization. All pair/episode variants kept together. Implement descriptive per-hand force vector/magnitude/direction, onset/loss, class calibration and robot/object errors by horizon/contact regime. Configurable gate requires finite held-out metrics, coverage, supplied thresholds, trained updates and provenance; default closed, never iteration-only. | hoi/validation.py; tests and diagnostic output | Mechanical tests distinct from physical diagnostics and learned performance. Missing required strata/thresholds/data keep NOT_READY and P3 PARTIAL. |
| P3.9 named WM migration | P0 checkpoint audit input action/xi columns shift; target checkpoint missing | Named feature/action/xi slice maps, original scales must match; preserve hidden and robot rows, zero new inputs/object/force outputs, copy termination strictly. Explicit schema metadata, reject unknown/missing/shape mismatch. No optimizer transfer or strict=False. | hoi/checkpoint.py | Nonzero original weights parity; shifted action/xi trap; wrong cfg/key rejection; checkpoint round trip. External real candidate read-only architecture test separately labeled qualified provenance. |
| §§1.3.6,2,14,15 framework and protection | P0 file map and both reuse audits | Only justified shared change: allow optional per-step auxiliary targets and named loss diagnostics in original window_loss, default path unchanged. No new autoregressive training loop; adapter inherits original trainer update. | shared world_model/world_model.py only; all other code HOI-local | Original tests, explicit old-vs-default loss/gradient parity, complete diff and protected hashes; audit. |

Object scaling explicitly inherits matching robot units: input position/rotation/linear/angular
(1,1,1,2), output dv/domega/dp/dtheta (.3,.5,.01,.02), error pos/rot/linear/angular
(.05,.1,.5,1). These are transparent nominal scales consistent with the original
robot, not trained accuracy thresholds. Joint residual82 and output88 are distinct.
Support is a separate time-varying canonical input block, never episode xi.

Minimal shared interface: `WorldModel.window_loss(..., auxiliary_targets=None)`
slices optional per-transition tensors for the existing `_loss_step`; an overridable
loss-metric reducer names HOI components. Default calls retain original arithmetic,
state70 and termination outputs. HOI `_loss_step` returns weighted physical and
force component losses to that same loop. Compilation remains optional via the
original enable_compile. No production optimizer loop is added.

## Data sufficiency, bounded diagnostics and blockers

P2 run_03 archive has 96 controlled transitions, nine env/episode groups, two aliases
of ONE reference, left/right active counts10/8, zero onset events. Its serialized
NPZ omits xi33, although live HOIReplayBuffer stores xi correctly. Do not invent xi,
train on reference-derived forces, relabel old provenance, or use repeat runs as
independent motion data. Available 8 active-terminal rows do not supply eight-step
trajectory windows. No natural HOI dataset has been located in contact_supervision.

Implementation prerequisite: collect a small fresh physical diagnostic using
unchanged P1 scene + P2 recorder, explicitly refined reference, saving all live
fields INCLUDING measured xi, reference/asset/schema and complete identities.
P3.4 starts with ≤64 one-step paired transitions. Subsequent fitting diagnostic is
bounded to ≤512 controlled transitions and ≤200 updates of a small shared trunk,
using the original WorldModelTrainer. These are short implementation experiments,
not natural training or approved long runs. Preserve failures in new run dirs.
No claim of a task-trained model follows from successful forward/optimizer calls.

Additional experimental requirement (not necessary to implement interfaces): natural
rollouts spanning approach, onset, sustained manipulation, placement/loss/release,
with repeated independent episode seeds and enough active examples per hand; retain
failed episodes and raw four-substep labels. Use explicit refined reset frames,
original robot policy or reference PD targets through original action mapping,
short validated intervals, no object actuation/teleport after reset; record xi,
commands/actions/states/masks and exact controller/checkpoint provenance. Split
whole episodes before windows, group common reference identities, keep a separate
paired support diagnostic, inspect distributions before fixing sample budget/loss
weights. Longer collection/fitting needs explicit user approval. Historical M6.5
FAILs and refinement's missing physical validation remain unresolved.

Natural data insufficiency, absent onset coverage, single motion identity and
unvalidated task-level prediction thresholds block full P3 acceptance. The stage
must be PARTIAL unless real evidence closes them; no invented readiness threshold.
Support encoding remains a candidate until the prescribed experiment. If it fails,
record pending decision rather than implicitly selecting no context. Any change
to approved state/force/model/research semantics requires amendment and approval.

## Validation and review procedure

Numerical tolerances: float64 geometry/integration1e-9; float32 invariance/roundtrip
2e-6 absolute (relative2e-6); force roundtrip2e-5N+2e-6relative; migration robot
prediction/loss parity2e-6 (larger observed numerical effects documented, not hidden).
Test gradients for integration/loss, not P6 force-gradient fidelity. Run applicable
original robot tests and P1/P2 regression; missing LAFAN1 remains explicit skip.
No new physical skip can count as PASS. Exact commands, raw results, counts, metrics,
changed paths, review commits and preservation checks go in REPORT/VALIDATION and
outputs/hoi/contact_supervision/force_model_validation/. Every group is fully reviewed,
tested, staged by explicit path and committed under the user's authorization.

## P3.4 resolved decision

See SUPPORT_CONTEXT.md and support_balanced_analysis.json. Physical paired outcomes
are nonidentifiable without support placement; candidate18 reduces held-out object
normalized MSE .0048631→.00012414. Select the explicitly allowed two relative
support transforms18, separate from xi33. Force ridge errors worsen (left1.258→1.971N,
right1.244→1.935N); force generalization remains unresolved. Balanced pair split
corrects the original side bias; both results are retained. No research amendment.
