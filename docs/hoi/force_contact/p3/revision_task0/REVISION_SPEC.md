# Task 0: minimal joint robot–object–contact P3 revision

Status: **specification for review; implementation and experiments not started**.
Baseline: `main`, `9bf253c74c92ebce9857f7863c7298f99a3f9567`, checked against upstream
`Ir1222/SMHOI` main on 2026-11-11. The user's Task 0 explicitly replaces continuous
force prediction in P3 with prediction of **desired reference contact intent**.
This is a scoped research amendment to the force-specific P3 requirements in the
[master plan](../../SMHOI_FORCE_AWARE_IMPLEMENTATION_PLAN.md), including its P3
acceptance/reuse requirements. The master plan and previous reports remain intact.
This document does not redefine P4–P7 or authorize their execution.

## 1. Predictive contract

| Quantity | Existing P3 | Revised P3 |
|---|---:|---:|
| Robot / box / joint physical state | 71 / 13 / 84 | unchanged |
| Predictive state passed through HOI WM steps | 84 | 85 = physical84 + contact1 |
| Joint physical canonical features | 86 | unchanged |
| Current global contact feature | absent | 1 |
| Support / executed action / normalized xi | 18 / 29 / 33 | unchanged |
| Features passed to `DynamicsMLP` before action/xi | 104 | 105 |
| Complete network input | 166 | **167 = 86 + 1 + 18 + 29 + 33** |
| Dynamics output | 88 = 70 + 12 + 6 force | **83 = 70 + 12 + 1 contact logit** |
| Residual passed to physical integration | 82 | unchanged |

Use one existing `DynamicsMLP` with SiLU hidden layers and three named output row
groups: robot `[0:70]`, object `[70:82]`, next-contact logit `[82:83]`. Default hidden
widths remain `(2048,2048,2048)`; the existing diagnostic checkpoint uses
`(64,64,64)`. There is no separate object model. The existing termination head is
separate from the 83 dynamics outputs and remains unchanged.

For predictive state `z_t = [s_t, c_t]`, define:

```text
x_t = [JointStateSpec.canonical(s_t), c_t, SupportContext(s_t)]  # 105
y_t = DynamicsMLP(x_t, a_exec_t, normalize_xi(xi))              # 83
s_next = JointStateSpec.integrate(s_t, y_t[..., :82])          # 84
logit_next = y_t[..., 82:83]
p_next = sigmoid(logit_next)                                 # 1
z_next = [s_next, p_next]                                     # 85
```

Logged contact states/targets are binary; predicted contact states are probabilities
in `[0,1]`. The HOI `predict` interface returns `(z_next, logit_next)` and `step`
returns `z_next`. Only the initial logged contact bit seeds an autoregressive
window. Every subsequent step uses the predicted probability, including during
training; no rounding, thresholding, detachment, or future-label teacher forcing.
The metric threshold is `p >= 0.5` only. Physical integration, canonicalization,
support transforms, and physical error functions receive **only `[..., :84]`**.
Do not enlarge the simulator state, `JointStateSpec`, or P2 Replay storage.

## 2. Preserved geometry and label provenance

Robot StateSpec ordering, scales, xi33 normalization, and integration are unchanged.
Object physical fields remain position3, quaternion4 (`wxyz`), root linear velocity3
in world axes, and angular velocity3 in object body axes. Positions are environment
local. Object canonical15 remains robot-root-relative position in robot heading
axes, heading-relative orientation6, absolute object linear velocity in heading
axes, and object body angular velocity. Do not subtract robot velocity or substitute
COM velocity. Rotation6 uses the existing row-flattened first two matrix columns.

Retain the existing scaled 12D object residual and `dt=0.02 s` integration:
update velocities first; translate by `dt * v_next` plus heading-frame position
residual; right-multiply the object quaternion by the rotation increment from
`dt * omega_next` plus body-frame rotation residual. Retain both manifest-ordered
support transforms (18D), recomputed from each predicted physical state. Existing
[support identifiability evidence](../SUPPORT_CONTEXT.md) justifies this context
through object prediction; no new force experiment or xi extension is needed.

Authoritative motion asset:
`/home/han/pyx/SMHOI/outputs/hoi/asset_audit/part_b/reference_refined/reference_refined.npz`.
SHA256: `f6ba446c16e71ab3cffb09ec015ca9a224305f6ac82d01f3631d4d5990a5b5d7`.
It has 840 frames at 50 Hz and one boolean `object_contact[840]`: OFF at 0–259,
ON at 260–549, OFF at 550–839 (zero-based indices).

Reuse P2 `intent_ref` (last dimension 2), containing
`[object_contact[t], object_contact[t+1]]`, exactly.
This is desired reference intent, not measured hand contact, force, or termination.
There are no left/right intent labels. Native G1 hand–box sensing and raw measurements
remain diagnostic data; they do not supply classifier targets, inputs, or weights.
No fallback to the older reference asset is allowed.

## 3. Replay adapter, loss, and framework reuse

For `K=8`, existing Replay supplies states `[9,N,84]`, intent pairs `[8,N,2]`,
executed actions `[8,N,29]`, and episode xi `[N,33]`. In the HOI adapter:

```text
c_logged = cat((intent_ref[0, :, 0:1][None], intent_ref[:, :, 1:2]), dim=0)
z_logged = cat((states, c_logged), dim=-1)  # [9,N,85]
auxiliary_targets = {"contact_intent_next": intent_ref[..., 1:2]}
```

Check binary finite labels, exact canonical-reference agreement by clip hash/frame,
adjacent pair continuity, state/next-state alignment, and existing episode/clip/
terminal masks before use. The last bit comes from the final transition endpoint,
not the Replay adapter's terminal-row padding. Frame indices select targets and
existing termination references only; they are not network features.

Reuse `WorldModel.window_loss` and its existing auxiliary-target hook. The HOI
`_loss_step` returns next state85 and 83 loss components: unchanged weighted robot
70 and object12 squared normalized errors, plus one unreduced
`BCEWithLogitsLoss(logit_next, contact_intent_next)`. The original loop retains its
sum-over-components, mean-over-batch/window reduction and separate termination BCE.
Use explicit finite nonnegative robot/object/contact weights; record their values.
For the proposed balanced diagnostic, use ordinary BCE (`pos_weight=1`) and an
initial contact coefficient of 1. Preserve physical/termination coefficients when
migrating; record any later train-only scale calibration before fitting. Do not
reuse the old force coefficient as the contact coefficient or fit force statistics.

Remove ForceStatistics, force encode/decode, Huber/L1 regression, active/inactive
force balancing, and force supervision from the active WM/trainer/fit path. Report
contact BCE separately from termination BCE and physical NSE; the combined loss is
not a physical NSE. Adapt HOI `online_error`/`rollout_error` to the 85D recurrent
state and 84D physical metrics; the inherited physical-only helper cannot receive
85D states unchanged.

`JointWorldModelTrainer` continues to inherit the original optimizer/update method,
backpropagation, gradient clipping, and distributed reduction. No new Trainer,
ReplayBuffer, training loop implementation, or shared-source change is required.
Actor, Critic, TD(lambda), `Trainer.run()`, and original `sm-train` stay unchanged.
The 85D HOI interface is not wired into Actor imagination in this revision.

## 4. Smallest proposed additional diagnostic

The existing `collection_02` has 320 transitions, all `[1,1]`, at frames 260–269.
Its 240/80 episode split cannot validate OFF, onset, or release. Keep it and all
previous evidence. P2's 96 controlled rows are also not a replacement training set:
they use the older reference and omit measured xi required by this WM pipeline.

Propose **four independently reset eight-transition episodes, 32 transitions total**:

| Split / episode | Initial frame | Transition start frames | Endpoint | Boundary |
|---|---:|---|---:|---|
| Train onset | 256 | 256–263 | 264 | 259→260 |
| Train release | 546 | 546–553 | 554 | 549→550 |
| Validation onset | 255 | 255–262 | 263 | 259→260 |
| Validation release | 545 | 545–552 | 553 | 549→550 |

Each split has 16 transitions: seven `00`, one `01`, seven `11`, one `10`; eight
positive and eight negative next-bit targets. Each split has two full eight-step
windows. This covers pre-contact OFF, onset, sustained ON, release, and post-contact
OFF. Different boundary offsets avoid identical elapsed-step positions across splits.

Use P1 `ReferenceResetCache`/state IO at the specified initial frame, distinct recorded
random seeds, existing native action mapping and bounded diagnostic commands/noise,
and the unchanged P2 recorder. Capture actual physical states, executed actions,
xi, sensors, and intent pairs. Do not overwrite physical states after reset, alter
labels, suppress capacity/failure guards, or synthesize missing xi. Preserve whole
source episodes in the split; report the shared reference SHA and frame overlap.
Keep the existing 320 rows outside this primary balanced fit; use them only for
separately reported physical regression checks if useful.

This is a minimum implementation diagnostic, not independent motion/clip
generalization: the validation episodes share reference boundary frames with
training. One onset and release per split offer no robust statistical estimate.
Reference-reset stability at these frames is untested; the earlier collection had
a post-forward failure at step 11. Eight steps do not guarantee stability. A failed
capture or missing window is PARTIAL, not permission to relax guards. Intent may
also be unidentifiable from physical state/current bit for some trajectories;
failure does not authorize adding phase, future reference, or additional labels.

Collection is an experimental prerequisite for Gate B, not for implementing the
contract/unit tests. **Do not run it in Task 0.** A subsequent approval must cover
this 32-transition capture and, separately, a bounded diagnostic fit (proposed cap:
200 original-trainer updates, hidden64×3, CPU, eight-step windows). No 840-frame
campaign, long training, or automatic sweep is proposed. Use new, non-overwriting
paths under `outputs/hoi/contact_supervision/contact_intent_model_validation/`.

## 5. Minimal validation and proposed acceptance

Implementation acceptance requires the dimension/feedback/label-isolation tests and
strict migration tests in the [implementation audit](IMPLEMENTATION_AUDIT.md), with
original robot/P1/P2 regression checks preserved. Gate metrics require an actually
updated model and independent simulator episodes; successful forward passes and
synthetic arithmetic tests are not learned-performance evidence.

| Gate | Required evidence | Proposed bounded-diagnostic pass criterion for review |
|---|---|---|
| A: robot/object | One-step on **every valid transition**; eight-step on every valid full window. Report robot NSE, object NSE, object position/rotation/linear/angular RMSE, and constant-velocity and frozen migrated-model baselines. | Finite results; each body at each horizon has NSE no worse than the frozen initialized model within `atol=rtol=2e-6`; report both heads' optimizer updates. This is a preservation criterion, not a claim of task-level physical accuracy. |
| B: global reference intent | Confusion counts, accuracy, precision, recall, F1; exact onset/release hits, misses, false events, timing error; probability/BCE and classification at horizons 1–8. | One-step accuracy and ON-class F1 exceed contact persistence; both held-out onset/release are detected at the exact transition with no false events in the full eight-step trajectories; both horizon-eight endpoint classes are correct. |

These relative performance thresholds are proposals for this review, not previously
approved research thresholds or measured results. Freeze them before fitting. On
the proposed validation set, one-step persistence has accuracy/F1 `0.875`; recursive
persistence has eight-step endpoint accuracy `0/2`. Also report always-OFF/always-ON
baselines and aggregate eight-step sequence accuracy. Include the initial observed
bit when detecting the first predicted transition; never compare across resets.
Count unique events by reference hash/env/episode/frame and report overlapping-window
repeats separately. Missing target classes/events yield undefined metrics and PARTIAL,
not a successful zero. Declare precision undefined when there are no predicted
positives, report its denominator, and use F1=0 when positives exist but TP=0.

P3 diagnostic PASS requires implementation checks and both Gates A/B under agreed
criteria. Missing data, authorization, or completed validation remains PARTIAL;
measured criterion violations are FAIL for that gate. No force magnitude, direction,
normalization, force-gradient, natural-HOI campaign, or Actor force-optimization gate
is part of this simplified P3. Historical P3 PARTIAL findings remain historical.
Stop after review of this specification; source changes require explicit approval.
