# P3 Training Framework Reuse Audit

**Result: PASS for architectural reuse and tested robot-only compatibility.**
Overall P3 remains PARTIAL for experimental coverage and learned force readiness.

| Original component | Reuse and evidence |
|---|---|
| `world_model/state_spec.py::StateSpec` | Composed by JointStateSpec, never copied/reordered. Robot canonical/integration prefix exact equality; source hash unchanged. |
| `world_model/dynamics.py::DynamicsMLP` | Used directly with feat_dim104/out_dim88; same SiLU trunk/default 2048×3 and zero final affine. Three row blocks are robot 70/object 12/force 6. Source unchanged. |
| `world_model/world_model.py::WorldModel.window_loss` | Original autoregressive loop/reductions reused by subclass. Only optional per-step auxiliary target kwargs and metric aggregation added. Robot path old/current loss, metrics and every gradient exactly equal in eight-step regression. |
| `WorldModel::TerminationHead`, analytic termination features | Original head/feature computation reused, next-frame reference gathered explicitly, physical_failure target only. No new contact classifier. |
| `world_model/wm_trainer.py::WorldModelTrainer` | Joint adapter overrides batch/reference loss and online_error only. Constructor/sampling/update/Adam/allreduce/clip/step/aggregation are inherited. Identity assertion verifies `.update is WorldModelTrainer.update`; measured data produced real updates. Source unchanged. |
| `data/replay_buffer.py::ReplayBuffer` | Original ring storage/random eligible-row sampler reused through accepted P2 subclass. No duplicate ReplayBuffer. Source unchanged. |
| P2 `hoi/replay.py` | Existing gather extracted into `_gather_windows`; random sampling calls exactly the same gather. Exhaustive evaluation uses it without copying indexing semantics. |
| `training/loop.py::Trainer.run()` | Unchanged shared collection/UTD/scheduling/logging/checkpoint mainline. P3 has no new production main loop. P7 integration is deliberately not claimed. |
| Actor/BPTT, Critic/TD(lambda), optimizer/distributed/schedule | All original files unchanged. No force rewards, imagination transitions, policy updates or PPO framework introduced. |
| `sm-train`, config/entry point/dependency defaults | Unchanged. Default checkpoint strict load and actual external legacy WM migration tested. |

## Shared changes and why an HOI-local adapter alone is insufficient

Only `src/supermimic/world_model/world_model.py` changes in the shared runtime core:
11 added/2 removed lines. The original `_loss_step` invocation cannot receive
transition-indexed measured force/occupancy. An optional `auxiliary_targets` keyword
lets the existing autoregressive loop pass these to the HOI loss implementation.
An optional metric reducer reports three weighted components. Without this hook,
HOI would have to duplicate or bypass the original window loop. With no auxiliary
targets, original arithmetic, metrics and gradients are identical. No task branches
were added to Trainer, Actor, Critic or ReplayBuffer.

The HOI-local Replay change is5 added lines; it extracts gathering rather than
reimplementing windows. The only other previously existing file changed is
`tests/test_hoi_reconciliation.py`: the historical M6 baseline also hashed the old
shared WM source. Its old hash and exact reviewed replacement are both asserted;
all other historical files remain hash-checked. M6 evidence and verification helper
are unchanged. This is a version-specific integrity check, not a broad exclusion.

## Duplication assessment and scope

No production Trainer, StateSpec robot algorithm, ReplayBuffer, optimizer loop,
Actor BPTT or TD(lambda) is copied. New object state/force/support/metadata logic is
HOI-specific and lives under `src/supermimic/hoi/`.

`fit_p3_diagnostic.py` is a bounded harness calling the original trainer.update 200
times; it does not contain a second optimizer/backward/allreduce algorithm. The
support ridge comparison is a small diagnostic predictor, not a production WM.
`validation.py` has a no-grad prediction loop to calculate held-out metrics, not a
second training/autograd loop. No integrated lifecycle or stochastic resume claim.

Full test suite:102 passed, 17 missing-LAFAN skips. The explicit original-vs-current
robot regression uses accepted commit a991ee7 and strict equal weights, then checks
loss, metrics and all parameter gradients with atol=rtol=0. Real external legacy
2048×3 WM migration error is1.9073486328125e-6; strict diagnostic reload error0.
`integrity_final.json` confirms exactly three intentional existing-file changes,
all other baseline files and ten installed dependency hashes unchanged.

Unverified scopes remain explicit: LAFAN-dependent tests, distributed multi-GPU
training, compiled HOI fitting, full Trainer.run HOI integration and optimizer/Replay/
RNG resume. These are not implied by the successful CPU diagnostic or migration.
