# Revised P3 Training Framework Reuse Audit

Core implementation checks: **PASS**. Bounded learned validation: **PARTIAL**, since
the approved physical capture failed and fitting was not started. This audit
describes actual implemented reuse, not a second training architecture.

| Framework component | Actual reuse / evidence |
|---|---|
| Robot `StateSpec` | Unchanged source/order/scales/integration. Exact robot-only loss, metrics and gradient regression passed. |
| `DynamicsMLP` | Unchanged class and SiLU trunk; HOI constructor supplies feature105/output83, yielding input167 with action29/xi33. |
| Joint physical spec/support | Existing physical84/canonical86/residual82 and support18 files unchanged. Integration/canonicalization/support regression passed. |
| `WorldModel.window_loss` | Existing autoregressive loop and auxiliary-target hook reused unchanged. HOI override returns state85, physical82+BCE1 components and original termination outputs. |
| `WorldModelTrainer` | Original constructor/Adam/update/backward/allreduce/clipping inherited. Runtime identity assertion and optimizer-path tests passed. Bounded fit script calls only this update method. |
| Replay | Original Replay and P2 HOI Replay/recorder unchanged. Adapter constructs contact state after sampling; physical storage stays 84, intent remains the existing two-value sidecar. |
| Termination | Same separate head, reference-root features, BCE and physical-failure target. Contact does not replace termination. |
| Actor/Critic/TD(lambda)/imagination | Unchanged, unused by these diagnostics. No contact/force reward or Actor optimization added. |
| `Trainer.run()` / `sm-train` | Unchanged lifecycle, optimizer defaults, CLI entry and robot checkpoint behavior. HOI diagnostic scripts remain explicit entry points. |
| Checkpoints | Explicit named initialization under HOI, no global loader/default modification and no optimizer resume. Real joint FP32 parity passed; optional external robot FP32 multi-step limitation is reported separately. |

The only altered training adapter is `src/supermimic/hoi/wm_trainer.py`; the physical
state is always sliced to 84 before existing state/spec interfaces. Original shared
Python files have no edits. All existing-file changes are the 12 approved HOI/script/
test files listed in the final integrity manifest; the two additional test files
are local to `tests/hoi/`.

ForceStatistics remains in `force.py` and two historical utility tests. Its only
checkpoint use is a local import while validating persisted legacy statistics
before migration. The new model constructor, trainer loss, fit script and new
checkpoint loader do not fit/use force statistics. The old readiness function is
explicitly named `legacy_force_readiness_gate`; the historical verifier reproduces
its prior result without applying those requirements to revised P3.

Physical sensor aggregation is still checked when reading measured archives. This
protects P2 diagnostic provenance; it supplies no target/input/weight to the Contact
Head. Tests modifying force/occupancy and clean-action sidecars leave the new
training loss exactly unchanged. Corrupt reference intent now raises an error.

No second Trainer, ReplayBuffer, independent optimizer, Actor update or alternate
learned dynamics framework was introduced. `legacy_joint_step` is a read-only
checkpoint parity probe using stored linear tensors, not another trained model.
