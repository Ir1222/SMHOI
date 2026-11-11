# P2 Training Framework Reuse Audit

**PASS. No shared training source or installed dependency changed.** The original
SuperMimic algorithms remain the only training framework; none was executed by P2.

| Component | Reuse / modification | Evidence |
|---|---|---|
| `data/replay_buffer.py:ReplayBuffer` | HOI subclass directly reuses original allocation, ring writes, write pointers, episode counters, `end_episode`, `_rolled`, base `window_mask`, and `_sample_from_mask`. Inherits `sample_starts`. | Original file SHA unchanged; ring wrap, partial resets and eight-step tests pass. |
| Replay window gather | HOI-specific gather reads extra fields at the selected transition index and strengthens clip/frame/validity masking. It does not call the original successor-row `done` gather. | P0 `[1,1]` discrepancy becomes HOI `[0,1]`; physical-failure, timeout and clip-end tests pass. Original robot-only behavior is unchanged, including its documented discrepancy. |
| `data/recorder.py:ReplayRecorder` | Preserved byte-for-byte. New HOI RecorderTerm uses the same native lifecycle hooks with JointStateIO and explicit clip metadata; it does not copy robot-only recorder code or its terminal-label convention. | Terminal action and nonzero force copied before reset; no phantom transitions; selective external reset tested. |
| P1 scene/state/sensing | Factory, named StateIO84, reference cache and six original contact functions reused. Contact file only adds interval aggregation/capture and a common-point torque helper. | 420 protected files unchanged; original six contact functions AST-identical; P1 tests pass. |
| Native mjlab step/reset/metrics/termination managers | Reused directly. Register substep capture and recorder through manager APIs; no alternative production step loop. | 104 vector substeps checked; graph/eager values agree within declared tolerances. Diagnostic-only step wrapper observes inputs and calls the unchanged step. |
| Original action and xi33 | Reuse `NoisyJointPositionAction.executed_raw` and `SimContext.read`; do not append support fields. | Measured execution-noise std `0.0302866`; zero action-label alignment error at all four substeps. |
| Trainer.run / scheduler / UTD / distributed / logging / checkpoint lifecycle | Untouched, no alternate Trainer or CLI. | Protected hashes match baseline. |
| WM fitting / Actor BPTT / Critic TD(lambda) / optimizers | Untouched; no new learned model, reward or optimization path. | Protected hashes match; no training command executed. |
| Original robot-only regression | Original CUDA HDMI environment, StateIO, noise/context and eight-step Replay smoke test runs through existing P1 test. | Complete suite 90 passed / 17 existing missing-LAFAN1 skips; no P2/P1 physical skip. |

Shared modifications requiring justification: **none**. The HOI sampler repeats only
schema-specific tensor gathering/index construction; eligibility, random selection,
ring lifecycle and episode mechanics come from the original implementation. It does
not duplicate the complete ReplayBuffer or another training algorithm. Explicit
next-state sidecars intentionally retain terminal supervision immediately; native
k+1 row sampling retains its one-row latency for the latest nonterminal transition.

Diagnostics are controlled physical fixtures, not a policy benchmark. First full
suite exposed a line-search overflow in the original bilateral fixture; the guard
rejected it. The documented elevated-fixture correction passes without changing
solver settings, overflow checks, source assets or shared code. This does not
establish floor-interaction robustness or resolve historical M6.5 failures.
